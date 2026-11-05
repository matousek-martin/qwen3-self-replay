"""Remote disk probe and narrowly scoped, optional package-cache cleanup.

Executed on a worker via stdin; does not edit the registered worker payload.
"""
import json
import os
from pathlib import Path
import subprocess
import time

CACHE = Path('/root/.cache/thinkingcap/uv')
VENV = Path('/opt/thinkingcap-venv')
GIB = 1024 ** 3


def usage(path):
    s = os.statvfs(path)
    return {'path': path, 'total_bytes': s.f_blocks * s.f_frsize,
            'available_bytes': s.f_bavail * s.f_frsize,
            'used_percent': round(100 * (s.f_blocks - s.f_bfree) / max(1, s.f_blocks), 2),
            'inode_used_percent': round(100 * (s.f_files - s.f_ffree) / max(1, s.f_files), 2)}


def pressure(disk):
    if disk['used_percent'] >= 95 or disk['available_bytes'] < GIB or disk['inode_used_percent'] >= 95:
        return 'critical'
    if disk['used_percent'] >= 85 or disk['available_bytes'] < 4 * GIB or disk['inode_used_percent'] >= 85:
        return 'high'
    return 'healthy'


def cache_blocker(cache=CACHE, venv=VENV, proc=Path('/proc')):
    if not cache.is_dir() or cache.resolve() != cache or not venv.is_dir():
        return 'cache_missing_or_redirected'
    # A symlink-install mode would make cache cleanup unsafe for imports.
    for directory, dirs, files in os.walk(venv, followlinks=False):
        for name in dirs + files:
            p = Path(directory) / name
            if p.is_symlink() and p.resolve().is_relative_to(cache):
                return 'installed_environment_links_to_cache'
    for p in proc.glob('[0-9]*'):
        try:
            argv = (p / 'cmdline').read_bytes().split(b'\0')
            # Avoid colliding with dependency installation or another cleaner.
            names = {Path(a.decode(errors='replace')).name for a in argv[:3]}
            if names & {'uv', 'pip', 'pip3'}:
                return 'package_manager_active'
            if str(cache) in (p / 'maps').read_text():
                return 'cache_memory_mapped'
            for fd in (p / 'fd').iterdir():
                try:
                    if os.readlink(fd).startswith(str(cache) + '/'):
                        return 'cache_file_open'
                except FileNotFoundError:
                    pass
        except (FileNotFoundError, ProcessLookupError):
            continue  # A process exited during the scan.
        except PermissionError:
            return 'process_access_incomplete'
    return None


def main():
    before = {p: usage(p) for p in ['/', '/workspace']}
    action = {'status': 'not_needed'}
    if pressure(before['/']) != 'healthy':
        blocker = cache_blocker()
        binary = Path('/workspace/project/uv')
        if blocker or not binary.is_file() or binary.is_symlink():
            action = {'status': 'skipped', 'reason': blocker or 'uv_unavailable'}
        else:
            try:
                result = subprocess.run([str(binary), 'cache', 'clean', '--cache-dir', str(CACHE)],
                                        capture_output=True, timeout=90)
                action = {'status': 'cleaned' if result.returncode == 0 else 'failed',
                          'returncode': result.returncode}
            except subprocess.TimeoutExpired:
                action = {'status': 'timed_out'}
    after = {p: usage(p) for p in ['/', '/workspace']}
    print(json.dumps({'checked_at': time.time(), 'before': before, 'after': after,
                      'cleanup': action, 'freed_bytes': max(0, after['/']['available_bytes'] - before['/']['available_bytes'])}))


if __name__ == '__main__':
    main()
