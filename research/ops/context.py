#!/usr/bin/env python3
"""Read-only study navigation and local reference checks; no network or job control."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def local_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f'Path outside project: {value}')
    if not path.exists():
        raise ValueError(f'Missing path: {value}')
    return path


def read_observation(item, sources):
    path = local_path(sources[item['source']]['path'])
    value = json.loads(path.read_text())
    for part in item['field']:
        value = value[part]
    if not isinstance(value, (str, int, float, bool, type(None))):
        raise ValueError(f'Observation must select a scalar: {item["label"]}')
    return value, path


def check(manifest, path):
    if manifest['schema_version'] != 1 or manifest['id'] != path.parent.name:
        raise ValueError(f'Invalid schema/id: {path}')
    for name in manifest['documents']:
        local_path(name)
    for source in manifest['sources'].values():
        target = local_path(source['path'])
        if 'sha256' in source:
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
            if actual != source['sha256']:
                raise ValueError(f'Evidence changed; review report: {source["path"]}')
    for item in manifest['observations']:
        read_observation(item, manifest['sources'])


def check_links():
    files = [ROOT / 'README.md', ROOT / 'AGENTS.md', ROOT / 'experiments/README.md',
             ROOT / 'experiments/offline_training/README.md', ROOT / 'experiments/rl_training/README.md']
    files += list((ROOT / 'docs').rglob('*.md')) + list((ROOT / 'studies').rglob('*.md'))
    for path in files:
        for link in re.findall(r'\[[^\]]*\]\(([^\s)]+)\)', path.read_text()):
            if '://' in link or link.startswith(('#', 'mailto:')):
                continue
            target = link.split('#', 1)[0]
            if not (path.parent / target).exists():
                raise ValueError(f'Broken link in {path.relative_to(ROOT)}: {link}')
    return len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', nargs='?')
    parser.add_argument('--check', action='store_true', help='Validate all context references')
    args = parser.parse_args()
    manifests = {p.parent.name: (json.loads(p.read_text()), p)
                 for p in sorted((ROOT / 'studies').glob('*/manifest.json'))}
    if not manifests:
        raise ValueError('No study manifests found')
    if args.check:
        for manifest, path in manifests.values():
            check(manifest, path)
        count = check_links()
        print(f'PASS: {len(manifests)} manifests, source paths, evidence hashes, observation fields, {count} Markdown files.')
    if args.study:
        if args.study not in manifests:
            parser.error('Unknown study; choose: ' + ', '.join(manifests))
        manifest, path = manifests[args.study]
        check(manifest, path)
        print(manifest['title'])
        print('Read in order:')
        for name in ['AGENTS.md', 'README.md', *manifest['documents']]:
            print('  ' + str(local_path(name)))
        print('Local file observations read at ' + datetime.now(timezone.utc).isoformat())
        print('These are file values, not proof of live worker/provider state.')
        for item in manifest['observations']:
            value, target = read_observation(item, manifest['sources'])
            modified = datetime.fromtimestamp(target.stat().st_mtime, timezone.utc).isoformat()
            print(f'  {item["label"]}: {json.dumps(value, ensure_ascii=False)} (file modified {modified})')
        print('Evidence/code map: ' + str(path))
    elif not args.check:
        for key, (manifest, _) in manifests.items():
            print(f'{key}: {manifest["title"]}')
        print('Usage: python3 ops/context.py <study> | --check')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError, TypeError) as exc:
        print(f'Context error: {exc}', file=sys.stderr)
        sys.exit(1)
