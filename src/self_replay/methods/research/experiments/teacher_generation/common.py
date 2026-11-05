"""Small durable IO helpers; no shell evaluation of credentials."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
from contextlib import contextmanager
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'
PROMPTS = ROOT / 'lab/experiments/data/teacher-pilot-20261024/teacher-comparison-128.jsonl'
PROMPTS_SHA = 'ece4b3becde7e3b63ef297079fb4048e326f482437284aad74e6c054a19896ce'


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def durable_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def lock(path, blocking=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield


def read_key(path=ROOT / '.env'):
    # Strict single-line dotenv subset: quoted/unquoted values and comments.
    # No source/eval, interpolation, environment dump, or credential in errors.
    values = []
    for line in Path(path).read_text().splitlines():
        line = line.strip().removeprefix('export ')
        name, sep, value = line.partition('=')
        if sep and name.strip() == 'OPENAI_API_KEY':
            try:
                parts = shlex.split(value, comments=True)
            except ValueError:
                raise ValueError('Malformed OPENAI_API_KEY assignment') from None
            if len(parts) != 1 or not parts[0] or any(c.isspace() for c in parts[0]):
                raise ValueError('Invalid OPENAI_API_KEY assignment')
            values.append(parts[0])
    if len(values) != 1:
        raise ValueError('Expected one OPENAI_API_KEY assignment')
    return values[0]


def prompts():
    if file_hash(PROMPTS) != PROMPTS_SHA:
        raise ValueError('Original 128-prompt hash changed')
    rows = [json.loads(line) for line in PROMPTS.read_text().splitlines()]
    if len(rows) != 128 or len({r['id'] for r in rows}) != 128 or any(r['split'] != 'train' for r in rows):
        raise ValueError('Invalid training prompt cohort')
    return rows
