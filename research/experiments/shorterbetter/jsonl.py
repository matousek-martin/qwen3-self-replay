"""JSON Lines uses LF records; Unicode separators inside JSON strings are data."""
import json
from pathlib import Path


def read_rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]
