"""Load only the pinned, pure-Python answer verifier from the author's source.

No torch/VERL import, remote execution or dynamic dependency installation.
Original file, copyright header and source URL remain in the evidence bundle.
The verifier uses normalized string equality, not a symbolic proof of equality.
"""
import ast
import hashlib
import json
from pathlib import Path

FUNCTIONS = {'verify_correctness', 'is_equiv', 'remove_boxed',
             'last_boxed_only_string', 'fix_fracs', 'fix_a_slash_b',
             'remove_right_units', 'fix_sqrt', 'strip_string'}


def load_verifier(sources):
    sources = Path(sources)
    path = sources / 'shorterbetter_reward.py'
    expected = json.loads((sources / 'manifest.json').read_text())[path.name]['sha256']
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Pinned upstream verifier hash mismatch')
    tree = ast.parse(path.read_text())
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
            and node.name in FUNCTIONS]
    if {node.name for node in body} != FUNCTIONS:
        raise ValueError('Upstream verifier function set changed')
    namespace = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['verify_correctness']
