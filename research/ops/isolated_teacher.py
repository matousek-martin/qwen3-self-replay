"""Run the sealed API generator with canonical credentials and shared budget."""
import argparse
import json
from pathlib import Path
import sys
from isolated_cloud import ROOT, verify


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--approval', type=Path, required=True)
    p.add_argument('--max-requests', type=int)
    a = p.parse_args()
    verify(a.release)
    import experiments
    from experiments import cloud
    assert cloud.STATE.resolve() == (ROOT / 'lab/experiments/cloud').resolve()
    experiments.__path__ = [str(a.release.resolve() / 'experiments'), *experiments.__path__]
    from experiments.teacher_generation import runner, common
    assert common.ROOT == a.release.resolve()
    plan = json.loads((a.release / 'studies/teacher-pilot/openai-plan.json').read_text())
    runner.validate_plan(plan)
    result = runner.execute(plan, a.out, json.loads(a.approval.read_text()),
                            max_requests=a.max_requests,
                            key_loader=lambda: common.read_key(ROOT / '.env'))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
