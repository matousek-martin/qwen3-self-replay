"""Exercise the registered v2 evaluator on its full cohort before new training."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    for key in ['sources','inputs']:
        for name,expected in json.loads((ROOT/f'data/isolated/{key}.json').read_text()).items():
            path=(ROOT/'data' if key=='inputs' else ROOT)/name
            if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                raise ValueError('Preflight input/source changed: '+name)
    def run(*args):
        subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,check=True)
    run('-m','experiments.vllm_compat','--out',ROOT/'output/vllm-compat.json')
    for arm in ['base','existing-adapter']:
        target=ROOT/f'output/validation/{arm}.jsonl'
        args=['-m','experiments.teacher_generation.evaluate_matched','--data',ROOT/'data/val.jsonl','--out',target]
        if arm!='base':args+=['--adapter',ROOT/'data/evaluation-preflight-adapter']
        run(*args)
        rows=[json.loads(line) for line in target.read_text().splitlines()]
        if len(rows)!=800 or len({r['prompt_id'] for r in rows})!=800:
            raise ValueError('Incomplete preflight coverage')
        if any(type(r['correct']) is not bool or r['total_tokens']!=len(r['generated_token_ids']) for r in rows):
            raise ValueError('Malformed evaluation record')
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid','--format=csv,noheader'],text=True).strip()
    (ROOT/'output/role-result.json').write_text(json.dumps({'development_complete':True,
        'purpose':'Full-cohort v2 evaluator integration check only; existing adapter is not a teacher candidate.',
        'evaluations':2,'responses_per_evaluation':800,'fresh_baseline_same_worker':True,'gpu_identity':gpu},indent=2)+'\n')


if __name__=='__main__':main()
