"""Train/evaluate static teacher versus static student, using frozen schedules."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from experiments.gpu_policy import verify_gpu_names

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    config=json.loads((ROOT/'data/isolated/config.json').read_text())
    if a.seed not in config['seeds']:raise ValueError('Seed not registered')
    identity=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid','--format=csv,noheader'],text=True).strip()
    verify_gpu_names(identity.split(',')[0])
    for key in ['sources','inputs']:
        for path,expected in json.loads((ROOT/f'data/isolated/{key}.json').read_text()).items():
            f=(ROOT/'data' if key=='inputs' else ROOT)/path
            if hashlib.sha256(f.read_bytes()).hexdigest()!=expected:raise ValueError('Immutable '+key+' changed: '+path)
    def run(*args):
        subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,check=True)
    run('-m','unittest','experiments.rl_training.test_static_replay','-v')
    states={}
    for arm in ['student','teacher']:
        target=ROOT/'output/training'/f'{arm}-seed{a.seed}'
        params=dict(config['training'],seed=a.seed,out=str(target),static_arm=arm,
             static_replay=str(ROOT/'data/static/manifest.json'),
             static_replay_sha256=hashlib.sha256((ROOT/'data/static/manifest.json').read_bytes()).hexdigest())
        params['data']=str(ROOT/'data/pilot.jsonl');params['references']=str(ROOT/'data/shared/references.json')
        command=['-m','experiments.rl_training.train']
        for key,value in params.items():command+=['--'+key.replace('_','-'),str(value)]
        run(*command)
        states[arm]=json.loads((target/'status.json').read_text())
        if states[arm]['status']!='completed' or states[arm]['completed_groups']!=128:
            raise RuntimeError('Static paired training incomplete')
    run('-m','experiments.vllm_compat','--out',ROOT/'output/vllm-compat.json')
    for arm in ['base','student','teacher']:
        command=['-m','experiments.teacher_generation.evaluate_matched','--data',ROOT/'data/val.jsonl',
                 '--out',ROOT/f'output/validation/{arm}.jsonl']
        if arm!='base':command+=['--adapter',ROOT/f'output/training/{arm}-seed{a.seed}/adapter']
        run(*command)
    (ROOT/'output/role-result.json').write_text(json.dumps({'seed':a.seed,'training':states,
        'development_complete':True,'gpu_identity':identity,'primary':'800 frozen training-source validation; greedy; fresh paired base',
        'config':config},indent=2)+'\n')


if __name__=='__main__':main()
