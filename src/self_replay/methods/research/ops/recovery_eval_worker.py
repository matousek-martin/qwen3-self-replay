"""Evaluate saved final adapters; no training is performed by this role."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    data=ROOT/'data';out=ROOT/'output'
    for key in ['inputs','sources']:
        for path,expected in json.loads((data/f'isolated/{key}.json').read_text()).items():
            target=(data if key=='inputs' else ROOT)/path
            if hashlib.sha256(target.read_bytes()).hexdigest()!=expected:raise ValueError('Immutable file changed: '+path)
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True).strip()
    def run(*args):subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,check=True)
    config=json.loads((data/'recovery/manifest.json').read_text())
    run('-m','experiments.vllm_compat','--out',out/'vllm-compat.json')
    results={}
    for suite,meta in config['suites'].items():
        stage=data/'recovery'/suite;destination=out/suite
        destination.mkdir()
        run('-m','experiments.vllm_lora_smoke','--config',stage/'evaluation.json','--out',destination/'lora-smoke.json')
        run('-m','experiments.study_eval','--config',stage/'evaluation.json','--out',destination/'study')
        result=json.loads((destination/'study/study-result.json').read_text())
        if not result['complete']:raise RuntimeError('Incomplete recovered evaluation')
        candidates=json.loads((stage/'evaluation.json').read_text())['candidates'][1:]
        for i,left in enumerate(candidates):
            for right in candidates[i+1:]:
                run(ROOT/'ThinkingCap-Test/compare.py',destination/'study/runs'/left,
                    destination/'study/runs'/right,'--out',destination/f'{right}-vs-{left}.json')
        provenance={'original_job':meta['original_job'],'adapter_hashes':meta['adapter_hashes'],
                    'gpu_identity':gpu,'baseline':'fresh on this evaluation worker','result':result}
        (destination/'recovery-result.json').write_text(json.dumps(provenance,indent=2)+'\n')
        results[suite]=provenance
    (out/'role-result.json').write_text(json.dumps({'development_complete':True,'evaluation_only':True,
                                                 'gpu_identity':gpu,'suites':results},indent=2)+'\n')


if __name__=='__main__':main()
