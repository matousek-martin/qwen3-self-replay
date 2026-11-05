"""Registered deterministic continuation/resume check using the preserved checkpoint.

Reuses previous engineering data only as technical evidence; the main experiment
still starts from the pinned original model, after all new GPU checks pass.
"""
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

from .config import assert_launch_ready
from .distributed import atomic_json
from .launch_checks import seed_resume
from .prepare import sha
from .throughput import read_rows

ROOT = Path(__file__).resolve().parents[2]


def compare(reference, resumed, world, step=32):
    from safetensors import safe_open
    import torch
    for rank in range(world):
        def completions(folder):
            return [(r['prompt_id'],r['completion_ids_sha256'],r['reward'])
                    for r in read_rows(folder/f'rank-{rank:03d}/rollouts.jsonl')
                    if r['split']=='train' and r['optimizer_step']==step]
        if not completions(reference) or completions(reference)!=completions(resumed):
            raise RuntimeError('Deterministic resume prompts/completions/rewards differ')
        for root in (reference,resumed):
            folder=root/f'rank-{rank:03d}'
            d=json.loads((folder/'determinism-check.json').read_text())
            if not d['enabled'] or d['warn_only']:
                raise RuntimeError('Determinism not enforced on all ranks')
            if world > 1:
                reduction=json.loads((folder/'ddp-reduction-check.json').read_text())
                if reduction['method']!='gather_then_fp32_sum_rank_order' or reduction['world_size']!=world:
                    raise RuntimeError('Fixed rank-order reduction not enforced')
            for name in ('precision-check','gpu-memory-check','gpu-sampling-check'):
                if json.loads((folder/(name+'.json')).read_text())['status']!='passed':
                    raise RuntimeError('Deterministic GPU check failed: '+name)
            rows=[r for r in read_rows(folder/'parameter-updates.jsonl') if r['optimizer_step']>step]
            if [r['optimizer_step'] for r in rows]!=list(range(step+1,step+5)) or not all(r['rank_samples_identical'] for r in rows):
                raise RuntimeError('Incomplete deterministic update/rank evidence')
    maximum=0.; equal=True; tensors=0
    with safe_open(reference/'model/model.safetensors',framework='pt',device='cpu') as a, \
         safe_open(resumed/'model/model.safetensors',framework='pt',device='cpu') as b:
        if set(a.keys())!=set(b.keys()):raise RuntimeError('Export tensor names differ')
        for name in a.keys():
            left,right=a.get_tensor(name),b.get_tensor(name)
            torch.testing.assert_close(left,right,rtol=1e-6,atol=1e-8)
            equal &= torch.equal(left,right)
            maximum=max(maximum,float((left-right).abs().max()));tensors+=1
    from transformers import AutoConfig,AutoTokenizer
    if AutoConfig.from_pretrained(resumed/'model',local_files_only=True).model_type!='qwen3':
        raise RuntimeError('Wrong export architecture')
    AutoTokenizer.from_pretrained(resumed/'model',local_files_only=True)
    return {'status':'passed','start_step':step,'end_step':step+4,'tensors':tensors,
            'max_abs_difference':maximum,'bit_identical':equal,'rtol':1e-6,'atol':1e-8}


def run(protocol,path,lease,evidence):
    registration=protocol['execution']['resume_repair']
    parent=ROOT/registration['parent_directory']
    checkpoint=parent/'checkpoint-28'
    for name,expected in registration['checkpoint_sha256'].items():
        if sha(checkpoint/name)!=expected:raise RuntimeError('Parent checkpoint hash mismatch: '+name)
    technical=copy.deepcopy(protocol)
    technical['id']+='-resume-diagnostic'
    technical['execution'].update(phase='benchmark',gated_full_training=False,resume_diagnostic=True,resume_trace=True)
    technical['execution'].pop('resume_repair')
    technical['training'].update(rollout_batches=9,optimizer_updates_total=36,training_completions_per_arm=4608)
    technical['monitoring']['every_rollout_batches']=1
    technical['technical_resume_parent']={'protocol_sha256':json.loads((parent/'run-provenance.json').read_text())['protocol_sha256'],
                                          'optimizer_step':28,'checkpoint_sha256':registration['checkpoint_sha256']}
    technical_path=ROOT/'output/deterministic-protocol.json'
    atomic_json(technical_path,technical)
    technical_evidence=dict(evidence,protocol_sha256=sha(technical_path))
    assert_launch_ready(technical,technical_evidence)
    evidence_path=ROOT/'output/deterministic-readiness.json'
    atomic_json(evidence_path,technical_evidence)
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1',
             '--nproc_per_node='+str(lease['gpu_count']),'--max_restarts=0','-m','experiments.shorterbetter.train',
             '--protocol',str(technical_path),'--arm','shorterbetter','--seed',str(protocol['seeds'][0]),
             '--execute','--readiness',str(evidence_path),'--job-name',lease['job_name']]
    reference,resumed=ROOT/'output/deterministic-reference',ROOT/'output/resume-check'
    seed_resume(parent,reference,lease['gpu_count'])
    state_path=reference/'checkpoint-28/trainer_state.json'
    state=json.loads(state_path.read_text())
    atomic_json(reference/'checkpoint-runtime-overrides.json',{'original':{k:state.get(k) for k in ('save_steps','eval_steps','max_steps')},
                'registered':{'save_steps':4,'eval_steps':4,'max_steps':36},'model_optimizer_rng_unchanged':True})
    state.update(save_steps=4,eval_steps=4,max_steps=36)
    atomic_json(state_path,state)
    atomic_json(ROOT/'output/phase.json',{'phase':'deterministic_resume_reference','time':time.time()})
    subprocess.run(command+['--out',str(reference),'--resume',str(reference/'checkpoint-28')],check=True)
    seed_resume(reference,resumed,lease['gpu_count'],step=32)
    atomic_json(ROOT/'output/phase.json',{'phase':'gpu_resume_verification','time':time.time()})
    subprocess.run(command+['--out',str(resumed),'--resume',str(resumed/'checkpoint-32')],check=True)
    result=compare(reference,resumed,lease['gpu_count'])
    from .verify_cheap import optimizer_compare, compact_technical_artifacts
    result['optimizer']=optimizer_compare(reference,resumed,step=36)
    atomic_json(ROOT/'output/deterministic-resume-result.json',result)
    curve=json.loads((parent/'learning-curve.json').read_text())['checkpoints']
    if [r['optimizer_step'] for r in curve]!=[0,28] or any(r['questions']!=128 for r in curve):
        raise RuntimeError('Missing matched validation evidence')
    timing=json.loads((parent/'throughput.json').read_text())
    if timing['status']!='measured_projection':raise RuntimeError('Missing original throughput evidence')
    steps={r['optimizer_step']:r['time'] for r in read_rows(reference/'steps.jsonl')}
    durations=[]
    for start in (28,32):
        starts=[r['started_at'] for rank in range(lease['gpu_count'])
                for r in read_rows(reference/f'rank-{rank:03d}/generation-events.jsonl')
                if r.get('mode')=='train' and r.get('optimizer_step')==start and r.get('status')=='started']
        durations.append(steps[start+4]-min(starts))
    slowest=max(timing['slowest_measured_rollout_seconds'],*durations)
    predicted=time.time()-lease['created_at']+600+1.2*(300*slowest+31*timing['slowest_validation_seconds'])+900
    atomic_json(ROOT/'output/full-run-projection.json',{'status':'measured_projection','projected_full_job_hours':predicted/3600,
                'projected_full_job_reserved_usd':predicted/3600*lease['budget_rate'],
                'deterministic_rollout_seconds':durations,'basis':'Conservative max of prior >=5 rollouts and 2 deterministic rollouts; informational, no cap.'})
    evidence.update(trainer_gpu_smoke=True,distributed_gpu_check=True,resume_equivalence=True,export_reload=True,
                    matched_validation=True,cost_projection=True,deterministic_resume_result=result,
                    prior_engineering_provenance_sha256=sha(parent/'run-provenance.json'))
    assert_launch_ready(protocol,evidence)
    full_evidence=ROOT/'output/readiness-at-start.json'
    atomic_json(full_evidence,evidence)
    command[command.index('--protocol')+1]=str(path)
    command[command.index('--readiness')+1]=str(full_evidence)
    compact_technical_artifacts(reference,resumed,checkpoint=32,final_step=36,extra=(reference/'checkpoint-28',))
    atomic_json(ROOT/'output/phase.json',{'phase':'main_training','time':time.time()})
    subprocess.run(command+['--out',str(ROOT/'output/training')],check=True)
    atomic_json(ROOT/'output/phase.json',{'phase':'finished','time':time.time()})
