"""Bounded four-domain pilots with matched validation and measured hardware timing."""
import json,subprocess,sys,time
from pathlib import Path
from experiments.shorterbetter.config import assert_launch_ready
from experiments.shorterbetter.distributed import atomic_json
from experiments.shorterbetter.throughput import read_rows
ROOT=Path(__file__).resolve().parents[3]

def projection(output,protocol,lease):
    output=Path(output);ends={r['optimizer_step']:r['time'] for r in read_rows(output/'steps.jsonl')}
    times=[ends[r['optimizer_step']+4]-r['time'] for r in read_rows(output/'rollout-starts.jsonl') if r['optimizer_step']+4 in ends]
    evals=[r['eval_runtime'] for r in read_rows(output/'rank-000/metrics.jsonl') if 'eval_runtime' in r]
    if len(times)<2 or not evals:return {'status':'insufficient_measurements','rollout_seconds':times,'evaluation_seconds':evals}
    measured=times[1:] # First rollout includes one-time optimizer allocation.
    panel=protocol['data']['monitor_subset_questions'];full_panel=64;eval_scale=full_panel/panel
    setup=max(0,read_rows(output/'rollout-starts.jsonl')[0]['time']-lease['created_at']-evals[0])
    overhead=setup+900
    if any(protocol.get('execution', {}).get(k) for k in ('accelerated_multidomain_benchmark', 'hardware_comparison', 'multidomain_full_training')):
        targets = {}
        for iterations in sorted({100,300,protocol.get('training',{}).get('rollout_batches',300)}):
            validations = iterations//10 + 1
            a = overhead + iterations*min(measured) + validations*min(evals)*eval_scale
            b = overhead + 1.2*(iterations*max(measured) + validations*max(evals)*eval_scale)
            targets[str(iterations)] = {'hours_range':[a/3600,b/3600],
                'conservative_cost_range_usd':[a/3600*lease['budget_rate'],b/3600*lease['budget_rate']],
                'internal_validations':validations,'validation_questions':full_panel}
        return {'status':'early_measured_projection','gpu':lease['gpu'],'measured_gpu_count':lease['gpu_count'],
            'rollout_seconds':times,'steady_rollout_samples':len(measured),'warmup_rollouts_excluded':1,
            'evaluation_seconds':evals,'targets':targets,'setup_seconds':setup,'archive_allowance_seconds':900,
            'cost_basis':'Canonical reservation rate, not an invoice or future provider price.',
            'limitations':['Only early full-length rollouts; 20% planning margin is not a statistical confidence interval.',
                'Validation scaled linearly from16 to64questions; a larger confirmation and competition evaluation are additional.',
                'Changing dataset size does not measure its benefit; timings assume unchanged sequence lengths and backend.',
                'Benchmark does not authorize launching100 or300rollouts.']}
    low=overhead+300*min(measured)+31*min(evals)*eval_scale
    high=overhead+1.2*(300*max(measured)+31*max(evals)*eval_scale)
    return {'status':'early_measured_projection','gpu':'NVIDIA GeForce RTX 4090','measured_gpu_count':1,
        'rollout_seconds':times,'warmup_rollouts_excluded':1,'steady_rollout_samples':len(measured),
        'evaluation_seconds':evals,'pilot_validation_questions':panel,'projected_validation_questions':full_panel,
        'target_rollouts':300,'target_prompts_per_rollout':64,'target_generations_per_prompt':8,
        'single4090_hours_range':[low/3600,high/3600],
        'single4090_cost_range_usd':[low/3600*lease['rate'],high/3600*lease['rate']] if 'rate' in lease else None,
        'two4090_scenarios_hours':{'ideal_2x':(overhead+(low-overhead)/2)/3600,'no_speedup':high/3600},
        'two4090_measured':False,'setup_seconds':setup,'archive_allowance_seconds':900,
        'limitations':['Only early pilot rollouts; not a guarantee.','64-question validation projected linearly from the smaller pilot panel.',
          'TwoGPU timing is hypothetical; native DDP full-length memory and communication must be measured, can be slower than singleGPU.',
          'No official competition evaluation included; no inference that all4 domains improved.']}

def run(protocol,path,lease,evidence,resume=None):
    accelerated = protocol['execution'].get('accelerated_multidomain_benchmark', False)
    gpu, world, rollouts = ('NVIDIA A100-SXM4-80GB', 2, 2) if accelerated else ('NVIDIA GeForce RTX 4090', 1, 3)
    comparison = protocol['execution'].get('hardware_comparison')
    full = protocol['execution'].get('multidomain_full_training') is True
    if full:
        from .full_run import validate
        gpu, world, rollouts = validate(protocol)
        accelerated = True
    if comparison:
        from experiments.shorterbetter.hardware_benchmark import validate, ROLLOUTS
        gpu, world, _, _ = validate(protocol)
        rollouts = ROLLOUTS
        accelerated = True
    if (lease['gpu']!=gpu or lease['gpu_count']!=world
        or protocol['training']['rollout_batches']!=rollouts or protocol['training']['prompts_per_rollout_batch']!=64
        or protocol['training']['generations_per_prompt']!=8):raise ValueError('Wrong pilot dimensions')
    assert_launch_ready(protocol,evidence)
    from .rewards import code_check
    if not code_check('print(int(input())+1)',{'inputs':['2'],'outputs':['3']}):raise RuntimeError('Worker code sandbox positive fixture failed')
    if code_check('import socket; socket.socket(); print(3)',{'inputs':['2'],'outputs':['3']}):raise RuntimeError('Worker code sandbox network isolation failed')
    atomic_json(ROOT/'output/sandbox-check.json',{'status':'passed','chroot':True,'uid':65534,'seccomp':True,'no_network':True})
    ready=ROOT/'output/readiness-at-start.json';atomic_json(ready,evidence)
    out=ROOT/'output/training'
    atomic_json(ROOT/'output/phase.json',{'phase':'main_training' if full else ('hardware_comparison' if comparison else ('multidomain_a100x2_benchmark' if accelerated else 'multidomain_pilot')),
        'time':time.time(),'target_rollouts':rollouts})
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1',f'--nproc_per_node={world}','--max_restarts=0',
       '-m','experiments.shorterbetter.train','--protocol',str(path),'--arm','shorterbetter','--seed','17','--execute',
       '--readiness',str(ready),'--job-name',lease['job_name'],'--out',str(out)]
    if resume is not None:
        if not protocol['execution'].get('full_run_continuation'):
            raise ValueError('Resume requires a registered continuation')
        command += ['--resume', str(resume)]
    if comparison:
        topology = subprocess.run(['nvidia-smi', 'topo', '-m'], capture_output=True, text=True, check=True)
        (ROOT/'output/gpu-topology.txt').write_text(topology.stdout)
        with (ROOT/'output/gpu-utilization.csv').open('w') as metrics:
            monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,name,memory.used,utilization.gpu,utilization.memory',
                                        '--format=csv', '-l', '2'], stdout=metrics, stderr=subprocess.DEVNULL)
            try:
                subprocess.run(command, check=True)
            finally:
                monitor.terminate()
                monitor.wait(timeout=10)
    else:
        subprocess.run(command,check=True)
    finished=json.loads((out/'training-finished.json').read_text())
    atomic_json(ROOT/'output/pilot-projection.json',projection(out,protocol,lease))
    if comparison:
        from experiments.shorterbetter.hardware_benchmark import measure
        measured = measure(out, lease)
        from experiments.shorterbetter.prepare import sha
        measured['comparison_identity'] = {'model': protocol['model'],
            'data_manifest_sha256': protocol['data']['manifest_sha256'],
            'trainer_source_sha256': sha(ROOT/'experiments/shorterbetter/train.py'),
            'cpu_verification_sha256': evidence['cpu_verification_sha256'],
            'seed': protocol['seeds'][0], 'prompts_per_rollout':64, 'generations_per_prompt':8}
        expected = protocol['data'].get('expected_train_prompt_ids_by_step')
        if expected:
            for step, ids in measured['prompt_ids_by_step'].items():
                if step not in expected or sorted(set(ids)) != expected[step]:
                    raise RuntimeError('Measured prompts differ from the matched hardware plan')
        atomic_json(ROOT/'output/hardware-benchmark.json', measured)
    state=json.loads((out/'trainer_state.json').read_text());step=state['global_step'];checkpoint=out/f'checkpoint-{step}'
    required=['model.safetensors','optimizer.pt','scheduler.pt','trainer_state.json',
              *([f'rng_state_{rank}.pth' for rank in range(world)] if world > 1 else ['rng_state.pth'])]
    if any(not (checkpoint/f).is_file() for f in required):raise RuntimeError('Missing complete pilot checkpoint')
    # Export reload verifies saved tensor names/shapes, without retaining a second GPU model.
    from safetensors import safe_open
    with safe_open(str(checkpoint/'model.safetensors'),framework='pt',device='cpu') as a, safe_open(str(out/'model/model.safetensors'),framework='pt',device='cpu') as b:
        if set(a.keys())!=set(b.keys()):raise RuntimeError('Export tensor set mismatch')
        import torch
        for k in a.keys():
            if not torch.equal(a.get_tensor(k),b.get_tensor(k)):raise RuntimeError('Export weights mismatch '+k)
    curve=json.loads((out/'learning-curve.json').read_text())
    matched = all(len(r['checkpoints'])>=2 and r['checkpoints'][-1]['optimizer_step']==step for r in curve['domains'].values())
    if accelerated:
        from collections import Counter, defaultdict
        groups = defaultdict(list)
        for rank in range(world):
            folder=out/f'rank-{rank:03d}'
            for filename in ('gpu-memory-check.json','gpu-sampling-check.json','precision-check.json'):
                if json.loads((folder/filename).read_text())['status']!='passed':raise RuntimeError('GPU check missing: '+filename)
            updates=read_rows(folder/'parameter-updates.jsonl')
            if len(updates)!=step or not all(r['rank_samples_identical'] for r in updates):
                raise RuntimeError('Distributed parameter-update verification failed')
            for row in read_rows(folder/'rollouts.jsonl'):
                if row['split']=='train':groups[row['optimizer_step']].append(row)
        for rows in groups.values():
            if (len(rows)!=512 or Counter(r['task'] for r in rows)!={d:128 for d in ('math','coding','tools','instructions')}
                    or len(set(r['prompt_id'] for r in rows))!=64
                    or set(Counter(r['prompt_id'] for r in rows).values())!={8}):
                raise RuntimeError('Actual distributed rollout groups differ from registered exposure')
        atomic_json(ROOT/'output/distributed-pilot-check.json',{'status':'passed','world_size':world,
            'verified_generation_steps':sorted(groups),'optimizer_updates':step,'global_groups_balanced':True,
            'sampled_rank_weights_identical':True,'exact_split_run_resume_tested':bool(full and evidence.get('resume_equivalence'))})
    result={'status':'passed' if step==rollouts*4 and matched else 'stopped_early',
      'optimizer_steps':step,'rollout_batches':step/4,'full_training_started':full,
      'matched_final_validation':matched,'measured_gpu_count':world,'gpu':gpu,
      'checkpoint':str(checkpoint.relative_to(ROOT/'output')),'export_reload':'bit-identical',
      'domain_validation':{d:r['checkpoints'][-1] for d,r in curve['domains'].items()},
      'limitation':('One seed, repeatedly inspected development panel; no independent efficacy confirmation.' if full else
                   'Four development questions/domain: operational pilot, not evidence of stable improvement or convergence.')}
    atomic_json(ROOT/'output'/('multidomain-training-result.json' if full else 'multidomain-pilot-result.json'),result)
    atomic_json(ROOT/'output/phase.json',{'phase':'finished','time':time.time()})
