"""Prepare a named, bounded single4090 pilot; canonical cloud owns all rentals."""
import argparse,fcntl,hashlib,json,os,re,shutil,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import cloud
from experiments.shorterbetter.prepare import sha
from experiments.shorterbetter.train import check_data
from experiments.shorterbetter.config import training_arguments,assert_launch_ready
from ops.isolated_cloud import seal

BASE=ROOT/'lab/experiments/shorterbetter-multidomain'

def prepare(accelerated=False):
    data=BASE/'data-v3';manifest=check_data(data)
    if manifest['domain_counts']['train']!={d:1000 for d in ('math','coding','tools','instructions')}:raise ValueError('Need exactly1000/domain')
    tests=BASE/('a100x2-tests.log' if accelerated else 'tests.log')
    counts=re.findall(r'Ran (\d+) tests',tests.read_text())
    if not counts or int(counts[-1])<119 or not tests.read_text().rstrip().endswith('OK'):raise ValueError('Tests not passed')
    test_count=int(counts[-1])
    ddp=Path('/tmp/sb-multidomain-a100x2-cpu-20261028/ddp-result.json' if accelerated else '/tmp/sb-multidomain-shared-regression-20261028/ddp-result.json')
    if json.loads(ddp.read_text())['status']!='passed':raise ValueError('CPU regression missing')
    if accelerated and json.loads(ddp.read_text())['mode']!='multidomain':raise ValueError('Four-domain DDP fixture missing')
    terminal=Path('/tmp/sb-multidomain-a100x2-terminal-20261028/ddp-result.json')
    if accelerated and (json.loads(terminal.read_text())['status']!='passed'
            or json.loads(terminal.read_text())['mode']!='multidomain-stop'):
        raise ValueError('Terminal evaluation/checkpoint fixture missing')
    if cloud.list_pods():raise RuntimeError('Reconcile current provider pods before this sequential pilot')
    state=json.loads((cloud.STATE/'ledger.json').read_text())
    old=[j for n,j in state['jobs'].items() if n.startswith('shorterbetter-scaled-multidomain-')]
    if any(not j.get('released_at') for j in old):raise RuntimeError('Existing pilot needs reconciliation')
    cap=10. if accelerated else 5.
    scope='shorterbetter_multidomain_a100x2_benchmark' if accelerated else 'shorterbetter_multidomain_pilot'
    prefix='shorterbetter-scaled-multidomain-a100x2-' if accelerated else 'shorterbetter-scaled-multidomain-'
    sequence=[j for n,j in state['jobs'].items() if n.startswith(prefix)]
    if sum(j.get('charged_estimate_usd',0) for j in sequence)+cap>(20 if accelerated else 10):raise RuntimeError('Pilot repair allowance exhausted; preserve evidence')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dt%H%M%Sz').lower();name=prefix+stamp
    directory=BASE/(('a100x2-' if accelerated else 'pilot4090-')+stamp);directory.mkdir()
    p=json.loads((ROOT/'studies/shorterbetter/protocol.json').read_text())
    p.update(id='shorterbetter-four-domains-pilot-v1',status='locked',paid_launch_ready=True,
             question='Does the four-domain trainer function correctly and what is full-length single4090 throughput?',
             scope='Technical/early four-domain pilot; not exact paper reproduction, not final competition evidence.')
    p['training'].update(rollout_batches=3,optimizer_updates_total=12,training_completions_per_arm=1536,
        parameter_dtype='float32',world_size_initial=1,backend='Pinned TRL/vLLM; FP32 AdamW foreach=False; BF16 compute; exact chunked projection')
    p['data']={'directory':'data/multidomain','sources':manifest['sources'],'train_questions':4000,'per_domain':1000,
        'monitor_subset_questions':16,'development_questions':256,'confirmation_questions':256,
        'screening':manifest['screening'],'manifest_sha256':sha(data/'manifest.json')}
    p['arms']={'shorterbetter':p['arms']['shorterbetter']};p['seeds']=[17]
    p['reward']['correctness']='Math boxed numeric; APPS sandbox tests; tool-call structured equality; supported IF constraints. Final answer only; original SOL formula.'
    p['monitoring'].update(every_rollout_batches=3,baseline_at_step_zero=True,per_domain=True,automatic_parameter_changes=False)
    from experiments.shorterbetter.stopping import DEFAULTS
    p['stopping']={**DEFAULTS,'regression_min_rollouts':3,'regression_patience':1,'minimum_rollouts':50}
    p['execution']={'scaled':True,'phase':'benchmark','multidomain':True,'world_size':1,'gpu':'NVIDIA GeForce RTX 4090',
        'vllm_gpu_memory_utilization':.45,'eval_batch_size':8,'save_total_limit':1,'deterministic_training':True,
        'deterministic_ddp':False,'single_gpu_chunked_projection':True,'adam_foreach':False}
    p['budget']={'proposed_total_pilot_cap_usd':5.,'cap_authorized':True,'scope':'shorterbetter_multidomain_pilot','repair_sequence_ceiling_usd':10.}
    p['authorization']={'preparation':True,'pilot_single4090_authorized':True,'technical_repairs_authorized':True,
        'full_training_authorized':False,'telegram_progress_authorized':True,
        'user_instruction':'Prepare1000questions/domain; run small pilot on1RTX4090, stop after verification; estimate1/2GPU duration; fix errors until functional.'}
    p['full_run_target']={'rollout_batches':300,'prompts_per_rollout_batch':64,'generations_per_prompt':8,'optimizer_updates':1200,'launch_authorized':False,'validation_questions':64,'validation_every_rollouts':10}
    p['phases']=['Memory/sampling and sandbox preflight.','Fresh base validation16questions across4domains, 3fullrollouts, matched final validation.','Checkpoint/export verification, archive/release and report.']
    p['selection']={'rule':'No final efficacy selection from this tiny panel; retain pilot checkpoint.'}
    if accelerated:
        p.update(id='shorterbetter-four-domains-a100x2-benchmark-v1',
            question='Does two-A100 full-length DDP work on the unchanged four-domain data, and how long would100/300rollouts take?')
        p['training'].update(rollout_batches=2,optimizer_updates_total=8,training_completions_per_arm=1024,
            world_size_initial=2,backend='Pinned TRL/vLLM; native DDP forward; FP32 AdamW foreach=False; BF16 compute; deterministic rank-order reduction')
        p['monitoring'].update(every_rollout_batches=1,terminal_evaluation_required=True)
        p['execution'].update(world_size=2,gpu='NVIDIA A100-SXM4-80GB',deterministic_ddp=True,
            single_gpu_chunked_projection=False,accelerated_multidomain_benchmark=True,
            terminal_evaluation_required=True,shutdown_reserve_seconds=1200)
        p['stopping'].update(regression_min_rollouts=1,regression_patience=2)
        p['budget']={'proposed_total_pilot_cap_usd':cap,'cap_authorized':True,'scope':scope,'repair_sequence_ceiling_usd':20.}
        p['authorization']={'preparation':True,'pilot_a100x2_authorized':True,'technical_repairs_authorized':True,
            'full_training_authorized':False,'telegram_progress_authorized':True,
            'user_instruction':'Test our training on two A100s, report measured duration, then consider100iterations; do not automatically launch100or300.'}
        p['full_run_target']['additional_projection_rollouts']=100
        p['phases']=['Full-length memory/sampling and sandbox preflight on bothA100s.',
            'Fresh matched baseline, two full64x8rollouts, validation after each rollout; force terminal evaluation if stopped early.',
            'Verify distributed exposure/updates, full checkpoint and export; archive/release, Telegram100/300iteration forecasts.']
        p['lineage']={'kind':'Fresh base model for a matched hardware benchmark; not a resume of checkpoint10.',
            'reference_single4090_job':'shorterbetter-scaled-multidomain-20261028t121420z',
            'train_sha256':manifest['files']['train.jsonl'],'monitor_sha256':manifest['files']['monitor.jsonl']}
    cloud.write(directory/'protocol.json',p)
    shutil.copy2(tests,directory/'tests.log');shutil.copy2(ddp,directory/'ddp-result.json')
    if accelerated:shutil.copy2(terminal,directory/'terminal-ddp-result.json')
    proof={'status':'cpu_passed_gpu_pending','tests':test_count,'data_audit':str(data/'audit.json'),'data_audit_sha256':sha(data/'audit.json'),
        'test_log_sha256':sha(tests),'ddp_result_sha256':sha(ddp),
        'source_sha256':{str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'experiments/shorterbetter').rglob('*.py')}}
    cloud.write(directory/'cpu-verification.json',proof)
    evidence={k:True for k in ('data_audit','reward_parity','backend_cpu_parity','distributed_cpu_check','stopping_tests',
        'backup_release_integration','four_domain_verifiers','balanced_sampler','data_domains')}
    evidence.update(protocol_sha256=sha(directory/'protocol.json'),budget_reservation=False,pilot_cap_enforced=False,
        trainer_gpu_smoke=False,resume_equivalence=False,export_reload=False,matched_validation=False,cost_projection=False)
    assert_launch_ready(p,{**evidence,'budget_reservation':True,'pilot_cap_enforced':True})
    cloud.write(directory/'readiness.json',evidence)
    cloud.write(directory/'resolved-training-arguments.json',training_arguments(p,'shorterbetter',17,'output/training'))
    release=ROOT/'lab/isolated'/name;release.mkdir()
    shutil.copytree(ROOT/'experiments',release/'experiments',ignore=shutil.ignore_patterns('.git','__pycache__','.venv*','runs','.env*','*.log','launch'))
    shutil.copytree(ROOT/'ThinkingCap-Test',release/'ThinkingCap-Test',ignore=shutil.ignore_patterns('.git','__pycache__','.venv*','runs','.env*','*.log','questions','img'))
    launch=release/'experiments/shorterbetter/launch';launch.mkdir()
    for f in ('protocol.json','readiness.json','cpu-verification.json'):shutil.copy2(directory/f,launch/f)
    payload=release/'lab/experiments/data';payload.mkdir(parents=True)
    oldpayload=ROOT/'lab/isolated/shorterbetter-retry4-20261027T075430Z/lab/experiments/data'
    shutil.copytree(oldpayload/'wheels',payload/'wheels',copy_function=os.link)
    shutil.copytree(oldpayload/'shorterbetter/sources',payload/'shorterbetter/sources',copy_function=os.link)
    (payload/'pilot.jsonl').write_text('{}\n') # Existing allocator readiness sentinel, not consumed as training.
    (payload/'multidomain').mkdir()
    for f in ('train.jsonl','monitor.jsonl','manifest.json','audit.json'):shutil.copy2(data/f,payload/'multidomain'/f)
    shutil.copy2(oldpayload.parents[1]/'gpu-requirements.lock',release/'lab/gpu-requirements.lock')
    digest=seal(release)
    auth={'scope':scope,'authorized':True,'job_name':name,'gpu':p['execution']['gpu'],
      'gpu_count':p['execution']['world_size'],'max_cost_usd':cap,'max_runtime_hours':2 if accelerated else 3,
      'max_gpu_rate':1.79 if accelerated else .89,'pilot_rollouts':p['training']['rollout_batches'],
      'source_user_instruction':p['authorization']['user_instruction'],'full_training_authorized':False,'payload_sha256':digest}
    cloud.write(directory/'authorization.json',auth);cloud.write(cloud.STATE/'authorizations'/f'{name}.json',auth)
    current={'directory':str(directory.relative_to(ROOT)),'release':str(release.relative_to(ROOT)),'job_name':name,'authorization':str((directory/'authorization.json').relative_to(ROOT))}
    cloud.write(BASE/'current.json',current);cloud.write(ROOT/'lab/experiments/shorterbetter-scaled/current.json',current)
    registry=ROOT/'lab/experiments/shorterbetter-scaled/notifications/jobs.json'
    with (registry.parent/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);rows=json.loads(registry.read_text());rows[name]={'telegram_authorized':True,'directory':str(directory)};cloud.write(registry,rows)
    print(json.dumps({**current,'payload_sha256':digest},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a100x2',action='store_true')
    prepare(parser.parse_args().a100x2)
