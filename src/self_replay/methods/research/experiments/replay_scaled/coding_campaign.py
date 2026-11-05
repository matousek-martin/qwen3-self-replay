"""Sequential screening/train/eval on one rental, bounded per-stage and overall."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from .core import sha,write

OUT=Path('output/coding-campaign')


def read(p):return json.loads(Path(p).read_text())


def run(command, deadline, log):
    remaining=deadline-time.time()-180
    if remaining<=0:raise RuntimeError('Stage budget exhausted before '+str(command[1]))
    with Path(log).open('w') as f:
        p=subprocess.Popen(list(map(str,command)),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            rc=p.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=120)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            raise RuntimeError('Stage deadline reached; previous complete checkpoints preserved')
        if rc:raise RuntimeError(f'Worker exited {rc}; see {log}')


def prepare_evaluation():
    from experiments import study_eval as study,full_replay_eval
    cp=Path('output/training/checkpoints/group-000128');manifest=read(cp/'complete.json')
    if manifest['group']!=128 or manifest['world_size']!=1 or manifest['pilot']:raise RuntimeError('Invalid branch checkpoint')
    for name,digest in manifest['files'].items():
        if sha(cp/name)!=digest:raise RuntimeError('Candidate checkpoint SHA mismatch')
    stage=OUT/'evaluation-input';stage.mkdir();candidate='coding-replay-768-128'
    shutil.copy2('data/replay-evaluation/splits.json',stage/'splits.json')
    adapter=stage/'adapters'/candidate;shutil.copytree(cp/'adapter',adapter)
    original=read(adapter/'adapter_config.json');write(adapter/'adapter_config.json',dict(original,revision=study.BASE_REVISION))
    hashes=study.check_adapter(adapter)
    write(stage/'candidate-lock.json',dict(schema_version=1,candidate=candidate,primary_candidate=candidate,adapter_files_sha256=hashes,
        split_manifest_sha256=sha(stage/'splits.json'),locked_at=study.now(),selection='Registered final128 coding replay steps from parent768'))
    config=dict(candidates=['base',candidate],samples=8,coding_replay_branch='coding-replay-768-v1',parent_group=768,training_groups=128,
        baseline_cache_required=True,evaluation_only=True,training_authorized=False,evaluator_files_sha256=study.evaluator_hashes(),
        training_provenance=dict(parent_group=768,branch_groups=128,checkpoint_manifest_sha256=sha(cp/'complete.json'),
                                 original_adapter_config=original,original_weights_sha256=sha(adapter/'adapter_model.safetensors')))
    write(stage/'evaluation.json',config);full_replay_eval.inputs((stage/'evaluation.json').resolve())
    return stage/'evaluation.json'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lease=read('worker-lease.json');rate=lease['budget_rate'];hard=lease['hard_deadline']
    if lease['max_cost_usd']>18 or rate<=0:raise RuntimeError('Invalid campaign envelope')
    phase='screening'
    try:
        # Allocation, transfer and setup are charged to the screening envelope.
        screen_deadline=min(hard,lease['created_at']+4/rate*3600)
        os.environ['CODING_SCREENING_DEADLINE']=str(screen_deadline)
        write(OUT/'progress.json',dict(phase=phase,stage_deadline=screen_deadline,updated_at=time.time()))
        run([sys.executable,'-m','experiments.replay_scaled.preload'],screen_deadline,OUT/'preload.log')
        run([sys.executable,'-m','experiments.replay_scaled.coding_screen'],screen_deadline,OUT/'screening.log')
        if not read('output/coding-screen/completion.json')['eligible']:
            write(OUT/'completion.json',dict(complete=True,training_started=False,reason='insufficient_eligible_questions'))
            return
        phase='technical_check';begin=time.time();deadline=min(hard-600,begin+10/rate*3600)
        os.environ['CODING_TRAINING_DEADLINE']=str(deadline)
        write(OUT/'progress.json',dict(phase=phase,stage_deadline=deadline,updated_at=time.time()))
        command=['torchrun','--standalone','--nproc_per_node=1','-m','experiments.replay_scaled.worker','--data','output/coding-data']
        run(command+['--out','output/preflight-reference','--pilot'],deadline,OUT/'preflight-reference.log')
        run(command+['--out','output/preflight-resumed','--pilot','--resume','output/preflight-reference/checkpoints/group-000002'],deadline,OUT/'preflight-resumed.log')
        run([sys.executable,'-m','experiments.replay_scaled.check','--reference','output/preflight-reference','--resumed','output/preflight-resumed',
             '--world','1','--out','output/gpu-verification.json'],deadline,OUT/'gpu-verification.log')
        if not read('output/gpu-verification.json')['passed']:raise RuntimeError('GPU smoke/resume did not pass')
        phase='training';write(OUT/'progress.json',dict(phase=phase,stage_deadline=deadline,updated_at=time.time()))
        run(command+['--out','output/training'],deadline,OUT/'training.log')
        if not read('output/training/completion.json')['complete']:
            write(OUT/'completion.json',dict(complete=False,training_started=True,reason='training_stage_budget_stop'))
            return
        phase='full_evaluation';deadline=min(hard-300,time.time()+4/rate*3600)
        write(OUT/'progress.json',dict(phase=phase,stage_deadline=deadline,updated_at=time.time()))
        config=prepare_evaluation()
        run([sys.executable,'-m','experiments.vllm_compat','--out','output/vllm-compat.json'],deadline,OUT/'vllm-compat.log')
        run([sys.executable,'-m','experiments.vllm_lora_smoke','--config',config,'--out','output/adapter-routing.json'],deadline,OUT/'adapter-routing.log')
        run([sys.executable,'-m','experiments.full_replay_eval','--config',config,'--out','output/full-evaluation'],deadline,OUT/'full-evaluation.log')
        write(OUT/'completion.json',dict(complete=True,training_started=True,training_groups=128,evaluation_complete=True,
            comparison_sha256=sha(Path('output/full-evaluation/comparison.json')),finished_at=time.time()))
        write(OUT/'progress.json',dict(phase='complete',updated_at=time.time()))
    except Exception as e:
        write(OUT/'failure.json',dict(phase=phase,error=str(e),detected_at=time.time()))
        raise


if __name__=='__main__':main()
