"""One bounded SFT pilot, complete checkpoint before each development validation."""
import os
from pathlib import Path
import shutil
import sys
import time
from .runtime import DATA,read,write,sha,check_inputs,check_lease,SCOPE
from experiments.replay_scaled.coding_campaign import run

OUT=Path('output/opencoder-campaign')


def evaluation_input():
    from experiments import study_eval as study,full_replay_eval
    cp=Path('output/training/checkpoints/step-000084');m=read(cp/'complete.json')
    if m['step']!=84 or m['technical']:raise RuntimeError('Invalid final SFT checkpoint')
    for f,h in m['files'].items():
        if sha(cp/f)!=h:raise RuntimeError('Final checkpoint corrupt')
    stage=OUT/'evaluation-input';stage.mkdir();candidate='opencoder-sft-768-84'
    shutil.copy2('data/replay-evaluation/splits.json',stage/'splits.json')
    adapter=stage/'adapters'/candidate;shutil.copytree(cp/'adapter',adapter)
    original=read(adapter/'adapter_config.json');write(adapter/'adapter_config.json',dict(original,revision=study.BASE_REVISION))
    write(stage/'candidate-lock.json',dict(schema_version=1,candidate=candidate,primary_candidate=candidate,
          adapter_files_sha256=study.check_adapter(adapter),split_manifest_sha256=sha(stage/'splits.json'),locked_at=study.now(),
          selection='Registered final84 SFT updates; parent768,2000distinct coding questions'))
    config=dict(candidates=['base',candidate],samples=8,opencoder_sft_branch=SCOPE,parent_group=768,optimizer_steps=84,
        baseline_cache_required=True,evaluation_only=True,training_authorized=False,evaluator_files_sha256=study.evaluator_hashes(),
        training_provenance=dict(parent_group=768,optimizer_steps=84,checkpoint_manifest_sha256=sha(cp/'complete.json'),
             original_adapter_config=original,original_weights_sha256=sha(adapter/'adapter_model.safetensors')))
    write(stage/'evaluation.json',config);full_replay_eval.inputs((stage/'evaluation.json').resolve())
    return stage/'evaluation.json'


def main():
    OUT.mkdir(parents=True,exist_ok=True);cfg=check_inputs();lease=check_lease()
    if not 0<lease['max_cost_usd']<=15:raise RuntimeError('Invalid envelope')
    hard=lease['hard_deadline'];train_deadline=min(hard-600,lease['created_at']+10/lease['budget_rate']*3600)
    phase='setup'
    def phase_set(value):
        nonlocal phase
        phase=value;write(OUT/'progress.json',dict(phase=value,updated_at=time.time()))
    def call(module,args,log,deadline=train_deadline):
        check_lease();run([sys.executable,'-m',module]+list(map(str,args)),deadline,OUT/(log+'.log'))
    try:
        call('experiments.replay_scaled.preload',[],'preload')
        call('experiments.vllm_compat',['--out','output/vllm-compat.json'],'vllm-compat')
        phase_set('technical_check')
        call('experiments.opencoder_sft.train',['--technical','--until',2,'--out','output/smoke-full'],'smoke-full')
        call('experiments.opencoder_sft.train',['--technical','--until',2,'--out','output/smoke-resumed','--resume','output/smoke-full/checkpoints/step-000001'],'smoke-resumed')
        call('experiments.opencoder_sft.check',[],'resume-check')
        if not read('output/gpu-verification.json')['passed']:raise RuntimeError('Technical check failed')
        phase_set('baseline')
        call('experiments.opencoder_sft.validate',['--step',0,'--adapter',DATA/'adapter'],'validation-0')
        previous=None
        for step in [16,32,48,64,80,84]:
            phase_set('training');args=['--until',step]
            if previous:args+=['--resume',previous]
            call('experiments.opencoder_sft.train',args,f'train-{step}')
            previous=Path('output/training/checkpoints')/f'step-{step:06d}'
            phase_set('validation')
            call('experiments.opencoder_sft.validate',['--step',step,'--adapter',previous/'adapter'],f'validation-{step}')
        write('output/training/completion.json',dict(complete=True,optimizer_steps=84,coding_questions=2000))
        phase_set('full_evaluation');config=evaluation_input()
        deadline=min(hard-300,time.time()+4/lease['budget_rate']*3600)
        call('experiments.vllm_lora_smoke',['--config',config,'--out','output/adapter-routing.json'],'adapter-routing',deadline)
        call('experiments.full_replay_eval',['--config',config,'--out','output/full-evaluation'],'full-evaluation',deadline)
        write(OUT/'completion.json',dict(complete=True,evaluation_complete=True,optimizer_steps=84,
            comparison_sha256=sha(Path('output/full-evaluation/comparison.json')),finished_at=time.time()))
        phase_set('complete')
    except Exception as e:
        write(OUT/'failure.json',dict(phase=phase,error=str(e),detected_at=time.time()));raise

if __name__=='__main__':main()
