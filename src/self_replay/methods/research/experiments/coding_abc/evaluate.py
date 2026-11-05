"""Frozen official evaluation of the last complete expandedABC checkpoint."""
import shutil,subprocess,sys,time
from pathlib import Path
from experiments.opencoder_sft.runtime import read,write,sha

def main():
    cfg=read('data/coding-abc/plan.json')
    if not cfg.get('final_official_evaluation'):return
    from experiments import study_eval as study,full_replay_eval
    out=Path('output/training');latest=read(out/'latest-checkpoint.json');group=latest['group']
    if group<=0:raise RuntimeError('No trained checkpoint for official evaluation')
    cp=out/'checkpoints'/f'group-{group:06d}';m=read(cp/'complete.json')
    for file,h in m['files'].items():
        if sha(cp/file)!=h:raise RuntimeError('Incomplete final adapter')
    stage=Path('output/abc-evaluation-input');stage.mkdir();candidate=f"coding-abc-{cfg['arm']}-{group}"
    shutil.copy2('data/replay-evaluation/splits.json',stage/'splits.json')
    adapter=stage/'adapters'/candidate;shutil.copytree(cp/'adapter',adapter)
    original=read(adapter/'adapter_config.json');write(adapter/'adapter_config.json',dict(original,revision=study.BASE_REVISION))
    write(stage/'candidate-lock.json',dict(schema_version=1,candidate=candidate,primary_candidate=candidate,
        adapter_files_sha256=study.check_adapter(adapter),split_manifest_sha256=sha(stage/'splits.json'),locked_at=study.now(),
        selection='Preregistered last complete checkpoint at one-pass or morning deadline; no quality-based selection'))
    config=dict(candidates=['base',candidate],samples=8,coding_abc_branch='expanded-v2',arm=cfg['arm'],parent_group=768,
        training_groups=group,target_groups=cfg['steps'],baseline_cache_required=True,evaluation_only=True,training_authorized=False,
        evaluator_files_sha256=study.evaluator_hashes(),training_provenance=dict(parent_group=768,arm=cfg['arm'],training_groups=group,
            target_groups=cfg['steps'],checkpoint_manifest_sha256=sha(cp/'complete.json'),plan_sha256=sha('data/coding-abc/plan.json'),
            original_weights_sha256=sha(adapter/'adapter_model.safetensors')))
    write(stage/'evaluation.json',config);full_replay_eval.inputs((stage/'evaluation.json').resolve())
    p=read(out/'progress.json');write(out/'progress.json',dict(p,phase='full_evaluation',updated_at=time.time()))
    deadline=min(read('worker-lease.json')['hard_deadline'],cfg['overall_deadline'])-120
    for command in ([sys.executable,'-m','experiments.vllm_compat','--out','output/vllm-compat.json'],
        [sys.executable,'-m','experiments.vllm_lora_smoke','--config',str(stage/'evaluation.json'),'--out','output/adapter-routing.json'],
        [sys.executable,'-m','experiments.full_replay_eval','--config',str(stage/'evaluation.json'),'--out','output/full-evaluation']):
        remaining=deadline-time.time()
        if remaining<=0:raise RuntimeError('Morning evaluation deadline reached; checkpoint preserved')
        subprocess.run(command,check=True,timeout=remaining)
    write(out/'progress.json',dict(p,phase='complete',updated_at=time.time(),official_evaluation_complete=True))

if __name__=='__main__':main()
