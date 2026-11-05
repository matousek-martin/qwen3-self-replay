"""Frozen full-weight candidate versus a fresh baseline; official evaluator unchanged."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

from experiments import study_eval as study
from experiments.full_replay_eval import COUNTS
from experiments.shorterbetter.hardware import verify_worker


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def inputs(config_path):
    config_path=Path(config_path).resolve()
    config=json.loads(config_path.read_text())
    if (config.get('evaluation_only') is not True or config.get('training_authorized') is not False
            or config.get('samples')!=8 or config.get('evaluation_seed')!=9001
            or config.get('candidates')!=['base','shorterbetter-final']
            or config.get('model')!=study.BASE_MODEL or config.get('revision')!=study.BASE_REVISION):
        raise ValueError('Unregistered full paired evaluation')
    source=study.REPO/'questions';rows=study.read_jsonl(source/'questions.jsonl')
    if (dict(Counter(r['dataset'] for r in rows))!=COUNTS
            or len({(r['dataset'],r['id']) for r in rows})!=2759
            or digest(source/'questions.jsonl')!=config['cohort_sha256']
            or digest(source/'manifest.json')!=config['questions_manifest_sha256']
            or study.evaluator_hashes()!=config['evaluator_files_sha256']):
        raise ValueError('Frozen official evaluator or full cohort changed')
    candidate=study.inside(config_path.parent,config['candidate']['path'])
    expected=config['candidate']['files_sha256']
    actual={f.name:digest(f) for f in candidate.iterdir() if f.is_file()}
    if actual!=expected or 'model.safetensors' not in actual or 'config.json' not in actual:
        raise ValueError('Frozen full-weight candidate changed')
    early = (config.get('paired_short_evaluation') is True
             and config['candidate'].get('optimizer_step') in range(20,121,20)
             and config['candidate'].get('source_training_job','').startswith('shorterbetter-scaled-paired-'))
    if (actual['model.safetensors']!=config['candidate']['original_weights_sha256']
            or (config['candidate'].get('optimizer_step') not in (160,426) and not early)):
        raise ValueError('Candidate is not an authorized checkpoint')
    original=config['candidate']['original_model_config']
    inference=json.loads((candidate/'config.json').read_text())
    if dict(original,dtype='bfloat16')!=inference:
        raise ValueError('Only explicit BF16 inference metadata may differ from the original config')
    return config,source,rows,candidate


def tokenizer_parity(rows,candidate,out):
    from transformers import AutoTokenizer,AutoConfig
    base=AutoTokenizer.from_pretrained(study.BASE_MODEL,revision=study.BASE_REVISION)
    new=AutoTokenizer.from_pretrained(candidate)
    cfg=AutoConfig.from_pretrained(study.BASE_MODEL,revision=study.BASE_REVISION)
    if str(cfg.dtype) not in ('bfloat16','torch.bfloat16'):
        raise ValueError('Pinned baseline inference precision differs from registered BF16')
    # Use the exact official chat-template call used by build_jobs.
    for row in rows:
        a=base.apply_chat_template(row['messages'],tokenize=True,add_generation_prompt=True,enable_thinking=True)
        b=new.apply_chat_template(row['messages'],tokenize=True,add_generation_prompt=True,enable_thinking=True)
        if a!=b:raise ValueError('Candidate changed an official tokenized prompt')
    if base.eos_token_id!=new.eos_token_id:
        raise ValueError('Candidate EOS token differs')
    study.write_json(out/'tokenizer-parity.json',{'status':'passed','questions':len(rows),
        'same_tokenized_prompts':True,'inference_dtype':'bfloat16','original_FP32_weights_unchanged':True})


def main():
    signal.signal(signal.SIGTERM,study.interrupt)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args();args.config=args.config.resolve()
    config,source,rows,candidate=inputs(args.config)
    if args.check_only:
        print(json.dumps({'status':'inputs_verified','questions':len(rows),'responses_per_model':22072,'training':False}));return
    out=args.out.resolve()
    if out.exists():raise ValueError('No implicit restart into an existing evaluation directory')
    out.mkdir(parents=True)
    lease=json.loads((study.ROOT/'worker-lease.json').read_text())
    if lease['job_name']!=config['job_name'] or lease['max_cost_usd']!=config['max_cost_usd']:
        raise ValueError('Evaluation lease identity/cap mismatch')
    names=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).splitlines()
    verify_worker(names,lease)
    gpu_command=['nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv,noheader']
    identity=subprocess.check_output(gpu_command,text=True).strip()
    study.write_json(out/'provenance.json',dict(config,gpu_identity=identity,started_at=study.now(),
        matched_baseline_regenerated=None,same_physical_worker_required=False,fresh_holdout=False,
        runner_sha256=digest(__file__),lease=lease))
    study.preflight_nltk();tokenizer_parity(rows,candidate,out)
    study.write_json(out/'phase.json',{'phase':'grader_smoke','updated_at':time.time()})
    study.run_process([sys.executable,study.REPO/'evaluate.py','--questions',source,
        '--out',out/'grader-smoke','--smoke','--workers',4],out/'logs/grader-smoke.log',None)
    smoke=json.loads((out/'grader-smoke/results.json').read_text())
    if smoke.get('failed_benchmarks'):raise ValueError('Official grader smoke failed')
    for name in config['candidates']:
        if subprocess.check_output(gpu_command,text=True).strip()!=identity:
            raise RuntimeError('Physical evaluator GPU changed')
        inputs(args.config)
        if time.time()>=lease['hard_deadline']-600:
            raise RuntimeError('Insufficient remaining evaluation lease')
        study.write_json(out/'phase.json',{'phase':name,'updated_at':time.time()})
        study.write_json(out/'status'/f'{name}.json',{'state':'running','started_at':study.now()})
        run=out/'runs'/name
        command=[sys.executable,study.REPO/'evaluate.py','--questions',source,'--out',run,
                 '--samples',8,'--seed',9001,'--workers',4]
        if name!='base':command+=['--model',candidate]
        study.run_process(command,out/'logs'/f'{name}.log',max(1,lease['hard_deadline']-time.time()-300))
        model,revision=(study.BASE_MODEL,study.BASE_REVISION) if name=='base' else (str(candidate),None)
        result=study.validate_records(run,rows,8,config['cohort_sha256'],'full',model=model,revision=revision)
        if result.get('adapter') or result['selection']!='full':
            raise ValueError('Wrong candidate or evaluation selection')
        from experiments.baseline_cache import status_fields
        study.write_json(out/'status'/f'{name}.json',{'state':'complete','finished_at':study.now(),
            'responses':22072,'records_sha256':digest(run/'records.jsonl'),**(status_fields(run) if name=='base' else {})})
    inputs(args.config)
    if subprocess.check_output(gpu_command,text=True).strip()!=identity:
        raise RuntimeError('Physical evaluator GPU changed during comparison')
    study.run_process([sys.executable,study.REPO/'compare.py',out/'runs/base',out/'runs/shorterbetter-final',
        '--out',out/'comparison.json'],out/'logs/compare.log',600)
    comparison=json.loads((out/'comparison.json').read_text())
    if comparison['ungraded_benchmarks'] or any(comparison['full_evaluation'].values()):
        raise ValueError('Official comparator rejected full evaluation')
    study.write_json(out/'completion.json',{'complete':True,'finished_at':study.now(),
        'responses_per_model':22072,'comparison_sha256':digest(out/'comparison.json')})
    study.write_json(out/'phase.json',{'phase':'finished','updated_at':time.time()})


if __name__=='__main__':main()
