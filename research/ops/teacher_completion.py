"""Authorized v2 continuation: verified pairs, three paired seeds, full validation."""
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from isolated_cloud import ROOT,verify,digest
from completion_jobs import run_job
import teacher_pipeline as previous
from teacher_student_samples import normalize
from teacher_review_completion import review as completion_review

OLD=ROOT/'lab/isolated/launch-20261025'
BASE=ROOT/'lab/isolated/completion-20261026'
STATE=ROOT/'lab/experiments/teacher-generation/completion-v2-20261026'
SOURCES=previous.SOURCES


def write(path,value):previous.write(path,value)


def progress(phase,**extra):
    value={'phase':phase,'updated_at':time.time(),**extra};write(STATE/'pipeline.json',value)
    print(json.dumps(value),flush=True)


def clone(name):
    destination=BASE/name
    if destination.exists():
        verify(destination);return destination
    verify(BASE/'teacher-template-v3')
    shutil.copytree(BASE/'teacher-template-v3',destination,
                    ignore=shutil.ignore_patterns('payload-manifest.json','__pycache__'))
    return destination


def generation(round_index):
    release=clone(f'teacher-student-round{round_index}')
    if not (release/'payload-manifest.json').exists():
        plan={'attempt_start':1+8*round_index,'attempt_end':9+8*round_index,'all_original_prompts':128,
               'selection':'first answer/length/review-passing sample by attempt order; no test outcomes',
               'protocol_sha256':digest(BASE/'teacher-v2-protocol.json')}
        if round_index>=4:
            plan['sampling_extension_sha256']=digest(BASE/'teacher-v2-sampling-extension.json')
        write(release/'lab/experiments/data/student-sampling-plan.json',plan)
        (release/'experiments/run-teacher-additional.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\nsource /workspace/project/experiments/setup_gpu.sh\nexport OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1\npython -m experiments.teacher_student_samples\n')
        previous.lock_inputs_and_sources(release)
    name=f'teacher-completion-student-r{round_index}-'+verify(release)[:10]
    policy_path=ROOT/'lab/experiments/completion-20261026/execution-policy.json'
    policy=json.loads(policy_path.read_text()) if policy_path.exists() else {}
    name=policy.get('sampling_job_overrides',{}).get(name,name)
    progress('student_generation',round=round_index,job=name)
    output=run_job(release,name,'experiments/run-teacher-additional.sh',2)
    rows=[json.loads(x) for x in (output/'student.jsonl').read_text().splitlines()]
    if len(rows)!=1024 or len({(r['prompt_id'],r['attempt']) for r in rows})!=1024:
        raise ValueError('Student generation coverage incomplete')
    return rows


def main():
    STATE.mkdir(parents=True,exist_ok=True)
    with (STATE/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runner,common=previous.setup_api()
        previous.STATE=STATE;previous.BASE=BASE
        from transformers import AutoTokenizer
        from experiments.rl_training.train import verifier,reasoning_is_complete
        tokenizer=AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B',revision=common.REVISION,local_files_only=True)
        grader=verifier();cohort={r['id']:r for r in common.prompts()}
        oldstate=ROOT/'lab/experiments/teacher-generation/isolated-pipeline-20261025'
        allteachers=json.loads((oldstate/'reviewed-teachers/verified.json').read_text())
        teacher=[r for r in allteachers if r['teacher_model']=='gpt-6.1-sol' and r['variant']=='explicit' and r['replay_eligible']]
        teacherids={r['prompt_id'] for r in teacher}
        original=ROOT/'lab/experiments/cloud/teacher-student-inference-07bcc40720/output/student.jsonl'
        samples=[]
        for r in map(json.loads,original.read_text().splitlines()):
            r=dict(r,attempt=0,completion=normalize(r['completion']))
            r['request_key']='student-v2-original-'+r['response_sha256'];samples.append(r)
        selected={};reviews=[];reviewed=set()
        # Replay previously persisted decisions on restart without paid repeats.
        extension=BASE/'teacher-v2-sampling-extension.json'
        rounds=json.loads(extension.read_text())['max_additional_attempts_per_prompt']//8 if extension.exists() else 4
        cohort_extension_used=False
        for round_index in range(rounds):
            if round_index==4 and (BASE/'teacher-cohort-extension-v21.json').exists():
                counts=Counter(r['source'] for r in selected.values())
                if any(counts[s]<16 for s in SOURCES if s!='skywork'):
                    raise RuntimeError('Registered extra cohort only addresses Skywork; another source is short')
                needed=max(0,16-counts['skywork'])
                if needed:
                    progress('cohort_expansion',required_extra_skywork_pairs=needed)
                    subprocess.run([sys.executable,ROOT/'ops/teacher_extra_pairs.py',str(needed)],cwd=ROOT,check=True)
                    extra=json.loads((ROOT/'lab/experiments/teacher-generation/completion-extra-v21/result.json').read_text())
                    cohort.update({r['id']:r for r in extra['cohort']});teacher+=extra['teacher']
                    selected.update({r['prompt_id']:r for r in extra['student']});reviews+=extra['reviews']
                    samples+=extra['all_student_samples'];cohort_extension_used=True
                    write(STATE/'cohort-extension-used.json',{'protocol_version':'2.1','protocol_sha256':extra['protocol_sha256'],
                        'extra_pairs':len(extra['student']),'result':'lab/experiments/teacher-generation/completion-extra-v21/result.json'})
                    break
            samples+=generation(round_index)
            for attempt in range(0,9+8*round_index):
                items=[]
                for r in sorted(samples,key=lambda x:(x['attempt'],x['prompt_id'])):
                    pid=r['prompt_id'];row=cohort[pid]
                    if r['attempt']!=attempt or pid in selected or pid not in teacherids or r['request_key'] in reviewed:continue
                    valid=(r['accepted_answer'] and r['finish_reason']=='stop' and reasoning_is_complete(r['completion'])
                           and grader(r['completion'],row['answer'],row['task'])
                           and len(tokenizer.encode(r['completion'],add_special_tokens=False))+1<=2048)
                    if valid:items.append(dict(r,prompt=row['prompt'],answer=row['answer']))
                if not items:continue
                label=f'student-review-attempt-{attempt}'
                progress('reviewing_student_pairs',round=round_index,attempt=attempt,candidates=len(items))
                verdicts=completion_review(items,label,runner,common)
                reviews+=verdicts;verdicts={r['request_key']:r for r in verdicts}
                for r in items:
                    reviewed.add(r['request_key'])
                    if verdicts[r['request_key']]['verdict']=='pass':selected[r['prompt_id']]=r
                counts=Counter(r['source'] for r in selected.values())
                write(STATE/'pair-progress.json',{'review_passed_per_source':dict(counts),'required':16,
                     'sampled':len(samples),'reviewed_candidates':len(reviewed),'attempt':attempt})
                if min(counts[s] for s in SOURCES)>=16:break
            if min(Counter(r['source'] for r in selected.values())[s] for s in SOURCES)>=16:break
        pools=previous.build_pools(teacher,list(selected.values()),cohort,tokenizer,reviews,common)
        if pools is None:raise RuntimeError(f'{8*rounds} additional attempts exhausted without enough independently verified pairs; do not claim completion or relax grading')
        ids,arms=pools
        write(STATE/'selected-students.json',list(selected.values()))
        write(STATE/'student-reviews.json',reviews)
        write(STATE/'all-student-samples.json',samples)
        previous.backup_api(STATE/'student-review-attempt-0','student-review-attempt-0') if (STATE/'student-review-attempt-0').exists() else None
        release=clone('teacher-matched-v2')
        if not (release/'payload-manifest.json').exists():
            static=release/'lab/experiments/data/static';static.mkdir()
            manifest={'schema_version':1,'prompt_ids':ids,'pools':{},'schedules':{},
                      'selection_sha256':digest(STATE/'selected-students.json'),'protocol_sha256':digest(BASE/'teacher-v2-protocol.json')}
            if extension.exists():manifest['sampling_extension_sha256']=digest(extension)
            if cohort_extension_used:manifest['cohort_extension_sha256']=digest(BASE/'teacher-cohort-extension-v21.json')
            for arm,records in arms.items():
                path=static/(arm+'-pool.jsonl');path.write_text(''.join(json.dumps(r)+'\n' for r in records))
                manifest['pools'][arm]={'file':path.name,'sha256':digest(path)}
            for seed in [17,29,43]:
                rng=random.Random(seed+104729);path=static/f'schedule-{seed}.json';write(path,[rng.choice(ids) for _ in range(128)])
                manifest['schedules'][str(seed)]={'file':path.name,'sha256':digest(path)}
                (release/f'experiments/run-isolated-seed{seed}.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\nsource /workspace/project/experiments/setup_gpu.sh\nexport OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1\npython -m experiments.teacher_generation.paired_worker --seed '+str(seed)+'\n')
            write(static/'manifest.json',manifest)
            training=json.loads((OLD/'replay-selection/lab/experiments/data/replay-selection-v1/protocol.json').read_text())['training']
            training['replay_admission']='all_correct'
            config={'id':'teacher-completion-v2','seeds':[17,29,43],'hours':6,'training':training,
                'protocol_sha256':digest(BASE/'teacher-v2-protocol.json'),
                'authorization':json.loads((BASE/'authorization.json').read_text()),'pool_manifest_sha256':digest(static/'manifest.json'),
                'primary_evaluation':{'questions':800,'data':'val.jsonl','greedy':True,'backend':'vllm',
                                     'max_new_tokens':4096,'final_answer_cap':1024,'seed':17}}
            if extension.exists():config['sampling_extension_sha256']=digest(extension)
            config['protocol_version']='2.1' if cohort_extension_used else '2'
            if cohort_extension_used:config['cohort_extension_sha256']=digest(BASE/'teacher-cohort-extension-v21.json')
            write(release/'lab/experiments/data/isolated/config.json',config)
            previous.lock_inputs_and_sources(release)
        progress('matched_training_campaign',release=str(release),pairs=len(ids))
        subprocess.run([sys.executable,ROOT/'ops/isolated_campaign.py','--release',release,
                        '--state',ROOT/'lab/experiments/teacher-completion-v2'],cwd=ROOT,check=True)
        progress('development_complete',training_launched=True,pairs=len(ids))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        progress('needs_inspection',error=str(exc));raise
