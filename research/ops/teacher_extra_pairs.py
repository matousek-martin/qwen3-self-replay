"""Conditional v2.1 expansion within the already frozen pilot512 training set."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

from isolated_cloud import ROOT, verify, digest
from completion_jobs import run_job
import teacher_pipeline as previous
from teacher_review_completion import review

BASE=ROOT/'lab/isolated/completion-20261026'
STATE=ROOT/'lab/experiments/teacher-generation/completion-extra-v21'


def generate_students(round_index):
    release=BASE/f'teacher-extra-student-r{round_index}'
    if not release.exists():
        verify(BASE/'teacher-template-v3')
        shutil.copytree(BASE/'teacher-template-v3',release,
                        ignore=shutil.ignore_patterns('payload-manifest.json','__pycache__'))
        sampler=release/'experiments/teacher_student_samples.py'
        original=sampler.read_text()
        old="ROOT/'data/teacher-pilot-20261024/teacher-comparison-128.jsonl'"
        if original.count(old)!=1:raise ValueError('Sampler source changed')
        sampler.chmod(0o644)
        sampler.write_text(original.replace(old,"ROOT/'data'/plan['cohort_file']"))
        target=release/'lab/experiments/data/teacher-extra-skywork-64.jsonl'
        shutil.copy2(BASE/'teacher-extra-skywork-64.jsonl',target)
        previous.write(release/'lab/experiments/data/student-sampling-plan.json',{
            'attempt_start':1+8*round_index,'attempt_end':9+8*round_index,
            'cohort_file':target.name,'cohort_sha256':digest(target),'questions':64,
            'protocol_sha256':digest(BASE/'teacher-cohort-extension-v21.json')})
        (release/'experiments/run-teacher-extra.sh').write_text(
            '#!/usr/bin/env bash\nset -euo pipefail\nsource /workspace/project/experiments/setup_gpu.sh\n'
            'export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1\npython -m experiments.teacher_student_samples\n')
        previous.lock_inputs_and_sources(release)
    name=f'teacher-completion-extra-r{round_index}-'+verify(release)[:10]
    out=run_job(release,name,'experiments/run-teacher-extra.sh',2)
    rows=[json.loads(x) for x in (out/'student.jsonl').read_text().splitlines()]
    if len(rows)!=512 or len({(r['prompt_id'],r['attempt']) for r in rows})!=512:
        raise ValueError('Extra student sampling coverage incomplete')
    return rows


def complete(required):
    if not 1<=required<=16:raise ValueError('Invalid extra pair requirement')
    STATE.mkdir(parents=True,exist_ok=True)
    marker=STATE/'result.json'
    if marker.exists():
        result=json.loads(marker.read_text())
        if len(result['student'])<required:raise ValueError('Cached extra cohort insufficient')
        return result
    protocol=json.loads((BASE/'teacher-cohort-extension-v21.json').read_text())
    path=BASE/'teacher-extra-skywork-64.jsonl'
    if digest(path)!=protocol['extra_cohort_sha256']:raise ValueError('Extra cohort changed')
    cohort={r['id']:r for r in map(json.loads,path.read_text().splitlines())}
    if len(cohort)!=64 or any(r['source']!='skywork' or r['split']!='train' for r in cohort.values()):
        raise ValueError('Extra cohort must be the registered Skywork training rows')
    # setup_api must read the original sealed API implementation first.
    runner,common=previous.setup_api()
    previous.STATE=STATE;previous.BASE=BASE
    from experiments.teacher_generation.verify import verify_response
    from transformers import AutoTokenizer
    from experiments.rl_training.train import verifier,reasoning_is_complete
    tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B',revision=common.REVISION,local_files_only=True)
    grade=verifier()
    pricing={'id':'gpt-6.1-sol','input_per_million':2.,'output_per_million':10.,
             'cached_input_per_million':.1,'input_ceiling_per_million':5.}

    def teacher(row):
        payload={'model':pricing['id'],'input':row['prompt'],
                 'instructions':runner.VARIANTS['explicit']+' '+runner.FORMAT,
                 'reasoning':{'effort':'low'},'max_output_tokens':2048,'service_tier':'default','store':False}
        bound=len(common.canonical(payload).encode())+2048
        identity={'prompt_id':row['id'],'prompt_sha256':common.digest(row),'variant':'explicit','payload':payload}
        request={**identity,'request_key':common.digest(identity),'source':row['source'],
                 'input_token_bound':bound,'reserved_usd':(bound*5+2048*10)/1e6,'pricing':pricing}
        plan={'schema_version':1,'study':'teacher-pilot-extra-v21','provider':'openai',
              'requests':[request],'reservation_usd':math.ceil(request['reserved_usd']*1e6)/1e6,
              'code_sha256':runner.source_hashes(),'controller_sha256':digest(Path(__file__)),
              'protocol_sha256':digest(BASE/'teacher-cohort-extension-v21.json')}
        directory=STATE/'teacher-requests'/request['request_key'];directory.mkdir(parents=True,exist_ok=True)
        plan_path=directory/'registered-plan.json'
        if plan_path.exists() and json.loads(plan_path.read_text())!=plan:raise ValueError('Extra teacher plan changed')
        previous.write(plan_path,plan)
        approval_path=directory/'approval.json'
        if approval_path.exists():approval=json.loads(approval_path.read_text())
        else:
            approval={'approved':True,'scope':'teacher-pilot-openai','plan_sha256':common.digest(plan),
                'max_usd':plan['reservation_usd'],'expires_at':time.time()+7*86400,
                'evidence_path':str(BASE/'authorization.json'),'evidence_sha256':digest(BASE/'authorization.json')}
            previous.write(approval_path,approval)
        receipt=directory/'run/requests'/(request['request_key']+'.json')
        if not receipt.exists():
            try:runner.execute(plan,directory/'run',approval,key_loader=lambda:common.read_key(ROOT/'.env'))
            except RuntimeError as exc:
                if str(exc)!='Request stopped; evidence and full campaign reservation retained':raise
        record=json.loads(receipt.read_text())
        if record.get('request')!=request:raise ValueError('Extra teacher receipt identity changed')
        if record.get('state')!='completed':
            if record.get('wire',{}).get('http_status') not in (500,502,503,504):
                raise RuntimeError('Unclassified extra teacher API failure; no resend')
            return None  # Failed generation retained, rejected, never resent.
        if common.digest(record['response'])!=record['response_sha256'] or record['response']!=json.loads(record['wire']['body']):
            raise ValueError('Extra teacher response evidence changed')
        return row,record

    # Generation has no dependency on teacher results; overlap independent work.
    with ThreadPoolExecutor(max_workers=1) as sampling:
        first=sampling.submit(generate_students,0)
        generated=[];ordered=list(cohort.values())
        with ThreadPoolExecutor(max_workers=4) as api:
            for offset in range(0,len(ordered),4):
                generated.extend(r for r in api.map(teacher,ordered[offset:offset+4]) if r is not None)
        teachers=[]
        for row,record in generated:
            body=record['response'];verified=verify_response(body,row,tok)
            if not verified['accepted_answer'] or len(tok.encode(verified['completion'],add_special_tokens=False))+1>2048:continue
            teachers.append({**verified,'prompt_id':row['id'],'source':row['source'],'request_key':record['request']['request_key'],
                'teacher_model':body['model'],'variant':'explicit','response_sha256':common.digest(body),
                'cost':record['cost'],'prompt':row['prompt'],'answer':row['answer']})
        verdicts={r['request_key']:r for r in review(teachers,'teacher-explanation-review',runner,common)}
        teachers=[dict(r,replay_eligible=True,explanation_review=verdicts[r['request_key']])
                  for r in teachers if verdicts[r['request_key']]['verdict']=='pass']
        previous.write(STATE/'teacher-verified.json',teachers)
        teacherids={r['prompt_id'] for r in teachers}
        if len(teacherids)<required:raise RuntimeError('Extra teacher yield below requirement; quality gate unchanged')
        selected={};reviews=[];all_samples=[]
        for round_index in range(4):
            samples=first.result() if round_index==0 else generate_students(round_index)
            all_samples+=samples
            for attempt in range(1+8*round_index,9+8*round_index):
                items=[]
                for r in sorted(samples,key=lambda x:x['prompt_id']):
                    pid=r['prompt_id'];row=cohort[pid]
                    if r['attempt']!=attempt or pid in selected or pid not in teacherids:continue
                    if (r['accepted_answer'] and r['finish_reason']=='stop' and reasoning_is_complete(r['completion'])
                        and grade(r['completion'],row['answer'],row['task'])
                        and len(tok.encode(r['completion'],add_special_tokens=False))+1<=2048):
                        items.append(dict(r,prompt=row['prompt'],answer=row['answer']))
                verdicts=review(items,f'student-extra-attempt-{attempt}',runner,common) if items else []
                reviews+=verdicts;passed={r['request_key'] for r in verdicts if r['verdict']=='pass'}
                for r in items:
                    if r['request_key'] in passed:selected[r['prompt_id']]=r
                previous.write(STATE/'pair-progress.json',{'passed_extra_pairs':len(selected),'required':required,
                    'attempt':attempt,'sampled':len(all_samples)})
                if len(selected)>=required:break
            if len(selected)>=required:break
        if len(selected)<required:raise RuntimeError('Expanded cohort still insufficient; never relax review')
    result={'protocol_version':'2.1','protocol_sha256':digest(BASE/'teacher-cohort-extension-v21.json'),
            'cohort':list(cohort.values()),'teacher':teachers,'student':list(selected.values()),
            'reviews':reviews,'all_student_samples':all_samples}
    previous.write(marker,result)
    previous.backup_api(STATE/'teacher-requests','extra-teacher-generation')
    previous.progress('development_complete',matched_extra_pairs=len(selected),sampled=len(all_samples),
                      reviewed_teacher_answers=len(teachers))
    return result


if __name__=='__main__':
    complete(int(sys.argv[1]))
