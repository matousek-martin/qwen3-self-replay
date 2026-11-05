"""Publish verified completion results as jobs finish; no allocation or messaging."""
import argparse
from collections import Counter
import hashlib
import fcntl
import json
from pathlib import Path
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import cloud

STATE=ROOT/'lab/experiments/completion-20261026'
RECOVERY='completion-evaluations-v2-0735c95b47'
SOURCES=['arc_easy','commonsense_qa','gsm8k','skywork']
REPORT_VERSION=6


def records(path):return [json.loads(x) for x in path.read_text().splitlines()]


def running_detail(study, complete):
    if complete:return 'Dokončeno a archiv ověřen'
    if study=='teacher-pilot':
        base=ROOT/'lab/experiments/teacher-generation/completion-v2-20261026'
        state=json.loads((base/'pipeline.json').read_text()) if (base/'pipeline.json').exists() else {}
        if state.get('phase')=='needs_inspection':return 'Přerušeno; chyba je v pipeline.json'
        if state.get('phase')=='cohort_expansion':
            return 'Doplňování párů z dalších registrovaných Skywork tréninkových otázek (v2.1)'
        if state.get('phase') in ('matched_training_campaign','development_complete'):
            policy=json.loads((STATE/'execution-policy.json').read_text())
            prerequisites=policy.get('prerequisites',{}).get('teacher-completion-v2',[])
            if any(not (STATE/p['name']/'state.json').exists() or
                   json.loads((STATE/p['name']/'state.json').read_text()).get('phase')!='complete'
                   for p in prerequisites):
                return 'Páry připraveny; čeká na dokončení kontroly evaluátoru'
            return 'Páry připraveny; probíhá dokončení tří tréninků a evaluací'
        progress=base/'pair-progress.json'
        if progress.exists():
            counts=json.loads(progress.read_text())['review_passed_per_source']
            return 'Kontrola párů: '+', '.join(f'{label} {counts.get(s,0)}/16' for s,label in zip(SOURCES,['ARC','CQA','GSM','Skywork']))
        return 'Doplňování studentských řešení'
    suite={'replay-selection':'replay-seed43','adaptive-replay':'adaptive-seed29',
           'on-policy-distillation':'opd-seed17'}[study]
    output=cloud.STATE/RECOVERY/'output'/suite
    statuses=[json.loads(p.read_text()) for p in (output/'study/status').glob('*.json')]
    finished=sum(r.get('state')=='complete' for r in statuses)
    expected=4 if study=='on-policy-distillation' else 3
    detail=f'Obnova evaluace: {finished}/{expected} modelů (včetně baseline)'
    if (output/'recovery-result.json').exists():
        recovery_state=STATE/RECOVERY/'state.json'
        verified=recovery_state.exists() and json.loads(recovery_state.read_text()).get('phase')=='complete'
        detail='Obnovená evaluace hotová a archiv ověřen' if verified else 'Obnovená evaluace hotová; čeká na ověření celého archivu'
    if study=='adaptive-replay':
        training=cloud.STATE/'adaptive-completion-v1-seed43-2c09b3417b/output/training'
        arms=[]
        for arm in ['fixed-replay-seed43','adaptive-replay-seed43']:
            path=training/arm/'metrics.jsonl'
            if path.exists():
                lines=path.read_text().splitlines()
                if lines:arms.append(f"{'fixed' if arm.startswith('fixed') else 'adaptive'} {json.loads(lines[-1])['group']}/128")
        detail+='; seed 43: '+(', '.join(arms) if arms else 'čeká na trénink')
    return detail


def paired_summary(pairs,teacher=False,resamples=10000,aggregation='domain_geometric'):
    import numpy as np
    seed_data=[];expected=None;group_keys=None
    def index(rows):
        result={}
        for r in rows:
            key=(r['source'],r['prompt_id'],0) if teacher else (r['dataset'],r['id'],r['sample_index'])
            if key in result or type(r.get('correct')) is not bool or r.get('grading_error'):
                raise ValueError('Duplicate or ungraded paired record')
            result[key]=r
        return result
    for seed,leftpath,rightpath in pairs:
        left,right=index(records(leftpath)),index(records(rightpath))
        if set(left)!=set(right) or (expected is not None and expected!=set(left)):
            raise ValueError('Paired/seed evaluation coverage differs')
        expected=set(left)
        grouped={}
        for key in sorted(left):
            group=key[0] if teacher or not key[0].startswith('bfcl_') else 'bfcl'
            prompt=key[:2];a,b=left[key],right[key]
            token='total_tokens' if teacher else 'output_tokens'
            reasoning='thinking_tokens' if teacher else 'reasoning_tokens'
            grouped.setdefault(group,{}).setdefault(prompt,[]).append([float(a['correct']),float(b['correct']),a[token],b[token],
                a.get(reasoning,0),b.get(reasoning,0),float(a.get('truncated',False)),float(b.get('truncated',False)),
                float(bool(a.get('parse_error'))),float(bool(b.get('parse_error')))])
        if teacher and (len(left)!=800 or set(grouped)!=set(SOURCES)):
            raise ValueError('Incomplete teacher validation')
        if not teacher and (len(left)!=1024 or set(grouped)!={'bfcl','ifeval','mbpp','svamp'}):
            raise ValueError('Incomplete frozen tuning evaluation')
        keys={g:sorted(ps) for g,ps in grouped.items()}
        if group_keys is not None and group_keys!=keys:raise ValueError('Prompt clusters changed')
        group_keys=keys
        seed_data.append({g:np.array([np.mean(ps[p],axis=0) for p in keys[g]]) for g,ps in grouped.items()})
    arrays={g:np.stack([d[g] for d in seed_data]) for g in group_keys}
    def point(means):
        # means: seeds x domains x metrics. Protocols deliberately differ:
        # replay geometrically averages seed/domain ratios, adaptive first
        # averages seed means; teacher uses the ratio of overall means.
        delta=float(np.mean(means[:,:,1]-means[:,:,0]))
        if teacher:
            ratio=float(np.mean(means[:,:,3])/np.mean(means[:,:,2]))
        elif aggregation=='seed_domain_geometric':
            ratio=float(np.exp(np.log(means[:,:,3]/means[:,:,2]).mean()))
        else:
            averaged=means.mean(axis=0)
            ratio=float(np.exp(np.log(averaged[:,3]/averaged[:,2]).mean()))
        return delta,ratio
    all_means=np.stack([a.mean(axis=1) for a in arrays.values()],axis=1)
    point_est=point(all_means)
    labels=['control_accuracy','candidate_accuracy','control_mean_tokens','candidate_mean_tokens',
            'control_mean_thinking_tokens','candidate_mean_thinking_tokens','control_truncation_rate','candidate_truncation_rate',
            'control_parse_failure_rate','candidate_parse_failure_rate']
    per_seed=[]
    for i,(seed,_,_) in enumerate(pairs):
        means=all_means[i:i+1];delta,ratio=point(means)
        per_seed.append({'seed':seed,'accuracy_delta':delta,'token_ratio':ratio,'token_reduction':1-ratio,
                         'domains':{g:dict(zip(labels,a[i].mean(axis=0).tolist())) for g,a in arrays.items()}})
    rng=np.random.default_rng(20261025);draws=[];domain_draws=[]
    for _ in range(resamples):
        seeds=rng.integers(0,len(pairs),size=len(pairs));means=[]
        for a in arrays.values():
            indices=rng.integers(0,a.shape[1],size=a.shape[1])
            means.append(a[seeds[:,None],indices[None,:]].mean(axis=1))
        means=np.stack(means,axis=1)
        draws.append(point(means))
        domain_draws.append((means[:,:,1]-means[:,:,0]).mean(axis=0))
    intervals=np.quantile(np.array(draws),[.025,.975],axis=0)
    domain_intervals=np.quantile(np.array(domain_draws),[.025,.975],axis=0)
    multiplier=1 if teacher else 2
    totals=np.sum(np.concatenate(list(arrays.values()),axis=1),axis=(0,1))*multiplier
    return {'seed_count':len(pairs),'per_seed':per_seed,'accuracy_delta':point_est[0],
            'token_ratio':point_est[1],'token_reduction':1-point_est[1],
            'ci95':{'accuracy_delta':intervals[:,0].tolist(),'token_ratio':intervals[:,1].tolist()},
            'domains':{g:{**dict(zip(labels,all_means[:,i].mean(axis=0).tolist())),
                          'accuracy_delta_ci95':domain_intervals[:,i].tolist()} for i,g in enumerate(arrays)},
            'pooled_totals':{'responses':len(expected)*len(pairs),
                'control_generated_tokens':float(totals[2]),'candidate_generated_tokens':float(totals[3]),
                'control_thinking_tokens':float(totals[4]),'candidate_thinking_tokens':float(totals[5]),
                'token_ratio':float(totals[3]/totals[2])},
            'bootstrap':{'replicates':resamples,'seed':20261025,'method':'paired seed resampling and stratified prompt clusters; samples averaged inside prompt',
                         'limitation':'Only three training seeds at full completion; partial summaries are exploratory.'},
            'token_aggregation':'ratio of means over four equal-sized sources' if teacher else aggregation}


def verified_job(name,ledger):
    j=ledger['jobs'].get(name,{})
    if not (j.get('released_at') and j.get('backup_verified') and j.get('state')=='complete'):return None
    marker=cloud.STATE/name/'output/role-result.json'
    if not marker.exists():return None
    fingerprint=str(j['released_at'])
    cache=STATE/'verified-jobs.json';known=json.loads(cache.read_text()) if cache.exists() else {}
    if known.get(name)!=fingerprint:
        if cloud.verify_backup(name)['exit_code']!=0:raise ValueError('Failed worker cannot count as complete')
        known[name]=fingerprint;cloud.write(cache,known)
    return marker.parent


def training_evidence(study,seeds):
    ledger=json.loads((cloud.STATE/'ledger.json').read_text())
    names=[]
    for seed in seeds:
        if study=='replay-selection':names.append(f'replay-selection-seed{seed}-20261025T160813Z')
        elif study=='adaptive-replay':names.append(f'adaptive-isolated-v2-seed{seed}-079b11b144' if seed!=43 else 'adaptive-completion-v1-seed43-2c09b3417b')
        elif study=='on-policy-distillation':names.append(f'opd-isolated-v2-seed{seed}-12b6f7061c')
        else:
            campaign=json.loads((ROOT/'lab/experiments/teacher-completion-v2/campaign.json').read_text())
            names.append(campaign['trials'][str(seed)]['job'])
    result={}
    for name in names:
        job=ledger['jobs'][name];output=cloud.STATE/name/'output';arms={}
        for path in (output/'training').glob('*/metrics.jsonl'):
            rows=records(path)
            if not rows:continue
            last=rows[-1];admissions=Counter();by_source={}
            for row in rows:
                # Static teacher/student replay has no online admission event.
                admission=row.get('replay_admission') or {}
                action=admission.get('action','not_recorded')
                admissions[action]+=1
                d=by_source.setdefault(row['source'],{'groups':0,'eligible':0,'admission_events':Counter()})
                d['groups']+=1;d['eligible']+=bool(admission.get('eligible'))
                d['admission_events'][action]+=1
            arms[path.parent.name]={'metrics':str(path),'completed_groups':last['group'],
                'optimizer_updates':last.get('optimizer_updates'),'generated_tokens':last.get('generated_tokens_total'),
                'rl_and_replay_training_tokens':last.get('training_tokens_total'),
                'main_training_tokens':sum(r.get('trained_tokens',0) for r in rows),
                'replay_tokens':sum(r.get('replay_tokens',0) for r in rows),
                'elapsed_seconds':last.get('elapsed_seconds'),'peak_gpu_bytes':max(r.get('peak_gpu_bytes',0) for r in rows),
                'admission_events':dict(admissions),'by_source':by_source,
                'group_success_count_histogram':dict(Counter(str(r.get('group_success_count')) for r in rows)),
                'final_pool_size':last.get('replay_pool_size')}
            distillation=path.parent/'distillation.jsonl'
            if distillation.exists():
                scored=records(distillation)
                arms[path.parent.name]['distillation']={'records':str(distillation),'groups':len(scored),
                    'scored_tokens':sum(r['scored_token_count'] for r in scored),
                    'sources':dict(Counter(r['source'] for r in scored))}
        role=output/'role-result.json';provenance=output/'isolation-provenance.json'
        meta=json.loads(role.read_text()) if role.exists() else json.loads(provenance.read_text()) if provenance.exists() else {}
        result[name]={'historical_worker_state':job['state'],'actual_gpu':meta.get('gpu_identity',meta.get('gpu')),
                      'released':bool(job.get('released_at')),'ledger_estimated_usd':job.get('charged_estimate_usd'),
                      'provenance':str(role if role.exists() else provenance),'arms':arms}
        cache=output/'preparation/teacher-cache.json'
        if cache.exists():
            info=json.loads(cache.read_text());entries=info.pop('entries')
            result[name]['teacher_cache']={**info,'prompts':len(entries),
                'verified_prompts':sum(bool(r['accepted']) for r in entries.values()),'source':str(cache)}
    attempts={n:{'state':j['state'],'ledger_estimated_usd':j.get('charged_estimate_usd'),
                 'released':bool(j.get('released_at'))} for n,j in ledger['jobs'].items()
              if n.startswith('completion-evaluations-')}
    return {'training_jobs':result,'shared_recovery_attempts':None if study=='teacher-pilot' else attempts,
            'shared_recovery_job':None if study=='teacher-pilot' else RECOVERY,
            'shared_recovery_ledger_estimated_usd':None if study=='teacher-pilot' else ledger['jobs'].get(RECOVERY,{}).get('charged_estimate_usd'),
            'cost_note':'Ledger estimates, not a provider invoice. Recovery evaluation job is shared across three studies; do not count its cost three times. Teacher generation/review costs remain in the shared ledger.'}


def publish(study,pairs,complete,extra):
    destination=STATE/(study+'-results.json')
    fingerprints=[(seed,str(a),hashlib.sha256(a.read_bytes()).hexdigest(),str(b),hashlib.sha256(b.read_bytes()).hexdigest()) for seed,a,b in pairs]
    if destination.exists():
        existing=json.loads(destination.read_text())
        if (existing.get('report_version')==REPORT_VERSION and existing.get('input_fingerprints')==json.loads(json.dumps(fingerprints))
            and existing['complete']==complete and all(existing.get(k)==v for k,v in extra.items())):return existing
    summary=paired_summary(pairs,teacher=study=='teacher-pilot',
        aggregation='seed_domain_geometric' if study=='replay-selection' else 'domain_geometric') if pairs else None
    baseline={}
    if pairs:
        for label,index in [('control',1),('candidate',2)]:
            baseline_pairs=[(p[0],p[1].with_name('base.jsonl') if study=='teacher-pilot' else
                             p[1].parent.parent/'base/records.jsonl',p[index]) for p in pairs]
            baseline[label]=paired_summary(baseline_pairs,teacher=study=='teacher-pilot',
                aggregation='seed_domain_geometric' if study=='replay-selection' else 'domain_geometric')
    result={'study':study,'complete':complete,'updated_at':time.time(),'input_fingerprints':fingerprints,
            'report_version':REPORT_VERSION,'summary':summary,'fresh_baseline_comparisons':baseline,
            'training_evidence':training_evidence(study,[p[0] for p in pairs]),**extra}
    if study in ['replay-selection','adaptive-replay'] and summary:
        def accuracy_passed(r):
            return r['ci95']['accuracy_delta'][0]>=-.01 and all(d['accuracy_delta_ci95'][0]>=-.01 for d in r['domains'].values())
        result['criteria']={'accuracy_noninferiority_macro_and_domains':accuracy_passed(summary),
            'candidate_baseline_accuracy_noninferiority_macro_and_domains':accuracy_passed(baseline['candidate']),
            'token_saving_ci':summary['ci95']['token_ratio'][1]<1,
            'every_seed_token_saving':all(s['token_ratio']<1 for s in summary['per_seed'])}
        result['benefit_established']=complete and all(result['criteria'].values())
    if study=='teacher-pilot' and summary:
        domains={g:sum(seed['domains'][g]['candidate_accuracy']-seed['domains'][g]['control_accuracy']
                       for seed in summary['per_seed'])/len(pairs) for g in SOURCES}
        result['criteria']={'point_token_ratio_at_most_095':summary['token_ratio']<=.95,
                            'point_accuracy_delta_at_least_minus001':summary['accuracy_delta']>=-.01,
                            'accuracy_noninferiority_ci':summary['ci95']['accuracy_delta'][0]>=-.01,
                            'token_saving_ci':summary['ci95']['token_ratio'][1]<1,
                            'domain_guardrail':all(delta>=-.03 for delta in domains.values()),
                            'domain_accuracy_deltas':domains}
        result['benefit_established']=complete and all(result['criteria'][k] for k in result['criteria'] if k!='domain_accuracy_deltas')
    if study=='on-policy-distillation' and summary and extra.get('against_offline_hard'):
        comparisons=[summary,extra['against_offline_hard']]
        result['screening_passed']=all(r['token_reduction']>=.05 and r['accuracy_delta']>=-.01 for r in comparisons)
        result['accuracy_noninferiority_established']=all(r['ci95']['accuracy_delta'][0]>=-.01 for r in comparisons)
    cloud.write(destination,result)
    lines=[f'# {study}: výsledky dokončení', '',
           ('Kompletní registrovaný rozsah dokončen.' if complete else 'Průběžné výsledky; experiment ještě není dokončen.'),'',
           f'Autoritativní data: [{destination.name}]({destination})','']
    if summary:
        lines+=['| Seed | Rozdíl přesnosti (p. b.) | Úspora celkových tokenů |','|---|---:|---:|']
        for r in summary['per_seed']:lines.append(f"| {r['seed']} | {100*r['accuracy_delta']:+.2f} | {100*r['token_reduction']:+.2f}% |")
        lines+=['',f"Souhrn: přesnost {100*summary['accuracy_delta']:+.2f} p. b.; úspora tokenů {100*summary['token_reduction']:+.2f}%.",
                f"95% interval rozdílu přesnosti: {summary['ci95']['accuracy_delta']}; interval poměru tokenů: {summary['ci95']['token_ratio']}.",'',
                'Vyšší úspora znamená kratší odpovědi. Intervaly používají párové vzorkování seedů a otázek; malý počet seedů omezuje závěr. Nejde o nový nepoužitý holdout.','']
        lines+=['| Doména | Přesnost kontrola → kandidát | Celkové tokeny kontrola → kandidát | Thinking tokeny kontrola → kandidát |',
                '|---|---:|---:|---:|']
        for domain,d in summary['domains'].items():
            lines.append(f"| {domain} | {100*d['control_accuracy']:.2f}% → {100*d['candidate_accuracy']:.2f}% | {d['control_mean_tokens']:.1f} → {d['candidate_mean_tokens']:.1f} | {d['control_mean_thinking_tokens']:.1f} → {d['candidate_mean_thinking_tokens']:.1f} |")
        lines+=['',f"Kandidát proti čerstvé baseline: přesnost {100*baseline['candidate']['accuracy_delta']:+.2f} p. b.; úspora tokenů {100*baseline['candidate']['token_reduction']:+.2f}%.",
                'JSON obsahuje také baseline porovnání kontrolní větve, intervaly jednotlivých domén a četnost zkrácení či chyb parsování.', '']
    if 'benefit_established' in result:
        lines += ['Předem stanovená kritéria přínosu splněna: '+('ano' if result['benefit_established'] else 'ne / zatím neprůkazné'), '']
    if 'screening_passed' in result:
        lines += ['Screening pro další OPD replikace: '+('splněn' if result['screening_passed'] else 'nesplněn'), '']
    lines += [extra.get('interpretation',''),'', 'Původní neúspěšné běhy a jejich zálohy zůstávají zachované. Evaluace obnovených modelů používají čerstvou baseline na témže evaluačním workeru.','']
    (ROOT/'studies'/study/'REPORT-COMPLETION-20261026.md').write_text('\n'.join(lines))
    return result


def tick():
    STATE.mkdir(parents=True,exist_ok=True)
    ledger=json.loads((cloud.STATE/'ledger.json').read_text())
    recovery=verified_job(RECOVERY,ledger)
    replay=[]
    for seed in [17,29]:
        p=verified_job(f'replay-selection-seed{seed}-20261025T160813Z',ledger)
        if p:replay.append((seed,p/f'study/runs/control-seed{seed}/records.jsonl',p/f'study/runs/mixed-seed{seed}/records.jsonl'))
    if recovery:
        p=recovery/'replay-seed43/study/runs';replay.append((43,p/'control-seed43/records.jsonl',p/'mixed-seed43/records.jsonl'))
    result={'replay-selection':publish('replay-selection',replay,len(replay)==3,{'interpretation':'Verified-correct admission proti all-correct; tři původní training seedy, tuning benchmark.'})}
    adaptive=[]
    p=verified_job('adaptive-isolated-v2-seed17-079b11b144',ledger)
    if p:adaptive.append((17,p/'study/runs/fixed-replay-seed17/records.jsonl',p/'study/runs/adaptive-replay-seed17/records.jsonl'))
    if recovery:
        p=recovery/'adaptive-seed29/study/runs';adaptive.append((29,p/'fixed-replay-seed29/records.jsonl',p/'adaptive-replay-seed29/records.jsonl'))
    p=verified_job('adaptive-completion-v1-seed43-2c09b3417b',ledger)
    if p:adaptive.append((43,p/'study/runs/fixed-replay-seed43/records.jsonl',p/'study/runs/adaptive-replay-seed43/records.jsonl'))
    result['adaptive-replay']=publish('adaptive-replay',adaptive,len(adaptive)==3,{'interpretation':'Adaptive versus fixed replay při nezměněném all-correct admission.'})
    opd=[];soft=None
    if recovery:
        p=recovery/'opd-seed17/study/runs';opd=[(17,p/'offline-soft-seed17/records.jsonl',p/'on-policy-soft-seed17/records.jsonl')]
        soft=paired_summary([(17,p/'offline-hard-seed17/records.jsonl',p/'on-policy-soft-seed17/records.jsonl')])
    result['on-policy-distillation']=publish('on-policy-distillation',opd,bool(opd),{'against_offline_hard':soft,
        'interpretation':'Primární on-policy-soft versus offline-soft; doplňkově offline-hard. Dokončení znamená počáteční seed17 pilot, nikoli tři replikace. Další seedy jsou podmíněny původním screeningem.'})
    teacher=[];campaign=ROOT/'lab/experiments/teacher-completion-v2/campaign.json'
    if campaign.exists():
        for seed,trial in json.loads(campaign.read_text())['trials'].items():
            p=verified_job(trial['job'],ledger)
            if p:teacher.append((int(seed),p/'validation/student.jsonl',p/'validation/teacher.jsonl'))
    extended=(ROOT/'lab/experiments/teacher-generation/completion-v2-20261026/cohort-extension-used.json').exists()
    result['teacher-pilot']=publish('teacher-pilot',teacher,len(teacher)==3,{'protocol_version':'2.1' if extended else '2',
        'interpretation':('V2.1 přidává registrované Skywork tréninkové otázky; ' if extended else 'V2: ')+
        'učitelský replay proti ověřenému studentovu replay po více pokusech. Původní v1 nezískal dost párů. 800 training-source validation otázek, greedy vLLM; stejné podmínky pro všechny větve.'})
    summary={'updated_at':time.time(),'complete':all(r['complete'] for r in result.values()),
             'experiments':{k:{'complete':r['complete'],'seed_count':r['summary']['seed_count'] if r['summary'] else 0} for k,r in result.items()},
             'ledger_liability_usd':cloud.liability(ledger['jobs']),'ceiling_usd':cloud.USABLE}
    cloud.write(STATE/'summary.json',summary)
    lines=['# Dokončení čtyř experimentů', '',
           'Aktualizováno automaticky: '+datetime.now(ZoneInfo('Europe/Prague')).strftime('%Y-%m-%d %H:%M %Z')+'. Výsledky jsou z ověřených archivů, průběh z pracovních stavů.', '',
           '| Experiment | Vyhodnocené seedy | Stav | Výsledky |',
           '|---|---:|---|---|']
    for study,row in summary['experiments'].items():
        expected=1 if study=='on-policy-distillation' else 3
        status=running_detail(study,row['complete'])
        lines.append(f"| {study} | {row['seed_count']}/{expected} | {status} | [Report](../{study}/REPORT-COMPLETION-20261026.md) |")
    lines+=['','Dokončení znamená provedené porovnání, nikoli automaticky prokázaný přínos metody. OPD je počáteční seed17 pilot; teacher používá protokol '+str(result['teacher-pilot']['protocol_version'])+'.','',
            '[Strojový stav](../../lab/experiments/completion-20261026/summary.json) · [Provozní předání](HANDOFF.md)','']
    (ROOT/'studies/parallel-launch/RESULTS-20261026.md').write_text('\n'.join(lines))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
    while True:
        try:
            STATE.mkdir(parents=True,exist_ok=True)
            with (STATE/'report.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                summary=tick()
            print(json.dumps(summary),flush=True)
        except Exception as exc:
            if not a.watch:raise
            print(json.dumps({'report_update_failed':str(exc),'previous_reports_preserved':True}),flush=True)
            time.sleep(120)
            continue
        if not a.watch or summary['complete']:break
        time.sleep(120)
