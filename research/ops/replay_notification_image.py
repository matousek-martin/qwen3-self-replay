"""Local, read-only Self-replay report renderer for codex-telegram.

Invoked at send time so already-running controllers adopt the new format without
restart. Never sends messages or touches model/worker/campaign state.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path
from zoneinfo import ZoneInfo
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
CLOUD = ROOT / 'lab/experiments/cloud'
EVALS = ROOT / 'lab/experiments/replay-scaled-evaluation'
TOKENIZER = ROOT / 'lab/models/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer.json'
OUT = ROOT / 'lab/analysis/replay-notification-images'
DOMAINS = [('overall', 'Celkem'), ('math', 'Matematika'), ('coding', 'Kódování'), ('tools', 'Nástroje'), ('instructions', 'Instrukce')]
OFFICIAL = {'math': 'svamp', 'coding': 'mbpp', 'tools': 'bfcl', 'instructions': 'ifeval'}


def read(path):
    return json.loads(Path(path).read_text())


def saving(before, after):
    return None if before is None or after is None or before <= 0 else 100 * (1 - after / before)


def number(value, sign=False):
    return '—' if value is None else (f'{value:+.1f}' if sign else f'{value:.1f}').replace('.', ',')


def percent(value, approx=False):
    return '—' if value is None else ('≈ ' if approx else '') + number(value) + ' %'


def normalize_official(raw):
    groups = {}
    for dataset, m in raw['results'].items():
        groups.setdefault('bfcl' if dataset.startswith('bfcl_') else dataset, []).append(m)
    result = {}
    for domain, benchmark in OFFICIAL.items():
        rows = groups[benchmark]
        n = sum(m['n'] for m in rows)
        if n <= 0 or any(m.get('ungraded', 0) for m in rows):
            raise ValueError('Incomplete official metrics')
        result[domain] = {dest: sum(m[source]*m['n'] for m in rows)/n for dest, source in
                          [('accuracy', 'accuracy'), ('tokens', 'mean_output_tokens'), ('thinking', 'mean_reasoning_tokens')]}
    result['overall'] = {k: statistics.mean(v[k] for v in result.values()) for k in ('accuracy', 'tokens', 'thinking')}
    return result


def add_savings(metrics, baseline, official=False, comparison=None):
    for domain, _ in DOMAINS:
        m, b = metrics[domain], baseline[domain]
        m['delta'] = 100*(m['accuracy'] - b['accuracy'])
        m['saving'] = saving(b['tokens'], m['tokens'])
        m['thinking_saving'] = saving(b['thinking'], m['thinking'])
    if official:
        # Official score uses a geometric mean over four benchmark ratios.
        # Apply the same weighting to the separately measured thinking tokens.
        for key, field in [('saving', 'tokens'), ('thinking_saving', 'thinking')]:
            ratios = [metrics[d][field]/baseline[d][field] for d in OFFICIAL if baseline[d][field] > 0]
            metrics['overall'][key] = 100*(1-math.prod(ratios)**(1/len(ratios))) if len(ratios)==4 else None
        if comparison is not None:
            if comparison['ungraded_benchmarks'] or any(comparison['full_evaluation'].values()):
                raise ValueError('Official evaluation incomplete')
            assert math.isclose(metrics['overall']['saving'], 100*comparison['average']['token_reduction'], abs_tol=1e-7)
            metrics['overall']['saving'] = 100*comparison['average']['token_reduction']
    return metrics


def official_report(group, preview=False):
    candidates = [(read(p), p.parent) for p in EVALS.glob('*/campaign.json')]
    matched = [(c,p) for c,p in candidates if c.get('checkpoint_group') == group and c.get('state') in ('running','complete')]
    campaign, reg = max(matched, key=lambda x:x[0].get('created_at',0))
    output = CLOUD / campaign['job_name'] / 'output/full-evaluation'
    base_raw = read(output/'runs/base/results.json')
    baseline = normalize_official(base_raw)
    contract_keys = ['revision','samples','seed','sampling','max_tokens','max_model_len','cohort_sha256','revisions']
    def compatible(raw):
        return all(raw.get(k)==base_raw.get(k) for k in contract_keys) and normalize_official(raw)==baseline
    points = [{'step':0, 'metrics':add_savings(normalize_official(base_raw),baseline,True)}]
    for c,p in sorted(candidates, key=lambda x:x[0].get('checkpoint_group',0)):
        step = c.get('checkpoint_group',0)
        if c.get('state')!='complete' or step>group or not c.get('backup_verified') or not c.get('released'):
            continue
        folder = CLOUD/c['job_name']/'output/full-evaluation'
        if not read(folder/'completion.json').get('complete'):
            continue
        raw = read(folder/'runs/base/results.json')
        if not compatible(raw):
            continue
        metrics = normalize_official(read(folder/'runs'/f'replay-{step}'/'results.json'))
        points.append({'step':step, 'metrics':add_savings(metrics,baseline,True,read(folder/'comparison.json'))})
    if not any(p['step']==group for p in points):
        if not preview:
            raise ValueError('Completed, backed-up result required before sending')
        points.append({'step':group,'metrics':None})
    stamp = dt.datetime.fromtimestamp(campaign.get('finished_at',campaign.get('observed_at',campaign['created_at'])),ZoneInfo('Europe/Prague'))
    elapsed = max(0, int(stamp.timestamp() - campaign['created_at']))
    duration = f'{elapsed//3600:02d}:{elapsed%3600//60:02d}:{elapsed%60:02d}'
    return dict(kind='official',group=group,baseline=baseline,points=points,approx=False,
                subtitle='Plná evaluace • 2 759 otázek × 8 odpovědí',
                meta='RTX 4090 • čas '+duration+' • baseline z cache • '+stamp.strftime('%d. %m. %Y, %H:%M %Z'),
                footer='Úspory: geometrický průměr čtyř oblastí; přesnost: aritmetický průměr.',
                caption=(f'ThinkingCap / Self-replay — NÁHLED plné evaluace checkpointu {group}; čeká na dokončení.' if preview else
                         f'ThinkingCap / Self-replay — plná evaluace checkpointu {group} dokončena. GPU uvolněna.'),preview=preview)


def development_report(group):
    pointer=read(ROOT/'lab/experiments/replay-scaled/current.json')
    reg=(ROOT/pointer['path']).parent
    launch=read(reg/'launch.json')
    folder=CLOUD/launch['job_name']/'output/training'
    live=read(reg/'live.json')
    points=[p for p in live['validation']['checkpoints'] if p['iteration']<=group]
    if not points or points[0]['iteration']!=0 or points[-1]['iteration']!=group:
        raise ValueError('Validation history does not match message')
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(TOKENIZER))
    history=[]
    for p in points:
        path=folder/'validations'/f'{p["iteration"]:06d}.jsonl'
        rows=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        if rows:
            pairs={(r['id'],r['sample']) for r in rows}
            if len(rows)!=p['metrics']['overall']['responses'] or len(pairs)!=len(rows):
                rows=[]  # An in-flight backup is not a complete validation.
        for r in rows:
            r['thinking']=len(tokenizer.encode(r['text'].split('</think>',1)[0].removeprefix('<think>'),add_special_tokens=False).ids)
        metrics={}
        for domain,_ in DOMAINS:
            m=p['metrics'][domain];rs=[r for r in rows if domain=='overall' or r['task']==domain]
            if rs and (not math.isclose(statistics.mean(r['total_tokens'] for r in rs),m['mean_tokens']) or
                       not math.isclose(statistics.mean(r['correct'] for r in rs),m['accuracy'])):
                raise ValueError('Validation artifact/summary mismatch')
            metrics[domain]={'accuracy':m['accuracy'],'tokens':m['mean_tokens'],
                             'thinking':statistics.mean(r['thinking'] for r in rs) if rs else None}
        history.append({'step':p['iteration'],'metrics':metrics})
    baseline=history[0]['metrics']
    for point in history:
        add_savings(point['metrics'],baseline)
    progress=live['progress'];elapsed=int(progress.get('elapsed',0))
    stamp=dt.datetime.fromtimestamp(points[-1]['completed_at'],ZoneInfo('Europe/Prague'))
    return dict(kind='development',group=group,baseline=baseline,points=history,approx=True,
                subtitle=f'Development • checkpoint {group} • '+stamp.strftime('%d. %m. %Y, %H:%M %Z'),
                meta=f'{progress.get("gpu_count",2)}× A100 • {progress["current"]} / {progress["total"]} otázek • čas {elapsed//3600:02d}:{elapsed%3600//60:02d}:{elapsed%60:02d}',
                footer='≈ Thinking dopočten ze zachovaného textu; — znamená dosud nedostupný údaj.',
                caption=f'ThinkingCap / Self-replay — dokončena development validace checkpointu {group}.',preview=False)


def render(report, destination):
    points=report['points'];latest=points[-1]['metrics'];baseline=report['baseline'];approx=report['approx']
    # Keep a phone-sized card as history grows: baseline plus latest six results.
    shown=points if len(points)<=7 else [points[0]]+points[-6:]
    height=1770
    img=Image.new('RGB',(1500,height),'#eef2f7');draw=ImageDraw.Draw(img)
    def text(x,y,value,size=26,color='#17263b',bold=False):
        f=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans'+('-Bold' if bold else '')+'.ttf',size)
        draw.text((x,y),value,font=f,fill=color)
    def panel(box):draw.rounded_rectangle(box,radius=22,fill='white')
    text(60,42,'ThinkingCap / Self-replay',46,bold=True)
    text(60,106,report['subtitle'],26,color='#53667d')
    text(60,150,report['meta'],24,color='#53667d')
    if latest:
        m=latest['overall']
        cards=[('PŘESNOST',percent(100*m['accuracy']),number(m['delta'],True)+' p. b. proti baseline'),
               ('ÚSPORA VŠECH TOKENŮ',percent(m['saving']),'Thinking + finální odpověď'),
               ('ÚSPORA THINKING',percent(m['thinking_saving'],approx),'Proti baseline')]
    else:
        cards=[(label,'—','Čeká na dokončení evaluace') for label in ['PŘESNOST','ÚSPORA VŠECH TOKENŮ','ÚSPORA THINKING']]
    for i,(label,value,detail) in enumerate(cards):
        x=60+i*465;panel((x,212,x+450,414))
        text(x+22,237,label,23,bold=True);text(x+22,276,value,56,color='#087f72',bold=True);text(x+22,358,detail,21,color='#53667d')
    panel((60,442,1440,960));text(88,467,'Vývoj proti baseline'+(' • poslední checkpointy' if len(points)>7 else ''),32,bold=True)
    cols=[90,340,600,900,1170]
    for x,label in zip(cols,['Checkpoint','Přesnost','Δ přesnosti','Úspora všech','Úspora thinking']):text(x,532,label,23,bold=True)
    for i,p in enumerate(shown):
        y=582+i*49;m=p['metrics']['overall'] if p['metrics'] else None
        row=['Baseline' if p['step']==0 else str(p['step'])]
        row+= [percent(100*m['accuracy']),number(m['delta'],True)+' p. b.',percent(m['saving']),percent(m['thinking_saving'],approx)] if m else ['Čeká','—','—','—']
        if i==len(shown)-1:draw.rounded_rectangle((78,y-5,1420,y+43),radius=8,fill='#dcf3ed')
        for x,value in zip(cols,row):text(x,y,value,25,bold=i==len(shown)-1)
    panel((60,985,1440,1450));text(88,1010,f'Checkpoint {report["group"]} podle oblastí',32,bold=True)
    for x,label in zip([90,360,760,1100],['Oblast','Přesnost: baseline → nyní','Úspora všech','Úspora thinking']):text(x,1078,label,22,bold=True)
    for i,(domain,label) in enumerate(DOMAINS):
        m=latest[domain] if latest else None
        row=[label,number(100*baseline[domain]['accuracy'])+' → '+(percent(100*m['accuracy']) if m else 'čeká'),percent(m['saving']) if m else '—',percent(m['thinking_saving'],approx) if m else '—']
        for x,value in zip([90,360,760,1100],row):text(x,1134+i*57,value,26,bold=i==0)
    panel((60,1475,1440,1650))
    previous=points[-2] if len(points)>1 else None
    text(88,1497,'Změna od checkpointu '+str(previous['step']) if previous else 'Srovnání',29,bold=True)
    if latest and previous:
        pm=previous['metrics'];m=latest['overall'];b=pm['overall']
        st=saving(b['tokens'],m['tokens']);sr=saving(b['thinking'],m['thinking'])
        if report['kind']=='official':
            st=100*(1-math.prod(latest[d]['tokens']/pm[d]['tokens'] for d in OFFICIAL)**.25)
            sr=100*(1-math.prod(latest[d]['thinking']/pm[d]['thinking'] for d in OFFICIAL)**.25)
        text(88,1550,f'Přesnost {number(100*(m["accuracy"]-b["accuracy"]),True)} p. b. • úspora tokenů {percent(st)} • thinking {percent(sr,approx)}',24)
        delta=100*(latest['coding']['accuracy']-pm['coding']['accuracy'])
        text(88,1594,f'Kódování: {percent(100*pm["coding"]["accuracy"])} → {percent(100*latest["coding"]["accuracy"])} ({number(delta,True)} p. b.)',24,color='#a54630' if delta<0 else '#087f72')
    else:
        text(88,1550,'NÁHLED • Výsledky checkpointu '+str(report['group'])+' se doplní po dokončení.',25)
        text(88,1594,'Starší řádky obsahují skutečné výsledky plné evaluace.',24,color='#53667d')
    text(60,1680,report['footer'],22,color='#53667d')
    text(60,1718,'Úspory proti baseline; všechny tokeny zahrnují thinking i finální odpověď.',22,color='#53667d')
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    img.save(destination)
    destination.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return destination


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preview-full',type=int);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.preview_full:
        report=official_report(args.preview_full,True)
        path=render(report,args.output or OUT/f'preview-official-{args.preview_full}.png')
    else:
        message=json.load(sys.stdin)['message']
        match=re.search(r'Plná evaluace checkpointu\s*(\d+) dokončena',message)
        if match:report=official_report(int(match[1]))
        else:
            match=re.search(r'Metriky z validace: iterace (\d+)',message)
            if not match:raise ValueError('Unsupported result message')
            report=development_report(int(match[1]))
        path=render(report,OUT/(report['kind']+'-'+str(report['group'])+'-'+hashlib.sha256(message.encode()).hexdigest()[:12]+'.png'))
    print(json.dumps({'photo':str(path),'caption':report['caption']},ensure_ascii=False))

if __name__=='__main__':main()
