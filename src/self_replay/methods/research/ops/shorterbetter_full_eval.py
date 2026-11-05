"""Launch one authorized evaluation and notify only after verified release."""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import html
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,'/srv/playground')
from experiments import cloud
from ops.isolated_cloud import verify
from ops.notify_shorterbetter_scaled import send_html
from project_alerts import Alerts


def notify(directory,launch):
    name=launch['job_name'];job=json.loads((cloud.STATE/'ledger.json').read_text())['jobs'][name]
    if not job.get('released_at'):raise RuntimeError('Cannot report completion before GPU release')
    if any(p['id']==job.get('pod_id') for p in cloud.list_pods(job.get('provider','vast'))):
        raise RuntimeError('Evaluation GPU is still allocated')
    out=cloud.STATE/name/'output/full-evaluation'
    lines=['<b>Project: ThinkingCap / ShorterBetter</b>']
    success=job.get('backup_verified') and (out/'completion.json').exists()
    if success:
        cloud.verify_backup(name)
        comparison=json.loads((out/'comparison.json').read_text())
        if comparison['ungraded_benchmarks'] or any(comparison['full_evaluation'].values()):
            raise ValueError('Incomplete official full evaluation')
        config=json.loads((directory/'evaluation.json').read_text())
        if config['candidate'].get('alpha') is not None:
            lines.append(f"Experiment: α={config['candidate']['alpha']:g}")
        step=config['candidate']['optimizer_step']
        label=f'iterace {step//4} (checkpoint {step})' if step%4==0 else f'checkpointu {step}'
        lines+=[f'Summary: Plná evaluace {label} dokončena.',
                'Evaluace: 2 759 otázek × 8 odpovědí pro baseline i kandidáta',
                f"GPU: 1× {job['gpu'].removeprefix('NVIDIA ')} (vast)",
                'GPU uvolněna: ano','Záloha: ověřena',
                f"Průměrná změna accuracy: {comparison['average']['accuracy_delta']*100:+.2f} p. b.",
                f"Oficiální úspora tokenů: {comparison['average']['token_reduction']*100:.1f} %"]
        for domain,label in [('svamp','Matematika (SVAMP)'),('mbpp','Kódování (MBPP+)'),('ifeval','Instrukce (IFEval)')]:
            r=comparison['results'][domain]
            lines+=['',f'<b>{label}</b>',f"Accuracy: {r['base']['accuracy']:.1%} → {r['candidate']['accuracy']:.1%}",
                    f"Tokeny / odpověď: {r['base']['mean_output_tokens']:.0f} → {r['candidate']['mean_output_tokens']:.0f}"]
        bf=[r for k,r in comparison['results'].items() if k.startswith('bfcl_')]
        n=sum(r['paired_rows'] for r in bf)
        def pooled(side,key):return sum(r['paired_rows']*r[side][key] for r in bf)/n
        lines+=['','<b>Nástroje (BFCL)</b>',f"Accuracy: {pooled('base','accuracy'):.1%} → {pooled('candidate','accuracy'):.1%}",
                f"Tokeny / odpověď: {pooled('base','mean_output_tokens'):.0f} → {pooled('candidate','mean_output_tokens'):.0f}"]
    else:
        lines+=['Summary: Plná evaluace nebyla dokončena; zachované výstupy vyžadují kontrolu.',
                'GPU uvolněna: ano',f"Záloha ověřena: {'ano' if job.get('backup_verified') else 'ne'}"]
    elapsed=max(0,int(job['released_at']-job['created_at']));h,rest=divmod(elapsed,3600);m,s=divmod(rest,60)
    lines+=['',f'Time elapsed: {h:02d}:{m:02d}:{s:02d}',
            f'Timestamp: {datetime.now(ZoneInfo("Europe/Prague")):%Y-%m-%d %H:%M:%S %Z} (Praha)']
    message='\n'.join(lines)
    formatter=lambda *args:message
    alerts=Alerts(directory/'notifications.sqlite3','ThinkingCap / ShorterBetter',sender=send_html,formatter=formatter)
    key=name+':final'
    if not alerts.db.execute('SELECT 1 FROM incidents WHERE key=?',(key,)).fetchone():
        alerts.emit(key,'Results' if success else 'Technical','Full evaluation complete' if success else 'Full evaluation incomplete')
    cloud.write(directory/'completion-status.json',{'success':bool(success),'released':True,'backup_verified':bool(job.get('backup_verified'))})
    pointer=directory.parent/'current.json'
    if pointer.exists():
        current=json.loads(pointer.read_text())
        if current.get('job_name')==name:
            current.update(status='completed_released' if success else 'failed_released',
                           released_at=job['released_at'],backup_verified=bool(job.get('backup_verified')))
            cloud.write(pointer,current)


def main():
    os.umask(0o077)
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--notify-only',action='store_true')
    args=parser.parse_args();directory=args.directory.resolve();launch=json.loads((directory/'launch.json').read_text())
    name=launch['job_name']
    if not args.notify_only:
        if verify(Path(launch['release']))!=launch['payload_sha256']:raise ValueError('Evaluation payload changed')
        jobs=json.loads((cloud.STATE/'ledger.json').read_text())['jobs']
        if name in jobs:
            if not jobs[name].get('released_at'):cloud.watch(name)
        else:
            command=[sys.executable,str(ROOT/'ops/isolated_cloud.py'),'run','--release',launch['release'],
                     '--name',name,'--entry','experiments/shorterbetter/run-scaled.sh','--gpu',launch['gpu'],
                     '--gpu-count','1','--hardware-scope','shorterbetter-scaled-v1','--hours','1',
                     '--max-cost-usd',str(launch['max_cost_usd']),'--max-runtime-hours',str(launch['max_runtime_hours']),
                     '--provider','vast','--vast-offer-id',str(launch['offer_id']),'--vast-disk-gb','50','--data-upload-timeout','900']
            result=subprocess.run(command)
            if result.returncode:
                job=json.loads((cloud.STATE/'ledger.json').read_text())['jobs'].get(name,{})
                if not job.get('released_at'):raise RuntimeError('Controller failed; reconcile existing instance before retry')
    notify(directory,launch)


if __name__=='__main__':main()
