import datetime
import html
import sqlite3
import subprocess
from zoneinfo import ZoneInfo

def deliver_once(db_path,key,message,sender=None):
    with sqlite3.connect(db_path,timeout=15) as db:
        db.execute('CREATE TABLE IF NOT EXISTS deliveries (key TEXT PRIMARY KEY,status TEXT,message TEXT)')
        if db.execute('SELECT 1 FROM deliveries WHERE key=?',(key,)).fetchone():return False
        db.execute('INSERT INTO deliveries VALUES (?,?,?)',(key,'uncertain',message));db.commit()
        if sender:sender(message)
        else:
            subprocess.run(['/usr/local/bin/codex-telegram','send',message,'--parse-mode','HTML'],check=True,timeout=30,capture_output=True)
        db.execute('UPDATE deliveries SET status=? WHERE key=?',('accepted',key));db.commit()
    return True

def validation_message(progress,point,baseline):
    elapsed=int(progress.get('elapsed',0));stamp=datetime.datetime.fromtimestamp(point['completed_at'],ZoneInfo('Europe/Prague')).strftime('%Y-%m-%d %H:%M:%S %Z (Praha)')
    lines=['<b>Project: ThinkingCap / Self-replay scaled</b>','Summary: Dokončena development validace.',
           f"GPU: {progress.get('gpu_count',2)}× A100-SXM4-80GB",f"Iterations (otázky): {progress['current']} / {progress['total']}",
           f"Time elapsed: {elapsed//3600:02d}:{elapsed%3600//60:02d}:{elapsed%60:02d}",f'Timestamp: {stamp}',
           f"Metriky z validace: iterace {point['iteration']}",'Development monitor; není to plná oficiální evaluace.']
    for domain,label in [('overall','Celkem'),('math','Matematika'),('coding','Kódování'),('tools','Nástroje'),('instructions','Instrukce')]:
        v=point['metrics'][domain];b=baseline['metrics'][domain]
        lines+=['',label,f"Accuracy: {100*b['accuracy']:.1f}% → {100*v['accuracy']:.1f}% ({100*(v['accuracy']-b['accuracy']):+.1f} p. b.)",
                f"Tokeny / odpověď: {b['mean_tokens']:.0f} → {v['mean_tokens']:.0f}"]
        lines.append(f"Useknuté odpovědi: {100*v['truncation']:.1f}%")
        if 'accuracy_delta_ci' in v:lines.append(f"95% interval změny: {100*v['accuracy_delta_ci'][0]:+.1f} až {100*v['accuracy_delta_ci'][1]:+.1f} p. b.")
    return '\n'.join(lines)
