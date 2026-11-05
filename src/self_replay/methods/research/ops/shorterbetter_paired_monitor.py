"""Deduplicated orchestration failure alert for the authorized paired campaign."""
import html
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,'/srv/playground')
from ops.shorterbetter_paired import ROOT,read
from ops.notify_shorterbetter_scaled import send_html
from project_alerts import Alerts
from datetime import datetime
from zoneinfo import ZoneInfo

def main():
 pointer=ROOT/'lab/experiments/shorterbetter-paired/current.json'
 if not pointer.exists():return
 path=ROOT/read(pointer)['path'];c=read(path)
 if c.get('state')!='needs_inspection':return
 message='\n'.join(['<b>Project: ThinkingCap / ShorterBetter</b>',
  'Summary: Automatické navázání experimentů vyžaduje technickou kontrolu.',
  'Stav: '+html.escape(c.get('error','Neznámá chyba')[:350]),
  f'Timestamp: {datetime.now(ZoneInfo("Europe/Prague")):%Y-%m-%d %H:%M:%S %Z} (Praha)'])
 alerts=Alerts(path.parent/'controller-alerts.sqlite3','ThinkingCap / ShorterBetter',sender=send_html,formatter=lambda *args:message)
 key='controller-failure:'+str(c.get('error_at'))
 if not alerts.db.execute('SELECT 1 FROM incidents WHERE key=?',(key,)).fetchone():alerts.emit(key,'Technical','Paired controller stopped')
if __name__=='__main__':main()
