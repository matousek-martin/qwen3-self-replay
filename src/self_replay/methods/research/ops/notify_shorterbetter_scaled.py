"""Authorized progress/stop/completion notices; durable at-most-once delivery intents."""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import fcntl
import html
import json
import os
from pathlib import Path
import sys
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, '/srv/playground')
from experiments import cloud
from project_alerts import Alerts

DIRECTORY = ROOT / 'lab/experiments/shorterbetter-scaled/notifications'
DISPLAY_TIMEZONE = ZoneInfo('Europe/Prague')
REASONS = {
    'registered_updates_completed': 'registrovaný počet aktualizací dokončen',
    'persistent_regression': 'opakované zhoršení přesnosti nebo dokončování odpovědí',
    'no_useful_progress_within_window': 'bez užitečného pokroku v registrovaném okně; není to důkaz neúčinnosti metody',
    'useful_plateau_needs_confirmation': 'užitečné zkrácení se stabilizovalo; potřebuje nezávislé potvrzení',
    'deadline_or_stop_request': 'časový/finanční limit nebo požadavek na zastavení',
    'throughput_benchmark_not_efficacy': 'výkonnostní test dokončen; účinnost metody tím není potvrzena',
    'out_of_memory': 'nedostatek GPU paměti',
    'numerical_failure': 'numerická chyba nebo nesynchronizované/neúčinné aktualizace vah',
    'worker_error': 'chyba workeru; podrobnosti v uloženém logu',
    'runtime_projection_exceeded': 'naměřená rychlost přesahuje povolený časový nebo finanční limit',
    'engineering_or_training_failure': 'technická kontrola nebo trénink selhaly; podrobnosti v archivu',
}


def send_html(message):
    result = subprocess.run(['/usr/local/bin/codex-telegram', 'send', message, '--parse-mode', 'HTML'],
                            capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError('Telegram delivery may be uncertain; automatic retry suppressed')


LIVE_FILES = ('phase.json', 'training/progress.json', 'training/learning-curve.json', 'training/training-series.json',
              'training/decision.json', 'verification-reference/progress.json',
              'verification-resumed/progress.json', 'deterministic-resume-result.json',
              'launch-failure.json')


def refresh_live_status(job, output):
    """Small independent read so checkpoint transfers cannot hide live progress."""
    pod = cloud.api('pods/' + job['pod_id'])
    code = "from pathlib import Path\nimport json,time\np=Path('/workspace/project/output')\nfiles={}\n"
    code += 'for name in ' + repr(LIVE_FILES) + ':\n'
    code += " f=p/name\n files[name]=json.loads(f.read_text()) if f.is_file() else None\n"
    code += "print(json.dumps({'observed_at':time.time(),'files':files}))\n"
    value = json.loads(cloud.remote(pod, "python3 - <<'PY'\n" + code + "PY", timeout=20))
    if set(value['files']) != set(LIVE_FILES):
        raise ValueError('Incomplete live notification snapshot')
    cloud.write(output.parent/'notification-live.json', value)
    return value


def status_reader(output, job):
    snapshot_path = output.parent/'notification-live.json'
    snapshot = (json.loads(snapshot_path.read_text())
                if snapshot_path.exists() and not job.get('released_at') else {})
    def read(relative):
        if relative in snapshot.get('files', {}):
            return snapshot['files'][relative] or {}
        path = output/relative
        return json.loads(path.read_text()) if path.exists() else {}
    return read, snapshot


def format_notification(project, kind, summary, job, output, protocol=None, now=None):
    """One readable HTML message; current progress and last evaluation stay distinct."""
    now = time.time() if now is None else now
    protocol = protocol or {}
    read_status, live = status_reader(output, job)
    def read(name):
        return read_status('training/'+name)
    curve = read('learning-curve.json')
    checkpoints = curve.get('checkpoints', [])
    progress = read('progress.json')
    current = progress.get('rollout_batches')
    if current is None:
        current = checkpoints[-1].get('rollout_batches', 0) if checkpoints else 0
    total = protocol.get('training', {}).get('rollout_batches')
    iterations = f'{current:g} / {total:g}' if total is not None else f'{current:g} / ?'
    elapsed = max(0, int((job.get('released_at') or now) - job.get('created_at', now)))
    hours, rest = divmod(elapsed, 3600)
    minutes, seconds = divmod(rest, 60)
    gpu = job.get('gpu', 'GPU').removeprefix('NVIDIA ')
    provider = job.get('provider', 'runpod')
    esc = html.escape
    lines = [f'<b>Project: {esc(project)}</b>']
    if protocol.get('execution', {}).get('paired_short_experiment'):
        alpha = protocol['arms']['shorterbetter']['alpha']
        lines.append(f'Experiment: {"kontrolní" if alpha == 2 else "vyšší váha správnosti"} (α={alpha:g})')
    if kind not in ('Progress', 'Training'):
        lines.append(f'Summary: {esc(summary)}')
    full = protocol.get('execution', {}).get('multidomain_full_training')
    phase = read_status('phase.json').get('phase')
    if full:
        stages = {'deterministic_resume_reference':'technické ověření před tréninkem',
                  'gpu_resume_verification':'test obnovy checkpointu před tréninkem',
                  'main_training':'hlavní trénink' if current else 'hlavní běh — inicializace / vstupní validace',
                  'finished':'výpočet dokončen'}
        lines.append('Fáze: ' + esc(stages.get(phase, 'příprava workeru')))
        technical = {'deterministic_resume_reference':'verification-reference',
                     'gpu_resume_verification':'verification-resumed'}.get(phase)
        if technical:
            step = read_status(technical+'/progress.json').get('optimizer_step')
            lines.append(f'Technická kontrola: {step} / 8 aktualizací' if step is not None
                         else 'Technická kontrola: inicializace')
            lines.append(f'Technické kroky se do {total:g} tréninkových iterací nepočítají.')
    lines += [f'GPU: {job.get("gpu_count", 1)}× {esc(gpu)} ({esc(provider)})',
             f'Iterations{" (trénink)" if full else ""}: {iterations}',
             f'Time elapsed: {hours:02d}:{minutes:02d}:{seconds:02d}',
             f'Timestamp: {datetime.fromtimestamp(now, DISPLAY_TIMEZONE):%Y-%m-%d %H:%M:%S %Z} (Praha)']
    if live:
        age = max(0, int(now-live['observed_at']))
        lines.append(f'Stav ověřen: {datetime.fromtimestamp(live["observed_at"], DISPLAY_TIMEZONE):%H:%M:%S %Z} (Praha)'
                     + (f' (před {age//60} min; živé spojení není aktuální)' if age > 300 else ''))
    if checkpoints:
        lines += ['', f'Metriky z validace: iterace {checkpoints[-1]["rollout_batches"]:g}',
                  'Hodnoty: baseline → poslední validace']
        def metrics(label, first, last):
            change = (last['accuracy'] - first['accuracy']) * 100
            tokens = first['mean_total_tokens']
            saving = (1 - last['mean_total_tokens']/tokens)*100 if tokens else None
            delta = f' ({saving:+.1f} % úspora)' if saving is not None else ''
            return ['', f'<b>{esc(label)}</b>',
                f'Accuracy: {first["accuracy"]:.1%} → {last["accuracy"]:.1%} ({change:+.1f} p. b.)',
                f'Tokeny / odpověď: {tokens:.0f} → {last["mean_total_tokens"]:.0f}{delta}']
        lines += metrics('Celkem', checkpoints[0], checkpoints[-1])
        if 'truncation_rate' in checkpoints[-1]:
            lines.append(f'Useknuté odpovědi: {checkpoints[-1]["truncation_rate"]:.1%}')
        labels = {'math': 'Matematika', 'coding': 'Kódování', 'tools': 'Nástroje', 'instructions': 'Instrukce'}
        domains = curve.get('domains', {})
        for domain in list(labels) + sorted(set(domains) - set(labels)):
            rows = domains.get(domain, {}).get('checkpoints', [])
            if rows:
                lines += metrics(labels.get(domain, domain), rows[0], rows[-1])
    return '\n'.join(lines)


def notices(name, job, output, now=None, validation_only=False):
    """Pure planning: completion release/backup verification is performed by the caller."""
    now = time.time() if now is None else now
    planned = []
    credit = job.get('last_account_credit')
    funding = job.get('incremental_funding', {})
    if funding and credit is not None and not job.get('released_at'):
        if credit <= funding['stop_credit_usd']:
            planned.append(('credit-stop', 'Billing', f'Kredit Vast {credit:.2f} USD; žádám uložení a ukončení běhu před vyčerpáním kreditu.'))
        elif credit <= funding['warning_credit_usd']:
            planned.append(('credit-low', 'Billing', f'Kredit Vast {credit:.2f} USD; doplň kredit pro pokračování do 300 iterací; při 12 USD se běh bezpečně ukončí.'))
    read_status, _ = status_reader(output, job)
    def read(filename, default):
        return read_status('training/'+filename) or default
    curve = read('learning-curve.json', {}).get('checkpoints', [])
    decision = read('decision.json', {})
    phase = read_status('phase.json').get('phase')
    failures = [json.loads(p.read_text()) for folder in ('training', 'startup-check', 'engineering-check', 'deterministic-reference', 'resume-check', 'verification-reference', 'verification-resumed')
                for p in sorted((output/folder).glob('failure-rank-*.json'))]
    launch_failure = read_status('launch-failure.json')
    if launch_failure:
        failures.append({'category': launch_failure['reason']})
    failure = next((r for r in failures if r['category'] != 'worker_error'), failures[0] if failures else None)
    if failure:
        planned.append(('failure', 'Technical', 'Běh selhal: ' + REASONS[failure['category']] + '; probíhá záloha a uvolnění GPU.'))
    if job.get('state') == 'running':
        purpose = 'krátké technické ověření, nikoli hlavní trénink' if job.get('verification_only') else 'worker spuštěn; následuje stav aktuální fáze'
        planned.append(('start', 'Training', f"Běží {job.get('gpu_count', 1)}× {job.get('gpu', 'GPU')}; {purpose}."))
        if phase:
            labels = {'cheap_verification_reference': 'Ověřuji skutečné aktualizace vah na 4090 s krátkými vstupy.',
                'cheap_verification_resume': 'Porovnávám obnovený checkpoint s nepřerušeným během na 4090.',
                'startup_verification': 'Ověřuji numeriku a skutečné aktualizace vah.',
                'throughput_verification': 'Měřím plné rollouty a validaci; zatím technický test.',
                'gpu_resume_verification': 'Ověřuji obnovení GPU checkpointu proti nepřerušenému běhu.',
                'deterministic_resume_reference': 'Ověřuji deterministický výpočet a checkpoint před hlavním tréninkem.',
                'main_training': 'Technické kontroly prošly; začíná hlavní trénink z původního modelu.'}
            labels['multidomain_pilot'] = 'Krátký pilot čtyř oblastí na jedné RTX4090; po třech rolloutech uložení a uvolnění GPU.'
            labels['multidomain_a100x2_benchmark'] = 'Ověření čtyř oblastí na2×A100 80GB; dvě plné iterace s evaluací, potom záloha a uvolnění GPU; 100iterací zatím nespouštím.'
            if phase in labels:
                planned.append(('phase-'+phase, 'Progress', labels[phase]))
    projection_path = output/'full-run-projection.json'
    if projection_path.exists():
        projection=json.loads(projection_path.read_text())
        hours=projection.get('projected_full_job_hours')
        if isinstance(hours,(int,float)) and hours>0:
            planned.append(('projection','Progress',
                f"Naměřený konzervativní odhad celého běhu: {hours:.1f} h; při aktuální sazbě přibližně {hours*job.get('rate',0):.0f} USD za GPU. Kvalitativní stop může běh zkrátit; nejde o časový limit."))
    if curve:
        r = curve[-1]
        text = f"Validace po iteraci {r['rollout_batches']:g} dokončena."
        planned.append((f"eval-{r['optimizer_step']}", 'Progress', text))
    if job.get('state') == 'running' and not job.get('user_pause_requested_at') and now - job['created_at'] >= 3600:
        progress = read('progress.json', {})
        technical_stage = {'cheap_verification_reference':'verification-reference',
                           'cheap_verification_resume':'verification-resumed',
                           'startup_verification':'startup-check',
                           'deterministic_resume_reference':'deterministic-reference',
                           'throughput_verification':'engineering-check',
                           'gpu_resume_verification':'resume-check'}.get(phase)
        if technical_stage:
            path = output/technical_stage/'progress.json'
            progress = json.loads(path.read_text()) if path.exists() else {}
        elapsed = now-job['created_at']
        pilot_phase = phase in ('hardware_comparison', 'multidomain_pilot', 'multidomain_a100x2_benchmark')
        label = 'Technická kontrola' if technical_stage else ('Technický pilot' if pilot_phase else 'Trénink')
        text = (f"{label} pokračuje.\n"
                f"Konzervativní náklad: {elapsed/3600*job.get('budget_rate', 0):.2f} USD")
        if technical_stage:
            text += ' Ověřování před hlavním tréninkem; nejde o konečný výsledek metody.'
        elif not curve:
            text += ' Zatím není kompletní baseline; zlepšení nelze posoudit.'
        planned.append((f"hour-{int(elapsed // 3600)}", 'Progress', text))
    if decision.get('stop'):
        reason = ('pozastavení na přání uživatele kvůli přípravě dat pro všechny oblasti ThinkingCap'
                  if job.get('user_pause_requested_at') else REASONS.get(decision['reason'], decision['reason']))
        if job.get('user_stop_requested_at'):
            reason = 'zastavení na výslovné přání uživatele'
        planned.append(('stop', 'Training', 'Ukončuji běh: ' + reason + '; následuje uložení a uvolnění GPU.'))
    if job.get('released_at'):
        reason = REASONS[failure['category']] if failure else REASONS.get(decision.get('reason'), job.get('state', 'ukončen'))
        if job.get('user_pause_requested_at') and not failure:
            reason = 'pozastaven na přání uživatele; před dalším tréninkem připravíme data a validaci všech čtyř oblastí'
        if job.get('user_stop_requested_at') and not failure:
            reason = 'zastaven na výslovné přání uživatele; bez automatického restartu'
        verification = output/'verification-result.json'
        if verification.exists() and json.loads(verification.read_text()).get('status') == 'passed':
            reason = 'technické ověření aktualizací a obnovy checkpointu prošlo; nejde o výsledek účinnosti metody'
        pilot = output/'multidomain-pilot-result.json'
        full_result = output/'multidomain-training-result.json'
        if full_result.exists():
            result = json.loads(full_result.read_text())
            reason = f"hlavní běh {result['status']}, {result['rollout_batches']:g}/300 iterací; {reason}"
        if pilot.exists():
            result = json.loads(pilot.read_text())
            reason = f"pilot čtyř oblastí {result['status']}, {result['rollout_batches']:g} rolloutů; checkpoint zachován, plný běh nespouštím"
            validation = result.get('domain_validation', {})
            if validation and any(r.get('optimizer_step') != result.get('optimizer_steps') for r in validation.values()):
                reason += '; závěrečná evaluace neproběhla, zlepšení modelu není ověřeno'
            projection_path = output/'pilot-projection.json'
            if job.get('backup_verified') and projection_path.exists():
                projection = json.loads(projection_path.read_text())
                if projection.get('status') == 'early_measured_projection' and 'targets' in projection:
                    parts = []
                    for target, value in projection['targets'].items():
                        low, high = value['hours_range']
                        cost_low, cost_high = value['conservative_cost_range_usd']
                        parts.append(f"{target} iterací: {low:.1f}–{high:.1f} h, konzervativní rezerva {cost_low:.0f}–{cost_high:.0f} USD")
                    planned.append(('pilot-duration','Results',
                        f"Naměřeno na {job.get('gpu_count', 1)}× {job.get('gpu', 'GPU')}: "+ '\n'.join(parts) +
                        f". Jen {projection['steady_rollout_samples']} časových vzorků po zahřátí; zahrnuje průběžnou interní validaci, nikoli velkou potvrzovací/soutěžní evaluaci. Plný běh nespouštím."))
                elif projection.get('status') == 'early_measured_projection':
                    low, high = projection['single4090_hours_range']
                    two = projection['two4090_scenarios_hours']
                    planned.append(('pilot-duration', 'Results',
                        f"Odhad 300 iterací včetně interní validace: 1×4090 {low:.1f}–{high:.1f} h "
                        f"(měřeno jen {projection['steady_rollout_samples']} iterací po zahřátí). "
                        f"2×4090 nebyly měřeny: ideální 2× zrychlení {two['ideal_2x']:.1f} h, "
                        f"scénář bez zrychlení {two['no_speedup']:.1f} h; skutečnost může být i pomalejší. "
                        "Nezahrnuje finální soutěžní evaluaci; celý trénink nespouštím."))
        planned.append(('final', 'Results', f"Běh ukončen: {reason}\nGPU uvolněny: ano\n"
            f"Záloha: {'ověřena' if job.get('backup_verified') else 'neověřena'}\n"
            f"Nákladový odhad: {job.get('charged_estimate_usd', 0):.2f} USD"))
    if validation_only:
        # Routine full-run updates require a new post-training measurement.
        # Existing event IDs preserve delivery deduplication across polls.
        operational = {'failure', 'credit-low', 'credit-stop', 'stop', 'final'}
        measured = len(curve) >= 2 and curve[-1].get('optimizer_step', 0) > 0
        planned = [event for event in planned if event[0] in operational
                   or (measured and event[0].startswith('eval-'))]
    return planned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    registry_path = DIRECTORY / 'jobs.json'
    if not registry_path.exists():
        return
    with (DIRECTORY / 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        registry = json.loads(registry_path.read_text())
        jobs = json.loads((cloud.STATE / 'ledger.json').read_text())['jobs']
        alerts = Alerts(DIRECTORY / 'delivery.sqlite3', 'ThinkingCap / ShorterBetter', sender=send_html)
        for name, config in registry.items():
            if config.get('telegram_authorized') is not True or name not in jobs:
                continue
            job = dict(jobs[name])
            if job.get('hardware_scope') != 'shorterbetter-scaled-v1':
                continue
            output = cloud.STATE / name / 'output'
            protocol_path = Path(config.get('directory', output))/'protocol.json'
            protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
            if (protocol.get('execution', {}).get('multidomain_full_training')
                    and job.get('pod_id') and not job.get('released_at')):
                try:
                    refresh_live_status(job, output)
                except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
                    cloud.write(output.parent/'notification-live-error.json',
                                {'at':time.time(),'type':type(exc).__name__})
            alerts.formatter = lambda project, kind, text, action, detected: format_notification(
                project, kind, text, job, output, protocol)
            for key, kind, text in notices(name, job, output,
                    validation_only=bool(protocol.get('execution', {}).get('multidomain_full_training'))):
                if (key.startswith('eval-') and protocol.get('continuation_parent')
                        and int(key.split('-')[1]) <= protocol['continuation_parent']['optimizer_step']):
                    continue  # Inherited validation is not a new measurement.
                attempt = job.get('notification_attempt', 0)
                incident = name + (f':attempt-{attempt}' if attempt else '') + ':' + key
                # Never retry changed/uncertain delivery for this specific event.
                if alerts.db.execute('SELECT 1 FROM incidents WHERE key=?', (incident,)).fetchone():
                    continue
                if key == 'final':
                    if job.get('pod_id') and any(p['id'] == job['pod_id'] or p['name'] == job.get('pod_name')
                                               for p in cloud.list_pods(job.get('provider', 'runpod'))):
                        continue
                    if job.get('backup_verified'):
                        cloud.verify_backup(name)
                if args.dry_run:
                    print(json.dumps({'event': incident, 'message': format_notification(
                        'ThinkingCap / ShorterBetter', kind, text, job, output, protocol)}, ensure_ascii=False))
                else:
                    alerts.emit(incident, kind, text)


if __name__ == '__main__':
    main()
