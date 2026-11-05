"""Run all registered teacher/student seeds and publish verified full comparisons."""
import argparse
import json
import math
from pathlib import Path
import sys

from isolated_cloud import ROOT, verify
from parallel_campaign import run
from experiments import cloud


def publish(directory):
    state = json.loads((directory/'campaign.json').read_text())
    rows = []
    with cloud.ledger() as ledger:
        jobs = ledger['jobs']
        for seed, trial in state['trials'].items():
            job = jobs.get(trial['job'], {})
            if (trial['state'] != 'complete' or job.get('state') != 'complete'
                    or not job.get('backup_verified') or not job.get('released_at')):
                continue
            output = cloud.STATE/trial['job']/'output/full-evaluation'
            for arm in ('student', 'teacher'):
                path = output/f'{arm}-seed{seed}-vs-base.json'
                result = json.loads(path.read_text())
                if result['ungraded_benchmarks'] or any(result['full_evaluation'].values()):
                    raise ValueError('Incomplete official comparison')
                rows.append(dict(seed=int(seed), arm=arm, comparison=str(path),
                                 average=result['average'], domains=result['results']))
    aggregate = {}
    for arm in ('student', 'teacher'):
        selected = [r for r in rows if r['arm'] == arm]
        if selected:
            aggregate[arm] = dict(seeds=len(selected),
                accuracy_delta=sum(r['average']['accuracy_delta'] for r in selected)/len(selected),
                token_reduction=1-math.exp(sum(math.log1p(-r['average']['token_reduction'])
                                               for r in selected)/len(selected)),
                note='Descriptive equal-seed aggregation; confidence intervals remain per seed, not pooled.')
    complete = len(rows) == 6
    cloud.write(directory/'results.json', dict(complete=complete, comparisons=rows, aggregate=aggregate,
                 full_contest_evaluation=True, fresh_holdout=False, responses_per_model=22072))
    lines = ['# Plná evaluace teacher a student replay', '',
             'Všechny tři seedy dokončeny a zálohy ověřeny.' if complete else 'Průběžné ověřené výsledky.', '',
             'Každý model: 2 759 otázek × 8 odpovědí. Každý seed má čerstvou baseline na stejném workeru.',
             'Použité benchmarky nejsou nový nedotčený holdout. Není prováděn trénink.', '',
             '| Větev | Seed | Celkových tokenů proti baseline | Změna přesnosti | 95% CI přesnosti |',
             '|---|---:|---|---:|---|']
    for row in rows:
        m = row['average'];saving = 100*m['token_reduction']
        ci = m['intervals']['accuracy_delta']
        lines.append(f"| {row['arm']} | {row['seed']} | o {abs(saving):.2f} % {'méně' if saving >= 0 else 'více'} | "
                     f"{100*m['accuracy_delta']:+.2f} p. b. | [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}] p. b. |")
    lines += ['', 'Úplné domény, intervaly a zdrojové cesty jsou v [results.json]('+str(directory/'results.json')+').', '']
    (ROOT/'studies/teacher-pilot/REPORT-FULL-20261026.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--publish-only', action='store_true')
    args = parser.parse_args()
    release = args.release.resolve();directory = args.state.resolve()
    payload = verify(release)
    config = json.loads((release/'lab/experiments/data/isolated/config.json').read_text())
    try:
        if not args.publish_only:
            run(release, directory, config, payload, 3)
    finally:
        if (directory/'campaign.json').exists():
            publish(directory)


if __name__ == '__main__':
    main()
