"""Export measured training median output length and accuracy, overall/by domain.

Read-only with respect to the worker and canonical backups. One plotted point is
one complete balanced rollout whose four optimizer updates have finished.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics

DOMAINS = ('math', 'coding', 'tools', 'instructions')


def collect(training):
    groups = defaultdict(list)
    sources = {}
    progress = training/'progress.json'
    if not progress.exists():
        raise ValueError('Missing training progress; cannot establish completed iterations')
    data = progress.read_bytes()
    sources[str(progress)] = hashlib.sha256(data).hexdigest()
    completed_step = json.loads(data)['optimizer_step']
    paths = sorted(training.glob('rank-*/rollouts.jsonl'))
    if len(paths) != 4:
        raise ValueError('Expected all four rank logs')
    for path in paths:
        data = path.read_bytes()
        sources[str(path)] = hashlib.sha256(data).hexdigest()
        for line in data.splitlines():
            row = json.loads(line)
            if row['split'] == 'train':
                groups[row['optimizer_step']].append(row)
    result, excluded = [], []
    for step, rows in sorted(groups.items()):
        if step % 4 or step + 4 > completed_step:
            excluded.append({'optimizer_step': step, 'reason': 'optimizer updates not complete'})
            continue
        counts = Counter(r['prompt_id'] for r in rows)
        if (len(rows) != 512 or len(counts) != 64 or set(counts.values()) != {8}
                or Counter(r['task'] for r in rows) != {d:128 for d in DOMAINS}):
            excluded.append({'optimizer_step': step, 'reason': 'incomplete balanced four-rank snapshot'})
            continue
        for domain in ('overall', *DOMAINS):
            selected = rows if domain == 'overall' else [r for r in rows if r['task'] == domain]
            result.append({'iteration':step//4+1, 'generation_optimizer_step':step,
                           'domain':domain, 'responses':len(selected),
                           'median_output_tokens':statistics.median(r['total_tokens'] for r in selected),
                           'mean_output_tokens':statistics.mean(r['total_tokens'] for r in selected),
                           'accuracy':statistics.mean(int(r['correct']) for r in selected),
                           'truncation_rate':statistics.mean(int(r['truncated']) for r in selected)})
    if not result:
        raise ValueError('No complete measured rollouts available')
    return result, {'source_sha256':sources, 'completed_optimizer_step':completed_step,
                    'excluded_rollouts':excluded,
                    'measurement':'Generated output tokens (thinking + answer); changing training batches, not fixed validation.',
                    'iteration_semantics':'Point k uses outputs generated before the four updates of rollout k; only finished rollouts included.'}


def moving_average(values, window):
    return [None if i+1 < window else statistics.mean(values[i-window+1:i+1])
            for i in range(len(values))]


def render(training, protocol_path, out, window=5):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    if window < 1:
        raise ValueError('Smoothing window must be positive')
    rows, provenance = collect(training)
    protocol_data = protocol_path.read_bytes()
    protocol = json.loads(protocol_data)
    out.mkdir(parents=True, exist_ok=True)
    with (out/'training-metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    labels = {'overall':'All four domains', 'math':'Mathematics', 'coding':'Coding',
              'tools':'Tool use', 'instructions':'Instruction following'}
    reward = protocol['arms']['shorterbetter']
    with PdfPages(out/'training-curves.pdf') as pdf:
        for domain, label in labels.items():
            selected = [r for r in rows if r['domain'] == domain]
            x = [r['iteration'] for r in selected]
            length = [r['median_output_tokens'] for r in selected]
            accuracy = [100*r['accuracy'] for r in selected]
            fig, left = plt.subplots(figsize=(10, 5.5))
            right = left.twinx()
            smooth = len(x) >= window and window > 1
            red, blue = '#d52e35', '#2456c4'
            left.plot(x, length, color=red, alpha=.30 if smooth else 1, marker='.', linewidth=1)
            right.plot(x, accuracy, color=blue, alpha=.30 if smooth else 1, marker='.', linewidth=1)
            if smooth:
                left.plot(x, moving_average(length, window), color=red, linewidth=2,
                          label=f'Median output length ({window}-rollout trailing mean)')
                right.plot(x, moving_average(accuracy, window), color=blue, linewidth=2,
                           label=f'Accuracy ({window}-rollout trailing mean)')
            else:
                left.lines[0].set_label('Median output length (raw)')
                right.lines[0].set_label('Accuracy (raw)')
            left.set_xlabel('Completed rollout iteration (4 optimizer updates each)')
            left.set_ylabel('Median generated output tokens', color=red)
            right.set_ylabel('Training accuracy (%)', color=blue)
            left.set_ylim(bottom=0)
            right.set_ylim(0, 100)
            left.set_xlim(min(x)-.25, max(x)+.25)
            left.grid(alpha=.18)
            handles, names = left.get_legend_handles_labels()
            h, n = right.get_legend_handles_labels()
            fig.legend(handles+h, names+n, loc='upper center', bbox_to_anchor=(.5,.875), frameon=False, fontsize=9)
            fig.suptitle(f'ShorterBetter / Qwen3-0.6B — {label}', fontsize=15, y=.98)
            fig.text(.5, .91, f'α = {reward["alpha"]:g}, length β = {reward["length_beta"]:g} · seed {protocol["seeds"][0]} · '
                     f'{selected[0]["responses"]} responses / rollout', fontsize=10, ha='center')
            fig.text(.5,.025, 'Changing training batches; not held-out validation. '
                     + ('Faint lines: raw measurements; bold lines: trailing means.' if smooth else 'Raw measurements; no extrapolation.'),
                     ha='center', fontsize=8)
            fig.subplots_adjust(top=.76, bottom=.15, left=.10, right=.90)
            fig.savefig(out/f'training-{domain}.png', dpi=180)
            pdf.savefig(fig)
            plt.close(fig)
    provenance.update(protocol_sha256=hashlib.sha256(protocol_data).hexdigest(),
                      smoothing={'method':'trailing arithmetic mean of per-rollout statistics', 'window':window},
                      reward=reward, measured_rollouts=sum(r['domain']=='overall' for r in rows),
                      no_additional_training_started=True)
    (out/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    return provenance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--window', type=int, default=5)
    args = parser.parse_args()
    result = render(args.training, args.protocol, args.out, args.window)
    print(json.dumps({'output':str(args.out), 'measured_rollouts':result['measured_rollouts']}))
