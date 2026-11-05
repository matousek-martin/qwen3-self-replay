"""Plot development evidence only; never reads confirmation outputs."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


ORDER = ['sft-random', 'sft-short', 'sft-compatible', 'sft-dpo-matched',
         'dpo-16', 'dpo-32', 'dpo-48', 'dpo', 'grpo', 'dr-grpo',
         'rl-fixed', 'rl-adaptive', 'rl-difficulty', 'rl-progressive', 'rl-replay']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path,
                        default=Path('lab/experiments/study/development-report.json'))
    parser.add_argument('--out', type=Path,
                        default=Path('lab/experiments/study/development-tradeoffs.png'))
    args = parser.parse_args()
    raw = args.report.read_bytes()
    report = json.loads(raw)
    rows = sorted(report['candidates'],
                  key=lambda c: ORDER.index(c['name']) if c['name'] in ORDER else len(ORDER))
    complete = [c for c in rows if c['evaluation_state'] == 'complete']
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.4 + .39 * len(rows)), sharey=True)
    fig.subplots_adjust(left=.19, right=.97, top=.84, bottom=.17, wspace=.18)
    fig.suptitle('ThinkingCap: less thinking, preserved accuracy?',
                 x=.04, y=.98, ha='left', fontsize=18, weight='bold')
    fig.text(.04, .935,
             f"Development only — {len(complete)}/{len(rows)} comparisons complete",
             fontsize=12, color='#444444')
    fig.text(.04, .905, '512 questions · 2 samples per question · 4 domains',
             fontsize=10, color='#555555')
    thinking, accuracy = axes
    thinking.set_title('Thinking-token reduction (%)', loc='left', fontsize=12, pad=14)
    accuracy.set_title('Accuracy change (percentage points)', loc='left', fontsize=12, pad=14)
    savings, bounds = [0], [-1, 0]
    plotted = []
    for y, candidate in enumerate(rows):
        if candidate['evaluation_state'] != 'complete':
            for axis in axes:
                axis.text(.02, y, candidate['evaluation_state'],
                          transform=axis.get_yaxis_transform(), va='center',
                          color='#888888', fontsize=9)
            continue
        macro = candidate['analysis']['macro']
        reduction = macro['thinking_tokens']['reduction']
        delta = 100 * macro['accuracy_delta']
        low, high = [100 * v for v in macro['intervals']['accuracy_delta']]
        color = '#336b91'
        if reduction is None:
            thinking.text(.02, y, 'undefined: zero baseline tokens',
                          transform=thinking.get_yaxis_transform(), va='center', fontsize=8)
        else:
            reduction *= 100
            savings.append(reduction)
            thinking.barh(y, reduction, height=.48, color=color, alpha=.85)
            thinking.annotate(f'{reduction:+.1f}%', (reduction, y),
                              xytext=(5 if reduction >= 0 else -5, 0),
                              textcoords='offset points', va='center',
                              ha='left' if reduction >= 0 else 'right', fontsize=9)
        accuracy.hlines(y, low, high, color=color, linewidth=1.8)
        accuracy.plot([low, high], [y, y], '|', color=color, markersize=7)
        accuracy.plot(delta, y, 'o', color=color, markersize=5)
        bounds.extend([low, high, delta])
        plotted.append({'candidate': candidate['name'], 'thinking_reduction_percent': reduction,
                        'accuracy_delta_pp': delta, 'accuracy_interval_pp': [low, high]})
    # Leave room for labels on negative bars without colliding with row names.
    thinking.set_xlim(min(savings) - 6, max(5, max(savings)) + 7)
    accuracy.set_xlim(min(bounds) - .7, max(bounds) + .7)
    accuracy.axvline(-1, linestyle='--', color='#b07b25', linewidth=1)
    for axis in axes:
        axis.axvline(0, color='#444444', linewidth=.8)
        axis.set_ylim(len(rows) - .35, -.65)
        axis.set_axisbelow(True)
        axis.grid(axis='x', color='#e6e6e6', linewidth=.7)
        axis.spines[['top', 'right', 'left']].set_visible(False)
        axis.tick_params(axis='y', length=0)
        axis.tick_params(axis='x', labelsize=9)
    thinking.set_yticks(range(len(rows)), [c['name'] for c in rows], fontsize=10)
    accuracy.tick_params(axis='y', labelleft=False)
    fig.text(.04, .105,
             'Accuracy bars: paired 95% question-cluster intervals. Crossing zero does not establish preservation.\n'
             'Dashed line: −1 pp exploratory tolerance, not the final accuracy target.\n'
             'Thinking savings are descriptive point estimates; include incorrect and truncated responses.\n'
             'Registered selection also considers total-output tokens and individual-domain regressions.',
             fontsize=9, color='#444444', va='top', linespacing=1.5)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, facecolor='white')
    fig.savefig(args.out.with_suffix('.svg'), facecolor='white')
    plt.close(fig)
    metadata = {'source_report': str(args.report), 'source_sha256': hashlib.sha256(raw).hexdigest(),
                'source_generated_at': report['generated_at'], 'scope': 'exploratory development only',
                'completed': len(complete), 'planned': len(rows), 'plotted_values': plotted,
                'matplotlib_version': matplotlib.__version__}
    args.out.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'png': str(args.out), 'svg': str(args.out.with_suffix('.svg')),
                      'completed': len(complete), 'planned': len(rows)}))


if __name__ == '__main__':
    main()
