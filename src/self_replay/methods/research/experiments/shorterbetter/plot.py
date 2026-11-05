"""Plot measured internal validation checkpoints; no synthetic curve generation."""
import argparse
import json
from pathlib import Path


def render(report, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = report['checkpoints']
    if not rows:
        raise ValueError('No measured checkpoints to plot')
    x = [r['rollout_batches'] for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True, constrained_layout=True)
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.spines[['top', 'right']].set_visible(False)
    ax = axes[0]
    ax.axhline(0, color='black', linewidth=1, linestyle='--', label='Počáteční model')
    ax.plot(x, [r['accuracy_delta_pp'] for r in rows], 'o-', color='#2457b2', label='Správná odpověď')
    ax.fill_between(x, [r['paired_question_accuracy_ci95_pp'][0] for r in rows],
                    [r['paired_question_accuracy_ci95_pp'][1] for r in rows], color='#2457b2', alpha=0.15)
    ax.plot(x, [r['completed_accuracy_delta_pp'] for r in rows], 's--', color='#bc600b',
            label='Správná, dokončená odpověď s reasoningem')
    ax.fill_between(x, [r['completed_accuracy_ci95_pp'][0] for r in rows],
                    [r['completed_accuracy_ci95_pp'][1] for r in rows], color='#bc600b', alpha=0.1)
    ax.set_ylabel('Změna přesnosti (p. b.)')
    ax.legend(fontsize=8, loc='best')
    ax.set_title('ShorterBetter: průběh interní validace\nPásma: párové 95% intervaly po otázkách; průběžné měření, nikoli finální důkaz')
    axes[1].plot(x, [r['mean_total_tokens'] for r in rows], 'o-', color='#2457b2')
    axes[1].axhline(rows[0]['mean_total_tokens'], color='black', linestyle='--', linewidth=1)
    axes[1].set_ylabel('Průměr celkových tokenů')
    axes[2].plot(x, [100 * r['truncation_rate'] for r in rows], 'o-', color='#bf3737', label='Uříznuté limitem')
    axes[2].plot(x, [100 * r['invalid_thinking_rate'] for r in rows], 's--', color='#9855a1', label='Neplatný formát reasoningu')
    axes[2].set_ylabel('Podíl odpovědí (%)')
    axes[2].set_xlabel('Dokončené rollout iterace (64 otázek × 8 odpovědí)')
    axes[2].set_ylim(bottom=0)
    axes[2].legend(fontsize=8)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + '.tmp.png')
    fig.savefig(temporary, dpi=150, format='png')
    plt.close(fig)
    temporary.replace(destination)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report')
    parser.add_argument('destination')
    args = parser.parse_args()
    render(json.loads(Path(args.report).read_text()), args.destination)
