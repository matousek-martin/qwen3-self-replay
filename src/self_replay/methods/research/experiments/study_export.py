"""Export the verified development report without opening evaluation outcomes."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path


def percent(value):
    return None if value is None else value * 100


def export(report_path, output_path):
    raw = report_path.read_bytes()
    report = json.loads(raw)
    fields = ['candidate', 'evaluation_state', 'scope', 'evaluation_job',
              'base_accuracy_percent', 'candidate_accuracy_percent',
              'accuracy_delta_pp', 'accuracy_ci_lower_pp', 'accuracy_ci_upper_pp',
              'thinking_token_reduction_percent', 'total_output_token_reduction_percent',
              'total_output_reduction_ci_lower_percent', 'total_output_reduction_ci_upper_percent',
              'strict_preservation', 'passes_exploratory_point_screen']
    domains = ('bfcl', 'ifeval', 'mbpp', 'svamp')
    for domain in domains:
        fields += [domain + '_accuracy_delta_pp', domain + '_accuracy_ci_lower_pp',
                   domain + '_accuracy_ci_upper_pp', domain + '_thinking_token_reduction_percent']
    fields += ['warnings']
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    completed = 0
    for candidate in report['candidates']:
        row = {'candidate': candidate['name'], 'evaluation_state': candidate['evaluation_state'],
               'scope': 'exploratory development',
               'evaluation_job': candidate.get('evaluation_provenance', {}).get('job'),
               'warnings': ' | '.join(candidate.get('warnings', []))}
        if candidate['evaluation_state'] == 'complete':
            completed += 1
            analysis = candidate['analysis']
            macro = analysis['macro']
            accuracy_ci = macro['intervals']['accuracy_delta']
            output_ci = macro['intervals']['token_reduction']
            row.update(base_accuracy_percent=percent(macro['base_accuracy']),
                       candidate_accuracy_percent=percent(macro['candidate_accuracy']),
                       accuracy_delta_pp=percent(macro['accuracy_delta']),
                       accuracy_ci_lower_pp=percent(accuracy_ci[0]),
                       accuracy_ci_upper_pp=percent(accuracy_ci[1]),
                       thinking_token_reduction_percent=percent(macro['thinking_tokens']['reduction']),
                       total_output_token_reduction_percent=percent(macro['token_reduction']),
                       total_output_reduction_ci_lower_percent=percent(output_ci[0]),
                       total_output_reduction_ci_upper_percent=percent(output_ci[1]),
                       strict_preservation=macro['strict_preservation']['classification'],
                       passes_exploratory_point_screen=macro['passes_exploratory_point_screen'])
            for domain in domains:
                item = analysis['domains'][domain]
                row[domain + '_accuracy_delta_pp'] = percent(item['accuracy_delta'])
                interval = item['intervals']['accuracy_delta']
                row[domain + '_accuracy_ci_lower_pp'] = percent(interval[0])
                row[domain + '_accuracy_ci_upper_pp'] = percent(interval[1])
                row[domain + '_thinking_token_reduction_percent'] = percent(item['thinking_tokens']['reduction'])
        writer.writerow(row)
    if completed != report['completed_development_count']:
        raise ValueError('Complete candidate count differs from report summary')
    if len(report['candidates']) != report['planned_candidate_count']:
        raise ValueError('Candidate count differs from registered scope')
    body = stream.getvalue().encode('utf-8')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(body)
    metadata = {'generated_at': datetime.now(timezone.utc).isoformat(),
                'source_report': str(report_path.resolve()),
                'source_report_sha256': hashlib.sha256(raw).hexdigest(),
                'csv_sha256': hashlib.sha256(body).hexdigest(),
                'completed': completed, 'planned': report['planned_candidate_count'],
                'scope': report['scope'],
                'notes': ['Positive token reduction means fewer tokens; negative means more.',
                          'Accuracy changes and their intervals are percentage points.',
                          'Intervals are paired 95% question-cluster estimates from the source report.',
                          'Thinking-token reductions are descriptive; no thinking-token interval is claimed.',
                          'Blank metrics mean the comparison is not complete, not a zero effect.',
                          'A passed exploratory point screen does not establish preserved accuracy.']}
    output_path.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return {'csv': str(output_path), 'completed': completed, 'planned': report['planned_candidate_count']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path,
                        default=Path('lab/experiments/study/development-report.json'))
    parser.add_argument('--out', type=Path,
                        default=Path('lab/experiments/study/development-comparisons.csv'))
    args = parser.parse_args()
    print(json.dumps(export(args.report, args.out)))


if __name__ == '__main__':
    main()
