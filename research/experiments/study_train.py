"""Complete one preregistered RL arm per bounded rental."""
import argparse
import json
from experiments.pilot import DATA, OUT, run, validate, write

ARMS = {
    'grpo': ('correctness', 'grpo'),
    'dr-grpo': ('correctness', 'dr_grpo'),
    'rl-fixed': ('fixed', 'dr_grpo'),
    'rl-adaptive': ('adaptive', 'dr_grpo'),
    'rl-difficulty': ('difficulty', 'dr_grpo'),
    'rl-progressive': ('progressive', 'dr_grpo'),
    'rl-replay': ('replay', 'dr_grpo'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm', choices=ARMS)
    args = parser.parse_args()
    variant, optimizer = ARMS[args.arm]
    target = OUT / 'training' / args.arm
    command = ['-m', 'experiments.rl_training.train', '--data', DATA / 'pilot.jsonl',
               '--out', target, '--variant', variant, '--optimizer', optimizer,
               '--max-steps', 128, '--max-seconds', 0, '--max-new-tokens', 2048,
               '--final-cap', 1024, '--group-size', 4, '--save-every', 32,
               '--references', DATA / 'shared/references.json']
    if variant == 'replay':
        command += ['--replay-reward', 'fixed']
    run(*command)
    result = json.loads((target / 'status.json').read_text())
    if result['status'] != 'completed' or result['completed_groups'] != 128:
        raise RuntimeError('Incomplete matched training budget; retain checkpoint for diagnosis')
    if result['optimizer_updates'] < 1:
        raise RuntimeError('No optimizer updates; not an informative training experiment')
    validate(args.arm, target / 'adapter')
    write(OUT / 'role-result.json', {'role': args.arm, 'result': result,
          'scope': '128-group controlled pilot; training-source validation only'})


if __name__ == '__main__':
    main()
