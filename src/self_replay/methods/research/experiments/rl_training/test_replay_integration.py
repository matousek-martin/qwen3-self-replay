import itertools
import json
from pathlib import Path
import subprocess
import sys
import unittest

from .adaptive_replay import replay_rewards
from .objectives import rewards, random_streams
from .replay_selection import admit_replay, select_replay
from .test_replay_selection import sample
from .train import group_rewards, make_parser, verifier

ROOT = Path(__file__).resolve().parents[2]


class IntegrationTests(unittest.TestCase):
    def test_dispatch_matches_historical_fixed_for_all_patterns(self):
        args = make_parser().parse_args(['--data','unused','--out','unused','--variant','replay','--replay-reward','fixed'])
        for flags in itertools.product((False,True), repeat=4):
            lengths=[10,100,500,2048]
            self.assertEqual(group_rewards(flags,lengths,args,512), rewards(flags,lengths,'replay',.1,512,'fixed'))
            args.replay_reward='adaptive'
            self.assertEqual(group_rewards(flags,lengths,args,512),replay_rewards(flags,lengths,'adaptive',.1,512))
            args.replay_reward='fixed'

    def test_reward_dispatch_cannot_change_admission_or_sampling(self):
        check=verifier()
        buffers=[{},{}]
        rngs=[random_streams(17)[1],random_streams(17)[1]]
        args=make_parser().parse_args(['--data','unused','--out','unused','--variant','replay'])
        for step,flags in enumerate(itertools.product((False,True), repeat=4)):
            group=[sample(n,ok,str(i)) for i,(n,ok) in enumerate(zip([5,10,7,11],flags))]
            for i,mode in enumerate(('fixed','adaptive')):
                args.replay_reward=mode
                group_rewards(flags,[g['total_tokens'] for g in group],args,512)
                admit_replay(buffers[i],str(step%3),[1],select_replay(group,'2','math',check,'all_correct'),2)
            self.assertEqual(buffers[0],buffers[1])
            if buffers[0]: self.assertEqual(rngs[0].choice(list(buffers[0])),rngs[1].choice(list(buffers[1])))

    def test_cli_default_adaptive_and_rejection(self):
        parser=make_parser()
        self.assertEqual(parser.parse_args(['--data','x','--out','y']).replay_reward,'correctness')
        command=[sys.executable,'-m','experiments.rl_training.train','--data',str(ROOT/'lab/experiments/data/pilot.jsonl'),'--out','/tmp/unused-adaptive','--variant','replay','--replay-reward','adaptive','--dry-run']
        result=subprocess.run(command,text=True,capture_output=True,check=True)
        config=json.loads(result.stdout)
        self.assertEqual(config['replay_admission'],'all_correct')
        self.assertEqual(config['replay_reward_definition'],'existing-group-adaptive-v1')
        command[command.index('--variant')+1]='fixed'
        result=subprocess.run(command,text=True,capture_output=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('--replay-reward adaptive requires --variant replay',result.stderr)
