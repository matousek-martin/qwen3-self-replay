import copy
import itertools
import random
import unittest
from unittest.mock import Mock

from .objectives import random_streams
from .replay_selection import admit_replay, select_replay
from .train import make_parser, verifier, verify_training_device


def sample(length=8, correct=True, name='s', **changes):
    return dict({'ids': [length] * length, 'total_tokens': length, 'sample_id': name,
                 'completion': 'Compute one plus one.</think>\nFinal answer: ' + ('2' if correct else '3'),
                 'correct': correct, 'reasoning_valid': True, 'truncated': False}, **changes)


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.check = staticmethod(verifier())

    def select(self, group, rule='all_correct'):
        return select_replay(group, '2', 'math', self.check, rule)

    def test_exhaustive_success_patterns(self):
        for flags in itertools.product((False, True), repeat=4):
            group = [sample(n, ok, str(n)) for n, ok in zip((3, 7, 4, 6), flags)]
            for rule in ('all_correct', 'verified_correct'):
                result = self.select(group, rule)
                admitted = all(flags) if rule == 'all_correct' else any(flags)
                expected = min((g for g, ok in zip(group, flags) if ok),
                               key=lambda g: g['total_tokens']) if admitted else None
                self.assertEqual(result, expected)

    def test_flags_are_not_authority(self):
        forged = sample(correct=False)
        forged['correct'] = True
        self.assertIsNone(self.select([forged], 'verified_correct'))
        valid = sample(correct=True)
        valid['correct'] = False
        valid['reasoning_valid'] = False
        self.assertEqual(self.select([valid]), valid)

    def test_truncation_and_malformed_cannot_enter(self):
        invalid = [sample(truncated=True), sample(completion='Final answer: 2'),
                   sample(completion='<think></think>Final answer: 2'),
                   sample(completion='x</think>y</think>Final answer: 2')]
        good = sample(20)
        for bad in invalid:
            self.assertIsNone(self.select([bad]))
            self.assertIsNone(self.select([bad, good]))
            self.assertIs(self.select([bad, good], 'verified_correct'), good)
        missing = sample(); del missing['truncated']
        self.assertIsNone(self.select([missing], 'verified_correct'))

    def test_ties_preserve_generation_order_and_no_mutation(self):
        group = [sample(name='first'), sample(name='second')]
        before = copy.deepcopy(group)
        self.assertIs(self.select(group), group[0])
        self.assertEqual(group, before)

    def test_empty_invalid_rule_and_lengths(self):
        self.assertIsNone(self.select([]))
        with self.assertRaises(ValueError): self.select([], 'invalid')
        for bad in (sample(0), sample(total_tokens=99)):
            with self.assertRaises(ValueError): self.select([bad])

    def test_multiple_choice_verifier(self):
        good = sample(completion='Consider the options.</think>\nFinal answer: B')
        wrong = sample(3, completion='Consider the options.</think>\nFinal answer: A')
        result = select_replay([wrong, good], 'B', 'multiple_choice', self.check, 'verified_correct')
        self.assertIs(result, good)

    def test_fifo_shortening_ties_eviction_and_copy(self):
        buffer = {}
        first = sample(10)
        self.assertEqual(admit_replay(buffer, 'a', [1], first, 2)['action'], 'inserted')
        first['ids'][0] = 99
        self.assertEqual(buffer['a'][1][0], 10)
        admit_replay(buffer, 'b', [2], sample(), 2)
        event = admit_replay(buffer, 'a', [1], sample(5), 2)
        self.assertEqual(event['action'], 'shortened')
        self.assertEqual(list(buffer), ['a', 'b'])
        self.assertEqual(admit_replay(buffer, 'a', [9], sample(5), 2)['action'], 'retained')
        self.assertEqual(buffer['a'][0], [1])
        self.assertEqual(admit_replay(buffer, 'c', [3], sample(), 2)['evicted_prompt_id'], 'a')
        before = copy.deepcopy(buffer)
        self.assertEqual(admit_replay(buffer, 'x', [4], None, 2)['action'], 'rejected')
        self.assertEqual(buffer, before)
        with self.assertRaises(ValueError): admit_replay({}, 'a', [1], first, 0)

    def test_historical_buffer_and_sampling_regression(self):
        old, new = {}, {}
        data_rng = random.Random(918)
        _, old_rng = random_streams(17)
        _, new_rng = random_streams(17)
        for step in range(300):
            prompt = str(data_rng.randrange(5))
            group = [sample(data_rng.randrange(2, 12), data_rng.random() < .8, str(i)) for i in range(4)]
            # Literal historical admission, replacement, eviction and sampling.
            if all(g['correct'] for g in group):
                shortest = min(group, key=lambda g: g['total_tokens'])
                previous = old.get(prompt)
                if previous is None or len(shortest['ids']) < len(previous[1]):
                    old[prompt] = ([1], shortest['ids'])
                if len(old) > 3: del old[next(iter(old))]
            admit_replay(new, prompt, [1], self.select(group), 3)
            self.assertEqual(list(old.items()), list(new.items()))
            if old:
                self.assertEqual(old[old_rng.choice(list(old))], new[new_rng.choice(list(new))])

    def test_different_admission_does_not_change_prompt_rng(self):
        rng_a, replay_a = random_streams(43)
        rng_b, replay_b = random_streams(43)
        order_a, order_b = list(range(20)), list(range(20))
        for _ in range(8):
            rng_a.shuffle(order_a); rng_b.shuffle(order_b)
            self.assertEqual(order_a, order_b)
            replay_a.choice([1])
            for _ in range(50): replay_b.choice([1, 2, 3])

    def test_default_and_explicit_cli(self):
        args = make_parser().parse_args(['--data', 'data', '--out', 'out'])
        self.assertEqual(args.replay_admission, 'all_correct')
        args = make_parser().parse_args(['--data', 'data', '--out', 'out', '--replay-admission', 'verified_correct'])
        self.assertEqual(args.replay_admission, 'verified_correct')

    def test_worker_device_guard(self):
        torch = Mock()
        torch.cuda.is_available.return_value = True
        torch.cuda.is_bf16_supported.return_value = True
        torch.cuda.get_device_name.return_value = 'NVIDIA GeForce RTX 4090'
        self.assertEqual(verify_training_device(torch, 'cuda'), 'NVIDIA GeForce RTX 4090')
        for name in ('NVIDIA L4', 'NVIDIA RTX 4090', 'NVIDIA GeForce RTX 3090'):
            torch.cuda.get_device_name.return_value = name
            with self.assertRaises(RuntimeError): verify_training_device(torch, 'cuda')
        with self.assertRaises(RuntimeError): verify_training_device(torch, 'cpu')
        torch.cuda.is_available.return_value = False
        with self.assertRaises(RuntimeError): verify_training_device(torch, 'cuda')


if __name__ == '__main__':
    unittest.main()
