import unittest
from pipeline import DuplicateIndex, BASE_REVISION, balanced_pilot, screen, select_rows, split_sources, SOURCES


def prompt(identifier='one', text='What is two plus two?', split='train'):
    return {'id': identifier, 'prompt': [{'role': 'user', 'content': text}],
            'answer': '4', 'task': 'math', 'source': 'gsm8k', 'split': split}


def rollout(sample=0, tokens=12, completion='We add two and two. </think>\nFinal answer: 4', **kwargs):
    return dict({'prompt_id': 'one', 'sample_id': sample, 'completion': completion,
                 'total_tokens': tokens, 'truncated': False, 'finish_reason': 'eos',
                 'model_revision': BASE_REVISION, 'generation': {
                     'temperature': 0.6, 'top_p': 0.95, 'top_k': 20,
                     'max_new_tokens': 1024, 'seed': sample}}, **kwargs)


class PipelineTests(unittest.TestCase):
    def test_duplicate_index_near_and_exact(self):
        index = DuplicateIndex()
        index.add('eval', 'Alice has five green apples and Bob gives her two more green apples today.')
        self.assertEqual(index.match('ALICE has five green apples and Bob gives her two more green apples today!')[1], 'exact')
        self.assertEqual(index.match('Alice has five green apples and Bob gives her two more green apples today. How many?')[1], 'lexical_near')
        self.assertIsNone(index.match('A spaceship travels around the moon.'))

    def test_screen_prioritizes_validation_without_answers(self):
        train, val, excluded = screen([prompt('a'), prompt('b', 'Different question')],
                                      [prompt('v', split='val')], [])
        self.assertEqual([r['id'] for r in train], ['b'])
        self.assertEqual(excluded[0]['matched_id'], 'val:v')
        self.assertNotIn('answer', excluded[0])

    def test_benchmark_answer_not_used(self):
        benchmark = {'id': 'eval', 'messages': [{'role': 'user', 'content': 'What is two plus two?'}]}
        self.assertEqual(len(screen([prompt()], [], [benchmark])[0]), 0)

    def test_splits_reproducible_disjoint(self):
        grouped = {source: [prompt(f'{source}-{i}') for i in range(8)] for source in SOURCES}
        a, b = split_sources(grouped, 0, 2)
        self.assertEqual((a, b), split_sources(grouped, 0, 2))
        self.assertFalse({r['id'] for r in a} & {r['id'] for r in b})

    def test_balanced_pilot(self):
        rows = [dict(prompt(f'{s}-{i}'), source=s) for s in ['a', 'b'] for i in range(5)]
        pilot = balanced_pilot(rows, 6, 0)
        self.assertEqual(sum(r['source'] == 'a' for r in pilot), 3)

    def test_selection_regrades_and_excludes_truncation(self):
        rows = [rollout(0, 12), rollout(1, 20, 'We check twice: two and two makes four. </think>\nFinal answer: 4'),
                rollout(2, 6, 'Guess. </think>\nFinal answer: 5'), rollout(3, 5, truncated=True),
                rollout(4, 5, 'Final answer: 4'), rollout(5, 5, ' </think>\nFinal answer: 4')]
        outputs = select_rows([prompt()], rows)
        self.assertEqual(outputs['sft-short.jsonl'][0]['sample_id'], 0)
        self.assertEqual(len(outputs['sft-random.jsonl']), 1)
        self.assertEqual(outputs['dpo.jsonl'][0]['rejected_sample_id'], 1)
        self.assertEqual(sum(r['accepted'] for r in outputs['selection-audit.jsonl']), 2)

    def test_all_wrong_no_fabricated_pairs(self):
        outputs = select_rows([prompt()], [rollout(completion='Wrong. </think>\nFinal answer: 7')])
        self.assertEqual(outputs['dpo.jsonl'], [])
        self.assertEqual(outputs['sft-short.jsonl'], [])

    def test_equal_lengths_no_dpo_pair(self):
        outputs = select_rows([prompt()], [rollout(0), rollout(1, completion='Two twice equals four. </think>\nFinal answer: 4')])
        self.assertEqual(outputs['dpo.jsonl'], [])

    def test_compatibility_prefers_high_score_and_requires_complete_pool(self):
        candidates = [rollout(0, 12), rollout(1, 20, 'We check carefully. </think>\nFinal answer: 4')]
        scores = [{'prompt_id': 'one', 'sample_id': sample, 'total_tokens': length,
                   'model_revision': BASE_REVISION, 'verified_answer_mean_logprob': value}
                  for sample, length, value in [(0, 12, -1.0), (1, 20, -0.1)]]
        result = select_rows([prompt()], candidates, compatibility_scores=scores)
        self.assertEqual(result['sft-compatible.jsonl'][0]['sample_id'], 1)
        self.assertEqual([r['prompt_id'] for r in result['sft-compatible.jsonl']],
                         [r['prompt_id'] for r in result['sft-short.jsonl']])
        with self.assertRaisesRegex(ValueError, 'Incomplete compatibility scores'):
            select_rows([prompt()], candidates, compatibility_scores=scores[:1])

    def test_compatibility_rejects_foreign_or_nonfinite_scores(self):
        score = {'prompt_id': 'one', 'sample_id': 0, 'total_tokens': 12,
                 'model_revision': BASE_REVISION, 'verified_answer_mean_logprob': -1.0}
        for changed in [dict(score, sample_id=2), dict(score, model_revision='wrong'),
                        dict(score, verified_answer_mean_logprob=float('nan'))]:
            with self.assertRaises(ValueError):
                select_rows([prompt()], [rollout()], compatibility_scores=[changed])

    def test_validation_and_bad_provenance_fail(self):
        with self.assertRaises(ValueError):
            select_rows([prompt(split='val')], [rollout()])
        with self.assertRaises(ValueError):
            select_rows([prompt()], [rollout(model_revision='main')])
        with self.assertRaises(ValueError):
            select_rows([prompt()], [rollout(), rollout()])
        with self.assertRaises(ValueError):
            select_rows([prompt()], [rollout(truncated=None)])


if __name__ == '__main__':
    unittest.main()
