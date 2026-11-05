from types import SimpleNamespace
import unittest
from .sampling import independent_parameters


class SamplingTests(unittest.TestCase):
    def test_repeated_prompts_get_independent_reproducible_streams(self):
        prompts = ['problem A'] * 8 + ['problem B'] * 8
        original = SimpleNamespace(n=1, seed=17, temperature=.9, top_p=1., max_tokens=6000)
        def draw(phase, step, batch=prompts):
            return independent_parameters(batch, original, seed=17, phase=phase, step=step)
        initial = draw('eval', 0)
        self.assertEqual(len({p.seed for p in initial}), 16)
        self.assertEqual([p.seed for p in initial], [p.seed for p in draw('eval', 20)])
        self.assertEqual([p.seed for p in initial[8:]], [p.seed for p in draw('eval', 0, prompts[8:])])
        self.assertNotEqual([p.seed for p in draw('train', 0)], [p.seed for p in draw('train', 4)])
        self.assertEqual(original.seed, 17)
        for p in initial:
            self.assertEqual((p.n, p.temperature, p.top_p, p.max_tokens), (1, .9, 1., 6000))


if __name__ == '__main__':
    unittest.main()
