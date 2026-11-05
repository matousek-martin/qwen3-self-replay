"""CPU checks against frozen upstream trainer code, not just our own formula."""
import ast
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

from .config import training_arguments, assert_launch_ready

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / 'lab/experiments/shorterbetter-preparation-20261026/sources'


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads((ROOT / 'studies/shorterbetter/protocol.json').read_text())

    def test_rollout_exposure_is_not_confused_with_optimizer_steps(self):
        settings = training_arguments(self.protocol, 'shorterbetter', 17, '/tmp/sb-test')
        self.assertEqual(settings['max_steps'], 1200)
        self.assertEqual(settings['generation_batch_size'], 512)
        self.assertEqual(settings['gradient_accumulation_steps'], 128)
        self.assertEqual(settings['max_steps'] * settings['gradient_accumulation_steps'], 153600)
        self.assertEqual(settings['eval_steps'], 100)

    def test_control_changes_reward_only(self):
        a = training_arguments(self.protocol, 'shorterbetter', 29, '/tmp/a')
        b = training_arguments(self.protocol, 'correctness-control', 29, '/tmp/b')
        a.pop('output_dir'); b.pop('output_dir')
        self.assertEqual(a, b)

    def test_unverified_protocol_cannot_launch(self):
        with self.assertRaises(RuntimeError):
            assert_launch_ready(self.protocol, {})

    def test_frozen_actual_grpo_config_accepts_fields(self):
        if importlib.util.find_spec('torch') is None:
            self.skipTest('Use documented CPU torch environment')
        spec = importlib.util.spec_from_file_location('sb_pinned_grpo_config', SOURCES / 'grpo_config.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        settings = training_arguments(self.protocol, 'shorterbetter', 17, '/tmp/sb-test')
        settings.update(use_cpu=True, bf16=False)
        args = module.GRPOConfig(**settings)
        self.assertEqual(args.steps_per_generation, 512)
        self.assertEqual(args.steps_per_generation // args.gradient_accumulation_steps, 4)

    def test_actual_trl_loss_and_gradients_match_scalar_reference(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Use documented CPU torch environment')
        tree = ast.parse((SOURCES / 'grpo_trainer.py').read_text())
        function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_compute_loss')
        scope = {'torch': torch, 'nanmin': torch.min, 'nanmax': torch.max}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'pinned-trl', 'exec'), scope)
        for kl_beta in (0.0, 0.001):
            logits = torch.tensor([[-0.1, -0.9, -0.4], [-0.2, -0.4, -0.7]],
                                  dtype=torch.float64, requires_grad=True)
            old = torch.tensor([[-0.5, -0.4, -0.6], [-0.1, -0.8, -0.1]], dtype=torch.float64)
            ref = old - 0.1
            mask = torch.tensor([[1., 1., 0.], [1., 1., 1.]], dtype=torch.float64)
            advantages = torch.tensor([0.7, -0.4], dtype=torch.float64)
            trainer = SimpleNamespace(
                _get_per_token_logps_and_entropies=lambda *a, **k: (logits, torch.zeros_like(logits)),
                top_entropy_quantile=1.0, beta=kl_beta, importance_sampling_level='token',
                epsilon_low=0.2, epsilon_high=0.2, args=SimpleNamespace(delta=None),
                use_vllm=True, vllm_importance_sampling_correction=False,
                loss_type='grpo', current_gradient_accumulation_steps=128,
                model=SimpleNamespace(training=True),
                _metrics={'train': defaultdict(list)},
                accelerator=SimpleNamespace(gather=lambda x: x))
            inputs = {'prompt_ids': torch.ones(2, 1, dtype=torch.long),
                      'prompt_mask': torch.ones(2, 1), 'completion_ids': torch.ones(2, 3, dtype=torch.long),
                      'completion_mask': mask, 'advantages': advantages,
                      'old_per_token_logps': old, 'ref_per_token_logps': ref}
            actual = scope['_compute_loss'](trainer, None, inputs)
            terms = []
            for i, length in enumerate((2, 3)):
                sample = []
                for j in range(length):
                    ratio = (logits[i, j] - old[i, j]).exp()
                    policy = -torch.minimum(ratio * advantages[i],
                                            ratio.clamp(0.8, 1.2) * advantages[i])
                    delta = ref[i, j] - logits[i, j]
                    sample.append(policy + kl_beta * (delta.exp() - delta - 1))
                terms.append(torch.stack(sample).mean())
            expected = torch.stack(terms).mean() / 128
            torch.testing.assert_close(actual, expected)
            ga = torch.autograd.grad(actual, logits, retain_graph=True)[0]
            ge = torch.autograd.grad(expected, logits)[0]
            torch.testing.assert_close(ga, ge)
            self.assertEqual(ga[0, 2], 0)


if __name__ == '__main__':
    unittest.main()
