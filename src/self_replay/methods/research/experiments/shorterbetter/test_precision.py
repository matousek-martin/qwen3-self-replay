"""Regression for erased low-LR updates and precision checks after resume."""
import copy
import unittest

import torch
from .precision import inspect_precision, require_precision_config, parameter_sample_indices
from .config import training_arguments


class PrecisionTests(unittest.TestCase):
    def test_real_qwen_embedding_sample_indices_do_not_round_out_of_bounds(self):
        size = 151936 * 1024
        old = torch.linspace(0, size - 1, 512, dtype=torch.float32).long()
        self.assertEqual(old[-1].item(), size)  # Actual failure, absent in tiny CPU fixtures.
        indices = parameter_sample_indices(size)
        self.assertEqual(indices.dtype, torch.int64)
        self.assertEqual(indices[0].item(), 0)
        self.assertEqual(indices[-1].item(), size - 1)
        self.assertTrue(bool(((indices >= 0) & (indices < size)).all()))
        self.assertEqual(indices.unique().numel(), 512)

    def test_parameter_sampling_handles_scalar_and_small_tensors(self):
        for size in (1, 2, 128, 512, 513):
            indices = parameter_sample_indices(size)
            self.assertEqual(indices.numel(), min(size, 512))
            self.assertEqual(indices[-1].item(), size - 1)
            self.assertEqual(indices.unique().numel(), min(size, 512))

    def test_small_updates_accumulate_only_with_fp32_parameters(self):
        # Numerical fixture only, not experiment training data or a model score.
        initial = torch.tensor([.02], dtype=torch.bfloat16)
        results = {}
        for dtype in (torch.bfloat16, torch.float32):
            parameter = torch.nn.Parameter(initial.to(dtype).clone())
            optimizer = torch.optim.AdamW([parameter], lr=1e-6, weight_decay=.01, foreach=False)
            for _ in range(80):
                parameter.grad = torch.full_like(parameter, .001)
                optimizer.step()
            results[dtype] = parameter.detach()
        self.assertTrue(torch.equal(results[torch.bfloat16], initial))
        self.assertGreater(float(initial.float() - results[torch.float32]), 7e-5)
        self.assertFalse(torch.equal(results[torch.float32].bfloat16(), initial))

    def test_precision_guard_catches_original_bug_and_corrupted_resume(self):
        model = torch.nn.Linear(2, 2, bias=False).float()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()
        self.assertEqual(inspect_precision(model, optimizer, True)['status'], 'passed')
        saved = copy.deepcopy(optimizer.state_dict())
        resumed = torch.optim.AdamW(model.parameters(), lr=1e-6)
        resumed.load_state_dict(saved)
        self.assertEqual(inspect_precision(model, resumed, True)['status'], 'passed')
        resumed.state[model.weight]['exp_avg'] = resumed.state[model.weight]['exp_avg'].bfloat16()
        with self.assertRaisesRegex(RuntimeError, 'moments'):
            inspect_precision(model, resumed, True)
        model.bfloat16()
        with self.assertRaisesRegex(RuntimeError, 'parameters'):
            inspect_precision(model, optimizer)

    def test_historical_configuration_is_preserved_but_rejected_for_new_execution(self):
        import json
        from pathlib import Path
        protocol = json.loads((Path(__file__).resolve().parents[2] / 'studies/shorterbetter/pilot-protocol.json').read_text())
        historical = training_arguments(protocol, 'shorterbetter', 17, '/tmp/unused')
        self.assertEqual(historical['model_init_kwargs']['dtype'], 'bfloat16')
        with self.assertRaisesRegex(RuntimeError, 'read-only'):
            require_precision_config(historical)
        protocol['training']['parameter_dtype'] = 'float32'
        repaired = training_arguments(protocol, 'shorterbetter', 17, '/tmp/unused')
        require_precision_config(repaired)
        historical['model_init_kwargs']['dtype'] = 'float32'
        self.assertEqual(historical, repaired)  # only the registered precision changes


if __name__ == '__main__':
    unittest.main()
