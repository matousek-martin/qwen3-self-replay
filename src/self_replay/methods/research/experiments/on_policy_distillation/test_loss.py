import copy
import math
import unittest
from types import SimpleNamespace

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from .loss import (assert_tokenizer_alignment, completion_mask, freeze_teacher,
                   full_vocab_forward_kl, qwen_forward_kl)


class Tokenizer:
    def __init__(self):
        self.vocab = {"a": 0, "b": 1, "<eos>": 2}
        self.backend_tokenizer = SimpleNamespace(to_str=lambda: '{"model":"BPE"}')
        self.special_tokens_map = {"eos_token": "<eos>"}
        self.chat_template = "template"

    def get_vocab(self):
        return self.vocab


class LossTests(unittest.TestCase):
    def test_hand_calculated_value_and_gradient(self):
        logits = torch.tensor([[math.log(.6), math.log(.4)]], dtype=torch.double, requires_grad=True)
        teacher = torch.tensor([[math.log(.25), math.log(.75)]], dtype=torch.double, requires_grad=True)
        loss = full_vocab_forward_kl(logits, teacher, torch.tensor([True]))
        self.assertAlmostEqual(loss.item(), .25 * math.log(.25/.6) + .75 * math.log(.75/.4))
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.tensor([[.35, -.35]], dtype=torch.double))
        self.assertIsNone(teacher.grad)

    def test_finite_difference_gradcheck(self):
        torch.manual_seed(17)
        student = torch.randn(2, 3, 5, dtype=torch.double, requires_grad=True)
        teacher = torch.randn_like(student)
        mask = torch.tensor([[False, True, True], [True, False, True]])
        self.assertTrue(torch.autograd.gradcheck(
            lambda x: full_vocab_forward_kl(x, teacher, mask, temperature=.7), (student,)))

    def test_identical_zero_and_extreme_finite(self):
        x = torch.tensor([[1000., -1000., 0.]], requires_grad=True)
        m = torch.tensor([True])
        self.assertEqual(full_vocab_forward_kl(x, x, m).item(), 0)
        loss = full_vocab_forward_kl(x, -x, m)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_mask_excludes_nan_and_gradient(self):
        x = torch.tensor([[[float('nan'), 1.], [0., 1.], [7., 0.]]], requires_grad=True)
        teacher = torch.zeros_like(x)
        mask = torch.tensor([[False, True, False]])
        loss = full_vocab_forward_kl(x, teacher, mask)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        torch.testing.assert_close(x.grad[:, [0, 2]], torch.zeros(1, 2, 2))

    def test_causal_shift_prompt_eos_padding(self):
        # EOS in prompt is not the completion EOS; pad shares EOS ID.
        ids = torch.tensor([[9, 2, 3, 9, 9, 9], [1, 2, 3, 4, 5, 9]])
        attention = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 0]])
        mask = completion_mask(ids, attention, [2, 3], {9})
        torch.testing.assert_close(mask, torch.tensor([[False, True, True, False, False],
                                                      [False, False, True, True, False]]))

    def test_no_synthetic_eos_at_truncation(self):
        ids = torch.tensor([[1, 2, 3]])
        torch.testing.assert_close(completion_mask(ids, torch.ones_like(ids), [1], {9}),
                                   torch.tensor([[True, True]]))

    def test_empty_mask_bad_temperature_shapes_and_nonfinite(self):
        x = torch.ones(1, 2)
        for mask, temp in [(torch.tensor([False]), 1), (torch.tensor([True]), 0),
                           (torch.tensor([True]), float('nan'))]:
            with self.assertRaises(ValueError):
                full_vocab_forward_kl(x, x, mask, temp)
        with self.assertRaises(ValueError):
            full_vocab_forward_kl(x, x[:, :1], torch.tensor([True]))
        with self.assertRaises(ValueError):
            full_vocab_forward_kl(x * float('nan'), x, torch.tensor([True]))

    def test_invalid_prompt_and_left_padding_rejected(self):
        ids = torch.tensor([[1, 2, 3]])
        for attention, prompt in [(torch.tensor([[0, 1, 1]]), [1]),
                                  (torch.tensor([[1, 0, 1]]), [1]),
                                  (torch.ones_like(ids), [0]), (torch.ones_like(ids), [3])]:
            with self.assertRaises(ValueError):
                completion_mask(ids, attention, prompt, {9})

    def test_tokenizer_checks_more_than_size(self):
        student = Tokenizer()
        assert_tokenizer_alignment(student, Tokenizer(), 4, 4)  # padded logit row retained
        changes = [lambda t: t.vocab.update(a=1, b=0),
                   lambda t: setattr(t, 'chat_template', 'different'),
                   lambda t: t.special_tokens_map.update(eos_token='b'),
                   lambda t: setattr(t, 'backend_tokenizer', SimpleNamespace(to_str=lambda: '{}'))]
        for change in changes:
            teacher = Tokenizer()
            change(teacher)
            with self.assertRaises(ValueError):
                assert_tokenizer_alignment(student, teacher, 4, 4)
        with self.assertRaises(ValueError):
            assert_tokenizer_alignment(student, Tokenizer(), 4, 5)


class QwenProjectionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)
        config = Qwen3Config(vocab_size=13, hidden_size=16, intermediate_size=24,
                            num_hidden_layers=1, num_attention_heads=2,
                            num_key_value_heads=1, head_dim=8, attention_dropout=0.0)
        self.student = Qwen3ForCausalLM(config).double().eval()
        self.teacher = freeze_teacher(Qwen3ForCausalLM(config).double())
        self.ids = torch.tensor([[1, 2, 3, 4, 5], [2, 3, 6, 5, 5]])
        self.attention = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]])
        self.mask = completion_mask(self.ids, self.attention, [2, 1], {5})

    def test_chunked_matches_dense_loss_and_all_parameter_gradients(self):
        dense_student = copy.deepcopy(self.student)
        dense = full_vocab_forward_kl(
            dense_student(self.ids, attention_mask=self.attention).logits[:, :-1],
            self.teacher(self.ids, attention_mask=self.attention).logits[:, :-1], self.mask)
        dense.backward()
        for size in [1, 3, 99]:
            self.student.zero_grad(set_to_none=True)
            actual = qwen_forward_kl(self.student, self.teacher, self.ids,
                                     self.attention, self.mask, size)
            torch.testing.assert_close(actual, dense)
            actual.backward()
            for a, b in zip(self.student.parameters(), dense_student.parameters()):
                torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-8)
            self.assertTrue(all(p.grad is None for p in self.teacher.parameters()))

    def test_frozen_teacher_enforced(self):
        self.teacher.train()
        with self.assertRaises(ValueError):
            qwen_forward_kl(self.student, self.teacher, self.ids, self.attention, self.mask)
        self.teacher.eval().requires_grad_(True)
        with self.assertRaises(ValueError):
            qwen_forward_kl(self.student, self.teacher, self.ids, self.attention, self.mask)

    def test_padding_cannot_be_selected(self):
        bad = self.mask.clone()
        bad[1, -1] = True
        with self.assertRaises(ValueError):
            qwen_forward_kl(self.student, self.teacher, self.ids, self.attention, bad)

    def test_forward_scoring_without_grad(self):
        with torch.no_grad():
            loss = qwen_forward_kl(self.student, self.teacher, self.ids, self.attention, self.mask)
        self.assertFalse(loss.requires_grad)
        self.assertTrue(torch.isfinite(loss))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
