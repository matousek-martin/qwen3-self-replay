"""Compare values AND every model gradient against the full Qwen vocabulary head."""
import unittest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from .memory import qwen_logps


class MemoryTests(unittest.TestCase):
    def test_complete_model_matches_dense_logps_entropy_and_gradients(self):
        torch.manual_seed(17)
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=41, hidden_size=32,
            intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=8, attention_dropout=0.0,
            tie_word_embeddings=True))
        model.config._attn_implementation = 'sdpa'
        ids = torch.randint(0, 41, (2, 13))
        mask = torch.ones_like(ids)
        mask[0, :2] = 0
        target = ids[:, -9:]
        for training in (False, True):
            model.train(training)
            if training:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            model.zero_grad(set_to_none=True)
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, -10:-1].float() / .9
            dense = logits.log_softmax(-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
            entropy = -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1)
            # Non-uniform coefficients exercise token ordering and padded positions.
            weights = torch.linspace(-.7, 1.2, 18).reshape(2, 9)
            (dense * weights).sum().backward()
            expected = {n: p.grad.clone() for n, p in model.named_parameters()}
            model.zero_grad(set_to_none=True)
            actual, ent = qwen_logps(model, ids, mask, 9, .9, True, chunk_tokens=4)
            torch.testing.assert_close(actual, dense, atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(ent, entropy, atol=2e-6, rtol=2e-6)
            self.assertFalse(ent.requires_grad)
            (actual * weights).sum().backward()
            for n, p in model.named_parameters():
                torch.testing.assert_close(p.grad, expected[n], atol=2e-5, rtol=2e-4, msg=n)
            with torch.no_grad():
                actual, ent = qwen_logps(model, ids, mask, 9, .9, False, chunk_tokens=4)
                torch.testing.assert_close(actual, dense, atol=2e-6, rtol=2e-6)
                self.assertIsNone(ent)


if __name__ == '__main__':
    unittest.main()
