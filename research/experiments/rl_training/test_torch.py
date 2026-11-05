"""Run in the GPU dependency environment; tiny random model needs no downloads."""

import unittest

try:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM
except ImportError:
    torch = None

from .train import token_logprobs


@unittest.skipIf(torch is None, "Requires torch, transformers and peft")
class TensorTests(unittest.TestCase):
    def test_chunked_projection_matches_dense_values_and_gradients(self):
        torch.manual_seed(1)
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                            head_dim=8, max_position_embeddings=128, attention_dropout=0)
        base = Qwen3ForCausalLM(config)
        model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, lora_dropout=0,
                                              target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        prompt = torch.tensor([[1, 5, 9]])
        completion = torch.tensor([[10, 11, 12, 2]])
        temperature = 0.6
        tokens = torch.cat([prompt, completion], dim=1)
        logits = model(tokens, use_cache=False).logits[:, prompt.shape[1] - 1:-1, :].float() / temperature
        expected = logits.log_softmax(-1).gather(-1, completion.unsqueeze(-1)).squeeze(0).squeeze(-1)
        (-expected.mean()).backward()
        dense_grad = {name: p.grad.clone() for name, p in model.named_parameters() if p.requires_grad}
        model.zero_grad(set_to_none=True)
        actual, entropy = token_logprobs(model, prompt, completion, chunk_size=2, temperature=temperature)
        (-actual.mean()).backward()
        torch.testing.assert_close(actual, expected)
        self.assertEqual(len(entropy), len(completion[0]))
        self.assertTrue(torch.isfinite(entropy).all())
        self.assertTrue(any(gradient.abs().sum() > 0 for gradient in dense_grad.values()))
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                torch.testing.assert_close(parameter.grad, dense_grad[name], rtol=2e-4, atol=2e-6)

    def test_single_update_ratio_has_policy_gradient(self):
        lp = torch.tensor([-0.5, -1.0], requires_grad=True)
        loss = -((lp - lp.detach()).exp() * 0.75).sum() / 8
        loss.backward()
        torch.testing.assert_close(lp.grad, torch.full_like(lp, -0.75 / 8))


if __name__ == "__main__":
    unittest.main()
