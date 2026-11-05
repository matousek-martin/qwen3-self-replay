import importlib.util
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import canonical_completion, encode_example, generation_seed, sequence_logprob
from score import score_rows, valid_reasoning
from score_compatibility import answer_conditioned_example


class FakeTokenizer:
    eos_token_id = 999

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] and kwargs["add_generation_prompt"]
        return [1, 2, 3]

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, tokens, **kwargs):
        return "assistant\n"


class DataTests(unittest.TestCase):
    def test_completion_mask_and_eos(self):
        result = encode_example(FakeTokenizer(), [], "ab", 6)
        self.assertEqual(result, {"input_ids": [1, 2, 3, 97, 98, 999],
                                  "prompt_length": 3, "completion_tokens": 3})

    def test_no_silent_truncation(self):
        with self.assertRaises(ValueError):
            encode_example(FakeTokenizer(), [], "abc", 6)

    def test_open_thinking_marker_not_duplicated(self):
        self.assertEqual(canonical_completion("<think>\nabc</think>\n42"), "<think>\nabc</think>\n42")
        self.assertEqual(canonical_completion("<think>\nabc</think>\n42", True), "abc</think>\n42")
        self.assertEqual(canonical_completion("abc</think>\n42"), "abc</think>\n42")

    def test_seeds_stable_per_prompt_not_batch(self):
        self.assertEqual(generation_seed(42, "abc", 1), generation_seed(42, "abc", 1))
        self.assertNotEqual(generation_seed(42, "abc", 1), generation_seed(42, "abc", 2))

    def test_scoring_rejects_truncated_success(self):
        prompts = [{"id": "a", "source": "gsm8k", "task": "math", "answer": "42"}]
        records = [{"prompt_id": "a", "sample_id": 0, "completion": "42 is the result</think>\nFinal answer: 42",
                    "truncated": True, "total_tokens": 20}]
        score = score_rows(prompts, records, 1)["sources"]["all"]
        self.assertEqual(score["accuracy"], 0)
        self.assertEqual(score["truncation_rate"], 1)

    def test_missing_samples_and_duplicates_rejected(self):
        prompts = [{"id": "a", "source": "gsm8k", "task": "math", "answer": "42"}]
        records = [{"prompt_id": "a", "sample_id": 0, "completion": "x</think>\nFinal answer: 42",
                    "truncated": False, "total_tokens": 20}]
        with self.assertRaises(ValueError):
            score_rows(prompts, records, 2)
        with self.assertRaises(ValueError):
            score_rows(prompts, records * 2, 2)

    def test_thinking_required(self):
        self.assertFalse(valid_reasoning("Final answer: 42"))
        self.assertFalse(valid_reasoning("<think></think>\nFinal answer: 42"))
        self.assertTrue(valid_reasoning("some reasoning</think>\nFinal answer: 42"))

    def test_compatibility_excludes_original_final_answer(self):
        tok = FakeTokenizer()
        result = answer_conditioned_example(tok, [], "reason</think>\nFinal answer: LEAK", "42", 100)
        prefix = "".join(chr(c) for c in result["input_ids"][3:result["prompt_length"]])
        self.assertEqual(prefix, "reason</think>\n\nFinal answer:")
        self.assertNotIn("LEAK", prefix)
        self.assertEqual(result["completion_tokens"], 3)


TORCH_AVAILABLE = all(importlib.util.find_spec(name) for name in ["torch", "transformers", "peft"])


@unittest.skipUnless(TORCH_AVAILABLE, "CPU torch, transformers, peft required for numerical tests")
class NumericalTests(unittest.TestCase):
    def make_model(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from peft import LoraConfig, get_peft_model
        torch.manual_seed(7)
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                            head_dim=8, max_position_embeddings=64, attention_dropout=0.0)
        model = get_peft_model(Qwen3ForCausalLM(config), LoraConfig(r=2, lora_alpha=4,
                              target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM", lora_dropout=0.0))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.config.use_cache = False
        model.train()
        return model

    def test_chunked_loss_and_adapter_gradients_equal_full_logits(self):
        import torch
        import torch.nn.functional as F
        torch.set_num_threads(1)
        model = self.make_model()
        example = {"input_ids": [1, 2, 3, 4, 5, 6, 7], "prompt_length": 3, "completion_tokens": 4}
        chunked = sequence_logprob(model, example, 2)
        chunked.backward()
        grads = {name: param.grad.clone() for name, param in model.named_parameters() if param.requires_grad}
        model.zero_grad(set_to_none=True)
        ids = torch.tensor([example["input_ids"]])
        logits = model(input_ids=ids, use_cache=False).logits[0, 2:-1]
        expected = -F.cross_entropy(logits.float(), ids[0, 3:], reduction="sum")
        expected.backward()
        torch.testing.assert_close(chunked.detach(), expected.detach(), rtol=1e-5, atol=1e-6)
        for name, param in model.named_parameters():
            if param.requires_grad:
                torch.testing.assert_close(grads[name], param.grad, rtol=1e-4, atol=1e-6)

    def test_dpo_at_initial_reference_has_log_two_loss_and_gradient(self):
        import torch
        import torch.nn.functional as F
        torch.set_num_threads(1)
        model = self.make_model()
        chosen = {"input_ids": [1, 2, 3, 4, 5], "prompt_length": 3, "completion_tokens": 2}
        rejected = {"input_ids": [1, 2, 3, 6, 7, 8], "prompt_length": 3, "completion_tokens": 3}
        model.eval()
        with torch.no_grad(), model.disable_adapter():
            ref_margin = sequence_logprob(model, chosen, 2) - sequence_logprob(model, rejected, 2)
        model.train()
        margin = sequence_logprob(model, chosen, 2) - sequence_logprob(model, rejected, 2) - ref_margin
        loss = -F.logsigmoid(0.1 * margin)
        self.assertAlmostEqual(loss.item(), math.log(2), places=5)
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
