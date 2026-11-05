"""Exercise the actual study subclass through baseline, optimizer and checkpoint on CPU."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from .train import append
from .precision import PrecisionGuard, inspect_precision


class IntegrationTests(unittest.TestCase):
    def test_split_run_at_rollout_boundary_preserves_weights_and_prompts(self):
        """Pinned TRL/Transformers restore Adam, RNG, sampler and four-step rollout boundaries."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tree = ast.parse(Path(__file__).with_name('train.py').read_text())
            cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == 'StudyTrainer')
            vocab = {'[PAD]': 0, '[BOS]': 1, '[EOS]': 2, **{f'w{i}': i for i in range(3, 41)}}
            backend = Tokenizer(WordLevel(vocab, unk_token='w3'))
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', bos_token='[BOS]', eos_token='[EOS]')
            tokenizer.model_input_names = ['input_ids', 'attention_mask']
            torch.manual_seed(17)
            model = Qwen3ForCausalLM(Qwen3Config(vocab_size=41, hidden_size=32, intermediate_size=48,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                eos_token_id=2, pad_token_id=0, bos_token_id=1, attention_dropout=0.0))
            model.save_pretrained(root / 'initial')
            data = Dataset.from_list([{'prompt': f'w3 w{i}'} for i in range(4, 36)])
            observations = {}
            class StopAtFour(TrainerCallback):
                def on_step_end(self, args, state, control, **kwargs):
                    if state.global_step == 4:
                        control.should_training_stop = True
                        control.should_save = True
            def run(label, resume=None, stop=False):
                out = root / ('split' if label != 'whole' else 'whole')
                out.mkdir(exist_ok=True)
                scope = dict(globals(), out=out, protocol={'execution': {}})
                exec(compile(ast.Module(body=[cls], type_ignores=[]), 'actual-study-trainer', 'exec'), scope)
                observations[label] = []
                def reward(prompts, completions, trainer_state, **kwargs):
                    observations[label].append((trainer_state.global_step, prompts, completions))
                    return [float(i % 8) for i in range(len(completions))]
                config = GRPOConfig(output_dir=str(out), use_cpu=True, bf16=True, use_vllm=False,
                    per_device_train_batch_size=1, gradient_accumulation_steps=8, generation_batch_size=32,
                    num_generations=8, max_completion_length=8, max_prompt_length=4, max_steps=8,
                    beta=.001, loss_type='grpo', learning_rate=1e-6, lr_scheduler_type='constant',
                    eval_strategy='no', save_strategy='steps', save_steps=4, logging_steps=4,
                    report_to=[], disable_tqdm=True, seed=17)
                trainer = scope['StudyTrainer'](model=str(root / 'initial'), args=config,
                    processing_class=tokenizer, reward_funcs=reward, train_dataset=data,
                    callbacks=[PrecisionGuard(out)] + ([StopAtFour()] if stop else []))
                trainer.train(resume_from_checkpoint=str(resume) if resume else None)
                self.assertEqual(inspect_precision(trainer.model, trainer.optimizer, True)['status'], 'passed')
                return {n: p.detach().clone() for n, p in trainer.model.named_parameters()}
            whole = run('whole')
            run('first', stop=True)
            resumed = run('second', resume=root / 'split/checkpoint-4')
            self.assertEqual(observations['whole'], observations['first'] + observations['second'])
            for name in whole:
                torch.testing.assert_close(whole[name], resumed[name], rtol=0, atol=0)

    def test_initial_evaluation_then_optimizer_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            # Compile the real nested subclass, supplying its logging closure.
            tree = ast.parse(Path(__file__).with_name('train.py').read_text())
            cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == 'StudyTrainer')
            scope = dict(globals(), out=out, protocol={'execution': {}})
            exec(compile(ast.Module(body=[cls], type_ignores=[]), 'actual-study-trainer', 'exec'), scope)
            vocab = {'[PAD]': 0, '[BOS]': 1, '[EOS]': 2, **{f'w{i}': i for i in range(3, 41)}}
            backend = Tokenizer(WordLevel(vocab, unk_token='w3'))
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]',
                                                bos_token='[BOS]', eos_token='[EOS]')
            tokenizer.model_input_names = ['input_ids', 'attention_mask']
            torch.manual_seed(17)
            model = Qwen3ForCausalLM(Qwen3Config(vocab_size=41, hidden_size=32,
                intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                num_key_value_heads=2, head_dim=8, eos_token_id=2, pad_token_id=0,
                bos_token_id=1, attention_dropout=0.0))
            model.save_pretrained(out / 'initial')
            model = Qwen3ForCausalLM.from_pretrained(out / 'initial')
            original = {n: p.detach().clone() for n, p in model.named_parameters()}
            observed = []
            def reward(completions, trainer_state, **kwargs):
                observed.append(trainer_state.global_step)
                return [float(i % 8) for i in range(len(completions))]
            config = GRPOConfig(output_dir=str(out), use_cpu=True, bf16=True, use_vllm=False,
                per_device_train_batch_size=1, gradient_accumulation_steps=8,
                generation_batch_size=8, num_generations=8, max_completion_length=8,
                max_prompt_length=4, max_steps=1, beta=.001, loss_type='grpo', learning_rate=1e-6,
                gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
                eval_on_start=True, eval_strategy='steps', eval_steps=1,
                per_device_eval_batch_size=8, save_strategy='steps', save_steps=1,
                logging_steps=1, report_to=[], disable_tqdm=True, seed=17)
            trainer = scope['StudyTrainer'](model=model, args=config, processing_class=tokenizer,
                reward_funcs=reward, train_dataset=Dataset.from_list([{'prompt': f'w3 w{i}'} for i in range(4, 12)]),
                eval_dataset=Dataset.from_list([{'prompt': 'w4 w5'}]), callbacks=[PrecisionGuard(out)])
            self.assertFalse(hasattr(trainer, 'current_gradient_accumulation_steps'))
            trainer.train()
            self.assertEqual(trainer.state.global_step, 1)
            self.assertEqual(observed[0], 0)
            self.assertEqual(observed[-1], 1)
            self.assertTrue(any(not torch.equal(p, original[n]) for n, p in model.named_parameters()))
            changed = sum(int((p != original[n]).sum()) for n, p in model.named_parameters())
            total = sum(p.numel() for p in model.parameters())
            self.assertGreater(changed / total, .1)
            self.assertEqual(inspect_precision(model, trainer.optimizer, True)['status'], 'passed')
            print('BF16-autocast/FP32-parameter GRPO changed_fraction:', changed / total)
            self.assertEqual(json.loads((out / 'precision-check.json').read_text())['optimizer_moment_dtype'], 'float32')
            for filename in ('optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json'):
                self.assertTrue((out / 'checkpoint-1' / filename).is_file(), filename)
            loaded = Qwen3ForCausalLM.from_pretrained(out / 'checkpoint-1')
            for n, p in model.named_parameters():
                torch.testing.assert_close(p, dict(loaded.named_parameters())[n])


if __name__ == '__main__':
    unittest.main()
