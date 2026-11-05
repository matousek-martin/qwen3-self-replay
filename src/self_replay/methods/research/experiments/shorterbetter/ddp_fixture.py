"""Two-process CPU regression fixture for the real trainer; no target-model training."""
import ast
import json
import os
from pathlib import Path
import sys
import time
import torch
from accelerate import PartialState
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
from trl import GRPOConfig, GRPOTrainer
from .train import append
from .distributed import gather_parts, global_rewards
from .precision import PrecisionGuard, UpdateGuard, inspect_precision
from .control import ScaledControl
from .stopping import DEFAULTS


def main():
    process = PartialState(cpu=True)
    # Accelerate 1.10 labels distributed CPU as cpu:0; torch.load 2.8 cannot
    # restore to that indexed CPU device. Fixture-only normalization; CUDA
    # workers retain their rank-local cuda:N device unchanged.
    process.device = torch.device('cpu')
    out = Path(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else 'four'
    final_step = 1 if mode == 'multidomain-stop' else (8 if mode in ('eight', 'resume') else 4)
    rank_out = out / f'rank-{process.process_index:03d}'
    rank_out.mkdir(parents=True, exist_ok=True)
    vocab = {'[PAD]': 0, '[BOS]': 1, '[EOS]': 2, **{f'w{i}': i for i in range(3, 41)}}
    backend = Tokenizer(WordLevel(vocab, unk_token='w3'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', bos_token='[BOS]', eos_token='[EOS]')
    tokenizer.model_input_names = ['input_ids', 'attention_mask']
    torch.manual_seed(17)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=41, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        eos_token_id=2, pad_token_id=0, bos_token_id=1, attention_dropout=0.0))
    original = torch.cat([p.detach().flatten() for p in model.parameters()])
    if process.is_main_process:
        model.save_pretrained(out / 'initial')
    process.wait_for_everyone()
    tree = ast.parse(Path(__file__).with_name('train.py').read_text())
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == 'StudyTrainer')
    multidomain = mode in ('multidomain','multidomain-stop','multidomain-full')
    fixture_execution = {'multidomain':True,'deterministic_ddp':True} if multidomain else {}
    if mode == 'multidomain-stop':fixture_execution.update(terminal_evaluation_required=True,shutdown_reserve_seconds=1200)
    scope = dict(globals(), out=out, protocol={'execution': fixture_execution,
                 'data': {'cover_all_training_rows': mode == 'multidomain-full'}}, rank_out=rank_out, world=process.num_processes)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), 'actual-study-trainer', 'exec'), scope)
    class FixtureTrainer(scope['StudyTrainer']):
        def _load_optimizer_and_scheduler(self, checkpoint):
            # CPU-only compatibility shim for the pinned Transformers loader;
            # all actual optimizer/scheduler loading remains upstream code.
            from unittest.mock import patch
            original_load = torch.load
            def cpu_load(*args, **kwargs):
                if str(kwargs.get('map_location', '')).startswith('cpu:'):
                    kwargs['map_location'] = 'cpu'
                return original_load(*args, **kwargs)
            with patch.object(torch, 'load', cpu_load):
                return super()._load_optimizer_and_scheduler(checkpoint)

        def _generate_single_turn(self, prompts, images):
            parts, rank = gather_parts(prompts)
            flat = [p for part in parts for p in part]
            counts, completions = {}, []
            for p in flat:
                i = counts.get(p, 0)
                counts[p] = i + 1
                completions.append([4 + i % 8] * (1 + i % 5) + [2])
            start = sum(len(p) for p in parts[:rank])
            return [self.processing_class.encode(p, add_special_tokens=False) for p in prompts], completions[start:start+len(prompts)], None, {}
    def data(split, count):
        if mode == 'multidomain-full' and split == 'train':
            tasks = [d for d, n in zip(('math','coding','tools','instructions'), (4,6,8,10)) for _ in range(n)]
            return Dataset.from_list([{'id':f'{split}-{i}', 'prompt':f'w3 w{i+4}', 'split':split,
                                       'task':task} for i,task in enumerate(tasks)])
        return Dataset.from_list([{'id': f'{split}-{i}', 'prompt': f'w3 w{i+4}', 'split': split,
            **({'task':('math','coding','tools','instructions')[i%4]} if multidomain else {})} for i in range(count)])
    def reward(completion_ids, id, split, trainer_state, task=None, **kwargs):
        correct = [ids[0] % 2 == 0 for ids in completion_ids]
        lengths = [len(ids) for ids in completion_ids]
        rewards, targets = global_rewards(id, correct, lengths, alpha=2., length_beta=.001)
        for i,(identifier, ok, length) in enumerate(zip(id, correct, lengths)):
            append(rank_out / 'rollouts.jsonl', {'optimizer_step': trainer_state.global_step,
                'split': split[0], 'prompt_id': identifier, 'correct': ok, 'total_tokens': length,
                **({'task':task[i]} if task is not None else {}),
                'truncated': False, 'closed_thinking': True})
        return rewards
    config = GRPOConfig(output_dir=str(out), use_cpu=True, bf16=True, use_vllm=False,
        per_device_train_batch_size=1, gradient_accumulation_steps=8//process.num_processes, generation_batch_size=32,
        per_device_eval_batch_size=4, num_generations=8, max_completion_length=8,
        max_prompt_length=4, max_steps=8 if mode == 'stop' else (4 if mode=='multidomain-stop' else final_step), beta=.001, loss_type='grpo', learning_rate=1e-6,
        lr_scheduler_type='constant', eval_strategy='steps', eval_on_start=mode != 'resume', eval_steps=4,
        save_strategy='steps', save_steps=4, report_to=[], disable_tqdm=True, seed=17,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        ddp_find_unused_parameters=False)
    trainer = FixtureTrainer(model=str(out / 'initial'), args=config, processing_class=tokenizer,
        reward_funcs=reward, train_dataset=data('train', 16), eval_dataset=data('development', 4 if multidomain else 2),
        callbacks=[PrecisionGuard(rank_out), UpdateGuard(rank_out)])
    trainer._scaled_native_forward = True
    stopping = {**DEFAULTS, **({'minimum_rollouts': 1, 'stagnation_patience': 1} if mode == 'stop' else {})}
    trainer.add_callback(ScaledControl(out, {'stopping': stopping, 'data': {'monitor_subset_questions': 4 if multidomain else 2},
        'execution': {**fixture_execution,'phase': 'training' if mode == 'stop' else 'benchmark'}},
        {'hard_deadline': time.time()+(300 if mode=='multidomain-stop' else 3600), 'job_name': 'fixture'}, trainer.accelerator))
    trainer.train(resume_from_checkpoint=str(out/'checkpoint-4') if mode == 'resume' else None)
    inspect_precision(trainer.model, trainer.optimizer, True)
    assert trainer.state.global_step == final_step
    if mode == 'stop':
        process.wait_for_everyone()
        assert json.loads((out/'decision.json').read_text())['stop'] is True
    weights = torch.cat([p.detach().flatten() for p in trainer.model.parameters()])
    gathered = [torch.empty_like(weights) for _ in range(process.num_processes)]
    torch.distributed.all_gather(gathered, weights)
    for item in gathered:
        torch.testing.assert_close(weights, item, rtol=0, atol=0)
    changed = float((weights != original).float().mean())
    assert changed > .1, changed
    reloaded = Qwen3ForCausalLM.from_pretrained(out/f'checkpoint-{final_step}')
    for name, p in trainer.model.named_parameters():
        torch.testing.assert_close(p.detach().cpu(), dict(reloaded.named_parameters())[name], rtol=0, atol=0)
    if process.is_main_process:
        curve=json.loads((out / 'learning-curve.json').read_text())
        assert curve['checkpoints'][-1]['questions'] == (4 if multidomain else 2)
        if multidomain:
            assert set(curve['domains'])=={'math','coding','tools','instructions'}
            rows=[json.loads(line) for rank in range(process.num_processes)
                  for line in (out/f'rank-{rank:03d}'/'rollouts.jsonl').read_text().splitlines()]
            from collections import Counter
            train_rows=[r for r in rows if r['split']=='train']
            assert Counter(r['task'] for r in train_rows)=={d:8 for d in curve['domains']}
        saved_step=1 if mode=='multidomain-stop' else 4
        for rank in range(process.num_processes):
            assert (out / f'checkpoint-{saved_step}/rng_state_{rank}.pth').exists()
        if mode=='multidomain-stop':
            assert [r['optimizer_step'] for r in curve['checkpoints']]==[0,1]
            assert json.loads((out/'decision.json').read_text())['reason']=='deadline_or_stop_request'
        (out / 'ddp-result.json').write_text(json.dumps({'status': 'passed', 'world_size': process.num_processes,
            'rank_weights_bit_identical': True, 'changed_fraction': changed,
            'optimizer_steps': final_step, 'checkpoint_reload_identical': True, 'mode': mode,
            'scope': 'Synthetic tiny CPU fixture; CUDA/vLLM not tested'})+'\n')
    process.wait_for_everyone()
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
