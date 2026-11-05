"""Prepared full-weight GRPO worker; dry-run by default, never provisions GPUs.

Execution needs a locked protocol, verified readiness evidence, and an existing
shared-controller reservation. Hardware policy is enforced before model load.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
import signal
from pathlib import Path
import time

from .config import training_arguments, assert_launch_ready, load_protocol
from .objective import grouped_rewards
from .prepare import sha
from .upstream import load_verifier

ROOT = Path(__file__).resolve().parents[2]


def append(path, row):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def check_data(path):
    manifest = json.loads((path / 'manifest.json').read_text())
    audit = json.loads((path / 'audit.json').read_text())
    if audit['status'] != 'passed' or audit['manifest_sha256'] != sha(path / 'manifest.json'):
        raise ValueError('Missing or stale data audit')
    # Do not load confirmation examples or benchmark prompts onto training worker.
    for name in ('train.jsonl', 'monitor.jsonl'):
        if sha(path / name) != manifest['files'][name]:
            raise ValueError('Training/validation artifact changed: ' + name)
    if manifest.get('id') in ('shorterbetter-four-domains-v1', 'shorterbetter-four-domains-xlam-v1', 'shorterbetter-four-domains-full-v1'):
        if set(manifest['verifier_sha256']) != {'rewards.py', 'code_worker.py'}:
            raise ValueError('Incomplete registered domain verifiers')
        for name, expected in manifest['verifier_sha256'].items():
            if sha(Path(__file__).with_name('multidomain') / name) != expected:
                raise ValueError('Verifier changed after data audit: ' + name)
    return manifest


def check_dataset_protocol(manifest, protocol):
    if manifest.get('id') != 'shorterbetter-four-domains-full-v1':
        return
    if protocol.get('data', {}).get('cover_all_training_rows') is not True:
        raise ValueError('Full dataset requires cover_all_training_rows; legacy sampler truncates domains')
    if protocol['data'].get('train_questions') != manifest['counts']['train.jsonl']:
        raise ValueError('Registered training count differs from full dataset')


def monitor_rows(path, protocol):
    """Select a registered development panel without rewriting audited data."""
    from .jsonl import read_rows
    rows = read_rows(path)
    count = protocol['data']['monitor_subset_questions']
    selected = protocol['data'].get('monitor_ids')
    if selected is not None:
        if len(selected) != count or len(set(selected)) != count:
            raise ValueError('Invalid registered monitor IDs')
        by_id = {r['id']: r for r in rows}
        if len(by_id) != len(rows) or any(i not in by_id for i in selected):
            raise ValueError('Monitor ID missing or duplicated')
        rows = [by_id[i] for i in selected]
        if any(r['split'] != 'development' for r in rows):
            raise ValueError('Monitoring must use development examples')
        if protocol.get('execution', {}).get('multidomain') and Counter(r['task'] for r in rows) != {
                d: count//4 for d in ('math', 'coding', 'tools', 'instructions')}:
            raise ValueError('Unbalanced registered monitor panel')
        return rows
    if not 1 <= count <= len(rows):
        raise ValueError('Invalid registered monitor subset size')
    return rows[:count]


def finish_successfully(trainer):
    """Avoid the pinned vLLM allocator's known CPython-finalization crash.

    Call only after successful computation and artifact writes; synchronization
    failures must propagate instead of being converted into successful exits.
    """
    import sys
    import torch
    torch.cuda.synchronize()
    trainer.accelerator.wait_for_everyone()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', default=str(ROOT / 'studies/shorterbetter/protocol.json'))
    parser.add_argument('--arm', choices=['shorterbetter', 'correctness-control'], required=True)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--out', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--technical-check', action='store_true',
                        help='Two complete rollout batches for engineering verification, never an efficacy screen')
    parser.add_argument('--startup-check', action='store_true',
                        help='One real optimizer update before long baseline; isolated engineering output, never resumed')
    parser.add_argument('--readiness')
    parser.add_argument('--job-name')
    parser.add_argument('--resume')
    parser.add_argument('--baseline-check', action='store_true',
                        help='Fresh pinned-base evaluation only, isolated from resumed training')
    args = parser.parse_args()
    if args.baseline_check and (args.resume or args.startup_check or args.technical_check):
        raise ValueError('Baseline check must be isolated and fresh')
    protocol = load_protocol(args.protocol)
    data_path = ROOT / protocol['data']['directory']
    manifest = check_data(data_path)
    check_dataset_protocol(manifest, protocol)
    out = Path(args.out).resolve()
    settings = training_arguments(protocol, args.arm, args.seed, out)
    if args.technical_check:
        settings.update(max_steps=8, save_steps=4, eval_steps=4)
    if args.startup_check:
        if args.technical_check or args.resume:
            raise ValueError('Startup check must be a fresh isolated engineering run')
        settings.update(max_steps=1, eval_on_start=False, eval_strategy='no',
                        save_strategy='no', logging_steps=1)
    if not args.execute:
        print(json.dumps({'status': 'prepared_not_executed', 'settings': settings,
                          'data_counts': manifest['counts'],
                          'reward': protocol['arms'][args.arm]}, indent=2))
        return
    if not args.readiness or not args.job_name:
        raise RuntimeError('Execution requires readiness evidence and controller reservation')
    from .precision import require_precision_config, inspect_precision, PrecisionGuard
    require_precision_config(settings)
    evidence = json.loads(Path(args.readiness).read_text())
    if evidence.get('protocol_sha256') != sha(args.protocol):
        raise RuntimeError('Readiness does not match the locked protocol')
    if args.technical_check:
        if evidence.get('data_audit') is not True or evidence.get('reward_parity') is not True:
            raise RuntimeError('Technical check requires verified data and reward evidence')
    else:
        assert_launch_ready(protocol, evidence)
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    scaled = protocol.get('execution', {}).get('scaled', False)
    if world != protocol.get('execution', {}).get('world_size', 1):
        raise RuntimeError('Worker world size differs from registered global batch configuration')
    if world > 1 and not scaled:
        raise RuntimeError('Distributed training requires a scaled registration')
    lease_path = ROOT / 'worker-lease.json'
    if lease_path.is_file():
        job = json.loads(lease_path.read_text())
        if job['job_name'] != args.job_name or job['max_cost_usd'] != protocol['budget']['proposed_total_pilot_cap_usd']:
            raise RuntimeError('Worker lease identity/cap mismatch')
    else:
        ledger = json.loads((ROOT / 'lab/experiments/cloud/ledger.json').read_text())
        job = ledger['jobs'].get(args.job_name)
    expiry = job.get('hard_deadline', 0) if job and scaled else (job or {}).get('deadline', 0)
    if job and scaled and job.get('completion_budget_scope') == 'finish_registered_1200_updates' and not expiry:
        expiry = float('inf')
    if not job or job.get('released_at') or expiry <= time.time():
        raise RuntimeError('Missing active shared-controller reservation')
    import importlib.metadata
    for package, version in [('torch', '2.8.0'), ('transformers', '4.57.1'),
                             ('trl', '0.24.0'), ('vllm', '0.10.2')]:
        actual = importlib.metadata.version(package).split('+')[0]
        if actual != version:
            raise RuntimeError(f'Worker version mismatch: {package} {actual} != {version}')
    import torch
    from experiments.gpu_policy import verify_gpu_names
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if scaled:
        from .hardware import verify_worker
        verify_worker(names, job)
        if job['gpu_count'] != world or job['gpu'] != protocol['execution']['gpu']:
            raise RuntimeError('Registered hardware differs from worker lease')
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    else:
        verify_gpu_names('\n'.join(names))
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('bf16 required')
    if args.resume and protocol.get('id', '').startswith('shorterbetter-diagnostic-'):
        raise RuntimeError('Exact resume not yet verified for this initial diagnostic; preserve stopped run')
    from accelerate import PartialState
    process = PartialState()
    process.wait_for_everyone()
    if rank == 0 and out.exists() and any(out.iterdir()) and not args.resume:
        raise RuntimeError('Refusing nonempty output without explicit resume')
    process.wait_for_everyone()
    if args.resume:
        checkpoint = Path(args.resume).resolve()
        if checkpoint.parent != out:
            raise RuntimeError('Cross-run resume/parameter branches need their own registered protocol')
        state = json.loads((checkpoint / 'trainer_state.json').read_text())
        if state['global_step'] % settings['save_steps']:
            raise RuntimeError('Resume only at a saved complete-rollout boundary')
        for name in ('optimizer.pt', 'scheduler.pt', f'rng_state_{rank}.pth' if world > 1 else 'rng_state.pth'):
            if not (checkpoint / name).is_file():
                raise RuntimeError('Incomplete resume checkpoint: ' + name)
        prior = json.loads((out / 'run-provenance.json').read_text())
        bridge = (protocol.get('execution', {}).get('resume_diagnostic') is True
                  and protocol.get('execution', {}).get('phase') == 'benchmark'
                  and protocol.get('technical_resume_parent', {}).get('protocol_sha256') == prior['protocol_sha256']
                  and protocol.get('technical_resume_parent', {}).get('optimizer_step') == state['global_step'])
        continuation = protocol.get('execution', {}).get('full_run_continuation') is True
        if continuation:
            from .multidomain.continuation import validate_lineage
            validate_lineage(protocol, prior, state, ROOT)
            bridge = True
        if ((prior['protocol_sha256'] != sha(args.protocol) and not bridge) or prior['arm'] != args.arm
                or prior['seed'] != args.seed or prior['technical_check'] != args.technical_check):
            raise RuntimeError('Resume lineage mismatch')
        settings['eval_on_start'] = False
    out.mkdir(parents=True, exist_ok=True)
    rank_out = out / f'rank-{rank:03d}' if scaled else out
    rank_out.mkdir(parents=True, exist_ok=True)
    from datasets import Dataset
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer
    sources = ROOT / 'lab/experiments/shorterbetter-preparation-20261026/sources'
    if protocol.get('execution', {}).get('multidomain'):
        from .multidomain.rewards import verify
    else:
        verify = load_verifier(sources)
    # TRL passes model.name_or_path to vLLM without a revision argument: use a
    # pinned local snapshot for policy, reference, tokenizer AND rollout engine.
    model_path = snapshot_download(protocol['model']['repo'], revision=protocol['model']['revision'])
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    reward_settings = protocol['arms'][args.arm]
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def reward(prompts, completions, completion_ids, id, answer, split, trainer_state, task=None, **kwargs):
        if len(set(split)) != 1:
            raise ValueError('Mixed training/validation batch')
        correct, texts = [], []
        for completion, gold in zip(completions, answer):
            text = completion if isinstance(completion, str) else completion[-1]['content']
            texts.append(text)
            correct.append(bool(verify(text, gold)))
        lengths = [len(tokens) for tokens in completion_ids]
        from .distributed import global_rewards
        values, targets = global_rewards(id, correct, lengths, **reward_settings)
        for i, text in enumerate(texts):
            append(rank_out / 'rollouts.jsonl', {
                'optimizer_step': trainer_state.global_step, 'split': split[i],
                'prompt_id': id[i], 'correct': correct[i], 'total_tokens': lengths[i],
                **({'task': task[i]} if task is not None else {}),
                'sol': targets[id[i]], 'reward': values[i],
                'truncated': completion_ids[i][-1] != tokenizer.eos_token_id,
                'closed_thinking': text.count('</think>') == 1 and bool(text.split('</think>')[0].strip()),
                **({'completion': text} if not scaled or split[i] == 'development' else {}),
                'completion_ids_sha256': hashlib.sha256(json.dumps(completion_ids[i]).encode()).hexdigest(),
            })
        return values

    class LogCallback(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            if protocol.get('execution', {}).get('multidomain') and state.is_world_process_zero and state.global_step % 4 == 0:
                append(out / 'rollout-starts.jsonl', {'optimizer_step': state.global_step, 'time': time.time()})

        def on_step_end(self, args, state, control, **kwargs):
            if state.is_world_process_zero:
                append(out / 'steps.jsonl', {'optimizer_step': state.global_step, 'time': time.time()})
            reserve = protocol.get('execution', {}).get('shutdown_reserve_seconds', 600) if scaled else 1800
            if stop_requested or time.time() >= (job.get('hard_deadline') or float('inf')) - reserve:
                control.should_training_stop = True
                control.should_save = True
                append(rank_out / 'budget-stop.jsonl', {'optimizer_step': state.global_step,
                       'time': time.time(), 'reason': 'signal_or_cleanup_window'})
            return control

        def on_evaluate(self, args, state, control, **kwargs):
            if scaled:
                return  # ScaledControl performs one synchronized global report/decision.
            from .monitor import summarize
            from .plot import render
            report = summarize(out / 'rollouts.jsonl')
            (out / 'learning-curve.json').write_text(json.dumps(report, indent=2) + '\n')
            render(report, out / 'learning-curve.png')

        def on_log(self, args, state, control, logs=None, **kwargs):
            append(rank_out / 'metrics.jsonl', {'optimizer_step': state.global_step,
                   'rollout_batches': state.global_step / 4, 'time': time.time(), **(logs or {})})
            for key in ('loss', 'grad_norm'):
                value = (logs or {}).get(key)
                if isinstance(value, (float, int)) and not math.isfinite(value):
                    raise RuntimeError('Nonfinite training diagnostic: ' + key)

    class StudyTrainer(GRPOTrainer):
        def _get_train_sampler(self, dataset=None):
            if protocol.get('execution', {}).get('multidomain'):
                from .multidomain.sampler import BalancedRepeatSampler
                return BalancedRepeatSampler(dataset if dataset is not None else self.train_dataset,
                    mini_repeat_count=self.num_generations,
                    batch_size=self.args.generation_batch_size // self.num_generations,
                    repeat_count=self.num_iterations * self.args.steps_per_generation,
                    seed=self.args.seed,
                    cover_all=protocol.get('data', {}).get('cover_all_training_rows', False))
            return super()._get_train_sampler(dataset)

        def _generate_and_score_completions(self, inputs):
            result = super()._generate_and_score_completions(inputs)
            if protocol.get('execution', {}).get('resume_trace') and self.model.training:
                def cpu(value):
                    return value.detach().cpu() if isinstance(value, torch.Tensor) else value
                torch.save({k: cpu(v) for k, v in result.items()},
                           rank_out/f'buffer-step-{self.state.global_step}.pt')
            return result

        def _prepare_inputs(self, inputs):
            result = super()._prepare_inputs(inputs)
            if protocol.get('execution', {}).get('resume_trace') and self.model.training:
                hashes = {k: hashlib.sha256(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
                          for k, v in result.items() if isinstance(v, torch.Tensor)}
                append(rank_out/'input-trace.jsonl', {'optimizer_step': self.state.global_step,
                       'split': 'train', 'micro_step': self._step % self.args.steps_per_generation, 'hashes': hashes})
            return result

        def training_step(self, model, inputs, *args, **kwargs):
            if protocol.get('execution', {}).get('deterministic_ddp'):
                from torch.nn.parallel import DistributedDataParallel
                from .deterministic_ddp import rank_order_average
                if not isinstance(model, DistributedDataParallel):
                    raise RuntimeError('Registered deterministic averaging requires actual DDP')
                if not getattr(model, '_study_fixed_rank_hook', False):
                    model.register_comm_hook(None, rank_order_average)
                    model._study_fixed_rank_hook = True
                    (rank_out/'ddp-reduction-check.json').write_text(json.dumps({
                        'method': 'gather_then_fp32_sum_rank_order', 'world_size': world,
                        'registered_before_first_backward': True})+'\n')
            loss = super().training_step(model, inputs, *args, **kwargs)
            if protocol.get('execution', {}).get('resume_trace') and self.accelerator.sync_gradients:
                tensors = {n: {'parameter': hashlib.sha256(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
                               'gradient': hashlib.sha256(p.grad.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
                           for n, p in self.model.named_parameters() if p.grad is not None
                           and n in ('model.embed_tokens.weight', 'model.layers.0.self_attn.q_proj.weight')}
                append(rank_out/'gradient-trace.jsonl', {'optimizer_step': self.state.global_step, 'split': 'train', 'tensors': tensors})
            return loss

        def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask,
                logits_to_keep, batch_size=None, compute_entropy=False, **kwargs):
            if self.accelerator.num_processes > 1 or getattr(self, '_scaled_native_forward', False):
                # Call the wrapped model so DDP prepares its reducer. Never
                # unwrap/bypass DDP with the historical low-memory projection.
                return super()._get_per_token_logps_and_entropies(
                    model, input_ids, attention_mask, logits_to_keep,
                    batch_size=1, compute_entropy=compute_entropy, **kwargs)
            from .memory import qwen_logps
            if any(value is not None for value in kwargs.values()):
                raise ValueError('This memory adapter supports text-only inputs')
            raw = self.accelerator.unwrap_model(model)
            with self.accelerator.autocast():
                return qwen_logps(raw, input_ids, attention_mask, logits_to_keep,
                                  self.temperature, compute_entropy)

        def _generate_single_turn(self, prompts, images):
            # TRL colocate sends repeated prompts with n=1. A shared scalar
            # seed would produce duplicated samples and destroy GRPO variance.
            from .sampling import independent_parameters
            phase = 'train' if self.model.training else 'eval'
            original_generate = None
            if self.use_vllm and self.vllm_mode == 'colocate':
                original_generate = self.llm.generate
                def generate(inputs, sampling_params, **kwargs):
                    from .distributed import global_sampling_parameters
                    params = global_sampling_parameters(inputs, sampling_params, seed=self.args.seed,
                                                    phase=phase, step=self.state.global_step)
                    if protocol.get('execution', {}).get('verification_only') and not getattr(self, '_engineering_check', False):
                        # Registered technical fixture: varied short caps ensure
                        # nonzero SOL advantages even when every answer truncates.
                        # Never used in efficacy runs or scientific comparisons.
                        for ordinal, parameter in enumerate(params):
                            parameter.max_tokens = 32 * (1 + ordinal % 8)
                    log_dir = Path(self.args.output_dir) / f'rank-{self.accelerator.process_index:03d}' if scaled else out
                    append(log_dir / 'sampling-seeds.jsonl', {'phase': phase,
                           'optimizer_step': self.state.global_step,
                           'engineering': getattr(self, '_engineering_check', False),
                           'seeds': [p.seed for p in params]})
                    return original_generate(inputs, sampling_params=params, **kwargs)
                self.llm.generate = generate
            started = time.time()
            event = {'optimizer_step': self.state.global_step,
                     'mode': 'train' if self.model.training else 'eval',
                     'engineering': getattr(self, '_engineering_check', False),
                     'sequences': len(prompts), 'started_at': started}
            log_dir = Path(self.args.output_dir) / f'rank-{self.accelerator.process_index:03d}' if getattr(self, '_scaled_native_forward', False) else out
            append(log_dir / 'generation-events.jsonl', {**event, 'status': 'started'})
            try:
                result = super()._generate_single_turn(prompts, images)
                if self.use_vllm and self._last_loaded_step != self.state.global_step:
                    raise RuntimeError('vLLM weight synchronization did not reach the current optimizer step')
                append(log_dir / 'generation-events.jsonl', {**event, 'status': 'finished',
                       'vllm_loaded_step': self._last_loaded_step if self.use_vllm else None,
                       'elapsed_seconds': time.time() - started,
                       'generated_tokens': sum(len(ids) for ids in result[1])})
                return result
            finally:
                if original_generate is not None:
                    self.llm.generate = original_generate

        def evaluate(self, *args, **kwargs):
            # Validation must not advance the RNG stream used by training.
            import random
            import numpy as np
            # Transformers initializes this only on entering its optimizer
            # loop, but TRL's loss already needs it for eval_on_start.
            if not hasattr(self, 'current_gradient_accumulation_steps'):
                self.current_gradient_accumulation_steps = self.args.gradient_accumulation_steps
            py_state, np_state = random.getstate(), np.random.get_state()
            try:
                with torch.random.fork_rng(devices=[self.accelerator.device.index] if self.accelerator.device.type == 'cuda' else []):
                    return super().evaluate(*args, **kwargs)
            finally:
                random.setstate(py_state)
                np.random.set_state(np_state)

    def dataset(name):
        from .jsonl import read_rows
        rows = monitor_rows(data_path / name, protocol) if name == 'monitor.jsonl' else read_rows(data_path / name)
        keys = ('id', 'prompt', 'answer', 'split', 'task') if protocol.get('execution', {}).get('multidomain') else ('id', 'prompt', 'answer', 'split')
        return Dataset.from_list([{k: row[k] for k in keys} for row in rows])

    provenance = {'protocol_sha256': sha(args.protocol), 'data_manifest_sha256': sha(data_path / 'manifest.json'),
                  'gpu_names': names, 'arm': args.arm, 'seed': args.seed, 'settings': settings,
                  'technical_check': args.technical_check,
                  'startup_check': args.startup_check,
                  'job_name': args.job_name, 'trainer_source_sha256': sha(__file__),
                  'source_manifest_sha256': sha(sources / 'manifest.json')}
    if args.resume and bridge and rank == 0:
        (out/'resume-parent-provenance.json').write_text(json.dumps(prior, indent=2)+'\n')
        parent_key = 'continuation_parent' if continuation else 'technical_resume_parent'
        provenance[parent_key] = protocol[parent_key]
        (out/'run-provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    if not args.resume and rank == 0:
        (out / 'run-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    trainer = StudyTrainer(model=model_path, reward_funcs=reward,
                           args=GRPOConfig(**settings), processing_class=tokenizer,
                           train_dataset=dataset('train.jsonl'), eval_dataset=dataset('monitor.jsonl'),
                           callbacks=[PrecisionGuard(rank_out), LogCallback()])
    trainer._scaled_native_forward = scaled and not protocol.get('execution', {}).get('single_gpu_chunked_projection')
    if protocol.get('execution', {}).get('deterministic_training'):
        deterministic = {'enabled': torch.are_deterministic_algorithms_enabled(),
                         'warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
                         'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}
        if not deterministic['enabled'] or deterministic['warn_only']:
            raise RuntimeError('Registered deterministic GPU training is not enabled')
        (rank_out/'determinism-check.json').write_text(json.dumps(deterministic, indent=2)+'\n')
    if scaled:
        from .control import ScaledControl
        from .precision import UpdateGuard
        trainer.add_callback(UpdateGuard(rank_out))
        trainer.add_callback(ScaledControl(out, protocol, job, trainer.accelerator))
    trainer.create_optimizer()
    if protocol.get('execution', {}).get('adam_foreach') is False:
        for group in trainer.optimizer.param_groups:
            group['foreach'] = False
    inspect_precision(trainer.model, trainer.optimizer)
    if (not args.resume or protocol.get('execution', {}).get('resume_diagnostic')
            or protocol.get('execution', {}).get('full_run_continuation')):
        from .gpu_check import check, check_sampling
        with torch.random.fork_rng(devices=[trainer.accelerator.device.index]):
            check(trainer, rank_out)
            check_sampling(trainer, rank_out)
    if args.baseline_check:
        trainer.state.is_world_process_zero = trainer.is_world_process_zero()
        trainer.state.is_local_process_zero = trainer.is_local_process_zero()
        trainer.evaluate()
        finish_successfully(trainer)
    def parameter_fingerprints():
        return {name: hashlib.sha256(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                for name, parameter in trainer.model.named_parameters()}
    initial_fingerprints = parameter_fingerprints() if args.startup_check else None
    trainer.train(resume_from_checkpoint=args.resume)
    if args.startup_check:
        final_fingerprints = parameter_fingerprints()
        changed = [name for name in initial_fingerprints if initial_fingerprints[name] != final_fingerprints[name]]
        (rank_out / 'parameter-update-check.json').write_text(json.dumps({
            'status': 'passed' if changed and trainer.state.global_step == 1 else 'failed',
            'optimizer_steps': trainer.state.global_step, 'changed_parameter_tensors': len(changed),
            'total_parameter_tensors': len(initial_fingerprints), 'changed_names': changed,
            'initial_sha256': initial_fingerprints, 'final_sha256': final_fingerprints}, indent=2) + '\n')
        if not changed or trainer.state.global_step != 1:
            raise RuntimeError('Startup optimizer check did not change model weights')
    trainer.save_model(str(out / 'model'))
    if rank == 0:
        tokenizer.save_pretrained(out / 'model')
    trainer.save_state()
    # Local marker is NOT the shared controller's backup-verified completion.
    if rank == 0:
        (out / 'training-finished.json').write_text(json.dumps({
        'optimizer_steps': trainer.state.global_step,
        'technical_check': args.technical_check,
        'startup_check': args.startup_check,
        'phase': protocol.get('execution', {}).get('phase', 'historical_pilot'),
        'training_completed': not (args.technical_check or args.startup_check) and trainer.state.global_step == settings['max_steps'],
        'backup_verified': False, 'gpu_released': False}) + '\n')
    # This pinned torch/vLLM sleep allocator has repeatedly aborted in CPython
    # finalization (CUDAPluggableAllocator::raw_delete). Only after ALL writes
    # and success checks above have completed, let the OS destroy this single
    # process CUDA context. Exceptions never reach this path and remain failures.
    finish_successfully(trainer)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        import sys
        if '--out' in sys.argv:
            from .distributed import atomic_json
            message = str(exc).lower()
            category = ('out_of_memory' if 'out of memory' in message else
                        'numerical_failure' if any(k in message for k in
                            ('nonfinite', 'float32', 'parameter updates', 'different sampled parameters')) else
                        'worker_error')
            try:
                failure_out = Path(sys.argv[sys.argv.index('--out') + 1])
                atomic_json(failure_out / f"failure-rank-{int(os.environ.get('RANK', '0')):03d}.json",
                            {'category': category, 'exception_type': type(exc).__name__, 'time': time.time()})
            except (OSError, ValueError, IndexError):
                pass  # Never obscure the original failure or expose raw errors in Telegram.
        raise
