"""Resolve paper rollout iterations separately from optimizer steps."""
import json
from pathlib import Path


def training_arguments(protocol, arm, seed, out):
    if arm not in protocol['arms'] or seed not in protocol['seeds']:
        raise ValueError('Unregistered arm/seed')
    t = protocol['training']
    execution = protocol.get('execution', {})
    world = execution.get('world_size', 1)
    if execution.get('multidomain_full_training') or execution.get('multidomain_full_verification'):
        from .multidomain.full_run import validate
        validate(protocol)
    if execution.get('hardware_comparison'):
        from .hardware_benchmark import validate
        validate(protocol)
    if execution.get('accelerated_multidomain_benchmark'):
        if (not execution.get('multidomain') or not execution.get('scaled')
                or execution.get('phase') != 'benchmark' or world != 2
                or execution.get('gpu') != 'NVIDIA A100-SXM4-80GB'
                or t['rollout_batches'] != 2 or t['prompts_per_rollout_batch'] != 64
                or t['generations_per_prompt'] != 8 or t['max_completion_tokens'] != 6000
                or execution.get('single_gpu_chunked_projection')
                or execution.get('shutdown_reserve_seconds') != 1200
                or not execution.get('terminal_evaluation_required')):
            raise ValueError('Unregistered two-A100 multidomain benchmark configuration')
    if 'shutdown_reserve_seconds' in execution and not 600 <= execution['shutdown_reserve_seconds'] <= 3600:
        raise ValueError('Invalid shutdown reserve')
    if execution.get('single_gpu_chunked_projection') and (world != 1 or not execution.get('multidomain')):
        raise ValueError('Chunked projection is scoped to the single-GPU multidomain pilot')
    if type(world) is not int or world not in (1, 2, 4, 8):
        raise ValueError('Unsupported registered world size')
    # One GPU, one completion per microbatch. 16 prompts * 8 completions =
    # 128 sequences per optimizer update; four updates per rollout batch.
    generation_size = t['prompts_per_rollout_batch'] * t['generations_per_prompt']
    optimizer_size = t['prompts_per_optimizer_minibatch'] * t['generations_per_prompt']
    if optimizer_size % world or generation_size % world:
        raise ValueError('Global batches must divide evenly across ranks')
    if generation_size % optimizer_size:
        raise ValueError('Non-integral rollout/minibatch ratio')
    updates_per_rollout = generation_size // optimizer_size
    cadence = protocol['monitoring']['every_rollout_batches'] * updates_per_rollout
    result = {
        'output_dir': str(out), 'seed': seed, 'data_seed': seed,
        'model_init_kwargs': {'revision': protocol['model']['revision'],
                              # Missing field preserves historical registrations;
                              # new execution requires explicit FP32 parameters.
                              'dtype': t.get('parameter_dtype', 'bfloat16'), 'attn_implementation': 'sdpa'},
        'max_steps': t['rollout_batches'] * updates_per_rollout,
        'per_device_train_batch_size': 1,
        'gradient_accumulation_steps': optimizer_size // world,
        'generation_batch_size': generation_size,
        'num_generations': t['generations_per_prompt'],
        'num_iterations': 1, 'max_prompt_length': 1500,
        'max_completion_length': t['max_completion_tokens'],
        'learning_rate': t['learning_rate'], 'lr_scheduler_type': 'constant',
        'warmup_steps': 0, 'weight_decay': 0.01, 'max_grad_norm': 1.0,
        'optim': 'adamw_torch', 'adam_beta1': 0.9, 'adam_beta2': 0.999,
        'adam_epsilon': 1e-8, 'bf16': True, 'gradient_checkpointing': True,
        'gradient_checkpointing_kwargs': {'use_reentrant': False},
        'disable_dropout': True, 'beta': t['kl_coefficient'],
        'epsilon': 0.2, 'loss_type': 'grpo', 'scale_rewards': 'group',
        'mask_truncated_completions': False, 'sync_ref_model': False,
        'temperature': 0.9, 'top_p': 1.0, 'top_k': None,
        'use_vllm': True, 'vllm_mode': 'colocate',
        'vllm_gpu_memory_utilization': protocol.get('execution', {}).get('vllm_gpu_memory_utilization', 0.25),
        'vllm_enable_sleep_mode': True,
        'vllm_importance_sampling_correction': False,
        'use_liger_loss': False,
        'eval_strategy': 'steps', 'eval_on_start': True,
        'eval_steps': cadence,
        'per_device_eval_batch_size': protocol.get('execution', {}).get('eval_batch_size', 8),
        'save_strategy': 'steps', 'save_steps': cadence,
        'save_only_model': False, 'save_safetensors': True,
        'save_total_limit': execution.get('save_total_limit'), 'load_best_model_at_end': False,
        'logging_steps': updates_per_rollout,
        'logging_nan_inf_filter': False,
        'report_to': [], 'remove_unused_columns': False,
        'dataloader_num_workers': 0,
    }
    if execution.get('scaled'):
        result.update(ddp_find_unused_parameters=False, ddp_timeout=600)
        if execution.get('deterministic_training'):
            result['full_determinism'] = True
        if execution.get('resume_diagnostic'):
            if execution.get('phase') != 'benchmark':
                raise ValueError('Resume diagnostic must be a separate technical protocol')
            result.update(eval_on_start=False, eval_strategy='no', save_steps=4)
    if execution.get('verification_only'):
        if (not execution.get('scaled') or execution.get('phase') != 'benchmark'
                or execution.get('gpu') != 'NVIDIA GeForce RTX 4090' or world not in (1, 2)
                or t['rollout_batches'] != 2 or generation_size != 64 or optimizer_size != 16
                or t['max_completion_tokens'] != 256):
            raise ValueError('Unregistered cheap verification dimensions')
        result.update(max_prompt_length=128, eval_on_start=False, eval_strategy='no',
                      save_steps=4, save_total_limit=2, per_device_eval_batch_size=1)
    if execution.get('multidomain_full_verification'):
        result.update(eval_on_start=True, eval_strategy='steps', eval_steps=4, save_steps=4)
    elif execution.get('paired_short_experiment'):
        result['save_steps'] = execution['preserve_every_rollouts'] * updates_per_rollout
    return result


def assert_launch_ready(protocol, evidence):
    """No unreviewed draft can be used to start a paid full training job."""
    required = ('data_audit', 'reward_parity', 'trainer_gpu_smoke',
                'resume_equivalence', 'matched_validation', 'export_reload',
                'cost_projection', 'budget_reservation', 'backup_release_integration')
    if protocol.get('execution', {}).get('scaled'):
        required = ('data_audit', 'reward_parity', 'backend_cpu_parity',
                    'distributed_cpu_check', 'stopping_tests', 'budget_reservation',
                    'pilot_cap_enforced', 'backup_release_integration')
        if protocol['execution'].get('completion_driven_authorized'):
            required = tuple(k for k in required if k != 'pilot_cap_enforced') + ('completion_driven_authorized',)
        if protocol['execution']['phase'] == 'training':
            required += ('trainer_gpu_smoke', 'distributed_gpu_check', 'resume_equivalence',
                         'export_reload', 'matched_validation', 'cost_projection')
        elif protocol['execution']['phase'] != 'benchmark':
            raise RuntimeError('Unregistered scaled phase')
        if not protocol.get('budget', {}).get('cap_authorized'):
            raise RuntimeError('Scaled spending envelope is not authorized')
        if protocol['execution'].get('multidomain'):
            required += ('four_domain_verifiers', 'balanced_sampler', 'data_domains')
    elif protocol.get('id', '').startswith('shorterbetter-diagnostic-'):
        # This bounded diagnostic is itself the first GPU integration run.
        # Do not falsely mark unperformed GPU/resume checks as passed.
        required = ('data_audit', 'reward_parity', 'backend_cpu_parity',
                    'budget_reservation', 'pilot_cap_enforced', 'backup_release_integration')
        if not protocol.get('budget', {}).get('cap_authorized'):
            raise RuntimeError('Diagnostic spending envelope is not authorized')
    missing = [key for key in required if evidence.get(key) is not True]
    if protocol.get('status') != 'locked' or missing:
        raise RuntimeError('Experiment is not launch-ready: ' + ', '.join(missing or ['protocol_not_locked']))


def load_protocol(path):
    return json.loads(Path(path).read_text())


def rng_devices(device, current_cuda_device):
    """Accelerate may report plain cuda for a single process, without an index."""
    if device.type != 'cuda':
        return []
    return [current_cuda_device() if device.index is None else device.index]
