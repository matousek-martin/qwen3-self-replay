"""Maximum-length engineering check; backward only, no optimizer update."""
import json
import time
import torch


def check_sampling(trainer, out):
    """Exercise actual vLLM through the study override, without reward/update."""
    length, mode = trainer.max_completion_length, trainer.model.training
    trainer._engineering_check = True
    trainer.max_completion_length = 128
    trainer.model.eval()
    try:
        prompt = trainer.train_dataset[0]['prompt']
        _, completions, _, _ = trainer._generate_single_turn([prompt] * 8, None)
        distinct = len({tuple(ids) for ids in completions})
        report = {'status': 'passed' if distinct > 1 else 'failed',
                  'engineering_only': True, 'optimizer_updates': 0,
                  'samples': len(completions), 'distinct_completions': distinct,
                  'max_completion_tokens': 128}
        (out / 'gpu-sampling-check.json').write_text(json.dumps(report, indent=2) + '\n')
        if len(completions) != 8 or distinct <= 1:
            raise RuntimeError('GPU sampler produced no diversity; inspect request seeds before paid baseline')
    finally:
        trainer.max_completion_length = length
        trainer.model.train(mode)
        trainer._engineering_check = False


def check(trainer, out):
    started = time.time()
    model = trainer.accelerator.unwrap_model(trainer.model)
    device = trainer.accelerator.device
    completion_length = trainer.args.max_completion_length
    prompt_length = trainer.args.max_prompt_length
    # Artificial IDs are a memory stress fixture, never training examples.
    ids = (torch.arange(prompt_length + completion_length, device=device) % model.config.vocab_size)[None]
    mask = torch.ones_like(ids)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    previous_mode = model.training
    previous_accumulation = getattr(trainer, 'current_gradient_accumulation_steps', None)
    trainer.current_gradient_accumulation_steps = trainer.args.gradient_accumulation_steps
    eval_batch = trainer.args.per_device_eval_batch_size
    report = {'status': 'started', 'optimizer_updates': 0, 'prompt_tokens': prompt_length,
              'completion_tokens': completion_length, 'evaluation_batch_size': eval_batch}
    path = out / 'gpu-memory-check.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    torch.cuda.reset_peak_memory_stats()
    try:
        model.eval()
        with torch.no_grad():
            lp, ent = trainer._get_per_token_logps_and_entropies(trainer.model, ids.repeat(eval_batch, 1),
                mask.repeat(eval_batch, 1), completion_length, compute_entropy=True)
            if not torch.isfinite(lp).all() or not torch.isfinite(ent).all():
                raise RuntimeError('Nonfinite maximum-length evaluation')
            ref, _ = trainer._get_per_token_logps_and_entropies(trainer.ref_model, ids, mask, completion_length)
        old = lp[:1].detach()
        del lp, ent
        model.train()
        # Keep an Adam moment-sized allocation live during backward to exercise
        # the steady-state budget before spending time on baseline generation.
        moment_bytes = sum(p.numel() * p.element_size() * 2 for p in model.parameters() if p.requires_grad)
        moment_reserve = torch.empty(moment_bytes, dtype=torch.uint8, device=device)
        inputs = {'prompt_ids': ids[:, :prompt_length], 'prompt_mask': mask[:, :prompt_length],
                  'completion_ids': ids[:, prompt_length:], 'completion_mask': mask[:, prompt_length:],
                  'advantages': torch.ones(1, device=device), 'old_per_token_logps': old,
                  'ref_per_token_logps': ref}
        with trainer.accelerator.autocast():
            loss = trainer._compute_loss(trainer.model, inputs)
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
            raise RuntimeError('Nonfinite or missing maximum-length gradient')
        # Also verify generation fits beside policy/reference, gradients and
        # optimizer-sized storage, before committing to a long baseline.
        loss_value = float(loss.detach())
        del loss, inputs, old, ref, gradients
        torch.cuda.empty_cache()
        trainer.llm.wake_up()
        free, total = torch.cuda.mem_get_info()
        report['generation_free_bytes_with_adam_and_gradients'] = free
        report['device_total_bytes'] = total
        trainer.llm.sleep(level=1)
        report.update(status='passed', loss=loss_value,
                      adam_moment_reserve_bytes=moment_bytes,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      elapsed_seconds=time.time() - started)
        del moment_reserve
    except Exception as exc:
        report.update(status='failed', error=str(exc), elapsed_seconds=time.time() - started)
        raise
    finally:
        model.zero_grad(set_to_none=True)
        model.train(previous_mode)
        if previous_accumulation is None:
            del trainer.current_gradient_accumulation_steps
        else:
            trainer.current_gradient_accumulation_steps = previous_accumulation
        for metrics in trainer._metrics.values():
            metrics.clear()
        torch.cuda.empty_cache()
        path.write_text(json.dumps(report, indent=2) + '\n')
