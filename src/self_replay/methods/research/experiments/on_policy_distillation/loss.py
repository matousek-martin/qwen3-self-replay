"""Conditional forward KL on detached, sampled prefixes (not sequence KL).

No optimizer or rollout loop lives here. CPU use is for synthetic gradient tests;
the integrating trainer must enforce RTX 4090 before any model training.
"""
import json
import math

import torch
from torch.utils.checkpoint import checkpoint


def assert_tokenizer_alignment(student, teacher, student_vocab, teacher_vocab):
    """Fail closed: equal vocabulary sizes alone are insufficient."""
    checks = {
        "vocabulary": student.get_vocab() == teacher.get_vocab(),
        "backend": json.loads(student.backend_tokenizer.to_str())
        == json.loads(teacher.backend_tokenizer.to_str()),
        "special_tokens": student.special_tokens_map == teacher.special_tokens_map,
        "chat_template": student.chat_template == teacher.chat_template,
        "logit_vocabulary": student_vocab == teacher_vocab,
    }
    if not all(checks.values()):
        raise ValueError("Tokenizer mismatch: " + ", ".join(k for k, v in checks.items() if not v))
    ids = list(student.get_vocab().values())
    if not ids or min(ids) < 0 or max(ids) >= student_vocab:
        raise ValueError("Tokenizer IDs outside logit vocabulary")


def completion_mask(input_ids, attention_mask, prompt_lengths, eos_token_ids):
    """Return [B,L-1] mask for logits[:, :-1] predicting input_ids[:, 1:].

    Right padding only. Include the first completion EOS; exclude all subsequent
    positions. EOS in the prompt is irrelevant. Never infer padding from token ID
    because Qwen training uses pad_token_id == eos_token_id.
    """
    if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape:
        raise ValueError("Expected matching [batch, sequence] IDs and attention")
    if input_ids.dtype != torch.long or input_ids.shape[1] < 2:
        raise ValueError("Need int64 IDs and at least two positions")
    if not torch.all((attention_mask == 0) | (attention_mask == 1)):
        raise ValueError("Attention mask must be binary")
    attention = attention_mask.bool()
    if not attention[:, 0].all() or (attention[:, 1:] & ~attention[:, :-1]).any():
        raise ValueError("Only contiguous right padding is supported")
    prompts = torch.as_tensor(prompt_lengths, device=input_ids.device)
    if prompts.shape != (input_ids.shape[0],) or prompts.dtype != torch.long:
        raise ValueError("Need one int64 prompt length per sequence")
    if (prompts < 1).any() or (prompts >= attention.sum(-1)).any():
        raise ValueError("Each sequence needs a prompt and at least one completion token")
    positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :]
    completion = attention & (positions >= prompts[:, None])
    eos = torch.zeros_like(attention)
    for token in eos_token_ids:
        eos |= input_ids == token
    eos &= completion
    earlier_eos = eos.long().cumsum(-1) - eos.long()
    return (completion & (earlier_eos == 0))[:, 1:]


def _token_kl(student_logits, teacher_logits, temperature):
    # Preserve float64 for gradcheck; accumulate low precision inputs in FP32.
    dtype = torch.float64 if student_logits.dtype == torch.float64 else torch.float32
    logp = (student_logits.to(dtype) / temperature).log_softmax(-1)
    logq = (teacher_logits.detach().to(device=logp.device, dtype=dtype) / temperature).log_softmax(-1)
    if not torch.isfinite(logp).all() or not torch.isfinite(logq).all():
        raise ValueError("Nonfinite active distribution")
    return (logq.exp() * (logq - logp)).sum(-1)


def full_vocab_forward_kl(student_logits, teacher_logits, target_mask, temperature=1.0):
    """Mean KL(q||p) over active positions; no implicit temperature² multiplier.

    Logits must ALREADY align to targets, with shape [..., V]. For causal model
    outputs pass logits[:, :-1, :] and completion_mask(...). Teacher is detached.
    Padding is selected out BEFORE softmax, so NaNs in padding cannot leak.
    """
    if student_logits.shape != teacher_logits.shape or student_logits.ndim < 2:
        raise ValueError("Student/teacher logit shapes must match")
    if not student_logits.is_floating_point() or not teacher_logits.is_floating_point():
        raise ValueError("Floating point logits required")
    if target_mask.shape != student_logits.shape[:-1] or target_mask.dtype != torch.bool:
        raise ValueError("Need a boolean target mask matching logit positions")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    if not target_mask.any():
        raise ValueError("No active completion tokens")
    return _token_kl(student_logits[target_mask],
                     teacher_logits[target_mask.to(teacher_logits.device)], temperature).mean()


def freeze_teacher(teacher):
    teacher.eval()
    teacher.requires_grad_(False)
    for parameter in teacher.parameters():
        parameter.grad = None
    return teacher


def qwen_forward_kl(student, teacher, input_ids, attention_mask, target_mask, chunk_size=32):
    """Memory-bounded lm_head projection; exact conditional KL with tau=1.

    Supports Qwen3ForCausalLM and a PEFT-wrapped student; batch 1 is recommended.
    Both models must be on the same device. Call alignment checks at startup and
    completion_mask for every batch. Full causal backbones run once; checkpoint
    recomputes each head chunk during backward without retaining [B,L,V] logits.
    """
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if target_mask.dtype != torch.bool or target_mask.shape != input_ids[:, 1:].shape:
        raise ValueError("Mask must match shifted targets")
    if not target_mask.any():
        raise ValueError("No active completion tokens")
    if attention_mask.shape != input_ids.shape or not torch.all(
            (attention_mask == 0) | (attention_mask == 1)):
        raise ValueError("Invalid attention mask")
    if (target_mask & ~(attention_mask[:, 1:].bool() & attention_mask[:, :-1].bool())).any():
        raise ValueError("Target mask selects padding or a target without an attended prefix")
    if teacher.training or any(p.requires_grad for p in teacher.parameters()):
        raise ValueError("Teacher must be frozen and in eval mode")
    base = student.get_base_model() if hasattr(student, "get_base_model") else student
    if base.config.vocab_size != teacher.config.vocab_size:
        raise ValueError("Logit vocabularies differ")
    with torch.no_grad():
        teacher_hidden = teacher.model(input_ids=input_ids, attention_mask=attention_mask,
                                       use_cache=False, return_dict=True).last_hidden_state
        teacher_hidden = teacher_hidden[:, :-1, :][target_mask]
    student_hidden = base.model(input_ids=input_ids, attention_mask=attention_mask,
                               use_cache=False, return_dict=True).last_hidden_state
    student_hidden = student_hidden[:, :-1, :][target_mask]

    def project(student_chunk, teacher_chunk):
        with torch.no_grad():
            teacher_logits = teacher.lm_head(teacher_chunk)
        student_logits = base.lm_head(student_chunk)
        return _token_kl(student_logits, teacher_logits, 1.0).sum()

    chunks = []
    for start in range(0, len(student_hidden), chunk_size):
        sh = student_hidden[start:start + chunk_size]
        th = teacher_hidden[start:start + chunk_size]
        chunks.append(checkpoint(project, sh, th, use_reentrant=False)
                      if torch.is_grad_enabled() else project(sh, th))
    return torch.stack(chunks).sum() / len(student_hidden)
