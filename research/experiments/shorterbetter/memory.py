"""Exact Qwen3 token log probabilities with a bounded vocabulary projection.

Keep complete causal attention, but project hidden states in token chunks and
recompute those projections during backward. No vocabulary/sequence truncation.
Single-process text-only adapter; the pinned TRL GRPO objective is unchanged.
"""
import torch
from torch.utils.checkpoint import checkpoint


def projected_statistics(hidden, head, targets, temperature, compute_entropy, chunk_tokens=128):
    def project(h, ids):
        # Match Accelerate's fp32 output conversion before TRL temperature scaling.
        logits = head(h).float() / temperature
        log_z = torch.logsumexp(logits, dim=-1)
        selected = logits.gather(-1, ids.unsqueeze(-1)).squeeze(-1) - log_z
        if compute_entropy:
            with torch.no_grad():
                entropy = log_z.detach() - (logits.softmax(-1) * logits).sum(-1)
        else:
            entropy = torch.zeros_like(selected)
        return selected, entropy

    logps, entropies = [], []
    for start in range(0, hidden.shape[1], chunk_tokens):
        h = hidden[:, start:start + chunk_tokens]
        ids = targets[:, start:start + chunk_tokens]
        if torch.is_grad_enabled() and (h.requires_grad or any(p.requires_grad for p in head.parameters())):
            lp, ent = checkpoint(project, h, ids, use_reentrant=False)
        else:
            lp, ent = project(h, ids)
        logps.append(lp)
        entropies.append(ent)
    return torch.cat(logps, dim=1), torch.cat(entropies, dim=1) if compute_entropy else None


def qwen_logps(model, input_ids, attention_mask, logits_to_keep, temperature,
               compute_entropy=False, chunk_tokens=128):
    if model.config.model_type != 'qwen3' or not hasattr(model, 'lm_head'):
        raise ValueError('Memory adapter is verified only for text Qwen3 causal LM')
    if not 0 < logits_to_keep < input_ids.shape[1]:
        raise ValueError('Invalid completion length')
    logps, entropies = [], []
    # Generation groups stay intact; only scoring uses single-sequence forwards.
    for i in range(input_ids.shape[0]):
        hidden = model.model(input_ids=input_ids[i:i+1], attention_mask=attention_mask[i:i+1],
                             use_cache=False, return_dict=True).last_hidden_state
        hidden = hidden[:, -(logits_to_keep + 1):-1]
        lp, ent = projected_statistics(hidden, model.lm_head, input_ids[i:i+1, -logits_to_keep:],
                                        temperature, compute_entropy, chunk_tokens)
        logps.append(lp)
        if compute_entropy:
            entropies.append(ent)
    return torch.cat(logps, dim=0), torch.cat(entropies, dim=0) if compute_entropy else None
