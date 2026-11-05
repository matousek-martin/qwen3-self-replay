"""Shared IO and tokenization for ThinkingCap pilots; importable without torch."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import time

MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def append_jsonl(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def prompt_id(row):
    value = row.get("id", row.get("prompt_id"))
    if value is None:
        raise ValueError("Prompt row lacks id/prompt_id")
    return str(value)


def generation_seed(seed, row_id, sample_id):
    digest = hashlib.sha256(f"{seed}:{row_id}:{sample_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % (2**31)


def prompt_tokens(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=True
    )


def canonical_completion(text, prompt_has_open_think=False):
    """Keep emitted thinking markers unless the prompt already contains the opener.

    The pinned Qwen3 tokenizer ends its thinking-enabled template at the
    assistant role header: the model itself emits <think>. Some other Qwen
    templates include the opener, so detect that boundary instead of assuming.
    """
    stripped = text.lstrip()
    if prompt_has_open_think and stripped.startswith("<think>"):
        stripped = stripped[len("<think>"):].lstrip("\n")
        return stripped
    return text


def prompt_has_open_think(tokenizer, prefix):
    return tokenizer.decode(prefix, skip_special_tokens=False).rstrip().endswith("<think>")


def encode_example(tokenizer, messages, completion, max_seq_len):
    prefix = prompt_tokens(tokenizer, messages)
    suffix = tokenizer.encode(canonical_completion(completion, prompt_has_open_think(tokenizer, prefix)),
                              add_special_tokens=False)
    if not suffix:
        raise ValueError("Empty completion")
    if suffix[-1] != tokenizer.eos_token_id:
        suffix.append(tokenizer.eos_token_id)
    if len(prefix) + len(suffix) > max_seq_len:
        raise ValueError(f"Sequence has {len(prefix) + len(suffix)} tokens, exceeds {max_seq_len}; refusing truncation")
    return {"input_ids": prefix + suffix, "prompt_length": len(prefix), "completion_tokens": len(suffix)}


class Deadline:
    """Soft deadline; signals request a checkpoint at the next safe boundary."""

    def __init__(self, seconds=0):
        self.started = time.monotonic()
        self.seconds = seconds
        self.signaled = False
        signal.signal(signal.SIGTERM, self._signal)
        signal.signal(signal.SIGINT, self._signal)

    def _signal(self, signum, frame):
        self.signaled = True

    @property
    def elapsed(self):
        return time.monotonic() - self.started

    @property
    def expired(self):
        return self.signaled or (self.seconds > 0 and self.elapsed >= self.seconds)


def load_model(adapter=None, trainable=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    if not torch.cuda.is_available():
        raise RuntimeError("These GPU pilots require CUDA. Pure IO tests run on CPU.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Pilot implementation requires a bf16-capable GPU for matched precision")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", trust_remote_code=False
    ).to("cuda")
    if adapter:
        config = json.loads((Path(adapter) / "adapter_config.json").read_text())
        if config.get("base_model_name_or_path") != MODEL or config.get("revision") != REVISION:
            raise ValueError("Adapter base model/revision must match the pinned baseline")
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    return model, tokenizer


def sequence_logprob(model, encoded, chunk_size=128):
    """Sum completion log probabilities without retaining full sequence x vocab logits.

    Only the Qwen3 backbone runs for hidden states; each LM-head/cross-entropy
    chunk is checkpointed, so its large vocabulary logits are recomputed during
    backward instead of being retained for every chunk. Base LM head is frozen.
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=model.device)
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = base.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
    start = encoded["prompt_length"] - 1
    hidden = hidden[0, start:-1]
    targets = ids[0, start + 1:]
    head = base.get_output_embeddings()

    def chunk_logprob(h, labels):
        return -F.cross_entropy(head(h).float(), labels, reduction="sum")

    total = torch.zeros((), dtype=torch.float32, device=model.device)
    for offset in range(0, targets.numel(), chunk_size):
        h, target = hidden[offset:offset + chunk_size], targets[offset:offset + chunk_size]
        if torch.is_grad_enabled() and h.requires_grad:
            total = total + checkpoint(chunk_logprob, h, target, use_reentrant=False)
        else:
            total = total + chunk_logprob(h, target)
    return total
