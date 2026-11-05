"""Single-update, on-policy LoRA GRPO pilots. No external trainer dependency.

Run from project root: python -m experiments.rl_training.train --help
These small pilots are method adaptations, not full paper reproductions.
"""

import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
import math
import statistics
import time
from pathlib import Path

from .objectives import VARIANTS, advantages, loss_denominator, random_streams, reasoning_is_complete, rewards, training_cap
from .runtime import RuntimeLimit
from .replay_selection import ADMISSION_RULES, admit_replay, select_replay
from .adaptive_replay import REPLAY_REWARDS, REWARD_DEFINITION, replay_rewards
from experiments.gpu_policy import verify_gpu_names

ROOT = Path(__file__).resolve().parents[2]
MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def verify_training_device(torch, device):
    """Enforce the training policy even when invoked without the cloud wrapper."""
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Model training requires NVIDIA GeForce RTX 4090 with CUDA")
    names = verify_gpu_names(torch.cuda.get_device_name(torch.cuda.current_device()))
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This pilot requires bf16 support")
    return names[0]


def group_rewards(successes, lengths, args, reference):
    if args.variant == "replay":
        return replay_rewards(successes, lengths, args.replay_reward,
                              args.length_coefficient, reference)
    return rewards(successes, lengths, args.variant, args.length_coefficient,
                   reference, replay_reward=args.replay_reward)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("Empty training data")
    ids = set()
    for row in rows:
        if row.get("split") != "train":
            raise ValueError("Every training row must explicitly have split=train")
        if row.get("task") not in ("math", "multiple_choice"):
            raise ValueError("This pilot only implements the approved math/choice verifier")
        if row.get("source") not in ("gsm8k", "skywork", "commonsense_qa", "arc_easy"):
            raise ValueError("This pilot only accepts the four approved training sources")
        if not row.get("id") or row["id"] in ids:
            raise ValueError("Need unique, nonempty prompt IDs")
        if not row.get("prompt") or "answer" not in row or "source" not in row:
            raise ValueError("Missing required prompt, answer or source")
        ids.add(row["id"])
    return rows


def verifier():
    path = ROOT / "ThinkingCap-Test/utils/train/rewards.py"
    spec = importlib.util.spec_from_file_location("thinkingcap_training_rewards", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.is_task_correct


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def token_logprobs(model, prompt_ids, completion_ids, chunk_size=64, temperature=1.0):
    """Compute completion logps while checkpointing the large output projection.

    This deliberately uses Qwen3's standard backbone and lm_head. The model and
    exported adapter themselves are unchanged and require no custom load code.
    """
    import torch
    from torch.utils.checkpoint import checkpoint

    base = model.get_base_model()
    tokens = torch.cat((prompt_ids, completion_ids), dim=1)
    hidden = base.model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                        use_cache=False, return_dict=True).last_hidden_state
    hidden = hidden[:, prompt_ids.shape[1] - 1:-1, :].squeeze(0)
    targets = completion_ids.squeeze(0)
    logps, entropies = [], []

    def project(h, labels):
        logits = base.lm_head(h).float() / temperature
        logp = logits.log_softmax(-1)
        chosen = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        # Diagnostic only: entropy of the untruncated temperature-scaled policy.
        with torch.no_grad():
            entropy = -(logp.exp() * logp).sum(-1)
        return chosen, entropy

    for start in range(0, len(targets), chunk_size):
        h, labels = hidden[start:start + chunk_size], targets[start:start + chunk_size]
        lp, ent = checkpoint(project, h, labels, use_reentrant=False) if torch.is_grad_enabled() else project(h, labels)
        logps.append(lp)
        entropies.append(ent)
    return torch.cat(logps), torch.cat(entropies)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--variant", choices=VARIANTS, default="correctness")
    parser.add_argument("--optimizer", choices=("grpo", "dr_grpo"), default="dr_grpo")
    parser.add_argument("--scale-rewards", choices=("auto", "yes", "no"), default="auto",
                        help="auto: group std for GRPO, no std for Dr. GRPO")
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--max-seconds", type=float, default=0,
                        help="Optional elapsed-time limit; 0 (default) runs until the requested groups complete or a stop signal arrives")
    parser.add_argument("--runtime-control", help="Optional JSON file with max_seconds and/or stop; reread at most once per second")
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--final-cap", type=int, default=1024)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--length-coefficient", type=float, default=0.1)
    parser.add_argument("--length-reference", type=float, default=1024,
                        help="Fallback length reference. Use --references from base training rollouts for experiments.")
    parser.add_argument("--references", help="JSON mapping training source to mean base rollout length")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--kl-beta", type=float, default=0.0)
    parser.add_argument("--replay-weight", type=float, default=0.1)
    parser.add_argument("--replay-reward", choices=REPLAY_REWARDS, default="correctness",
                        help="For replay only: choose the RL reward, not replay scoring; fixed shares control references and coefficient")
    parser.add_argument("--replay-capacity", type=int, default=128)
    parser.add_argument("--replay-admission", choices=ADMISSION_RULES, default="all_correct",
                        help="Replay only: admit all-correct groups or verified correct solutions from any group")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--save-every", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main():
    args = make_parser().parse_args()
    if args.variant != "replay" and args.replay_admission != "all_correct":
        raise SystemExit("--replay-admission requires --variant replay")
    if args.variant != "replay" and args.replay_reward == "adaptive":
        raise SystemExit("--replay-reward adaptive requires --variant replay")
    rows = load_rows(args.data)
    if args.group_size < 2 or args.max_steps < 1:
        raise SystemExit("Need group-size >= 2 and positive steps")
    RuntimeLimit.validate_seconds(args.max_seconds)
    if args.max_new_tokens < 1 or args.final_cap < 1 or args.final_cap > args.max_new_tokens:
        raise SystemExit("Need 0 < final-cap <= max-new-tokens")
    if not 0 <= args.length_coefficient < 1 or args.length_reference <= 0:
        raise SystemExit("Invalid length coefficient or reference")
    if min(args.chunk_size, args.rank, args.save_every, args.max_prompt_tokens, args.replay_capacity) <= 0:
        raise SystemExit("Chunk size, rank, save interval, prompt cap and replay capacity must be positive")
    if args.temperature <= 0 or not 0 < args.top_p <= 1 or args.top_k < 0:
        raise SystemExit("Invalid sampling parameters")
    if min(args.learning_rate, args.kl_beta, args.replay_weight) < 0:
        raise SystemExit("Learning rate and loss coefficients must be nonnegative")
    references = json.loads(Path(args.references).read_text()) if args.references else {}
    if any(not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in references.values()):
        raise SystemExit("Length references must be positive finite numbers")
    scale_rewards = args.optimizer == "grpo" if args.scale_rewards == "auto" else args.scale_rewards == "yes"
    config = {**vars(args), "model": MODEL, "revision": REVISION, "scale_rewards_resolved": scale_rewards,
              "replay_reward_definition": REWARD_DEFINITION if args.variant == "replay" and args.replay_reward == "adaptive" else None,
              "data_sha256": sha256(args.data), "references_sha256": sha256(args.references) if args.references else None,
              "prompt_count": len(rows), "algorithm": "single_update_on_policy_group_relative",
              "sampling_logprob_note": "temperature-scaled full-vocabulary likelihood; top-k/top-p sampling support is not differentiated"}
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

    config["training_gpu"] = verify_training_device(torch, args.device)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit("Output directory is nonempty; use a fresh directory (resume is deliberately not implicit)")
    out.mkdir(parents=True, exist_ok=True)
    config["packages"] = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "accelerate")}
    atomic_json(out / "config.json", config)
    start = time.monotonic()
    runtime = RuntimeLimit(args.max_seconds, args.runtime_control, started=start)
    runtime.install_signal_handlers()

    def expired():
        return runtime.expired

    class Deadline(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return expired()

    torch.manual_seed(args.seed)
    rng, replay_rng = random_streams(args.seed)
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=dtype,
                                               attn_implementation="sdpa").to(args.device)
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0,
                                          base_model_name_or_path=MODEL, revision=REVISION,
                                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                                          task_type="CAUSAL_LM", bias="none"))
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0)
    is_correct = verifier()
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    eos.discard(None)
    if tokenizer.eos_token_id is not None:
        eos.add(tokenizer.eos_token_id)
    thinking_end = tokenizer.convert_tokens_to_ids("</think>")
    replay = {}
    order = list(range(len(rows)))
    rng.shuffle(order)
    total_tokens, updates, completed_groups, training_tokens_total = 0, 0, 0, 0
    data_cursor = 0
    skipped_long = 0
    reason = "max_steps"
    error = None

    def save(label):
        checkpoint = out / label
        model.save_pretrained(checkpoint, safe_serialization=True)
        tokenizer.save_pretrained(checkpoint)
        atomic_json(checkpoint / "training_state.json", {"updates": updates, "completed_groups": completed_groups,
                    "generated_tokens": total_tokens, "elapsed_seconds": time.monotonic() - start,
                    "data_cursor": data_cursor, "seed": args.seed, "runtime_policy": runtime.summary(),
                    "restart_note": "Adapter only; not an exact optimizer/RNG resume checkpoint"})
        return checkpoint

    try:
        with (out / "metrics.jsonl").open("w", buffering=1) as metrics, (out / "rollouts.jsonl").open("w", buffering=1) as traces:
            while completed_groups < args.max_steps:
                if expired():
                    reason = runtime.reason
                    break
                if data_cursor and data_cursor % len(order) == 0:
                    rng.shuffle(order)
                row = rows[order[data_cursor % len(order)]]
                data_cursor += 1
                prompt_ids = tokenizer.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True,
                                                           enable_thinking=True, return_tensors="pt").to(args.device)
                if prompt_ids.shape[1] > args.max_prompt_tokens:
                    skipped_long += 1
                    if skipped_long >= len(rows) and completed_groups == 0:
                        raise RuntimeError("No training prompt fits the explicit prompt-token cap")
                    continue
                cap = training_cap(completed_groups, args.max_steps, args.max_new_tokens, args.final_cap, args.variant)
                group_start = time.monotonic()
                group = []
                model.eval()
                # Generate the entire group from one unchanged policy before any update.
                with torch.inference_mode():
                    generated = model.generate(prompt_ids, attention_mask=torch.ones_like(prompt_ids),
                        do_sample=True, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                        max_new_tokens=cap, pad_token_id=tokenizer.pad_token_id, use_cache=True,
                        num_return_sequences=args.group_size,
                        stopping_criteria=StoppingCriteriaList([Deadline()]))[:, prompt_ids.shape[1]:]
                    for sample, sequence in enumerate(generated):
                        # Clone outside inference mode below before autograd consumes IDs.
                        ids = sequence.tolist()
                        # Batched generation pads finished sequences up to the longest.
                        first_eos = next((i for i, token in enumerate(ids) if token in eos), None)
                        if first_eos is not None:
                            ids = ids[:first_eos + 1]
                        total_tokens += len(ids)
                        text = tokenizer.decode(ids, skip_special_tokens=True)
                        truncated = not ids or ids[-1] not in eos
                        valid = reasoning_is_complete(text)
                        success = bool(not truncated and valid and is_correct(text, row["answer"], row["task"]))
                        record = {"prompt_id": row["id"], "sample_id": f"step{completed_groups}-sample{sample}",
                                  "completion": text, "total_tokens": len(ids), "truncated": truncated,
                                  "reasoning_tokens": ids.index(thinking_end) if thinking_end in ids else len(ids),
                                  "reasoning_valid": valid, "correct": success, "model_revision": REVISION,
                                  "policy_update": updates, "finish_reason": "length_or_deadline" if truncated else "eos",
                                  "generation": {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
                                                 "max_new_tokens": cap, "seed": args.seed}}
                        traces.write(json.dumps(record) + "\n")
                        group.append({**record, "ids": ids})
                if len(group) != args.group_size or expired():
                    reason = runtime.reason
                    break
                successes = [g["correct"] for g in group]
                lengths = [g["total_tokens"] for g in group]
                reference = references.get(row["source"], args.length_reference)
                reward = group_rewards(successes, lengths, args, reference)
                advantage = advantages(reward, scale=scale_rewards)
                zero_variance = max(reward) == min(reward)
                replay_admission = None
                if args.variant == "replay":
                    shortest = select_replay(group, row["answer"], row["task"], is_correct,
                                             args.replay_admission)
                    replay_admission = admit_replay(replay, row["id"], prompt_ids[0].tolist(),
                                                    shortest, args.replay_capacity)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss_sum, kl_sum, entropy_sum = 0.0, 0.0, 0.0
                trained_tokens = 0
                has_gradient = False
                for g, adv in zip(group, advantage):
                    if adv == 0 and args.kl_beta == 0:
                        continue
                    completion = torch.tensor([g["ids"]], device=args.device)
                    reference_lp = None
                    if args.kl_beta:
                        with torch.no_grad(), model.disable_adapter():
                            reference_lp, _ = token_logprobs(model, prompt_ids, completion, args.chunk_size, args.temperature)
                    lp, entropy = token_logprobs(model, prompt_ids, completion, args.chunk_size, args.temperature)
                    # One optimizer update per on-policy group: old policy == current.
                    # exp(lp - stop_gradient(lp)) has value 1 and the correct policy gradient.
                    ratio = (lp - lp.detach()).exp()
                    kl = torch.zeros_like(lp)
                    if reference_lp is not None:
                        difference = reference_lp - lp
                        kl = difference.exp() - difference - 1
                    denominator = loss_denominator(args.optimizer, len(g["ids"]), args.max_new_tokens, args.group_size)
                    loss = (-ratio * adv + args.kl_beta * kl).sum() / denominator
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite policy loss")
                    loss.backward()
                    has_gradient = True
                    trained_tokens += len(g["ids"])
                    loss_sum += loss.detach().item()
                    kl_sum += kl.detach().sum().item()
                    entropy_sum += entropy.detach().sum().item()
                replay_loss = 0.0
                replay_tokens = 0
                replay_prompt_id = None
                if args.variant == "replay" and replay and args.replay_weight:
                    replay_prompt_id = replay_rng.choice(list(replay))
                    saved_prompt, saved_completion = replay[replay_prompt_id]
                    lp, _ = token_logprobs(model, torch.tensor([saved_prompt], device=args.device),
                                           torch.tensor([saved_completion], device=args.device), args.chunk_size, 1.0)
                    loss = -args.replay_weight * lp.mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite replay loss")
                    loss.backward()
                    has_gradient = True
                    replay_loss = loss.detach().item()
                    replay_tokens = len(saved_completion)
                if has_gradient:
                    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    updates += 1
                else:
                    norm = 0.0
                completed_groups += 1
                training_tokens_total += trained_tokens + replay_tokens
                metrics.write(json.dumps({"group": completed_groups, "optimizer_updates": updates, "prompt_id": row["id"],
                    "source": row["source"], "success_rate": sum(successes) / len(successes),
                    "mean_total_tokens": statistics.mean(lengths), "correct_mean_tokens": statistics.mean([n for n, ok in zip(lengths, successes) if ok]) if any(successes) else None,
                    "mean_reasoning_tokens": statistics.mean(g["reasoning_tokens"] for g in group),
                    "wrong_mean_tokens": statistics.mean([n for n, ok in zip(lengths, successes) if not ok]) if not all(successes) else None,
                    "truncation_rate": sum(g["truncated"] for g in group) / len(group),
                    "invalid_reasoning_rate": sum(not g["reasoning_valid"] for g in group) / len(group),
                    "reward_mean": statistics.mean(reward), "zero_variance": zero_variance, "all_wrong": not any(successes),
                    "all_correct": all(successes), "loss": loss_sum, "replay_loss": replay_loss,
                    "group_success_count": sum(successes), "task": row["task"],
                    "replay_admission_rule": args.replay_admission, "replay_admission": replay_admission,
                    "replay_sampled_prompt_id": replay_prompt_id,
                    "adaptive_target_total_tokens": min([n for n, ok in zip(lengths, successes) if ok], default=None)
                        if args.variant == "replay" and args.replay_reward == "adaptive" else None,
                    "replay_pool_size": len(replay), "gradient_norm": float(norm),
                    "kl": kl_sum / trained_tokens if trained_tokens and args.kl_beta else None,
                    "policy_entropy": entropy_sum / trained_tokens if trained_tokens else None,
                    "trained_tokens": trained_tokens, "replay_tokens": replay_tokens,
                    "training_tokens_total": training_tokens_total, "generated_tokens_total": total_tokens,
                    "training_cap": cap, "length_reference": reference, "elapsed_seconds": time.monotonic() - start,
                    "group_seconds": time.monotonic() - group_start,
                    "peak_gpu_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else 0}) + "\n")
                print(f"group={completed_groups} updates={updates} correct={sum(successes)}/{len(successes)} tokens={statistics.mean(lengths):.1f}", flush=True)
                if completed_groups % args.save_every == 0:
                    save(f"checkpoint-{completed_groups:04d}")
    except Exception as exc:
        error = str(exc)
        reason = "error"
        raise
    finally:
        checkpoint = save("adapter")
        hashes = {p.name: sha256(p) for p in checkpoint.glob("*.safetensors")}
        atomic_json(out / "status.json", {"status": "failed" if error else "completed" if reason == "max_steps" else "partial",
                    "stop_reason": reason, "error": error, "completed_groups": completed_groups,
                    "optimizer_updates": updates, "generated_tokens": total_tokens,
                    "training_tokens_total": training_tokens_total,
                    "elapsed_seconds": time.monotonic() - start, "skipped_long_prompts": skipped_long,
                    "adapter": str(checkpoint), "checkpoint_sha256": hashes, "runtime_policy": runtime.summary()})


if __name__ == "__main__":
    main()
