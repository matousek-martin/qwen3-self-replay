"""CPU-only replay admission and FIFO storage; no RNG or model dependencies."""

from .objectives import reasoning_is_complete

ADMISSION_RULES = ("all_correct", "verified_correct")


def select_replay(group, answer, task, is_correct, admission="all_correct"):
    """Return the first shortest verified completion, or None.

    Regrade text with the training verifier; serialized correctness flags are
    never authority. Token IDs and EOS-derived truncation come from the caller's
    live tokenizer/generator, not an external replay dataset. Length includes EOS.
    """
    if admission not in ADMISSION_RULES:
        raise ValueError("Unknown replay admission rule: " + admission)
    if not group:
        return None
    eligible = []
    for sample in group:
        ids = sample["ids"]
        if not ids or sample["total_tokens"] != len(ids):
            raise ValueError("Replay requires nonempty IDs and matching total-token length")
        text = sample["completion"]
        if (sample.get("truncated") is False and reasoning_is_complete(text)
                and is_correct(text, answer, task)):
            eligible.append(sample)
    if not eligible or (admission == "all_correct" and len(eligible) != len(group)):
        return None
    return min(eligible, key=lambda sample: sample["total_tokens"])


def admit_replay(buffer, prompt_id, prompt_ids, selected, capacity):
    """Keep one shortest solution per prompt, preserving historical FIFO order.

    Updating an existing prompt does not refresh its insertion position. Equal
    lengths retain the old solution. Return an auditable event for every group.
    """
    if capacity < 1:
        raise ValueError("Replay capacity must be positive")
    event = {"eligible": selected is not None, "action": "rejected",
             "selected_sample_id": None, "selected_tokens": None, "evicted_prompt_id": None}
    if selected is None:
        return event
    event.update(selected_sample_id=selected["sample_id"], selected_tokens=len(selected["ids"]))
    previous = buffer.get(prompt_id)
    if previous is None or len(selected["ids"]) < len(previous[1]):
        buffer[prompt_id] = (list(prompt_ids), list(selected["ids"]))
        event["action"] = "inserted" if previous is None else "shortened"
    else:
        event["action"] = "retained"
    if len(buffer) > capacity:
        event["evicted_prompt_id"] = next(iter(buffer))
        del buffer[event["evicted_prompt_id"]]
    return event
