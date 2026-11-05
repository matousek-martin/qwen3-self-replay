"""Canonical replay framing and fail-closed validation for the pinned Qwen student."""

OPEN = '<think>'
CLOSE = '</think>'


def normalize_completion(text):
    """Repair the historical missing opener without changing solution content."""
    text = text.strip()
    if text.count(CLOSE) != 1 or text.count(OPEN) > 1:
        raise ValueError('Replay requires exactly one reasoning block')
    if OPEN not in text:
        text = OPEN + '\n' + text
    if not text.startswith(OPEN):
        raise ValueError('Replay opening tag must precede reasoning')
    reasoning, answer = text[len(OPEN):].split(CLOSE)
    if not reasoning.strip() or not answer.strip() or OPEN in answer:
        raise ValueError('Replay requires reasoning and final answer')
    return text


def validate_tokens(tokenizer, prompt, completion):
    """Check the actual concatenated sequence and the loss-bearing target prefix."""
    opening = tokenizer.convert_tokens_to_ids(OPEN)
    closing = tokenizer.convert_tokens_to_ids(CLOSE)
    if opening == closing or tokenizer.encode(OPEN, add_special_tokens=False) != [opening]:
        raise ValueError('Unexpected tokenizer thinking tags')
    if opening in prompt or closing in prompt:
        raise ValueError('Prompt must not inject replay reasoning tags')
    if (not completion or completion[0] != opening or completion.count(opening) != 1
            or completion.count(closing) != 1 or completion[-1] != tokenizer.eos_token_id
            or len(completion) > 2048):
        raise ValueError('Invalid replay target tokens or length')
    text = tokenizer.decode(completion[:-1], skip_special_tokens=False)
    if normalize_completion(text) != text:
        raise ValueError('Replay token target is not canonically framed')
