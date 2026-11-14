#!/usr/bin/env python3
"""Export selected user-visible Codex conversations and tool-call metadata locally.

Never exports system/developer messages or assistant reasoning. Tool arguments and
outputs are omitted because they may contain secrets; metadata preserves the actual
call sequence. Explicit input sessions only; this does not publish any conversation.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

PATTERNS = [
    r"gh[pousr]_[A-Za-z0-9]{25,}",
    r"hf_[A-Za-z0-9]{25,}",
    r"sk-[A-Za-z0-9_-]{30,}",
    r"\b\d{8,12}:[A-Za-z0-9_-]{30,}\b",
    r"-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----",
    r"(?i)bearer\s+[a-z0-9_.-]+",
]


def redact(text):
    for pattern in PATTERNS:
        text = re.sub(pattern, "[REDACTED CREDENTIAL]", text, flags=re.S)
    text = re.sub(
        r"(?im)^.*(?:api[_ -]?key|access[_ -]?token|password|secret[_ -]?key)\s*[:=].*$",
        "[REDACTED CREDENTIAL LINE]",
        text,
    )
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[REDACTED EMAIL]", text)
    text = re.sub(r"https://nullstudio\.cz/thinkingcap/[^\s)<>]+", "[PRIVATE DASHBOARD URL]", text)
    text = re.sub(
        r"(?im)^.*(?:account balance|credit balance|zůstatek|kredit)\b.*\d.*$",
        "[ACCOUNT BALANCE LINE OMITTED]",
        text,
    )
    text = re.sub(
        r"https?://[^\s<>\"']*(?:token|key|secret|signature)=[^\s<>\"']*",
        "[REDACTED AUTHENTICATED URL]",
        text,
        flags=re.I,
    )
    return text


def export(paths, out):
    pieces = [
        "SELECTED AGENT CONVERSATIONS AND TOOL TRACE METADATA\n",
        "Exported for the ThinkingCap submission, 2026-11-14.\n"
        "This is a selected, redacted export, not the entire project's chat history.\n"
        "User and visible assistant messages retain their original language.\n"
        "System/developer instructions, injected workspace context and private reasoning are excluded.\n"
        "Tool arguments/results are omitted; recorded tool names, call IDs and timestamps remain.\n"
        "Known credential patterns and email addresses are redacted. No claim of complete traces is made.\n"
        "Source SHA-256 values identify the private log snapshots used for this export.\n",
    ]
    counts = {"sessions": 0, "messages": 0, "tool_calls": 0, "tool_results": 0}
    for path in paths:
        raw = path.read_bytes()
        pieces.append(f"\nSESSION {path.stem}\nSource SHA-256: {hashlib.sha256(raw).hexdigest()}\n")
        counts["sessions"] += 1
        for line in raw.decode().splitlines():
            entry = json.loads(line)
            if entry.get("type") != "response_item":
                continue
            p = entry["payload"]
            timestamp = entry.get("timestamp", "not recorded")
            kind = p.get("type")
            if kind == "message" and p.get("role") in ("user", "assistant"):
                if p.get("channel") == "analysis":
                    continue
                if p.get("role") == "assistant" and p.get("phase") not in (
                    "commentary",
                    "final_answer",
                    None,
                ):
                    continue
                text = "\n".join(
                    c.get("text", "")
                    for c in p.get("content", [])
                    if c.get("type") in ("input_text", "output_text", "text")
                )
                if not text.strip() or text.startswith(
                    (
                        "# AGENTS.md instructions",
                        "<environment_context>",
                        "<permissions instructions>",
                    )
                ):
                    continue
                if (
                    p.get("role") == "assistant"
                    and p.get("phase") is None
                    and p.get("channel") not in ("commentary", "final")
                ):
                    continue
                pieces.append(f"\n[{timestamp}] {p['role'].upper()}\n{redact(text)}\n")
                counts["messages"] += 1
            elif kind in ("function_call", "custom_tool_call"):
                pieces.append(
                    f"[{timestamp}] TOOL CALL {p.get('name', 'unknown')} "
                    f"id={p.get('call_id', 'unavailable')} [arguments omitted]\n"
                )
                counts["tool_calls"] += 1
            elif kind in ("function_call_output", "custom_tool_call_output"):
                pieces.append(
                    f"[{timestamp}] TOOL RESULT id={p.get('call_id', 'unavailable')} "
                    "[output omitted]\n"
                )
                counts["tool_results"] += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(pieces))
    print(json.dumps(counts))
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    export(args.session, args.out)
