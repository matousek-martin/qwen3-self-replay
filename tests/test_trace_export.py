import importlib.util
import json

from self_replay.paths import repository

spec = importlib.util.spec_from_file_location(
    "export_agent_traces", repository() / "scripts/export_agent_traces.py"
)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def test_trace_export_excludes_private_reasoning_instructions_and_tool_payload(tmp_path):
    entries = []
    for role, channel, text in [
        ("system", None, "system-instruction-sentinel"),
        ("assistant", "analysis", "hidden-analysis-sentinel"),
        ("user", None, "# AGENTS.md instructions for example"),
        ("user", None, "Please compare these results."),
        ("assistant", "final", "The result is inconclusive."),
    ]:
        entries.append(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": role,
                    "channel": channel,
                    "content": [{"type": "output_text", "text": text}],
                },
            }
        )
    entries.append(
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "exec_command",
                "call_id": "call-1",
                "arguments": "private tool payload",
            },
        }
    )
    src = tmp_path / "session.jsonl"
    src.write_text("\n".join(json.dumps(e) for e in entries))
    out = tmp_path / "trace.txt"
    counts = exporter.export([src], out)
    text = out.read_text()
    assert "Please compare these results" in text and "The result is inconclusive" in text
    for private in (
        "hidden-analysis-sentinel",
        "system-instruction-sentinel",
        "private tool payload",
        "AGENTS.md",
    ):
        assert private not in text
    assert "TOOL CALL exec_command" in text
    assert counts["messages"] == 2 and counts["tool_calls"] == 1


def test_trace_redacts_emails_and_authenticated_urls():
    assert "person@example.org" not in exporter.redact("Email: person@example.org")
    assert "secret-value" not in exporter.redact("https://example.org/?token=secret-value")
