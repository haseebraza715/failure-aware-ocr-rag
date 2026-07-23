"""API accounting contracts for one logical request with lifecycle events."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from faar.api_logging import ApiCallLogger, new_record
from faar.recovery import VisualFallback
from faar.settings import AppSettings


def test_started_and_succeeded_events_count_as_one_api_request(tmp_path: Path) -> None:
    log_path = tmp_path / "vlm_calls.jsonl"
    logger = ApiCallLogger(log_path)

    logger.log(
        new_record(
            provider="anthropic",
            model="claude-sonnet-4-5",
            operation="visual_fallback",
            status="started",
            metadata={"example_id": "ex1"},
        )
    )
    logger.log(
        new_record(
            provider="anthropic",
            model="claude-sonnet-4-5",
            operation="visual_fallback",
            status="succeeded",
            prompt_tokens=100,
            completion_tokens=20,
            cost_usd=0.01,
            metadata={"example_id": "ex1"},
        )
    )

    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [record["status"] for record in records] == ["started", "succeeded"]
    assert logger.call_count == 1
    assert logger.total_cost_usd == 0.01


def test_openai_request_is_logged_before_execution_and_usage_after(
    monkeypatch,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"image")
    log_path = tmp_path / "logs/vlm_calls.jsonl"
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-placeholder")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-2024-11-20")

    class FakeCompletions:
        def create(self, **kwargs):
            records = [json.loads(line) for line in log_path.read_text().splitlines()]
            assert records[-1]["status"] == "started"
            assert kwargs["model"] == "gpt-4o-2024-11-20"
            return SimpleNamespace(
                model="gpt-4o-2024-11-20",
                usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
                choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr("faar.recovery.OpenAI", lambda: fake_client)
    settings = AppSettings(project_root=tmp_path)
    settings.recovery.vlm_backend = "openai"
    settings.recovery.log_vlm_calls = True

    result = VisualFallback(settings).answer("Question?", [image_path], "")

    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert result["status"] == "succeeded"
    assert [record["status"] for record in records] == ["started", "succeeded"]
    assert records[0]["metadata"]["request_id"] == records[1]["metadata"]["request_id"]
    assert records[1]["prompt_tokens"] == 100
    assert records[1]["completion_tokens"] == 20
    assert records[1]["cost_usd"] > 0
    assert records[1]["metadata"]["request_model"] == "gpt-4o-2024-11-20"
    assert records[1]["metadata"]["response_model"] == "gpt-4o-2024-11-20"
    assert result["response_model"] == "gpt-4o-2024-11-20"
    assert result["completed_at_utc"]
