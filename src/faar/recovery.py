from __future__ import annotations

import base64
import os
import re
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from openai import OpenAI
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .api_logging import (
    anthropic_cost_rates,
    estimate_anthropic_cost_usd,
    estimate_openai_cost_usd,
    make_api_usage,
    make_vlm_logger,
    new_record,
    openai_cost_rates,
    vlm_cost_rates,
    zero_api_usage,
)
from .settings import AppSettings
from .types import RetrievalHit


@lru_cache(maxsize=1)
def _load_byt5(model_name: str, revision: str | None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, revision=revision)
    return tokenizer, model


class ByT5Corrector:
    def __init__(self, model_name: str, revision: str | None = None) -> None:
        self.model_name = model_name
        self.revision = revision

    def correct(self, text: str, max_new_tokens: int = 128) -> str:
        return self.propose_correction(text, max_new_tokens=max_new_tokens)["text"]

    def propose_correction(self, text: str, max_new_tokens: int = 128) -> dict[str, str | bool]:
        if not text.strip():
            return {"text": text, "candidate": text, "applied": False, "reason": "empty_input"}
        should_attempt, skip_reason = _should_attempt_correction(text)
        if not should_attempt:
            return {"text": text, "candidate": text, "applied": False, "reason": skip_reason}
        corrected = self._generate_correction(text, max_new_tokens=max_new_tokens)
        accepted, reason = _should_accept_correction(text, corrected)
        return {
            "text": corrected if accepted else text,
            "candidate": corrected,
            "applied": accepted,
            "reason": reason,
        }

    def _generate_correction(self, text: str, max_new_tokens: int = 128) -> str:
        # Bound inference input to keep Phase 3 batch runs tractable.
        clipped = text[:512]
        tokenizer, model = _load_byt5(self.model_name, self.revision)
        prompt = f"correct ocr noise: {clipped}"
        tokens = tokenizer(prompt, return_tensors="pt", truncation=True)
        output = model.generate(**tokens, max_new_tokens=min(max_new_tokens, 64))
        return tokenizer.decode(output[0], skip_special_tokens=True).strip()


class VisualFallback:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.logger = make_vlm_logger(settings.project_root, enabled=settings.recovery.log_vlm_calls)

    _PAID_BACKENDS = {"openai", "claude-sonnet-4-5", "anthropic", "claude"}

    def answer(self, question: str, image_paths: list[Path], fallback_context: str) -> dict:
        if not self.settings.recovery.api_enabled and self.settings.recovery.vlm_backend in self._PAID_BACKENDS:
            return {
                "backend": "openai" if self.settings.recovery.vlm_backend == "openai" else "anthropic",
                "status": "skipped",
                "reason": "api_disabled",
                "answer": "",
                "used_images": [],
                "cost_rates": vlm_cost_rates(self.settings.recovery.vlm_backend),
                "api_usage": zero_api_usage(),
            }
        if self.settings.recovery.vlm_backend == "openai":
            return self._answer_with_openai(question, image_paths)
        if self.settings.recovery.vlm_backend in {"claude-sonnet-4-5", "anthropic", "claude"}:
            return self._answer_with_anthropic(question, image_paths, fallback_context)
        return {
            "backend": "mock",
            "status": "skipped",
            "reason": "vlm_backend_not_configured",
            "answer": "",
            "used_images": [str(path) for path in image_paths],
            "fallback_context": fallback_context[:500],
            "api_usage": zero_api_usage(),
        }

    def _answer_with_openai(self, question: str, image_paths: list[Path]) -> dict:
        if not image_paths:
            return {
                "backend": "openai",
                "status": "skipped",
                "reason": "no_images_provided",
                "answer": "",
                "used_images": [],
                "api_usage": zero_api_usage(),
            }
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for VLM_BACKEND=openai.")
        client = OpenAI(max_retries=0)
        request_id = str(uuid4())
        content: list[dict] = [{"type": "text", "text": f"Answer the question using only the page image.\n\nQuestion: {question}"}]
        for path in image_paths:
            encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                }
            )
        try:
            self.logger.log(
                new_record(
                    provider="openai",
                    model=self.settings.recovery.openai_model,
                    operation="visual_fallback",
                    status="started",
                    metadata={"request_id": request_id, "image_count": len(image_paths)},
                )
            )
            response = client.chat.completions.create(
                model=self.settings.recovery.openai_model,
                messages=[{"role": "user", "content": content}],
                timeout=self.settings.recovery.request_timeout_seconds,
            )
        except Exception as exc:  # pragma: no cover - external runtime dependent
            completed_at_utc = datetime.now(UTC).isoformat()
            cost_rates = openai_cost_rates()
            self.logger.log(
                new_record(
                    provider="openai",
                    model=self.settings.recovery.openai_model,
                    operation="visual_fallback",
                    status="failed",
                    metadata={
                        "request_id": request_id,
                        "error": type(exc).__name__,
                        "request_model": self.settings.recovery.openai_model,
                        "response_model": None,
                        "completed_at_utc": completed_at_utc,
                        "cost_rates": cost_rates,
                    },
                )
            )
            return {
                "backend": "openai",
                "status": "failed",
                "reason": f"openai_error:{type(exc).__name__}",
                "answer": "",
                "used_images": [str(path) for path in image_paths],
                "request_model": self.settings.recovery.openai_model,
                "response_model": None,
                "completed_at_utc": completed_at_utc,
                "cost_rates": cost_rates,
                "api_usage": make_api_usage(api_requests=1),
            }
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        response_model = getattr(response, "model", None) or self.settings.recovery.openai_model
        completed_at_utc = datetime.now(UTC).isoformat()
        cost_rates = openai_cost_rates()
        cost_usd = estimate_openai_cost_usd(prompt_tokens, completion_tokens, cost_rates)
        self.logger.log(
            new_record(
                provider="openai",
                model=self.settings.recovery.openai_model,
                operation="visual_fallback",
                status="succeeded",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost_usd,
                metadata={
                    "request_id": request_id,
                    "image_count": len(image_paths),
                    "request_model": self.settings.recovery.openai_model,
                    "response_model": response_model,
                    "completed_at_utc": completed_at_utc,
                    "cost_rates": cost_rates,
                },
            )
        )
        return {
            "backend": "openai",
            "status": "succeeded",
            "reason": "visual_fallback_answer_generated",
            "answer": response.choices[0].message.content or "",
            "used_images": [str(path) for path in image_paths],
            "request_model": self.settings.recovery.openai_model,
            "response_model": response_model,
            "completed_at_utc": completed_at_utc,
            "cost_rates": cost_rates,
            "api_usage": make_api_usage(
                api_requests=1,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost_usd,
            ),
        }

    def _answer_with_anthropic(self, question: str, image_paths: list[Path], fallback_context: str) -> dict:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is required for VLM_BACKEND=claude-sonnet-4-5.")
        try:  # pragma: no cover - optional external runtime
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - optional external runtime
            raise RuntimeError("The `anthropic` package is required for VLM_BACKEND=claude-sonnet-4-5.") from exc

        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), max_retries=0)
        request_id = str(uuid4())
        content: list[dict] = [
            {
                "type": "text",
                "text": (
                    "Answer the question using the page image when available and the OCR context only as supporting evidence.\n\n"
                    f"Question: {question}\n\nOCR context:\n{fallback_context[:4000]}"
                ),
            }
        ]
        for path in image_paths:
            encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
            content.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": encoded},
                }
            )
        self.logger.log(
            new_record(
                provider="anthropic",
                model=self.settings.recovery.anthropic_model,
                operation="visual_fallback",
                status="started",
                metadata={"request_id": request_id, "image_count": len(image_paths)},
            )
        )
        try:
            response = client.messages.create(
                model=self.settings.recovery.anthropic_model,
                max_tokens=256,
                messages=[{"role": "user", "content": content}],
                timeout=self.settings.recovery.request_timeout_seconds,
            )
        except Exception as exc:  # pragma: no cover - external runtime dependent
            completed_at_utc = datetime.now(UTC).isoformat()
            cost_rates = anthropic_cost_rates()
            self.logger.log(
                new_record(
                    provider="anthropic",
                    model=self.settings.recovery.anthropic_model,
                    operation="visual_fallback",
                    status="failed",
                    metadata={
                        "request_id": request_id,
                        "error": type(exc).__name__,
                        "request_model": self.settings.recovery.anthropic_model,
                        "response_model": None,
                        "completed_at_utc": completed_at_utc,
                        "cost_rates": cost_rates,
                    },
                )
            )
            return {
                "backend": "anthropic",
                "status": "failed",
                "reason": f"anthropic_error:{type(exc).__name__}",
                "answer": "",
                "used_images": [str(path) for path in image_paths],
                "request_model": self.settings.recovery.anthropic_model,
                "response_model": None,
                "completed_at_utc": completed_at_utc,
                "cost_rates": cost_rates,
                "api_usage": make_api_usage(api_requests=1),
            }
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        text_parts = [getattr(block, "text", "") for block in getattr(response, "content", []) if getattr(block, "type", "") == "text"]
        response_model = getattr(response, "model", None) or self.settings.recovery.anthropic_model
        completed_at_utc = datetime.now(UTC).isoformat()
        cost_rates = anthropic_cost_rates()
        cost_usd = estimate_anthropic_cost_usd(self.settings.recovery.anthropic_model, prompt_tokens, completion_tokens)
        self.logger.log(
            new_record(
                provider="anthropic",
                model=self.settings.recovery.anthropic_model,
                operation="visual_fallback",
                status="succeeded",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost_usd,
                metadata={
                    "request_id": request_id,
                    "image_count": len(image_paths),
                    "request_model": self.settings.recovery.anthropic_model,
                    "response_model": response_model,
                    "completed_at_utc": completed_at_utc,
                    "cost_rates": cost_rates,
                },
            )
        )
        return {
            "backend": "anthropic",
            "status": "succeeded",
            "reason": "visual_fallback_answer_generated",
            "answer": "\n".join(part for part in text_parts if part).strip(),
            "used_images": [str(path) for path in image_paths],
            "request_model": self.settings.recovery.anthropic_model,
            "response_model": response_model,
            "completed_at_utc": completed_at_utc,
            "cost_rates": cost_rates,
            "api_usage": make_api_usage(
                api_requests=1,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost_usd,
            ),
        }


def semantic_backtrack(query: str, hits: list[RetrievalHit]) -> str:
    context = " ".join(hit.chunk.text for hit in hits)
    if not context:
        return query
    anchor_words = context.split()[:24]
    return f"{query} {' '.join(anchor_words)}"


def _should_attempt_correction(text: str) -> tuple[bool, str]:
    if _contains_cjk(text):
        return False, "non_latin_source"
    if _looks_formula_like(text):
        return False, "formula_like_source"
    if _weird_char_ratio(text) < 0.08 and not _has_ocr_like_token_noise(text):
        return False, "source_not_noisy_enough"
    return True, "eligible"


def _should_accept_correction(source: str, candidate: str) -> tuple[bool, str]:
    if not candidate.strip():
        return False, "empty_correction"
    if _normalize_whitespace(source) == _normalize_whitespace(candidate):
        return False, "no_change"
    if _numeric_signature(source) != _numeric_signature(candidate):
        return False, "numeric_signature_changed"
    if not 0.6 <= _length_ratio(source, candidate) <= 1.4:
        return False, "length_shift_too_large"
    if _informative_token_overlap(source, candidate) < 0.5:
        return False, "low_token_preservation"
    if _weird_char_ratio(candidate) > _weird_char_ratio(source) + 0.01:
        return False, "noise_not_reduced"
    return True, "accepted"


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", text))


def _looks_formula_like(text: str) -> bool:
    formula_markers = ("\\", "{", "}", "^", "_", "operatorname", "Delta", "alpha", "beta")
    return any(marker in text for marker in formula_markers)


def _weird_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    weird = sum(1 for char in text if not (char.isalnum() or char.isspace() or char in ".,;:!?$%()-/'\"&"))
    return weird / max(len(text), 1)


def _has_ocr_like_token_noise(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]+\d+[A-Za-z]*|\d+[A-Za-z]{2,}|[A-Za-z]{2,}\d+[A-Za-z]*", text))


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _numeric_signature(text: str) -> list[str]:
    return re.findall(r"(?:US\s*)?\$?\d[\d,]*(?:\.\d+)?%?", text)


def _informative_token_overlap(source: str, candidate: str) -> float:
    source_tokens = _informative_tokens(source)
    candidate_tokens = _informative_tokens(candidate)
    if not source_tokens:
        return 1.0
    overlap = len(source_tokens & candidate_tokens)
    return overlap / len(source_tokens)


def _informative_tokens(text: str) -> set[str]:
    tokens = {token.lower() for token in re.findall(r"[A-Za-z0-9$%]+", text)}
    return {token for token in tokens if len(token) >= 3 or any(char.isdigit() for char in token)}


def _length_ratio(source: str, candidate: str) -> float:
    if not source:
        return 1.0
    return len(candidate) / len(source)
