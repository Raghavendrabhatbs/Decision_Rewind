from __future__ import annotations

import json
from typing import Any, Dict

from groq import APIConnectionError, APIStatusError, APITimeoutError, Groq

from backend.app.config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
)


# Keep the prompt comfortably below provider request limits. The LLM should
# receive the decision-rewind evidence, not the entire application state/log.
MAX_EVIDENCE_CHARS = 14_000
MAX_LOG_CHARS = 5_000
MAX_GRAPH_CHARS = 2_500


SYSTEM_PROMPT = """
You are the Decision Rewind AI assistant. Explain the deterministic analysis in
simple language for a student, engineer, or project judge.

The deterministic Decision Rewind engine is authoritative. It decides the
counterfactual outputs, affected/unaffected decisions, provenance, and whether
selective rewind is justified. You only explain those results.

For every question about a correction, answer using EXACTLY these short sections:

## What changed?
State the corrected feature(s) and old → new value in one short sentence.

## What happened after correction?
Explain the historical output → counterfactual output for the affected decisions.
Mention that the existing trained model was re-run on the corrected input; it
was NOT retrained.

## Why were these decisions affected?
Explain the provenance chain in simple terms, for example:
failed_logins → D1 → D2 → D5.

## Why were the other decisions not affected?
Name the unaffected decisions and give one short reason.

## Why is selective rewind safe?
Explain that only decisions whose counterfactual result differs and whose
provenance connects them to the correction are rewound. Unaffected decisions
remain unchanged.

## Evidence
Give at most 2 short bullets from the supplied universal logs or verification.

Rules:
- Do not invent facts, logs, model outputs, or dependencies.
- Do not reproduce the raw evidence JSON.
- Do not use tables.
- Do not use long paragraphs.
- Keep the complete answer under about 300 words.
- Use simple, presentation-friendly language.
- If evidence is missing, say exactly what is missing.
"""


def _compact_value(value: Any, max_chars: int) -> Any:
    """Return a JSON-safe, bounded representation of an evidence value."""
    try:
        encoded = json.dumps(value, sort_keys=True, default=str, ensure_ascii=True)
    except Exception:
        encoded = str(value)
    if len(encoded) <= max_chars:
        return value
    preview = encoded[:max_chars]
    return {
        "truncated": True,
        "preview": preview,
        "original_chars": len(encoded),
    }


def _compact_evidence(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Keep high-value reasoning evidence within a strict request budget."""
    preferred_keys = [
        "reasoning_mode", "dataset", "event_id", "correction",
        "historical_state", "current_state", "corrected_state",
        "historical_decisions", "counterfactual_decisions",
        "decision_impacts", "affected_decisions", "unaffected_decisions",
        "provenance_paths", "provenance_graph", "universal_log_context",
        "verification",
    ]

    per_key_limits = {
        "universal_log_context": MAX_LOG_CHARS,
        "provenance_graph": MAX_GRAPH_CHARS,
        "historical_state": 2_000,
        "current_state": 2_000,
        "corrected_state": 2_000,
        "historical_decisions": 2_500,
        "counterfactual_decisions": 2_500,
        "decision_impacts": 3_000,
        "provenance_paths": 2_500,
        "verification": 2_000,
    }

    compact: Dict[str, Any] = {}
    for key in preferred_keys:
        if key not in evidence:
            continue
        compact[key] = _compact_value(
            evidence[key], per_key_limits.get(key, 1_500)
        )

    # Add a tiny preview of any other useful frontend field only when there is
    # still room. Never allow arbitrary request payloads to grow unbounded.
    for key, value in evidence.items():
        if key in compact or key in {"changes", "proposed_changes"}:
            continue
        candidate = _compact_value(value, 500)
        trial = dict(compact)
        trial[key] = candidate
        if len(json.dumps(trial, sort_keys=True, default=str, ensure_ascii=True)) <= MAX_EVIDENCE_CHARS:
            compact = trial

    # Final deterministic budget enforcement. Preserve the fields that are
    # essential to Decision Rewind reasoning and progressively shrink the
    # largest verbose fields until the JSON request is below the limit.
    shrink_order = [
        ("universal_log_context", 2_500),
        ("provenance_graph", 1_200),
        ("provenance_paths", 1_500),
        ("decision_impacts", 1_800),
        ("historical_decisions", 1_500),
        ("counterfactual_decisions", 1_500),
        ("historical_state", 1_000),
        ("current_state", 1_000),
        ("corrected_state", 1_000),
        ("verification", 1_000),
    ]
    for key, limit in shrink_order:
        encoded = json.dumps(compact, sort_keys=True, default=str, ensure_ascii=True)
        if len(encoded) <= MAX_EVIDENCE_CHARS:
            break
        if key in compact:
            compact[key] = _compact_value(evidence.get(key, compact[key]), limit)

    encoded = json.dumps(compact, sort_keys=True, default=str, ensure_ascii=True)
    if len(encoded) > MAX_EVIDENCE_CHARS:
        # Last resort: preserve the core causal reasoning fields only.
        core = [
            "reasoning_mode", "event_id", "correction", "corrected_state",
            "counterfactual_decisions", "decision_impacts",
            "affected_decisions", "unaffected_decisions",
            "provenance_paths", "universal_log_context",
        ]
        compact = {
            key: _compact_value(compact[key], 800)
            for key in core if key in compact
        }
    return compact


class LLMProvider:
    def __init__(
        self,
        provider: str = LLM_PROVIDER,
        api_key: str = LLM_API_KEY,
        base_url: str = LLM_BASE_URL,
        model: str = LLM_MODEL,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
    ):
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def source(self) -> str:
        return f"{self.provider}:{self.model}"

    def explain(self, question: str, evidence: Dict[str, Any]) -> Dict[str, Any]:
        if not self.api_key:
            raise RuntimeError(
                "LLM is not configured. Set GROQ_API_KEY in the project .env file and restart the API."
            )
        if self.provider.lower() != "groq":
            raise RuntimeError(
                f"Unsupported LLM provider: {self.provider}. Configure LLM_PROVIDER=groq."
            )

        compact_evidence = _compact_evidence(evidence)
        evidence_json = json.dumps(
            compact_evidence, sort_keys=True, default=str, ensure_ascii=True
        )

        # Use Groq's native client endpoint. Do not override it with a custom
        # OpenAI-compatible base URL unless a future provider explicitly needs it.
        client = Groq(api_key=self.api_key, timeout=45)
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Question:\n{question}\n\n"
                            f"Decision Rewind evidence (bounded JSON):\n{evidence_json}"
                        ),
                    },
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except APITimeoutError as error:
            raise RuntimeError("The Groq API request timed out.") from error
        except APIConnectionError as error:
            raise RuntimeError("Could not reach the Groq API.") from error
        except APIStatusError as error:
            if error.status_code == 401:
                message = "Groq rejected the API key. Check GROQ_API_KEY in the project .env file."
            elif error.status_code == 403:
                message = "The Groq API denied access. Check model access and network restrictions."
            elif error.status_code == 413:
                message = (
                    "The Groq request was too large (HTTP 413). The AI evidence was compacted; "
                    "if this persists, reduce the universal-log context or use a model with a larger context window."
                )
            elif error.status_code == 429:
                message = "The Groq API rate limit or quota was reached."
            else:
                detail = str(error)
                try:
                    detail = error.response.text
                except Exception:
                    pass
                message = f"The Groq API returned HTTP {error.status_code}: {detail}"
            raise RuntimeError(message) from error

        message = response.choices[0].message
        answer = message.content if hasattr(message, "content") else ""
        if isinstance(answer, list):
            answer = "".join(
                part.get("text", "") for part in answer if isinstance(part, dict)
            )
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("The Groq API returned an empty answer.")
        return {
            "source": self.source,
            "answer": answer.strip(),
            "evidence": evidence,
        }
