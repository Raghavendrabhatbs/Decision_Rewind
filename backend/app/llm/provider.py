from __future__ import annotations

import json
from typing import Any, Dict

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from backend.app.config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
)


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
            raise RuntimeError(f"Unsupported LLM provider: {self.provider}. Configure LLM_PROVIDER=groq.")

        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=45)
        instructions = (
            "You are the DECISION-REWIND executive investigation assistant. Provide clean, professional, "
            "and well-structured responses formatted with markdown headings, clear bullet points, or clean markdown tables. "
            "PRESENTATION GUIDELINES: "
            "1. Do NOT dump raw JSON or dictionary syntax (such as `{\"source\":\"failed_logins\",...}`). "
            "Always translate technical data into human-readable descriptions (e.g. `failed_logins -> D2`). "
            "2. When explaining affected decisions, present a clear executive summary followed by structured bullet points or a clean summary table with columns: Decision, Changed Feature(s), Provenance Path, and Status. "
            "3. For application-state claims, answer using only the supplied application evidence. Persisted database state, frozen model outputs, counterfactual replay, "
            "provenance, and deterministic verification are authoritative. Universal Logs provide supporting temporal evidence and may be partial. "
            "4. Do not invent state, logs, metrics, decisions, or operations. Never authorize, execute, or approve recovery and never override verification. "
            "5. If sources conflict, explicitly describe the conflict and prioritize authoritative persisted state. "
            "Dependency paths indicate possible impact, not proof that a decision changed. "
            "Provide concise reasoning and evidence references, not hidden chain-of-thought."
        )
        try:
            response = client.responses.create(
                model=self.model,
                instructions=instructions,
                input=(
                    f"Question:\n{question}\n\n"
                    f"Available application context (JSON):\n"
                    f"{json.dumps(evidence, sort_keys=True, default=str)}"
                ),
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            )
        except APITimeoutError as error:
            raise RuntimeError("The Groq API request timed out.") from error
        except APIConnectionError as error:
            raise RuntimeError("Could not reach the Groq API.") from error
        except APIStatusError as error:
            if error.status_code == 413:
                try:
                    compact_evidence = {k: v for k, v in evidence.items() if k != "universal_log_context"}
                    compact_evidence["workflow_audit"] = evidence.get("workflow_audit", [])[:8]
                    compact_evidence["universal_log_context"] = {
                        "note": "Trimmed due to token limit",
                        "matched_events": (evidence.get("universal_log_context") or {}).get("matched_events", [])[:5],
                    }
                    response = client.responses.create(
                        model=self.model,
                        instructions=instructions,
                        input=(
                            f"Question:\n{question}\n\n"
                            f"Available application context (JSON):\n"
                            f"{json.dumps(compact_evidence, sort_keys=True, default=str)}"
                        ),
                        temperature=self.temperature,
                        max_output_tokens=min(self.max_tokens, 768),
                    )
                    evidence = compact_evidence
                except Exception as retry_err:
                    raise RuntimeError("Groq token limit reached (HTTP 413). Please ask a more specific question.") from retry_err
            elif error.status_code == 401:
                message = "Groq rejected the API key. Check GROQ_API_KEY in the project .env file."
                raise RuntimeError(message) from error
            elif error.status_code == 403:
                message = "The Groq API denied access. Check model access and network restrictions."
                raise RuntimeError(message) from error
            elif error.status_code == 429:
                message = "The Groq API rate limit or quota was reached."
                raise RuntimeError(message) from error
            else:
                message = f"The Groq API returned HTTP {error.status_code}."
                raise RuntimeError(message) from error

        answer = response.output_text
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("The Groq API returned an empty answer.")
        return {"source": self.source, "answer": answer.strip(), "evidence": evidence}
