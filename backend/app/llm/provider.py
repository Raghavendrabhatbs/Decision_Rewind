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
        try:
            response = client.responses.create(
                model=self.model,
                instructions=(
                    "You are the DECISION-REWIND investigation assistant. Answer directly using only the supplied "
                    "application evidence. Persisted database state, frozen model outputs, counterfactual replay, "
                    "provenance, and deterministic verification are authoritative. Universal Logs provide "
                    "supporting temporal evidence and may be partial. Do not invent state, logs, metrics, decisions, "
                    "or operations. Never authorize, execute, or approve recovery and never override verification. "
                    "If sources conflict, explicitly describe the conflict and prioritize authoritative persisted "
                    "state. Dependency paths indicate possible impact, not proof that a decision changed. If "
                    "universal_log_context.truncated is true, state that the supplied log evidence is partial. "
                    "Use whole-history summaries only for aggregate counts and supplied events for specific log "
                    "details. Provide concise reasoning and evidence references, not hidden chain-of-thought."
                ),
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
            if error.status_code == 401:
                message = "Groq rejected the API key. Check GROQ_API_KEY in the project .env file."
            elif error.status_code == 403:
                message = "The Groq API denied access. Check model access and network restrictions."
            elif error.status_code == 429:
                message = "The Groq API rate limit or quota was reached."
            else:
                message = f"The Groq API returned HTTP {error.status_code}."
            raise RuntimeError(message) from error

        answer = response.output_text
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("The Groq API returned an empty answer.")
        return {"source": self.source, "answer": answer.strip(), "evidence": evidence}
