#!/usr/bin/env python3
"""Draft evidence-cited answers with a Groq-hosted language model."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from typing import Any

from scripts.agents.groq_retry import call_with_retry

try:
    from groq import Groq
except ImportError:  # pragma: no cover - reported when the client is used
    Groq = None


CompletionFunction = Callable[[str, str], str]
ALLOWED_ANSWERS = {"yes", "no", "maybe"}


class SummarizerAgent:
    """Create a candidate answer grounded in supplied evidence."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        completion_function: CompletionFunction | None = None,
    ) -> None:
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self._completion_function = completion_function
        if completion_function is None:
            if Groq is None:
                raise RuntimeError("groq is not installed; install the groq package")
            key = api_key or os.getenv("GROQ_API_KEY")
            if not key:
                raise RuntimeError("GROQ_API_KEY is required for the live summarizer")
            self.client = Groq(api_key=key)

    @staticmethod
    def _format_evidence(evidence: list[dict[str, Any]]) -> str:
        formatted = []
        for index, item in enumerate(evidence, 1):
            source = item.get("source_dataset", "unknown")
            identifier = item.get("pmid") or item.get("url") or "unknown"
            text = item.get("chunk_text") or item.get("text") or ""
            formatted.append(f"EVIDENCE_{index} | source={source} | id={identifier}\n{text}")
        return "\n\n".join(formatted)

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are a biomedical evidence summarizer. Answer only from the supplied evidence. "
            "Return JSON with exactly these keys: answer, rationale, cited_evidence. "
            "answer must be exactly yes, no, or maybe. cited_evidence must be a non-empty array "
            "of evidence labels such as EVIDENCE_1. Every factual claim in rationale must be "
            "supported by a cited evidence item. Do not use outside knowledge or invent citations. "
            "If the evidence does not support a defensible answer, choose maybe and explain the limitation."
        )

    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        if self._completion_function is not None:
            return self._completion_function(system_prompt, user_prompt)
        response = call_with_retry(
            lambda: self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        )
        return response.choices[0].message.content or ""

    def summarize(self, question: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
        if not question.strip():
            raise ValueError("question must not be empty")
        if not evidence:
            raise ValueError("evidence must not be empty")

        raw = self._complete(
            self._system_prompt(),
            f"Question:\n{question}\n\nEvidence:\n{self._format_evidence(evidence)}",
        )
        try:
            draft = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("Summarizer returned invalid JSON") from exc

        answer = str(draft.get("answer", "")).strip().lower()
        citations = draft.get("cited_evidence")
        valid_labels = {f"EVIDENCE_{index}" for index in range(1, len(evidence) + 1)}
        if answer not in ALLOWED_ANSWERS:
            raise ValueError("Summarizer answer must be yes, no, or maybe")
        if not isinstance(citations, list) or not citations:
            raise ValueError("Summarizer must cite at least one evidence item")
        if not set(citations).issubset(valid_labels):
            raise ValueError("Summarizer cited an evidence label that was not supplied")

        return {
            "answer": answer,
            "rationale": str(draft.get("rationale", "")).strip(),
            "cited_evidence": citations,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Draft an evidence-cited answer with Groq")
    parser.add_argument("question")
    parser.add_argument("evidence_json", help="JSON list of evidence objects")
    args = parser.parse_args()
    evidence = json.loads(args.evidence_json)
    print(json.dumps(SummarizerAgent().summarize(args.question, evidence), indent=2))


if __name__ == "__main__":
    main()