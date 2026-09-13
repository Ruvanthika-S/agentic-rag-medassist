#!/usr/bin/env python3
"""Verify a drafted biomedical answer against the retrieved evidence."""

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
FINAL_ANSWERS = {"yes", "no", "maybe", "insufficient_evidence"}


class VerifierAgent:
    """Independently check a summary and enforce explicit negative knowledge."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        completion_function: CompletionFunction | None = None,
    ) -> None:
        self.model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        self._completion_function = completion_function
        if completion_function is None:
            if Groq is None:
                raise RuntimeError("groq is not installed; install the groq package")
            key = api_key or os.getenv("GROQ_API_KEY")
            if not key:
                raise RuntimeError("GROQ_API_KEY is required for the live verifier")
            self.client = Groq(api_key=key)

    @staticmethod
    def _format_evidence(evidence: list[dict[str, Any]]) -> str:
        parts = []
        for index, item in enumerate(evidence, 1):
            identifier = item.get("pmid") or item.get("url") or "unknown"
            parts.append(
                f"EVIDENCE_{index} | source={item.get('source_dataset', 'unknown')} | "
                f"id={identifier}\n{item.get('chunk_text') or item.get('text') or ''}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are an independent biomedical evidence verifier. Check every claim in the "
            "draft against the supplied evidence text, not general knowledge. Return JSON "
            "with exactly: verdict, answer, rationale, evidence_used. verdict must be "
            "approved or rejected. answer must be yes, no, maybe, or insufficient_evidence. "
            "Approve only when the answer and cited claims are directly supported. If the "
            "evidence is insufficient, contradictory, or the citations do not support the "
            "draft, reject it and set answer to insufficient_evidence. Never guess. "
            "evidence_used must contain only supplied labels such as EVIDENCE_1."
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

    def verify(
        self,
        question: str,
        draft: dict[str, Any],
        evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not question.strip():
            raise ValueError("question must not be empty")
        if not evidence:
            raise ValueError("evidence must not be empty")

        user_prompt = (
            f"Question:\n{question}\n\nDraft answer:\n{json.dumps(draft, ensure_ascii=False)}\n\n"
            f"Evidence:\n{self._format_evidence(evidence)}"
        )
        try:
            result = json.loads(self._complete(self._system_prompt(), user_prompt))
        except json.JSONDecodeError as exc:
            raise ValueError("Verifier returned invalid JSON") from exc

        verdict = str(result.get("verdict", "")).strip().lower()
        answer = str(result.get("answer", "")).strip().lower()
        evidence_used = result.get("evidence_used")
        valid_labels = {f"EVIDENCE_{index}" for index in range(1, len(evidence) + 1)}
        if verdict not in {"approved", "rejected"}:
            raise ValueError("Verifier verdict must be approved or rejected")
        if answer not in FINAL_ANSWERS:
            raise ValueError("Verifier answer is not a supported final answer")
        if not isinstance(evidence_used, list) or not set(evidence_used).issubset(valid_labels):
            raise ValueError("Verifier referenced an unavailable evidence label")

        if verdict == "approved":
            if answer == "insufficient_evidence":
                raise ValueError("An insufficient-evidence answer cannot be approved")
            if not evidence_used:
                raise ValueError("An approved answer must cite evidence")
        else:
            answer = "insufficient_evidence"

        return {
            "verdict": verdict,
            "answer": answer,
            "rationale": str(result.get("rationale", "")).strip(),
            "evidence_used": evidence_used,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify an evidence-cited draft with Groq")
    parser.add_argument("question")
    parser.add_argument("draft_json")
    parser.add_argument("evidence_json")
    args = parser.parse_args()
    result = VerifierAgent().verify(
        args.question, json.loads(args.draft_json), json.loads(args.evidence_json)
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()