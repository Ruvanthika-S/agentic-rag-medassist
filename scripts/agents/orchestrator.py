#!/usr/bin/env python3
"""Coordinate retrieval, fallback search, summarization, and verification."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

from scripts.agents.retriever_agent import RetrieverAgent
from scripts.agents.summarizer_agent import SummarizerAgent
from scripts.agents.verifier_agent import VerifierAgent
from scripts.agents.web_fallback_agent import WebFallbackAgent


class Orchestrator:
    """Run the agentic pipeline with bounded iterations and elapsed time."""

    def __init__(
        self,
        retriever: Any | None = None,
        fallback: Any | None = None,
        summarizer: Any | None = None,
        verifier: Any | None = None,
        confidence_threshold: float = 0.75,
        timeout_seconds: float = 120.0,
        max_iterations: int = 1,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        self.retriever = retriever or RetrieverAgent()
        self.fallback = fallback or WebFallbackAgent(
            confidence_threshold=confidence_threshold
        )
        self.summarizer = summarizer or SummarizerAgent()
        self.verifier = verifier or VerifierAgent()
        self.confidence_threshold = confidence_threshold
        self.timeout_seconds = timeout_seconds
        self.max_iterations = max_iterations

    def _timed_out(self, started: float) -> bool:
        return time.monotonic() - started > self.timeout_seconds

    @staticmethod
    def _confidence_level(confidence: float, threshold: float) -> str:
        if confidence >= threshold:
            return "high"
        if confidence >= threshold / 2:
            return "medium"
        return "low"

    @staticmethod
    def _insufficient_response(
        question: str, rationale: str, evidence: list[dict[str, Any]], confidence: float,
        used_web_fallback: bool, threshold: float
    ) -> dict[str, Any]:
        return {
            "question": question,
            "answer": "insufficient_evidence",
            "rationale": rationale,
            "evidence_used": evidence,
            "confidence_level": Orchestrator._confidence_level(confidence, threshold),
            "used_web_fallback": used_web_fallback,
        }

    def run(self, question: str, top_k: int = 5) -> dict[str, Any]:
        if not question.strip():
            raise ValueError("question must not be empty")
        started = time.monotonic()

        retrieval = self.retriever.retrieve(question, top_k=top_k)
        if self._timed_out(started):
            return self._insufficient_response(
                question, "Pipeline timed out after retrieval.", retrieval.get("results", []),
                float(retrieval.get("retrieval_confidence", 0.0)), False,
                self.confidence_threshold,
            )

        enriched = self.fallback.maybe_search(question, retrieval)
        confidence = float(enriched.get("retrieval_confidence", 0.0))
        evidence = list(enriched.get("results", []))
        used_web = bool(enriched.get("used_web_fallback", False))
        if not evidence:
            return self._insufficient_response(
                question, "No evidence was retrieved.", evidence, confidence, used_web,
                self.confidence_threshold,
            )
        if self._timed_out(started):
            return self._insufficient_response(
                question, "Pipeline timed out before summarization.", evidence, confidence,
                used_web, self.confidence_threshold,
            )

        for _ in range(self.max_iterations):
            draft = self.summarizer.summarize(question, evidence)
            if self._timed_out(started):
                return self._insufficient_response(
                    question, "Pipeline timed out before verification.", evidence, confidence,
                    used_web, self.confidence_threshold,
                )
            verified = self.verifier.verify(question, draft, evidence)
            if verified.get("answer") == "insufficient_evidence":
                rationale = verified.get("rationale") or "Evidence did not support the draft."
                used_labels = set(verified.get("evidence_used", []))
                used_evidence = [
                    item for index, item in enumerate(evidence, 1)
                    if f"EVIDENCE_{index}" in used_labels
                ]
                return self._insufficient_response(
                    question, rationale, used_evidence, confidence, used_web,
                    self.confidence_threshold,
                )

            used_labels = set(verified.get("evidence_used", []))
            used_evidence = [
                item for index, item in enumerate(evidence, 1)
                if f"EVIDENCE_{index}" in used_labels
            ]
            return {
                "question": question,
                "answer": verified["answer"],
                "rationale": verified.get("rationale", ""),
                "evidence_used": used_evidence,
                "confidence_level": self._confidence_level(confidence, self.confidence_threshold),
                "used_web_fallback": used_web,
            }

        return self._insufficient_response(
            question, "Maximum verification iterations reached.", evidence, confidence,
            used_web, self.confidence_threshold,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the biomedical agentic RAG pipeline")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    print(json.dumps(Orchestrator(timeout_seconds=args.timeout).run(args.question, args.top_k), indent=2))


if __name__ == "__main__":
    main()