#!/usr/bin/env python3
"""Trigger DuckDuckGo evidence retrieval when local confidence is low."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from typing import Any

try:
    from ddgs import DDGS
except ImportError:  # pragma: no cover - exercised through the installation check
    DDGS = None


SearchFunction = Callable[[str, int], list[dict[str, Any]]]


class WebFallbackAgent:
    """Add web evidence only when the retriever's heuristic is below threshold."""

    def __init__(
        self,
        confidence_threshold: float = 0.75,
        max_results: int = 5,
        search_function: SearchFunction | None = None,
    ) -> None:
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if max_results < 1:
            raise ValueError("max_results must be at least 1")
        self.confidence_threshold = confidence_threshold
        self.max_results = max_results
        self.search_function = search_function or self._duckduckgo_search

    @staticmethod
    def _duckduckgo_search(query: str, max_results: int) -> list[dict[str, Any]]:
        if DDGS is None:
            raise RuntimeError("ddgs is not installed")
        return list(DDGS().text(query, safesearch="moderate", max_results=max_results))

    @staticmethod
    def _normalize_result(result: dict[str, Any]) -> dict[str, Any]:
        title = str(result.get("title") or "").strip()
        body = str(result.get("body") or result.get("snippet") or "").strip()
        url = str(result.get("href") or result.get("url") or "").strip()
        return {
            "chunk_text": body,
            "pmid": "",
            "source_dataset": "web",
            "title": title,
            "url": url,
        }

    def maybe_search(
        self, query: str, retrieval: dict[str, Any]
    ) -> dict[str, Any]:
        if not query.strip():
            raise ValueError("query must not be empty")
        confidence = float(retrieval.get("retrieval_confidence", 0.0))
        local_results = list(retrieval.get("results", []))
        if confidence >= self.confidence_threshold:
            return {
                "results": local_results,
                "retrieval_confidence": confidence,
                "used_web_fallback": False,
                "web_results": [],
            }

        try:
            web_results = [
                self._normalize_result(item)
                for item in self.search_function(query, self.max_results)
            ]
            error = None
        except Exception as exc:  # Keep the orchestrator informed without hiding failure.
            web_results = []
            error = f"{type(exc).__name__}: {exc}"

        response = {
            "results": local_results + web_results,
            "retrieval_confidence": confidence,
            "used_web_fallback": True,
            "web_results": web_results,
        }
        if error:
            response["web_search_error"] = error
        return response


def main() -> None:
    parser = argparse.ArgumentParser(description="Run confidence-gated web fallback")
    parser.add_argument("query")
    parser.add_argument("--confidence", type=float, required=True)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--max-results", type=int, default=5)
    args = parser.parse_args()
    response = WebFallbackAgent(
        confidence_threshold=args.threshold, max_results=args.max_results
    ).maybe_search(args.query, {"retrieval_confidence": args.confidence, "results": []})
    print(json.dumps(response, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()