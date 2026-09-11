"""Retry helpers for transient Groq API failures."""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable
from typing import Any, TypeVar


T = TypeVar("T")
RETRY_SECONDS_RE = re.compile(r"try again in\s+([0-9]+(?:\.[0-9]+)?)s", re.IGNORECASE)


def _retry_delay(exc: Exception, attempt: int) -> float:
    match = RETRY_SECONDS_RE.search(str(exc))
    if match:
        return min(float(match.group(1)) + 1.0, float(os.getenv("GROQ_MAX_RETRY_WAIT", "900")))
    return min(2**attempt, float(os.getenv("GROQ_MAX_RETRY_WAIT", "900")))


def call_with_retry(call: Callable[[], T]) -> T:
    """Retry rate-limit errors, while allowing other errors to fail immediately."""
    max_retries = int(os.getenv("GROQ_MAX_RETRIES", "3"))
    for attempt in range(max_retries + 1):
        try:
            return call()
        except Exception as exc:
            is_rate_limit = getattr(exc, "status_code", None) == 429 or type(exc).__name__ == "RateLimitError"
            if not is_rate_limit or attempt >= max_retries:
                raise
            delay = _retry_delay(exc, attempt)
            print(
                f"Groq rate limit reached; retry {attempt + 1}/{max_retries} "
                f"after {delay:.1f}s: {exc}",
                flush=True,
            )
            time.sleep(delay)
    raise RuntimeError("unreachable")