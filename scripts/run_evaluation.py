#!/usr/bin/env python3
"""Evaluate the full agentic biomedical RAG pipeline on MIRAGE subsets."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.agents.orchestrator import Orchestrator


def load_questions(path: Path, dataset: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = []
    for question_id, item in payload.items():
        options = item.get("options") or {}
        answer_key = str(item.get("answer", "")).strip()
        correct_answer = str(options.get(answer_key, answer_key)).strip().lower()
        questions.append(
            {
                "dataset": dataset,
                "question_id": str(question_id),
                "question": str(item.get("question", "")).strip(),
                "correct_answer": correct_answer,
            }
        )
    return questions


def evidence_pmids(evidence: list[dict[str, Any]]) -> list[str]:
    return list(dict.fromkeys(str(item["pmid"]) for item in evidence if item.get("pmid")))


def evaluate_question(orchestrator: Orchestrator, item: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        output = orchestrator.run(item["question"])
        answer = str(output.get("answer", "")).strip().lower()
        result = {
            "question": item["question"],
            "question_id": item["question_id"],
            "dataset": item["dataset"],
            "our_answer": answer,
            "correct_answer": item["correct_answer"],
            "match": answer == item["correct_answer"],
            "confidence_level": output.get("confidence_level", "unknown"),
            "used_web_fallback": bool(output.get("used_web_fallback", False)),
            "evidence_pmids": evidence_pmids(output.get("evidence_used", [])),
            "rationale": output.get("rationale", ""),
            "status": "ok",
        }
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        print(
            f"ERROR {item['dataset']} {item['question_id']}: {message}",
            flush=True,
        )
        result = {
            "question": item["question"],
            "question_id": item["question_id"],
            "dataset": item["dataset"],
            "our_answer": None,
            "correct_answer": item["correct_answer"],
            "match": False,
            "confidence_level": "error",
            "used_web_fallback": False,
            "evidence_pmids": [],
            "rationale": "Evaluation error: " + str(exc),
            "error": type(exc).__name__,
            "error_message": message,
            "status": "error",
        }
    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return result


def print_summary(results: list[dict[str, Any]]) -> None:
    total = len(results)
    scored = [item for item in results if item.get("status", "ok") == "ok"]
    matches = sum(bool(item["match"]) for item in scored)
    accuracy = matches / len(scored) * 100 if scored else 0.0
    print("\n=== Evaluation Summary ===")
    print(f"Questions: {total:,}")
    print(f"Accuracy: {matches:,}/{len(scored):,} scored ({accuracy:.2f}%)")
    for dataset in ("pubmedqa", "bioasq"):
        subset = [item for item in results if item["dataset"] == dataset]
        subset_scored = [item for item in subset if item.get("status", "ok") == "ok"]
        subset_matches = sum(bool(item["match"]) for item in subset_scored)
        subset_accuracy = subset_matches / len(subset_scored) * 100 if subset_scored else 0.0
        label = "PubMedQA" if dataset == "pubmedqa" else "BioASQ-Y/N"
        print(f"{label} Accuracy: {subset_matches:,}/{len(subset_scored):,} scored ({subset_accuracy:.2f}%)")
    insufficient = sum(
        item.get("status", "ok") == "ok" and item.get("our_answer") == "insufficient_evidence"
        for item in results
    )
    fallback = sum(bool(item["used_web_fallback"]) for item in results)
    errors = sum(item.get("status") == "error" for item in results)
    print(f"Insufficient evidence answers: {insufficient:,}")
    print(f"Web fallback triggered: {fallback:,}")
    print(f"Evaluation errors: {errors:,}")
    if results:
        print(f"Average time per question: {sum(item['elapsed_seconds'] for item in results) / len(results):.2f}s")


def load_existing_results(path: Path) -> tuple[list[dict[str, Any]], set[tuple[str, str]]]:
    if not path.exists():
        return [], set()
    results = []
    completed: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                result = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Warning: ignoring malformed result line {line_number}: {exc}")
                continue
            key = (str(result.get("dataset", "")), str(result.get("question_id", "")))
            if not all(key):
                print(f"Warning: ignoring result line {line_number} without dataset/question_id")
                continue
            if result.get("status") == "error" or "error" in result:
                print(
                    f"Retrying prior API-error result {key[0]} {key[1]}: "
                    f"{result.get('error_message') or result.get('rationale', '')}",
                    flush=True,
                )
                continue
            if key in completed:
                continue
            results.append(result)
            completed.add(key)
    return results, completed


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the agentic RAG orchestrator")
    parser.add_argument("--pubmedqa", default="data/benchmark_subsets/pubmedqa.json")
    parser.add_argument("--bioasq", default="data/benchmark_subsets/bioasq.json")
    parser.add_argument("--output", default="data/eval_results/full_run.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N questions")
    parser.add_argument("--pubmedqa-limit", type=int, default=0)
    parser.add_argument("--bioasq-limit", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing output records")
    args = parser.parse_args()
    if any(value < 0 for value in (args.limit, args.pubmedqa_limit, args.bioasq_limit)):
        raise ValueError("limits cannot be negative")

    questions = load_questions(Path(args.pubmedqa), "pubmedqa")
    bioasq_questions = load_questions(Path(args.bioasq), "bioasq")
    if args.pubmedqa_limit:
        questions = questions[: args.pubmedqa_limit]
    if args.bioasq_limit:
        bioasq_questions = bioasq_questions[: args.bioasq_limit]
    questions.extend(bioasq_questions)
    if args.limit:
        questions = questions[: args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_results, completed = (
        ([], set()) if args.no_resume else load_existing_results(output_path)
    )
    pending_questions = [
        item for item in questions if (item["dataset"], item["question_id"]) not in completed
    ]
    if not args.no_resume and output_path.exists():
        existing_line_count = sum(1 for line in output_path.open("r", encoding="utf-8") if line.strip())
        if existing_line_count > len(existing_results):
            with output_path.open("w", encoding="utf-8") as handle:
                for result in existing_results:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(
                f"Removed {existing_line_count - len(existing_results):,} prior error/duplicate records "
                "before resuming.",
                flush=True,
            )
    orchestrator = Orchestrator() if pending_questions else None
    results = existing_results[:]
    print(
        f"Loaded {len(existing_results):,} existing results; "
        f"{len(pending_questions):,} questions pending -> {output_path}"
    )
    output_mode = "w" if args.no_resume else "a"
    with output_path.open(output_mode, encoding="utf-8") as handle:
        for index, item in enumerate(pending_questions, 1):
            assert orchestrator is not None
            result = evaluate_question(orchestrator, item)
            results.append(result)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{index}/{len(questions)}] {item['dataset']} {item['question_id']} "
                f"answer={result['our_answer']} correct={result['correct_answer']} "
                f"match={result['match']} time={result['elapsed_seconds']:.2f}s"
            )
    print_summary(results)


if __name__ == "__main__":
    main()