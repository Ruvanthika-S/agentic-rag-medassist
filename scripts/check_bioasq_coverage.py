#!/usr/bin/env python3
"""Measure BioASQ yes/no source-PMID coverage in the local PubMed corpus."""

import argparse
import json
import re
from pathlib import Path
from typing import Any


CORPUS_URL_PMID_RE = re.compile(
    r"https?://pubmed\.ncbi\.nlm\.nih\.gov/(\d+)(?:/)?(?:[?#].*)?$",
    re.IGNORECASE,
)


def normalized_text(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()


def pmids_from_document(document: Any) -> set[str]:
    value = str(document or "").strip()
    match = re.search(r"(?:pubmed/|pmid[=:]?)(\d{5,9})(?:\D|$)", value, re.IGNORECASE)
    if match:
        return {match.group(1)}
    if value.isdigit():
        return {value}
    return set()


def raw_questions(raw_dir: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    by_id: dict[str, dict] = {}
    by_text: dict[str, dict] = {}
    for path in sorted(raw_dir.rglob("*.json")):
        with path.open("r", encoding="utf-8") as raw_file:
            payload = json.load(raw_file)
        for question in payload.get("questions", []):
            if question.get("id"):
                by_id[str(question["id"])] = question
            if question.get("body"):
                by_text[normalized_text(question["body"])] = question
    return by_id, by_text


def corpus_pmids(corpus_path: Path) -> tuple[set[str], list[str]]:
    present: set[str] = set()
    samples: list[str] = []
    with corpus_path.open("r", encoding="utf-8") as corpus_file:
        for line in corpus_file:
            record = json.loads(line)
            match = CORPUS_URL_PMID_RE.match(str(record.get("url") or "").strip())
            if match:
                pmid = match.group(1)
                present.add(pmid)
                if pmid not in samples and len(samples) < 3:
                    samples.append(pmid)
    return present, samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="mirage_repo/benchmark.json")
    parser.add_argument("--raw-dir", default="mirage_repo/rawdata/bioasq")
    parser.add_argument("--corpus", default="data/pubmed_subset/pubmed_subset.jsonl")
    args = parser.parse_args()

    with Path(args.benchmark).open("r", encoding="utf-8") as benchmark_file:
        bioasq = json.load(benchmark_file)["bioasq"]
    raw_by_id, raw_by_text = raw_questions(Path(args.raw_dir))
    available_pmids, sample_pmids = corpus_pmids(Path(args.corpus))

    raw_matches = 0
    questions_with_source_pmid = 0
    covered_questions = 0
    unmatched_ids: list[str] = []
    uncovered_ids: list[str] = []

    for benchmark_id, item in bioasq.items():
        raw = raw_by_id.get(str(benchmark_id)) or raw_by_text.get(normalized_text(item.get("question")))
        if raw is None:
            unmatched_ids.append(str(benchmark_id))
            continue
        raw_matches += 1
        source_pmids = set().union(*(pmids_from_document(document) for document in raw.get("documents", [])))
        if source_pmids:
            questions_with_source_pmid += 1
        if source_pmids & available_pmids:
            covered_questions += 1
        else:
            uncovered_ids.append(str(benchmark_id))

    total = len(bioasq)
    percentage = (covered_questions / total * 100) if total else 0.0
    print(f"Sample extracted corpus PMIDs ({len(sample_pmids)} available): {sample_pmids}")
    print(f"BioASQ-Y/N questions: {total:,}")
    print(f"Matched raw BioASQ entries: {raw_matches:,}/{total:,}")
    print(f"Questions with a PMID in documents: {questions_with_source_pmid:,}/{total:,}")
    print(f"Questions with a source PMID in corpus: {covered_questions:,}/{total:,} ({percentage:.2f}%)")
    if unmatched_ids:
        print(f"Unmatched benchmark IDs: {', '.join(unmatched_ids)}")
    if uncovered_ids:
        print(f"Questions without corpus coverage: {len(uncovered_ids):,}")


if __name__ == "__main__":
    main()