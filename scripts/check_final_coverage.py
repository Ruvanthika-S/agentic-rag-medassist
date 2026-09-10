#!/usr/bin/env python3
"""Check benchmark source-PMID coverage in the rebuilt pooled corpus."""

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_pmids(values: Any) -> set[str]:
    if not isinstance(values, list):
        values = [values] if values is not None else []
    return {str(value).strip() for value in values if str(value).strip()}


def load_corpus_pmids(path: Path) -> set[str]:
    pmids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            pmid = str(record.get("pmid") or "").strip()
            if pmid:
                pmids.add(pmid)
    return pmids


def check_dataset(path: Path, corpus_pmids: set[str]) -> tuple[int, int, set[str]]:
    benchmark = load_json(path)
    covered = 0
    missing_pmids: set[str] = set()

    for item in benchmark.values():
        source_pmids = normalize_pmids(item.get("PMID"))
        if source_pmids & corpus_pmids:
            covered += 1
        else:
            missing_pmids.update(source_pmids)

    return len(benchmark), covered, missing_pmids


def report(label: str, total: int, covered: int) -> None:
    percentage = covered / total * 100 if total else 0.0
    print(f"{label}: {covered:,}/{total:,} questions covered ({percentage:.2f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check final benchmark PMID coverage")
    parser.add_argument("--pubmedqa", default="data/benchmark_subsets/pubmedqa.json")
    parser.add_argument("--bioasq", default="data/benchmark_subsets/bioasq.json")
    parser.add_argument("--corpus", default="data/pooled_corpus/corpus.jsonl")
    args = parser.parse_args()

    corpus_pmids = load_corpus_pmids(Path(args.corpus))
    pubmedqa_total, pubmedqa_covered, pubmedqa_missing = check_dataset(
        Path(args.pubmedqa), corpus_pmids
    )
    bioasq_total, bioasq_covered, bioasq_missing = check_dataset(
        Path(args.bioasq), corpus_pmids
    )

    total = pubmedqa_total + bioasq_total
    covered = pubmedqa_covered + bioasq_covered

    print(f"Corpus PMIDs loaded: {len(corpus_pmids):,}")
    report("PubMedQA", pubmedqa_total, pubmedqa_covered)
    report("BioASQ-Y/N", bioasq_total, bioasq_covered)
    report("Total", total, covered)

    if pubmedqa_missing:
        print(f"PubMedQA source PMIDs missing from covered questions: {len(pubmedqa_missing):,}")
    if bioasq_missing:
        print(f"BioASQ source PMIDs missing from covered questions: {len(bioasq_missing):,}")


if __name__ == "__main__":
    main()