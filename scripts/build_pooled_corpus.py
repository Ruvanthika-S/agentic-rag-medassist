#!/usr/bin/env python3
"""Build a pooled evaluation corpus from BioASQ and PubMedQA source evidence.

Pools snippet/context text for MIRAGE benchmark questions, adds random PubMed
distractors from the existing pubmed_subset sample, and writes a deduplicated
corpus keyed by PMID (or MedRAG id for distractors without a real PMID).

Usage:
  python scripts/build_pooled_corpus.py
  python scripts/build_pooled_corpus.py --distractor-count 1500  # timing sample
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PUBMEDQA_URL = (
    "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"
)
PMID_FROM_DOCUMENT_RE = re.compile(
    r"(?:pubmed/|pubmed\.ncbi\.nlm\.nih\.gov/)(\d{5,9})(?:\D|$)",
    re.IGNORECASE,
)
NUMERIC_PMID_RE = re.compile(r"^\d{5,9}$")


class CorpusRecord:
    __slots__ = ("pmid", "texts", "sources", "tasks")

    def __init__(self) -> None:
        self.pmid = ""
        self.texts: set[str] = set()
        self.sources: set[str] = set()
        self.tasks: set[str] = set()

    def add_text(self, text: str) -> None:
        cleaned = " ".join(str(text or "").split())
        if cleaned:
            self.texts.add(cleaned)

    def merged_text(self) -> str:
        return "\n\n".join(sorted(self.texts, key=len, reverse=True))

    def merged_source_dataset(self) -> str:
        return ",".join(sorted(self.sources))

    def merged_source_task(self) -> str:
        return ",".join(sorted(self.tasks))

    def to_dict(self) -> dict[str, str]:
        return {
            "pmid": self.pmid,
            "text": self.merged_text(),
            "source_dataset": self.merged_source_dataset(),
            "source_task": self.merged_source_task(),
        }


def extract_pmid_from_document(document: Any) -> str | None:
    match = PMID_FROM_DOCUMENT_RE.search(str(document or "").strip())
    return match.group(1) if match else None


def resolve_record_pmid(record: dict[str, Any]) -> str | None:
    for key in ("pmid", "PMID"):
        value = str(record.get(key) or "").strip()
        if NUMERIC_PMID_RE.match(value):
            return value

    url_pmid = extract_pmid_from_document(record.get("url"))
    if url_pmid:
        return url_pmid

    record_id = str(record.get("id") or "").strip()
    if NUMERIC_PMID_RE.match(record_id):
        return record_id

    return None


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_pubmedqa_ori(path: Path) -> dict[str, Any]:
    if path.exists():
        print(f"Loading cached PubMedQA data from {path}")
        return load_json(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading PubMedQA ori_pqal.json -> {path}")
    try:
        with urllib.request.urlopen(PUBMEDQA_URL, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Failed to download PubMedQA data from {PUBMEDQA_URL}") from exc

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
        handle.write("\n")

    return payload


def load_raw_bioasq_questions(raw_dir: Path) -> dict[str, tuple[str, dict[str, Any]]]:
    questions: dict[str, tuple[str, dict[str, Any]]] = {}
    for path in sorted(raw_dir.rglob("*.json")):
        task_name = path.parent.name
        payload = load_json(path)
        for question in payload.get("questions", []):
            question_id = str(question.get("id") or "").strip()
            if question_id:
                questions[question_id] = (task_name, question)
    return questions


def pool_bioasq(
    benchmark: dict[str, Any],
    raw_questions: dict[str, tuple[str, dict[str, Any]]],
) -> dict[str, CorpusRecord]:
    pooled: dict[str, CorpusRecord] = {}
    missing_ids: list[str] = []
    non_yesno_ids: list[str] = []

    for benchmark_id in benchmark:
        raw_entry = raw_questions.get(str(benchmark_id))
        if raw_entry is None:
            missing_ids.append(str(benchmark_id))
            continue

        task_name, question = raw_entry
        if question.get("type") != "yesno":
            non_yesno_ids.append(str(benchmark_id))
            continue

        for snippet in question.get("snippets") or []:
            pmid = extract_pmid_from_document(snippet.get("document"))
            if not pmid:
                continue

            record = pooled.setdefault(pmid, CorpusRecord())
            record.pmid = pmid
            record.sources.add("bioasq")
            record.tasks.add(task_name)
            record.add_text(snippet.get("text"))

    if missing_ids:
        print(f"Warning: {len(missing_ids):,} BioASQ benchmark IDs not found in raw data")
    if non_yesno_ids:
        print(f"Warning: skipped {len(non_yesno_ids):,} non-yes/no BioASQ benchmark entries")

    return pooled


def pool_pubmedqa(
    benchmark: dict[str, Any],
    ori_pqal: dict[str, Any],
) -> dict[str, CorpusRecord]:
    pooled: dict[str, CorpusRecord] = {}
    missing_pmids: list[str] = []

    for item in benchmark.values():
        for raw_pmid in item.get("PMID") or []:
            pmid = str(raw_pmid).strip()
            entry = ori_pqal.get(pmid)
            if entry is None:
                missing_pmids.append(pmid)
                continue

            record = pooled.setdefault(pmid, CorpusRecord())
            record.pmid = pmid
            record.sources.add("pubmedqa")
            record.tasks.add("ori_pqal")
            for context in entry.get("CONTEXTS") or []:
                record.add_text(context)

    if missing_pmids:
        print(f"Warning: {len(missing_pmids):,} PubMedQA benchmark PMIDs missing from ori_pqal")

    return pooled


def merge_pools(
    bioasq_pool: dict[str, CorpusRecord],
    pubmedqa_pool: dict[str, CorpusRecord],
) -> dict[str, CorpusRecord]:
    merged: dict[str, CorpusRecord] = {}

    for pmid, record in bioasq_pool.items():
        merged_record = merged.setdefault(pmid, CorpusRecord())
        merged_record.pmid = pmid
        merged_record.texts.update(record.texts)
        merged_record.sources.update(record.sources)
        merged_record.tasks.update(record.tasks)

    for pmid, record in pubmedqa_pool.items():
        merged_record = merged.setdefault(pmid, CorpusRecord())
        merged_record.pmid = pmid
        merged_record.texts.update(record.texts)
        merged_record.sources.update(record.sources)
        merged_record.tasks.update(record.tasks)

    return merged


def build_distractor_text(record: dict[str, Any]) -> str:
    title = str(record.get("title") or "").strip()
    abstract = str(record.get("abstract") or "").strip()
    if title and abstract:
        return f"{title}\n\n{abstract}"
    return title or abstract


def load_distractor_candidates(pubmed_subset_path: Path) -> tuple[list[dict[str, Any]], int]:
    candidates: list[dict[str, Any]] = []
    skipped_no_text = 0

    with pubmed_subset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if not build_distractor_text(record):
                skipped_no_text += 1
                continue
            candidates.append(record)

    return candidates, skipped_no_text


def sample_distractors(
    candidates: list[dict[str, Any]],
    existing_pmids: set[str],
    count: int,
    seed: int,
) -> tuple[list[CorpusRecord], dict[str, int]]:
    rng = random.Random(seed)
    shuffled = candidates[:]
    rng.shuffle(shuffled)

    selected: list[CorpusRecord] = []
    stats = Counter()
    seen_distractor_pmids: set[str] = set()

    for record in shuffled:
        if len(selected) >= count:
            break

        real_pmid = resolve_record_pmid(record)
        if real_pmid:
            pmid = real_pmid
            stats["real_pmid"] += 1
            if pmid in existing_pmids:
                stats["skipped_overlap_with_pooled"] += 1
                continue
        else:
            pmid = str(record.get("id") or "").strip()
            stats["medrag_id_as_pmid"] += 1
            if not pmid:
                stats["skipped_missing_id"] += 1
                continue

        if pmid in seen_distractor_pmids:
            stats["skipped_duplicate_distractor"] += 1
            continue

        seen_distractor_pmids.add(pmid)
        corpus_record = CorpusRecord()
        corpus_record.pmid = pmid
        corpus_record.sources.add("distractor")
        corpus_record.tasks.add("pubmed_subset")
        corpus_record.add_text(build_distractor_text(record))
        selected.append(corpus_record)

    stats["selected"] = len(selected)
    return selected, dict(stats)


def write_corpus(records: list[CorpusRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def validate_unique_pmids(records: list[CorpusRecord]) -> tuple[bool, int]:
    pmids = [record.pmid for record in records]
    duplicate_count = len(pmids) - len(set(pmids))
    return duplicate_count == 0, duplicate_count


def print_summary(records: list[CorpusRecord], elapsed_seconds: float) -> None:
    source_counts = Counter(record.merged_source_dataset() for record in records)
    task_counts = Counter(record.merged_source_task() for record in records)

    print("\n=== Pooled Corpus Summary ===")
    print(f"Total documents: {len(records):,}")
    print(f"Elapsed time: {elapsed_seconds:.2f}s")
    print("Breakdown by source_dataset:")
    for source in sorted(source_counts):
        print(f"  {source}: {source_counts[source]:,}")

    print("Breakdown by source_task (top 10):")
    for task, count in task_counts.most_common(10):
        print(f"  {task}: {count:,}")

    unique_ok, duplicate_count = validate_unique_pmids(records)
    if unique_ok:
        print("Duplicate PMIDs in final file: none")
    else:
        print(f"Duplicate PMIDs in final file: {duplicate_count:,} (ERROR)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pooled evaluation corpus with distractors")
    parser.add_argument("--bioasq-benchmark", default="data/benchmark_subsets/bioasq.json")
    parser.add_argument("--pubmedqa-benchmark", default="data/benchmark_subsets/pubmedqa.json")
    parser.add_argument("--bioasq-raw-dir", default="mirage_repo/rawdata/bioasq")
    parser.add_argument("--pubmedqa-cache", default="data/pubmedqa/ori_pqal.json")
    parser.add_argument("--pubmed-subset", default="data/pubmed_subset/pubmed_subset.jsonl")
    parser.add_argument("--output", default="data/pooled_corpus/corpus.jsonl")
    parser.add_argument("--distractor-count", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-distractors",
        action="store_true",
        help="Build only the benchmark-aligned pooled documents (no distractors)",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=0,
        help="Optional cap on total output documents (useful for timing samples)",
    )
    args = parser.parse_args()

    started = time.perf_counter()

    bioasq_benchmark = load_json(Path(args.bioasq_benchmark))
    pubmedqa_benchmark = load_json(Path(args.pubmedqa_benchmark))
    raw_questions = load_raw_bioasq_questions(Path(args.bioasq_raw_dir))
    ori_pqal = ensure_pubmedqa_ori(Path(args.pubmedqa_cache))

    print(f"BioASQ benchmark questions: {len(bioasq_benchmark):,}")
    print(f"PubMedQA benchmark questions: {len(pubmedqa_benchmark):,}")
    print(f"Raw BioASQ questions loaded: {len(raw_questions):,}")

    bioasq_pool = pool_bioasq(bioasq_benchmark, raw_questions)
    pubmedqa_pool = pool_pubmedqa(pubmedqa_benchmark, ori_pqal)
    pooled = merge_pools(bioasq_pool, pubmedqa_pool)

    print(f"Pooled benchmark documents (unique PMIDs): {len(pooled):,}")
    print(f"  BioASQ-only PMIDs: {sum(1 for r in pooled.values() if r.sources == {'bioasq'}):,}")
    print(
        f"  PubMedQA-only PMIDs: {sum(1 for r in pooled.values() if r.sources == {'pubmedqa'}):,}"
    )
    print(
        "  Overlap PMIDs: "
        f"{sum(1 for r in pooled.values() if r.sources == {'bioasq', 'pubmedqa'}):,}"
    )

    final_records = list(pooled.values())

    if not args.skip_distractors:
        pubmed_subset_path = Path(args.pubmed_subset)
        if not pubmed_subset_path.exists():
            raise FileNotFoundError(f"PubMed subset not found: {pubmed_subset_path}")

        print(f"Loading distractor candidates from {pubmed_subset_path}")
        candidates, skipped_no_text = load_distractor_candidates(pubmed_subset_path)
        print(f"Distractor candidates with text: {len(candidates):,} (skipped empty: {skipped_no_text:,})")

        distractors, distractor_stats = sample_distractors(
            candidates=candidates,
            existing_pmids=set(pooled.keys()),
            count=args.distractor_count,
            seed=args.seed,
        )
        print("Distractor sampling stats:")
        for key, value in sorted(distractor_stats.items()):
            print(f"  {key}: {value:,}")

        if distractor_stats.get("real_pmid", 0) == 0:
            print(
                "Note: pubmed_subset records do not contain real PMIDs; "
                "using MedRAG ids as pmid for distractors and skipping PMID overlap checks."
            )

        if distractor_stats.get("selected", 0) < args.distractor_count:
            print(
                f"Warning: requested {args.distractor_count:,} distractors but only selected "
                f"{distractor_stats.get('selected', 0):,}"
            )

        final_records.extend(distractors)

    if args.max_documents and len(final_records) > args.max_documents:
        print(
            f"Capping output from {len(final_records):,} to {args.max_documents:,} documents "
            "(timing sample mode)"
        )
        final_records = final_records[: args.max_documents]

    output_path = Path(args.output)
    write_corpus(final_records, output_path)
    print(f"Wrote corpus to {output_path}")

    elapsed = time.perf_counter() - started
    print_summary(final_records, elapsed)


if __name__ == "__main__":
    main()
