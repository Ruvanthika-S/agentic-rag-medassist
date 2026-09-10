#!/usr/bin/env python3
"""Load a manageable subset from the MedRAG/pubmed HF dataset and save as JSONL.

Usage example:
  python scripts/load_pubmed_subset.py --max-snippets 40000 --output-dir data/pubmed_subset

This script streams the Hugging Face dataset and writes up to --max-snippets
examples that pass a minimal filtering (have an abstract/title and min length).
"""

"""Why does the command have those extra bits like --max-snippets 40000?
Those are called arguments — they let you reuse the same script with different settings without rewriting the code each time. Like a form with blank fields: same form, different values each time you fill it in. So today you ran it for 40,000 — tomorrow, if you wanted 10,000 instead, you'd just change that one number in the command, not touch the code at all."""
import argparse
import json
import os
import random
from typing import Iterable

from datasets import load_dataset




def normalize_example(example: dict) -> dict:
    # Try common field names used in various PubMed exports
    pmid = example.get("PMID") or example.get("pmid") or example.get("paper_id") or example.get("pmcid")
    snippet_id = example.get("id") or pmid
    title = example.get("title") or example.get("paper_title") or ""
    abstract = (
        example.get("abstract")
        or example.get("content")
        or example.get("contents")
        or example.get("summary")
        or example.get("text")
        or example.get("full_text")
        or ""
    )
    journal = example.get("journal") or ""
    year = example.get("year") or example.get("pub_year") or example.get("publication_year") or ""
    url = example.get("url") or example.get("source_url") or (
        f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
    )
    return {
        "id": snippet_id,
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "journal": journal,
        "year": year,
        "url": url,
        "raw": example,
    }


def stream_dataset(dataset_name: str):
    hf_token = (
        os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HF_TOKEN")
    )

    print(f"Loading dataset: {dataset_name}")

    try:
        return load_dataset(
            dataset_name,
            split="train",
            streaming=True,
            token=hf_token if hf_token else None,
        )

    except Exception as e:
        print("\nERROR while loading dataset")
        print(type(e).__name__)
        print(e)
        raise


def matches_keywords(text: str, keywords: list[str]) -> bool:
    if not keywords:
        return True
    text_l = text.lower()
    for k in keywords:
        if k.lower() in text_l:
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="MedRAG/pubmed", help="Hugging Face dataset id")
    parser.add_argument("--max-snippets", type=int, default=40000)
    parser.add_argument("--min-abstract-words", type=int, default=50)
    parser.add_argument("--output-dir", default="data/pubmed_subset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keywords", help="Comma-separated keywords to filter (optional)")
    parser.add_argument("--stream-max", type=int, default=0, help="Optional: stop after streaming this many raw items (for testing)")
    parser.add_argument("--debug-samples", type=int, default=10, help="How many rejected samples to save for inspection")
    args = parser.parse_args()

    random.seed(args.seed)
    keywords = [k.strip() for k in args.keywords.split(",")] if args.keywords else []

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "pubmed_subset.jsonl")

    print(f"Streaming dataset {args.dataset} (streaming=True)")
    iterable = stream_dataset(args.dataset)

    # Reservoir sampling for an unbiased random sample across the full stream
    reservoir = []
    processed = 0
    candidates = 0
    kept = 0
    rejected_samples = []

    for item in iterable:
        processed += 1
        if processed % 500000 == 0:
            print(f"Streamed {processed:,} items, reservoir size {len(reservoir):,}")

        try:
            ex = normalize_example(item)
        except Exception:
            # include raw item in rejected samples if possible
            if len(rejected_samples) < args.debug_samples:
                rejected_samples.append({"reason": "normalize_exception", "raw": item})
            continue

        abstract = ex.get("abstract") or ""
        title = ex.get("title") or ""

        # Normalize abstract if it's a list or nested structure
        if isinstance(abstract, list):
            # If list of strings, join; if list of dicts, try to extract text-like fields
            pieces = []
            for part in abstract:
                if isinstance(part, str):
                    pieces.append(part)
                elif isinstance(part, dict):
                    # common keys
                    for k in ("text", "passage", "abstract", "body"):
                        if k in part and isinstance(part[k], str):
                            pieces.append(part[k])
                            break
            abstract = " \n ".join(pieces)
        elif isinstance(abstract, dict):
            # try to pull a text field
            for k in ("text", "passage", "abstract", "body", "summary"):
                if k in abstract and isinstance(abstract[k], str):
                    abstract = abstract[k]
                    break

        # Ensure abstract is a string
        if not isinstance(abstract, str):
            abstract = str(abstract)

        # Safe word count
        try:
            abstract_words = len(abstract.split())
        except Exception:
            abstract_words = 0

        if abstract_words < args.min_abstract_words:
            if len(rejected_samples) < args.debug_samples:
                rejected_samples.append({"reason": "short_abstract", "abstract_preview": abstract[:200], "raw": ex.get("raw")})
            continue

        if keywords:
            if not (matches_keywords(title, keywords) or matches_keywords(abstract, keywords)):
                if len(rejected_samples) < args.debug_samples:
                    rejected_samples.append({"reason": "no_keyword_match", "title": title, "abstract_preview": abstract[:200], "raw": ex.get("raw")})
                continue

        # Candidate passed all filters
        candidates += 1

        row = {
            "title": title,
            "abstract": abstract,
            "id": ex.get("id"),
            "pmid": ex.get("pmid"),
            "journal": ex.get("journal"),
            "year": ex.get("year"),
            "url": ex.get("url"),
        }

        # Reservoir sampling logic (use candidate count)
        if len(reservoir) < args.max_snippets:
            reservoir.append(row)
        else:
            # i is count of candidates seen so far (1-based)
            i = candidates
            # choose a random index in [0, i-1]
            j = random.randrange(i)
            if j < args.max_snippets:
                reservoir[j] = row

        # Optional early stop for very large runs (useful for testing)
        if args.stream_max and processed >= args.stream_max:
            print(f"Reached stream max {args.stream_max:,}, stopping early")
            break

    selected = reservoir
    kept = len(selected)

    print(f"Finished streaming. Processed {processed:,} raw items, candidates {candidates:,}, kept {kept:,} examples")

    # write rejected samples for inspection
    if rejected_samples:
        rej_path = os.path.join(args.output_dir, "rejected_samples.jsonl")
        print(f"Writing {len(rejected_samples):,} rejected samples to {rej_path}")
        with open(rej_path, "w", encoding="utf-8") as rf:
            for r in rejected_samples:
                rf.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Writing {len(selected):,} items to {out_path}")
    with open(out_path, "w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Done.")


if __name__ == "__main__":
    main()
