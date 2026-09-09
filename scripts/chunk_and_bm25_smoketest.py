#!/usr/bin/env python3
"""Chunk pooled corpus JSONL into overlapping passages and run a BM25 smoke test.

Produces:
 - <output-dir>/pooled_chunks.jsonl (one chunk per line with metadata)
 - <output-dir>/bm25_index.pkl (pickle with {'bm25': bm25, 'docs': docs, 'metas': metas})

Run:
    python scripts/chunk_and_bm25_smoketest.py

"""
import argparse
import json
import os
import pickle
import re
from typing import List

from rank_bm25 import BM25Okapi
from tqdm import tqdm


WORD_RE = re.compile(r"\w+")


def tokenize(text: str) -> List[str]:
    return WORD_RE.findall(text.lower())


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50):
    words = tokenize(text)
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        chunks.append((start, end, chunk_text))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/pooled_corpus/corpus.jsonl")
    parser.add_argument("--output-dir", default="data/pooled_chunks")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_chunks = os.path.join(args.output_dir, "pooled_chunks.jsonl")
    bm25_pickle = os.path.join(args.output_dir, "bm25_index.pkl")

    docs_text = []
    metas = []

    print(f"Reading input {args.input}")
    with open(args.input, "r", encoding="utf-8") as f_in, open(out_chunks, "w", encoding="utf-8") as f_out:
        for line in tqdm(f_in):
            item = json.loads(line)
            pmid = str(item.get("pmid") or "")
            source_dataset = str(item.get("source_dataset") or "")
            combined = str(item.get("text") or "").strip()
            chunked = chunk_text(combined, chunk_size=args.chunk_size, overlap=args.overlap)
            for idx, (s, e, chunk_text_str) in enumerate(chunked):
                meta = {
                    "pmid": pmid,
                    "source_dataset": source_dataset,
                    "chunk_id": idx,
                    "start_word": s,
                    "end_word": e,
                }
                out = {"text": chunk_text_str, "meta": meta}
                f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
                docs_text.append(chunk_text_str)
                metas.append(meta)

    print(f"Wrote chunks to {out_chunks} (total chunks: {len(docs_text):,})")

    print("Building BM25 index")
    tokenized = [tokenize(d) for d in docs_text]
    bm25 = BM25Okapi(tokenized)

    print(f"Saving BM25 index to {bm25_pickle}")
    with open(bm25_pickle, "wb") as pf:
        pickle.dump({"bm25": bm25, "docs": docs_text, "metas": metas}, pf)

    # Smoke test queries
    queries = [
        "What are the side effects of aspirin?",
        "Treatment for hypertension in adults",
        "Symptoms of diabetes"
    ]

    for q in queries:
        q_tok = tokenize(q)
        scores = bm25.get_scores(q_tok)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:5]
        print(f"\nQuery: {q}")
        for rank, (doc_idx, score) in enumerate(ranked, start=1):
            meta = metas[doc_idx]
            snippet = docs_text[doc_idx][:300].replace('\n', ' ')
            print(f"{rank}. score={score:.2f} pmid={meta['pmid']} source={meta['source_dataset']} chunk={meta['chunk_id']}\n   {snippet}...\n")


if __name__ == "__main__":
    main()
