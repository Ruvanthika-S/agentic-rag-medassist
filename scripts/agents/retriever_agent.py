#!/usr/bin/env python3
"""Hybrid BM25 and MedCPT retriever for the pooled biomedical corpus."""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from scripts.embed_and_faiss import mean_pool


class RetrieverAgent:
    """Retrieve pooled chunks using weighted reciprocal-rank fusion."""

    def __init__(
        self,
        bm25_index_path: str = "data/pooled_chunks/bm25_index.pkl",
        faiss_index_path: str = "data/pooled_faiss_index/faiss_index.idx",
        dense_metadata_path: str = "data/pooled_faiss_index/metas.jsonl",
        model_name: str = "ncbi/MedCPT-Article-Encoder",
        device: str | None = None,
        rrf_k: int = 60,
        bm25_weight: float = 0.5,
    ) -> None:
        if not 0 <= bm25_weight <= 1:
            raise ValueError("bm25_weight must be between 0 and 1")

        with Path(bm25_index_path).open("rb") as handle:
            payload = pickle.load(handle)
        self.bm25 = payload["bm25"]
        self.docs: list[str] = payload["docs"]
        self.metas: list[dict[str, Any]] = payload["metas"]
        self.faiss_index = faiss.read_index(faiss_index_path)
        self.dense_metas = self._load_jsonl(dense_metadata_path)

        if len(self.docs) != self.faiss_index.ntotal or len(self.docs) != len(self.metas):
            raise ValueError("BM25 and FAISS artifacts do not have matching lengths")
        if len(self.dense_metas) != len(self.docs):
            raise ValueError("Dense metadata does not align with the index")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.dense_weight = 1.0 - bm25_weight

    @staticmethod
    def _load_jsonl(path: str) -> list[dict[str, Any]]:
        with Path(path).open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle]

    def _embed_query(self, query: str) -> np.ndarray:
        encoded = self.tokenizer(
            [query], padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            output = self.model(**encoded)
            embedding = mean_pool(output.last_hidden_state, encoded["attention_mask"])
        vector = embedding.cpu().numpy().astype("float32")
        vector /= np.maximum(np.linalg.norm(vector, axis=1, keepdims=True), 1e-12)
        return vector

    def retrieve(self, query: str, top_k: int = 5) -> dict[str, Any]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        bm25_tokens = re.findall(r"\w+", query.lower())
        bm25_scores = np.asarray(self.bm25.get_scores(bm25_tokens))
        candidate_count = min(len(self.docs), max(top_k * 10, top_k))
        bm25_ids = np.argsort(-bm25_scores)[:candidate_count]
        dense_scores_full, dense_ids_full = self.faiss_index.search(
            self._embed_query(query), candidate_count
        )
        bm25_rank = {int(index): rank for rank, index in enumerate(bm25_ids, 1)}
        dense_rank = {int(index): rank for rank, index in enumerate(dense_ids_full[0], 1)}
        raw_dense_scores = {int(index): float(score) for index, score in zip(dense_ids_full[0], dense_scores_full[0])}
        candidates = set(bm25_rank) | set(dense_rank)

        fused = []
        for index in candidates:
            score = 0.0
            if index in bm25_rank:
                score += self.bm25_weight / (self.rrf_k + bm25_rank[index])
            if index in dense_rank:
                score += self.dense_weight / (self.rrf_k + dense_rank[index])
            fused.append((score, index))
        fused.sort(reverse=True)

        results = []
        for combined_score, index in fused[:top_k]:
            meta = self.metas[index]
            results.append(
                {
                    "chunk_text": self.docs[index],
                    "pmid": meta.get("pmid", self.dense_metas[index].get("pmid", "")),
                    "source_dataset": meta.get(
                        "source_dataset", self.dense_metas[index].get("source_dataset", "")
                    ),
                    "bm25_score": float(bm25_scores[index]),
                    "faiss_score": raw_dense_scores.get(index, 0.0),
                    "combined_score": float(combined_score),
                }
            )

        max_rrf_score = 1.0 / (self.rrf_k + 1)
        confidence = results[0]["combined_score"] / max_rrf_score if results else 0.0
        return {"results": results, "retrieval_confidence": float(min(confidence, 1.0))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hybrid retrieval against the pooled indexes")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    response = RetrieverAgent().retrieve(args.query, top_k=args.top_k)
    print(json.dumps(response, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()