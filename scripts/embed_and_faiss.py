#!/usr/bin/env python3
"""Compute dense embeddings for chunks and build a FAISS index.

Usage:
  python scripts/embed_and_faiss.py \
    --input data/pubmed_chunks/pubmed_chunks.jsonl \
    --model <HF-model-id-for-MedCPT-or-encoder> \
    --output-dir data/faiss_index \
    --batch-size 32

Notes:
- This uses a transformer encoder and mean-pools last_hidden_state over tokens.
- Works on CPU if no GPU is available (slower).
"""
import argparse
import json
import os
from typing import List

import numpy as np
import torch
from tqdm import tqdm

try:
    import faiss
except Exception:
    faiss = None

from transformers import AutoTokenizer, AutoModel


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(1)
    counts = mask.sum(1).clamp(min=1e-9)
    return summed / counts


def load_chunks(path: str, max_chunks: int = 0):
    texts = []
    metas = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_chunks and i >= max_chunks:
                break
            obj = json.loads(line)
            texts.append(obj.get("text") or obj.get("chunk") or "")
            metas.append(obj.get("meta") or {})
    return texts, metas


def compute_embeddings(texts: List[str], model_name: str, batch_size: int, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # Respect tokenizer/model maximum sequence length to avoid expansion errors
    # Use safe, explicit max length for BERT-based encoders
    max_len = 512

    all_embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
            batch = texts[i : i + batch_size]
            # Explicitly enforce truncation to 512 tokens to avoid model errors
            try:
                enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
            except Exception:
                # Fallback: tokenize per sample with truncation and pad manually
                small_encs = [tokenizer(t, padding=True, truncation=True, return_tensors="pt", max_length=512) for t in batch]
                input_ids = torch.nn.utils.rnn.pad_sequence([e["input_ids"].squeeze(0) for e in small_encs], batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
                attention_mask = torch.nn.utils.rnn.pad_sequence([e["attention_mask"].squeeze(0) for e in small_encs], batch_first=True, padding_value=0).to(device)
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                last_hidden = out.last_hidden_state
                pooled = mean_pool(last_hidden, attention_mask)
                emb = pooled.cpu().numpy()
                all_embs.append(emb)
                continue
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = out.last_hidden_state
            pooled = mean_pool(last_hidden, attention_mask)
            emb = pooled.cpu().numpy()
            all_embs.append(emb)
    return np.vstack(all_embs)


def l2_normalize(a: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return a / norms


def build_faiss_index(embs: np.ndarray, index_path: str):
    if faiss is None:
        raise RuntimeError("faiss is not installed; install faiss-cpu or faiss-gpu")
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    faiss.write_index(index, index_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/pubmed_chunks/pubmed_chunks.jsonl")
    parser.add_argument("--model", required=True, help="Hugging Face model id for MedCPT or other encoder")
    parser.add_argument("--output-dir", default="data/faiss_index")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-chunks", type=int, default=0, help="Optional: limit number of chunks for quick test")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("Loading chunks...")
    texts, metas = load_chunks(args.input, max_chunks=args.max_chunks)
    print(f"Loaded {len(texts):,} chunks")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Computing embeddings (this may take a while on CPU)")
    embs = compute_embeddings(texts, args.model, args.batch_size, device)
    print("Normalizing embeddings")
    embs = l2_normalize(embs)

    emb_path = os.path.join(args.output_dir, "embeddings.npy")
    meta_path = os.path.join(args.output_dir, "metas.jsonl")
    index_path = os.path.join(args.output_dir, "faiss_index.idx")

    print(f"Saving embeddings to {emb_path}")
    np.save(emb_path, embs)

    print(f"Saving metas to {meta_path}")
    with open(meta_path, "w", encoding="utf-8") as mf:
        for m in metas:
            mf.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"Building FAISS index (will be saved to {index_path})")
    build_faiss_index(embs.astype(np.float32), index_path)

    print("Done. Index and artifacts are in:", args.output_dir)


if __name__ == "__main__":
    main()
