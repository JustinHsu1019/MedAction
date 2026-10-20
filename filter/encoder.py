"""
encoder.py — MedCPT encoder + FAISS index construction.

Device:
  - Default CUDA (override to "cpu" if no GPU)
  - Override via env var:  MEDCPT_DEVICE=cpu python filter.py
"""
import os
import numpy as np
import torch
import faiss
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel


# ── Device setting (change here) ─────────────────────────────
DEVICE = os.environ.get("MEDCPT_DEVICE", "cuda")   # "cpu" or "cuda"


class MedCPTEncoder:
    def __init__(self, device=None):
        self.device = device or DEVICE
        if self.device == "cuda" and not torch.cuda.is_available():
            print("[warn] CUDA unavailable, falling back to CPU")
            self.device = "cpu"
        print(f"Initializing MedCPT on: {self.device.upper()}")
        self.tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
        self.model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts, batch_size=256):
        all_embeds = []
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Encoding ({self.device.upper()})"):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=128, return_tensors="pt"
            ).to(self.device)
            outputs = self.model(**inputs)
            emb = torch.nn.functional.normalize(
                outputs.last_hidden_state[:, 0, :], p=2, dim=1
            )
            all_embeds.append(emb.cpu().numpy())
        return np.vstack(all_embeds)


def load_nodes_with_faiss(csv_path, encoder, cache_suffix="_medcpt_emb.npy", max_rows=None):
    """Load a node CSV into a DataFrame and build a FAISS index over it."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    print(f"Reading: {os.path.basename(csv_path)}...")

    df = None
    for s in [',', '\t']:
        try:
            temp_df = pd.read_csv(csv_path, sep=s, dtype=str, on_bad_lines="skip", nrows=5)
            if temp_df.shape[1] > 1:
                df = pd.read_csv(csv_path, sep=s, dtype=str, on_bad_lines="skip", engine="c")
                break
        except Exception:
            continue

    if df is None:
        df = pd.read_csv(csv_path, sep=None, dtype=str, on_bad_lines="skip", engine="python")

    df = df.fillna("")
    df.columns = [c.strip().lower() for c in df.columns]

    id_candidates   = ["node_id", "id", "concept_id", "entity_id", "key"]
    name_candidates = ["node_name", "name", "label", "concept_name", "preferred_name"]

    id_col   = next((c for c in id_candidates if c in df.columns), df.columns[0])
    name_col = next((c for c in name_candidates if c in df.columns), None)
    if name_col is None:
        name_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    if max_rows:
        df = df.iloc[:max_rows].copy()

    texts      = df[name_col].tolist()
    cache_path = csv_path.replace(".csv", cache_suffix)

    if os.path.exists(cache_path):
        print(f"Loading cache: {os.path.basename(cache_path)}")
        embeds = np.load(cache_path)
        if len(embeds) != len(texts):
            print("[warn] cache length mismatch, re-encoding...")
            embeds = encoder.encode(texts, batch_size=256)
            np.save(cache_path, embeds.astype("float32"))
    else:
        print(f"Encoding {len(texts)} items...")
        embeds = encoder.encode(texts, batch_size=256)
        np.save(cache_path, embeds.astype("float32"))

    dim   = embeds.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeds.astype("float32"))

    print(f"{os.path.basename(csv_path)} loaded | {len(df)} nodes\n")
    return df, index, id_col, name_col
