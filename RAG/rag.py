import os
import glob
import json
import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "intfloat/e5-large-v2"

def _safe_str(x):
    if x is None:
        return ""
    return str(x)

def load_db_from_folder(folder: str):
    """
    Load all JSON files from a folder.
    Each JSON contains either a dict or list[dict] records. Each record must include:
      - embedding : list[float]
    """
    all_records = []
    pattern = os.path.join(folder, "*.json")
    files = sorted(glob.glob(pattern))

    for path in files:
        print(f"  Loading {path} ...")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                data = [data]
            for rec in data:
                if "embedding" not in rec:
                    raise ValueError(f"Record in {path} missing 'embedding' (id={rec.get('id')})")
                all_records.append(rec)

    print(f"Total records loaded: {len(all_records)}")
    if len(all_records) == 0:
        return [], np.zeros((0, 1), dtype=np.float32)

    emb_matrix = np.array([r["embedding"] for r in all_records], dtype=np.float32)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    emb_matrix_norm = emb_matrix / np.clip(norms, 1e-9, None)
    return all_records, emb_matrix_norm

def build_context(results, max_chars: int = 4000) -> str:
    pieces = []
    for r in results:
        header = (
            f"--- Source: {r['id']} "
            f"(page {r['page']}, chapter={r['chapter']}, "
            f"section={r['section']}, subsection={r['subsection']}) ---\n"
        )
        pieces.append(header + _safe_str(r.get("text", "")) + "\n")

    ctx = "\n".join(pieces)
    if len(ctx) > max_chars:
        ctx = ctx[:max_chars] + "\n...[truncated]..."
    return ctx

class RAGSearch:
    def __init__(self, chunks_folder: str, embed_model_name: str = EMBED_MODEL_NAME):
        print(f"Loading embedding model: {embed_model_name}")
        self.model = SentenceTransformer(embed_model_name)  # add device="cuda" if you want

        self.db, self.emb_norm = load_db_from_folder(chunks_folder)
        if len(self.db) == 0:
            raise RuntimeError(f"Chunks DB empty. Put JSON chunk files with embeddings in: {chunks_folder}")

    def embed_query(self, text: str) -> np.ndarray:
        vec = self.model.encode(text, convert_to_numpy=True).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec if norm == 0 else (vec / norm)

    def search(self, query_text: str, top_k: int):
        q = self.embed_query(query_text)  # (D,)
        sims = self.emb_norm @ q          # (N,)
        top_idx = np.argsort(-sims)[:top_k]

        results = []
        for idx in top_idx:
            rec = self.db[idx]
            results.append(
                {
                    "id": rec.get("id"),
                    "score": float(sims[idx]),
                    "page": rec.get("page"),
                    "chapter": rec.get("chapter"),
                    "section": rec.get("section", rec.get("section_num")),
                    "subsection": rec.get("subsection", rec.get("subsection_num")),
                    "source": rec.get("source"),
                    "book_title": rec.get("book_title"),
                    "text": rec.get("text", ""),
                }
            )
        return results

