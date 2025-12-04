import os
import json
import glob
import argparse

import numpy as np
import pandas as pd
from scipy.io import savemat
from sentence_transformers import SentenceTransformer
from llm import LLM

# ===========================
# CONFIG
# ===========================
CHUNKS_FOLDER = "./Knowledge_Base/Chunks"
MODEL_NAME = "thenlper/gte-large"
TOP_K_DEFAULT = 15

print(f"Loading embedding model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)  # add device="cuda" if you have GPU

def _safe_str(x):
    """Convert None to empty string, otherwise to string."""
    if x is None:
        return ""
    return str(x)


def load_db_from_folder(folder: str):
    """
    Load all JSON files from a folder.
    Each file is expected to contain a list of chunk dicts, each with an 'embedding' key.
    Returns:
        db: list[dict]
        emb_matrix_norm: np.ndarray of shape (N, D), normalized embeddings
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
                    raise ValueError(
                        f"Record in {path} missing 'embedding' key (id={rec.get('id')})"
                    )
                all_records.append(rec)

    print(f"Total records loaded: {len(all_records)}")

    emb_matrix = np.array([r["embedding"] for r in all_records], dtype=np.float32)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    emb_matrix_norm = emb_matrix / np.clip(norms, 1e-9, None)

    return all_records, emb_matrix_norm


def embed_query(text: str) -> np.ndarray:
    """
    Embed a query string using thenlper/gte-large and L2 normalize.
    """
    vec = model.encode(text, convert_to_numpy=True)
    vec = vec.astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def search(query_text: str, top_k: int = TOP_K_DEFAULT):
    """
    Search the loaded DB for the top_k most similar chunks to the query_text.
    Returns a list of dicts:
      {id, score, page, chapter, text, source, book_title}
    """
    if len(DB) == 0:
        raise RuntimeError(
            "Database is empty. Did you put JSON files with embeddings in the folder?"
        )

    q = embed_query(query_text)  # shape (D,)
    sims = EMB_MATRIX_NORM @ q  # shape (N,)

    top_idx = np.argsort(-sims)[:top_k]

    results = []
    for idx in top_idx:
        rec = DB[idx]
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


def signal_params_to_query(params: dict) -> str:
    """
    Convert a signal configuration (e.g. from Excel row) into a rich semantic query.
    Expected keys:
      Standard, phyFormat, Modulation, CodingRate, MCS,
      Bandwidth, Sampling Rate (Hz), FFT Size
    """
    standard = params.get("Standard", "11g")
    phy = params.get("phyFormat", "")
    mod = params.get("Modulation", "")
    code = params.get("CodingRate", "")
    mcs = params.get("MCS", "")
    bw = params.get("Bandwidth", "")
    fs = params.get("Sampling Rate (Hz)", "")
    nfft = params.get("FFT Size", "")

    q = f"""
    Explain an IEEE 802.{standard} OFDM wireless LAN signal with:
    - PHY format: {phy}
    - Modulation: {mod} with coding rate {code} (MCS {mcs})
    - Channel bandwidth: {bw}, sampling rate {fs}, FFT size {nfft}.

    I want detailed information on:
    - The OFDM PHY structure (subcarriers, pilots, guard interval, symbol duration).
    - How this configuration appears as complex IQ samples.
    - Typical SNR and channel conditions where {mod} with coding rate {code} (MCS {mcs}) is used.
    - The trade-off between robustness and throughput for this mode.
    - Best-case scenarios and deployment environments where such a signal is recommended.
    - Any explanations in the book regarding IEEE 802.{standard}, OFDM, BPSK, and low-rate MCS modes.
    """
    return " ".join(q.split())


def build_context(results, max_chars: int = 4000) -> str:
    """
    Combine search results into a single context string for LLM prompting.
    """
    pieces = []
    for r in results:
        header = (
            f"--- Source: {r['id']} "
            f"(page {r['page']}, chapter={r['chapter']}, "
            f"section={r['section']}, subsection={r['subsection']}) ---\n"
        )
        pieces.append(header + r["text"] + "\n")

    ctx = "\n".join(pieces)
    if len(ctx) > max_chars:
        ctx = ctx[:max_chars] + "\n...[truncated]..."
    return ctx


def build_llm_prompt(params: dict, context: str) -> str:
    """
    Create a prompt that an LLM can use to generate a rich label describing this signal.
    """
    return f"""
        You are an RF/Wi-Fi physical layer expert helping to label Wi-Fi IQ datasets for a generative model.

        The IQ sample has the following configuration:

        - Standard: IEEE 802.{params.get("Standard")}
        - PHY format: {params.get("phyFormat")}
        - Modulation: {params.get("Modulation")}
        - Coding rate: {params.get("CodingRate")} (MCS {params.get("MCS")})
        - Bandwidth: {params.get("Bandwidth")}
        - Sampling rate: {params.get("Sampling Rate (Hz)")}
        - FFT size: {params.get("FFT Size")}

        You are given context extracted from technical books on OFDM and IEEE 802.11:

        CONTEXT:
        {context}

        TASK:
        Using ONLY the information that can be reasonably inferred from the context above, write a single rich, multi-sentence label for this IQ signal that includes:

        1. What this signal represents at the PHY layer (OFDM structure, subcarriers, pilots, guard interval, symbol duration, etc.).
        2. The implications of using {params.get("Modulation")} with coding rate {params.get("CodingRate")} (MCS {params.get("MCS")}): robustness vs throughput.
        3. Typical SNR range, channel conditions, and deployment scenarios where this configuration is appropriate (e.g., low-SNR indoor coverage, long-range links, etc.), if supported by the context.
        4. Any notes about how the IQ samples would look statistically or spectrally (e.g., constellation density, energy distribution) that follow from the context.

        Write the label as a dense, natural-language description that a diffusion model training pipeline could treat as a detailed caption for the IQ sample.

        If some detail is not described in the context, explicitly say "not specified in context" instead of guessing.

        Return ONLY the label text, no extra commentary or bullet points.
    """.strip()


# ===========================
# NEW: IQ loader & main loop
# ===========================

def load_iq_from_bin(
    file_path: str,
    dtype=np.float32,
) -> np.ndarray:
    """
    Load IQ samples from a binary file assumed to contain interleaved float32:
    [Re0, Im0, Re1, Im1, ...].

    Returns:
        complex64 numpy array of shape (N,)
    """
    raw = np.fromfile(file_path, dtype=dtype)
    if raw.size < 2:
        raise ValueError(f"File {file_path} too small to contain IQ data.")
    if raw.size % 2 != 0:
        # drop last odd sample if needed
        raw = raw[:-1]

    i_samples = raw[0::2]
    q_samples = raw[1::2]
    iq = i_samples + 1j * q_samples
    return iq.astype(np.complex64)


def process_excel(excel_path: str, out_dir: str):
    """
    For each row in the Excel file:
      - Skip if File_Path is missing/empty.
      - Load the IQ .bin file.
      - Run RAG search + LLM to generate rich label.
      - Save a .mat file with keys:
          'data' -> complex IQ samples
          'label' -> generated LLM label (string)
    """
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_excel(excel_path)

    required_cols = [
        "Standard",
        "phyFormat",
        "Modulation",
        "CodingRate",
        "MCS",
        "Bandwidth",
        "Sampling Rate (Hz)",
        "FFT Size",
        "File_Path",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Excel file missing required column: {col}")

    llm = LLM()

    for idx, row in df.iterrows():
        file_path = str(row.get("File_Path") or "").strip()
        if not file_path:
            print(f"[Row {idx}] Missing File_Path, skipping.")
            continue

        if not os.path.isfile(file_path):
            print(f"[Row {idx}] File not found: {file_path}, skipping.")
            continue

        print(f"\n=== Processing row {idx} ===")
        print(f"  File: {file_path}")

        # Build params dict for this row
        params = {
            "Standard": row.get("Standard"),
            "phyFormat": row.get("phyFormat"),
            "Modulation": row.get("Modulation"),
            "CodingRate": row.get("CodingRate"),
            "MCS": row.get("MCS"),
            "Bandwidth": row.get("Bandwidth"),
            "Sampling Rate (Hz)": row.get("Sampling Rate (Hz)"),
            "FFT Size": row.get("FFT Size"),
        }

        # 1) Build semantic query
        query_text = signal_params_to_query(params)

                # 2) RAG search over your chunks
        results = search(query_text, top_k=TOP_K_DEFAULT)

        # Convert results into MATLAB-friendly struct format
        rag_results_struct = []
        for r in results:
            rag_results_struct.append({
                "id":         _safe_str(r.get("id")),
                "score":      float(r.get("score", 0.0)),
                "page":       _safe_str(r.get("page")),
                "chapter":    _safe_str(r.get("chapter")),
                "section":    _safe_str(r.get("section")),
                "subsection": _safe_str(r.get("subsection")),
                "source":     _safe_str(r.get("source")),
                "book_title": _safe_str(r.get("book_title")),
                "text":       _safe_str(r.get("text")),
            })
        # MATLAB requires list -> object array
        rag_results_struct = np.array(rag_results_struct, dtype=object)

        # 3) Build context for LLM
        context = build_context(results, max_chars=4000)

        # 4) Build prompt & generate label
        prompt = build_llm_prompt(params, context)
        raw_label = llm.generate(
            prompt,
            max_new_tokens=384,
            temperature=0.2,
            top_p=0.9,
        )

        if isinstance(raw_label, (list, dict)):
            label_text = json.dumps(raw_label)
        else:
            label_text = str(raw_label).strip()

        # 5) Load IQ data from binary file
        try:
            iq_data = load_iq_from_bin(file_path)
        except Exception as e:
            print(f"  ERROR loading IQ from {file_path}: {e}")
            continue

        # 6) Create .mat file name and save
        mat_name = f"{row.get("Standard")}_{row.get("phyFormat")}_{row.get("Modulation")}_{row.get("MCS")}_{int(row.get("Sampling Rate (Hz)")) // 1000000}MHz_FFT{row.get("FFT Size")}.mat"
        mat_path = os.path.join(out_dir, mat_name)

        mat_dict = {
            "data": iq_data,
            "label": label_text,
            "rag_results": rag_results_struct,   # <-- NEW FIELD
        }

        savemat(mat_path, mat_dict)
        print(f"  Saved MAT file: {mat_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAG + LLM labeling of Wi-Fi IQ .bin files into .mat files."
    )
    parser.add_argument(
        "--excel",
        type=str,
        required=True,
        help="Path to Excel file containing signal parameters and File_Path column.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory where .mat files will be saved.",
    )

    args = parser.parse_args()

    # Load DB once
    DB, EMB_MATRIX_NORM = load_db_from_folder(CHUNKS_FOLDER)

    # Process all rows in the Excel
    process_excel(args.excel, args.out_dir)
