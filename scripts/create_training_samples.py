#!/usr/bin/env python3
import os
import re
import json
import glob
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import savemat

# Local Wrappers
from RAG.llm import LLM

# ===========================
# CONFIG
# ===========================
EMBED_MODEL_NAME = "intfloat/e5-large-v2"


# ===========================
# Helpers
# ===========================
def _safe_str(x):
    if x is None:
        return ""
    return str(x)

def sanitize_filename(s: str, max_len: int = 180) -> str:
    s = _safe_str(s).strip()
    if not s:
        return "sample"
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-zA-Z0-9._-]+", "", s)
    return s[:max_len] if len(s) > max_len else s


def existing_signal_indices(out_dir: str, stem: str):
    pattern = re.compile(rf"^{re.escape(stem)}_s(\d+)\.mat$")
    found = []
    try:
        for name in os.listdir(out_dir):
            m = pattern.match(name)
            if m:
                found.append(int(m.group(1)))
    except FileNotFoundError:
        return []
    return sorted(found)


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


def signal_params_to_query(params: dict) -> str:
    """
    Convert a simple-signal configuration into a rich semantic query for your RAG KB.
    """
    mod = params.get("Modulation", "")
    N = params.get("Number of Samples", "")
    amp = params.get("Amplitude", "")
    fc = params.get("Center Frequency (Hz)", "")
    phase = params.get("Phase (degrees)", "")
    fs = params.get("Sampling Rate (Hz)", "")
    sps = params.get("Samples Per Symbol", 1)

    q = f"""
    Explain a simple complex IQ signal that uses digital modulation {mod}.
    The signal is represented as complex baseband (I/Q) samples.

    Known generation parameters:
    - Number of samples (N): {N}
    - Sampling rate (fs): {fs} Hz
    - Samples per symbol (sps): {sps}
    - Amplitude: {amp}
    - Center frequency (fc): {fc} Hz
    - Initial phase: {phase} degrees

    I want detailed information focused on digital modulation (not Wi-Fi / OFDM):
    - What {mod} means in terms of constellation geometry and bits-per-symbol (if applicable).
    - How {mod} appears in the IQ plane (constellation shape/density) and time-domain waveform.
    - Typical channel/SNR regimes where {mod} is preferred (robustness vs spectral efficiency).
    - Practical scenarios where {mod} is a good choice.
    - Notes about amplitude scaling, carrier/center frequency offsets, and phase on observed IQ samples.
    - BER vs SNR or decision-boundary intuition for {mod} (from standard comms texts).
    """
    return " ".join(q.split())


def build_llm_prompt(params: dict, context: str) -> str:
    mod   = params.get("Modulation", "")
    N     = params.get("Number of Samples", "")
    amp   = params.get("Amplitude", "")
    fc    = params.get("Center Frequency (Hz)", "")
    phase = params.get("Phase (degrees)", "")
    fs    = params.get("Sampling Rate (Hz)", "")
    sps   = params.get("Samples Per Symbol", 1)

    return f"""
        SIGNAL CONFIG (authoritative; use exactly these values):
        Modulation={mod}; N={N}; sampling_rate_hz={fs}; samples_per_symbol={sps}; amplitude={amp}; center_frequency_hz={fc}; initial_phase_deg={phase}.

        CONTEXT (RAG output; ONLY use facts stated here; do not invent):
        {context}

        Write the TWO-SENTENCE label now.
    """.strip()


def _label_style_for_index(idx: int) -> str:
    styles = ["advanced", "less_advanced", "simple"]
    return styles[int(idx) % len(styles)]


def build_single_label(params: dict, context: str, llm, style: str, variation_idx: int, used_labels=None, use_rag: bool = True):
    """
    Create one label for one IQ/bits sample.
    The label style cycles by generation index so repeated samples from the same
    modulation template do not all receive the same complexity level.
    """
    mod = _safe_str(params.get("Modulation")).strip().upper().replace("-", "").replace(" ", "")
    N = int(params.get("Number of Samples"))
    fs = float(params.get("Sampling Rate (Hz)"))
    amp = float(params.get("Amplitude"))
    fc = float(params.get("Center Frequency (Hz)"))
    phase = float(params.get("Phase (degrees)"))
    sps = int(params.get("Samples Per Symbol", 1) or 1)
    style = str(style).strip().lower()

    template_map = {
        "simple": f"Generate a {mod} IQ signal with the provided bitstream.",
        "less_advanced": (
            f"Generate a {mod} baseband IQ waveform with N={N}, fs={fs} Hz, "
            f"samples_per_symbol={sps}, amplitude={amp}, center_frequency={fc} Hz, and phase={phase} deg."
        ),
        "advanced": (
            f"Create a {mod} communication waveform in complex IQ form, preserving constellation geometry, "
            f"symbol timing, and demodulation-friendly structure under the stated parameters."
        ),
    }
    seed_label = template_map.get(style, template_map["advanced"])
    prior_labels = list(used_labels or [])
    prior_text = "\n".join(f"- {x}" for x in prior_labels[-6:]) if prior_labels else "(none yet)"
    context_header = "CONTEXT (use only facts stated here):" if use_rag else "PURE LLM MODE (no retrieved RAG context):"
    context_instruction = (
        "Use the retrieved context for modulation facts and do not invent unsupported details."
        if use_rag
        else "Use your general digital communications knowledge, grounded by the signal config, without citing retrieved sources."
    )

    prompt = f"""
        SIGNAL CONFIG (authoritative):
        Modulation={mod}; N={N}; sampling_rate_hz={fs}; samples_per_symbol={sps}; amplitude={amp}; center_frequency_hz={fc}; initial_phase_deg={phase}.

        LABEL STYLE:
        {style}

        VARIATION INDEX:
        {variation_idx}

        PREVIOUS LABELS FOR THIS TEMPLATE (avoid repeating their wording):
        {prior_text}

        {context_header}
        {context}

        Write exactly TWO sentences as a training label for this signal.
        {context_instruction}
        Keep the style as requested, keep it modulation-grounded, and use distinct wording from the previous labels.
    """.strip()

    try:
        raw = llm.generate(
            prompt,
            max_new_tokens=196,
            temperature=0.35 + 0.08 * (variation_idx % 3),
            top_p=0.9,
        )
        label = json.dumps(raw) if isinstance(raw, (list, dict)) else str(raw).strip()
        if not label:
            label = seed_label
    except Exception:
        label = seed_label

    if used_labels is not None and label in used_labels:
        label = f"{seed_label} Variation {variation_idx + 1}."
    return {"style": style, "label": label}


def save_constellation_plot(path: str, iq_data: np.ndarray, title: str, stride: int = 1):
    iq = np.asarray(iq_data).reshape(-1).astype(np.complex64, copy=False)
    step = max(1, int(stride))
    pts = iq[::step]

    plt.figure(figsize=(6, 6))
    plt.plot(pts.real, pts.imag, ".", markersize=2)
    plt.xlabel("In-Phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.title(title)
    plt.grid(True, alpha=0.4)
    plt.axhline(0.0, color="black", linewidth=0.5)
    plt.axvline(0.0, color="black", linewidth=0.5)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# ===========================
# Modulation / Signal Generation
# ===========================

def _norm_to_unit_avg_power(x: np.ndarray) -> np.ndarray:
    """Normalize constellation to unit average power."""
    p = np.mean(np.abs(x)**2)
    if p <= 0:
        return x
    return x / np.sqrt(p)

def _bits_to_int(bits_2d: np.ndarray) -> np.ndarray:
    """
    bits_2d: shape (Nsym, k), MSB-first.
    Returns ints 0..(2^k - 1).
    """
    k = bits_2d.shape[1]
    w = (1 << np.arange(k-1, -1, -1, dtype=np.int64))
    return (bits_2d.astype(np.int64) * w).sum(axis=1)

def _gray_to_binary(g: np.ndarray) -> np.ndarray:
    """Vectorized Gray->binary for non-negative ints."""
    b = g.copy()
    shift = 1
    # enough shifts to cover up to 8 bits (256QAM) safely
    while shift < 32:
        b ^= (b >> shift)
        shift <<= 1
    return b

def bits_per_symbol(mod: str) -> int:
    m = mod.strip().upper().replace("-", "").replace(" ", "")
    if m == "BPSK":
        return 1
    if m == "QPSK":
        return 2
    if m == "8PSK":
        return 3
    if m == "16QAM":
        return 4
    if m == "64QAM":
        return 6
    if m == "256QAM":
        return 8
    raise ValueError(f"Unsupported Modulation '{mod}'. Supported: BPSK, QPSK, 8PSK, 16QAM, 64QAM, 256QAM.")

def gen_symbols_from_bits(mod: str, bits: np.ndarray) -> np.ndarray:
    """
    Map bits -> complex symbols (unit average power).
    bits: shape (n_bits,), dtype int8 {0,1}
    """
    m = mod.strip().upper().replace("-", "").replace(" ", "")

    if m == "BPSK":
        # 1 bit/sym: 0->-1, 1->+1
        b = (2.0 * bits.astype(np.float32) - 1.0).astype(np.float32)
        return b.astype(np.complex64)

    if m == "QPSK":
        # 2 bits/sym, Gray-ish mapping via I/Q bits:
        # b0 -> I, b1 -> Q, each in {-1,+1}, then normalize by sqrt(2)
        if bits.size % 2 != 0:
            raise ValueError("QPSK requires bits length multiple of 2")
        bb = bits.reshape(-1, 2)
        I = (2.0 * bb[:, 0].astype(np.float32) - 1.0)
        Q = (2.0 * bb[:, 1].astype(np.float32) - 1.0)
        syms = (I + 1j * Q) / np.sqrt(2.0)
        return syms.astype(np.complex64)

    if m == "8PSK":
        # 3 bits/sym -> Gray coded phase index
        if bits.size % 3 != 0:
            raise ValueError("8PSK requires bits length multiple of 3")
        bb = bits.reshape(-1, 3)
        g = _bits_to_int(bb)                 # gray index 0..7
        idx = _gray_to_binary(g)             # binary phase index 0..7
        phase = 2.0 * np.pi * idx / 8.0
        syms = np.exp(1j * phase)
        syms = _norm_to_unit_avg_power(syms)
        return syms.astype(np.complex64)

    if m in ("16QAM", "64QAM", "256QAM"):
        M = int(m.replace("QAM", ""))
        k = int(np.log2(M))
        if bits.size % k != 0:
            raise ValueError(f"{m} requires bits length multiple of {k}")

        bb = bits.reshape(-1, k)

        # Square QAM: split bits into I and Q halves
        k2 = k // 2
        gI = _bits_to_int(bb[:, :k2])   # Gray index on I axis
        gQ = _bits_to_int(bb[:, k2:])   # Gray index on Q axis

        # Convert Gray index -> binary index 0..(sqrt(M)-1)
        bI = _gray_to_binary(gI)
        bQ = _gray_to_binary(gQ)

        L = int(np.sqrt(M))  # levels per axis
        # Map 0..L-1 to odd levels: -(L-1), -(L-3), ..., +(L-1)
        # amplitude = 2*index - (L-1)
        aI = (2.0 * bI.astype(np.float32) - (L - 1)).astype(np.float32)
        aQ = (2.0 * bQ.astype(np.float32) - (L - 1)).astype(np.float32)

        syms = aI + 1j * aQ
        syms = _norm_to_unit_avg_power(syms)
        return syms.astype(np.complex64)

    raise ValueError(f"Unsupported Modulation '{mod}'. Supported: BPSK, QPSK, 8PSK, 16QAM, 64QAM, 256QAM.")


def generate_iq_from_excel_params(params: dict):
    """
    Generate IQ samples and the underlying bitstream from Excel-driven params.

    Params expected:
      Modulation, Number of Samples (N), Sampling Rate (Hz) (fs), Amplitude,
      Center Frequency (Hz) (fc), Phase (degrees)
    Optional:
      Samples Per Symbol (sps), Seed

    Output:
      x: complex64 IQ samples, length N
      bits: int8 bitstream used (length n_syms * bits_per_symbol)
    """
    mod = _safe_str(params.get("Modulation")).strip()
    N = int(params.get("Number of Samples"))
    fs = float(params.get("Sampling Rate (Hz)"))
    amp = float(params.get("Amplitude"))
    fc = float(params.get("Center Frequency (Hz)"))
    phase_deg = float(params.get("Phase (degrees)"))

    sps = int(params.get("Samples Per Symbol", 1) or 1)
    if sps <= 0:
        sps = 1

    seed = params.get("Seed", None)
    seed = None if (seed is None or (isinstance(seed, float) and np.isnan(seed)) or str(seed).strip() == "") else int(seed)
    rng = np.random.default_rng(seed)

    # Decide number of symbols needed, then bits needed for that modulation
    n_syms = int(np.ceil(N / float(sps)))
    k = bits_per_symbol(mod)
    n_bits = n_syms * k

    bits = rng.integers(0, 2, size=n_bits, dtype=np.int8)
    syms = gen_symbols_from_bits(mod, bits)  # complex symbols (unit avg power)

    # Upsample symbols by sps (simple rectangular pulse shaping)
    # If you later add pulse shaping (RRC), this is where it goes.
    x_bb = np.repeat(syms, sps)[:N].astype(np.complex64)

    # Apply carrier rotation / frequency offset if fc != 0 (stays "complex IQ", but now passband-ish rotation)
    n = np.arange(N, dtype=np.float64)
    phase = np.deg2rad(phase_deg)
    rot = np.exp(1j * (2.0 * np.pi * fc * n / fs + phase)).astype(np.complex64)

    x = (amp * x_bb * rot).astype(np.complex64)
    return x, bits.astype(np.int8)


# ===========================
# Main Excel Loop
# ===========================
def process_excel(
    excel_path: str,
    out_dir: str,
    top_k: int,
    chunks_folder: str,
    signals_per_row: int,
    plot_constellation: bool = False,
    plot_stride: int = 1,
    use_rag: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_excel(excel_path)

    required_cols = [
        "Modulation",
        "Number of Samples",
        "Sampling Rate (Hz)",
        "Amplitude",
        "Center Frequency (Hz)",
        "Phase (degrees)",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Excel file missing required column: {col}")

    # Optional cols
    if "Samples Per Symbol" not in df.columns:
        df["Samples Per Symbol"] = 1
    if "Seed" not in df.columns:
        df["Seed"] = np.nan
    if "Output Name" not in df.columns:
        df["Output Name"] = ""

    if use_rag:
        from RAG.rag import RAGSearch
        rag = RAGSearch(chunks_folder=chunks_folder)
    else:
        rag = None
    llm = LLM()
    if not use_rag:
        llm.system_prompt = """
            You are a digital communications (PHY) expert labeling simple digitally modulated IQ signals (non-Wi-Fi).

            You will receive authoritative signal configuration values and no retrieved RAG context.
            Your task is to produce EXACTLY TWO sentences total, on a SINGLE LINE, plain natural language only.

            Sentence 1 must describe the modulation type, constellation or symbol-level meaning, and all provided signal parameters.
            Sentence 2 must use general digital communications knowledge to describe expected IQ/constellation/time-domain behavior and relevant robustness or BER/SNR intuition.

            STRICT RULES:
            - Exactly TWO sentences.
            - No bullet points, no lists, no colon-separated fields.
            - Do not mention RAG, retrieval, sources, or missing context.
            - Stay grounded in the SIGNAL CONFIG and common digital modulation facts.
        """.strip()

    manifest_rows = []

    for idx, row in df.iterrows():
        print(f"\n=== Processing row {idx} ===")

        base_params = {
            "Modulation": row.get("Modulation"),
            "Number of Samples": row.get("Number of Samples"),
            "Sampling Rate (Hz)": row.get("Sampling Rate (Hz)"),
            "Amplitude": row.get("Amplitude"),
            "Center Frequency (Hz)": row.get("Center Frequency (Hz)"),
            "Phase (degrees)": row.get("Phase (degrees)"),
            "Samples Per Symbol": row.get("Samples Per Symbol", 1),
            "Seed": row.get("Seed", None),
        }

        # 0) Generate the IQ + bits using EXCEL params
        try:
            _ = bits_per_symbol(_safe_str(base_params.get("Modulation")).strip())
        except Exception as e:
            print(f"  ERROR parsing row {idx}: {e}")
            continue

        # 1) Optionally build RAG context from base params.
        if use_rag:
            query_text = signal_params_to_query(base_params)
            results = rag.search(query_text, top_k=top_k)
            context = build_context(results, max_chars=4000)
        else:
            results = []
            context = (
                "No retrieved context was used. Generate the label from the authoritative "
                "signal configuration and general digital modulation knowledge."
            )

        # MATLAB-friendly struct array for rag_results
        rag_results_struct = []
        for r in results:
            rag_results_struct.append(
                {
                    "id": _safe_str(r.get("id")),
                    "score": float(r.get("score", 0.0)),
                    "page": _safe_str(r.get("page")),
                    "chapter": _safe_str(r.get("chapter")),
                    "section": _safe_str(r.get("section")),
                    "source": _safe_str(r.get("source")),
                    "book_title": _safe_str(r.get("book_title")),
                    "text": _safe_str(r.get("text")),
                }
            )
        rag_results_struct = np.array(rag_results_struct, dtype=object)

        # 4) Multi-label generation for this single IQ/bits sample
        # 4) Save N distinct signals for this modulation template.
        output_name = _safe_str(row.get("Output Name")).strip()
        if output_name:
            stem = sanitize_filename(output_name)
        else:
            stem = sanitize_filename(
                f"{row.get('Modulation')}_N{row.get('Number of Samples')}_sps{row.get('Samples Per Symbol', 1)}_amp{row.get('Amplitude')}_fc{row.get('Center Frequency (Hz)')}_ph{row.get('Phase (degrees)')}_row{idx}"
            )

        mod_name = _safe_str(base_params.get("Modulation")).strip().upper().replace("-", "").replace(" ", "")
        sps_val = int(base_params.get("Samples Per Symbol", 1) or 1)
        k_val = bits_per_symbol(mod_name)

        base_seed = base_params.get("Seed", None)
        if isinstance(base_seed, float) and np.isnan(base_seed):
            base_seed = None
        base_seed = None if base_seed is None or str(base_seed).strip() == "" else int(base_seed)
        row_rng = np.random.default_rng(base_seed if base_seed is not None else (idx + 1) * 7919)
        used_labels = []
        existing_indices = existing_signal_indices(out_dir, stem)
        target_total = max(1, int(signals_per_row))
        existing_set = set(existing_indices)
        if len(existing_indices) >= target_total:
            print(f"  Found {len(existing_indices)} existing samples for {stem}; nothing new to generate.")
            continue
        print(f"  Found {len(existing_indices)} existing samples for {stem}; generating up to {target_total}.")

        for j in range(target_total):
            if j in existing_set:
                continue
            signal_seed = int(base_seed + j) if base_seed is not None else int(row_rng.integers(0, 2**31 - 1))
            params = dict(base_params)
            params["Seed"] = signal_seed
            try:
                iq_data, bits = generate_iq_from_excel_params(params)
            except Exception as e:
                print(f"  ERROR generating IQ for row {idx}, sample {j}: {e}")
                continue

            style = _label_style_for_index(j)
            label_info = build_single_label(params, context, llm, style, j, used_labels=used_labels, use_rag=use_rag)
            used_labels.append(label_info["label"])

            file_stem = f"{stem}_s{j:03d}"
            mat_path = os.path.join(out_dir, f"{file_stem}.mat")
            mat_dict = {
                "data": iq_data.astype(np.complex64),
                "bits": bits.astype(np.int8),
                "modulation": np.array([mod_name], dtype=object),
                "bits_per_symbol": np.array([[k_val]], dtype=np.int32),
                "samples_per_symbol": np.array([[sps_val]], dtype=np.int32),
                "label": label_info["label"],
                "label_mode": np.array(["rag_llm" if use_rag else "pure_llm"], dtype=object),
                "rag_results": rag_results_struct,
            }
            savemat(mat_path, mat_dict)
            if plot_constellation:
                plot_path = os.path.join(out_dir, f"{file_stem}_constellation.png")
                plot_title = f"{mod_name} Constellation"
                save_constellation_plot(plot_path, iq_data, plot_title, stride=plot_stride)
            manifest_rows.append(
                {
                    "template_row": idx,
                    "signal_index": j,
                    "file": os.path.basename(mat_path),
                    "modulation": mod_name,
                    "seed": signal_seed,
                    "label_style": label_info["style"],
                    "label_mode": "rag_llm" if use_rag else "pure_llm",
                    "bits_len": int(bits.size),
                    "number_of_samples": int(base_params["Number of Samples"]),
                }
            )
            print(f"  Saved MAT file: {mat_path}  [sample {j+1}/{target_total}, style={label_info['style']}]")
        print("  Keys: data (complex64), bits (int8), modulation (str), bits_per_symbol (int), samples_per_symbol (int), label (str), label_mode (str), rag_results (struct array)")

    if manifest_rows:
        manifest_path = os.path.join(out_dir, "generation_manifest.csv")
        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
        print(f"Saved generation manifest: {manifest_path}")


def main():
    ap = argparse.ArgumentParser(description="Excel → generate simple modulated IQ → LLM label → .mat")
    ap.add_argument("--excel", type=str, required=True, help="Path to Excel file with signal parameters.")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for .mat files.")
    ap.add_argument("--top_k", type=int, default=5, help="Top-k RAG chunks used for LLM context.")
    ap.add_argument(
        "--chunks_folder",
        type=str,
        default="./RAG/Knowledge_Base/Chunks",
        help="Folder with JSON chunk files (must include embeddings).",
    )
    ap.add_argument(
        "--signals_per_row",
        type=int,
        default=1,
        help="Number of distinct IQ samples to generate from each Excel row template.",
    )
    ap.add_argument(
        "--plot_constellation",
        action="store_true",
        help="Save one constellation PNG per generated IQ sample.",
    )
    ap.add_argument(
        "--plot_stride",
        type=int,
        default=1,
        help="Subsample factor for constellation plotting (e.g. 2 keeps every other point).",
    )
    ap.add_argument(
        "--no_rag",
        "--pure_llm",
        action="store_true",
        help="Skip RAG retrieval and generate labels with the LLM only.",
    )
    args = ap.parse_args()

    process_excel(
        args.excel,
        args.out_dir,
        args.top_k,
        args.chunks_folder,
        args.signals_per_row,
        plot_constellation=args.plot_constellation,
        plot_stride=args.plot_stride,
        use_rag=not args.no_rag,
    )


if __name__ == "__main__":
    main()
