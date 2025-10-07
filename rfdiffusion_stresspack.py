#!/usr/bin/env python3
import argparse
import glob
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import scipy.io as sio

def add_awgn(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    p_signal = np.mean(np.abs(x)**2)
    snr_lin = 10**(snr_db/10.0)
    p_noise = p_signal / snr_lin if snr_lin > 0 else p_signal * 10
    noise = (rng.normal(0, np.sqrt(p_noise/2), size=x.shape)
             + 1j * rng.normal(0, np.sqrt(p_noise/2), size=x.shape))
    return x + noise

def time_stretch_2d(x: np.ndarray, scale: float) -> np.ndarray:
    T, F = x.shape
    t_old = np.arange(T)
    t_new = np.linspace(0, T-1, int(round(T*scale)))
    xr = np.empty((t_new.shape[0], F), dtype=float)
    xi = np.empty((t_new.shape[0], F), dtype=float)
    for f in range(F):
        xr[:, f] = np.interp(t_new, t_old, np.real(x[:, f]), left=0.0, right=0.0)
        xi[:, f] = np.interp(t_new, t_old, np.imag(x[:, f]), left=0.0, right=0.0)
    y = xr + 1j*xi
    if y.shape[0] < T:
        y = np.pad(y, ((0, T-y.shape[0]), (0,0)), mode='constant')
    elif y.shape[0] > T:
        y = y[:T, :]
    return y

def freq_shift_bins(x: np.ndarray, shift: int) -> np.ndarray:
    return np.roll(x, shift=shift, axis=1)

def tf_random_mask(x: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    mask = rng.random(size=x.shape) < frac
    return np.where(mask, 0.0 + 0.0j, x)

def splice_segments(x1: np.ndarray, x2: np.ndarray, seg_len: int = 100) -> np.ndarray:
    T = x1.shape[0]
    out = np.zeros_like(x1)
    i = 0
    flip = False
    while i < T:
        j = min(T, i + seg_len)
        out[i:j] = x2[i:j] if flip else x1[i:j]
        flip = not flip
        i = j
    return out

def swap_cond_indices(cvec: np.ndarray, idx_pairs: Tuple[Tuple[int,int], ...]) -> np.ndarray:
    c = cvec.copy()
    for i, j in idx_pairs:
        if i < c.size and j < c.size:
            c[i], c[j] = c[j], c[i]
    return c

def process_file(path: str,
                 out_dir: Path,
                 snr_list: List[float],
                 time_scales: List[float],
                 freq_shifts: List[int],
                 tf_mask_frac: float,
                 splice_len: int,
                 swap_cond: List[int],
                 seed: int = 123):
    data = sio.loadmat(path, squeeze_me=False, struct_as_record=False)
    if "feature" not in data or "cond" not in data:
        raise ValueError(f"{path} missing required variables 'feature' and/or 'cond'")

    feature = data["feature"]
    cond = data["cond"]
    cond_vec = np.asarray(cond).astype(np.int32).ravel()

    base_name = Path(path).stem
    rng = np.random.default_rng(seed)

    manifest_rows = []

    def save_variant(tag: str, name: str, feat_var: np.ndarray, cond_var: np.ndarray):
        out_name = f"{base_name}__{tag}__{name}.mat"
        out_path = out_dir / out_name
        sio.savemat(out_path, {"feature": feat_var, "cond": cond_var.reshape(1, -1)})
        manifest_rows.append({
            "src": path,
            "file": str(out_path),
            "tag": tag,
            "name": name,
            "feature_shape": tuple(int(x) for x in feat_var.shape),
            "cond_vec": cond_var.tolist()
        })

    # 1) AWGN
    # for snr in snr_list:
    #     v = add_awgn(feature, snr_db=snr, rng=rng)
    #     save_variant("awgn", f"snr{snr:+.0f}dB", v, cond_vec)

    # # 2) Time stretch
    # for s in time_scales:
    #     v = time_stretch_2d(feature, scale=float(s))
    #     save_variant("tstretch", f"s{s:.2f}", v, cond_vec)

    # # 3) Frequency shift
    # for k in freq_shifts:
    #     v = freq_shift_bins(feature, shift=int(k))
    #     save_variant("fshift", f"k{int(k):+d}", v, cond_vec)

    # # 4) TF random mask
    # if tf_mask_frac and tf_mask_frac > 0.0:
    #     v_mask = tf_random_mask(feature, frac=float(tf_mask_frac), rng=rng)
    #     save_variant("tfmask", f"p{tf_mask_frac:.2f}", v_mask, cond_vec)

    # # # 5) Splice with shifted copy
    # if splice_len and splice_len > 0:
    #     x_alt = freq_shift_bins(feature, shift=3)
    #     v_splice = splice_segments(feature, x_alt, seg_len=int(splice_len))
    #     save_variant("splice", f"alt3_seg{int(splice_len)}", v_splice, cond_vec)

    # # 6) Condition corruption (pairwise swaps along the list)
    if swap_cond and len(swap_cond) >= 2:
        pairs = list(zip(swap_cond[::2], swap_cond[1::2]))
        cond_swapped = swap_cond_indices(cond_vec, tuple(pairs))
        save_variant("condswap", "_".join([f"{i}_{j}" for i, j in pairs]), feature, cond_swapped)

    return manifest_rows

def main():
    ap = argparse.ArgumentParser(description="Generate stress-test variants for RF-Diffusion .mat samples")
    ap.add_argument("--input", nargs="+", required=True,
                    help="Input .mat files or glob patterns (e.g., data/*.mat)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--snr", type=float, nargs="*", default=[-5, 0, 5, 10, 20],
                    help="SNRs (dB) for AWGN variants")
    ap.add_argument("--time-scale", type=float, nargs="*", default=[0.90, 0.95, 1.05, 1.10],
                    help="Time-stretch scales")
    ap.add_argument("--freq-shift", type=int, nargs="*", default=[-4, -2, 2, 4],
                    help="Frequency-bin shifts")
    ap.add_argument("--tf-mask-frac", type=float, default=0.15,
                    help="Fraction of TF bins to mask (0..1)")
    ap.add_argument("--splice-len", type=int, default=120,
                    help="Segment length for splicing with shifted copy")
    ap.add_argument("--swap-cond", type=int, nargs="*", default=[],
                    help="Indices to swap in the condition vector, given as pairs: e.g., --swap-cond 0 1 2 3")
    ap.add_argument("--seed", type=int, default=123, help="Random seed")
    args = ap.parse_args()

    # Expand glob patterns
    input_files = []
    for pat in args.input:
        input_files.extend(glob.glob(pat))
    if not input_files:
        raise SystemExit("No input files matched. Check --input patterns.")

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for path in input_files:
        rows = process_file(path,
                            out_dir=out_dir,
                            snr_list=args.snr,
                            time_scales=args.time_scale,
                            freq_shifts=args.freq_shift,
                            tf_mask_frac=args.tf_mask_frac,
                            splice_len=args.splice_len,
                            swap_cond=args.swap_cond,
                            seed=args.seed)
        all_rows.extend(rows)

    # Write manifest
    manifest = pd.DataFrame(all_rows)
    manifest_path = out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"Wrote {len(all_rows)} variants across {len(input_files)} inputs to {out_dir}")
    print(f"Manifest: {manifest_path}")

if __name__ == "__main__":
    main()
