#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import scipy.io as scio


def save_complex_bin_interleaved_iq(path: Path, x: np.ndarray) -> None:
    """Save complex64 x as interleaved float32 IQ: [I0,Q0,I1,Q1,...]."""
    x = np.asarray(x, dtype=np.complex64).ravel()
    iq = np.empty(2 * x.size, dtype=np.float32)
    iq[0::2] = np.real(x).astype(np.float32)
    iq[1::2] = np.imag(x).astype(np.float32)
    iq.tofile(path)


def main():
    ap = argparse.ArgumentParser(description="Generate one RAW (no-noise) BPSK complex IQ sample and save to .bin")
    ap.add_argument("--out-mat", type=str, required=True, help="Output .mat filename to save 'data','label','bits' for training")
    ap.add_argument("--label", type=str, required=True, help="Label / prompt to save in the .mat file")
    ap.add_argument("--N", type=int, default=1024, help="Number of IQ samples (1xN)")
    ap.add_argument("--fs", type=float, default=1_000_000.0, help="Sample rate (Hz)")

    # only tunables you asked for
    ap.add_argument("--amp", type=float, default=1.0, help="Amplitude")
    ap.add_argument("--fc", type=float, default=0.0, help="Center frequency (Hz)")
    ap.add_argument("--phase_deg", type=float, default=0.0, help="Initial phase (degrees)")

    ap.add_argument("--seed", type=int, default=None, help="Optional seed for repeatability (omit for random)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # BPSK bits -> ±1 (symbol-rate: one symbol per sample)
    bits = rng.integers(0, 2, size=args.N, dtype=np.int8)
    b = (2.0 * bits.astype(np.float32) - 1.0).astype(np.float32)  # ±1

    n = np.arange(args.N, dtype=np.float64)
    phase = np.deg2rad(args.phase_deg)
    rot = np.exp(1j * (2.0 * np.pi * args.fc * n / args.fs + phase))

    x = (args.amp * b.astype(np.complex64) * rot.astype(np.complex64)).astype(np.complex64)

    # Save a .mat file with 'data','label','bits' (required for training loader)
    mat_path = Path(args.out_mat)
    # Ensure complex array is stored as complex64
    data_to_save = x.astype(np.complex64)
    # bits were generated as integers 0/1
    bits_to_save = bits.astype(np.int8)
    mat_dict = {
        #'data': data_to_save,
        'prompt': args.label,
        'bits': bits_to_save,
    }
    scio.savemat(mat_path, mat_dict)
    print(f"Saved {mat_path} (.mat) with keys: data, label, bits")


if __name__ == "__main__":
    main()
