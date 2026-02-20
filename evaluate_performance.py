#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path
from typing import Optional
import numpy as np


# ==========================================================
# IO
# ==========================================================

def load_bits(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix.lower() == ".npy":
        b = np.load(p)
        b = np.asarray(b).astype(np.uint8).ravel()
    else:
        b = np.fromfile(p, dtype=np.uint8).ravel()
    if b.size == 0:
        raise ValueError("Empty bits file")
    return (b & 1).astype(np.uint8)


def load_iq(path: str, var: Optional[str] = None) -> np.ndarray:
    p = Path(path)
    suf = p.suffix.lower()

    if suf == ".npy":
        x = np.load(p)
        x = np.asarray(x)
        if np.iscomplexobj(x):
            return x.astype(np.complex64).ravel()
        if x.ndim == 2 and x.shape[1] == 2:
            return (x[:, 0] + 1j * x[:, 1]).astype(np.complex64).ravel()
        raise ValueError("Unsupported npy IQ format (expected complex array or Nx2 I/Q)")

    if suf == ".mat":
        import scipy.io as scio
        m = scio.loadmat(p)
        if var is None:
            for k in ("data", "iq", "IQ", "x"):
                if k in m:
                    var = k
                    break
        if var not in m:
            raise ValueError(f"MAT variable not found. Keys={list(m.keys())}")
        x = np.asarray(m[var]).squeeze()
        if np.iscomplexobj(x):
            return x.astype(np.complex64).ravel()
        if x.ndim == 2 and x.shape[1] == 2:
            return (x[:, 0] + 1j * x[:, 1]).astype(np.complex64).ravel()
        raise ValueError("Unsupported MAT IQ format (expected complex array or Nx2 I/Q)")

    # default: raw float32 interleaved IQ (I,Q,I,Q,...)
    raw = np.fromfile(p, dtype=np.float32)
    if raw.size % 2 != 0:
        raise ValueError("IQ bin must have even number of float32 values (I,Q,...)")
    return (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)


# ==========================================================
# Gray helpers
# ==========================================================

def binary_to_gray(b: np.ndarray) -> np.ndarray:
    b = b.astype(np.uint32)
    return (b ^ (b >> 1)).astype(np.uint32)


def int_to_bits_msb(x: np.ndarray, k: int) -> np.ndarray:
    x = x.astype(np.uint32)
    shifts = np.arange(k - 1, -1, -1, dtype=np.uint32)
    return ((x[:, None] >> shifts[None, :]) & 1).astype(np.uint8)


# ==========================================================
# Normalization
# ==========================================================

def norm_to_unit_avg_power(x: np.ndarray) -> np.ndarray:
    p = float(np.mean(np.abs(x) ** 2))
    if p <= 0.0:
        return x.astype(np.complex64)
    return (x / np.sqrt(p)).astype(np.complex64)


# ==========================================================
# Demods
# ==========================================================

def demod_bpsk(sym: np.ndarray) -> np.ndarray:
    # 1 if I>=0 else 0
    return (np.real(sym) >= 0).astype(np.uint8)


def demod_qpsk(sym: np.ndarray) -> np.ndarray:
    # [b0,b1] where b0 -> I sign, b1 -> Q sign
    b0 = (np.real(sym) >= 0).astype(np.uint8)
    b1 = (np.imag(sym) >= 0).astype(np.uint8)
    return np.stack([b0, b1], axis=1).reshape(-1)


def demod_8psk_gray(sym: np.ndarray) -> np.ndarray:
    """
    8PSK Gray demod:
      idx = round(angle * 8 / 2pi) mod 8
      g = binary_to_gray(idx)
      bits = MSB-first (3 bits)
    """
    M = 8
    ang = np.angle(sym).astype(np.float64)
    idx = (np.round(ang * M / (2.0 * np.pi)).astype(np.int64) % M).astype(np.uint32)
    g = binary_to_gray(idx)
    bits = int_to_bits_msb(g, 3)
    return bits.reshape(-1)


def _gray_to_binary_arr(g: np.ndarray) -> np.ndarray:
    b = g.astype(np.uint32).copy()
    shift = 1
    while shift < 32:
        b ^= (b >> shift)
        shift <<= 1
    return b


def _axis_to_gray_bits(v: np.ndarray, levels: int, bits_per_axis: int) -> np.ndarray:
    # Quantize to nearest odd PAM levels: -(L-1), ..., +(L-1)
    idx = np.round((v + (levels - 1)) / 2.0).astype(np.int64)
    idx = np.clip(idx, 0, levels - 1).astype(np.uint32)
    g = binary_to_gray(idx)
    return int_to_bits_msb(g, bits_per_axis)  # [N, bits_per_axis]


def demod_square_qam_gray(sym: np.ndarray, M: int) -> np.ndarray:
    """
    Gray-demod for square QAM (16/64/256), matching create_training_samples.py mapping.
    """
    if M not in (16, 64, 256):
        raise ValueError("M must be one of 16, 64, 256")
    L = int(np.sqrt(M))
    k = int(np.log2(M))
    k2 = k // 2

    # Normalize measured symbols to unit average power before slicing.
    s = norm_to_unit_avg_power(sym)
    # For square QAM with odd levels, peak amp before normalization is (L-1).
    # Undo average-power normalization to return to odd-level grid domain.
    mean_level_pow = (2.0 / 3.0) * (L**2 - 1)
    scale = np.sqrt(mean_level_pow)
    x = np.real(s) * scale
    y = np.imag(s) * scale

    bI = _axis_to_gray_bits(x, levels=L, bits_per_axis=k2)
    bQ = _axis_to_gray_bits(y, levels=L, bits_per_axis=k2)
    return np.concatenate([bI, bQ], axis=1).reshape(-1).astype(np.uint8)


# ==========================================================
# AWGN
# ==========================================================

def add_awgn(sym: np.ndarray, ebn0_db: float, k: int, rng: np.random.Generator) -> np.ndarray:
    """
    Complex AWGN based on Eb/N0 using measured Es on symbol-rate samples.
      Es = mean(|sym|^2)
      N0 = Es/(k*EbN0)
      noise = sqrt(N0/2)*(nI + j*nQ)
    """
    ebn0 = 10.0 ** (ebn0_db / 10.0)
    Es = float(np.mean(np.abs(sym) ** 2))
    N0 = Es / (float(k) * ebn0)
    sigma = np.sqrt(N0 / 2.0)
    n = (rng.standard_normal(sym.shape) + 1j * rng.standard_normal(sym.shape)).astype(np.complex64) * sigma
    return (sym + n).astype(np.complex64)


# ==========================================================
# MAIN
# ==========================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iq", required=True, help="IQ: .npy/.mat/.bin(float32 interleaved I,Q)")
    ap.add_argument("--bits", required=True, help="Bits: .bin uint8 (0/1) or .npy")
    ap.add_argument("--mod", required=True, choices=["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "256QAM"])
    ap.add_argument("--ebn0_start", type=float, default=-5.0)
    ap.add_argument("--ebn0_stop", type=float, default=20.0)
    ap.add_argument("--ebn0_step", type=float, default=1.0)
    ap.add_argument("--csv", default="ber.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mat_var", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    bits = load_bits(args.bits)
    sym = norm_to_unit_avg_power(load_iq(args.iq, var=args.mat_var))

    mod = args.mod.upper()
    if mod == "BPSK":
        k = 1
        demod = demod_bpsk
    elif mod == "QPSK":
        k = 2
        demod = demod_qpsk
    elif mod == "8PSK":
        k = 3
        demod = demod_8psk_gray
    elif mod == "16QAM":
        k = 4
        demod = lambda z: demod_square_qam_gray(z, M=16)
    elif mod == "64QAM":
        k = 6
        demod = lambda z: demod_square_qam_gray(z, M=64)
    elif mod == "256QAM":
        k = 8
        demod = lambda z: demod_square_qam_gray(z, M=256)
    else:
        raise ValueError("Unsupported modulation")

    # Trim to match symbol count
    max_bits = min(bits.size, sym.size * k)
    max_bits -= (max_bits % k)
    n_sym = max_bits // k
    sym = sym[:n_sym]
    bits = bits[:n_sym * k]

    if n_sym == 0:
        raise ValueError("No data after trimming (check IQ/bits lengths)")

    rows = []
    ebn0_vals = np.arange(args.ebn0_start, args.ebn0_stop + 1e-9, args.ebn0_step, dtype=np.float64)

    for ebn0_db in ebn0_vals:
        y = add_awgn(sym, float(ebn0_db), k=k, rng=rng)
        bh = demod(y).astype(np.uint8)

        L = min(bh.size, bits.size)
        err = int(np.count_nonzero((bh[:L] ^ bits[:L]) & 1))
        ber = err / float(L) if L > 0 else 1.0

        print(f"Eb/N0={ebn0_db:6.2f} dB  BER={ber:.6e}  (errors={err}, Nbits={L})")
        rows.append((float(ebn0_db), float(ber), int(L), int(err)))

    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ebn0_db", "ber"])
        for r in rows:
            w.writerow([f"{r[0]:.6f}", f"{r[1]:.10e}"])

    print("Saved:", args.csv)


if __name__ == "__main__":
    main()
