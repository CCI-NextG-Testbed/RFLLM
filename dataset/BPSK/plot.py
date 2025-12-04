import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Carrier maps
# ----------------------------
def carriers_11ag_fft64():
    # 802.11a/g: data = ±{1..26} \ {±7, ±21}; pilots at -21, -7, +7, +21
    data_bins = []
    pilot_bins = [-21, -7, 7, 21]
    used = list(range(-26, 0)) + list(range(1, 27))
    for k in used:
        if k not in pilot_bins:
            data_bins.append(k)
    return data_bins, pilot_bins

def carriers_generic(N):
    # Occupy ~80% of bins (skip DC & edges); add up to 4 pilot candidates.
    usable = int(round(0.8 * N))
    half = usable // 2
    neg = list(range(-half, 0))
    pos = list(range(1, half + 1))
    pilot_candidates = []
    if half >= 22:
        pilot_candidates = [-(half//3), -(half//6), (half//6), (half//3)]
    elif half >= 10:
        pilot_candidates = [-7, -3, 3, 7]
    pilots = [p for p in pilot_candidates if p in (neg + pos)]
    data = [k for k in (neg + pos) if k not in pilots]
    return data, pilots

def bin_to_fft_index(k, N):
    return k % N

# ----------------------------
# IO
# ----------------------------
def load_bin_interleaved_float32(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 2 != 0:
        raise ValueError("BIN length is not even (I/Q).")
    i = raw[0::2]
    q = raw[1::2]
    return (i + 1j * q).astype(np.complex64)

# ----------------------------
# Simple CP-based alignment (coarse)
# ----------------------------
def estimate_symbol_start_cp(iq: np.ndarray, N: int, cp_len: int, search_len: int = 200000) -> int:
    """
    Find an index n maximizing correlation between segment[n:n+cp] and segment[n+N:n+N+cp].
    Works best if the file contains contiguous OFDM symbols and enough SNR.
    """
    L = min(search_len, len(iq) - (N + cp_len) - 1)
    if L <= 0:
        return 0
    # Vectorized correlation metric
    a = np.lib.stride_tricks.as_strided(
        iq, shape=(L, cp_len), strides=(iq.strides[0], iq.strides[0])
    )
    b = np.lib.stride_tricks.as_strided(
        iq[N:], shape=(L, cp_len), strides=(iq.strides[0], iq.strides[0])
    )
    # Sum over CP window of a * conj(b)
    metric = np.abs(np.einsum('ij,ij->i', a, np.conj(b)))
    idx = int(np.argmax(metric))
    return idx

# ----------------------------
# Plots
# ----------------------------
def plot_time(iq, fs_hz, n=5000, title_note=""):
    n = min(n, iq.size)
    t = np.arange(n) / fs_hz
    plt.figure()
    plt.title(f"Time domain (first {n} samples) {title_note}")
    plt.plot(t, iq[:n].real, label="I")
    plt.plot(t, iq[:n].imag, label="Q")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend(); plt.grid(True)

def plot_spectrum(iq, fs_hz, nfft=65536, title_note=""):
    N = min(nfft, iq.size)
    X = np.fft.fftshift(np.fft.fft(iq[:N], n=N))
    f = np.fft.fftshift(np.fft.fftfreq(N, d=1.0/fs_hz))
    psd_db = 20 * np.log10(np.abs(X) + 1e-12)
    plt.figure()
    plt.title(f"Spectrum ({N}-point FFT) {title_note}")
    plt.plot(f, psd_db)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)

def plot_constellation(iq, fs_hz, N, cp_len, data_bins,
                       max_syms=2000, title_note="",
                       amp_floor_ratio=0.25, imag_ratio_tol=0.2):
    
    sym_len = N + cp_len
    num_syms_total = iq.size // sym_len
    num_syms = min(num_syms_total, max_syms)
    if num_syms == 0:
        return

    pts_all = []

    for s in range(num_syms):
        blk = iq[s*sym_len:(s+1)*sym_len]
        no_cp = blk[cp_len:cp_len+N]
        X = np.fft.fftshift(np.fft.fft(no_cp))

        # Collect DATA subcarriers (complex values)
        d = np.array([X[(k % N)] for k in data_bins], dtype=np.complex64)
        if d.size == 0:
            continue

        # Per-symbol amplitude floor to remove guards/near-zero junk
        mags = np.abs(d)
        m_med = np.median(mags) + 1e-12
        keep = mags >= (amp_floor_ratio * m_med)
        d = d[keep]
        if d.size == 0:
            continue

        # Per-symbol common phase estimate via circular mean (pilot-less CPE)
        u = d / (np.abs(d) + 1e-12)             # normalize to unit circle
        phi = np.angle(np.mean(u))              # common phase angle
        d_corr = d * np.exp(-1j * phi)          # rotate onto the real axis line

        # Off-axis rejection: keep points close to the BPSK line
        re = np.abs(d_corr.real) + 1e-12
        im = np.abs(d_corr.imag)
        keep2 = im <= (imag_ratio_tol * re)     # e.g., ±11.3° cone if tol=0.2
        d_corr = d_corr[keep2]
        if d_corr.size == 0:
            continue

        # Normalize power (optional, cosmetic)
        d_corr = d_corr / (np.mean(np.abs(d_corr)) + 1e-12)

        pts_all.append(d_corr)

    if not pts_all:
        return
    pts = np.concatenate(pts_all)

    # Final plot: should be two tight clusters near ±1 on real axis, imag ~ 0
    plt.figure()
    plt.title(f"BPSK Constellation (DATA ONLY, per-symbol phase aligned) {title_note}")
    plt.scatter(pts.real, pts.imag, s=6, alpha=0.6)
    plt.axhline(0, linewidth=1, linestyle="--")
    plt.axvline(0, linewidth=1, linestyle="--")
    plt.xlabel("I"); plt.ylabel("Q"); plt.grid(True)

def plot_constellation(iq, fs_hz, N, cp_len, data_bins,
                       max_syms=2000, title_note="",
                       power_keep_percentile=50,      # keep power >= P30 (looser; raise to 50–70 to be stricter)
                       origin_eps_ratio=0.05,         # drop |X| < 0.05 * median(|X|)
                       theta_deg=12.0,                # keep only angles within ±theta of I-axis (0° or 180°)
                       real_min_ratio=0.05,           # require |Re| >= 0.05 * median(|X|)
                       tone_consistency=True,         # drop tones whose median angle is off-axis
                       tone_theta_deg=8.0):           # per-tone median angle window

    sym_len = N + cp_len
    num_syms = min(iq.size // sym_len, max_syms)
    if num_syms == 0:
        return

    pts = []
    tone_ids = []   # track which data_bin each sample came from (for tone-consistency)
    for s in range(num_syms):
        blk = iq[s*sym_len:(s+1)*sym_len]
        no_cp = blk[cp_len:cp_len+N]
        X = np.fft.fftshift(np.fft.fft(no_cp))
        for idx, k in enumerate(data_bins):
            pts.append(X[k % N])
            tone_ids.append(idx)
    pts = np.asarray(pts, dtype=np.complex64)
    tone_ids = np.asarray(tone_ids, dtype=np.int32)
    if pts.size == 0:
        return

    # --- Power filter ---
    power = (pts.real**2 + pts.imag**2)
    p_thresh = np.percentile(power, power_keep_percentile)
    keep = power >= p_thresh
    pts = pts[keep]; tone_ids = tone_ids[keep]
    if pts.size == 0:
        return

    # --- Remove near-origin points ---
    mags = np.abs(pts)
    med_mag = np.median(mags) + 1e-12
    pts = pts[mags > origin_eps_ratio * med_mag]
    tone_ids = tone_ids[mags > origin_eps_ratio * med_mag]
    if pts.size == 0:
        return

    # --- Require some real energy (avoid vertical/near-Q points) ---
    keep = np.abs(pts.real) >= (real_min_ratio * med_mag)
    pts = pts[keep]; tone_ids = tone_ids[keep]
    if pts.size == 0:
        return

    # --- Axis-angle gate: keep samples near 0° or 180° ---
    ang = np.angle(pts)                      # (-π, π]
    # distance to the nearest I-axis (0 or π)
    ang_to_I = np.minimum(np.abs(ang), np.abs(np.pi - np.abs(ang)))
    keep = ang_to_I <= np.deg2rad(theta_deg)
    pts = pts[keep]; tone_ids = tone_ids[keep]
    if pts.size == 0:
        return
    
    # --- Normalize and plot (unchanged) ---
    pts = pts / (np.mean(np.abs(pts)) + 1e-12)

    plt.figure()
    filt_text = (f"[power≥P{power_keep_percentile}, |X|>{origin_eps_ratio:.2f}·median, "
                 f"angle≤±{theta_deg:.1f}°, |Re|>{real_min_ratio:.2f}·median"
                 + (f", tone±{tone_theta_deg:.1f}°" if tone_consistency else "") + "]")
    plt.title(f"Constellation (Data Subcarriers, BPSK) {title_note}\n{filt_text}")
    plt.scatter(pts.real, pts.imag, s=6, alpha=0.6)
    plt.axhline(0, linewidth=1); plt.axvline(0, linewidth=1)
    plt.xlabel("I"); plt.ylabel("Q"); plt.grid(True)
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Plot BPSK Wi-Fi-style OFDM from a .bin (float32 IQ interleaved)")
    ap.add_argument("--infile", type=str, required=True, help="Path to .bin (float32 interleaved IQ)")
    # REQUIRED parameters (per user request)
    ap.add_argument("--bandwidth", type=float, required=True, help="Channel bandwidth (Hz), e.g., 20e6")
    ap.add_argument("--fs", type=float, required=True, help="Sampling rate (Hz), typically equals bandwidth")
    ap.add_argument("--coding_rate", type=str, required=True, help="Coding rate (e.g., 1/2, 3/4) [for metadata]")
    ap.add_argument("--fft", type=int, required=True, help="FFT size (e.g., 64 for 802.11a/g)")
    # Optional
    ap.add_argument("--gi_fraction", type=float, default=0.25, help="Guard interval as fraction of FFT (default 0.25)")
    ap.add_argument("--try_align", action="store_true", help="Try CP-based symbol alignment before plotting constellation")
    ap.add_argument("--max_syms", type=int, default=2000, help="Max symbols to use for constellation")
    ap.add_argument("--time_samples", type=int, default=5000, help="How many samples to show in time plot")
    args = ap.parse_args()

    path = Path(args.infile)
    iq = load_bin_interleaved_float32(path)

    fs = float(args.fs)
    N = int(args.fft)
    cp_len = int(round(args.gi_fraction * N))

    # Choose carrier map
    data_bins, pilot_bins = (carriers_11ag_fft64() if N == 64 else carriers_generic(N))

    # Optional coarse CP alignment for better constellation (does nothing to time/spectrum plots)
    align_note = ""
    iq_for_const = iq
    if args.try_align:
        start = estimate_symbol_start_cp(iq, N, cp_len)
        align_note = f"[aligned @ {start}]"
        iq_for_const = iq[start:]

    title_note = f"(BW={args.bandwidth:.0f} Hz, Fs={fs:.0f} Hz, CR={args.coding_rate}, FFT={N}, GI={args.gi_fraction:.2f})"

    # Plots
    plot_time(iq, fs, n=args.time_samples, title_note=title_note)
    plot_spectrum(iq, fs, title_note=title_note)
    plot_constellation(iq_for_const, fs, N, cp_len, data_bins, max_syms=args.max_syms,
                       title_note=f"{title_note} {align_note}")

    plt.show()

if __name__ == "__main__":
    main()