import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Carrier maps ----------------
def carriers_11ag_fft64():
    data_bins = []
    pilot_bins = [-21, -7, 7, 21]
    used = list(range(-26, 0)) + list(range(1, 27))
    for k in used:
        if k not in pilot_bins:
            data_bins.append(k)
    return data_bins, pilot_bins

def carriers_generic(N):
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

def bin_to_fft_index(k, N):  # signed bin -> FFT index
    return k % N

# ---------------- IO ----------------
def load_bin_interleaved_float32(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 2 != 0:
        raise ValueError("BIN length is not even (I/Q).")
    i = raw[0::2]; q = raw[1::2]
    return (i + 1j * q).astype(np.complex64)

# ---------------- CP-based coarse alignment (optional) ----------------
def estimate_symbol_start_cp(iq: np.ndarray, N: int, cp_len: int, search_len: int = 200000) -> int:
    L = min(search_len, len(iq) - (N + cp_len) - 1)
    if L <= 0:
        return 0
    a = np.lib.stride_tricks.as_strided(iq, shape=(L, cp_len), strides=(iq.strides[0], iq.strides[0]))
    b = np.lib.stride_tricks.as_strided(iq[N:], shape=(L, cp_len), strides=(iq.strides[0], iq.strides[0]))
    metric = np.abs(np.einsum('ij,ij->i', a, np.conj(b)))
    return int(np.argmax(metric))

# ---------------- Plots: Time & Spectrum ----------------
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

# ---------------- Constellation: square M-QAM (no rotation/EQ) ----------------
def qam_constellation_points(M):
    """
    Square M-QAM ideal points, normalized to unit average symbol energy.
    Levels are ±1, ±3, ... up to ±(sqrt(M)-1), scaled by 1/sqrt((2/3)(M-1)).
    """
    m = int(np.sqrt(M))
    if m * m != M:
        raise ValueError("M must be a perfect square (e.g., 16, 64, 256).")
    levels = np.arange(-(m-1), (m-1)+1, 2, dtype=np.float32)  # e.g., [-3,-1,1,3]
    a = 1.0/np.sqrt((2.0/3.0)*(M-1))  # normalization for unit Es
    xs, ys = np.meshgrid(levels, levels)
    pts = a * (xs.flatten() + 1j*ys.flatten())
    return pts, a, levels

def nearest_qam_distance(pts, ideals):
    """
    For each sample in 'pts', compute distance to nearest ideal QAM point.
    Returns distances and indices.
    """
    # Broadcast: [N,1] - [1,K] -> [N,K]
    diffs = pts[:, None] - ideals[None, :]
    d2 = (diffs.real**2 + diffs.imag**2)
    idx = np.argmin(d2, axis=1)
    dmin = np.sqrt(d2[np.arange(pts.shape[0]), idx])
    return dmin, idx

def plot_constellation_qam(iq, fs_hz, N, cp_len, data_bins, M,
                           max_syms=2000, title_note="",
                           power_keep_percentile=30,   # keep strongest Pxx
                           origin_eps_ratio=0.05,      # drop |X| < eps*median(|X|)
                           grid_tol_ratio=0.35):       # keep if dist_to_nearest <= grid_tol_ratio * step
    """
    QAM constellation with robust filtering:
      1) power percentile
      2) near-origin reject
      3) nearest-grid gating using a fraction of the step between adjacent QAM points
    No rotation/equalization; visualization is "as received" with outlier trimming.
    """
    sym_len = N + cp_len
    num_syms = min(iq.size // sym_len, max_syms)
    if num_syms == 0:
        return

    # Collect tones
    pts = []
    for s in range(num_syms):
        blk = iq[s*sym_len:(s+1)*sym_len]
        no_cp = blk[cp_len:cp_len+N]
        X = np.fft.fftshift(np.fft.fft(no_cp))
        for k in data_bins:
            pts.append(X[bin_to_fft_index(k, N)])
    pts = np.asarray(pts, dtype=np.complex64)
    if pts.size == 0:
        return

    # 1) Power filter
    power = (pts.real**2 + pts.imag**2)
    p_thresh = np.percentile(power, power_keep_percentile)
    keep = power >= p_thresh
    pts = pts[keep]
    if pts.size == 0:
        return

    # 2) Kill near-origin
    mags = np.abs(pts)
    med = np.median(mags) + 1e-12
    pts = pts[mags > (origin_eps_ratio * med)]
    if pts.size == 0:
        return

    # # 3) Nearest-grid gating for M-QAM
    # ideals, a, levels = qam_constellation_points(M)
    # # Adjacent grid spacing after normalization:
    # step = (levels[1] - levels[0]) * a if len(levels) > 1 else 0.0
    # dmin, _ = nearest_qam_distance(pts, ideals)
    # # keep points close to some ideal corner by distance gate
    # tol = grid_tol_ratio * max(step, 1e-6)
    # pts = pts[dmin <= tol]
    # if pts.size == 0:
    #     print("No Points")
    #     return

    # Normalize for plotting (cosmetic)
    pts = pts / (np.mean(np.abs(pts)) + 1e-12)

    plt.figure()
    plt.title(f"Constellation (Data Subcarriers, {M}-QAM) {title_note}\n"
              f"[power≥P{power_keep_percentile}, |X|>{origin_eps_ratio:.2f}·median, dist≤{grid_tol_ratio:.2f}·step]")
    plt.scatter(pts.real, pts.imag, s=6, alpha=0.6)
    plt.axhline(0, linewidth=1); plt.axvline(0, linewidth=1)
    plt.xlabel("I"); plt.ylabel("Q"); plt.grid(True)

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Plot M-QAM Wi-Fi OFDM (time, spectrum, constellation) from .bin")
    ap.add_argument("--infile", type=str, required=True, help="Path to .bin (float32 interleaved IQ)")
    # REQUIRED params
    ap.add_argument("--bandwidth", type=float, required=True, help="Channel bandwidth (Hz), e.g., 20e6")
    ap.add_argument("--fs", type=float, required=True, help="Sampling rate (Hz)")
    ap.add_argument("--coding_rate", type=str, required=True, help="Coding rate (e.g., 1/2, 3/4) [metadata only]")
    ap.add_argument("--fft", type=int, required=True, help="FFT size (e.g., 64 for 802.11a/g)")
    # Optional
    ap.add_argument("--qam", type=int, default=16, help="Square M-QAM order (e.g., 16, 64, 256)")
    ap.add_argument("--gi_fraction", type=float, default=0.25, help="Guard interval fraction (default 0.25)")
    ap.add_argument("--try_align", action="store_true", help="Try CP-based symbol alignment before plots")
    ap.add_argument("--max_syms", type=int, default=2000, help="Max symbols for constellation")
    ap.add_argument("--time_samples", type=int, default=5000, help="Samples to show in time plot")
    ap.add_argument("--nfft", type=int, default=65536, help="FFT size for spectrum plot")
    # Constellation filters
    ap.add_argument("--power_keep_percentile", type=int, default=30, help="Constellation power filter percentile")
    ap.add_argument("--origin_eps_ratio", type=float, default=0.05, help="Drop |X| < this*median(|X|)")
    ap.add_argument("--grid_tol_ratio", type=float, default=0.35, help="Distance gate as fraction of QAM step")
    args = ap.parse_args()

    path = Path(args.infile)
    iq = load_bin_interleaved_float32(path)

    fs = float(args.fs)
    N = int(args.fft)
    cp_len = int(round(args.gi_fraction * N))

    # data/pilot layout
    data_bins, _ = (carriers_11ag_fft64() if N == 64 else carriers_generic(N))

    # optional coarse CP alignment (for better per-symbol FFT)
    iq_for_plots = iq
    if args.try_align:
        start = estimate_symbol_start_cp(iq, N, cp_len)
        iq_for_plots = iq[start:]

    title_note = f"(BW={args.bandwidth:.0f} Hz, Fs={fs:.0f} Hz, CR={args.coding_rate}, FFT={N}, GI={args.gi_fraction:.2f})"

    # Time & Spectrum
    plot_time(iq_for_plots, fs, n=args.time_samples, title_note=title_note)
    plot_spectrum(iq_for_plots, fs, nfft=args.nfft, title_note=title_note)

    # Constellation (M-QAM)
    plot_constellation_qam(
        iq_for_plots, fs, N, cp_len, data_bins, args.qam,
        max_syms=args.max_syms, title_note=title_note,
        power_keep_percentile=args.power_keep_percentile,
        origin_eps_ratio=args.origin_eps_ratio,
        grid_tol_ratio=args.grid_tol_ratio
    )

    plt.show()

if __name__ == "__main__":
    main()