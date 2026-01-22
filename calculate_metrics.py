import argparse
import os
import math
import shutil
import tempfile

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import spectrogram

import torch
import torch.nn.functional as F

from pytorch_fid import fid_score
from sentence_transformers import CrossEncoder


# -----------------------
# Cross-Encoder STS scorer (semantic)
# -----------------------
def build_cross_encoder(model_name: str, device: str):
    # device: "cuda" or "cpu"
    return CrossEncoder(model_name, device=device)


def cross_encoder_scores(ce: CrossEncoder, prompts: list, labels: list) -> np.ndarray:
    """
    Returns per-pair STS scores for (prompt_i, label_i).
    """
    pairs = list(zip(prompts, labels))
    scores = ce.predict(pairs)
    return np.asarray(scores, dtype=float)


# -----------------------
# Helper: MAT -> Python string(s)
# -----------------------
def mat_to_strings(arr):
    """
    Convert a MATLAB-loaded string/char cell/array to a list[str].
    """
    if isinstance(arr, str):
        return [arr]

    if not isinstance(arr, np.ndarray):
        return [str(arr)]

    if arr.shape == () or arr.size == 1:
        return [str(arr.item())]

    flat = arr.ravel()
    return [str(x) for x in flat]


# -----------------------
# Your SSIM helpers (unchanged logic)
# -----------------------
@torch.jit.script
def gaussian(window_size: int, tfdiff: float):
    g = torch.tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * tfdiff ** 2)) for x in range(window_size)])
    return g / g.sum()


@torch.jit.script
def create_window(height: int, width: int):
    h_window = gaussian(height, 1.5).unsqueeze(1)
    w_window = gaussian(width, 1.5).unsqueeze(1)
    _2D_window = h_window.mm(w_window.t()).unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(1, 1, height, width).contiguous()
    return window


def eval_ssim(pred, data, height, width, device):
    """
    pred, data: complex tensors of shape [1, 1, H, W]
    """
    window = create_window(height, width).to(torch.complex64).to(device)
    padding = (height // 2, width // 2)

    mu_pred = F.conv2d(pred, window, padding=padding, groups=1)
    mu_data = F.conv2d(data, window, padding=padding, groups=1)

    mu_pred_pow = mu_pred.pow(2.0)
    mu_data_pow = mu_data.pow(2.0)
    mu_pred_data = mu_pred * mu_data

    tfdiff_pred = F.conv2d(pred * pred, window, padding=padding, groups=1) - mu_pred_pow
    tfdiff_data = F.conv2d(data * data, window, padding=padding, groups=1) - mu_data_pow
    tfdiff_pred_data = F.conv2d(pred * data, window, padding=padding, groups=1) - mu_pred_data

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu_pred * mu_data + C1) * (2 * tfdiff_pred_data.real + C2)) / (
        (mu_pred_pow + mu_data_pow + C1) * (tfdiff_pred + tfdiff_data + C2)
    )
    return 2 * ssim_map.mean().real


# -----------------------
# IQ extraction utilities
# -----------------------
def extract_complex_iq(x: np.ndarray) -> np.ndarray:
    """
    Returns a 1D complex IQ vector from common .mat shapes:
      - complex array already
      - real/imag in last dim (..,2)
    """
    x = np.asarray(x)

    # If already complex, just flatten
    if np.iscomplexobj(x):
        return x.reshape(-1).astype(np.complex64)

    # If last dimension looks like [I,Q]
    if x.ndim >= 2 and x.shape[-1] == 2:
        i = x[..., 0].reshape(-1)
        q = x[..., 1].reshape(-1)
        return (i + 1j * q).astype(np.complex64)

    # Otherwise treat as real-only (fallback)
    return x.reshape(-1).astype(np.float32).astype(np.complex64)


# -----------------------
# SciPy spectrogram + saving (same-size GT vs Pred)
# -----------------------
def compute_scipy_spectrogram_db(
    iq: np.ndarray,
    fs: float,
    nperseg: int,
    noverlap: int,
    eps: float = 1e-12,
):
    """
    Returns:
      f_shift_mhz: (F,) frequency axis in MHz (fftshifted, centered at DC)
      t: (T,) time axis in seconds
      S_db: (F,T) dB magnitude spectrogram (fftshifted)
    """
    iq = extract_complex_iq(iq)

    f, t, S = spectrogram(
        iq,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=False,     # keep negative freqs so we can fftshift
        scaling="density",
        mode="magnitude",          # magnitude spectrogram
    )

    S = np.fft.fftshift(S, axes=0)
    f = np.fft.fftshift(f)

    S_db = 20.0 * np.log10(S + eps)
    f_shift_mhz = f / 1e6
    return f_shift_mhz, t, S_db


def crop_to_same_shape(A: np.ndarray, B: np.ndarray):
    """
    Crops two (F,T) matrices to the same (minF, minT).
    """
    Fm = min(A.shape[0], B.shape[0])
    Tm = min(A.shape[1], B.shape[1])
    return A[:Fm, :Tm], B[:Fm, :Tm]


def save_pair_spectrogram_images(
    S_gt_db: np.ndarray,
    S_pred_db: np.ndarray,
    out_dir: str,
    base_name: str,
    vmin: float = None,
    vmax: float = None,
):
    """
    Saves:
      - combined figure (side-by-side)
      - gt-only image
      - pred-only image

    Returns (gt_img_path, pred_img_path, combined_path)
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "img_metric/gt"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "img_metric/pred"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "img"), exist_ok=True)

    gt_path = os.path.join(out_dir, "img_metric/gt", f"{base_name}_gt.png")
    pred_path = os.path.join(out_dir,"img_metric/pred", f"{base_name}_pred.png")
    combo_path = os.path.join(out_dir, "img", f"{base_name}_pair.png")

    # Ensure exact same matrix size
    S_gt_db, S_pred_db = crop_to_same_shape(S_gt_db, S_pred_db)

    # Shared color scale
    if vmin is None:
        vmin = float(min(S_gt_db.min(), S_pred_db.min()))
    if vmax is None:
        vmax = float(max(S_gt_db.max(), S_pred_db.max()))

    # --- GT-only (no axes) ---
    plt.figure(figsize=(6, 6))
    plt.imshow(S_gt_db, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    plt.axis("off")
    plt.savefig(gt_path, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close()

    # --- Pred-only (no axes) ---
    plt.figure(figsize=(6, 6))
    plt.imshow(S_pred_db, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    plt.axis("off")
    plt.savefig(pred_path, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close()

    # --- Side-by-side (with titles + colorbars) ---
    plt.figure(figsize=(10, 4))
    ax1 = plt.subplot(1, 2, 1)
    im1 = ax1.imshow(S_gt_db, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax1.set_title("GT Spectrogram (dB)")
    plt.colorbar(im1, ax=ax1, orientation="horizontal", pad=0.08)

    ax2 = plt.subplot(1, 2, 2)
    im2 = ax2.imshow(S_pred_db, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax2.set_title("Pred Spectrogram (dB)")
    plt.colorbar(im2, ax=ax2, orientation="horizontal", pad=0.08)

    plt.tight_layout()
    plt.savefig(combo_path, dpi=300)
    plt.close()

    return gt_path, pred_path, combo_path


# -----------------------
# FID (per-sample: one GT img vs one Pred img)
# -----------------------
def calculate_fid_onepair(gt_img_path: str, pred_img_path: str, dims: int = 192):
    """
    Per-sample FID using pytorch_fid:
      - put gt image in its own temp folder
      - put pred image in its own temp folder
      - run FID over those folders
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with tempfile.TemporaryDirectory() as real_dir, tempfile.TemporaryDirectory() as gen_dir:
        shutil.copy(gt_img_path, os.path.join(real_dir, os.path.basename(gt_img_path)))
        shutil.copy(pred_img_path, os.path.join(gen_dir, os.path.basename(pred_img_path)))

        fid_value = fid_score.calculate_fid_given_paths(
            [real_dir, gen_dir],
            batch_size=1,
            device=device,
            dims=dims,
            num_workers=1,
        )

    return float(fid_value)


# -----------------------
# SSIM on spectrogram magnitude (using your eval_ssim)
# -----------------------
def ssim_on_spectrogram_db(S_gt_db: np.ndarray, S_pred_db: np.ndarray, device: str):
    """
    Computes SSIM on magnitude-like images derived from dB arrays.
    We convert dB -> linear magnitude proxy to keep values positive and stable.

    Returns scalar float.
    """
    S_gt_db, S_pred_db = crop_to_same_shape(S_gt_db, S_pred_db)
    H, W = S_gt_db.shape

    # Convert dB back to linear-ish magnitude (positive)
    # (Any monotonic transform works; this is stable.)
    gt_mag = np.power(10.0, S_gt_db / 20.0).astype(np.float32)
    pr_mag = np.power(10.0, S_pred_db / 20.0).astype(np.float32)

    gt_t = torch.from_numpy(gt_mag).to(device)
    pr_t = torch.from_numpy(pr_mag).to(device)

    gt_c = torch.complex(gt_t, torch.zeros_like(gt_t)).unsqueeze(0).unsqueeze(0)
    pr_c = torch.complex(pr_t, torch.zeros_like(pr_t)).unsqueeze(0).unsqueeze(0)

    val = eval_ssim(pr_c, gt_c, H, W, device=torch.device(device))
    return float(val)


# -----------------------
# Plot helpers
# -----------------------
def scatter_with_line(x, y, xlabel, ylabel, title, out_path):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, alpha=0.7, label="Samples")

    if len(x) >= 2 and np.isfinite(x).all() and np.isfinite(y).all():
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = m * x_line + b
        plt.plot(x_line, y_line, linewidth=2, label="Linear fit")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# -----------------------
# Main
# -----------------------
def main(args):
    # Devices
    ce_device = "cuda" if (args.ce_device == "auto" and torch.cuda.is_available()) else (
        "cpu" if args.ce_device == "auto" else args.ce_device
    )
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build cross-encoder
    ce = build_cross_encoder(args.cross_encoder, device=ce_device)

    # Load GT
    gt_mat = loadmat(args.ground_truth_file)
    gt_data = gt_mat["data"]
    gt_labels = mat_to_strings(gt_mat["label"])

    # Prediction files
    pred_files = sorted([f for f in os.listdir(args.prediction_folder) if f.lower().endswith(".mat")])
    if len(pred_files) == 0:
        raise RuntimeError(f"No .mat files found in {args.prediction_folder}")

    # Output dirs
    os.makedirs(args.out_dir, exist_ok=True)

    semantic_means = []
    fid_values = []
    ssim_values = []

    # Precompute GT spectrogram once (same parameters for all)
    _, _, S_gt_db_full = compute_scipy_spectrogram_db(
        gt_data, fs=args.fs, nperseg=args.nperseg, noverlap=args.noverlap
    )

    for i, fname in enumerate(pred_files):
        pred_mat = loadmat(os.path.join(args.prediction_folder, fname))
        pred_iq = pred_mat["iq"]
        pred_prompts = mat_to_strings(pred_mat["prompt"])

        # Align prompt/label lengths
        m = min(len(pred_prompts), len(gt_labels))
        pred_prompts = pred_prompts[:m]
        gt_labels_trim = gt_labels[:m]

        # Semantic score (CrossEncoder STS)
        sts = cross_encoder_scores(ce, pred_prompts, gt_labels_trim)
        sts_mean = float(np.mean(sts))
        semantic_means.append(sts_mean)

        # Pred spectrogram
        _, _, S_pred_db_full = compute_scipy_spectrogram_db(
            pred_iq, fs=args.fs, nperseg=args.nperseg, noverlap=args.noverlap
        )

        # Crop to same shape and compute shared vmin/vmax
        S_gt_db, S_pred_db = crop_to_same_shape(S_gt_db_full, S_pred_db_full)
        vmin = float(min(S_gt_db.min(), S_pred_db.min()))
        vmax = float(max(S_gt_db.max(), S_pred_db.max()))

        # Save images (same size + same scale)
        base = f"sample_{i:04d}"
        gt_img, pred_img, combo_img = save_pair_spectrogram_images(
            S_gt_db, S_pred_db, out_dir=args.out_dir, base_name=base, vmin=vmin, vmax=vmax
        )

        # FID on this pair
        fid = calculate_fid_onepair(gt_img, pred_img, dims=args.fid_dims)
        fid_values.append(fid)

        # SSIM on spectrogram (using your eval_ssim)
        ssim = ssim_on_spectrogram_db(S_gt_db, S_pred_db, device=torch_device)
        ssim_values.append(ssim)

        print(f"[{i}] {fname}")
        print(f"    STS(mean)={sts_mean:.4f} | FID={fid:.4f} | SSIM={ssim:.4f}")
        print(f"    saved: {combo_img}")

    semantic_means = np.asarray(semantic_means, dtype=float)
    fid_values = np.asarray(fid_values, dtype=float)
    ssim_values = np.asarray(ssim_values, dtype=float)

    # Plots
    plot1 = os.path.join(args.out_dir, "semantic_vs_fid.png")
    plot2 = os.path.join(args.out_dir, "semantic_vs_ssim.png")

    scatter_with_line(
        semantic_means, fid_values,
        xlabel="Semantic similarity (CrossEncoder STS)",
        ylabel="FID (spectrogram images, per-sample)",
        title="Semantic similarity vs FID",
        out_path=plot1,
    )

    scatter_with_line(
        semantic_means, ssim_values,
        xlabel="Semantic similarity (CrossEncoder STS)",
        ylabel="SSIM (spectrogram magnitude)",
        title="Semantic similarity vs SSIM",
        out_path=plot2,
    )

    # Quick summary
    if len(semantic_means) >= 2:
        corr_fid = float(np.corrcoef(semantic_means, fid_values)[0, 1])
        corr_ssim = float(np.corrcoef(semantic_means, ssim_values)[0, 1])
        print("\n=== Summary ===")
        print(f"Saved: {plot1}")
        print(f"Saved: {plot2}")
        print(f"Pearson corr (semantic, FID):  {corr_fid:.4f}")
        print(f"Pearson corr (semantic, SSIM): {corr_ssim:.4f}")
        print(f"FID mean/std:  {fid_values.mean():.4f} / {fid_values.std():.4f}")
        print(f"SSIM mean/std: {ssim_values.mean():.4f} / {ssim_values.std():.4f}")
    else:
        print("\nNot enough samples for correlation (need >=2).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic similarity vs FID/SSIM using SciPy spectrograms (with fs).")

    parser.add_argument("--prediction_folder", type=str, required=True,
                        help="Folder containing prediction .mat files (expects keys: 'iq' and 'prompt').")
    parser.add_argument("--ground_truth_file", type=str, required=True,
                        help="Ground-truth .mat file (expects keys: 'data' and 'label').")

    parser.add_argument("--fs", type=float, required=True,
                        help="Sampling frequency in Hz (e.g., 40000000).")

    parser.add_argument("--nperseg", type=int, default=512,
                        help="Spectrogram FFT window size (try 256/512/1024).")
    parser.add_argument("--noverlap", type=int, default=384,
                        help="Spectrogram overlap (typically 50-75% of nperseg).")

    parser.add_argument("--cross_encoder", type=str, default="cross-encoder/stsb-roberta-large",
                        help="CrossEncoder STS model name.")
    parser.add_argument("--ce_device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Device for CrossEncoder inference.")

    parser.add_argument("--fid_dims", type=int, default=192,
                        help="FID feature dims (must match your Inception choice in pytorch-fid).")

    parser.add_argument("--out_dir", type=str, default="./dataset/wifi/",
                        help="Output directory for plots and spectrogram images.")

    main(parser.parse_args())