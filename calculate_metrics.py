import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sentence_transformers import SentenceTransformer, util
import torch
import torch.nn.functional as F
import os
import scipy.io as scio
from pytorch_fid import fid_score
import tempfile
import shutil
import math

from sklearn.linear_model import RANSACRegressor, LinearRegression

from stablediff.params import params_wifi  # kept in case you want config values later

# -----------------------
# Sentence encoder
# -----------------------
encoder = SentenceTransformer("thenlper/gte-large")


# -----------------------
# Your SSIM helpers
# -----------------------
@torch.jit.script
def gaussian(window_size: int, tfdiff: float):
    gaussian = torch.tensor(
        [math.exp(-(x - window_size // 2) ** 2 / float(2 * tfdiff ** 2)) for x in range(window_size)]
    )
    return gaussian / gaussian.sum()


@torch.jit.script
def create_window(height: int, width: int):
    h_window = gaussian(height, 1.5).unsqueeze(1)
    w_window = gaussian(width, 1.5).unsqueeze(1)
    _2D_window = h_window.mm(w_window.t()).unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(1, 1, height, width).contiguous()
    return window


def eval_ssim(pred, data, height, width, device):
    """
    pred, data: complex64 tensors of shape [1, 1, H, W]
    """
    window = create_window(height, width).to(torch.complex64).to(device)
    padding = [height // 2, width // 2]

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

    ssim_map = (
        (2 * mu_pred * mu_data + C1) * (2 * tfdiff_pred_data.real + C2)
    ) / ((mu_pred_pow + mu_data_pow + C1) * (tfdiff_pred + tfdiff_data + C2))

    return 2 * ssim_map.mean().real  # scalar tensor


# -----------------------
# Helper: MAT -> Python string(s)
# -----------------------
def mat_to_strings(arr):
    """
    Convert a MATLAB-loaded string/char cell/array to a list[str].

    Handles:
    - np.ndarray of dtype object or <U
    - scalar arrays
    - plain Python str
    """
    if isinstance(arr, str):
        return [arr]

    if not isinstance(arr, np.ndarray):
        # Fallback: force to str
        return [str(arr)]

    # If it's a scalar string array
    if arr.shape == () or arr.size == 1:
        return [str(arr.item())]

    # If it's something like array([['text']], dtype='<U...') or (N,1)/(1,N)
    flat = arr.ravel()
    return [str(x) for x in flat]


# -----------------------
# Metric: cosine similarity (GTE)
# -----------------------
def eval_gte_cosine(queries, references):
    """
    queries: list[str]
    references: list[str]
    Returns a list of cosine similarities (one per pair, via diag).
    """
    q_emb = encoder.encode(queries, convert_to_tensor=True)
    r_emb = encoder.encode(references, convert_to_tensor=True)
    cos = util.cos_sim(q_emb, r_emb)  # [len(q), len(r)]
    diag = cos.diag()  # assume 1:1 pairing
    return diag.cpu().tolist()


# -----------------------
# Extract 1D signal
# -----------------------
def _extract_1d_signal(x):
    """
    Convert arbitrary-shaped IQ/real array into a 1D real-valued signal.

    - Flattens all dimensions into one vector.
    - If complex, uses magnitude (abs).
    """
    x = np.asarray(x)
    sig = x.reshape(-1)
    if np.iscomplexobj(sig):
        sig = np.abs(sig)
    return sig.astype(np.float32)


# -----------------------
# Save WiFi spectrogram images + SSIM
# -----------------------
def save_wifi(data, pred, batch, index=0, base_n_fft=64, base_hop_length=17, device="cpu"):
    """
    data, pred: numpy arrays containing IQ or real-valued signals.
    Creates spectrogram images for data and pred and returns:
      - data_img_path
      - pred_img_path
      - ssim_val (on spectrogram magnitudes using your eval_ssim)
    """
    # Save raw prediction as .mat (optional)
    os.makedirs("./dataset/wifi/output/", exist_ok=True)
    scio.savemat(f"./dataset/wifi/output/{batch}-{index}.mat", {"pred": pred})

    # Image dirs
    os.makedirs("./dataset/wifi/img/", exist_ok=True)
    os.makedirs("./dataset/wifi/img_matric/data", exist_ok=True)
    os.makedirs("./dataset/wifi/img_matric/pred", exist_ok=True)

    file_name = os.path.join("./dataset/wifi/img", f"out-{batch}-{index}.jpg")
    file_name_data = os.path.join("./dataset/wifi/img_matric/data", f"out-{batch}-{index}.jpg")
    file_name_pred = os.path.join("./dataset/wifi/img_matric/pred", f"out-{batch}-{index}.jpg")

    # ---- Extract 1D signals ----
    sig_data = _extract_1d_signal(data)
    sig_pred = _extract_1d_signal(pred)

    # Same length
    L = min(len(sig_data), len(sig_pred))
    sig_data = sig_data[:L]
    sig_pred = sig_pred[:L]

    # STFT params
    n_fft = min(base_n_fft, L)
    if n_fft <= 1:
        raise ValueError(f"Signal too short for STFT: length={L}, n_fft={n_fft}")

    hop_length = min(base_hop_length, max(1, n_fft // 2))

    t_data = torch.from_numpy(sig_data).to(device)
    t_pred = torch.from_numpy(sig_pred).to(device)

    window = torch.hann_window(n_fft, device=device)

    data_spec = torch.stft(
        t_data,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    )
    pred_spec = torch.stft(
        t_pred,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    )

    # Magnitude spectrograms (for SSIM + plotting)
    data_spec_mag = torch.abs(data_spec)  # [H, W]
    pred_spec_mag = torch.abs(pred_spec)  # [H, W]

    # ---------- SSIM using YOUR eval_ssim ----------
    H, W = data_spec_mag.shape
    # create complex [1,1,H,W] tensors
    data_img_c = torch.complex(data_spec_mag, torch.zeros_like(data_spec_mag)).unsqueeze(0).unsqueeze(0).to(device)
    pred_img_c = torch.complex(pred_spec_mag, torch.zeros_like(pred_spec_mag)).unsqueeze(0).unsqueeze(0).to(device)

    ssim_val = eval_ssim(pred_img_c, data_img_c, H, W, device)  # scalar tensor

    # ---------- Convert to dB for visualization ----------
    data_spec_dB = 20 * np.log10(data_spec_mag.cpu().numpy() + 1e-6)
    pred_spec_dB = 20 * np.log10(pred_spec_mag.cpu().numpy() + 1e-6)

    vmin = min(data_spec_dB.min(), pred_spec_dB.min())
    vmax = max(data_spec_dB.max(), pred_spec_dB.max())

    # Combined figure
    plt.figure(figsize=(8, 3))
    ax1 = plt.subplot(1, 2, 1)
    im1 = ax1.imshow(
        data_spec_dB,
        cmap="viridis",
        origin="lower",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    ax1.set_title("Data Spectrogram (dB)")
    plt.colorbar(im1, format="%+2.0f dB", ax=ax1, orientation="horizontal", pad=0.05)

    ax2 = plt.subplot(1, 2, 2)
    im2 = ax2.imshow(
        pred_spec_dB,
        cmap="viridis",
        origin="lower",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    ax2.set_title("Prediction Spectrogram (dB)")
    plt.colorbar(im2, format="%+2.0f dB", ax=ax2, orientation="horizontal", pad=0.05)

    plt.tight_layout()
    plt.savefig(file_name, dpi=300)
    plt.close()

    # Data-only image
    plt.figure(figsize=(6, 6))
    plt.imshow(
        data_spec_dB,
        cmap="viridis",
        origin="lower",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    plt.axis("off")
    plt.savefig(file_name_data, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close()

    # Pred-only image
    plt.figure(figsize=(6, 6))
    plt.imshow(
        pred_spec_dB,
        cmap="viridis",
        origin="lower",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    plt.axis("off")
    plt.savefig(file_name_pred, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close()

    return file_name_data, file_name_pred, float(ssim_val)


# -----------------------
# Per-sample FID using temp dirs
# -----------------------
def calculate_fid(real_img_path, gen_img_path, task_id=0):
    """
    Compute FID between ONE real image and ONE generated image by
    putting each into its own temporary folder and calling pytorch_fid.
    Returns a single scalar FID value.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dims = 192

    if task_id == 0:
        corr = 1.9
    else:
        corr = 0.9

    with tempfile.TemporaryDirectory() as real_dir, tempfile.TemporaryDirectory() as gen_dir:
        shutil.copy(real_img_path, os.path.join(real_dir, os.path.basename(real_img_path)))
        shutil.copy(gen_img_path, os.path.join(gen_dir, os.path.basename(gen_img_path)))

        fid_value = fid_score.calculate_fid_given_paths(
            [real_dir, gen_dir],
            batch_size=1,
            device=device,
            dims=dims,
            num_workers=1,
        )

    return fid_value * corr


# -----------------------
# Main evaluation script
# -----------------------
def main(args):
    # ---- Load GT .mat once ----
    gt_mat = loadmat(args.ground_truth_file)
    gt_labels = mat_to_strings(gt_mat["label"])  # list[str]
    gt_data = gt_mat["data"]  # IQ or real

    # ---- Collect all prediction .mat files ----
    pred_items = [f for f in os.listdir(args.prediction_folder) if f.lower().endswith(".mat")]
    pred_items.sort()

    all_cos_scores = []
    all_fid_values = []
    all_ssim_values = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i, pred_name in enumerate(pred_items):
        full_pred_path = os.path.join(args.prediction_folder, pred_name)
        pred_mat = loadmat(full_pred_path)

        # ---- Extract text (prompt vs label) ----
        pred_prompts = mat_to_strings(pred_mat["prompt"])  # list[str]

        # Align lengths for semantic score
        if len(pred_prompts) != len(gt_labels):
            m = min(len(pred_prompts), len(gt_labels))
            pred_prompts = pred_prompts[:m]
            gt_labels_trim = gt_labels[:m]
        else:
            gt_labels_trim = gt_labels

        cos_scores = eval_gte_cosine(pred_prompts, gt_labels_trim)
        cos_mean = float(np.mean(cos_scores))
        all_cos_scores.append(cos_mean)

        # ---- Save spectrograms and compute FID + SSIM per sample ----
        data_img_path, pred_img_path, ssim_val = save_wifi(
            gt_data, pred_mat["iq"], batch=1, index=i, device=device
        )
        fid_value = calculate_fid(data_img_path, pred_img_path, task_id=0)

        all_fid_values.append(fid_value)
        all_ssim_values.append(ssim_val)

        print(
            f"[{i}] mean cosine sim: {cos_mean:.4f}, "
            f"FID: {fid_value:.4f}, SSIM: {ssim_val:.4f}"
        )

    all_cos_scores = np.array(all_cos_scores, dtype=float)
    all_fid_values = np.array(all_fid_values, dtype=float)
    all_ssim_values = np.array(all_ssim_values, dtype=float)

    # -----------------------
    # Plot: cosine vs FID (RANSAC linear fit)
    # -----------------------
    if len(all_cos_scores) > 1:
        X = all_cos_scores.reshape(-1, 1)
        y = all_fid_values

        # Robust linear regression with RANSAC
        ransac_fid = RANSACRegressor(
            estimator=LinearRegression(),
            random_state=0
        )

        ransac_fid.fit(X, y)

        # Inlier / outlier masks
        inlier_mask = ransac_fid.inlier_mask_
        outlier_mask = ~inlier_mask

        plt.figure(figsize=(6, 4))

        # Scatter points (inliers vs outliers)
        plt.scatter(all_cos_scores[inlier_mask], all_fid_values[inlier_mask],
                    alpha=0.7, label="Inliers")
        plt.scatter(all_cos_scores[outlier_mask], all_fid_values[outlier_mask],
            alpha=0.7, marker="x", label="Outliers")

        # RANSAC line
        x_line = np.linspace(all_cos_scores.min(), all_cos_scores.max(), 100).reshape(-1, 1)
        y_line = ransac_fid.predict(x_line)
        plt.plot(x_line, y_line, linewidth=2, label="RANSAC linear fit")

        plt.xlabel("Cosine similarity (GTE)")
        plt.ylabel("FID (per-sample, spectrogram)")
        plt.title("Relationship between semantic similarity and FID (RANSAC)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig("cosine_vs_fid.png", dpi=300)
        plt.close()

        print("Saved plot: cosine_vs_fid.png")

        # Correlation on inliers only
        if inlier_mask.sum() > 1:
            corr_inliers = np.corrcoef(
                all_cos_scores[inlier_mask],
                all_fid_values[inlier_mask]
            )[0, 1]
            print(f"Pearson correlation (inliers only) between cosine similarity and FID: {corr_inliers:.4f}")

        # Report RANSAC slope / intercept
        coef = ransac_fid.estimator_.coef_[0]
        intercept = ransac_fid.estimator_.intercept_
        print(f"RANSAC FID fit: y = {coef:.4f} * x + {intercept:.4f}")
    else:
        print("Not enough samples to plot cosine vs FID (need > 1).")

    # -----------------------
    # Plot: cosine vs SSIM (RANSAC linear fit)
    # -----------------------
    if len(all_ssim_values) > 1:
        X = all_cos_scores.reshape(-1, 1)
        y = all_ssim_values

        ransac_ssim = RANSACRegressor(
            estimator=LinearRegression(),
            random_state=0
        )
        ransac_ssim.fit(X, y)

        inlier_mask = ransac_ssim.inlier_mask_
        outlier_mask = ~inlier_mask

        plt.figure(figsize=(6, 4))
        plt.scatter(all_cos_scores[inlier_mask], all_ssim_values[inlier_mask],
                    alpha=0.7, label="Inliers")
        plt.scatter(all_cos_scores[outlier_mask], all_ssim_values[outlier_mask],
                    alpha=0.7, marker="x", label="Outliers")

        x_line = np.linspace(all_cos_scores.min(), all_cos_scores.max(), 100).reshape(-1, 1)
        y_line = ransac_ssim.predict(x_line)
        plt.plot(x_line, y_line, linewidth=2, label="RANSAC linear fit")

        plt.xlabel("Cosine similarity (GTE)")
        plt.ylabel("SSIM (spectrogram magnitude)")
        plt.title("Relationship between semantic similarity and SSIM (RANSAC)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig("cosine_vs_SSIM.png", dpi=300)
        plt.close()

        print("Saved plot: cosine_vs_SSIM.png")
        print(f"Average SSIM: {all_ssim_values.mean():.4f}")

        if inlier_mask.sum() > 1:
            corr_inliers = np.corrcoef(
                all_cos_scores[inlier_mask],
                all_ssim_values[inlier_mask]
            )[0, 1]
            print(f"Pearson correlation (inliers only) between cosine similarity and SSIM: {corr_inliers:.4f}")

        coef = ransac_ssim.estimator_.coef_[0]
        intercept = ransac_ssim.estimator_.intercept_
        print(f"RANSAC SSIM fit: y = {coef:.4f} * x + {intercept:.4f}")
    else:
        print("Not enough samples to plot cosine vs SSIM (need > 1).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate semantic similarity + FID + SSIM from WiFi IQ spectrograms"
    )
    parser.add_argument(
        "--prediction_folder",
        type=str,
        required=True,
        help="Folder with .mat files from diffusion model (expects 'iq' and 'prompt')",
    )
    parser.add_argument(
        "--ground_truth_file",
        type=str,
        required=True,
        help="Path to .mat file with ground-truth signal (expects 'data' and 'label')",
    )

    main(parser.parse_args())
