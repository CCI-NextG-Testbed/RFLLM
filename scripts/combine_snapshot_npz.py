#!/usr/bin/env python3
import argparse
import glob
import math
import os

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np


def load_snapshots(snapshot_dir: str):
    pattern = os.path.join(snapshot_dir, "snapshot_epoch_*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No snapshot files found in {snapshot_dir}")
    snapshots = [np.load(path, allow_pickle=True) for path in files]
    return files, snapshots


def symbol_rate_view(x: np.ndarray, sps: int, max_symbols: int):
    x = np.asarray(x).reshape(-1)
    sps = max(1, int(sps))
    t = min(len(x) // sps, int(max_symbols))
    if t <= 0:
        return np.zeros((0,), dtype=np.complex64)
    return x[: t * sps].reshape(t, sps).mean(axis=1)


def render_gif(snapshot_dir: str, out_path: str, mods, max_symbols: int, fps: int):
    _, snapshots = load_snapshots(snapshot_dir)
    mods = [str(m).upper() for m in mods]
    mods = [m for m in mods if f"{m}_target" in snapshots[0].files and f"{m}_pred" in snapshots[0].files]
    if not mods:
        raise ValueError("No requested modulation keys were found in the snapshot files.")

    epochs = np.asarray([int(np.asarray(s["epoch"]).squeeze()) for s in snapshots], dtype=np.int64)

    nmods = len(mods)
    ncols = 2 if nmods > 1 else 1
    nrows = int(math.ceil(nmods / float(ncols)))
    fig, ax_grid = plt.subplots(nrows, ncols, figsize=(7.0 * ncols, 6.5 * nrows))
    axes = np.atleast_1d(ax_grid).ravel().tolist()
    for ax in axes[nmods:]:
        ax.set_visible(False)

    pred_scats = []
    for i, mod in enumerate(mods):
        s0 = snapshots[0]
        target = np.asarray(s0[f"{mod}_target"])
        pred = np.asarray(s0[f"{mod}_pred"])
        sps = int(np.asarray(s0[f"{mod}_sps"]).squeeze())

        target_sym = symbol_rate_view(target, sps=sps, max_symbols=max_symbols)
        pred_sym = symbol_rate_view(pred, sps=sps, max_symbols=max_symbols)

        axes[i].scatter(np.real(target_sym), np.imag(target_sym), s=10, c="gray", alpha=0.45, label="Tx")
        pred_sc = axes[i].scatter(np.real(pred_sym), np.imag(pred_sym), s=10, c="tab:orange", alpha=0.9, label="Pred")
        pred_scats.append((pred_sc, sps))

        axes[i].axhline(0, color="k", lw=0.6, ls="--")
        axes[i].axvline(0, color="k", lw=0.6, ls="--")
        axes[i].set_title(f"{mod} Symbols")
        axes[i].set_xlabel("I")
        axes[i].set_ylabel("Q")
        axes[i].set_aspect("equal", adjustable="box")
        axes[i].set_xlim(-2.0, 2.0)
        axes[i].set_ylim(-2.0, 2.0)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc="upper right", fontsize=8)

    title = fig.suptitle("", fontsize=12)

    def update(k):
        s = snapshots[k]
        for i, mod in enumerate(mods):
            pred = np.asarray(s[f"{mod}_pred"])
            pred_sc, sps = pred_scats[i]
            pred_sym = symbol_rate_view(pred, sps=sps, max_symbols=max_symbols)
            pts = np.column_stack([np.real(pred_sym), np.imag(pred_sym)]) if len(pred_sym) else np.zeros((0, 2))
            pred_sc.set_offsets(pts)
        title.set_text(f"Epoch {int(epochs[k])}")
        artists = [title]
        artists.extend([x[0] for x in pred_scats])
        return artists

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(epochs),
        interval=max(1, int(1000 / max(1, fps))),
        blit=False,
        repeat=True,
    )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    if out_path.lower().endswith(".gif"):
        ani.save(out_path, writer=animation.PillowWriter(fps=fps))
    else:
        ani.save(out_path, writer="ffmpeg", fps=fps)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Render a constellation GIF from training snapshot .npz files.")
    ap.add_argument(
        "--snapshot_dir",
        type=str,
        default="./model/simple/training_snapshots",
        help="Directory containing snapshot_epoch_*.npz files.",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="./results/training_snapshots.gif",
        help="Output GIF path.",
    )
    ap.add_argument(
        "--mods",
        nargs="+",
        default=["BPSK", "QPSK", "8PSK", "16QAM"],
        help="Modulations to include if present in the snapshots.",
    )
    ap.add_argument(
        "--max_symbols",
        type=int,
        default=256,
        help="Maximum number of symbol-rate points to show per panel.",
    )
    ap.add_argument(
        "--fps",
        type=int,
        default=2,
        help="GIF frames per second.",
    )
    args = ap.parse_args()

    render_gif(
        snapshot_dir=args.snapshot_dir,
        out_path=args.out,
        mods=args.mods,
        max_symbols=args.max_symbols,
        fps=args.fps,
    )
    print(f"Saved GIF: {args.out}")


if __name__ == "__main__":
    main()
