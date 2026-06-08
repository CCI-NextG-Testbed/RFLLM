from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


RESULTS_DIR = Path("results/Semantic_Sim/16QAM")
OUTPUT_PATH = Path("results/BER_Prompt_Impact_Constellations.pdf")
PANELS = [
    ("pred_test_0000.mat", "Non-Signal Prompt"),
    ("pred_test_0002.mat", "Abstract Signal Prompt"),
    ("pred_test_0003.mat", "Generic QAM Prompt"),
    ("pred_test_0004.mat", "Full 16QAM Prompt"),
]


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_iq(path: Path) -> np.ndarray:
    data = sio.loadmat(path)
    iq = np.asarray(data["iq"]).reshape(-1)
    return iq.astype(np.complex64)


def main() -> None:
    configure_matplotlib()
    iq_by_label = [(label, load_iq(RESULTS_DIR / filename)) for filename, label in PANELS]

    all_points = np.concatenate([iq for _, iq in iq_by_label])
    limit = float(np.percentile(np.abs(np.concatenate([all_points.real, all_points.imag])), 99.5))
    limit = max(limit * 1.08, 1e-3)

    fig, axes = plt.subplots(2, 2, figsize=(7.8, 6.9), constrained_layout=True)
    for ax, (label, iq) in zip(axes.ravel(), iq_by_label):
        ax.scatter(iq.real, iq.imag, s=8, alpha=0.58, linewidths=0, color="#d62728")
        ax.set_title(label, pad=6)
        ax.set_xlabel("In-phase (I)")
        ax.set_ylabel("Quadrature (Q)")
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT_PATH.with_suffix(".png"), dpi=600, bbox_inches="tight")
    print(f"Saved constellation grid to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
