from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results")
OUTPUT_PATH = RESULTS_DIR / "train_loss_over_epochs.pdf"


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "lines.linewidth": 1.8,
            "lines.markersize": 7.0,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_publication_figure(fig, output_path: Path) -> None:
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    if output_path.suffix.lower() != ".pdf":
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")


def load_runs():
    runs = []
    csv_paths = sorted(
        RESULTS_DIR.glob("samples_*/convergence.csv"),
        key=lambda path: int(path.parent.name.split("_", 1)[1]),
    )
    for csv_path in csv_paths:
        df = (
            pd.read_csv(csv_path)
            .dropna(subset=["test_loss"])
            .sort_values("epoch")
            .reset_index(drop=True)
        )
        df["folder"] = csv_path.parent.name
        runs.append(df)
    return runs


def label_for_run(df: pd.DataFrame) -> str:
    folder = df["folder"].iloc[0]
    sample_count = folder.split("_", 1)[1]
    return f"{sample_count} waveforms per modulation"


def main():
    configure_matplotlib()
    runs = load_runs()
    if not runs:
        raise SystemExit("No convergence CSV files found under results/samples_*/.")

    fig, ax_curve = plt.subplots(1, 1, figsize=(5.3, 3.8), constrained_layout=True)

    for df in runs:
        ax_curve.plot(
            df["epoch"],
            df["test_loss"],
            marker="o",
            linewidth=2.2,
            markersize=7.0,
            label=label_for_run(df),
        )

    ax_curve.set_xlabel("Epoch")
    ax_curve.set_ylabel("Test Loss")
    ax_curve.grid(True, alpha=0.3)
    ax_curve.legend(loc="upper right", frameon=True, borderpad=0.45, handlelength=1.8)
    save_publication_figure(fig, OUTPUT_PATH)
    print(f"Saved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
