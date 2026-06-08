from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results")
OUTPUT_PATH = RESULTS_DIR / "train_loss_over_epochs.png"


def load_runs():
    runs = []
    for csv_path in sorted(RESULTS_DIR.glob("samples_*/convergence.csv")):
        df = (
            pd.read_csv(csv_path)
            .dropna(subset=["train_loss"])
            .sort_values("epoch")
            .reset_index(drop=True)
        )
        df["folder"] = csv_path.parent.name
        runs.append(df)
    return runs


def label_for_run(df: pd.DataFrame) -> str:
    folder = df["folder"].iloc[0]
    train_samples = int(df["train_sample_count"].iloc[0])
    sample_count = folder.split("_", 1)[1]
    return f"{sample_count} samples ({train_samples} train)"


def main():
    runs = load_runs()
    if not runs:
        raise SystemExit("No convergence CSV files found under results/samples_*/.")

    fig, ax_curve = plt.subplots(1, 1, figsize=(10, 6))

    for df in runs:
        ax_curve.plot(
            df["epoch"],
            df["train_loss"],
            marker="o",
            linewidth=2,
            markersize=4,
            label=label_for_run(df),
        )

    ax_curve.set_xlabel("Epoch")
    ax_curve.set_ylabel("Train Loss")
    ax_curve.grid(True, alpha=0.3)
    ax_curve.legend()

    fig.suptitle(
        "Training Loss Over Epochs",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
