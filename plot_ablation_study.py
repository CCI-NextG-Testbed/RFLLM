from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RUNS = [
    "New Model=results/New_Model/convergence_new.csv",
    "No RAG=results/No_RAG/convergence.csv",
    "RF Diffusion Loss=results/RF_Diffusion_Loss/convergence.csv",
    "No Cosine Attention=results/No_Cosine_Attention/convergence.csv",
]
DEFAULT_OUTPUT = Path("results/ablation_study_metrics.png")
PREFERRED_MODULATION_ORDER = ["BPSK", "QPSK", "8PSK", "16QAM"]
MODULATION_COLORS = {
    "BPSK": "#1f77b4",
    "QPSK": "#ff7f0e",
    "8PSK": "#2ca02c",
    "16QAM": "#d62728",
}


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.parent.name, path

    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError(f"Run label is empty in {value!r}.")
    return label, Path(path.strip())


def summarize_run(
    label: str,
    csv_path: Path,
    row_policy: str,
    range_window: int,
) -> dict[str, float | str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing convergence CSV: {csv_path}")

    df = pd.read_csv(csv_path).dropna(subset=["train_loss", "test_loss"])
    if df.empty:
        raise ValueError(f"No valid train/test loss rows found in {csv_path}")

    df = df.sort_values("epoch")
    if row_policy == "best-test-loss":
        selected_rows = df.loc[[df["test_loss"].idxmin()]]
    elif row_policy == "last":
        selected_rows = df.tail(1)
    else:
        selected_rows = df.tail(range_window)

    evm_cols = [col for col in df.columns if col.startswith("test_evm_")]
    if not evm_cols:
        raise ValueError(f"No test_evm_* columns found in {csv_path}")

    summary = {
        "label": label,
    }
    for col in sorted(evm_cols):
        modulation = col.replace("test_evm_", "")
        values = selected_rows[col].dropna()
        mean_value = float(values.mean())
        min_value = float(values.min())
        max_value = float(values.max())
        summary[f"evm_{modulation}"] = mean_value
        summary[f"evm_{modulation}_min"] = min_value
        summary[f"evm_{modulation}_max"] = max_value
        summary[f"evm_{modulation}_err_low"] = mean_value - min_value
        summary[f"evm_{modulation}_err_high"] = max_value - mean_value
    return summary


def add_value_labels(
    ax,
    bars,
    values: list[float],
    upper_errors: list[float] | None = None,
    rotation: int = 0,
    y_offset: int = 5,
) -> None:
    upper_errors = upper_errors or [0.0] * len(values)
    for bar, value, upper_error in zip(bars, values, upper_errors):
        height = bar.get_height()
        ax.annotate(
            f"{value:.3g}",
            xy=(bar.get_x() + bar.get_width() / 2, height + upper_error),
            xytext=(0, y_offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=rotation,
        )


def sorted_evm_cols(metrics: pd.DataFrame) -> list[str]:
    evm_cols = [
        col
        for col in metrics.columns
        if col.startswith("evm_")
        and not col.endswith(("_min", "_max", "_err_low", "_err_high"))
    ]
    order = {f"evm_{modulation}": idx for idx, modulation in enumerate(PREFERRED_MODULATION_ORDER)}
    return sorted(evm_cols, key=lambda col: (order.get(col, len(order)), col))


def evm_range_errors(metrics: pd.DataFrame, column: str) -> list[list[float]]:
    return [
        metrics[f"{column}_err_low"].tolist(),
        metrics[f"{column}_err_high"].tolist(),
    ]


def modulation_color(column: str, fallback_color: str) -> str:
    modulation = column.replace("evm_", "")
    return MODULATION_COLORS.get(modulation, fallback_color)


def plot_grouped(metrics: pd.DataFrame, output_path: Path) -> None:
    labels = metrics["label"].tolist()
    x = np.arange(len(labels))
    evm_cols = sorted_evm_cols(metrics)
    group_width = 0.85
    width = group_width / max(len(evm_cols), 1)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    series = [
        (
            col,
            col.replace("evm_", "EVM "),
            modulation_color(col, color_cycle[idx % len(color_cycle)]),
        )
        for idx, col in enumerate(evm_cols)
    ]

    fig, ax = plt.subplots(figsize=(13, 6.5))
    offsets = np.linspace(
        -group_width / 2 + width / 2,
        group_width / 2 - width / 2,
        len(series),
    )

    for idx, (offset, (column, display, color)) in enumerate(zip(offsets, series)):
        values = metrics[column].tolist()
        upper_errors = metrics[f"{column}_err_high"].tolist()
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=display,
            color=color,
            yerr=evm_range_errors(metrics, column),
            capsize=4,
            error_kw={"ecolor": "black", "elinewidth": 1.1, "capthick": 1.1},
        )
        add_value_labels(ax, bars, values, upper_errors, y_offset=4 + (idx % 2) * 9)

    ax.set_xlabel("Ablation Setting")
    ax.set_ylabel("Metric Value")
    ax.set_title("Ablation Study Metrics (Mean with Min/Max Range)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.margins(x=0.08, y=0.18)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(
        ncols=1,
        loc="upper right",
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        edgecolor="0.85",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")


def plot_panels(metrics: pd.DataFrame, output_path: Path) -> None:
    labels = metrics["label"].tolist()
    x = np.arange(len(labels))
    evm_cols = sorted_evm_cols(metrics)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    series = [
        (
            col,
            col.replace("evm_", "EVM "),
            modulation_color(col, color_cycle[idx % len(color_cycle)]),
        )
        for idx, col in enumerate(evm_cols)
    ]

    fig, axes = plt.subplots(len(series), 1, figsize=(11, 9), sharex=True)
    for ax, (column, display, color) in zip(axes, series):
        values = metrics[column].tolist()
        upper_errors = metrics[f"{column}_err_high"].tolist()
        bars = ax.bar(
            x,
            values,
            color=color,
            width=0.55,
            yerr=evm_range_errors(metrics, column),
            capsize=4,
            error_kw={"ecolor": "black", "elinewidth": 1.1, "capthick": 1.1},
        )
        add_value_labels(ax, bars, values, upper_errors)
        ax.set_ylabel(display)
        ax.grid(axis="y", alpha=0.3)

    axes[-1].set_xlabel("Ablation Setting")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=20, ha="right")
    fig.suptitle("Ablation Study Metrics (Mean with Min/Max Range)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")


def plot_quadrants(metrics: pd.DataFrame, output_path: Path) -> None:
    labels = metrics["label"].tolist()
    x = np.arange(len(labels))
    evm_cols = sorted_evm_cols(metrics)

    if len(evm_cols) != 4:
        raise ValueError(
            f"Quadrant layout expects exactly 4 EVM metrics, found {len(evm_cols)}: "
            f"{', '.join(col.replace('evm_', '') for col in evm_cols)}"
        )

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharey=False)
    for idx, (ax, column) in enumerate(zip(axes.ravel(), evm_cols)):
        modulation = column.replace("evm_", "")
        values = metrics[column].tolist()
        upper_errors = metrics[f"{column}_err_high"].tolist()
        color = modulation_color(column, color_cycle[idx % len(color_cycle)])
        bars = ax.bar(
            x,
            values,
            color=color,
            width=0.62,
            yerr=evm_range_errors(metrics, column),
            capsize=4,
            error_kw={"ecolor": "black", "elinewidth": 1.1, "capthick": 1.1},
        )
        add_value_labels(ax, bars, values, upper_errors)
        ax.set_title(f"{modulation} EVM", fontsize=12)
        ax.set_ylabel("EVM")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.margins(y=0.25)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Ablation Study EVM by Modulation (Mean with Min/Max Range)", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot ablation bars for per-modulation EVM metrics."
    )
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run,
        default=None,
        help="Ablation run as 'X Label=path/to/convergence.csv'. Can be repeated.",
    )
    parser.add_argument(
        "--row",
        choices=["average-last-window", "last", "best-test-loss"],
        default="average-last-window",
        help="Which EVM values to plot. The default plots the average over the final range window.",
    )
    parser.add_argument(
        "--range-window",
        type=int,
        default=5,
        help="Number of final epochs used for the mean and min/max range.",
    )
    parser.add_argument(
        "--layout",
        choices=["quadrants", "panels", "grouped"],
        default="quadrants",
        help="Use 2x2 modulation quadrants, stacked panels, or grouped bars.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output image path.",
    )
    args = parser.parse_args()

    if args.range_window < 1:
        raise ValueError("--range-window must be at least 1.")

    runs = args.run or [parse_run(run) for run in DEFAULT_RUNS]
    summaries = [
        summarize_run(label, path, args.row, args.range_window)
        for label, path in runs
    ]
    metrics = pd.DataFrame(summaries)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.layout == "quadrants":
        plot_quadrants(metrics, args.output)
    elif args.layout == "grouped":
        plot_grouped(metrics, args.output)
    else:
        plot_panels(metrics, args.output)

    print(metrics.to_string(index=False))
    print(f"Saved ablation plot to {args.output}")


if __name__ == "__main__":
    main()
