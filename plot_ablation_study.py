from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RUNS = [
    "WavePrompt=results/New_Model/convergence_new.csv",
    "w/o RAG=results/No_RAG/convergence.csv",
    "w/o Loss=results/RF_Diffusion_Loss/convergence.csv",
    "w/o Cosine Attention=results/No_Cosine_Attention/convergence.csv",
]
DEFAULT_OUTPUT = Path("results/ablation_study_metrics.png")
DEFAULT_LOG_YMIN = 1e-3
PREFERRED_MODULATION_ORDER = ["BPSK", "QPSK", "8PSK", "16QAM"]
MODULATION_COLORS = {
    "BPSK": "#1f77b4",
    "QPSK": "#ff7f0e",
    "8PSK": "#2ca02c",
    "16QAM": "#d62728",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 15,
            "axes.titlesize": 16,
            "axes.labelsize": 16,
            "legend.fontsize": 14,
            "xtick.labelsize": 13,
            "ytick.labelsize": 14,
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


def apply_y_scale(
    ax,
    metrics: pd.DataFrame,
    evm_cols: list[str],
    y_scale: str,
    log_ymin: float,
) -> None:
    if y_scale != "log":
        ax.margins(y=0.18)
        ax.grid(axis="y", alpha=0.3)
        return

    positive_lows = []
    positive_highs = []
    for column in evm_cols:
        positive_lows.extend(metrics[f"{column}_min"][metrics[f"{column}_min"] > 0].tolist())
        positive_highs.extend(metrics[f"{column}_max"][metrics[f"{column}_max"] > 0].tolist())

    ax.set_yscale("log", base=10)
    if positive_lows and positive_highs:
        ax.set_ylim(min(log_ymin, min(positive_lows) / 1.8), max(positive_highs) * 1.8)
    ax.grid(axis="y", which="both", alpha=0.3)


def plot_grouped(
    metrics: pd.DataFrame,
    output_path: Path,
    y_scale: str,
    log_ymin: float,
) -> None:
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

    fig, ax = plt.subplots(figsize=(6.9, 3.9), constrained_layout=True)
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

    ax.set_xlabel("Ablation Setting")
    ax.set_ylabel("EVM")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.margins(x=0.08)
    apply_y_scale(ax, metrics, evm_cols, y_scale, log_ymin)
    ax.legend(
        ncols=1,
        loc="upper right",
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        edgecolor="0.85",
    )
    save_publication_figure(fig, output_path)


def plot_panels(
    metrics: pd.DataFrame,
    output_path: Path,
    y_scale: str,
    log_ymin: float,
) -> None:
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

    fig, axes = plt.subplots(
        len(series),
        1,
        figsize=(6.7, 5.4),
        sharex=True,
        constrained_layout=True,
    )
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
        ax.set_ylabel(display)
        apply_y_scale(ax, metrics, evm_cols, y_scale, log_ymin)

    axes[-1].set_xlabel("Ablation Setting")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=20, ha="right")
    save_publication_figure(fig, output_path)


def plot_quadrants(
    metrics: pd.DataFrame,
    output_path: Path,
    y_scale: str,
    log_ymin: float,
) -> None:
    labels = metrics["label"].tolist()
    x = np.arange(len(labels))
    evm_cols = sorted_evm_cols(metrics)

    if len(evm_cols) != 4:
        raise ValueError(
            f"Quadrant layout expects exactly 4 EVM metrics, found {len(evm_cols)}: "
            f"{', '.join(col.replace('evm_', '') for col in evm_cols)}"
        )

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8), sharey=False, constrained_layout=True)
    for idx, (ax, column) in enumerate(zip(axes.ravel(), evm_cols)):
        modulation = column.replace("evm_", "")
        values = metrics[column].tolist()
        upper_errors = metrics[f"{column}_err_high"].tolist()
        color = modulation_color(column, color_cycle[idx % len(color_cycle)])
        ax.bar(
            x,
            values,
            color=color,
            width=0.68,
            yerr=evm_range_errors(metrics, column),
            capsize=4,
            error_kw={"ecolor": "black", "elinewidth": 1.1, "capthick": 1.1},
        )
        ax.set_ylabel(f"{modulation} EVM")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        apply_y_scale(ax, metrics, evm_cols, y_scale, log_ymin)

    save_publication_figure(fig, output_path)


def main() -> None:
    configure_matplotlib()
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
        "--y-scale",
        choices=["log", "linear"],
        default="log",
        help="Y-axis scale for EVM bars. Use log to compare values on the same visual scale.",
    )
    parser.add_argument(
        "--log-ymin",
        type=float,
        default=DEFAULT_LOG_YMIN,
        help="Minimum y-axis value for log-scale plots.",
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
    if args.log_ymin <= 0:
        raise ValueError("--log-ymin must be greater than 0.")

    runs = args.run or [parse_run(run) for run in DEFAULT_RUNS]
    summaries = [
        summarize_run(label, path, args.row, args.range_window)
        for label, path in runs
    ]
    metrics = pd.DataFrame(summaries)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.layout == "quadrants":
        plot_quadrants(metrics, args.output, args.y_scale, args.log_ymin)
    elif args.layout == "grouped":
        plot_grouped(metrics, args.output, args.y_scale, args.log_ymin)
    else:
        plot_panels(metrics, args.output, args.y_scale, args.log_ymin)

    print(metrics.to_string(index=False))
    print(f"Saved ablation plot to {args.output}")


if __name__ == "__main__":
    main()
