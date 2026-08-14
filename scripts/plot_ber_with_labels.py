#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import re
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


STYLE_BY_FAMILY = {
    "bpsk": {"color": "#1f77b4", "marker": "o"},
    "qpsk": {"color": "#ff7f0e", "marker": "*"},
    "8psk": {"color": "#2ca02c", "marker": "D"},
    "16psk": {"gt_color": "#7DD3FC", "pred_color": "#0284C7", "marker": "s"},
    "32psk": {"gt_color": "#F0ABFC", "pred_color": "#C026D3", "marker": "s"},
    "4qam": {"gt_color": "#93C5FD", "pred_color": "#2563EB", "marker": "x"},
    "16qam": {"color": "#d62728", "marker": "d"},
    "64qam": {"gt_color": "#BBF7D0", "pred_color": "#22C55E", "marker": "*"},
}


def load_one_csv(path: str, snr_col: str, ber_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[snr_col] = pd.to_numeric(df[snr_col], errors="coerce")
    df[ber_col] = pd.to_numeric(df[ber_col], errors="coerce")
    df = df.dropna(subset=[snr_col, ber_col]).copy()
    df = df.rename(columns={snr_col: "snr_db", ber_col: "ber"})
    return df


def aggregate_df(df: pd.DataFrame, how: str) -> pd.DataFrame:
    if how == "none":
        return df.sort_values("snr_db")
    if how == "mean":
        return df.groupby("snr_db", as_index=False)["ber"].mean()
    if how == "median":
        return df.groupby("snr_db", as_index=False)["ber"].median()
    raise ValueError("Unknown aggregate mode")


def infer_line_style(label: str, path: str) -> str:
    role = infer_source_role(label, path)
    if role == "pred":
        return "None"
    return "-"


def infer_marker(label: str, path: str) -> Optional[str]:
    role = infer_source_role(label, path)
    if role == "pred":
        return "x"
    if role == "gt":
        return "s"

    family = infer_modulation_family(label, path)
    return STYLE_BY_FAMILY.get(family, {}).get("marker", "o")


def infer_marker_size(label: str, path: str) -> float:
    role = infer_source_role(label, path)
    if role == "gt":
        return 6.0
    if role == "pred":
        return 4.5
    return 5.0


def infer_source_role(label: str, path: str) -> str:
    series_name = f"{label} {Path(path).stem} {Path(path).parent}".lower()
    if any(token in series_name for token in ("pred", "prediction", "predicted")):
        return "pred"
    if any(token in series_name for token in ("gt", "ground_truth", "ground-truth", "ground truth")):
        return "gt"
    return ""


def infer_color(label: str, path: str, fallback_color: str) -> str:
    role = infer_source_role(label, path)
    family = infer_modulation_family(label, path)
    style = STYLE_BY_FAMILY.get(family, {})
    if "color" in style:
        return style["color"]
    if role == "pred":
        return style.get("pred_color", fallback_color)
    if role == "gt":
        return style.get("gt_color", fallback_color)
    return style.get("pred_color", fallback_color)


def infer_modulation_family(label: str, path: str) -> str:
    series_name = f"{label} {Path(path).stem}".lower()
    compact_name = re.sub(r"[^a-z0-9]+", "", series_name)
    match = re.search(r"(\d+qam|\d+psk|bpsk|qpsk|\d+pam|\d+fsk|ook)", compact_name)
    if match:
        return match.group(1)

    cleanup_tokens = {
        "ber",
        "gt",
        "ground",
        "truth",
        "pred",
        "prediction",
        "predicted",
        "snr",
        "vs",
    }
    tokens = [
        token
        for token in re.split(r"[^a-z0-9]+", series_name)
        if token and token not in cleanup_tokens
    ]
    return tokens[0] if tokens else compact_name


def main():
    ap = argparse.ArgumentParser(
        description="Plot BER vs SNR for multiple CSV files on the same semilog-y graph."
    )

    # multiple inputs
    ap.add_argument(
        "--csv",
        nargs="+",
        required=True,
        help="One or more CSV paths. Example: --csv metrics_a.csv metrics_b.csv",
    )
    ap.add_argument(
        "--labels",
        nargs="*",
        default=[],
        help="Optional labels (same count as --csv). If omitted, filenames are used.",
    )

    # column selection (in case different files differ)
    ap.add_argument("--snr_col", default="ebn0_db", help="SNR column name (default snr_db)")
    ap.add_argument("--ber_col", default="ber", help="BER column name (default ber)")

    # axis + behavior
    ap.add_argument("--out", default="", help="Optional output image path (e.g., ber_vs_snr.png)")
    ap.add_argument("--xmax", type=float, default=15.0, help="Max SNR on x-axis (default 10 dB)")
    ap.add_argument("--ymin", type=float, default=1e-5, help="Min BER on y-axis (default 1e-5)")
    ap.add_argument("--ymax", type=float, default=1.0, help="Max BER on y-axis (default 1)")
    ap.add_argument(
        "--clip_zero_to",
        type=float,
        default=1e-5,
        help="Replace BER<=0 with this value so it shows on log scale (default 1e-5)",
    )
    ap.add_argument(
        "--aggregate",
        choices=["none", "mean", "median"],
        default="mean",
        help="Aggregate multiple points per SNR (default mean). Use 'none' to plot all points.",
    )

    args = ap.parse_args()

    csv_paths = args.csv
    labels = args.labels

    if labels and (len(labels) != len(csv_paths)):
        raise SystemExit(
            f"--labels count ({len(labels)}) must match --csv count ({len(csv_paths)}) "
            f"or be omitted."
        )

    if not labels:
        labels = [Path(p).stem for p in csv_paths]

    plt.figure()

    xmins = []
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, (path, label) in enumerate(zip(csv_paths, labels)):
        df = load_one_csv(path, args.snr_col, args.ber_col)

        # limit x-range
        df = df[df["snr_db"] <= args.xmax].copy()

        # handle zeros/negatives for log scale
        df.loc[df["ber"] <= 0, "ber"] = args.clip_zero_to

        # aggregate
        df_plot = aggregate_df(df, args.aggregate).sort_values("snr_db")

        if len(df_plot) == 0:
            print(f"[WARN] No valid rows after filtering for: {path}")
            continue

        xmins.append(float(df_plot["snr_db"].min()))
        fallback_color = color_cycle[idx % len(color_cycle)]

        plt.semilogy(
            df_plot["snr_db"],
            df_plot["ber"],
            color=infer_color(label, path, fallback_color),
            marker=infer_marker(label, path),
            markersize=infer_marker_size(label, path),
            markerfacecolor="none",
            markeredgewidth=1.0,
            linewidth=1.4,
            linestyle=infer_line_style(label, path),
            label=label,
        )

    if not xmins:
        raise SystemExit("No data to plot (all files empty after parsing/filtering).")

    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.xlim(min(xmins), args.xmax)
    plt.ylim(args.ymin, args.ymax)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.title("BER vs SNR Comparison")
    plt.legend()

    if args.out:
        plt.savefig(args.out, dpi=200, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()
