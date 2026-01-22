#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
    ap.add_argument("--snr_col", default="snr_db", help="SNR column name (default snr_db)")
    ap.add_argument("--ber_col", default="ber", help="BER column name (default ber)")

    # axis + behavior
    ap.add_argument("--out", default="", help="Optional output image path (e.g., ber_vs_snr.png)")
    ap.add_argument("--xmax", type=float, default=10.0, help="Max SNR on x-axis (default 10 dB)")
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

    for path, label in zip(csv_paths, labels):
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

        plt.semilogy(
            df_plot["snr_db"],
            df_plot["ber"],
            marker="o",
            linestyle="-",
            label=label,
        )

    if not xmins:
        raise SystemExit("No data to plot (all files empty after parsing/filtering).")

    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.xlim(min(xmins), args.xmax)
    plt.ylim(args.ymin, args.ymax)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.title("BER vs SNR (Multiple Runs)")
    plt.legend()

    if args.out:
        plt.savefig(args.out, dpi=200, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()
