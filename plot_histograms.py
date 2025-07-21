#!/usr/bin/env python3
"""
Very small histogram-plot helper.
Usage:
    python3 simple_plot_histograms.py <output_dir>
It looks for files named  histogram_*.csv  inside <output_dir>.
"""

import sys, glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_count_col(df):
    """Return the column that stores bin counts."""
    for c in df.columns:
        if c.lower() in ("count", "frequency", "freq"):
            return c
    # fallback: first numeric, non-Intensity column
    for c in df.columns:
        if c.lower() != "intensity" and pd.api.types.is_numeric_dtype(df[c]):
            return c
    raise ValueError("No count column found")


def main(out_dir, max_plots=6):
    out_dir = Path(out_dir)
    csv_files = sorted(glob.glob(str(out_dir / "histogram_*.csv")))
    if not csv_files:
        print("No histogram_*.csv files found in", out_dir)
        return

    # ---------------- individual plots ----------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    shown = 0
    aggregate = np.zeros(256, dtype=np.int64)

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        cnt_col = find_count_col(df)

        counts = pd.to_numeric(df[cnt_col], errors="coerce").fillna(0).astype(np.int64)
        if len(counts) != 256:                      # skip malformed files
            print(f"Skipped {Path(csv_path).name} (rows={len(counts)})")
            continue

        # add to aggregate
        aggregate += counts.values

        # plot first few histograms
        if shown < max_plots:
            ax = axes[shown]
            ax.bar(df["Intensity"], counts, width=1.0)
            ax.set_title(Path(csv_path).stem, fontsize=8)
            ax.set_xlim(0, 255)
            shown += 1

    # hide unused sub-plots
    for i in range(shown, len(axes)):
        axes[i].axis("off")

    fig.tight_layout()
    fig.savefig(out_dir / "individual_histograms.png", dpi=300)
    print("Saved ➜ individual_histograms.png")

    # ---------------- aggregate plot ----------------
    plt.figure(figsize=(12, 6))
    plt.bar(range(256), aggregate, width=1.0, color="steelblue")
    plt.title(f"Aggregate histogram ({aggregate.sum():,} pixels)")
    plt.xlim(0, 255)
    plt.tight_layout()
    plt.savefig(out_dir / "aggregate_histogram.png", dpi=300)
    print("Saved ➜ aggregate_histogram.png")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 simple_plot_histograms.py <output_dir>")
        sys.exit(1)
    main(sys.argv[1])
