#!/usr/bin/env python3
import sys, glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def find_count_col(df):
    for c in df.columns:
        if c.lower() in ("count","frequency","freq"):
            return c
    for c in df.columns:
        if c.lower()!="intensity" and pd.api.types.is_numeric_dtype(df[c]):
            return c
    raise ValueError

def main(out_dir, max_plots=6):
    p=Path(out_dir)
    files=sorted(glob.glob(str(p/"histogram_*.csv")))
    if not files:
        print("No files"); return
    fig,axes=plt.subplots(2,3,figsize=(15,10))
    axes=axes.flatten(); shown=0; agg=np.zeros(256,dtype=np.int64)
    for fp in files:
        df=pd.read_csv(fp)
        cnt=df[find_count_col(df)].astype(int)
        if len(cnt)!=256: continue
        agg+=cnt.values
        if shown<max_plots:
            ax=axes[shown]
            ax.bar(df["Intensity"],cnt,width=1); ax.set_xlim(0,255)
            shown+=1
    for i in range(shown,6): axes[i].axis("off")
    fig.tight_layout(); fig.savefig(p/"individual_histograms.png",dpi=300)
    plt.show()
    plt.figure(figsize=(12,6))
    plt.bar(range(256),agg,width=1); plt.xlim(0,255)
    plt.tight_layout(); plt.savefig(p/"aggregate_histogram.png",dpi=300)
    plt.show()

if __name__=="__main__":
    if len(sys.argv)!=2:
        print("Usage: simple_plot_histograms.py <dir>"); sys.exit(1)
    main(sys.argv[1])
