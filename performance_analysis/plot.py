import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "npr_mean_llr_2curves.csv"
df = pd.read_csv(CSV_PATH)

cols = ["amp", "npr1_db", "metric1", "npr2_db", "metric2"]
for c in cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

def robust_group(df, snr_col, met_col):
    d = df.dropna(subset=["amp", snr_col, met_col]).copy()

    # trim metric outliers within each amp using IQR
    def iqr_trim(g):
        q1 = g[met_col].quantile(0.25)
        q3 = g[met_col].quantile(0.75)
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        return g[(g[met_col] >= lo) & (g[met_col] <= hi)]

    d = d.groupby("amp", group_keys=False).apply(iqr_trim)

    # use MEDIAN after trimming
    out = (
        d.groupby("amp", as_index=False)
         .agg(snr=(snr_col, "median"),
              metric=(met_col, "median"),
              n=(met_col, "count"),
              metric_std=(met_col, "std"))
         .sort_values("snr")
    )
    return out

gt = robust_group(df, "npr1_db", "metric1")
pr = robust_group(df, "npr2_db", "metric2")

W = 12 # window size in number of amp points
gt["metric_smooth"] = gt["metric"].rolling(W, center=True, min_periods=1).mean()
pr["metric_smooth"] = pr["metric"].rolling(W, center=True, min_periods=1).mean()

plt.figure(figsize=(9,5))
plt.plot(gt["snr"], gt["metric_smooth"], "-o", label="GT (smoothed)")
plt.plot(pr["snr"], pr["metric_smooth"], "-o", label="Pred (smoothed)")
plt.xlabel("NPR (dB)")
plt.ylabel("Mean LLR (smoothed)")
plt.title("NPR vs Mean LLR (Smoothed across amp)")
plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()
