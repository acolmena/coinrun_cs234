import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "results/grok_compare_agg.csv"
OUT_DIR = "results"
TAGS = ["train_rew_mean", "test_rew_mean", "train_minus_test_gap", "weight_norm"]

os.makedirs(OUT_DIR, exist_ok=True)
df = pd.read_csv(CSV_PATH)

for tag in TAGS:
    sub = df[df["tag"] == tag]
    if sub.empty:
        print("skip {} (no data)".format(tag))
        continue

    plt.figure(figsize=(7, 4))
    for run_id, g in sub.groupby("run_id"):
        g = g.sort_values("step")
        plt.plot(g["step"], g["mean"], label=run_id)

    plt.title(tag)
    plt.xlabel("step")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, "{}.png".format(tag))
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("wrote", out_path)