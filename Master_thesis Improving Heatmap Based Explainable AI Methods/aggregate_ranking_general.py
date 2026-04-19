import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--eval_path", required=True, help="Pfad zur per_image_method_metrics.csv")
parser.add_argument("--out_dir", required=True, help="Ausgabeordner für Ranking (wird angelegt)")
args = parser.parse_args()

EVAL_PATH = args.eval_path
OUT_DIR = args.out_dir
os.makedirs(OUT_DIR, exist_ok=True)

metrics = ["deletion_auc", "insertion_auc", "fraction_below_0.5"]
WEIGHTS = {"deletion_auc": 0.33, "insertion_auc": 0.33, "fraction_below_0.5": 0.33}

df = pd.read_csv(EVAL_PATH)
methods = df["method"].unique()
print("Gefundene Methoden:", methods)

mean_df = df.groupby("method")[metrics].mean()

norm_df = mean_df.copy()
for metric in metrics:
    if mean_df[metric].max() - mean_df[metric].min() == 0:
        norm_df[metric] = 0.0
    else:
        if metric == "deletion_auc":
            norm_df[metric] = 1 - (mean_df[metric] - mean_df[metric].min()) / (mean_df[metric].max() - mean_df[metric].min())
        else:
            norm_df[metric] = (mean_df[metric] - mean_df[metric].min()) / (mean_df[metric].max() - mean_df[metric].min())

norm_df["overall_score"] = sum(norm_df[m] * WEIGHTS.get(m, 0.0) for m in metrics)
ranked = norm_df.sort_values("overall_score", ascending=False)

rank_path = os.path.join(OUT_DIR, "combined_ranking.csv")
ranked.to_csv(rank_path)
print("\n Kombinierte Rangliste gespeichert unter:", rank_path)

best_method = ranked.index[0] if not ranked.empty else None
print("\n  Beste Methode insgesamt:", best_method)
print(ranked)

plt.figure(figsize=(8,5))
if not ranked.empty:
    ranked["overall_score"].plot(kind="bar", edgecolor="black")
    plt.title("Gesamtbewertung der Methoden (kombinierte Metriken)")
    plt.ylabel("Normalisierter Gesamtscore (0–1)")
    plt.xlabel("Methode")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "combined_ranking_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(" Diagramm gespeichert:", plot_path)
else:
    print("Keine Daten zum Plotten.")
