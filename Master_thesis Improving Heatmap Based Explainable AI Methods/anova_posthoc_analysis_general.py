import os
import argparse
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("--eval_path", required=True, help="Pfad zur per_image_method_metrics.csv")
parser.add_argument("--out_dir", required=True, help="Ausgabeordner für Statistiken (wird angelegt)")
args = parser.parse_args()

EVAL_PATH = args.eval_path
OUT_DIR = args.out_dir
os.makedirs(OUT_DIR, exist_ok=True)

metrics = ["deletion_auc", "insertion_auc", "fraction_below_0.5"]

df = pd.read_csv(EVAL_PATH)
print("Daten geladen mit Spalten:", df.columns.tolist())

methods_order = sorted(df["method"].unique())

for metric in metrics:
    print(f"\nAnalysiere Metrik: {metric}")

    data = df[["method", metric]].dropna().copy()
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    data = data.dropna(subset=[metric])
    if data.empty:
        print(f"Keine Daten für {metric}, überspringe.")
        continue

    groups = []
    group_names = []
    for m in methods_order:
        vals = data.loc[data["method"] == m, metric].values
        if vals.size > 0:
            groups.append(vals)
            group_names.append(m)

    if len(groups) < 2:
        print(f"Weniger als 2 Gruppen mit Daten für {metric}, überspringe ANOVA.")
        f_stat, p_val = np.nan, np.nan
    else:
        try:
            f_stat, p_val = stats.f_oneway(*groups)
        except Exception as e:
            print(f"Fehler bei scipy.stats.f_oneway für {metric}: {e}")
            f_stat, p_val = np.nan, np.nan

    anova_df = pd.DataFrame([{"metric": metric, "F": f_stat, "p_value": p_val}])
    anova_path = os.path.join(OUT_DIR, f"anova_{metric}.csv")
    anova_df.to_csv(anova_path, index=False)
    print("ANOVA gespeichert:", anova_path)
    print(anova_df)

    try:
        if data["method"].nunique() > 1 and data.shape[0] >= 3:
            tukey = pairwise_tukeyhsd(endog=data[metric], groups=data["method"], alpha=0.05)
            tukey_table = pd.DataFrame(tukey._results_table.data[1:], columns=tukey._results_table.data[0])
            tukey_path = os.path.join(OUT_DIR, f"tukey_{metric}.csv")
            tukey_table.to_csv(tukey_path, index=False)
            print("Tukey gespeichert:", tukey_path)
        else:
            print("Nicht genügend Gruppen/Daten für Tukey-Test, überspringe.")
    except Exception as e:
        print(f"Fehler beim Tukey-Test für {metric}: {e}")

    try:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x="method", y=metric, data=data, order=methods_order)
        sns.stripplot(x="method", y=metric, data=data, color="black", size=3, jitter=0.15, alpha=0.6, order=methods_order)
        plt.title(f"{metric} – ANOVA + Tukey-Test")
        plt.tight_layout()
        plot_path = os.path.join(OUT_DIR, f"anova_boxplot_{metric}.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print("Boxplot gespeichert:", plot_path)
    except Exception as e:
        print(f"Fehler beim Erstellen des Boxplots für {metric}: {e}")

print("\nANOVA + Post-hoc Analyse abgeschlossen.")
