import matplotlib.pyplot as plt
import pandas as pd

results = {
    "Imbalanced": {
        "SVM": {"Acc": 91.55, "Recall": 1.55},
        "RF":  {"Acc": 91.58, "Recall": 4.89},
        "XGB": {"Acc": 91.51, "Recall": 5.08},
    },
    "Balanced": {
        "SVM": {"Acc": 75.79, "Recall": 77.27},
        "RF":  {"Acc": 75.25, "Recall": 78.35},
        "XGB": {"Acc": 75.93, "Recall": 78.87},
    },
    "Biased": {
        "SVM": {"Acc": 80.16, "Recall": 94.31},
        "RF":  {"Acc": 80.32, "Recall": 93.13},
        "XGB": {"Acc": 80.13, "Recall": 92.09},
    }
}

models = ["SVM", "RF", "XGB"]
datasets = ["Imbalanced", "Balanced", "Biased"]

# =========================
# FIGURE 1: RECALL COMPARISON
# =========================
recall_df = pd.DataFrame({
    d: [results[d][m]["Recall"] for m in models]
    for d in datasets
}, index=models)

recall_df.plot(
    kind="bar",
    figsize=(7,5),
    edgecolor="black"
)

plt.title("Recall Comparison Across Dataset Distributions")
plt.ylabel("Recall (%)")
plt.xlabel("Model")
plt.xticks(rotation=0)
plt.legend(title="Dataset")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# =========================
# FIGURE 2: ACCURACY–RECALL TRADE-OFF 
# =========================
colors = {"SVM": "tab:blue", "RF": "tab:orange", "XGB": "tab:green"}
markers = {"Imbalanced": "o", "Balanced": "s", "Biased": "^"}

plt.figure(figsize=(7,6))

for dataset in datasets:
    for model in models:
        acc = results[dataset][model]["Acc"]
        rec = results[dataset][model]["Recall"]
        plt.scatter(
            acc, rec,
            color=colors[model],
            marker=markers[dataset],
            s=120,
            edgecolor="black"
        )

# Legends
for model, color in colors.items():
    plt.scatter([], [], color=color, label=model)

for dataset, marker in markers.items():
    plt.scatter([], [], color="black", marker=marker, label=dataset)

plt.xlabel("Accuracy (%)")
plt.ylabel("Recall (%)")
plt.title("Accuracy–Recall Trade-off Across Dataset Distributions")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="lower left", fontsize=9)
plt.tight_layout()
plt.show()

