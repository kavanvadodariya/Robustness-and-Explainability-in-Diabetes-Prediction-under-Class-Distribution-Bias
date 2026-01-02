import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from matplotlib.lines import Line2D

def collect_results(df, dataset_name, distribution):
    X = df.drop("diabetes", axis=1)
    y = df["diabetes"]

    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    rows = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        rows.append({
            "Dataset": dataset_name,
            "Distribution": distribution,
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Recall": recall_score(y_test, preds),
            "AUC": roc_auc_score(y_test, probs)
        })

    return rows
import matplotlib.pyplot as plt

plt.figure(figsize=(7,5))

markers = {
    "Imbalanced": "o",
    "Balanced": "s",
    "Biased": "^"
}

colors = {
    "SVM": "#1f77b4",            # blue
    "Random Forest": "#ff7f0e",  # orange
    "XGBoost": "#2ca02c"         # green
}

for _, row in nhanes_results.iterrows():
    plt.scatter(
        row["Accuracy"] * 100,
        row["Recall"] * 100,
        marker=markers[row["Distribution"]],
        color=colors[row["Model"]],
        s=120,
        edgecolors="black",
        alpha=0.95
    )

plt.xlabel("Accuracy (%)")
plt.ylabel("Recall (%)")
plt.title("Accuracy–Recall Trade-off Across Dataset Distributions (NHANES)")
plt.grid(True, linestyle="--", alpha=0.6)

# Custom legend (to match your plot style)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Imbalanced',
           markerfacecolor='black', markersize=8),
    Line2D([0], [0], marker='s', color='w', label='Balanced',
           markerfacecolor='black', markersize=8),
    Line2D([0], [0], marker='^', color='w', label='Biased',
           markerfacecolor='black', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='SVM',
           markerfacecolor=colors["SVM"], markersize=8),
    Line2D([0], [0], marker='o', color='w', label='RF',
           markerfacecolor=colors["Random Forest"], markersize=8),
    Line2D([0], [0], marker='o', color='w', label='XGB',
           markerfacecolor=colors["XGBoost"], markersize=8)
]

plt.legend(handles=legend_elements, loc="lower left", fontsize=9)
plt.tight_layout()
plt.show()
