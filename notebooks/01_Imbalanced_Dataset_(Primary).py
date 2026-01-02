# =========================
# METRICS + VISUALIZATION
# =========================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# -------------------------
# LOAD & PREPROCESS
# -------------------------
df = pd.read_csv("/content/diabetes_prediction_dataset.csv").dropna()

X = df.drop(
    ["diabetes", "HbA1c_level", "blood_glucose_level"],
    axis=1
)
y = df["diabetes"].values

X = pd.get_dummies(
    X,
    columns=["gender", "smoking_history"],
    drop_first=True
)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# -------------------------
# TRAIN–TEST SPLIT
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# -------------------------
# MODELS
# -------------------------
models = {
    "SVM": SVC(kernel="rbf", C=10, gamma=0.05, probability=True),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        eval_metric="logloss", random_state=42
    )
}

results = {}

# -------------------------
# ML METRICS
# -------------------------
for name, model in models.items():
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "ROC-AUC": roc_auc_score(y_test, probs),
        "fpr_tpr": roc_curve(y_test, probs)
    }



# -------------------------
# PRINT METRICS
# -------------------------
print("===== METRIC COMPARISON =====")
for model, metrics in results.items():
    print(
        f"{model} | "
        f"Acc: {metrics['Accuracy']*100:.2f}% | "
        f"Prec: {metrics['Precision']*100:.2f}% | "
        f"Recall: {metrics['Recall']*100:.2f}% | "
        f"AUC: {metrics['ROC-AUC']:.3f}"
    )
