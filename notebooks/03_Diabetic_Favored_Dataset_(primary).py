# =========================
# DIABETIC-FAVORED DATASET
# (70% Diabetic, 30% Non-Diabetic)
# =========================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("/content/diabetes_prediction_dataset.csv").dropna()

# Remove dominant biomarkers
df = df.drop(["HbA1c_level", "blood_glucose_level"], axis=1)

# -------------------------
# CREATE DIABETIC-FAVORED DATASET
# -------------------------
df_diab = df[df["diabetes"] == 1]
df_nondiab = df[df["diabetes"] == 0]

# Target ratio: 70% diabetic, 30% non-diabetic
n_diab = len(df_diab)
n_nondiab = int(n_diab * 0.43)   # 70:30 ratio approx

df_nondiab = df_nondiab.sample(n=n_nondiab, random_state=42)

df_biased = pd.concat([df_diab, df_nondiab]).sample(
    frac=1, random_state=42
)

# -------------------------
# FEATURES & TARGET
# -------------------------
X = df_biased.drop("diabetes", axis=1)
y = df_biased["diabetes"].values

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
    X, y,
    test_size=0.25,
    stratify=y,
    random_state=42
)

# -------------------------
# ML MODELS
# -------------------------
models = {
    "SVM": SVC(kernel="rbf", C=10, gamma=0.05, probability=True),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=5,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=42
    )
}

results = {}

# -------------------------
# TRAIN & EVALUATE ML
# -------------------------
for name, model in models.items():
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "ROC-AUC": roc_auc_score(y_test, probs)
    }



# -------------------------
# PRINT RESULTS
# -------------------------
print("===== DIABETIC-FAVORED DATASET RESULTS =====")
for model, m in results.items():
    print(
        f"{model} | "
        f"Acc: {m['Accuracy']*100:.2f}% | "
        f"Prec: {m['Precision']*100:.2f}% | "
        f"Recall: {m['Recall']*100:.2f}% | "
        f"AUC: {m['ROC-AUC']:.3f}"
    )
