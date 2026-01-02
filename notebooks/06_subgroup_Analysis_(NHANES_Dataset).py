# ======================================
# SUBGROUP ANALYSIS (NHANES)
# Age & Gender | XGBoost | Balanced Data
# ======================================

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# -----------------------
# 1. Separate X and y
# -----------------------
X = df_balanced.drop("diabetes", axis=1)
y = df_balanced["diabetes"]

# Keep a copy BEFORE scaling (for subgroup labels)
X_unscaled = X.copy()

# -----------------------
# 2. Scale features
# -----------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------
# 3. Train-test split
# -----------------------
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_scaled, y, X_unscaled.index,
    test_size=0.25,
    stratify=y,
    random_state=42
)

# -----------------------
# 4. Train XGBoost
# -----------------------
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

# -----------------------
# 5. Rebuild evaluation dataframe
# -----------------------
eval_df = X_unscaled.loc[idx_test].copy()
eval_df["diabetes"] = y_test.values
eval_df["y_prob"] = y_prob
eval_df["y_pred"] = y_pred

# -----------------------
# 6. Decode gender
# -----------------------
# gender_2 = Male (because drop_first=True)
eval_df["Gender"] = np.where(
    eval_df["gender_2.0"] == 1, "Male", "Female"
)

# -----------------------
# 7. Create age groups
# -----------------------
def age_group(age):
    if age <= 35:
        return "18–35"
    elif age <= 55:
        return "36–55"
    else:
        return "56+"

eval_df["AgeGroup"] = eval_df["age"].apply(age_group)

# -----------------------
# 8. Compute subgroup metrics
# -----------------------
results = []

def subgroup_metrics(df, label):
    if df["diabetes"].nunique() < 2:
        return
    recall = recall_score(df["diabetes"], df["y_pred"])
    auc = roc_auc_score(df["diabetes"], df["y_prob"])
    results.append([label, round(recall * 100, 2), round(auc, 3)])

# Gender-wise
for g in ["Male", "Female"]:
    subgroup_metrics(eval_df[eval_df["Gender"] == g], f"Gender: {g}")

# Age-wise
for ag in ["18–35", "36–55", "56+"]:
    subgroup_metrics(eval_df[eval_df["AgeGroup"] == ag], f"Age: {ag}")

# -----------------------
# 9. Results table
# -----------------------
subgroup_results = pd.DataFrame(
    results,
    columns=["Subgroup", "Recall (%)", "ROC-AUC"]
)

print("\nSubgroup Analysis Results (NHANES – Balanced Dataset):\n")
print(subgroup_results)
