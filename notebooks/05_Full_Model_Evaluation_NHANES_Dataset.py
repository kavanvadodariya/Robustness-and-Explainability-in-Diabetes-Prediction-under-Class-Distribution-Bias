import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score


#load datasets
demo = pd.read_sas("/content/DEMO_L.xpt")
bmx  = pd.read_sas("/content/BMX_L.xpt")
bpx  = pd.read_sas("/content/BPQ_L.xpt")
diq  = pd.read_sas("/content/DIQ_L.xpt")
smq  = pd.read_sas("/content/SMQ_L.xpt")

#merge datasets
df = demo.merge(bmx, on="SEQN", how="inner") \
         .merge(bpx, on="SEQN", how="inner") \
         .merge(diq, on="SEQN", how="inner") \
         .merge(smq, on="SEQN", how="inner")

#diabetes label
df = df[df["DIQ010"].isin([1, 2])]
df["diabetes"] = df["DIQ010"].map({1: 1, 2: 0})

#smoking history
def smoking_status(row):
    if row["SMQ020"] == 2:
        return "Never"
    elif row["SMQ040"] == 3:
        return "Former"
    elif row["SMQ040"] in [1, 2]:
        return "Current"
    else:
        return None

df["smoking_history"] = df.apply(smoking_status, axis=1)

#hypertension
df = df[df["BPQ020"].isin([1, 2])]
df["hypertension"] = df["BPQ020"].map({1: 1, 2: 0})

final_df = df[[
    "RIDAGEYR",
    "RIAGENDR",
    "BMXBMI",
    "hypertension",
    "smoking_history",
    "diabetes"
]].dropna()

final_df.rename(columns={
    "RIDAGEYR": "age",
    "RIAGENDR": "gender",
    "BMXBMI": "bmi"
}, inplace=True)


df = final_df.copy()

df = pd.get_dummies(
    df,
    columns=["gender", "smoking_history"],
    drop_first=True
)


df_imbalanced = df.copy()


df_diab = df[df["diabetes"] == 1]
df_nondiab = df[df["diabetes"] == 0]

df_balanced = pd.concat([
    df_diab,
    df_nondiab.sample(len(df_diab), random_state=42)
]).sample(frac=1, random_state=42)


df_biased = pd.concat([
    df_diab,
    df_nondiab.sample(int(len(df_diab) * 0.43), random_state=42)
]).sample(frac=1, random_state=42)


from sklearn.model_selection import train_test_split

def prepare_xy(df):
    X = df.drop("diabetes", axis=1)
    y = df["diabetes"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(
        X_scaled, y,
        test_size=0.25,
        stratify=y,
        random_state=42
    )

models = {
    "SVM": SVC(kernel="rbf", C=10, gamma=0.05, probability=True),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=42
    )
}

def evaluate_models(df, label):
    print(f"\n===== NHANES RESULTS ({label}) =====")
    X_train, X_test, y_train, y_test = prepare_xy(df)

    for name, model in models.items():
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        print(
            f"{name} | "
            f"Acc: {accuracy_score(y_test, preds)*100:.2f}% | "
            f"Recall: {recall_score(y_test, preds)*100:.2f}% | "
            f"AUC: {roc_auc_score(y_test, probs):.3f}"
        )

evaluate_models(df_imbalanced, "Imbalanced (Real-world)")
evaluate_models(df_balanced, "Balanced")
evaluate_models(df_biased, "Diabetic-Favored")
