import shap
import matplotlib.pyplot as plt

# Create SHAP explainer
explainer = shap.TreeExplainer(xgb_model)

# Compute SHAP values on test set
shap_values_bal = explainer.shap_values(X_test)

# Feature names
feature_names = X.columns.tolist()

# Global importance (bar plot)
plt.figure(figsize=(7,5))
shap.summary_plot(
    shap_values_bal,
    X_test,
    feature_names=feature_names,
    plot_type="bar",
    max_display=10,
    show=False
)
plt.title(
    "Global Feature Importance for Diabetes Prediction\n"
    "(XGBoost Model – Balanced Dataset)",
    pad=12
)
plt.xlabel("Mean |SHAP Value| (Average Impact on Model Output)")
plt.tight_layout()
plt.show()

# Directional impact (beeswarm)
plt.figure(figsize=(7,5))
shap.summary_plot(
    shap_values_bal,
    X_test,
    feature_names=feature_names,
    max_display=10,
    show=False
)
plt.title(
    "Feature Impact and Direction on Diabetes Prediction\n"
    "(XGBoost Model – Balanced Dataset)",
    pad=12
)
plt.xlabel("SHAP Value (Impact on Model Output)")
plt.tight_layout()
plt.show()
