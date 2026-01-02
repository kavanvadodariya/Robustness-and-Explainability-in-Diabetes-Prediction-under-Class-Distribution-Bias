# Robustness-and-Explainability-in-Diabetes-Prediction-under-Class-Distribution-Bias

This project investigates early-stage diabetes risk prediction using machine learning under different class distribution settings. The study evaluates how class imbalance affects model performance and emphasizes recall and ROC-AUC over accuracy.

### Datasets
- Diabetes Prediction Dataset (~100k samples)
- NHANES (Population-level survey data, SAS XPT format)

Note: NHANES data must be downloaded separately from the official CDC website.

### Features
- Demographic: Age, Gender
- Anthropometric: BMI
- Clinical/Lifestyle: Hypertension, Smoking History

Removed features:
- HbA1c
- Blood Glucose

### Experimental Setup
- Models: SVM (RBF), Random Forest, XGBoost
- Data splits: 75% train / 25% test (stratified)
- Dataset variants:
  - Imbalanced (original distribution)
  - Balanced (50:50 undersampling)
  - Diabetic-favored (70:30)
- Evaluation metrics: Accuracy, Precision, Recall, ROC-AUC

### Results Summary
- Models achieved high accuracy on imbalanced datasets but low recall for diabetic cases.
- Balancing the dataset significantly improved recall and ROC-AUC across all models.
- Random Forest and SVM occasionally outperformed XGBoost in balanced settings.
- Subgroup analysis showed higher recall in female populations and reduced recall in younger age groups.

### Explainability
SHAP was used to interpret XGBoost predictions. Age, BMI, hypertension, and smoking history were identified as the most influential features across datasets.

### Reproducibility
1. Clone the repository
2. Install dependencies
3. Download datasets
4. Run notebooks/scripts in order

## License
- Code: MIT License  
- Research Paper: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
