# 🏥 Medical Insurance Charges Regression

End-to-end machine learning portfolio project for predicting medical insurance charges with **Linear Regression**, **Polynomial Regression**, and **Decision Tree Regression**.

The repository covers data cleaning, EDA, leakage-safe preprocessing, cross-validation, hyperparameter tuning, model comparison, interpretation, business analysis, saved model pipelines, reports, and an interactive Streamlit application.

> **Portfolio focus:** reproducible regression workflow, transparent model comparison, business interpretation, and deployment-ready project structure.

## 🚀 Live Demo

**Streamlit App:** https://medical-insurance-charges-regression.streamlit.app/

**GitHub:** https://github.com/mightyalok00/medical-insurance-charges-regression

---

## 🎯 Project Objective

Predict the target variable `charges` using:

- `age`
- `sex`
- `bmi`
- `children`
- `smoker`
- `region`

The project compares three regression approaches to show the trade-off between **simplicity, interpretability, nonlinear modeling, and predictive performance**.

---

## 📌 Executive Summary

| Item | Result |
|---|---|
| Original dataset | **1,338 rows** |
| Cleaned dataset | **1,337 rows** |
| Exact duplicates removed | **1** |
| Missing values | **None found** |
| Train/test split | **80/20** |
| Cross-validation | **5-fold** |
| Models trained | **3** |
| Metrics | **MAE, MSE, RMSE, R²** |
| Best saved holdout model | **Decision Tree Regression** |
| Reproducibility | `random_state=42` |

The final test set is kept untouched during model selection and tuning. Preprocessing remains inside scikit-learn pipelines to reduce leakage risk.

---

## 📊 Three-Model Comparison

All three models use the same final holdout split.

| Model | Test MAE | Test MSE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|
| **Decision Tree Regression** | **2,621.31** | **18,886,631.25** | **4,345.88** | **0.8972** |
| Polynomial Regression | 2,867.32 | 21,585,843.72 | 4,646.06 | 0.8825 |
| Linear Regression | 4,177.05 | 35,478,020.68 | 5,956.34 | 0.8069 |

Source: `reports/model_comparison.csv`

### 🏆 Best model in this project

Based on the saved holdout-test results, **Decision Tree Regression** performs best because it has the lowest MAE, MSE, and RMSE and the highest R² among the three models.

Compared with Linear Regression:

- Polynomial Regression reduces Test RMSE by about **22%**.
- Decision Tree Regression reduces Test RMSE by about **27%**.

This suggests that nonlinear relationships and feature interactions are important in this dataset.

> “Best” means best on this project’s saved holdout metrics. A production model would still need external validation, monitoring, fairness testing, domain review, and stability checks.

---

## 🧠 Model Interpretation

### Linear Regression

A simple and transparent baseline. It is easy to explain but does not capture all nonlinear structure in the data.

### Polynomial Regression

A nonlinear extension of the linear baseline. It improves predictive performance by modeling interactions and nonlinear effects while retaining a regression form.

### Decision Tree Regression

The strongest saved holdout performer. It naturally captures thresholds and interactions without manual polynomial expansion. The project uses tuning and cross-validation to control complexity.

---

## 🔮 Streamlit Dashboard

The root-level `app.py` provides a professional interactive dashboard with:

- **Executive Overview**
- **Prediction Lab**
- **Model Performance**
- **Drivers & Insights**
- **Data Explorer**
- **Project Files**

### 🧠 Always-visible model selector

The **left sidebar** now contains the prediction model selector, so it is visible immediately when the app opens.

Available options:

- `Compare all models`
- `Linear Regression`
- `Polynomial Regression`
- `Decision Tree Regression`

The selection controls the **Prediction Lab** tab.

When `Compare all models` is selected, the app shows predictions from every successfully loaded model in a comparison table and chart. When a single model is selected, the app displays that model’s prediction and descriptive cost band.

The sidebar also shows:

- how many of the three model artifacts loaded successfully
- model-load diagnostics if any artifact fails
- the best saved holdout model

### 🎛️ Dataset filters

The sidebar also includes filters for:

- 🎂 Age
- ⚖️ BMI
- 👶 Children
- 🚻 Sex
- 🚬 Smoker status
- 📍 Region

The reset button now resets the dataset filters correctly.

---

## 💡 Key Business Findings

- Smokers have substantially higher observed mean charges than non-smokers in this dataset.
- Age and BMI show meaningful relationships with charges.
- Nonlinear models outperform the simple linear baseline on the saved holdout set.
- Predicted-cost quartiles support descriptive segmentation.
- Region, sex, and number of children show smaller descriptive differences than smoking status in this sample.

These are **predictive/associational findings, not causal claims**.

The project is for education and portfolio demonstration only and should not be used for real insurance pricing, underwriting, eligibility, or adverse-action decisions without actuarial, legal, regulatory, fairness, temporal, and external validation.

---

## 🔄 Machine Learning Workflow

```text
Raw Data
   ↓
Data Quality Checks
   ↓
Cleaning & EDA
   ↓
Feature / Target Separation
   ↓
80/20 Train-Test Split
   ↓
Leakage-Safe Pipelines
   ↓
Linear Regression
Polynomial Regression
Decision Tree Regression
   ↓
5-Fold Cross-Validation / Tuning
   ↓
Untouched Final Test Evaluation
   ↓
MAE / MSE / RMSE / R² Comparison
   ↓
Interpretation & Business Analysis
   ↓
Saved Models + Reports + Streamlit App
```

---

## 🛡️ Leakage Prevention

- `charges` is excluded from the feature matrix.
- preprocessing stays inside scikit-learn pipelines.
- polynomial degree selection uses training-only validation.
- Decision Tree tuning uses training-only validation.
- the final holdout test set is reserved for final evaluation.

---

## 📁 Project Structure

```text
medical-insurance-charges-regression/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── .streamlit/
│   └── config.toml
├── data/
│   ├── raw/insurance.csv
│   └── processed/insurance_cleaned.csv
├── notebooks/
│   └── insurance_model_training.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── utils.py
├── models/
│   ├── linear_regression.pkl
│   ├── polynomial_regression.pkl
│   └── decision_tree_regression.pkl
├── reports/
│   ├── model_comparison.csv
│   ├── model_diagnostics.csv
│   ├── cross_validation_results.csv
│   ├── linear_regression_coefficients.csv
│   ├── polynomial_regression_coefficients.csv
│   ├── decision_tree_feature_importance.csv
│   ├── predicted_cost_segments.csv
│   ├── question_coverage_checklist.csv
│   ├── project_report.md
│   └── project_report.pdf
├── images/
└── docs/
```

---

## ✅ Assignment Coverage

All **17 project requirements** are covered, including dataset structure, cleaning, categorical encoding, EDA, smoker analysis, leakage prevention, all three regression models, metric comparison, interpretation, loop-based training, professional folder structure, deep business analysis, complete three-model comparison, cross-validation, and hyperparameter tuning.

See `reports/question_coverage_checklist.csv` for the detailed audit.

---

## ▶️ Run Locally

```bash
git clone https://github.com/mightyalok00/medical-insurance-charges-regression.git
cd medical-insurance-charges-regression
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Notebook:

```bash
jupyter notebook notebooks/insurance_model_training.ipynb
```

---

## 🔧 Deployment Notes

The app uses paths relative to `app.py`, so it does not depend on a machine-specific Windows path.

Deployment dependencies are constrained to compatible major-version ranges in `requirements.txt`, including Streamlit and Altair.

If Streamlit Cloud still shows an older interface after a commit, wait for the redeploy and perform a hard refresh (`Ctrl + F5`). The current app should show the **model selector at the top of the left sidebar**.

---

## 📦 Saved Deliverables

- three trained `.pkl` pipelines
- executed Jupyter notebook
- cleaned dataset
- model comparison and cross-validation results
- coefficient and feature-importance reports
- predicted-cost segmentation output
- Markdown and PDF project reports
- visualization images
- professional Streamlit dashboard
- assignment coverage checklist
- supporting documentation

---

## 🧾 Conclusion

This project demonstrates a complete regression workflow from raw data through model evaluation and deployment.

**Linear Regression** provides a transparent baseline. **Polynomial Regression** improves performance by capturing nonlinear effects and interactions. **Decision Tree Regression** achieves the strongest saved holdout performance, with the lowest error metrics and highest R² among the three models.

For this dataset and evaluation setup, the tuned **Decision Tree Regression** is the strongest predictive model of the three. Model choice in a real production setting would still need to balance predictive accuracy with interpretability, stability, fairness, operational constraints, and validation on new data.

---

## 👤 Author

**Alok Agarwal**

GitHub: https://github.com/mightyalok00

Live App: https://medical-insurance-charges-regression.streamlit.app/
