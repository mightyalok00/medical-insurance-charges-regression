# 🏥 Medical Insurance Charges Regression

End-to-end machine learning portfolio project for predicting medical insurance charges using **Linear Regression**, **Polynomial Regression**, and **Decision Tree Regression**.

The project covers the complete workflow from raw data to deployment: data cleaning, exploratory analysis, leakage-safe preprocessing, model training, cross-validation, hyperparameter tuning, evaluation, interpretation, business analysis, saved model pipelines, and a professional Streamlit dashboard.

> **Portfolio focus:** reproducible regression workflow, model comparison, business interpretation, model selection, and deployment-ready project structure.

---

## 🚀 Live Demo

**Streamlit App:**  
https://medical-insurance-charges-regression.streamlit.app/

**GitHub Repository:**  
https://github.com/mightyalok00/medical-insurance-charges-regression

---

## 🎯 Project Objective

The objective is to estimate the target variable `charges` using the following customer attributes:

- `age`
- `sex`
- `bmi`
- `children`
- `smoker`
- `region`

Three regression approaches are trained and compared to understand the trade-offs between **simplicity, interpretability, nonlinearity, flexibility, and predictive performance**.

---

## 📌 Executive Summary

| Item | Result |
|---|---|
| Original dataset | **1,338 rows** |
| Cleaned dataset | **1,337 rows** |
| Exact duplicates removed | **1** |
| Missing values | **None found** |
| Target | `charges` |
| Train/test split | **80/20** |
| Cross-validation | **5-fold** |
| Models trained | **3** |
| Evaluation metrics | **MAE, MSE, RMSE, R²** |
| Best saved holdout model | **Decision Tree Regression** |
| Reproducibility seed | `random_state=42` |

The final test set is kept untouched during model selection and tuning. Preprocessing is contained inside scikit-learn pipelines to reduce leakage risk.

---

# 📊 Three-Model Comparison

All three models were trained and evaluated using the same final test split.

| Model | Test MAE | Test MSE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|
| **Decision Tree Regression** | **2,621.31** | **18,886,631.25** | **4,345.88** | **0.8972** |
| Polynomial Regression | 2,867.32 | 21,585,843.72 | 4,646.06 | 0.8825 |
| Linear Regression | 4,177.05 | 35,478,020.68 | 5,956.34 | 0.8069 |

Source: `reports/model_comparison.csv`

## 🏆 Which model performs best?

For this dataset and saved holdout-test evaluation, **Decision Tree Regression performs best**.

It has:

- the **lowest Test MAE**
- the **lowest Test MSE**
- the **lowest Test RMSE**
- the **highest Test R²**

The tuned Decision Tree achieved:

- **Test MAE:** 2,621.31
- **Test RMSE:** 4,345.88
- **Test R²:** 0.8972

### RMSE improvement over Linear Regression

- Polynomial Regression improves RMSE by approximately **22.0%**
- Decision Tree Regression improves RMSE by approximately **27.0%**

This indicates that nonlinear relationships and interactions are important in the dataset.

> **Important:** “Best” here means best on the saved holdout metrics for this project. Real production model selection would also require stability checks, external validation, fairness testing, monitoring, and domain review.

---

## 🧠 Model-by-Model Interpretation

### 1. Linear Regression

**Role:** baseline model

**Strengths**

- simple
- transparent
- easy to explain
- useful for directional interpretation

**Limitation**

Its lower R² suggests that a purely linear structure does not fully capture the relationships in insurance charges.

### 2. Polynomial Regression

**Role:** nonlinear extension of the linear baseline

**Strengths**

- captures nonlinear effects
- models interactions
- substantially improves RMSE over Linear Regression

**Limitation**

Polynomial feature expansion increases complexity and may reduce interpretability as the number of terms grows.

### 3. Decision Tree Regression

**Role:** strongest predictive model in the saved evaluation

**Strengths**

- captures nonlinear thresholds naturally
- models interactions without manual polynomial expansion
- strongest saved MAE, MSE, RMSE, and R² results

**Limitation**

Decision Trees can overfit if not controlled, which is why this project uses tuning and cross-validation.

---

# 🔮 Professional Streamlit Dashboard

The root-level `app.py` provides an interactive ML portfolio application.

### Main dashboard sections

- **Executive Overview**
- **Prediction Lab**
- **Model Performance**
- **Drivers & Insights**
- **Data Explorer**
- **Project Files**

### Model selector

The Prediction Lab now includes a model-selection filter with:

- `Compare all models`
- `Linear Regression`
- `Polynomial Regression`
- `Decision Tree Regression`

Users can therefore enter one customer profile and either:

1. compare predictions from all three saved models side by side, or
2. select one model and generate an individual prediction.

When **Compare all models** is selected, the app displays:

- prediction values from all three models
- a side-by-side comparison table
- a prediction comparison chart
- the portfolio default prediction from the strongest saved test model

The dashboard also identifies **Decision Tree Regression** dynamically from the saved model-comparison report as the lowest-Test-RMSE model.

---

## 🎛️ Dashboard Filters

The sidebar provides interactive dataset filters for:

- 🎂 Age
- ⚖️ BMI
- 👶 Number of children
- 🚻 Sex
- 🚬 Smoker status
- 📍 Region

The filtered dataset updates the dashboard metrics and visualizations in real time.

Users can also download the currently filtered dataset as CSV.

---

# 💡 Key Business Findings

- Smokers have substantially higher observed mean charges than non-smokers in this dataset.
- Age and BMI show meaningful relationships with medical insurance charges.
- The nonlinear models outperform the simple Linear Regression baseline.
- Predicted-cost quartiles support descriptive customer segmentation.
- Region, sex, and number of children show smaller descriptive differences than smoking status in this sample.

These insights are **predictive and associational, not causal**.

The project is intended for education and portfolio demonstration only. It should not be used for actual insurance pricing, underwriting, eligibility decisions, or adverse-action decisions without actuarial, regulatory, legal, fairness, temporal, and external validation.

---

# 🔄 Machine Learning Workflow

```text
Raw Data
   ↓
Data Quality Checks
   ↓
Cleaning & Exploratory Data Analysis
   ↓
Feature / Target Separation
   ↓
80/20 Train-Test Split
   ↓
Leakage-Safe Preprocessing Pipelines
   ↓
Linear Regression
Polynomial Regression
Decision Tree Regression
   ↓
5-Fold Cross-Validation / Hyperparameter Tuning
   ↓
Untouched Final Test Evaluation
   ↓
MAE / MSE / RMSE / R² Comparison
   ↓
Model Interpretation & Business Analysis
   ↓
Saved Model Pipelines
   ↓
Professional Streamlit Dashboard
```

---

# 🛡️ Leakage Prevention

The project follows a leakage-conscious training workflow:

- `charges` is removed from the feature matrix before model training
- preprocessing remains inside scikit-learn pipelines
- model tuning is performed using training data only
- polynomial degree selection is performed using training-only validation
- Decision Tree tuning is performed using training-only validation
- the holdout test set is reserved for final evaluation

This provides a more reliable comparison between the three regression approaches.

---

# 🔁 Reproducibility

The project uses reproducible settings wherever applicable:

```python
random_state=42
```

This is used for the train/test split, Decision Tree workflow, and shuffled cross-validation.

The saved `.pkl` files contain the trained preprocessing + regression pipelines so predictions can be reproduced directly from raw model inputs.

---

# 📁 Project Structure

```text
medical-insurance-charges-regression/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── data/
│   ├── raw/
│   │   └── insurance.csv
│   └── processed/
│       └── insurance_cleaned.csv
│
├── notebooks/
│   └── insurance_model_training.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── utils.py
│
├── models/
│   ├── linear_regression.pkl
│   ├── polynomial_regression.pkl
│   └── decision_tree_regression.pkl
│
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
│
├── images/
│   ├── actual_vs_predicted.png
│   ├── age_vs_charges.png
│   ├── bmi_vs_charges.png
│   ├── feature_importance.png
│   ├── model_comparison.png
│   └── smoker_vs_charges.png
│
└── docs/
    ├── Insurance_Regression_Project_Questions.docx
    ├── DEVELOPMENT_LOG.md
    └── supporting project documentation
```

---

# ✅ Assignment Coverage

All **17 project requirements** are covered.

The repository includes evidence for:

1. Dataset structure and data types
2. Data quality and cleaning
3. Categorical encoding
4. Numerical relationships and EDA
5. Smoking and insurance charges
6. Train-test split and leakage prevention
7. Linear Regression
8. Polynomial Regression
9. Decision Tree Regression
10. Model performance comparison
11. Feature importance and interpretation
12. Business use and limitations
13. Loop-based model training
14. Professional folder structure
15. Deep business analysis
16. Complete three-model comparison
17. Cross-validation and hyperparameter tuning

See:

```text
reports/question_coverage_checklist.csv
```

for the question-by-question audit.

---

# 🖼️ Visual Outputs

The project includes portfolio-ready visualizations such as:

- Actual vs Predicted charges
- Age vs Charges
- BMI vs Charges
- Smoker vs Charges
- Model Comparison
- Decision Tree Feature Importance

These are stored in the `images/` directory.

---

# 💾 Saved Model Artifacts

All three trained model pipelines are saved:

```text
models/
├── linear_regression.pkl
├── polynomial_regression.pkl
└── decision_tree_regression.pkl
```

This confirms the project trains, persists, and compares all three models rather than only reporting notebook outputs.

---

# ▶️ Run Locally

```bash
git clone https://github.com/mightyalok00/medical-insurance-charges-regression.git
cd medical-insurance-charges-regression

python -m venv .venv
```

### Windows

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

### Notebook

```bash
jupyter notebook notebooks/insurance_model_training.ipynb
```

---

# 📦 Main Deliverables

The repository includes:

- cleaned insurance dataset
- executed Jupyter notebook
- reusable preprocessing/training/evaluation code
- three trained regression pipelines
- cross-validation results
- model diagnostics
- coefficient reports
- Decision Tree feature-importance report
- predicted-cost segmentation
- project report in Markdown and PDF
- visualization assets
- assignment coverage checklist
- professional supporting documentation
- interactive Streamlit application

---

# 🧾 Final Conclusion

This project demonstrates a complete and reproducible regression workflow from raw data through model training, comparison, interpretation, persistence, and deployment.

**Linear Regression** establishes a transparent baseline but leaves meaningful nonlinear structure unexplained. **Polynomial Regression** improves predictive performance by modeling nonlinear effects and interactions. **Decision Tree Regression** achieves the strongest saved holdout performance, producing the lowest MAE, MSE, and RMSE and the highest R² among the three trained models.

### Final model conclusion

For this dataset and evaluation setup:

> **Decision Tree Regression is the strongest predictive model of the three based on the saved holdout metrics.**

However, the project does not treat test performance as the only consideration. A production insurance model would also require external validation, fairness assessment, calibration, monitoring, governance, legal review, actuarial review, and validation on newer data.

The final portfolio therefore demonstrates more than model accuracy: it shows **leakage prevention, reproducibility, model comparison, interpretability, business reasoning, responsible limitations, and deployment through Streamlit**.

---

## 👤 Author

**Alok Agarwal**

GitHub: https://github.com/mightyalok00

Live App: https://medical-insurance-charges-regression.streamlit.app/
