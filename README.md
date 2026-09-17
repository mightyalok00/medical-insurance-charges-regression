# Medical Insurance Charges Regression

End-to-end machine learning portfolio project for predicting medical insurance charges using **Linear Regression**, **Polynomial Regression**, and **Decision Tree Regression**. The project covers data cleaning, exploratory analysis, leakage-safe preprocessing, model training, cross-validation, evaluation, interpretation, business analysis, saved model artifacts, and a professional Streamlit dashboard.

> **Portfolio focus:** reproducible regression workflow, model comparison, business interpretation, and deployment-ready project structure.

## Live Demo

🚀 **Try the deployed Streamlit application:** [Medical Insurance Charges Regression App](https://medical-insurance-charges-regression.streamlit.app/)

---

## Project Objective

The goal is to predict the target variable `charges` from customer attributes including:

- `age`
- `sex`
- `bmi`
- `children`
- `smoker`
- `region`

The project compares three regression approaches to understand the trade-off between **simplicity, nonlinearity, interpretability, and predictive performance**.

---

## Executive Summary

- Original dataset: **1,338 rows**
- Cleaned dataset: **1,337 rows** after removing **1 exact duplicate**
- Missing values: **none found**
- Train/test strategy: **80/20 split** with the final test set kept untouched during model selection
- Validation: **5-fold cross-validation** on the training data
- Models trained: **3**
- Final evaluation metrics: **MAE, MSE, RMSE, and R²**
- Best-performing model on the saved test results: **Decision Tree Regression**
- Streamlit dashboard compares predictions from all three saved model pipelines

---

## Model Comparison

All three models were trained and evaluated on the same final test split.

| Model | Test MAE | Test MSE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|
| **Decision Tree Regression** | **2,621.31** | **18,886,631.25** | **4,345.88** | **0.8972** |
| Polynomial Regression | 2,867.32 | 21,585,843.72 | 4,646.06 | 0.8825 |
| Linear Regression | 4,177.05 | 35,478,020.68 | 5,956.34 | 0.8069 |

### Which model performs best?

Based on the saved holdout-test results, **Decision Tree Regression performs best** because it has:

- the **lowest Test MAE**
- the **lowest Test MSE**
- the **lowest Test RMSE**
- the **highest Test R²**

The tuned Decision Tree achieved a Test RMSE of **4,345.88** and Test R² of **0.8972**. Polynomial Regression finished second, while Linear Regression served as a useful baseline.

### Performance improvement

Compared with Linear Regression:

- Polynomial Regression reduced test RMSE by approximately **22.0%**
- Decision Tree Regression reduced test RMSE by approximately **27.0%**

This suggests that nonlinear relationships and interactions are important in this dataset.

---

## Model Interpretation

### Linear Regression

Linear Regression provides the simplest and most interpretable baseline. It is useful for understanding overall directional relationships, but its lower R² indicates that a purely linear structure does not fully capture the patterns in insurance charges.

### Polynomial Regression

Polynomial Regression improves substantially over the linear baseline by modeling nonlinear relationships and interactions. It offers a strong middle ground between interpretability and predictive performance.

### Decision Tree Regression

Decision Tree Regression produced the strongest test performance. It can naturally model nonlinear thresholds and feature interactions without requiring manual polynomial expansion.

For this project, the tuned Decision Tree is used as the primary prediction model while the Streamlit dashboard also displays predictions from all three trained models for comparison.

---

## Key Business Findings

- Smokers had substantially higher observed mean charges than non-smokers in this dataset.
- Age and BMI show meaningful relationships with insurance charges.
- Nonlinear models capture patterns that the simple linear baseline misses.
- Predicted-cost quartiles are included for descriptive customer segmentation.
- Region, sex, and number of children show smaller descriptive differences than smoking status in this sample.

> These findings are **predictive and associational, not causal**. The project is for educational and portfolio purposes and is not suitable for real insurance pricing, underwriting, eligibility, or adverse-action decisions without actuarial, legal, fairness, temporal, and external validation.

---

## Machine Learning Workflow

```text
Raw Data
   ↓
Data Quality Checks
   ↓
Cleaning & EDA
   ↓
Train/Test Split
   ↓
Leakage-Safe Preprocessing Pipelines
   ↓
Linear Regression
Polynomial Regression
Decision Tree Regression
   ↓
5-Fold Cross-Validation / Tuning
   ↓
Final Untouched Test Evaluation
   ↓
Model Comparison & Interpretation
   ↓
Saved Models + Reports + Streamlit Dashboard
```

---

## Leakage Prevention and Reproducibility

The workflow was designed to reduce data leakage:

- `charges` is separated from the feature matrix before training
- preprocessing stays inside scikit-learn pipelines
- cross-validation and tuning are performed using training data only
- the final test set is reserved for final evaluation
- `random_state=42` is used for reproducibility where applicable

---

## Streamlit Dashboard

The root-level `app.py` provides a professional interactive dashboard with:

- executive project overview
- side-by-side predictions from all three trained models
- age, sex, BMI, children, smoker, and region inputs
- model-performance comparison
- business and feature-importance insights
- predicted-cost segmentation
- interactive dataset filters
- filtered CSV download
- access to saved reports and diagnostics

### Live application

👉 [Open the Medical Insurance Charges Regression Streamlit App](https://medical-insurance-charges-regression.streamlit.app/)

Run it locally with:

```bash
streamlit run app.py
```

---

## Project Structure

```text
medical-insurance-charges-regression/
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
    └── additional professional documentation
```

---

## Assignment Coverage

All **17 project requirements** are covered, including:

- dataset structure and data types
- data quality and cleaning
- categorical encoding
- exploratory data analysis
- smoking vs. insurance charges analysis
- train/test split and leakage prevention
- Linear Regression
- Polynomial Regression
- Decision Tree Regression
- model-performance comparison
- feature importance and interpretation
- business use and limitations
- loop-based model training
- professional folder structure
- deep business analysis
- complete three-model comparison
- cross-validation and hyperparameter tuning

See `reports/question_coverage_checklist.csv` for the detailed audit.

---

## Run Locally

```bash
git clone https://github.com/mightyalok00/medical-insurance-charges-regression.git
cd medical-insurance-charges-regression

python -m venv .venv

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

To work with the notebook:

```bash
jupyter notebook notebooks/insurance_model_training.ipynb
```

---

## Saved Deliverables

The repository includes:

- three trained `.pkl` model pipelines
- executed Jupyter notebook
- cleaned dataset
- model comparison and cross-validation results
- coefficient and feature-importance reports
- business segmentation output
- project report in Markdown and PDF
- visualization images
- Streamlit application
- assignment coverage checklist
- professional supporting documentation

---

## Conclusion

This project demonstrates a complete regression workflow from raw data to deployment-ready portfolio presentation.

**Linear Regression** provides a transparent baseline but underfits some of the nonlinear structure in the data. **Polynomial Regression** improves performance by capturing nonlinear effects and interactions. **Decision Tree Regression** achieves the strongest saved test performance, with the lowest error metrics and the highest R² among the three models.

For this dataset and evaluation setup, the **tuned Decision Tree Regression is the strongest predictive model of the three**. However, model choice should still consider interpretability, stability, fairness, operational requirements, and validation on new data rather than relying only on a single test result.

The project therefore demonstrates not only model training, but also **responsible model comparison, leakage prevention, business interpretation, reproducibility, and deployment through Streamlit**.

---

## Author

**Alok Agarwal**

GitHub: [mightyalok00](https://github.com/mightyalok00)

Live App: [Medical Insurance Charges Regression](https://medical-insurance-charges-regression.streamlit.app/)
