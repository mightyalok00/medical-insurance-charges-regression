# Medical Insurance Charges Regression

Portfolio-ready regression project using **Linear Regression, Polynomial Regression, and Decision Tree Regression** to predict medical insurance `charges`.

## Key Findings / Business Impact

- Cleaned dataset: **1,337 rows** after removing **1 exact duplicate**; no missing values were found.
- Final model selection used training-only cross-validation and one untouched 20% test set.
- **Decision Tree Regression** achieved the lowest test RMSE (**4345.88**) and test R² (**0.897**).
- Polynomial Regression reduced RMSE by **22.00%** versus Linear Regression; the tuned Decision Tree reduced RMSE by **27.04%**.
- Smokers had substantially higher observed mean charges (**32,050.23**) than non-smokers (**8,440.66**).
- Predicted-cost quartiles are included for descriptive customer segmentation and business analysis.
- Model findings are **predictive/associational, not causal** and are not sufficient for real pricing, eligibility, or adverse-action decisions without actuarial, legal, fairness, temporal, and external validation.

## Assignment Coverage

All **17 project questions are explicitly covered**. See `reports/question_coverage_checklist.csv` for a question-by-question audit. The notebook includes:

- dataset structure, cleaning and encoding
- EDA and smoking-cost analysis
- leakage-safe train/test splitting
- Linear, Polynomial, and Decision Tree regression
- loop-based reusable training
- 5-fold cross-validation and hyperparameter tuning
- MAE, MSE, RMSE and R² comparison
- Linear and Polynomial coefficient interpretation
- Decision Tree feature importance
- predicted-cost customer segmentation
- pricing, risk-assessment and cost-management implications
- explicit overfitting/underfitting and complexity-vs-performance diagnosis
- limitations and fairness considerations

## Folder Structure

```text
Medical Insurance/
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
├── docs/Insurance_Regression_Project_Questions.docx
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Notebook Commenting

The notebook contains **23 executable code cells**, each beginning with a short comment explaining the purpose of that cell. The packaged notebook has been executed successfully end-to-end.

## Streamlit Dashboard

The root-level `app.py` provides an interactive portfolio dashboard with:

- insurance-charge prediction using the tuned Decision Tree pipeline
- age, sex, BMI, children, smoking-status, and region inputs
- model-comparison metrics and charts
- business insights and feature-importance output
- predicted-cost segmentation and cleaned-data preview

The prediction dashboard is for **educational/portfolio use only** and is not intended for real insurance pricing or eligibility decisions.

## Run Locally

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt

# Launch Streamlit dashboard
streamlit run app.py

# Or open the analysis notebook
jupyter notebook notebooks/insurance_model_training.ipynb
```

## Reproducibility

`random_state=42` is used for the train/test split, Decision Tree, and shuffled cross-validation. Preprocessing stays inside scikit-learn pipelines to reduce leakage risk.

## Windows Project Location

This package is prepared to be extracted so the project folder is:

```text
D:\Medical-Insurances\Medical Insurance
```

Run the dashboard from PowerShell:

```powershell
cd "D:\Medical-Insurances\Medical Insurance"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The dashboard uses paths relative to `app.py`, so models, reports, and data load correctly from `D:\Medical-Insurances\Medical Insurance` without hard-coded machine-specific paths. The notebook also resolves the project root automatically when launched either from the project root or from the `notebooks` folder.

## Dashboard Filters

The Streamlit dashboard includes emoji-based sidebar filters for:

- 🎂 Age range
- ⚖️ BMI range
- 👶 Number of children
- 🚻 Sex
- 🚬 Smoker status
- 📍 Region

The filtered dataset tab updates record count, average charges, average age, average BMI, descriptive statistics, and provides an ⬇️ CSV download for the current selection.
