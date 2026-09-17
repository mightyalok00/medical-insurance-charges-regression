# Reproducibility

- `random_state=42` is used for train/test splitting and reproducible model behavior where supported.
- Cross-validation uses shuffled folds with a fixed random state.
- Preprocessing is kept inside scikit-learn pipelines to reduce leakage risk.
- Saved model artifacts are stored under `models/`.
- Generated metrics and analysis outputs are stored under `reports/`.
