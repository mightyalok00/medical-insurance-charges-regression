# Data Leakage Prevention

The project reduces leakage risk by:

1. Separating the final test set before model selection.
2. Keeping categorical encoding and transformations inside scikit-learn pipelines.
3. Selecting polynomial degree with training-only cross-validation.
4. Tuning Decision Tree hyperparameters with training-only validation.
5. Evaluating the selected models on the untouched test set only after selection.

The target `charges` is never included in the predictor matrix.
