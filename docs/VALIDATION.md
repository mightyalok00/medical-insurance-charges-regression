# Validation Strategy

The project separates model selection from final evaluation.

- Training data is used for cross-validation and hyperparameter selection.
- Polynomial degree is selected with training-only validation.
- Decision Tree complexity is tuned with training-only validation.
- The final test set remains untouched until the selected models are evaluated.
- Train/test gaps are reviewed for signs of overfitting or underfitting.
