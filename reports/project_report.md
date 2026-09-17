# Insurance Charges Regression - Complete Project Report

## 1. Dataset and Cleaning
The source contains 1,338 rows and seven columns. One exact duplicate was removed, leaving 1,337 observations. No missing values were detected.

## 2. Validation Design
The dataset was split 80/20 with `random_state=42`. Polynomial degree and Decision Tree hyperparameters were selected only with 5-fold cross-validation on the training data. The test set was reserved for final model comparison.

## 3. Final Three-Model Comparison

| Model                    |   Train MAE |   Test MAE |    Test MSE |   Test RMSE |   Train R2 |   Test R2 |   R2 Gap |
|:-------------------------|------------:|-----------:|------------:|------------:|-----------:|----------:|---------:|
| Decision Tree Regression |     2567.47 |    2621.31 | 1.88866e+07 |     4345.88 |      0.855 |     0.897 |   -0.042 |
| Polynomial Regression    |     2877.25 |    2867.32 | 2.15858e+07 |     4646.06 |      0.834 |     0.883 |   -0.049 |
| Linear Regression        |     4181.9  |    4177.05 | 3.5478e+07  |     5956.34 |      0.73  |     0.807 |   -0.077 |

Polynomial Regression improved test RMSE by **22.00%** relative to the linear baseline. The tuned Decision Tree improved test RMSE by **27.04%**. The train-versus-test R² gaps are reported explicitly in `model_comparison.csv`, and `model_diagnostics.csv` provides a generalization diagnosis for each model.

## 4. Feature Contribution Across Model Families
Linear Regression coefficients are saved to `linear_regression_coefficients.csv`; Polynomial Regression terms and interactions are saved to `polynomial_regression_coefficients.csv`; Decision Tree split importance is saved to `decision_tree_feature_importance.csv`. Coefficients and tree importance are different measures and should not be treated as directly interchangeable. Because numerical predictors are standardized and polynomial features are expanded, coefficient magnitude must be interpreted with the transformed design matrix in mind.

## 5. Deep Business Analysis
Smokers had observed mean charges of **32,050.23**, compared with **8,440.66** for non-smokers. Age, BMI, smoking status, number of children, sex and region are all analyzed in grouped summaries. Age/BMI/smoking combinations are profiled, and model-based predicted-cost quartiles are saved to `predicted_cost_segments.csv`.

### Business applications
- **Pricing analysis:** compare expected-cost patterns across observable customer groups as an analytical input, not an automated premium-setting rule.
- **Customer segmentation:** use predicted-cost bands to describe portfolio composition and investigate which characteristics are associated with different expected costs.
- **Risk assessment:** use smoking, age, BMI and nonlinear interactions as predictive signals subject to actuarial and fairness review.
- **Cost management:** identify high predicted-cost groups for further study of preventive-care or care-management opportunities, not automatic adverse action.

## 6. Model Selection and Complexity
The final model is **Decision Tree Regression**, because it has the lowest held-out test RMSE (4345.88) and strongest test R² (0.897) in this run. Polynomial Regression demonstrates that nonlinear terms improve on the linear baseline. The tuned Decision Tree captures additional nonlinear interactions while cross-validation constrains depth and leaf/split sizes to control complexity.

## 7. Limitations and Fairness
This is an educational dataset with limited variables and unknown population representativeness. Model outputs describe associations, not causal effects. Real insurance pricing, eligibility, underwriting, or adverse-action use would require broader actuarial variables, legal review, protected-class/fairness analysis, calibration, temporal validation, external validation, monitoring, and governance. Sex and region differences in particular must not be converted into causal or normative conclusions.

## 8. Assignment Coverage
All 17 requested questions are mapped to evidence in `question_coverage_checklist.csv`.
