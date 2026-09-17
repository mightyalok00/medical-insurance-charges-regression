"""Train the three required regression models with loop-based evaluation."""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from preprocessing import build_preprocessor
from evaluate_models import regression_metrics

def build_models(poly_degree=2, tree_params=None):
    tree_params = tree_params or {"max_depth": 4, "random_state": 42}
    return {
        "Linear Regression": Pipeline([("prep", build_preprocessor()), ("model", LinearRegression())]),
        "Polynomial Regression": Pipeline([("prep", build_preprocessor()), ("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)), ("model", LinearRegression())]),
        "Decision Tree Regression": Pipeline([("prep", build_preprocessor()), ("model", DecisionTreeRegressor(**tree_params))]),
    }

def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = regression_metrics(y_test, model.predict(X_test))
        results.append({"Model": name, **metrics})
    return results
