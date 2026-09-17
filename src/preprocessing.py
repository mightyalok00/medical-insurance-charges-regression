"""Data loading and preprocessing helpers for the insurance regression project."""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["age", "bmi", "children"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]
TARGET = "charges"

def load_and_clean(path):
    """Load CSV, remove exact duplicates, and return a clean DataFrame."""
    df = pd.read_csv(Path(path))
    return df.drop_duplicates().reset_index(drop=True)

def build_preprocessor():
    """Scale numeric columns and one-hot encode categorical columns."""
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])
