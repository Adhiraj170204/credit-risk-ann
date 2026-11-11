"""Load and clean the Credit Scoring data, and build the preprocessing pipeline.

Target: Status -> 1 = "bad" (applicant defaulted), 0 = "good".
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

DATA = Path(__file__).resolve().parents[1] / "data" / "CreditScoring.csv"
SEED = 42

NUMERIC = ["Seniority", "Time", "Age", "Expenses", "Income", "Amount", "Price",
           "Finrat", "Savings", "loan_to_income"]
SKEWED = ["Assets", "Debt"]                      # heavy right tail -> log1p
CATEGORICAL = ["Home", "Marital", "Records", "Job"]


def load_clean() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df = df.iloc[:, :16].copy()                  # drop the pre-binned *R columns
    df["target"] = (df.pop("Status") == "bad").astype(int)
    # Engineered feature: loan amount relative to monthly income
    df["loan_to_income"] = df["Amount"] / df["Income"].clip(lower=1)
    return df


def split(df: pd.DataFrame):
    """Stratified 70 / 15 / 15 train / validation / test split."""
    X, y = df.drop(columns="target"), df["target"]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
    return X_tr, X_val, X_te, y_tr, y_val, y_te


def make_preprocessor() -> ColumnTransformer:
    """Scaling matters for neural nets: gradient descent converges badly on raw
    features that range from 0 to 300,000. Fitted on the training set only."""
    log_scale = Pipeline([("log", FunctionTransformer(np.log1p)),
                          ("scale", StandardScaler())])
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("skew", log_scale, SKEWED),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])


if __name__ == "__main__":
    df = load_clean()
    print(df.shape, f"default rate = {df.target.mean():.3f}")
    for part, y in zip(["train", "val", "test"], split(df)[3:]):
        print(part, len(y), f"{y.mean():.3f}")
