from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "Churn"
ID_COLUMN = "customerID"


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    cost: int


def load_data(path: str) -> pd.DataFrame:
    """Load and validate the IBM Telco Customer Churn sample."""
    data = pd.read_csv(path)
    required = {ID_COLUMN, TARGET, "tenure", "MonthlyCharges", "TotalCharges"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data = data.copy()
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data[TARGET] = data[TARGET].map({"No": 0, "Yes": 1})
    if data[TARGET].isna().any():
        raise ValueError("Churn must contain only 'Yes' or 'No'.")
    return data


def split_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return data.drop(columns=[TARGET, ID_COLUMN]), data[TARGET].astype(int)


def build_models(features: pd.DataFrame) -> dict[str, Pipeline]:
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = features.select_dtypes(exclude="number").columns.tolist()
    preprocessor = ColumnTransformer(
        [
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )
    return {
        "Logistic regression": Pipeline([("preprocessor", preprocessor), ("model", LogisticRegression(class_weight="balanced", max_iter=2_000, random_state=42))]),
        "Random forest": Pipeline([("preprocessor", preprocessor), ("model", RandomForestClassifier(n_estimators=250, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1))]),
    }


def choose_cost_threshold(y_true, probabilities: np.ndarray, false_negative_cost: int = 5, false_positive_cost: int = 1) -> ThresholdResult:
    """Choose a validation threshold using an explicit retention-cost assumption."""
    best = ThresholdResult(threshold=0.5, cost=np.iinfo(np.int32).max)
    y_array = np.asarray(y_true)
    for threshold in np.arange(0.10, 0.81, 0.01):
        predictions = (probabilities >= threshold).astype(int)
        false_negatives = int(((y_array == 1) & (predictions == 0)).sum())
        false_positives = int(((y_array == 0) & (predictions == 1)).sum())
        cost = false_negatives * false_negative_cost + false_positives * false_positive_cost
        if cost < best.cost:
            best = ThresholdResult(round(float(threshold), 2), cost)
    return best


def classification_metrics(y_true, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }
