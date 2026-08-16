import numpy as np
import pandas as pd

from src.churn import choose_cost_threshold, classification_metrics, load_data


def test_load_data_converts_target_and_total_charges(tmp_path):
    path = tmp_path / "sample.csv"
    pd.DataFrame({"customerID": ["A", "B"], "tenure": [1, 2], "MonthlyCharges": [20.0, 30.0], "TotalCharges": ["20.0", " "], "Churn": ["No", "Yes"]}).to_csv(path, index=False)
    data = load_data(str(path))
    assert data["Churn"].tolist() == [0, 1]
    assert np.isnan(data.loc[1, "TotalCharges"])


def test_cost_threshold_returns_valid_threshold():
    result = choose_cost_threshold(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.45, 0.9]))
    assert 0.10 <= result.threshold <= 0.80
    assert result.cost >= 0


def test_classification_metrics_are_bounded():
    metrics = classification_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), 0.5)
    assert all(0.0 <= value <= 1.0 for value in metrics.values())
