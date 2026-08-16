from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split

from src.churn import build_models, choose_cost_threshold, classification_metrics
from src.churn import load_data, split_features


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "telco_customer_churn.csv"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def _feature_importance(model) -> pd.Series:
    names = model.named_steps["preprocessor"].get_feature_names_out()
    estimator = model.named_steps["model"]
    values = estimator.coef_[0] if hasattr(estimator, "coef_") else estimator.feature_importances_
    return pd.Series(values, index=names).sort_values(key=np.abs, ascending=False)


def main() -> None:
    data = load_data(str(DATA_PATH))
    features, target = split_features(data)
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=42, stratify=target)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.25, random_state=42, stratify=y_train)

    fitted, rows = {}, []
    for name, model in build_models(features).items():
        model.fit(x_train, y_train)
        fitted[name] = model
        probabilities = model.predict_proba(x_valid)[:, 1]
        rows.append({"model": name, "validation_roc_auc": classification_metrics(y_valid, probabilities, 0.5)["roc_auc"]})

    comparison = pd.DataFrame(rows).sort_values("validation_roc_auc", ascending=False)
    best_name = str(comparison.iloc[0]["model"])
    best_model = fitted[best_name]
    validation_probabilities = best_model.predict_proba(x_valid)[:, 1]
    threshold_result = choose_cost_threshold(y_valid, validation_probabilities)
    test_probabilities = best_model.predict_proba(x_test)[:, 1]
    test_metrics = classification_metrics(y_test, test_probabilities, threshold_result.threshold)
    test_predictions = (test_probabilities >= threshold_result.threshold).astype(int)

    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(REPORTS / "model_comparison.csv", index=False)
    metrics = {
        "rows": int(len(data)),
        "overall_churn_rate": float(target.mean()),
        "selected_model": best_name,
        "decision_threshold": threshold_result.threshold,
        "validation_cost_units": threshold_result.cost,
        "cost_assumption": "false negative = 5 units; false positive = 1 unit",
        "test": test_metrics,
    }
    (REPORTS / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    data.groupby("Contract")["Churn"].mean().sort_values().mul(100).plot.barh(ax=axes[0, 0], color="#377eb8")
    axes[0, 0].set(title="Churn rate by contract", xlabel="Customers who churned (%)", ylabel="")
    for name, model in fitted.items():
        probabilities = model.predict_proba(x_test)[:, 1]
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
        auc = classification_metrics(y_test, probabilities, 0.5)["roc_auc"]
        axes[0, 1].plot(false_positive_rate, true_positive_rate, label=f"{name} ({auc:.3f})")
    axes[0, 1].plot([0, 1], [0, 1], "--", color="grey")
    axes[0, 1].set(title="Holdout ROC curves", xlabel="False-positive rate", ylabel="True-positive rate")
    axes[0, 1].legend()
    ConfusionMatrixDisplay(confusion_matrix(y_test, test_predictions), display_labels=["Stayed", "Churned"]).plot(ax=axes[1, 0], cmap="Blues", colorbar=False)
    axes[1, 0].set_title(f"{best_name} at threshold {threshold_result.threshold:.2f}")
    importance = _feature_importance(best_model).head(10).sort_values()
    importance.index = importance.index.str.replace("categorical__", "", regex=False).str.replace("numeric__", "", regex=False)
    importance.plot.barh(ax=axes[1, 1], color="#e6862a")
    axes[1, 1].set(title="Strongest model signals", xlabel="Coefficient / feature importance")
    fig.suptitle("Telecom Customer Churn — Model and Business View", fontsize=18)
    fig.tight_layout()
    fig.savefig(FIGURES / "churn_dashboard.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIGURES / "churn_dashboard.svg", bbox_inches="tight")
    plt.close(fig)

    summary = f"""# Business summary

This project analyses {len(data):,} customers from IBM's Telco Customer Churn sample.

- Overall churn rate: {target.mean():.1%}
- Selected model: {best_name}
- Holdout ROC AUC: {test_metrics['roc_auc']:.3f}
- Holdout recall at the cost-aware threshold: {test_metrics['recall']:.1%}
- Holdout precision at the cost-aware threshold: {test_metrics['precision']:.1%}
- Decision threshold: {threshold_result.threshold:.2f}

The threshold is selected on validation data under an explicit assumption that missing a churner costs five times as much as contacting a customer who would have stayed. This is a demonstration assumption, not a measured commercial cost.
"""
    (REPORTS / "business_summary.md").write_text(summary, encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
