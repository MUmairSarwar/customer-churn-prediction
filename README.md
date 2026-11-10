# Telecom Customer Churn Prediction

[![tests](https://github.com/MUmairSarwar/customer-churn-prediction/actions/workflows/tests.yml/badge.svg)](https://github.com/MUmairSarwar/customer-churn-prediction/actions/workflows/tests.yml)

An end-to-end machine-learning project for a practical retention question: **which customers should a telecom company contact before they leave?**

The repository uses IBM's public Telco Customer Churn sample, compares two interpretable classification baselines and selects a decision threshold using an explicit business-cost assumption. It is designed as a reproducible portfolio project rather than a one-off notebook.

![Project dashboard](reports/figures/churn_dashboard.svg)


## Data period and version history

- **December 2025:** a synthetic-data churn notebook was recorded in this repository.
- **August 2026:** the project was substantially redeveloped around IBM's public Telco Customer Churn sample, with separate train/validation/holdout sets, model comparisons, a cost-based threshold, reports and automated tests.
- **Dataset availability:** IBM's source repository records the dataset by September 2019. Its publication date does not establish the dates of the underlying customer observations.

The results below are from the August 2026 version on the IBM sample, not the earlier synthetic-data notebook.

## What the project does

- Cleans 7,043 customer records and handles blank `TotalCharges` values
- Separates identifiers from model features
- Uses train, validation and holdout test sets to avoid tuning on final results
- Compares class-weighted logistic regression and random forest models
- Reports ROC AUC, PR AUC, Brier score, precision, recall and F1
- Selects a retention threshold where a missed churner is assumed to cost five times as much as an unnecessary contact
- Produces a business summary, a model comparison and a one-page dashboard
- Runs automated tests in GitHub Actions

## Results

The selected logistic-regression model achieved **0.843 ROC AUC** and **0.633 PR AUC** on the untouched holdout set. At the validation-selected threshold of **0.30**, it identified **93.3% of churners**; precision was **43.5%**, showing the operational trade-off created by prioritising missed churners.

Run `python -m src.pipeline` to reproduce the committed results. Exact metrics are stored in [`reports/model_metrics.json`](reports/model_metrics.json). The 5:1 cost ratio is my own assumption, not a figure from a real company.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python -m src.pipeline
```

If the data file is missing, restore it from the official source with `python -m src.download_data`.

## Data source and limitations

The data comes from IBM's public [`telco-customer-churn-on-icp4d`](https://github.com/IBM/telco-customer-churn-on-icp4d/tree/master/data) sample. It is useful for demonstrating a workflow, but it is not current production data. Observed relationships are associations, not causal effects. Deployment would require current company data, monitoring, fairness review, cost validation and a controlled retention experiment.

## Other selected projects

- [Rhine-Main Heat Risk Forecasting](https://github.com/MUmairSarwar/germany-heat-risk-forecasting)
- [Retail Customer & Operations Analytics](https://github.com/MUmairSarwar/retail-customer-analytics)
- [Robust Federated Learning](https://github.com/MUmairSarwar/robust-federated-learning-ml-security)
- [Strategic Classification](https://github.com/MUmairSarwar/strategic-classification)

## Author

Muhammad Umair Sarwar - M.Sc. Mathematics student (Mathematics in Data Science), TU Darmstadt.

Code in this repository is available under the MIT License.
