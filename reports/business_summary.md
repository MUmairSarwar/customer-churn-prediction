# Business summary

This project analyses 7,043 customers from IBM's Telco Customer Churn sample.

- Overall churn rate: 26.5%
- Selected model: Logistic regression
- Holdout ROC AUC: 0.843
- Holdout recall at the cost-aware threshold: 93.3%
- Holdout precision at the cost-aware threshold: 43.5%
- Decision threshold: 0.30

The threshold is selected on validation data under an explicit assumption that missing a churner costs five times as much as contacting a customer who would have stayed. This is a demonstration assumption, not a measured commercial cost.
