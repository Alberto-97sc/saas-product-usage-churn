"""Smoke tests for baseline training (fast, deterministic)."""

from __future__ import annotations

from src.baseline import train_eval_split
from src.churn_data import load_raw_csv, prepare_features_and_target


def test_train_eval_split_metrics_in_valid_range():
    X, y = prepare_features_and_target(load_raw_csv())
    metrics = train_eval_split(X, y)
    assert set(metrics.keys()) == {"logistic_regression", "random_forest"}
    for name, m in metrics.items():
        assert 0.5 < m["roc_auc"] <= 1.0, name
        assert 0.0 < m["pr_auc"] <= 1.0, name
        assert 0.4 < m["test_churn_rate"] < 0.7
