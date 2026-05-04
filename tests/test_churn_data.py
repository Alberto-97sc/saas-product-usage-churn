"""Tests for CSV loading and feature/target preparation."""

from __future__ import annotations

from src.churn_data import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    load_raw_csv,
    prepare_features_and_target,
)


def test_load_raw_csv_shape_and_columns():
    df = load_raw_csv()
    assert len(df) == 2800
    assert df.shape[1] == 10
    assert all(c == c.lower() for c in df.columns)
    assert "churn" in df.columns


def test_prepare_features_and_target():
    df = load_raw_csv()
    X, y = prepare_features_and_target(df)
    assert X.shape == (2800, len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES))
    assert set(X.columns) == set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    assert y.isin([0, 1]).all()
    assert not y.isna().any()
