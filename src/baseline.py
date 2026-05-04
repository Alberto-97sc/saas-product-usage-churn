"""Baseline churn models: shared preprocessing + sklearn pipelines."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.churn_data import CATEGORICAL_FEATURES, NUMERIC_FEATURES

DEFAULT_RANDOM_STATE = 42


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
)


def make_pipelines() -> dict[str, Pipeline]:
    prep = make_preprocessor()
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("prep", prep),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=DEFAULT_RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("prep", make_preprocessor()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=12,
                        random_state=DEFAULT_RANDOM_STATE,
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


@dataclass(frozen=True)
class SplitConfig:
    test_size: float = 0.25
    random_state: int = DEFAULT_RANDOM_STATE


def train_eval_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    split: SplitConfig | None = None,
) -> dict[str, dict[str, float]]:
    """Fit baseline pipelines and return test ROC-AUC and PR-AUC per model."""
    split = split or SplitConfig()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split.test_size,
        random_state=split.random_state,
        stratify=y,
    )
    metrics: dict[str, dict[str, float]] = {}
    for name, pipe in make_pipelines().items():
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        metrics[name] = {
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "pr_auc": float(average_precision_score(y_test, proba)),
            "test_churn_rate": float(y_test.mean()),
        }
    return metrics
