"""Load and prepare the SaaS churn CSV used across notebooks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "data" / "raw" / "customer_subscription_churn_usage_patterns.csv"

NUMERIC_FEATURES = [
    "monthly_fee",
    "avg_weekly_usage_hours",
    "support_tickets",
    "payment_failures",
    "tenure_months",
    "last_login_days_ago",
]
CATEGORICAL_FEATURES = ["plan_type"]
TARGET = "churn"


def load_raw_csv(csv_path: Path | None = None) -> pd.DataFrame:
    path = csv_path or DEFAULT_CSV
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def prepare_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return feature matrix X and binary target y (1=churn, 0=retained)."""
    work = df.copy()
    y = (
        work[TARGET]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
    )
    if y.isna().any():
        bad = work.loc[y.isna(), TARGET].unique().tolist()
        raise ValueError(f"Unexpected churn labels: {bad}")

    missing_cols = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c not in work.columns]
    if missing_cols:
        raise KeyError(f"Missing expected columns: {missing_cols}")

    X = work[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    return X, y.astype(int)
