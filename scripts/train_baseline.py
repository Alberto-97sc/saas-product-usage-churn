#!/usr/bin/env python3
"""Train baseline churn models and print test metrics (no Jupyter required)."""

from __future__ import annotations

import sys
from pathlib import Path

# Repo root on path when run as `python scripts/train_baseline.py`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baseline import train_eval_split
from src.churn_data import load_raw_csv, prepare_features_and_target


def main() -> None:
    X, y = prepare_features_and_target(load_raw_csv())
    metrics = train_eval_split(X, y)
    churn = next(iter(metrics.values()))["test_churn_rate"]
    print(f"Rows: {len(X)} | Test churn rate (holdout): {churn:.1%}\n")
    for name, m in metrics.items():
        print(f"{name:22s}  ROC-AUC: {m['roc_auc']:.4f}   PR-AUC: {m['pr_auc']:.4f}")


if __name__ == "__main__":
    main()
