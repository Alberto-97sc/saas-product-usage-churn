# SaaS churn — usage & behavior

End-to-end **tabular churn** analysis: **EDA → preprocessing → baselines → interpretation**, with a **reproducible** Python environment, **automated tests**, **CI**, and a **CLI training script** (metrics without opening Jupyter).

> **Dataset:** synthetic CSV (methodology demo, not production KPIs).

---

## Scope

Customer **churn** is a common subscription-product problem. This repository demonstrates:

1. Explore churn vs **usage**, **recency**, **friction** (support / payments), and **plan/fee** signals (`notebooks/01_eda.ipynb`).
2. Build **sklearn pipelines** (scaling + one-hot) and compare **logistic regression** vs **random forest** with **stratified** splits and **ROC-AUC / PR-AUC** under class imbalance (`notebooks/02_modeling.ipynb`).
3. Summarize outcomes in a **written report** (last section of the modeling notebook).

---

## Quick start

```bash
git clone <your-fork-url> saas-product-usage-churn
cd saas-product-usage-churn
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                          # unit + smoke tests
python scripts/train_baseline.py   # prints holdout metrics (no notebook)
```

Open `notebooks/01_eda.ipynb` and `notebooks/02_modeling.ipynb` with the `.venv` kernel. The modeling notebook prepends the repo root to `sys.path` so `import src.churn_data` works from `notebooks/` or from the project root.

---

## Results snapshot (held-out test, stratified 75/25)

| Model | ROC-AUC | PR-AUC |
| --- | ---: | ---: |
| Logistic regression | 0.7003 | 0.7258 |
| Random forest | **0.7056** | **0.7625** |

Test churn prevalence ≈ **57%** — PR-AUC should be read against that baseline. Details and interpretation: **`notebooks/02_modeling.ipynb`** (written report at the end).

---

## Repository layout

```
saas-product-usage-churn/
├── data/
│   ├── raw/                     # Source CSV
│   └── processed/               # Placeholder for future artifacts
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory analysis + cohort views
│   └── 02_modeling.ipynb        # Baselines + written summary
├── scripts/
│   └── train_baseline.py        # CLI: train + print metrics
├── src/
│   ├── churn_data.py            # Load CSV + build X, y
│   └── baseline.py             # Shared preprocessing + model definitions
├── tests/                       # pytest
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## Tech stack

Python **3.12**, **pandas**, **NumPy**, **scikit-learn**, **Matplotlib**, **Jupyter** / **nbconvert**, **pytest**, **GitHub Actions** (CI).

---

## Status

**Delivered for the current scope:** reproducible EDA, baseline modeling, written interpretation, tests, and CI. Possible extensions: gradient boosting, SHAP, probability calibration / threshold tuning, or a small batch inference service.

---

## Resumen (ES)

Análisis de **churn** en un contexto tipo **SaaS** usando datos **sintéticos** (enfoque metodológico). Incluye **EDA** exploratorio, **modelos baseline** (regresión logística y bosque aleatorio), evaluación con métricas adecuadas para **clase desbalanceada** (ROC-AUC / PR-AUC), un **informe interpretativo** en el notebook de modelado, un **script** reproducible (`scripts/train_baseline.py`), **pruebas automatizadas** y **integración continua** para validar el flujo en cada cambio.

---

## License

MIT — see [`LICENSE`](LICENSE).
