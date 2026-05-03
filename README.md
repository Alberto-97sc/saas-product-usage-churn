# SaaS Product Usage Churn Prediction

## 📌 Project Overview

Customer churn is a critical problem for subscription-based digital products such as SaaS platforms.  
This project focuses on **predicting customer churn using product usage and behavioral data**, with the goal of identifying at-risk users and supporting data-driven retention decisions.

Rather than treating churn as a purely technical classification task, this project emphasizes **customer behavior, engagement patterns, and actionable insights** derived from model interpretation.

---

## 🎯 Objective

The main objective of this project is to:

> **Predict customer churn in a SaaS product using usage, engagement, and friction signals in order to identify customers at risk and understand the key drivers behind churn.**

---

## 📊 Dataset

The dataset used in this project contains **synthetic subscription-based customer data** designed to reflect realistic behavior in digital products such as SaaS platforms, OTT services, or online learning systems.

- **Records:** 2,800 customers  
- **Data type:** Synthetic and anonymized  
- **Format:** CSV  
- **Target variable:** `churn` (Yes / No)

Although the data is synthetic, it captures realistic patterns of customer behavior and is suitable for modeling churn drivers, product engagement, and retention strategies.

### Key Features

- **Subscription & pricing**
  - `plan_type`
  - `monthly_fee`

- **Product usage & engagement**
  - `avg_weekly_usage`
  - `last_login_days`

- **Customer friction signals**
  - `support_tickets`
  - `payment_failures`

- **Customer relationship**
  - `tenure_months`
  - `signup_date`

---

## 🧠 Project Approach

This project follows an end-to-end data science workflow:

1. **Exploratory Data Analysis (EDA)**  
   - Understand customer behavior and churn distribution  
   - Analyze engagement, usage, and friction patterns  

2. **Feature Engineering**  
   - Transform raw usage and temporal signals into meaningful features  

3. **Modeling**  
   - Train baseline and machine learning models for churn prediction  
   - Address class imbalance and model evaluation trade-offs  

4. **Evaluation & Interpretation**  
   - Evaluate performance using appropriate classification metrics  
   - Interpret model outputs to identify key churn drivers  

5. **Business Insights**  
   - Translate model results into actionable retention strategies  

---

## 📂 Repository Structure


```
saas-product-usage-churn/
│
├── data/
│   ├── raw/              # Original dataset
│   └── processed/        # Cleaned and feature-engineered data
│
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory data analysis
│   └── 02_modeling.ipynb     # Baseline models (train/test, metrics)
│
├── src/
│   └── churn_data.py         # Shared CSV load + feature/target prep
│
├── requirements.txt
└── README.md
```



## 🛠 Tools & Technologies

- **Programming:** Python  
- **Data Analysis:** pandas, NumPy  
- **Machine Learning:** scikit-learn  
- **Visualization:** Matplotlib, Seaborn  
- **Model Interpretation:** SHAP (planned)  

---

## 🛠 Environment setup

```bash
cd saas-product-usage-churn
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run notebooks from the repo root (or open in Jupyter / VS Code with the `.venv` kernel). `02_modeling.ipynb` adds the repo root to `sys.path` so `import src.churn_data` works whether the kernel cwd is the project root or `notebooks/`.

## 🚀 Status

🔧 **Work in progress**

Current focus:
- EDA (`notebooks/01_eda.ipynb`)
- Baseline modeling (`notebooks/02_modeling.ipynb`) — stratified split, logistic regression & random forest, ROC-AUC / PR-AUC

Next: stronger tabular models, threshold tuning, and interpretation (e.g. SHAP) as needed.

---

## 📌 Notes

- This project uses synthetic data for demonstration and learning purposes.
- The emphasis is on **methodology, reasoning, and interpretability**, rather than maximizing a single performance metric.
