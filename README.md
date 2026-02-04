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
│   └── 01_eda.ipynb      # Exploratory data analysis
│
├── src/
│   ├── data/             # Data loading and preprocessing
│   ├── features/         # Feature engineering logic
│   ├── models/           # Model training
│   └── evaluation/       # Model evaluation and metrics
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

## 🚀 Status

🔧 **Work in progress**

Current focus:
- Dataset exploration
- Understanding churn behavior through EDA

Future updates will include feature engineering, modeling, and interpretation.

---

## 📌 Notes

- This project uses synthetic data for demonstration and learning purposes.
- The emphasis is on **methodology, reasoning, and interpretability**, rather than maximizing a single performance metric.
