# Customer Churn Prediction with Business Impact

A professional, end-to-end Machine Learning project designed to identify telecommunication customers at risk of churning and translate model performance into tangible business impact (ROI).

![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.7+-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2+-red.svg)
![MLflow](https://img.shields.io/badge/MLflow-Experiment_Tracking-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)

##  Project Overview

This project goes beyond standard accuracy metrics (like AUC or F1-score) by explicitly answering the business question: **"How much money can this model save the company?"**

 **[Access the Live Interactive Dashboard Here](https://churn-prediction-ds.streamlit.app/)**

### Key Features
1. **Robust ML Pipeline**: Implements XGBoost, RandomForest, and Logistic Regression, handling class imbalance natively and via `SMOTE`.
2. **Business Feature Engineering**: Translates raw Telco data into actionable risk features (e.g., `ContractRisk`, `PaymentRisk`, `MonthlyChargesPerService`).
3. **MLOps & CI/CD**: Uses `MLflow` for experiment tracking and features a fully automated Continuous Training pipeline via **GitHub Actions** that updates the active model only if ROI improves.
4. **Interactive Dashboard**: A 5-page Streamlit application allowing non-technical stakeholders to explore data, visualize model results, make live predictions, and calculate campaign ROI.

## Architecture & Structure

```text
churn_prediction/
├── app/
│   └── app.py                  # Streamlit dashboard
├── data/
│   └── raw/                    # (Ignored in Git) Store the dataset here
├── mlruns/                     # MLflow experiment tracking logs
├── models/
│   └── best_model.joblib       # Serialized Scikit-Learn/XGBoost pipeline
├── notebooks/                  # EDA & prototyping (Jupyter)
├── src/
│   ├── business_metrics.py     # ROI & LTV calculations
│   ├── data_pipeline.py        # Data ingestion & cleaning
│   ├── feature_engineering.py  # Sklearn ColumnTransformer & business features
│   ├── model_pipeline.py       # Imblearn pipelines & evaluation metrics
│   └── train.py                # CLI script for end-to-end training
├── .venv/                      # Python virtual environment (ignored)
├── requirements.txt            # Project dependencies
└── README.md
```

## Quickstart

### 1. Local Setup
If you want to run this project locally to train new models or explore the code:

```bash
git clone https://github.com/zak-lsk/churn-prediction.git
cd churn-prediction
python -m venv .venv
```
Activate the environment (`.venv\Scripts\activate` on Windows) and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Get the Data
Download the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle.
Extract the CSV and place it exactly at: `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

### 3. ML Pipeline & Training
Run the CLI script to clean data, engineer features, train the model, and log the experiment to MLflow.
```bash
python src/train.py --model xgboost
```
You can view the experiment logs by running `mlflow ui`.

## Results & Business Impact

The baseline XGBoost model achieves strong recall (identifying ~86% of potential churners).
When translated into a targeted retention campaign (assuming a LTV of €2,400, retention cost of €150, and a 25% success rate), the model generates significant incremental net benefit compared to an untargeted (baseline) campaign.

*See the "Business Impact" page in the Live Streamlit app for dynamic calculations.*

---

## Future Work
- Implement hyperparameter tuning (GridSearchCV/Optuna).
- Containerize the application using Docker for scalable deployment.
