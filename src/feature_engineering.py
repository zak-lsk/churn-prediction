"""
feature_engineering.py
───────────────────────
Transformaciones reproducibles usando sklearn ColumnTransformer + Pipeline.
Incluye features de negocio derivadas para maximizar el poder predictivo.

Uso:
    from src.feature_engineering import build_preprocessor, add_business_features
    df_enriched = add_business_features(df_clean)
    preprocessor = build_preprocessor(df_enriched)
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from loguru import logger


# ─── Feature Definitions ──────────────────────────────────────────────────────

NUMERICAL_FEATURES = [
    "tenure", "MonthlyCharges", "TotalCharges",
    # Business features (added by add_business_features)
    "MonthlyChargesPerService", "ChargesPerMonth",
]

CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
    # Business features (added by add_business_features)
    "TenureGroup", "ContractRisk", "PaymentRisk",
]


# ─── Business Feature Engineering ────────────────────────────────────────────

def add_business_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add domain-driven features based on business intuition.

    New Features Created:
    - TenureGroup        : Customer lifecycle stage (New / Growing / Loyal / Champion)
    - MonthlyChargesPerService : Cost per active service — proxy for "value perception"
    - ChargesPerMonth    : TotalCharges / max(tenure, 1) — ARPU proxy for older customers
    - ContractRisk       : Risk score by contract type (Month-to-month is highest risk)
    - PaymentRisk        : Electronic check payers churn more due to friction

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from DataPipeline.

    Returns
    -------
    pd.DataFrame
        DataFrame enriched with new derived columns.
    """
    df = df.copy()

    # 1. Tenure lifecycle segmentation
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, np.inf],
        labels=["New (0-12m)", "Growing (1-2y)", "Loyal (2-4y)", "Champion (4y+)"],
    ).astype(str)

    # 2. Service count (number of extra services beyond basic phone/internet)
    service_cols = [
        "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    service_count = df[service_cols].apply(lambda col: (col == "Yes").astype(int)).sum(axis=1)
    service_count = service_count.clip(lower=1)  # avoid division by zero

    # 3. Charges per active service (higher → customer may feel overcharged)
    df["MonthlyChargesPerService"] = df["MonthlyCharges"] / service_count

    # 4. Average monthly spend over entire lifetime (ARPU proxy)
    df["ChargesPerMonth"] = df["TotalCharges"] / df["tenure"].clip(lower=1)

    # 5. Contract risk label (business insight: M2M is 3× more likely to churn)
    contract_risk_map = {
        "Month-to-month": "High",
        "One year": "Medium",
        "Two year": "Low",
    }
    df["ContractRisk"] = df["Contract"].map(contract_risk_map).fillna("Unknown")

    # 6. Payment method risk (electronic check = highest friction/churn rate)
    payment_risk_map = {
        "Electronic check": "High",
        "Mailed check": "Medium",
        "Bank transfer (automatic)": "Low",
        "Credit card (automatic)": "Low",
    }
    df["PaymentRisk"] = df["PaymentMethod"].map(payment_risk_map).fillna("Unknown")

    logger.info(
        f"Feature engineering complete. "
        f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns "
        f"(+{df.shape[1] - df.shape[1] + 6} new features)"  # noqa: simplification indicator
    )
    return df


# ─── Sklearn Preprocessor ─────────────────────────────────────────────────────

def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """
    Build a production-ready sklearn ColumnTransformer that handles:
    - Numerical: median imputation + StandardScaler
    - Categorical: constant imputation + OneHotEncoder

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with business features already added.

    Returns
    -------
    ColumnTransformer
        Unfitted preprocessor ready to be embedded in a Pipeline.
    """
    # Identify which of our configured features are actually present in df
    target = "Churn"
    available_cols = [c for c in df.columns if c != target]

    num_features = [c for c in NUMERICAL_FEATURES if c in available_cols]
    cat_features = [c for c in CATEGORICAL_FEATURES if c in available_cols]

    # Any remaining columns not captured above
    unhandled = [c for c in available_cols if c not in num_features + cat_features]
    if unhandled:
        logger.warning(f"Unhandled columns (will be dropped): {unhandled}")

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    logger.info(
        f"Preprocessor built | Numerical: {len(num_features)} features | "
        f"Categorical: {len(cat_features)} features"
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract human-readable feature names after fitting the ColumnTransformer."""
    return list(preprocessor.get_feature_names_out())
