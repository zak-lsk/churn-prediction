"""
data_pipeline.py
────────────────
Ingesta, limpieza y validación del dataset Telco Customer Churn.

Uso:
    from src.data_pipeline import DataPipeline
    dp = DataPipeline("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = dp.run()
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger


# ─── Constants ───────────────────────────────────────────────────────────────

TARGET_COL = "Churn"
CUSTOMER_ID_COL = "customerID"

BINARY_YES_NO_COLS = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

CATEGORICAL_COLS = [
    "gender", "InternetService", "Contract", "PaymentMethod",
    "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

NUMERICAL_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


# ─── Pipeline Class ───────────────────────────────────────────────────────────

class DataPipeline:
    """
    End-to-end data ingestion and cleaning pipeline for the Telco Churn dataset.

    Parameters
    ----------
    filepath : str | Path
        Path to the raw CSV file.

    Attributes
    ----------
    df_raw : pd.DataFrame
        Original DataFrame loaded from disk (unmodified).
    df : pd.DataFrame
        Cleaned DataFrame after calling .run() or .clean().
    """

    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        self.df_raw: pd.DataFrame | None = None
        self.df: pd.DataFrame | None = None

    # ── Public API ──────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """Execute the full pipeline: load → clean → validate."""
        self.load()
        self.clean()
        self.validate()
        return self.df

    def load(self) -> "DataPipeline":
        """Load raw CSV from disk."""
        if not self.filepath.exists():
            raise FileNotFoundError(
                f"Dataset not found at '{self.filepath}'.\n"
                "Please download it from Kaggle:\n"
                "  https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n"
                "and place the CSV at: data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
            )
        self.df_raw = pd.read_csv(self.filepath)
        self.df = self.df_raw.copy()
        logger.info(f"Loaded dataset: {self.df.shape[0]:,} rows x {self.df.shape[1]} columns")
        return self

    def clean(self) -> "DataPipeline":
        """Apply all cleaning steps sequentially."""
        self._fix_total_charges()
        self._encode_target()
        self._encode_senior_citizen()
        self._strip_service_columns()
        self._drop_customer_id()
        logger.info(f"Cleaning complete. Final shape: {self.df.shape}")
        return self

    def validate(self) -> "DataPipeline":
        """Run basic data quality checks and log warnings."""
        null_counts = self.df.isnull().sum()
        if null_counts.any():
            logger.warning(f"Null values detected after cleaning:\n{null_counts[null_counts > 0]}")
        else:
            logger.info("✓ No null values found after cleaning.")

        churn_rate = self.df[TARGET_COL].mean()
        logger.info(f"✓ Churn rate: {churn_rate:.1%}  ({self.df[TARGET_COL].sum()} positive cases)")

        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            logger.warning(f"{duplicates} duplicate rows detected.")
        else:
            logger.info("✓ No duplicate rows found.")

        return self

    # ── Private Cleaning Steps ───────────────────────────────────────────────

    def _fix_total_charges(self):
        """TotalCharges is stored as string with spaces for new customers → coerce to float."""
        self.df["TotalCharges"] = pd.to_numeric(self.df["TotalCharges"], errors="coerce")
        # New customers (tenure=0) have TotalCharges=0 logically
        nulls_before = self.df["TotalCharges"].isnull().sum()
        self.df.fillna({"TotalCharges": 0.0}, inplace=True)
        if nulls_before > 0:
            logger.info(f"Fixed {nulls_before} TotalCharges nulls → 0.0 (new customers with tenure=0)")

    def _encode_target(self):
        """Encode Churn: 'Yes'→1, 'No'→0."""
        self.df[TARGET_COL] = (self.df[TARGET_COL] == "Yes").astype(int)

    def _encode_senior_citizen(self):
        """SeniorCitizen is already 0/1 — map to 'No'/'Yes' for consistency."""
        self.df["SeniorCitizen"] = self.df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    def _strip_service_columns(self):
        """
        Columns like OnlineSecurity have values: 'Yes', 'No', 'No internet service'.
        Consolidate 'No internet service' and 'No phone service' → 'No'.
        """
        for col in BINARY_YES_NO_COLS:
            if col in self.df.columns:
                self.df[col] = self.df[col].replace(
                    {"No internet service": "No", "No phone service": "No"}
                )

    def _drop_customer_id(self):
        """Remove the customer ID as it carries no predictive signal."""
        if CUSTOMER_ID_COL in self.df.columns:
            self.df.drop(columns=[CUSTOMER_ID_COL], inplace=True)

    # ── Diagnostics ─────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Return a summary dict useful for logging and dashboards."""
        if self.df is None:
            raise RuntimeError("Pipeline not run yet. Call .run() first.")
        return {
            "n_rows": len(self.df),
            "n_features": self.df.shape[1] - 1,
            "churn_rate": float(self.df[TARGET_COL].mean()),
            "n_churned": int(self.df[TARGET_COL].sum()),
            "n_retained": int((self.df[TARGET_COL] == 0).sum()),
            "numeric_features": NUMERICAL_COLS,
            "categorical_features": CATEGORICAL_COLS,
        }
