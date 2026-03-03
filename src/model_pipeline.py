"""
model_pipeline.py
─────────────────
Construcción de pipelines de modelado, evaluación y comparativa de modelos.
Soporta RandomForest, XGBoost y LogisticRegression con SMOTE para desbalance.

Uso:
    from src.model_pipeline import ModelPipeline
    mp = ModelPipeline(df_enriched, preprocessor)
    results = mp.run_all_models()
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from loguru import logger
import warnings

warnings.filterwarnings("ignore")


# ─── Constants ────────────────────────────────────────────────────────────────

TARGET_COL = "Churn"
RANDOM_STATE = 42
TEST_SIZE = 0.2


# ─── Model Registry ───────────────────────────────────────────────────────────

def _get_model(model_name: str):
    """Return the unfitted estimator for the given model name."""
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=3,   # Handles class imbalance natively in XGBoost
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            verbosity=0,
        ),
    }
    if model_name not in models:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(models.keys())}")
    return models[model_name]


# ─── Model Pipeline Builder ───────────────────────────────────────────────────

def build_model_pipeline(
    preprocessor,
    model_name: str = "xgboost",
    use_smote: bool = True,
) -> ImbPipeline:
    """
    Build a complete imbalanced-learn Pipeline with preprocessing + SMOTE + model.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Fitted or unfitted sklearn ColumnTransformer.
    model_name : str
        One of: 'logistic_regression', 'random_forest', 'xgboost'.
    use_smote : bool
        Whether to apply SMOTE oversampling to the training set.

    Returns
    -------
    ImbPipeline
        Complete pipeline ready to fit.
    """
    estimator = _get_model(model_name)
    steps = [("preprocessor", preprocessor)]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)))
    steps.append(("classifier", estimator))
    return ImbPipeline(steps=steps)


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_model(
    pipeline: ImbPipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> dict:
    """
    Evaluate a fitted pipeline and return a structured results dict.

    Parameters
    ----------
    pipeline : ImbPipeline
        Fitted pipeline.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        True labels.
    threshold : float
        Decision threshold (default 0.5). Lower = more sensitive to churn.

    Returns
    -------
    dict
        Dictionary with all metrics and curve data for visualization.
    """
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # Core classification metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    # Curve data for plotting
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
    prec_curve, rec_curve, pr_thresholds = precision_recall_curve(y_test, y_prob)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": auc_roc,
        "confusion_matrix": cm,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "roc_curve": {"fpr": fpr, "tpr": tpr, "thresholds": roc_thresholds},
        "pr_curve": {
            "precision": prec_curve,
            "recall": rec_curve,
            "thresholds": pr_thresholds,
        },
    }


# ─── Main Pipeline Orchestrator ───────────────────────────────────────────────

class ModelPipeline:
    """
    High-level orchestrator for training, comparing, and selecting models.

    Parameters
    ----------
    df : pd.DataFrame
        Enriched DataFrame with business features.
    preprocessor : ColumnTransformer
        Unfitted sklearn ColumnTransformer from feature_engineering.build_preprocessor().
    threshold : float
        Decision threshold applied at prediction time.
    """

    def __init__(self, df: pd.DataFrame, preprocessor, threshold: float = 0.4):
        self.df = df
        self.preprocessor = preprocessor
        self.threshold = threshold
        self.X_train: pd.DataFrame | None = None
        self.X_test: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None
        self.y_test: pd.Series | None = None
        self.fitted_pipelines: dict = {}
        self.results: dict = {}
        self._split_data()

    def _split_data(self):
        X = self.df.drop(columns=[TARGET_COL])
        y = self.df[TARGET_COL]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        logger.info(
            f"Train/test split: {len(self.X_train):,} / {len(self.X_test):,} | "
            f"Churn rate train: {self.y_train.mean():.1%} | test: {self.y_test.mean():.1%}"
        )

    def train_model(self, model_name: str, use_smote: bool = True) -> dict:
        """Train a single model and return its evaluation metrics."""
        logger.info(f"Training {model_name} (SMOTE={use_smote})...")
        pipeline = build_model_pipeline(self.preprocessor, model_name, use_smote)
        pipeline.fit(self.X_train, self.y_train)
        self.fitted_pipelines[model_name] = pipeline

        metrics = evaluate_model(pipeline, self.X_test, self.y_test, self.threshold)
        self.results[model_name] = metrics
        logger.info(
            f"{model_name} | Precision: {metrics['precision']:.3f} | "
            f"Recall: {metrics['recall']:.3f} | F1: {metrics['f1']:.3f} | "
            f"AUC-ROC: {metrics['auc_roc']:.3f}"
        )
        return metrics

    def run_all_models(self) -> pd.DataFrame:
        """Train all models and return a comparison DataFrame."""
        for model_name in ["logistic_regression", "random_forest", "xgboost"]:
            self.train_model(model_name)

        comparison = pd.DataFrame([
            {
                "Model": name,
                "Precision": r["precision"],
                "Recall": r["recall"],
                "F1-Score": r["f1"],
                "AUC-ROC": r["auc_roc"],
            }
            for name, r in self.results.items()
        ]).set_index("Model").sort_values("AUC-ROC", ascending=False)

        logger.info(f"\nModel Comparison:\n{comparison.to_string()}")
        return comparison

    def get_best_pipeline(self, metric: str = "auc_roc") -> tuple[str, ImbPipeline]:
        """Return the name and fitted pipeline of the best-performing model."""
        if not self.results:
            raise RuntimeError("No models trained yet. Call run_all_models() first.")
        best_name = max(self.results, key=lambda k: self.results[k][metric])
        return best_name, self.fitted_pipelines[best_name]

    def cross_validate_best(self, cv: int = 5) -> dict:
        """Run stratified K-Fold cross-validation on the best model."""
        best_name, best_pipeline = self.get_best_pipeline()
        X = self.df.drop(columns=[TARGET_COL])
        y = self.df[TARGET_COL]
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
        auc_scores = cross_val_score(best_pipeline, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
        logger.info(
            f"CV AUC-ROC for {best_name}: "
            f"{auc_scores.mean():.3f} ± {auc_scores.std():.3f}"
        )
        return {"model": best_name, "cv_auc_mean": auc_scores.mean(), "cv_auc_std": auc_scores.std()}
