"""
train.py
────────
Script CLI de re-entrenamiento end-to-end con MLflow tracking.
Diseñado para ser ejecutado tanto manualmente como en un pipeline CI/CD.

Uso:
    python src/train.py
    python src/train.py --model xgboost --threshold 0.4 --no-smote
    python src/train.py --model random_forest --experiment MyExperiment

El modelo entrenado se guarda en:  models/best_model.joblib
Los experimentos se registran en:  mlruns/ (visible con `mlflow ui`)
"""

import sys
import os

# Ensure project root is in Python path when running from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import click
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from pathlib import Path
from datetime import datetime
from loguru import logger

from src.data_pipeline import DataPipeline
from src.feature_engineering import add_business_features, build_preprocessor
from src.model_pipeline import ModelPipeline, evaluate_model
from src.business_metrics import BusinessImpactCalculator, BusinessConfig


# ─── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ─── CLI ─────────────────────────────────────────────────────────────────────

@click.command()
@click.option(
    "--model",
    default="xgboost",
    type=click.Choice(["logistic_regression", "random_forest", "xgboost"]),
    help="Model algorithm to train.",
    show_default=True,
)
@click.option(
    "--threshold",
    default=0.4,
    type=float,
    help="Decision threshold for churn prediction (lower = higher recall).",
    show_default=True,
)
@click.option(
    "--no-smote",
    is_flag=True,
    default=False,
    help="Disable SMOTE oversampling (use only model-level class weighting).",
)
@click.option(
    "--experiment",
    default="ChurnPrediction",
    help="MLflow experiment name.",
    show_default=True,
)
@click.option(
    "--ltv",
    default=2400.0,
    type=float,
    help="Average customer Lifetime Value in EUR for business impact calculation.",
    show_default=True,
)
def train(model: str, threshold: float, no_smote: bool, experiment: str, ltv: float):
    """
    \b
    ==================================================
       Churn Prediction — Model Training Pipeline    
    ==================================================

    Runs the full ML pipeline:
        1. Load & clean data
        2. Add business features
        3. Train model with SMOTE (optional)
        4. Evaluate with recall-focused metrics
        5. Log everything to MLflow
        6. Save best model to models/best_model.joblib
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    use_smote = not no_smote

    logger.info("=" * 55)
    logger.info("   CHURN PREDICTION — TRAINING PIPELINE")
    logger.info("=" * 55)
    logger.info(f"Model:     {model}")
    logger.info(f"Threshold: {threshold}")
    logger.info(f"SMOTE:     {use_smote}")
    logger.info(f"Experiment:{experiment}")

    # ── 1. Data Loading & Cleaning ────────────────────────────────────────────
    logger.info("\n[1/5] Loading and cleaning data...")
    pipeline_data = DataPipeline(DATA_PATH)
    df_clean = pipeline_data.run()
    data_summary = pipeline_data.summary()

    # ── 2. Feature Engineering ────────────────────────────────────────────────
    logger.info("\n[2/5] Engineering business features...")
    df_enriched = add_business_features(df_clean)
    preprocessor = build_preprocessor(df_enriched)

    # ── 3. Model Training with MLflow Tracking ────────────────────────────────
    logger.info(f"\n[3/5] Training {model} and logging to MLflow...")
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=f"{model}_{run_id}"):

        # Log parameters
        mlflow.log_params({
            "model": model,
            "threshold": threshold,
            "use_smote": use_smote,
            "n_train_samples": data_summary["n_rows"],
            "churn_rate": round(data_summary["churn_rate"], 4),
        })

        # Train
        mp = ModelPipeline(df_enriched, preprocessor, threshold=threshold)
        metrics = mp.train_model(model_name=model, use_smote=use_smote)

        # ── 4. Evaluate & Log Metrics ─────────────────────────────────────────
        logger.info("\n[4/5] Evaluating and logging metrics...")
        mlflow.log_metrics({
            "precision": round(float(metrics["precision"]), 4),
            "recall":    round(float(metrics["recall"]), 4),
            "f1_score":  round(float(metrics["f1"]), 4),
            "auc_roc":   round(float(metrics["auc_roc"]), 4),
        })

        # Business impact
        biz_config = BusinessConfig(avg_customer_ltv=ltv)
        calc = BusinessImpactCalculator(config=biz_config)
        biz_report = calc.compute(
            y_true=mp.y_test,
            y_pred=metrics["y_pred"],
            y_prob=metrics["y_prob"],
        )
        mlflow.log_metrics({
            "net_benefit_model_eur":      round(biz_report["net_benefit_model"], 2),
            "incremental_net_benefit_eur": round(biz_report["incremental_net_benefit"], 2),
            "model_roi_pct":              round(biz_report["model_roi_pct"], 2),
            "churners_identified":        biz_report["churners_identified"],
            "false_alarms":               biz_report["false_alarms"],
        })

        # Print executive summary
        print("\n" + calc.format_summary(biz_report))

        # ── 5. Save Model ─────────────────────────────────────────────────────
        logger.info("\n[5/5] Saving model...")
        fitted_pipeline = mp.fitted_pipelines[model]
        output_path = MODELS_DIR / "best_model.joblib"
        joblib.dump(fitted_pipeline, output_path)

        # Log model artifact to MLflow
        mlflow.sklearn.log_model(
            fitted_pipeline,
            artifact_path="model",
            registered_model_name=f"churn_{model}",
        )
        mlflow.log_artifact(str(output_path), artifact_path="joblib")

        run_url = mlflow.get_tracking_uri()
        logger.info(f"✓ Model saved to: {output_path}")
        logger.info(f"✓ MLflow run logged. View with: mlflow ui")
        logger.info(f"  Tracking URI: {run_url}")
        logger.info("=" * 55)
        logger.success("Training pipeline completed successfully!")


if __name__ == "__main__":
    train()
