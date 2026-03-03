"""
business_metrics.py
───────────────────
Traduce métricas técnicas del modelo en impacto económico tangible para el negocio.

La lógica central: el valor del modelo no está en el AUC-ROC, sino en cuánto dinero
puede ahorrar la empresa identificando clientes en riesgo antes de que se vayan.

Uso:
    from src.business_metrics import BusinessImpactCalculator
    calc = BusinessImpactCalculator(avg_customer_ltv=2400, retention_cost=150)
    report = calc.compute(pipeline, X_test, y_test)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class BusinessConfig:
    """
    Business parameters to configure the ROI calculation.

    Parameters
    ----------
    avg_customer_ltv : float
        Average lifetime value (LTV) of a customer in euros/USD.
        Telco industry benchmark: ~2,000–3,000€ over 2.5 years.
    retention_cost : float
        Cost of a targeted retention action per customer (discount, call, offer).
        Telco benchmark: ~100–200€.
    acquisition_cost : float
        Cost to acquire a NEW customer to replace a churned one.
        Telco benchmark: 5–10× higher than retention cost.
    retention_success_rate : float
        Probability that a retention action succeeds (converts a churner to stayer).
        Industry benchmark: 20–30%.
    """
    avg_customer_ltv: float = 2400.0
    retention_cost: float = 150.0
    acquisition_cost: float = 900.0
    retention_success_rate: float = 0.25


# ─── Calculator ───────────────────────────────────────────────────────────────

class BusinessImpactCalculator:
    """
    Compute the real-world business impact of the churn model vs. the baseline
    (no model — random or blanket campaigns).

    Parameters
    ----------
    config : BusinessConfig
        Business parameters (LTV, costs, success rates).
    """

    def __init__(self, config: BusinessConfig | None = None):
        self.config = config or BusinessConfig()

    def compute(
        self,
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
    ) -> dict:
        """
        Compute the full business impact report.

        Parameters
        ----------
        y_true : array-like
            Ground truth labels (0 = retained, 1 = churned).
        y_pred : array-like
            Binary predictions from the model.
        y_prob : array-like
            Churn probability scores (for segmentation).

        Returns
        -------
        dict
            Structured report with savings, ROI, and segment breakdown.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)

        n_test = len(y_true)
        cfg = self.config

        # ── Confusion matrix components
        tp = int(((y_pred == 1) & (y_true == 1)).sum())  # Correctly identified churners
        fp = int(((y_pred == 1) & (y_true == 0)).sum())  # False alarms (wasted effort)
        fn = int(((y_pred == 0) & (y_true == 1)).sum())  # Missed churners
        tn = int(((y_pred == 0) & (y_true == 0)).sum())  # Correctly identified loyals

        total_actual_churners = tp + fn

        # ── MODEL SCENARIO: Targeted retention campaign
        # We send offers only to predicted churners (TP + FP)
        targeted_population = tp + fp
        successfully_retained = int(tp * cfg.retention_success_rate)
        retention_spend = targeted_population * cfg.retention_cost
        revenue_saved_model = successfully_retained * cfg.avg_customer_ltv

        # ── BASELINE SCENARIO: Random campaign (no model, spray-and-pray)
        # Company sends offers to ALL customers hoping to catch churners
        random_success_rate = total_actual_churners / n_test  # fraction of real churners
        baseline_retained = int(n_test * random_success_rate * cfg.retention_success_rate)
        baseline_spend = n_test * cfg.retention_cost
        revenue_saved_baseline = baseline_retained * cfg.avg_customer_ltv

        # ── INCREMENTAL VALUE of the model
        incremental_savings = (revenue_saved_model - retention_spend) - (revenue_saved_baseline - baseline_spend)
        model_roi = (
            (revenue_saved_model - retention_spend) / retention_spend * 100
            if retention_spend > 0 else 0.0
        )

        # ── RISK SEGMENTATION: High / Medium / Low
        risk_segments = self._segment_by_risk(y_prob, y_true)

        return {
            # Summary KPIs
            "total_customers_scored": n_test,
            "actual_churners": total_actual_churners,
            "churners_identified": tp,
            "false_alarms": fp,
            "missed_churners": fn,
            # Model scenario
            "targeted_population": targeted_population,
            "successfully_retained_model": successfully_retained,
            "retention_spend_model": retention_spend,
            "revenue_saved_model": revenue_saved_model,
            "net_benefit_model": revenue_saved_model - retention_spend,
            "model_roi_pct": model_roi,
            # Baseline scenario
            "baseline_spend": baseline_spend,
            "revenue_saved_baseline": revenue_saved_baseline,
            "net_benefit_baseline": revenue_saved_baseline - baseline_spend,
            # Incremental impact
            "incremental_net_benefit": incremental_savings,
            # Risk segments
            "risk_segments": risk_segments,
            # Config used
            "config": {
                "avg_customer_ltv": cfg.avg_customer_ltv,
                "retention_cost": cfg.retention_cost,
                "retention_success_rate": cfg.retention_success_rate,
                "acquisition_cost": cfg.acquisition_cost,
            },
        }

    def _segment_by_risk(
        self, y_prob: np.ndarray, y_true: np.ndarray
    ) -> pd.DataFrame:
        """Bucket customers into High / Medium / Low risk for action prioritization."""
        df = pd.DataFrame({"prob": y_prob, "actual": y_true})
        df["risk_segment"] = pd.cut(
            df["prob"],
            bins=[-0.001, 0.33, 0.66, 1.001],
            labels=["Low Risk", "Medium Risk", "High Risk"],
        )
        segment_summary = (
            df.groupby("risk_segment", observed=True)
            .agg(
                n_customers=("prob", "count"),
                avg_churn_prob=("prob", "mean"),
                actual_churners=("actual", "sum"),
            )
            .reset_index()
        )
        segment_summary["churn_rate"] = (
            segment_summary["actual_churners"] / segment_summary["n_customers"]
        )
        estimated_ltv_at_risk = segment_summary["actual_churners"] * self.config.avg_customer_ltv
        segment_summary["ltv_at_risk"] = estimated_ltv_at_risk
        return segment_summary

    def format_summary(self, report: dict) -> str:
        """Return a human-readable executive summary string."""
        r = report
        lines = [
            "=" * 60,
            "  BUSINESS IMPACT REPORT - CHURN PREDICTION MODEL",
            "=" * 60,
            f"  Customers scored:         {r['total_customers_scored']:>6,}",
            f"  Actual churners:           {r['actual_churners']:>6,}",
            f"  Identified by model:       {r['churners_identified']:>6,} ({r['churners_identified']/max(r['actual_churners'],1):.0%} recall)",
            "",
            "  -- MODEL CAMPAIGN (Targeted) ------------------",
            f"  Customers contacted:       {r['targeted_population']:>6,}",
            f"  Retention spend:           EUR {r['retention_spend_model']:>9,.0f}",
            f"  Revenue protected:         EUR {r['revenue_saved_model']:>9,.0f}",
            f"  Net benefit:               EUR {r['net_benefit_model']:>9,.0f}",
            f"  ROI:                       {r['model_roi_pct']:>8.1f}%",
            "",
            "  -- BASELINE CAMPAIGN (No Model) ---------------",
            f"  Customers contacted:       {r['total_customers_scored']:>6,}",
            f"  Retention spend:           EUR {r['baseline_spend']:>9,.0f}",
            f"  Revenue protected:         EUR {r['revenue_saved_baseline']:>9,.0f}",
            f"  Net benefit:               EUR {r['net_benefit_baseline']:>9,.0f}",
            "",
            "  -- INCREMENTAL VALUE OF THE MODEL -------------",
            f"  Extra net benefit:         EUR {r['incremental_net_benefit']:>9,.0f}",
            "=" * 60,
        ]
        return "\n".join(lines)
