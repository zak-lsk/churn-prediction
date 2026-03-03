"""
app.py
──────
Streamlit multi-page dashboard for the Churn Prediction project.
5 pages: Overview · Data Explorer · Model Results · Live Predictor · Business Impact

Run with:
    streamlit run app/app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction | Analytics Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1A1D27 0%, #252836 100%);
        border: 1px solid #2D3047;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover { transform: translateY(-2px); }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1117 0%, #1A1D27 100%);
        border-right: 1px solid #2D3047;
    }

    /* Headers */
    h1 { color: #6C63FF !important; font-weight: 700 !important; }
    h2 { color: #FAFAFA !important; font-weight: 600 !important; }
    h3 { color: #A0A3BD !important; font-weight: 500 !important; }

    /* Expander */
    .streamlit-expanderHeader { font-weight: 600; }

    /* Churn badge colors */
    .badge-high { color: #FF6B6B; font-weight: 700; }
    .badge-low  { color: #6BCB77; font-weight: 700; }
    .badge-med  { color: #FFD93D; font-weight: 700; }

    /* Divider */
    hr { border-color: #2D3047 !important; }
</style>
""", unsafe_allow_html=True)


# ─── Constants ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"

PURPLE = "#6C63FF"
GREEN  = "#6BCB77"
RED    = "#FF6B6B"
YELLOW = "#FFD93D"
DARK   = "#0E1117"


# ─── Data & Model Loaders ────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame | None:
    """Load and cache the cleaned dataset."""
    if not DATA_PATH.exists():
        return None
    from src.data_pipeline import DataPipeline
    from src.feature_engineering import add_business_features
    dp = DataPipeline(DATA_PATH)
    df = dp.run()
    df = add_business_features(df)
    return df


@st.cache_resource(show_spinner=False)
def load_model():
    """Load and cache the trained model."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


# ─── Sidebar Navigation ──────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown("## 📡 Churn Analytics")
        st.markdown("*Telco Customer Churn Prediction*")
        st.divider()
        page = st.radio(
            "Navigate",
            options=[
                "🏠 Overview",
                "📊 Data Explorer",
                "🤖 Model Results",
                "🔮 Live Predictor",
                "💶 Business Impact",
            ],
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown(
            "<small>Built with Python · XGBoost · Streamlit<br>"
            "Dataset: IBM Telco Customer Churn</small>",
            unsafe_allow_html=True,
        )
    return page


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 1 · OVERVIEW                                                        ║
# ╚════════════════════════════════════════════════════════════════════════════╝
def page_overview(df: pd.DataFrame | None):
    st.title("📡 Customer Churn Prediction")
    st.markdown(
        "**End-to-end ML project** identifying customers at risk of churning "
        "to enable targeted retention campaigns. Industry: **Telecommunications**."
    )
    st.divider()

    if df is None:
        _show_dataset_warning()
        return

    churn_rate = df["Churn"].mean()
    n_churned  = df["Churn"].sum()
    n_retained = len(df) - n_churned
    avg_monthly = df["MonthlyCharges"].mean()
    avg_tenure  = df["tenure"].mean()

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Customers", f"{len(df):,}")
    c2.metric("Churn Rate", f"{churn_rate:.1%}", delta=f"{churn_rate-0.26:.1%} vs industry avg")
    c3.metric("Churned", f"{n_churned:,}", delta=f"-{n_churned:,}", delta_color="inverse")
    c4.metric("Avg Monthly Charge", f"€{avg_monthly:.0f}")
    c5.metric("Avg Tenure", f"{avg_tenure:.0f} months")

    st.divider()
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("### Churn Distribution by Contract Type")
        churn_by_contract = (
            df.groupby(["Contract", "Churn"])
            .size()
            .reset_index(name="count")
        )
        churn_by_contract["label"] = churn_by_contract["Churn"].map({1: "Churned", 0: "Retained"})
        fig = px.bar(
            churn_by_contract,
            x="Contract", y="count", color="label",
            barmode="group",
            color_discrete_map={"Churned": RED, "Retained": GREEN},
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor=DARK, plot_bgcolor=DARK,
            legend_title="Customer Status",
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig, on_select='ignore')

    with col_right:
        st.markdown("### Overall Churn Split")
        fig_pie = px.pie(
            values=[n_churned, n_retained],
            names=["Churned", "Retained"],
            color_discrete_sequence=[RED, GREEN],
            hole=0.6,
            template="plotly_dark",
        )
        fig_pie.update_layout(
            paper_bgcolor=DARK,
            showlegend=True,
            margin=dict(l=0, r=0, t=30, b=0),
            annotations=[dict(
                text=f"{churn_rate:.1%}<br>Churn",
                x=0.5, y=0.5, font_size=18,
                showarrow=False,
                font_color="white",
            )],
        )
        st.plotly_chart(fig_pie, on_select='ignore')

    st.divider()
    st.markdown("### Monthly Charges Distribution: Churned vs Retained")
    fig_box = px.box(
        df.assign(label=df["Churn"].map({1: "Churned", 0: "Retained"})),
        x="label", y="MonthlyCharges", color="label",
        color_discrete_map={"Churned": RED, "Retained": GREEN},
        points="outliers",
        template="plotly_dark",
    )
    fig_box.update_layout(
        paper_bgcolor=DARK, plot_bgcolor=DARK,
        showlegend=False,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_box, on_select='ignore')


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 2 · DATA EXPLORER                                                   ║
# ╚════════════════════════════════════════════════════════════════════════════╝
def page_data_explorer(df: pd.DataFrame | None):
    st.title("📊 Data Explorer")
    st.markdown("Explore the raw patterns and feature distributions in the dataset.")

    if df is None:
        _show_dataset_warning()
        return

    # ── Filters
    st.sidebar.divider()
    st.sidebar.markdown("**Filters**")
    selected_contract = st.sidebar.multiselect(
        "Contract Type",
        df["Contract"].unique().tolist(),
        default=df["Contract"].unique().tolist(),
    )
    selected_internet = st.sidebar.multiselect(
        "Internet Service",
        df["InternetService"].unique().tolist(),
        default=df["InternetService"].unique().tolist(),
    )

    df_filtered = df[
        df["Contract"].isin(selected_contract) &
        df["InternetService"].isin(selected_internet)
    ]
    st.markdown(f"*Showing **{len(df_filtered):,}** of {len(df):,} customers*")
    st.divider()

    # ── Churn rate heatmap by service combos
    t1, t2, t3 = st.tabs(["Feature Distributions", "Correlation Heatmap", "Churn by Segment"])

    with t1:
        num_col = st.selectbox(
            "Select a numerical feature:",
            ["MonthlyCharges", "TotalCharges", "tenure", "MonthlyChargesPerService", "ChargesPerMonth"],
        )
        fig = px.histogram(
            df_filtered.assign(label=df_filtered["Churn"].map({1: "Churned", 0: "Retained"})),
            x=num_col, color="label", barmode="overlay", nbins=40,
            color_discrete_map={"Churned": RED, "Retained": GREEN},
            template="plotly_dark", opacity=0.75,
        )
        fig.update_layout(paper_bgcolor=DARK, plot_bgcolor=DARK,
                          margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, on_select='ignore')

    with t2:
        num_cols = ["tenure", "MonthlyCharges", "TotalCharges",
                    "MonthlyChargesPerService", "ChargesPerMonth", "Churn"]
        corr = df_filtered[num_cols].corr()
        fig_heat = px.imshow(
            corr.round(2),
            text_auto=True,
            color_continuous_scale="RdBu_r",
            template="plotly_dark",
            aspect="auto",
        )
        fig_heat.update_layout(paper_bgcolor=DARK, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_heat, on_select='ignore')

    with t3:
        cat_col = st.selectbox(
            "Segment by:",
            ["Contract", "InternetService", "PaymentMethod", "TenureGroup",
             "ContractRisk", "SeniorCitizen", "Dependents"],
        )
        churn_rate_seg = (
            df_filtered.groupby(cat_col)["Churn"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "Churn Rate", "count": "Customers"})
        )
        churn_rate_seg["Churn Rate"] = churn_rate_seg["Churn Rate"] * 100
        fig_seg = px.bar(
            churn_rate_seg.sort_values("Churn Rate", ascending=False),
            x=cat_col, y="Churn Rate",
            color="Churn Rate",
            color_continuous_scale=[[0, GREEN], [0.5, YELLOW], [1, RED]],
            text="Churn Rate",
            template="plotly_dark",
        )
        fig_seg.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_seg.update_layout(
            paper_bgcolor=DARK, plot_bgcolor=DARK,
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_seg, on_select='ignore')


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 3 · MODEL RESULTS                                                   ║
# ╚════════════════════════════════════════════════════════════════════════════╝
def page_model_results(df: pd.DataFrame | None, model=None):
    st.title("🤖 Model Results")
    st.markdown(
        "Performance evaluation of the **XGBoost Churn Classifier** with SMOTE oversampling "
        "to handle the class imbalance (~26% churn rate)."
    )

    if df is None:
        _show_dataset_warning()
        return

    if model is None:
        st.warning(
            "⚠️ No trained model found. Run the training pipeline first:\n"
            "```\npython src/train.py\n```",
            icon="⚠️",
        )
        return

    from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score
    from src.feature_engineering import add_business_features
    from src.data_pipeline import DataPipeline
    from sklearn.model_selection import train_test_split

    try:
        X = df.drop(columns=["Churn"])
        y = df["Churn"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        y_prob = model.predict_proba(X_test)[:, 1]
        threshold = st.slider("Decision Threshold (lower = higher recall)", 0.1, 0.9, 0.4, 0.05)
        y_pred = (y_prob >= threshold).astype(int)

        from sklearn.metrics import precision_score, recall_score, f1_score
        prec  = precision_score(y_test, y_pred, zero_division=0)
        rec   = recall_score(y_test, y_pred, zero_division=0)
        f1    = f1_score(y_test, y_pred, zero_division=0)
        auc   = roc_auc_score(y_test, y_prob)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precision", f"{prec:.1%}", help="Of customers flagged as churners, how many actually churned?")
        c2.metric("Recall",    f"{rec:.1%}",  help="Of actual churners, how many did we catch?")
        c3.metric("F1 Score",  f"{f1:.1%}",   help="Harmonic mean of Precision and Recall")
        c4.metric("AUC-ROC",   f"{auc:.3f}",  help="Overall discrimination ability of the model")

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, fill="tozeroy",
                                         fillcolor="rgba(108,99,255,0.15)",
                                         line=dict(color=PURPLE, width=2.5),
                                         name=f"AUC = {auc:.3f}"))
            fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1],
                                         line=dict(color="gray", dash="dash"),
                                         name="Random"))
            fig_roc.update_layout(
                xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                template="plotly_dark", paper_bgcolor=DARK, plot_bgcolor=DARK,
                legend=dict(x=0.6, y=0.1),
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_roc, on_select='ignore')

        with col2:
            st.markdown("### Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            labels = [["TN<br>Correct: Not churn", "FP<br>False alarm"],
                      ["FN<br>Missed churner", "TP<br>Correct: Churn"]]
            annotations_text = [
                [f"<b>{cm[0,0]}</b><br>{labels[0][0]}", f"<b>{cm[0,1]}</b><br>{labels[0][1]}"],
                [f"<b>{cm[1,0]}</b><br>{labels[1][0]}", f"<b>{cm[1,1]}</b><br>{labels[1][1]}"],
            ]
            fig_cm = px.imshow(
                cm, text_auto=True,
                color_continuous_scale=[[0, DARK], [1, PURPLE]],
                labels=dict(x="Predicted", y="Actual"),
                x=["Retained", "Churned"], y=["Retained", "Churned"],
                template="plotly_dark",
            )
            fig_cm.update_layout(paper_bgcolor=DARK, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_cm, on_select='ignore')

        st.divider()
        st.markdown("### Feature Importance")
        try:
            # XGBoost or RF feature importances from pipeline
            clf = model.named_steps["classifier"]
            preprocessor_step = model.named_steps["preprocessor"]
            if hasattr(clf, "feature_importances_"):
                feat_names = preprocessor_step.get_feature_names_out()
                importances = clf.feature_importances_
                fi_df = (
                    pd.DataFrame({"Feature": feat_names, "Importance": importances})
                    .sort_values("Importance", ascending=False)
                    .head(20)
                )
                # Clean up feature names
                fi_df["Feature"] = fi_df["Feature"].str.replace(r"^(num__|cat__)", "", regex=True)
                fig_fi = px.bar(
                    fi_df.sort_values("Importance"),
                    x="Importance", y="Feature", orientation="h",
                    color="Importance",
                    color_continuous_scale=[[0, "#2D3047"], [1, PURPLE]],
                    template="plotly_dark",
                )
                fig_fi.update_layout(
                    paper_bgcolor=DARK, plot_bgcolor=DARK,
                    coloraxis_showscale=False,
                    height=500,
                    margin=dict(l=0, r=0, t=10, b=0),
                )
                st.plotly_chart(fig_fi, on_select='ignore')
        except Exception as e:
            st.info(f"Feature importance not available for this model type: {e}")

    except Exception as e:
        st.error(f"Error during model evaluation: {e}")


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 4 · LIVE PREDICTOR                                                  ║
# ╚════════════════════════════════════════════════════════════════════════════╝
def page_live_predictor(df: pd.DataFrame | None, model=None):
    st.title("🔮 Live Churn Predictor")
    st.markdown(
        "Enter a customer's profile below and the model will estimate their **probability of churning**."
    )

    if df is None:
        _show_dataset_warning()
        return
    if model is None:
        st.warning("⚠️ No trained model found. Run `python src/train.py` first.")
        return

    st.divider()
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📋 Account Info**")
        gender       = st.selectbox("Gender", ["Male", "Female"], key="p_gender")
        senior       = st.selectbox("Senior Citizen", ["No", "Yes"], key="p_senior")
        partner      = st.selectbox("Partner", ["No", "Yes"], key="p_partner")
        dependents   = st.selectbox("Dependents", ["No", "Yes"], key="p_dep")
        tenure       = st.slider("Tenure (months)", 0, 72, 12, key="p_tenure")

    with col2:
        st.markdown("**📦 Services**")
        phone_service  = st.selectbox("Phone Service", ["Yes", "No"], key="p_phone")
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"], key="p_lines")
        internet       = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], key="p_internet")
        online_sec     = st.selectbox("Online Security", ["No", "Yes"], key="p_sec")
        online_backup  = st.selectbox("Online Backup", ["No", "Yes"], key="p_backup")
        device_prot    = st.selectbox("Device Protection", ["No", "Yes"], key="p_dev")
        tech_support   = st.selectbox("Tech Support", ["No", "Yes"], key="p_tech")
        streaming_tv   = st.selectbox("Streaming TV", ["No", "Yes"], key="p_tv")
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"], key="p_movies")

    with col3:
        st.markdown("**💳 Billing**")
        contract        = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="p_contract")
        paperless       = st.selectbox("Paperless Billing", ["Yes", "No"], key="p_paper")
        payment_method  = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            key="p_payment",
        )
        monthly_charges = st.slider("Monthly Charges (€)", 18.0, 120.0, 65.0, 0.5, key="p_monthly")
        total_charges   = st.number_input(
            "Total Charges (€)",
            min_value=0.0, max_value=10000.0,
            value=float(monthly_charges * max(tenure, 1)),
            key="p_total",
        )

    st.divider()
    if st.button("🔮  Predict Churn Risk", use_container_width="stretch", type="primary"):
        input_dict = {
            "gender": gender, "SeniorCitizen": senior, "Partner": partner,
            "Dependents": dependents, "tenure": tenure,
            "PhoneService": phone_service, "MultipleLines": multiple_lines,
            "InternetService": internet, "OnlineSecurity": online_sec,
            "OnlineBackup": online_backup, "DeviceProtection": device_prot,
            "TechSupport": tech_support, "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies, "Contract": contract,
            "PaperlessBilling": paperless, "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
        }
        input_df = pd.DataFrame([input_dict])

        # Add business features
        from src.feature_engineering import add_business_features
        input_enriched = add_business_features(input_df)

        try:
            prob = model.predict_proba(input_enriched)[0][1]
            pred = int(prob >= 0.4)

            # ── Result display
            if prob >= 0.66:
                color, icon, label = RED, "🔴", "HIGH RISK"
            elif prob >= 0.33:
                color, icon, label = YELLOW, "🟡", "MEDIUM RISK"
            else:
                color, icon, label = GREEN, "🟢", "LOW RISK"

            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1A1D27, #252836);
                        border: 2px solid {color}; border-radius: 16px;
                        padding: 32px; text-align: center; margin-top: 16px;">
                <div style="font-size: 48px;">{icon}</div>
                <div style="font-size: 32px; font-weight: 700; color: {color};">{label}</div>
                <div style="font-size: 52px; font-weight: 800; color: white; margin: 8px 0;">
                    {prob:.1%}
                </div>
                <div style="color: #A0A3BD; font-size: 16px;">probability of churn</div>
            </div>
            """, unsafe_allow_html=True)

            st.divider()
            col_a, col_b = st.columns(2)
            with col_a:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={"suffix": "%", "font": {"size": 36}},
                    gauge={
                        "axis": {"range": [0, 100], "tickfont": {"color": "white"}},
                        "bar": {"color": color},
                        "steps": [
                            {"range": [0, 33], "color": "#1A3D2B"},
                            {"range": [33, 66], "color": "#3D3A1A"},
                            {"range": [66, 100], "color": "#3D1A1A"},
                        ],
                        "threshold": {"line": {"color": "white", "width": 3}, "value": 40},
                    },
                ))
                fig_gauge.update_layout(
                    paper_bgcolor=DARK, font_color="white",
                    height=280, margin=dict(l=20, r=20, t=40, b=0),
                )
                st.plotly_chart(fig_gauge, on_select='ignore')

            with col_b:
                st.markdown("#### 💡 Risk Factors")
                risk_factors = []
                if contract == "Month-to-month":
                    risk_factors.append("📋 Month-to-month contract (+++ risk)")
                if payment_method == "Electronic check":
                    risk_factors.append("💳 Electronic check payment (+ risk)")
                if internet == "Fiber optic" and monthly_charges > 80:
                    risk_factors.append("💸 High monthly charges for Fiber (+risk)")
                if tenure < 12:
                    risk_factors.append("⏱️ New customer (<12 months) (+ risk)")
                if online_sec == "No" and internet != "No":
                    risk_factors.append("🔒 No Online Security addon (+ risk)")
                if not risk_factors:
                    risk_factors.append("✅ No major risk factors detected")

                for rf in risk_factors:
                    st.markdown(f"- {rf}")

                st.markdown("#### 🛡️ Recommended Action")
                if prob >= 0.66:
                    st.error("**Immediate retention call** + loyalty discount offer")
                elif prob >= 0.33:
                    st.warning("**Email campaign** with contract upgrade incentive")
                else:
                    st.success("**No action needed** — monitor quarterly")

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 5 · BUSINESS IMPACT                                                 ║
# ╚════════════════════════════════════════════════════════════════════════════╝
def page_business_impact(df: pd.DataFrame | None, model=None):
    st.title("💶 Business Impact Calculator")
    st.markdown(
        "Translate model performance into **real monetary value**. "
        "Adjust the business parameters to simulate different scenarios."
    )

    if df is None:
        _show_dataset_warning()
        return

    st.divider()
    st.markdown("### ⚙️ Business Parameters")
    col1, col2 = st.columns(2)

    with col1:
        ltv = st.slider(
            "Customer Lifetime Value (€)", 500, 5000, 2400, 100,
            help="Average revenue generated by a customer over their full lifecycle with the company.",
        )
        retention_cost = st.slider(
            "Retention Action Cost (€)", 50, 500, 150, 10,
            help="Cost per customer of a targeted retention action (call + discount offer).",
        )

    with col2:
        acquisition_cost = st.slider(
            "Customer Acquisition Cost (€)", 200, 2000, 900, 50,
            help="Cost to acquire a replacement customer (ads, sales, onboarding).",
        )
        success_rate = st.slider(
            "Retention Success Rate (%)", 10, 50, 25, 5,
            help="% of contacted at-risk customers who stay after a retention campaign.",
        ) / 100

    from src.business_metrics import BusinessImpactCalculator, BusinessConfig

    if model is not None and df is not None:
        from sklearn.model_selection import train_test_split
        X = df.drop(columns=["Churn"])
        y = df["Churn"]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.4).astype(int)
    else:
        # Demo mode with simulated data
        st.info("ℹ️ Showing an estimated scenario based on typical model performance.")
        n = 1409
        y_test = np.concatenate([np.ones(365), np.zeros(1044)])
        y_prob = np.concatenate([
            np.random.beta(5, 2, 365),
            np.random.beta(2, 5, 1044),
        ])
        y_pred = (y_prob >= 0.4).astype(int)

    config = BusinessConfig(
        avg_customer_ltv=ltv,
        retention_cost=retention_cost,
        acquisition_cost=acquisition_cost,
        retention_success_rate=success_rate,
    )
    calc = BusinessImpactCalculator(config=config)
    report = calc.compute(y_test, y_pred, y_prob)

    st.divider()
    st.markdown("### 📊 Results")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Churners Identified", f"{report['churners_identified']}")
    c2.metric("Revenue Protected (Model)", f"€{report['revenue_saved_model']:,.0f}")
    c3.metric("Net Benefit vs. No Model", f"€{report['incremental_net_benefit']:,.0f}")
    c4.metric("Campaign ROI", f"{report['model_roi_pct']:.0f}%")

    st.divider()
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### Model vs. Baseline Campaign Comparison")
        comparison_df = pd.DataFrame({
            "Scenario": ["Baseline (No Model)", "With Churn Model"],
            "Spend":       [report["baseline_spend"],          report["retention_spend_model"]],
            "Revenue Saved": [report["revenue_saved_baseline"], report["revenue_saved_model"]],
            "Net Benefit": [report["net_benefit_baseline"],    report["net_benefit_model"]],
        })
        fig_comp = go.Figure()
        fig_comp.add_bar(name="Spend (€)",        x=comparison_df["Scenario"], y=comparison_df["Spend"],        marker_color="#FF6B6B")
        fig_comp.add_bar(name="Revenue Saved (€)", x=comparison_df["Scenario"], y=comparison_df["Revenue Saved"], marker_color=GREEN)
        fig_comp.add_bar(name="Net Benefit (€)",   x=comparison_df["Scenario"], y=comparison_df["Net Benefit"],   marker_color=PURPLE)
        fig_comp.update_layout(
            barmode="group",
            template="plotly_dark", paper_bgcolor=DARK, plot_bgcolor=DARK,
            legend=dict(orientation="h", y=1.1),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_comp, on_select='ignore')

    with col_right:
        st.markdown("### Customer Risk Segmentation")
        seg = report["risk_segments"]
        fig_seg = px.bar(
            seg,
            x="risk_segment",
            y=["n_customers", "actual_churners"],
            barmode="group",
            color_discrete_sequence=[PURPLE, RED],
            labels={"value": "Count", "risk_segment": "Risk Segment"},
            template="plotly_dark",
        )
        fig_seg.update_layout(
            paper_bgcolor=DARK, plot_bgcolor=DARK,
            legend_title="",
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_seg, on_select='ignore')

    # Executive summary callout
    st.divider()
    st.markdown("### 📋 Executive Summary")
    st.markdown(f"""
    > **Using this churn prediction model**, the company can focus its retention budget on the
    > **{report['targeted_population']:,} customers** most likely to leave, rather than
    > contacting all {report['total_customers_scored']:,} customers blindly.
    >
    > This targeted approach is estimated to generate a net benefit of
    > **€{report['net_benefit_model']:,.0f}** vs. **€{report['net_benefit_baseline']:,.0f}**
    > for an untargeted campaign — an incremental gain of
    > **€{report['incremental_net_benefit']:,.0f}** with a campaign ROI of
    > **{report['model_roi_pct']:.0f}%**.
    >
    > *Note: Assumes {success_rate:.0%} retention success rate and €{retention_cost} cost per action.*
    """)


# ─── Dataset Warning Helper ───────────────────────────────────────────────────
def _show_dataset_warning():
    st.warning(
        "⚠️ **Dataset not found.**\n\n"
        "Download the Telco Customer Churn dataset from Kaggle and place it at:\n\n"
        "```\ndata/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv\n```\n\n"
        "🔗 [Download from Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)",
        icon="⚠️",
    )


# ─── Main Router ─────────────────────────────────────────────────────────────
def main():
    page = sidebar()

    with st.spinner("Loading data..."):
        df = load_data()

    with st.spinner("Loading model..."):
        model = load_model()

    if page == "🏠 Overview":
        page_overview(df)
    elif page == "📊 Data Explorer":
        page_data_explorer(df)
    elif page == "🤖 Model Results":
        page_model_results(df, model)
    elif page == "🔮 Live Predictor":
        page_live_predictor(df, model)
    elif page == "💶 Business Impact":
        page_business_impact(df, model)


if __name__ == "__main__":
    main()
