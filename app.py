"""
ML Model Monitoring and Drift Detection Dashboard
Interactive Streamlit application demonstrating Dataset Management, Statistical Drift Detection,
Model Performance Tracking, Automated MLOps Workflows, and OCI Architecture Equivalence.
"""

import os
import sys
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    initialize_datasets_on_disk,
    load_dataset,
    get_data_quality_report,
)
from src.drift_detector import (
    compute_drift_scorecard,
    compute_ecdf,
    calculate_psi,
)
from src.model_manager import (
    get_model,
    train_and_save_baseline_model,
    evaluate_batch,
)
from src.mlops_pipeline import (
    evaluate_retraining_trigger,
    record_monitoring_run,
    load_audit_history,
    generate_markdown_report,
)
from src.oci_mapping import (
    get_oci_service_mappings,
    get_oci_architecture_workflow,
)

# Page configuration
st.set_page_config(
    page_title="OCI ML Model Monitor & Drift Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize data and models
initialize_datasets_on_disk()
pipeline, model_meta = get_model()

# Custom CSS for polished look
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 14px 18px;
        border: 1px solid #e9ecef;
        margin-bottom: 10px;
    }
    .badge-healthy {
        background-color: #d4edda;
        color: #155724;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-critical {
        background-color: #f8d7da;
        color: #721c24;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 16px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.title("MLOps Monitor")
    st.caption("Oracle Cloud Infrastructure (OCI) MLOps Practice")

    st.markdown("---")
    st.subheader("🎯 Active Production Batch")
    batch_choice = st.selectbox(
        "Select Stream / Batch to Monitor:",
        options=[
            "Batch 1 - Stable Production (In-Distribution)",
            "Batch 2 - Moderate Feature Drift (Demographic Shift)",
            "Batch 3 - Severe Drift & Anomalies (Crisis & Outliers)",
            "Batch 4 - Pure Concept Drift (Target Relationship Inversion)",
        ],
        index=0,
    )

    batch_file_map = {
        "Batch 1 - Stable Production (In-Distribution)": "prod_batch_1_stable.csv",
        "Batch 2 - Moderate Feature Drift (Demographic Shift)": "prod_batch_2_drift.csv",
        "Batch 3 - Severe Drift & Anomalies (Crisis & Outliers)": "prod_batch_3_severe.csv",
        "Batch 4 - Pure Concept Drift (Target Relationship Inversion)": "prod_batch_4_concept.csv",
    }
    active_batch_file = batch_file_map[batch_choice]

    st.markdown("---")
    st.subheader("⚙️ Statistical Thresholds")
    p_val_thresh = st.slider("KS & Chi-Square Significance (α):", min_value=0.01, max_value=0.10, value=0.05, step=0.01)
    psi_thresh = st.slider("PSI Drift Threshold:", min_value=0.10, max_value=0.30, value=0.15, step=0.01)
    f1_sla_thresh = st.slider("F1 Degradation SLA Breach (%):", min_value=-25, max_value=-5, value=-10, step=1)

    st.markdown("---")
    if st.button("🔄 Retrain Baseline Model (v1.0.0)", use_container_width=True):
        with st.spinner("Retraining baseline scikit-learn pipeline..."):
            pipeline, model_meta = train_and_save_baseline_model()
            st.success("Model retrained and saved to disk!")
            st.rerun()

    st.markdown("---")
    st.caption(f"**Model ID:** `{model_meta.get('model_name')}`")
    st.caption(f"**Version:** `{model_meta.get('version')}` | **Status:** `ACTIVE`")
    st.caption(f"**Algorithm:** `{model_meta.get('algorithm')}`")

# ----------------- DATA LOADING & COMPUTATION -----------------
ref_df = load_dataset("reference_data.csv")
prod_df = load_dataset(active_batch_file)

# Run drift evaluation
drift_scorecard = compute_drift_scorecard(
    ref_df=ref_df,
    prod_df=prod_df,
    features=FEATURE_COLUMNS,
    numerical_features=NUMERICAL_FEATURES,
    categorical_features=CATEGORICAL_FEATURES,
    p_value_threshold=p_val_thresh,
)

# Run model evaluation
eval_results = evaluate_batch(
    pipeline=pipeline,
    batch_df=prod_df,
    ref_metrics=model_meta["reference_metrics"],
)

# Run automated retraining decision engine
retrain_decision = evaluate_retraining_trigger(
    drift_scorecard=drift_scorecard,
    eval_results=eval_results,
    f1_sla_threshold=float(f1_sla_thresh),
)

# ----------------- MAIN HEADER -----------------
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("🛡️ ML Model Monitoring & Drift Detection Dashboard")
    st.markdown(
        f"Real-time operational telemetry for **{model_meta.get('model_name')} (v{model_meta.get('version')})** "
        f"comparing **Reference Training Baseline** ({len(ref_df):,} samples) vs **{batch_choice}** ({len(prod_df):,} samples)."
    )
with col_h2:
    if retrain_decision["decision"] == "TRIGGER_RETRAINING":
        st.error(f"### {retrain_decision['status_label']}\nAutomated Retraining Alert")
    elif retrain_decision["decision"] == "WARNING_INVESTIGATE":
        st.warning(f"### {retrain_decision['status_label']}\nTelemetry Investigation Advised")
    else:
        st.success(f"### {retrain_decision['status_label']}\nModel Within Tolerances")

st.markdown("---")

# ----------------- TABS -----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 1. Dataset Management & Profiling",
    "🔍 2. Statistical Data Drift Detection",
    "📈 3. Model Performance & Degradation",
    "⚡ 4. MLOps Automation & Retraining Triggers",
    "☁️ 5. OCI Architecture Mapping Guide",
])

# ----------------- TAB 1: DATASET MANAGEMENT -----------------
with tab1:
    st.header("📊 Dataset Management & Data Quality Profiling")
    st.markdown("""
    Inspect baseline training data alongside incoming production batches. Track schema validation,
    missing value rates, distribution quantiles, and data quality anomalies before scoring.
    """)

    col_prof1, col_prof2 = st.columns(2)
    with col_prof1:
        st.subheader("Reference Training Baseline (`reference_data.csv`)")
        st.dataframe(ref_df.head(6), width="stretch")
        st.caption(f"Shape: {ref_df.shape[0]} rows × {ref_df.shape[1]} columns | Target: `{TARGET_COLUMN}`")
    with col_prof2:
        st.subheader(f"Active Production Stream (`{active_batch_file}`)")
        st.dataframe(prod_df.head(6), width="stretch")
        st.caption(f"Shape: {prod_df.shape[0]} rows × {prod_df.shape[1]} columns | Target: `{TARGET_COLUMN}`")

    st.markdown("---")
    st.subheader("Data Quality & Summary Statistics Comparison")

    col_q1, col_q2 = st.columns(2)
    with col_q1:
        st.markdown("**Reference Baseline Summary Statistics**")
        ref_quality = get_data_quality_report(ref_df)
        st.dataframe(ref_quality, width="stretch")

    with col_q2:
        st.markdown(f"**Production Batch Summary Statistics ({batch_choice[:10]})**")
        prod_quality = get_data_quality_report(prod_df)
        st.dataframe(prod_quality, width="stretch")

    # Missing value warning
    prod_missing = prod_df.isna().sum().sum()
    if prod_missing > 0:
        st.warning(
            f"⚠️ Data Quality Anomaly Detected: Ingested batch contains {prod_missing} missing values across features. "
            f"The scikit-learn `SimpleImputer` preprocessing pipeline handles these via median/mode imputation."
        )

# ----------------- TAB 2: DATA DRIFT DETECTION -----------------
with tab2:
    st.header("🔍 Statistical Data Drift Detection Engine")
    st.markdown("""
    Evaluate covariate drift between training reference and live inference distributions. Continuous features
    are tested using the **Kolmogorov–Smirnov (KS) test** and **Population Stability Index (PSI)**.
    Categorical features are tested via **Chi-Square ($\\chi^2$) Contingency tests**.
    """)

    # KPI summary metrics
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total Monitored Features", drift_scorecard["total_features"])
    with kpi2:
        st.metric(
            "Features with Detected Drift",
            f"{drift_scorecard['drifted_count']} / {drift_scorecard['total_features']}",
            delta=f"{drift_scorecard['drift_ratio']}% drifted",
            delta_color="inverse",
        )
    with kpi3:
        st.metric(
            "Critical Severity Drift (PSI ≥ 0.25)",
            drift_scorecard["high_drift_count"],
            delta_color="inverse",
        )
    with kpi4:
        st.metric(
            "Overall Distribution Status",
            drift_scorecard["overall_status"].replace("_", " "),
        )

    st.info(f"**Recommendation:** {drift_scorecard['recommendation']}")

    st.markdown("---")
    st.subheader("Interactive Feature Distribution & CDF Inspector")

    selected_feature = st.selectbox(
        "Select Feature to inspect statistical divergence:",
        options=FEATURE_COLUMNS,
        index=1 if "tenure_months" in FEATURE_COLUMNS else 0,
    )

    feat_detail = drift_scorecard["feature_details"][selected_feature]

    col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
    col_meta1.metric("Methodology", feat_detail["test_type"])
    col_meta2.metric("Test Statistic", feat_detail["stat"])
    col_meta3.metric("p-value", f"{feat_detail['p_value']:.5f}", delta="< 0.05 Drift" if feat_detail['p_value'] < p_val_thresh else "Stable")
    col_meta4.metric("PSI Score", feat_detail["psi"], delta=feat_detail["severity"])

    if selected_feature in NUMERICAL_FEATURES:
        col_fig1, col_fig2 = st.columns(2)

        # Plot 1: Histogram / Density overlay
        with col_fig1:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=ref_df[selected_feature].dropna(),
                name="Reference (Baseline)",
                opacity=0.6,
                marker_color="#2b5c8f",
                histnorm="probability density",
            ))
            fig_hist.add_trace(go.Histogram(
                x=prod_df[selected_feature].dropna(),
                name="Production (Batch)",
                opacity=0.6,
                marker_color="#e06666",
                histnorm="probability density",
            ))
            fig_hist.update_layout(
                title=f"Probability Density Overlay: {selected_feature}",
                barmode="overlay",
                xaxis_title=selected_feature,
                yaxis_title="Density",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=380,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # Plot 2: Empirical CDF (eCDF) for Kolmogorov-Smirnov test visualization
        with col_fig2:
            x_ref, y_ref = compute_ecdf(ref_df[selected_feature])
            x_prod, y_prod = compute_ecdf(prod_df[selected_feature])

            fig_ecdf = go.Figure()
            fig_ecdf.add_trace(go.Scatter(
                x=x_ref, y=y_ref,
                mode="lines",
                name="Reference Baseline CDF",
                line=dict(color="#2b5c8f", width=2.5),
            ))
            fig_ecdf.add_trace(go.Scatter(
                x=x_prod, y=y_prod,
                mode="lines",
                name="Production Batch CDF",
                line=dict(color="#e06666", width=2.5, dash="dash"),
            ))
            fig_ecdf.update_layout(
                title=f"Kolmogorov–Smirnov eCDF Comparison (D = {feat_detail['stat']})",
                xaxis_title=selected_feature,
                yaxis_title="Cumulative Probability F(x)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=380,
            )
            st.plotly_chart(fig_ecdf, use_container_width=True)

        # PSI Breakdown Table
        with st.expander(f"📋 View 10-Bucket Population Stability Index (PSI) Breakdown for `{selected_feature}`"):
            if "psi_breakdown" in feat_detail and isinstance(feat_detail["psi_breakdown"], pd.DataFrame):
                st.dataframe(feat_detail["psi_breakdown"], width="stretch")

    else:
        # Categorical bar chart comparison
        ref_cat_pct = ref_df[selected_feature].value_counts(normalize=True).reset_index()
        ref_cat_pct.columns = [selected_feature, "Reference %"]
        prod_cat_pct = prod_df[selected_feature].value_counts(normalize=True).reset_index()
        prod_cat_pct.columns = [selected_feature, "Production %"]

        merged_cat = pd.merge(ref_cat_pct, prod_cat_pct, on=selected_feature, how="outer").fillna(0)
        merged_cat["Reference %"] = (merged_cat["Reference %"] * 100).round(2)
        merged_cat["Production %"] = (merged_cat["Production %"] * 100).round(2)

        fig_cat = go.Figure()
        fig_cat.add_trace(go.Bar(
            x=merged_cat[selected_feature],
            y=merged_cat["Reference %"],
            name="Reference (Baseline %)",
            marker_color="#2b5c8f"
        ))
        fig_cat.add_trace(go.Bar(
            x=merged_cat[selected_feature],
            y=merged_cat["Production %"],
            name="Production (Batch %)",
            marker_color="#e06666"
        ))
        fig_cat.update_layout(
            title=f"Categorical Frequency Distribution: {selected_feature} (χ² p-val: {feat_detail['p_value']:.4f})",
            barmode="group",
            yaxis_title="Proportion (%)",
            height=380,
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Complete Feature Drift Scorecard")
    st.dataframe(drift_scorecard["summary_table"], width="stretch")

# ----------------- TAB 3: MODEL PERFORMANCE -----------------
with tab3:
    st.header("📈 Model Performance & Ground Truth Degradation")
    st.markdown("""
    Evaluate model quality on production batches when delayed feedback labels are acquired.
    Detect metric degradation (Accuracy, Precision, Recall, F1, ROC-AUC) against reference validation benchmarks.
    """)

    ref_m = model_meta["reference_metrics"]
    prod_m = eval_results.get("metrics", {})
    deltas = eval_results.get("deltas", {})

    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.metric(
            "Accuracy",
            f"{prod_m.get('accuracy', 0.0) * 100:.1f}%",
            delta=f"{deltas.get('accuracy', 0.0):+.2f}% vs Ref",
        )
    with m_col2:
        st.metric(
            "F1 Score",
            f"{prod_m.get('f1', 0.0):.3f}",
            delta=f"{deltas.get('f1', 0.0):+.2f}% vs Ref",
            delta_color="normal",
        )
    with m_col3:
        st.metric(
            "ROC-AUC",
            f"{prod_m.get('roc_auc', 0.0):.3f}",
            delta=f"{deltas.get('roc_auc', 0.0):+.2f}% vs Ref",
        )
    with m_col4:
        st.metric(
            "Precision",
            f"{prod_m.get('precision', 0.0):.3f}",
        )
    with m_col5:
        st.metric(
            "Recall",
            f"{prod_m.get('recall', 0.0):.3f}",
        )

    if eval_results.get("sla_breach", False):
        st.error(
            f"🚨 SLA BREACH: F1 score degradation ({deltas.get('f1', 0.0):+.2f}%) exceeds the "
            f"configured SLA threshold ({f1_sla_thresh}%). Model predictions have degraded below acceptable bounds!"
        )

    col_cm, col_roc = st.columns(2)
    with col_cm:
        cm = np.array(eval_results["confusion_matrix"])
        labels = ["Retain (0)", "Churn (1)"]
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=labels,
            y=labels,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted Label", y="Actual Ground Truth", color="Count"),
            title=f"Production Confusion Matrix ({batch_choice[:10]})"
        )
        fig_cm.update_layout(height=360)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_roc:
        roc_data = eval_results.get("roc_curve", {})
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=roc_data.get("fpr", []),
            y=roc_data.get("tpr", []),
            mode="lines",
            name=f"Production ROC (AUC = {prod_m.get('roc_auc', 0.0):.3f})",
            line=dict(color="#1f77b4", width=3),
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            name="Random Guessing (AUC = 0.5)",
            line=dict(color="grey", dash="dash"),
        ))
        fig_roc.update_layout(
            title="Receiver Operating Characteristic (ROC) Curve",
            xaxis_title="False Positive Rate (FPR)",
            yaxis_title="True Positive Rate (TPR)",
            height=360,
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("---")
    st.subheader("Model Feature Importance Profile")
    imp_dict = model_meta.get("feature_importances", {})
    top_features = list(imp_dict.keys())[:10]
    top_values = [imp_dict[k] for k in top_features]

    fig_imp = px.bar(
        x=top_values,
        y=top_features,
        orientation="h",
        title="Top 10 Feature Importances (Gradient Boosting Classifier)",
        labels={"x": "Importance Weight", "y": "Feature"},
        color=top_values,
        color_continuous_scale="Tealgrn"
    )
    fig_imp.update_layout(yaxis=dict(autorange="reversed"), height=350)
    st.plotly_chart(fig_imp, use_container_width=True)

# ----------------- TAB 4: MLOPS AUTOMATION & RETRAINING -----------------
with tab4:
    st.header("⚡ MLOps Automated Workflow & Retraining Triggers")
    st.markdown("""
    Enterprise MLOps combines statistical drift detection with SLA performance tracking to automate
    governance decisions. This module simulates event-driven alert triggers and generates audit compliance logs.
    """)

    col_dec1, col_dec2 = st.columns([2, 1])
    with col_dec1:
        st.subheader("Automated Governance Decision")
        st.markdown(f"**Status:** `{retrain_decision['status_label']}` | **Priority:** **{retrain_decision['priority']}**")
        st.markdown("**Root Cause Rationale:**")
        for reason in retrain_decision["reasons"]:
            st.markdown(f"- ⚠️ {reason}")

        st.markdown("**Automated Remediation Actions:**")
        for action in retrain_decision["recommended_actions"]:
            st.markdown(f"- ✅ {action}")

    with col_dec2:
        st.subheader("Simulate Audit Record")
        if st.button("📝 Log Run to Audit Trail", use_container_width=True, type="primary"):
            run_rec = record_monitoring_run(
                batch_name=batch_choice,
                model_metadata=model_meta,
                drift_scorecard=drift_scorecard,
                eval_results=eval_results,
                retrain_decision=retrain_decision,
            )
            st.success(f"Logged run: `{run_rec['run_id']}`")
            st.rerun()

        # Download Markdown Report
        md_report = generate_markdown_report(
            batch_name=batch_choice,
            model_metadata=model_meta,
            drift_scorecard=drift_scorecard,
            eval_results=eval_results,
            decision=retrain_decision,
        )
        st.download_button(
            label="📥 Download Executive Audit Report (.md)",
            data=md_report,
            file_name=f"ML_Drift_Audit_{active_batch_file.replace('.csv', '')}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.markdown("---")
    st.subheader("Simulated OCI CloudEvent Payload")
    st.caption("This JSON event payload is dispatched to the OCI Events Service when drift or SLA breach thresholds are met.")
    cloud_event = {
        "eventType": "com.oraclecloud.datascience.modelmonitoring.driftalert",
        "cloudEventsVersion": "0.1",
        "source": f"ocid1.datasciencemonitor.oc1.iad.{active_batch_file.split('.')[0]}",
        "eventTime": pd.Timestamp.now().isoformat() + "Z",
        "data": {
            "modelId": model_meta.get("oci_model_catalog_ocid"),
            "modelVersion": model_meta.get("version"),
            "activeBatch": batch_choice,
            "overallDriftStatus": drift_scorecard["overall_status"],
            "driftedFeaturesRatio": f"{drift_scorecard['drift_ratio']}%",
            "deltaF1": f"{eval_results.get('deltas', {}).get('f1', 0.0):+.2f}%",
            "actionRequired": retrain_decision["decision"],
        }
    }
    st.json(cloud_event)

    st.markdown("---")
    st.subheader("📜 Historical Audit Log")
    audit_history = load_audit_history()
    if audit_history:
        audit_rows = []
        for run in audit_history:
            audit_rows.append({
                "Run ID": run["run_id"],
                "Timestamp": run["timestamp"],
                "Batch": run["batch_name"][:25] + "...",
                "Model Version": run["model_version"],
                "Drifted Features": f"{run['drifted_features_count']} / {run['total_features']}",
                "Decision": run["decision"],
                "Priority": run["priority"],
            })
        st.dataframe(pd.DataFrame(audit_rows), width="stretch")
    else:
        st.info("No audit runs recorded yet. Click 'Log Run to Audit Trail' above to log the current execution.")

# ----------------- TAB 5: OCI CLOUD MAPPING GUIDE -----------------
with tab5:
    st.header("☁️ Oracle Cloud Infrastructure (OCI) Service Mapping Guide")
    st.markdown("""
    This project demonstrates how enterprise ML model monitoring and MLOps automation map to 
    native Oracle Cloud Infrastructure (OCI) services covered in the OCI Data Science & AI curriculum.
    """)

    st.subheader("1. Service Equivalence Matrix")
    mappings = get_oci_service_mappings()
    mapping_df = pd.DataFrame(mappings)
    st.table(mapping_df[["Icon", "Local Component", "Course Topic", "OCI Service", "Role in OCI"]])

    st.markdown("---")
    st.subheader("2. OCI Cloud Architecture Workflow")
    st.markdown("""
    The diagram below illustrates how an enterprise Oracle Cloud Infrastructure deployment
    manages the complete lifecycle: from Object Storage ingestion through Model Monitoring,
    Event dispatching, and automated Pipeline retraining.
    """)

    st.code(get_oci_architecture_workflow(), language="mermaid")

    col_oci1, col_oci2 = st.columns(2)
    with col_oci1:
        st.markdown("""
        ### Key OCI Best Practices Demonstrated:
        1. **Decoupled Monitoring & Ingestion:**
           - Data streams into **OCI Object Storage** without locking models.
           - **OCI Data Science Model Monitoring** runs asynchronously on scheduled intervals.
        2. **Immutable Governance:**
           - Models, schemas, and conda dependencies are versioned in the **OCI Model Catalog**.
           - Every audit run provides a deterministic provenance trail.
        """)
    with col_oci2:
        st.markdown("""
        ### Cloud Automation Mechanics:
        1. **Event-Driven Retraining:**
           - Metric alarms publish events to **OCI Events Service**.
           - **OCI Functions** parses the payload and kicks off an **OCI Data Science Pipeline**.
        2. **Observability & SLAs:**
           - Telemetry streams into **OCI Logging** and **OCI Monitoring MQL**.
           - On-call notification through **OCI Notifications (ONS)** via Slack/PagerDuty.
        """)

st.markdown("---")
st.caption("Built with Python, Pandas, SciPy, Scikit-Learn, Plotly & Streamlit | MLOps & OCI AI Architecture")
