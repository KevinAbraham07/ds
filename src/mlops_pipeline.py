"""
MLOps Automation & Orchestration Pipeline.
Simulates enterprise model monitoring governance, automated retraining decision logic,
audit run logging, and report generation compatible with OCI Events & Functions.
"""

import os
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
AUDIT_LOG_FILE = os.path.join(REPORTS_DIR, "audit_logs.json")


def load_audit_history() -> List[Dict[str, Any]]:
    """Load audit history log from disk."""
    if not os.path.exists(AUDIT_LOG_FILE):
        return []
    try:
        with open(AUDIT_LOG_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []


def save_audit_run(run_record: Dict[str, Any]) -> None:
    """Append a new monitoring execution to the audit log."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    history = load_audit_history()
    history.insert(0, run_record)  # latest first
    # Keep last 50 runs
    history = history[:50]
    with open(AUDIT_LOG_FILE, "w") as f:
        json.dump(history, f, indent=2)


def evaluate_retraining_trigger(
    drift_scorecard: Dict[str, Any],
    eval_results: Dict[str, Any],
    f1_sla_threshold: float = -10.0,
    acc_sla_threshold: float = -8.0
) -> Dict[str, Any]:
    """
    Automated Retraining Decision Engine.
    Combines data drift flags with performance degradation to issue automated action flags.
    """
    reasons = []
    actions = []

    high_drift_count = drift_scorecard["high_drift_count"]
    drift_ratio = drift_scorecard["drift_ratio"]
    deltas = eval_results.get("deltas", {})
    delta_f1 = deltas.get("f1", 0.0)
    delta_acc = deltas.get("accuracy", 0.0)

    # Condition 1: Critical Feature Drift
    if high_drift_count >= 2:
        reasons.append(f"Critical statistical drift detected in {high_drift_count} high-impact features (PSI >= 0.25).")

    # Condition 2: Broad Covariate Drift
    if drift_ratio >= 40.0:
        reasons.append(f"{drift_ratio}% of features have drifted significantly from baseline distribution.")

    # Condition 3: Ground-Truth Performance SLA Breach
    if delta_f1 < f1_sla_threshold:
        reasons.append(f"Model F1 score degraded by {delta_f1}%, breaching the {f1_sla_threshold}% SLA threshold.")
    if delta_acc < acc_sla_threshold:
        reasons.append(f"Model Accuracy degraded by {delta_acc}%, breaching the {acc_sla_threshold}% SLA threshold.")

    # Decision Matrix
    if len(reasons) >= 2 or delta_f1 < f1_sla_threshold:
        decision = "TRIGGER_RETRAINING"
        status_label = "🔴 RETRAINING REQUIRED"
        priority = "CRITICAL"
        actions = [
            "Dispatch automated trigger event to OCI Events service.",
            "Invoke OCI Data Science Pipeline to retrain model with combined baseline and incoming batch.",
            "Route high-risk predictions through human-in-the-loop fallback workflow.",
            "Notify MLOps on-call team via OCI Notification Topic (Slack/PagerDuty)."
        ]
    elif len(reasons) == 1 or drift_scorecard["overall_status"] == "MODERATE_DRIFT":
        decision = "WARNING_INVESTIGATE"
        status_label = "🟡 WARNING / REVIEW"
        priority = "MEDIUM"
        actions = [
            "Log warning telemetry to OCI Logging Service.",
            "Trigger automated data quality inspection job on upstream ingestion bucket.",
            "Schedule shadow evaluation run with next scheduled batch."
        ]
    else:
        decision = "PASS_HEALTHY"
        status_label = "🟢 PRODUCTION HEALTHY"
        priority = "LOW"
        actions = [
            "Model operating within acceptable statistical tolerances.",
            "No retraining required; continue active real-time scoring."
        ]

    return {
        "decision": decision,
        "status_label": status_label,
        "priority": priority,
        "reasons": reasons if reasons else ["All statistical and performance metrics within expected thresholds."],
        "recommended_actions": actions,
    }


def record_monitoring_run(
    batch_name: str,
    model_metadata: Dict[str, Any],
    drift_scorecard: Dict[str, Any],
    eval_results: Dict[str, Any],
    retrain_decision: Dict[str, Any]
) -> Dict[str, Any]:
    """Package a full monitoring execution record and persist to audit trail."""
    run_id = f"RUN-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4].upper()}"

    record = {
        "run_id": run_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "batch_name": batch_name,
        "model_version": model_metadata.get("version", "1.0.0"),
        "model_name": model_metadata.get("model_name", "Customer_Churn_Risk_Predictor"),
        "dataset_rows": eval_results.get("n_samples", 0),
        "data_drift_status": drift_scorecard["overall_status"],
        "drifted_features_count": drift_scorecard["drifted_count"],
        "total_features": drift_scorecard["total_features"],
        "metrics": eval_results.get("metrics", {}),
        "deltas": eval_results.get("deltas", {}),
        "decision": retrain_decision["decision"],
        "priority": retrain_decision["priority"],
        "reasons": retrain_decision["reasons"],
        "oci_event_payload": {
            "eventType": "com.oraclecloud.datascience.modelmonitoring.driftalert",
            "cloudEventsVersion": "0.1",
            "source": f"ocid1.datasciencemonitor.oc1.iad.{run_id}",
            "data": {
                "modelId": model_metadata.get("oci_model_catalog_ocid", "ocid1.datasciencemodel..."),
                "modelVersion": model_metadata.get("version", "1.0.0"),
                "status": retrain_decision["decision"],
                "driftRatio": drift_scorecard["drift_ratio"],
                "deltaF1": eval_results.get("deltas", {}).get("f1", 0.0),
                "action": "trigger_pipeline_run" if retrain_decision["decision"] == "TRIGGER_RETRAINING" else "none"
            }
        }
    }

    save_audit_run(record)
    return record


def generate_markdown_report(
    batch_name: str,
    model_metadata: Dict[str, Any],
    drift_scorecard: Dict[str, Any],
    eval_results: Dict[str, Any],
    decision: Dict[str, Any]
) -> str:
    """Generate an executive-ready Markdown audit and monitoring report."""
    metrics = eval_results.get("metrics", {})
    deltas = eval_results.get("deltas", {})
    ref_metrics = model_metadata.get("reference_metrics", {})

    report = f"""# 🛡️ MLOps Executive Monitoring & Drift Audit Report

**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Model Name:** `{model_metadata.get('model_name')}` (v{model_metadata.get('version')})  
**Evaluated Batch:** `{batch_name}` ({eval_results.get('n_samples')} records)  
**OCI Catalog OCID:** `{model_metadata.get('oci_model_catalog_ocid')}`  

---

## 1. Executive Summary & Automated Decision

- **Monitoring Status:** {decision['status_label']}
- **Action Priority:** **{decision['priority']}**
- **Decision Code:** `{decision['decision']}`

### Primary Finding & Rationale
"""
    for r in decision["reasons"]:
        report += f"- ⚠️ {r}\n"

    report += "\n### Recommended Automated Actions\n"
    for a in decision["recommended_actions"]:
        report += f"- [ ] {a}\n"

    report += f"""
---

## 2. Statistical Data Drift Breakdown

- **Total Monitored Features:** {drift_scorecard['total_features']}
- **Features with Significant Drift:** {drift_scorecard['drifted_count']} ({drift_scorecard['drift_ratio']}%)
- **Critical Severity Features (PSI >= 0.25):** {drift_scorecard['high_drift_count']}
- **Statistical Methodologies:** Two-sample Kolmogorov-Smirnov test (continuous), Chi-square contingency test (categorical), Population Stability Index (PSI).

### Feature Drift Scorecard

| Feature | Type | Status | Severity | Test Stat | p-value | PSI |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for _, row in drift_scorecard["summary_table"].iterrows():
        report += f"| {row['Feature']} | {row['Type']} | {row['Drift Detected']} | {row['Severity']} | {row['Test Statistic']} | {row['p-value']} | {row['PSI']} |\n"

    report += f"""
---

## 3. Model Performance & Ground Truth Degradation

| Metric | Baseline (v1.0.0) | Production ({batch_name}) | Delta (\\Delta) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | {ref_metrics.get('accuracy', 0.0):.4f} | {metrics.get('accuracy', 0.0):.4f} | {deltas.get('accuracy', 0.0):+.2f}% | {'🚨 Breach' if deltas.get('accuracy', 0.0) < -8.0 else '✅ Stable'} |
| **Precision** | {ref_metrics.get('precision', 0.0):.4f} | {metrics.get('precision', 0.0):.4f} | - | - |
| **Recall** | {ref_metrics.get('recall', 0.0):.4f} | {metrics.get('recall', 0.0):.4f} | - | - |
| **F1 Score** | {ref_metrics.get('f1', 0.0):.4f} | {metrics.get('f1', 0.0):.4f} | {deltas.get('f1', 0.0):+.2f}% | {'🚨 Breach' if deltas.get('f1', 0.0) < -10.0 else '✅ Stable'} |
| **ROC-AUC** | {ref_metrics.get('roc_auc', 0.0):.4f} | {metrics.get('roc_auc', 0.0):.4f} | {deltas.get('roc_auc', 0.0):+.2f}% | {'⚠️ Drift' if deltas.get('roc_auc', 0.0) < -8.0 else '✅ Stable'} |

---

## 4. OCI Cloud Architecture Reference

This monitoring run maps to the following Oracle Cloud Infrastructure services:
- **Telemetry & Drift Engine:** OCI Data Science Model Monitoring Service
- **Model Registry & Governance:** OCI Data Science Model Catalog
- **Event Dispatch:** OCI Events Service (`com.oraclecloud.datascience.modelmonitoring`)
- **Automated Remediation:** OCI Functions $\\rightarrow$ OCI Data Science Retraining Pipeline
"""
    return report
