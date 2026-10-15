"""
Oracle Cloud Infrastructure (OCI) Architecture Mapping Module.
Provides conceptual blueprints, service equivalence mappings, and architectural explanations
linking each project component directly to OCI AI & Data Science Cloud services.
"""

from typing import List, Dict, Any


def get_oci_service_mappings() -> List[Dict[str, Any]]:
    """Return comprehensive mapping table between local implementation and OCI services."""
    return [
        {
            "Local Component": "Dataset Manager (Reference & Prod Batches)",
            "Course Topic": "Machine Learning Lifecycle / Data Management",
            "OCI Service": "OCI Object Storage & OCI Data Flow",
            "Role in OCI": (
                "Reference datasets and streaming batch inference logs are stored in OCI Object Storage buckets. "
                "OCI Data Flow (managed Apache Spark) can be scheduled to preprocess incoming parquet batches."
            ),
            "Icon": "🗄️"
        },
        {
            "Local Component": "Model Artifact (`baseline_model.joblib` & metadata)",
            "Course Topic": "MLOps Practices / Model Versioning",
            "OCI Service": "OCI Data Science Model Catalog",
            "Role in OCI": (
                "The Model Catalog stores versioned model artifacts, conda environment runtimes, input schema, "
                "provenance metadata, and evaluation metrics with unique OCIDs for enterprise governance."
            ),
            "Icon": "📦"
        },
        {
            "Local Component": "Statistical Drift Engine (KS-Test, PSI, Chi-Square)",
            "Course Topic": "Monitoring & Drift Detection",
            "OCI Service": "OCI Data Science Model Monitoring",
            "Role in OCI": (
                "OCI Model Monitoring automatically schedules statistical drift evaluations comparing live model "
                "inference logs against the baseline dataset registered in the Model Catalog, calculating feature drift scores."
            ),
            "Icon": "📊"
        },
        {
            "Local Component": "Performance Degradation & Feedback Loop",
            "Course Topic": "MLOps Practices / Performance Tracking",
            "OCI Service": "OCI Monitoring & Oracle Cloud Logging Service",
            "Role in OCI": (
                "Emits custom performance metrics (Accuracy, F1, Loss) to OCI Monitoring MQL (Monitoring Query Language). "
                "Logs every prediction and audit trail to centralized OCI Logging with retention policies."
            ),
            "Icon": "📈"
        },
        {
            "Local Component": "Automated Retraining Decision Engine",
            "Course Topic": "Workflow Automation / Event-Driven MLOps",
            "OCI Service": "OCI Events + OCI Functions + OCI Notifications",
            "Role in OCI": (
                "When Model Monitoring emits an alarm rule, OCI Events captures the event pattern and triggers a serverless "
                "OCI Function. The Function sends a notification via OCI Notifications and launches an OCI Pipeline."
            ),
            "Icon": "⚡"
        },
        {
            "Local Component": "Retraining Execution Pipeline",
            "Course Topic": "Automated ML Lifecycle",
            "OCI Service": "OCI Data Science Pipelines",
            "Role in OCI": (
                "A managed, multi-step pipeline DAG that pulls newly labeled production data, retrains the model candidate, "
                "runs integration validation tests, and registers v1.1.0 into the Model Catalog."
            ),
            "Icon": "🔄"
        },
        {
            "Local Component": "Interactive Monitoring UI (Streamlit)",
            "Course Topic": "Workspace Design & Setup",
            "OCI Service": "OCI Container Instances / OKE / Oracle Analytics Cloud",
            "Role in OCI": (
                "The dashboard can be containerized with Docker and deployed serverless on OCI Container Instances or "
                "Oracle Container Engine for Kubernetes (OKE), or connected to Oracle Analytics Cloud (OAC)."
            ),
            "Icon": "🖥️"
        }
    ]


def get_oci_architecture_workflow() -> str:
    """Return Mermaid diagram definition of OCI End-to-End MLOps Architecture."""
    return """flowchart TD
    subgraph Data Layer ["1. Storage & Ingestion (OCI Object Storage)"]
        REF["Reference Baseline Dataset<br/>(oci://bucket/training/ref.parquet)"]
        PROD["Production Inference Stream<br/>(oci://bucket/prod/batch_2026.parquet)"]
    end

    subgraph Monitoring Layer ["2. OCI Data Science Model Monitoring"]
        MODEL["OCI Model Catalog<br/>Model v1.0.0 (OCID)"]
        DRIFT_ENG["OCI Model Monitoring Engine<br/>• KS-Test & PSI Calculation<br/>• Feature Distribution Drift<br/>• Ground Truth Metric Delta"]
    end

    subgraph Telemetry ["3. OCI Observability & Management"]
        LOGS["OCI Logging Service<br/>Audit & Prediction Logs"]
        METRICS["OCI Monitoring Service<br/>Alarms & MQL Metrics"]
    end

    subgraph Automation Layer ["4. Event-Driven Retraining Orchestration"]
        EVENTS["OCI Events Service<br/>Rule: ModelDriftThresholdExceeded"]
        FUNC["OCI Functions (Serverless)<br/>Parse Payload & Orchestrate"]
        NOTIF["OCI Notifications (ONS)<br/>Slack / PagerDuty / Email"]
        PIPELINE["OCI Data Science Pipeline<br/>1. Ingest New Data<br/>2. Retrain Candidate Model<br/>3. Evaluate & Register v1.1.0"]
    end

    REF --> DRIFT_ENG
    PROD --> DRIFT_ENG
    MODEL --> DRIFT_ENG
    DRIFT_ENG --> LOGS
    DRIFT_ENG --> METRICS
    METRICS -->|Alarm Triggered| EVENTS
    EVENTS --> FUNC
    FUNC --> NOTIF
    FUNC --> PIPELINE
    PIPELINE -->|Register New Artifact| MODEL
"""
