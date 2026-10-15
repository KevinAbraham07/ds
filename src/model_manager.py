"""
Model Management & Performance Evaluation Engine.
Handles model training, artifact persistence, inference, metric tracking,
confusion matrices, ROC curves, and performance degradation calculations.
"""

import os
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
)

from src.data_loader import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_dataset,
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "baseline_model.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")


def train_and_save_baseline_model() -> Tuple[Pipeline, Dict[str, Any]]:
    """Train a scikit-learn pipeline on reference data and save artifact + metadata."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = load_dataset("reference_data.csv")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Preprocessing pipelines
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    clf = GradientBoostingClassifier(
        n_estimators=120,
        learning_rate=0.08,
        max_depth=4,
        random_state=42
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf),
    ])

    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)

    # Save artifact
    joblib.dump(pipeline, MODEL_PATH)

    # Compute reference evaluation metrics on validation set (best practice)
    y_pred_val = pipeline.predict(X_val)
    y_prob_val = pipeline.predict_proba(X_val)[:, 1]

    ref_metrics = {
        "accuracy": round(float(accuracy_score(y_val, y_pred_val)), 4),
        "precision": round(float(precision_score(y_val, y_pred_val, zero_division=0)), 4),
        "recall": round(float(recall_score(y_val, y_pred_val, zero_division=0)), 4),
        "f1": round(float(f1_score(y_val, y_pred_val, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_val, y_prob_val)), 4),
        "brier_score": round(float(brier_score_loss(y_val, y_prob_val)), 4),
    }

    # Extract transformed feature names and importances
    ohe_cols = (
        pipeline.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(CATEGORICAL_FEATURES)
    )
    all_transformed_cols = list(NUMERICAL_FEATURES) + list(ohe_cols)
    importances = pipeline.named_steps["classifier"].feature_importances_

    importance_dict = {
        col: round(float(imp), 4)
        for col, imp in zip(all_transformed_cols, importances)
    }
    sorted_importance = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))

    with open(MODEL_PATH, "rb") as f:
        artifact_hash = hashlib.sha256(f.read()).hexdigest()[:16]

    metadata = {
        "model_name": "Customer_Churn_Risk_Predictor",
        "version": "1.0.0",
        "status": "PRODUCTION",
        "algorithm": "GradientBoostingClassifier",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "framework": "scikit-learn 1.9",
        "target": TARGET_COLUMN,
        "input_features": FEATURE_COLUMNS,
        "hyperparameters": {
            "n_estimators": 120,
            "learning_rate": 0.08,
            "max_depth": 4,
            "random_state": 42
        },
        "reference_dataset_rows": len(df),
        "reference_metrics": ref_metrics,
        "feature_importances": sorted_importance,
        "artifact_hash": artifact_hash,
        "oci_model_catalog_ocid": "ocid1.datasciencemodel.oc1.iad.amaaaaaa72..."
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    return pipeline, metadata


def get_model() -> Tuple[Pipeline, Dict[str, Any]]:
    """Load model artifact and metadata from disk, building if necessary."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
        return train_and_save_baseline_model()

    pipeline = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    return pipeline, metadata


def evaluate_batch(
    pipeline: Pipeline,
    batch_df: pd.DataFrame,
    ref_metrics: Dict[str, float]
) -> Dict[str, Any]:
    """
    Run model predictions on a production batch and compute performance degradation vs baseline.
    """
    X = batch_df[FEATURE_COLUMNS]
    y_true = batch_df[TARGET_COLUMN] if TARGET_COLUMN in batch_df.columns else None

    # Handle inferences
    y_pred = pipeline.predict(X)
    y_prob = pipeline.predict_proba(X)[:, 1]

    results = {
        "n_samples": len(batch_df),
        "predicted_positive_rate": round(float(np.mean(y_pred)) * 100, 2),
        "avg_confidence": round(float(np.mean(np.max(pipeline.predict_proba(X), axis=1))) * 100, 2),
        "has_ground_truth": y_true is not None,
        "predictions": y_pred,
        "probabilities": y_prob
    }

    if y_true is not None:
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = 0.5
        brier = float(brier_score_loss(y_true, y_prob))

        cm = confusion_matrix(y_true, y_pred)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)

        # Calculate degradation deltas
        delta_acc = round((acc - ref_metrics["accuracy"]) * 100, 2)
        delta_f1 = round((f1 - ref_metrics["f1"]) * 100, 2)
        delta_auc = round((auc - ref_metrics["roc_auc"]) * 100, 2)

        # SLA breach detection
        sla_breach = bool(delta_f1 < -10.0 or delta_acc < -8.0)

        results.update({
            "metrics": {
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "roc_auc": round(auc, 4),
                "brier_score": round(brier, 4),
            },
            "deltas": {
                "accuracy": delta_acc,
                "f1": delta_f1,
                "roc_auc": delta_auc,
            },
            "confusion_matrix": cm.tolist(),
            "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
            "pr_curve": {"precision": precision_curve.tolist(), "recall": recall_curve.tolist()},
            "sla_breach": sla_breach,
        })

    return results
