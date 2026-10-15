"""
Unit tests for ML Model Monitoring & Drift Detection Dashboard.
Verifies dataset generation, statistical tests (KS, PSI, Chi-Square), model inference, and MLOps triggers.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import (
    initialize_datasets_on_disk,
    load_dataset,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    get_data_quality_report,
)
from src.drift_detector import (
    calculate_psi,
    test_numerical_drift,
    test_categorical_drift,
    compute_drift_scorecard,
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
)


class TestMLModelMonitoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        initialize_datasets_on_disk()
        cls.ref_df = load_dataset("reference_data.csv")
        cls.batch1 = load_dataset("prod_batch_1_stable.csv")
        cls.batch2 = load_dataset("prod_batch_2_drift.csv")
        cls.batch3 = load_dataset("prod_batch_3_severe.csv")
        cls.batch4 = load_dataset("prod_batch_4_concept.csv")
        cls.pipeline, cls.metadata = get_model()

    def test_datasets_exist_and_non_empty(self):
        self.assertGreater(len(self.ref_df), 1000)
        self.assertGreater(len(self.batch1), 300)
        self.assertGreater(len(self.batch2), 300)
        self.assertGreater(len(self.batch3), 300)
        self.assertGreater(len(self.batch4), 300)

        for col in FEATURE_COLUMNS:
            self.assertIn(col, self.ref_df.columns)
        self.assertIn(TARGET_COLUMN, self.ref_df.columns)

    def test_data_quality_report(self):
        report = get_data_quality_report(self.ref_df)
        self.assertEqual(len(report), len(self.ref_df.columns))
        self.assertTrue("Missing Count" in report.columns)
        self.assertTrue("Unique Values" in report.columns)

    def test_psi_identical_distributions(self):
        rng = np.random.default_rng(42)
        arr = rng.normal(50, 10, 1000)
        psi_val, breakdown = calculate_psi(arr, arr)
        self.assertAlmostEqual(psi_val, 0.0, places=2)

    def test_psi_drifted_distributions(self):
        rng = np.random.default_rng(42)
        ref_arr = rng.normal(50, 10, 1000)
        drift_arr = rng.normal(75, 10, 1000)  # Significant shift
        psi_val, breakdown = calculate_psi(ref_arr, drift_arr)
        self.assertGreater(psi_val, 0.25)  # Should clearly exceed 0.25

    def test_ks_test_stable_batch(self):
        res = test_numerical_drift(self.ref_df["monthly_charges"], self.batch1["monthly_charges"])
        self.assertIn(res["severity"], ["LOW", "MEDIUM"])

    def test_ks_test_drifted_batch(self):
        # In batch 2, age and tenure were shifted
        res = test_numerical_drift(self.ref_df["age"], self.batch2["age"])
        self.assertTrue(res["drift_detected"])
        self.assertLess(res["p_value"], 0.05)

    def test_categorical_drift(self):
        res = test_categorical_drift(self.ref_df["contract_type"], self.batch2["contract_type"])
        self.assertIn("stat", res)
        self.assertIn("p_value", res)

    def test_drift_scorecard(self):
        scorecard = compute_drift_scorecard(
            ref_df=self.ref_df,
            prod_df=self.batch2,
            features=FEATURE_COLUMNS,
            numerical_features=NUMERICAL_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
        )
        self.assertIn("overall_status", scorecard)
        self.assertIn("summary_table", scorecard)
        self.assertGreaterEqual(scorecard["drifted_count"], 1)

    def test_model_evaluation_and_metrics(self):
        eval_res = evaluate_batch(self.pipeline, self.batch1, self.metadata["reference_metrics"])
        self.assertIn("accuracy", eval_res["metrics"])
        self.assertIn("f1", eval_res["metrics"])
        self.assertIn("roc_auc", eval_res["metrics"])
        self.assertIn("confusion_matrix", eval_res)
        self.assertFalse(eval_res["sla_breach"])

    def test_retraining_trigger_on_severe_drift(self):
        scorecard_severe = compute_drift_scorecard(
            ref_df=self.ref_df,
            prod_df=self.batch3,
            features=FEATURE_COLUMNS,
            numerical_features=NUMERICAL_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
        )
        eval_res_severe = evaluate_batch(self.pipeline, self.batch3, self.metadata["reference_metrics"])
        decision = evaluate_retraining_trigger(scorecard_severe, eval_res_severe)
        # Batch 3 has severe drift + performance drop
        self.assertEqual(decision["decision"], "TRIGGER_RETRAINING")
        self.assertEqual(decision["priority"], "CRITICAL")

    def test_audit_run_persistence(self):
        scorecard = compute_drift_scorecard(
            ref_df=self.ref_df,
            prod_df=self.batch1,
            features=FEATURE_COLUMNS,
            numerical_features=NUMERICAL_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
        )
        eval_res = evaluate_batch(self.pipeline, self.batch1, self.metadata["reference_metrics"])
        decision = evaluate_retraining_trigger(scorecard, eval_res)

        record = record_monitoring_run(
            batch_name="Test Batch 1",
            model_metadata=self.metadata,
            drift_scorecard=scorecard,
            eval_results=eval_res,
            retrain_decision=decision,
        )
        self.assertTrue(record["run_id"].startswith("RUN-"))

        history = load_audit_history()
        self.assertGreaterEqual(len(history), 1)
        self.assertEqual(history[0]["run_id"], record["run_id"])


if __name__ == "__main__":
    unittest.main()
