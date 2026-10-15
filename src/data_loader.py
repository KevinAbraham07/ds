"""
Dataset Management Module for ML Model Monitoring & Drift Detection.
Handles dataset synthesis, reference baseline generation, and realistic production batch simulation.
"""

import os
from typing import Dict, Tuple
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

NUMERICAL_FEATURES = [
    "age",
    "tenure_months",
    "monthly_charges",
    "total_spend",
    "credit_score",
    "support_calls",
    "payment_delay_days",
]

CATEGORICAL_FEATURES = [
    "contract_type",
    "internet_service",
    "payment_method",
]

FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "churn"


def calculate_churn_prob(
    df: pd.DataFrame,
    concept_shift: bool = False
) -> np.ndarray:
    """Calculate realistic ground-truth churn probability based on features."""
    z = (
        -1.2
        + 0.03 * (df["monthly_charges"] - 65.0)
        - 0.04 * (df["tenure_months"] - 30.0)
        - 0.005 * (df["credit_score"] - 650.0)
        + 0.45 * (df["support_calls"] - 1.5)
        + 0.08 * (df["payment_delay_days"] - 3.0)
        - 0.015 * (df["age"] - 42.0)
    )

    # Categorical effects
    contract_risk = df["contract_type"].map({
        "Month-to-Month": 0.6,
        "One-Year": -0.4,
        "Two-Year": -0.9
    }).fillna(0)

    service_risk = df["internet_service"].map({
        "Fiber Optic": 0.3,
        "DSL": -0.1,
        "No Internet": -0.3
    }).fillna(0)

    payment_risk = df["payment_method"].map({
        "Electronic Check": 0.4,
        "Mailed Check": 0.1,
        "Bank Transfer": -0.2,
        "Credit Card": -0.3
    }).fillna(0)

    z = z + contract_risk + service_risk + payment_risk

    if concept_shift:
        # Market shock: competitors target loyal, high-value users
        # Even two-year and high credit score customers churn rapidly
        z += 1.4 * (df["credit_score"] > 680).astype(float)
        z += 1.8 * (df["contract_type"] == "Two-Year").astype(float)

    # Sigmoid function
    prob = 1.0 / (1.0 + np.exp(-np.clip(z, -10, 10)))
    return prob


def generate_reference_data(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate baseline reference dataset representing original training distribution."""
    rng = np.random.default_rng(seed)

    age = rng.normal(loc=44.0, scale=12.0, size=n_samples).clip(18, 80).astype(int)
    tenure_months = rng.gamma(shape=3.0, scale=10.0, size=n_samples).clip(1, 72).astype(int)
    monthly_charges = rng.normal(loc=68.0, scale=18.0, size=n_samples).clip(20.0, 125.0)
    total_spend = (tenure_months * monthly_charges * rng.uniform(0.95, 1.05, size=n_samples)).round(2)
    credit_score = rng.normal(loc=675.0, scale=60.0, size=n_samples).clip(450, 850).astype(int)
    support_calls = rng.poisson(lam=1.6, size=n_samples).clip(0, 10)
    payment_delay_days = rng.exponential(scale=3.5, size=n_samples).clip(0, 30).astype(int)

    contract_type = rng.choice(
        ["Month-to-Month", "One-Year", "Two-Year"],
        p=[0.55, 0.25, 0.20],
        size=n_samples
    )
    internet_service = rng.choice(
        ["Fiber Optic", "DSL", "No Internet"],
        p=[0.45, 0.35, 0.20],
        size=n_samples
    )
    payment_method = rng.choice(
        ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"],
        p=[0.35, 0.20, 0.25, 0.20],
        size=n_samples
    )

    df = pd.DataFrame({
        "age": age,
        "tenure_months": tenure_months,
        "monthly_charges": np.round(monthly_charges, 2),
        "total_spend": total_spend,
        "credit_score": credit_score,
        "support_calls": support_calls,
        "payment_delay_days": payment_delay_days,
        "contract_type": contract_type,
        "internet_service": internet_service,
        "payment_method": payment_method,
    })

    probs = calculate_churn_prob(df, concept_shift=False)
    df[TARGET_COLUMN] = (rng.uniform(0, 1, size=n_samples) < probs).astype(int)
    return df


def generate_batch_1_stable(n_samples: int = 500, seed: int = 101) -> pd.DataFrame:
    """Production Batch 1: In-distribution with natural operational variance."""
    rng = np.random.default_rng(seed)

    age = rng.normal(loc=44.2, scale=12.1, size=n_samples).clip(18, 80).astype(int)
    tenure_months = rng.gamma(shape=3.0, scale=10.2, size=n_samples).clip(1, 72).astype(int)
    monthly_charges = rng.normal(loc=68.5, scale=17.8, size=n_samples).clip(20.0, 125.0)
    total_spend = (tenure_months * monthly_charges * rng.uniform(0.95, 1.05, size=n_samples)).round(2)
    credit_score = rng.normal(loc=673.0, scale=59.0, size=n_samples).clip(450, 850).astype(int)
    support_calls = rng.poisson(lam=1.65, size=n_samples).clip(0, 10)
    payment_delay_days = rng.exponential(scale=3.6, size=n_samples).clip(0, 30).astype(int)

    contract_type = rng.choice(
        ["Month-to-Month", "One-Year", "Two-Year"],
        p=[0.54, 0.26, 0.20],
        size=n_samples
    )
    internet_service = rng.choice(
        ["Fiber Optic", "DSL", "No Internet"],
        p=[0.46, 0.34, 0.20],
        size=n_samples
    )
    payment_method = rng.choice(
        ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"],
        p=[0.34, 0.21, 0.25, 0.20],
        size=n_samples
    )

    df = pd.DataFrame({
        "age": age,
        "tenure_months": tenure_months,
        "monthly_charges": np.round(monthly_charges, 2),
        "total_spend": total_spend,
        "credit_score": credit_score,
        "support_calls": support_calls,
        "payment_delay_days": payment_delay_days,
        "contract_type": contract_type,
        "internet_service": internet_service,
        "payment_method": payment_method,
    })

    probs = calculate_churn_prob(df, concept_shift=False)
    df[TARGET_COLUMN] = (rng.uniform(0, 1, size=n_samples) < probs).astype(int)
    return df


def generate_batch_2_drift(n_samples: int = 500, seed: int = 202) -> pd.DataFrame:
    """Production Batch 2: Moderate Covariate Drift.
    Younger demographic acquired from digital campaign with lower tenure and higher monthly fees.
    """
    rng = np.random.default_rng(seed)

    age = rng.normal(loc=31.0, scale=8.0, size=n_samples).clip(18, 80).astype(int)  # Shifted younger
    tenure_months = rng.exponential(scale=14.0, size=n_samples).clip(1, 72).astype(int)  # Lower tenure
    monthly_charges = rng.normal(loc=84.0, scale=15.0, size=n_samples).clip(25.0, 135.0)  # Higher fees
    total_spend = (tenure_months * monthly_charges * rng.uniform(0.95, 1.05, size=n_samples)).round(2)
    credit_score = rng.normal(loc=640.0, scale=65.0, size=n_samples).clip(450, 850).astype(int)
    support_calls = rng.poisson(lam=2.4, size=n_samples).clip(0, 10)  # Slightly higher
    payment_delay_days = rng.exponential(scale=5.0, size=n_samples).clip(0, 30).astype(int)

    contract_type = rng.choice(
        ["Month-to-Month", "One-Year", "Two-Year"],
        p=[0.75, 0.18, 0.07],  # Heavy month-to-month
        size=n_samples
    )
    internet_service = rng.choice(
        ["Fiber Optic", "DSL", "No Internet"],
        p=[0.70, 0.20, 0.10],  # Heavy Fiber Optic
        size=n_samples
    )
    payment_method = rng.choice(
        ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"],
        p=[0.52, 0.08, 0.20, 0.20],
        size=n_samples
    )

    df = pd.DataFrame({
        "age": age,
        "tenure_months": tenure_months,
        "monthly_charges": np.round(monthly_charges, 2),
        "total_spend": total_spend,
        "credit_score": credit_score,
        "support_calls": support_calls,
        "payment_delay_days": payment_delay_days,
        "contract_type": contract_type,
        "internet_service": internet_service,
        "payment_method": payment_method,
    })

    probs = calculate_churn_prob(df, concept_shift=False)
    df[TARGET_COLUMN] = (rng.uniform(0, 1, size=n_samples) < probs).astype(int)
    return df


def generate_batch_3_severe(n_samples: int = 500, seed: int = 303) -> pd.DataFrame:
    """Production Batch 3: Severe Drift & Data Quality Anomalies.
    Extreme support call surges, payment delays, credit score drops, and missing values.
    """
    rng = np.random.default_rng(seed)

    age = rng.normal(loc=38.0, scale=14.0, size=n_samples).clip(18, 80).astype(int)
    tenure_months = rng.uniform(1, 45, size=n_samples).astype(int)
    monthly_charges = rng.normal(loc=95.0, scale=22.0, size=n_samples).clip(20.0, 150.0)
    total_spend = (tenure_months * monthly_charges * rng.uniform(0.9, 1.1, size=n_samples)).round(2)
    credit_score = rng.normal(loc=550.0, scale=80.0, size=n_samples).clip(400, 820).astype(int)  # Severely impaired
    support_calls = rng.poisson(lam=5.5, size=n_samples).clip(0, 12)  # Major service crisis
    payment_delay_days = rng.normal(loc=16.0, scale=7.0, size=n_samples).clip(0, 30).astype(int)

    contract_type = rng.choice(
        ["Month-to-Month", "One-Year", "Two-Year"],
        p=[0.85, 0.10, 0.05],
        size=n_samples
    )
    internet_service = rng.choice(
        ["Fiber Optic", "DSL", "No Internet"],
        p=[0.60, 0.35, 0.05],
        size=n_samples
    )
    payment_method = rng.choice(
        ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"],
        p=[0.60, 0.20, 0.10, 0.10],
        size=n_samples
    )

    df = pd.DataFrame({
        "age": age,
        "tenure_months": tenure_months,
        "monthly_charges": np.round(monthly_charges, 2),
        "total_spend": total_spend,
        "credit_score": credit_score,
        "support_calls": support_calls,
        "payment_delay_days": payment_delay_days,
        "contract_type": contract_type,
        "internet_service": internet_service,
        "payment_method": payment_method,
    })

    probs = calculate_churn_prob(df, concept_shift=False)
    df[TARGET_COLUMN] = (rng.uniform(0, 1, size=n_samples) < probs).astype(int)

    # Inject realistic real-world data quality anomalies (e.g. 3% missing values in payment_delay_days and credit_score)
    missing_idx_1 = rng.choice(n_samples, size=int(0.04 * n_samples), replace=False)
    missing_idx_2 = rng.choice(n_samples, size=int(0.03 * n_samples), replace=False)
    df.loc[missing_idx_1, "payment_delay_days"] = np.nan
    df.loc[missing_idx_2, "credit_score"] = np.nan

    return df


def generate_batch_4_concept(n_samples: int = 500, seed: int = 404) -> pd.DataFrame:
    """Production Batch 4: Concept Drift.
    Feature distributions look reasonably normal, but relationships between features and churn inverted.
    """
    rng = np.random.default_rng(seed)

    age = rng.normal(loc=44.5, scale=11.8, size=n_samples).clip(18, 80).astype(int)
    tenure_months = rng.gamma(shape=3.1, scale=9.8, size=n_samples).clip(1, 72).astype(int)
    monthly_charges = rng.normal(loc=69.0, scale=17.5, size=n_samples).clip(20.0, 125.0)
    total_spend = (tenure_months * monthly_charges * rng.uniform(0.95, 1.05, size=n_samples)).round(2)
    credit_score = rng.normal(loc=678.0, scale=58.0, size=n_samples).clip(450, 850).astype(int)
    support_calls = rng.poisson(lam=1.7, size=n_samples).clip(0, 10)
    payment_delay_days = rng.exponential(scale=3.4, size=n_samples).clip(0, 30).astype(int)

    contract_type = rng.choice(
        ["Month-to-Month", "One-Year", "Two-Year"],
        p=[0.53, 0.27, 0.20],
        size=n_samples
    )
    internet_service = rng.choice(
        ["Fiber Optic", "DSL", "No Internet"],
        p=[0.45, 0.35, 0.20],
        size=n_samples
    )
    payment_method = rng.choice(
        ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"],
        p=[0.33, 0.22, 0.25, 0.20],
        size=n_samples
    )

    df = pd.DataFrame({
        "age": age,
        "tenure_months": tenure_months,
        "monthly_charges": np.round(monthly_charges, 2),
        "total_spend": total_spend,
        "credit_score": credit_score,
        "support_calls": support_calls,
        "payment_delay_days": payment_delay_days,
        "contract_type": contract_type,
        "internet_service": internet_service,
        "payment_method": payment_method,
    })

    # Concept shift true
    probs = calculate_churn_prob(df, concept_shift=True)
    df[TARGET_COLUMN] = (rng.uniform(0, 1, size=n_samples) < probs).astype(int)
    return df


def initialize_datasets_on_disk() -> Dict[str, str]:
    """Ensure data directory and all dataset files exist."""
    os.makedirs(DATA_DIR, exist_ok=True)

    datasets = {
        "Reference Baseline (Train)": ("reference_data.csv", generate_reference_data),
        "Batch 1 - Stable Production": ("prod_batch_1_stable.csv", generate_batch_1_stable),
        "Batch 2 - Moderate Feature Drift": ("prod_batch_2_drift.csv", generate_batch_2_drift),
        "Batch 3 - Severe Drift & Anomalies": ("prod_batch_3_severe.csv", generate_batch_3_severe),
        "Batch 4 - Pure Concept Drift": ("prod_batch_4_concept.csv", generate_batch_4_concept),
    }

    file_paths = {}
    for name, (fname, generator) in datasets.items():
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            df = generator()
            df.to_csv(fpath, index=False)
        file_paths[name] = fpath

    return file_paths


def load_dataset(filename: str) -> pd.DataFrame:
    """Load a specific dataset from data directory."""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        initialize_datasets_on_disk()
    return pd.read_csv(path)


def get_data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Generate comprehensive summary profiling and data quality statistics."""
    records = []
    for col in df.columns:
        col_type = "Categorical" if col in CATEGORICAL_FEATURES else ("Target" if col == TARGET_COLUMN else "Numerical")
        n_missing = int(df[col].isna().sum())
        pct_missing = round((n_missing / len(df)) * 100, 2)
        n_unique = int(df[col].nunique(dropna=True))

        if col in NUMERICAL_FEATURES:
            mean_val = round(float(df[col].mean()), 2)
            std_val = round(float(df[col].std()), 2)
            min_val = round(float(df[col].min()), 2)
            median_val = round(float(df[col].median()), 2)
            max_val = round(float(df[col].max()), 2)
        else:
            mean_val = np.nan
            std_val = np.nan
            min_val = np.nan
            median_val = np.nan
            max_val = np.nan

        records.append({
            "Feature": col,
            "Type": col_type,
            "Missing Count": n_missing,
            "Missing %": pct_missing,
            "Unique Values": n_unique,
            "Mean": mean_val,
            "Std Dev": std_val,
            "Min": min_val,
            "Median": median_val,
            "Max": max_val,
        })
    return pd.DataFrame(records)
