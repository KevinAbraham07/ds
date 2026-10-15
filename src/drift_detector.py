"""
Statistical Data Drift Detection Module.
Calculates Kolmogorov-Smirnov (KS) tests, Population Stability Index (PSI),
Wasserstein Distance, and Chi-Square contingency tests to detect feature distribution shifts.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_buckets: int = 10,
    epsilon: float = 1e-4
) -> Tuple[float, pd.DataFrame]:
    """
    Calculate Population Stability Index (PSI) between reference (expected)
    and production (actual) feature distributions.
    
    Thresholds:
      PSI < 0.10: No significant drift
      0.10 <= PSI < 0.25: Moderate drift
      PSI >= 0.25: Significant drift
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0, pd.DataFrame()

    # Determine quantile bin edges based on expected (reference) distribution
    percentiles = np.linspace(0, 100, num_buckets + 1)
    bin_edges = np.percentile(expected, percentiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Remove non-unique bin edges if data has low cardinality
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) <= 2:
        bin_edges = np.array([-np.inf, np.median(expected), np.inf])

    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    expected_pct = (expected_counts / len(expected)) + epsilon
    actual_pct = (actual_counts / len(actual)) + epsilon

    # Normalize back to 1.0 after epsilon smoothing
    expected_pct = expected_pct / expected_pct.sum()
    actual_pct = actual_pct / actual_pct.sum()

    # Calculate PSI per bucket
    psi_buckets = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    total_psi = float(np.sum(psi_buckets))

    bucket_labels = [f"Bin {i+1}" for i in range(len(psi_buckets))]
    breakdown_df = pd.DataFrame({
        "Bucket": bucket_labels,
        "Reference_Pct": np.round(expected_pct * 100, 2),
        "Production_Pct": np.round(actual_pct * 100, 2),
        "PSI_Contribution": np.round(psi_buckets, 4)
    })

    return total_psi, breakdown_df


def test_numerical_drift(
    ref_col: pd.Series,
    prod_col: pd.Series,
    p_value_threshold: float = 0.05,
    psi_threshold: float = 0.15
) -> Dict[str, Any]:
    """Perform KS-test, Wasserstein distance, and PSI on a continuous numerical feature."""
    ref_clean = ref_col.dropna().values
    prod_clean = prod_col.dropna().values

    if len(ref_clean) == 0 or len(prod_clean) == 0:
        return {
            "test_type": "KS-Test & PSI",
            "stat": 0.0,
            "p_value": 1.0,
            "psi": 0.0,
            "wasserstein": 0.0,
            "drift_detected": False,
            "severity": "LOW",
            "message": "Insufficient clean data"
        }

    # Two-sample Kolmogorov-Smirnov test
    ks_res = stats.ks_2samp(ref_clean, prod_clean)
    ks_stat = float(ks_res.statistic)
    p_val = float(ks_res.pvalue)

    # Earth Mover's / Wasserstein Distance
    # Standardize by reference std to make scale invariant
    ref_std = np.std(ref_clean) if np.std(ref_clean) > 0 else 1.0
    w_dist = float(stats.wasserstein_distance(ref_clean, prod_clean) / ref_std)

    # PSI
    psi_val, breakdown_df = calculate_psi(ref_clean, prod_clean)

    # Drift decision logic
    is_drift = bool(p_val < p_value_threshold or psi_val >= psi_threshold)

    if psi_val >= 0.25 or (p_val < 0.001 and ks_stat > 0.25):
        severity = "HIGH"
    elif psi_val >= 0.10 or p_val < p_value_threshold:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    return {
        "test_type": "Kolmogorov-Smirnov & PSI",
        "stat": round(ks_stat, 4),
        "p_value": round(p_val, 5),
        "psi": round(psi_val, 4),
        "wasserstein": round(w_dist, 4),
        "drift_detected": is_drift,
        "severity": severity,
        "psi_breakdown": breakdown_df
    }


def test_categorical_drift(
    ref_col: pd.Series,
    prod_col: pd.Series,
    p_value_threshold: float = 0.05
) -> Dict[str, Any]:
    """Perform Chi-Square goodness-of-fit test on categorical feature distributions."""
    ref_clean = ref_col.dropna()
    prod_clean = prod_col.dropna()

    all_categories = sorted(list(set(ref_clean.unique()).union(set(prod_clean.unique()))))

    ref_counts = ref_clean.value_counts().reindex(all_categories, fill_value=0)
    prod_counts = prod_clean.value_counts().reindex(all_categories, fill_value=0)

    contingency = np.array([ref_counts.values, prod_counts.values])

    try:
        chi2_stat, p_val, dof, _ = stats.chi2_contingency(contingency)
    except Exception:
        chi2_stat, p_val = 0.0, 1.0

    # Categorical PSI calculation
    ref_pct = (ref_counts.values / len(ref_clean)) + 1e-4
    prod_pct = (prod_counts.values / len(prod_clean)) + 1e-4
    ref_pct = ref_pct / ref_pct.sum()
    prod_pct = prod_pct / prod_pct.sum()
    psi_val = float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)))

    is_drift = bool(p_val < p_value_threshold or psi_val >= 0.15)

    if psi_val >= 0.25 or p_val < 0.001:
        severity = "HIGH"
    elif psi_val >= 0.10 or p_val < p_value_threshold:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    breakdown_df = pd.DataFrame({
        "Category": all_categories,
        "Reference_Pct": np.round((ref_counts.values / len(ref_clean)) * 100, 2),
        "Production_Pct": np.round((prod_counts.values / len(prod_clean)) * 100, 2),
    })

    return {
        "test_type": "Chi-Square Contingency & PSI",
        "stat": round(float(chi2_stat), 4),
        "p_value": round(float(p_val), 5),
        "psi": round(psi_val, 4),
        "wasserstein": np.nan,
        "drift_detected": is_drift,
        "severity": severity,
        "breakdown": breakdown_df
    }


def compute_drift_scorecard(
    ref_df: pd.DataFrame,
    prod_df: pd.DataFrame,
    features: List[str],
    numerical_features: List[str],
    categorical_features: List[str],
    p_value_threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Run statistical tests across all features and generate a comprehensive drift scorecard.
    """
    results = {}
    rows = []
    drifted_count = 0
    high_drift_count = 0

    for feat in features:
        if feat in numerical_features:
            res = test_numerical_drift(ref_df[feat], prod_df[feat], p_value_threshold)
        elif feat in categorical_features:
            res = test_categorical_drift(ref_df[feat], prod_df[feat], p_value_threshold)
        else:
            continue

        results[feat] = res
        if res["drift_detected"]:
            drifted_count += 1
        if res["severity"] == "HIGH":
            high_drift_count += 1

        rows.append({
            "Feature": feat,
            "Type": "Numerical" if feat in numerical_features else "Categorical",
            "Drift Detected": "🚨 DRIFT" if res["drift_detected"] else "✅ STABLE",
            "Severity": res["severity"],
            "Test Statistic": res["stat"],
            "p-value": res["p_value"],
            "PSI": res["psi"],
            "Wasserstein": res["wasserstein"] if not np.isnan(res.get("wasserstein", np.nan)) else "-"
        })

    summary_df = pd.DataFrame(rows)
    total_features = len(features)
    drift_ratio = drifted_count / total_features if total_features > 0 else 0.0

    if high_drift_count >= 2 or drift_ratio >= 0.40:
        overall_status = "CRITICAL_DRIFT"
        status_color = "red"
        recommendation = "🚨 High data drift detected in multiple critical features. Model predictions are high-risk. Urgent review / retraining pipeline trigger required."
    elif drifted_count >= 1 or drift_ratio >= 0.15:
        overall_status = "MODERATE_DRIFT"
        status_color = "orange"
        recommendation = "⚠️ Moderate feature drift detected. Monitor upcoming batches closely and prepare shadow retraining candidate."
    else:
        overall_status = "HEALTHY"
        status_color = "green"
        recommendation = "✅ No significant data drift detected. Production data distribution is aligned with baseline training dataset."

    return {
        "overall_status": overall_status,
        "status_color": status_color,
        "recommendation": recommendation,
        "total_features": total_features,
        "drifted_count": drifted_count,
        "high_drift_count": high_drift_count,
        "drift_ratio": round(drift_ratio * 100, 1),
        "summary_table": summary_df,
        "feature_details": results
    }


def compute_ecdf(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Compute empirical cumulative distribution function (eCDF) for plotting KS distance."""
    clean = np.sort(series.dropna().values)
    if len(clean) == 0:
        return np.array([]), np.array([])
    y = np.arange(1, len(clean) + 1) / len(clean)
    return clean, y
