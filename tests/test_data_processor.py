"""
tests/test_data_processor.py – Unit tests for the data processing pipeline.
============================================================================
Tests each stage: load, impute, validate, feature engineering, expert rules, scaling.
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import ALL_ML_FEATURES, EXPERT_RULES, RISK_PILLARS, SECTOR_LOANS_COLUMNS
from utils.data_processor import (
    DATA_PATH,
    FEATURE_GROUPS,
    _create_efficiency_ratio,
    _create_risk_to_profit_ratio,
    _impute_missing_values,
    _scale_feature_group,
    _validate_features,
    evaluate_expert_rules,
    load_data,
    process_data,
)


# ── Fixtures ─────────────────────────────────────────────────────────

def _make_sample_df(n: int = 20) -> pd.DataFrame:
    """Create a minimal DataFrame with all required columns for testing."""
    rng = np.random.default_rng(42)
    data = {
        "bank_id": [f"BANK_{i:03d}" for i in range(n)],
        "period": ["2024-Q1"] * n,
        "bank_type": rng.choice(["large", "medium", "small"], n).tolist(),
        "region": rng.choice(["north", "south", "central"], n).tolist(),
        "external_credit_rating": rng.choice(["A", "BBB", "BB"], n).tolist(),
        "stability": rng.choice(["high", "medium", "low"], n).tolist(),
        "total_assets": rng.uniform(50000, 200000, n),
        "total_loans": rng.uniform(30000, 150000, n),
        "obs_risk_indicator": rng.uniform(0, 1, n),
    }
    # All 26 ML features
    for feat in ALL_ML_FEATURES:
        if feat not in data:
            data[feat] = rng.uniform(0, 1, n)
    # Sector loan columns
    for col in SECTOR_LOANS_COLUMNS:
        data[col] = rng.uniform(1000, 50000, n)
    # Ensure specific ranges for rule-testable columns
    data["capital_adequacy_ratio"] = rng.uniform(0.05, 0.20, n)
    data["npl_ratio"] = rng.uniform(0.01, 0.06, n)
    data["liquidity_coverage_ratio"] = rng.uniform(0.8, 1.5, n)
    data["loan_to_deposit_ratio"] = rng.uniform(0.7, 1.3, n)
    data["sector_concentration_hhi"] = rng.uniform(0.1, 0.4, n)
    data["return_on_assets"] = rng.uniform(0.005, 0.02, n)
    data["nsfr"] = rng.uniform(0.8, 1.5, n)
    data["wholesale_dependency_ratio"] = rng.uniform(0.2, 0.7, n)
    data["top20_borrower_concentration"] = rng.uniform(0.1, 0.4, n)
    data["operating_expenses"] = rng.uniform(500, 2000, n)
    data["operating_income"] = rng.uniform(1000, 3000, n)
    return pd.DataFrame(data)


@pytest.fixture
def sample_df():
    return _make_sample_df(20)


@pytest.fixture
def sample_csv(sample_df, tmp_path):
    path = tmp_path / "test_data.csv"
    sample_df.to_csv(path, index=False)
    return str(path)


# ── Stage 1: Load ────────────────────────────────────────────────────

class TestLoadData:
    def test_load_valid_csv(self, sample_csv):
        df = load_data(sample_csv)
        assert len(df) == 20
        assert "bank_id" in df.columns

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_data("/nonexistent/path/data.csv")

    def test_load_empty_csv(self, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_text("col1,col2\n")
        with pytest.raises(pd.errors.EmptyDataError):
            load_data(str(path))

    def test_load_real_dataset(self):
        """Smoke test: load the actual project dataset."""
        if os.path.exists(DATA_PATH):
            df = load_data(DATA_PATH)
            assert len(df) > 0
            assert "bank_id" in df.columns


# ── Stage 2: Impute ──────────────────────────────────────────────────

class TestImpute:
    def test_no_missing_after_impute(self, sample_df):
        sample_df.loc[0, "npl_ratio"] = np.nan
        sample_df.loc[1, "capital_adequacy_ratio"] = np.nan
        result = _impute_missing_values(sample_df)
        assert result[ALL_ML_FEATURES].isna().sum().sum() == 0

    def test_impute_uses_median(self, sample_df):
        sample_df.loc[0, "npl_ratio"] = np.nan
        # Median is computed on the column WITH the NaN (pandas skipna=True)
        expected_median = sample_df["npl_ratio"].median()
        result = _impute_missing_values(sample_df)
        assert result.loc[0, "npl_ratio"] == pytest.approx(expected_median)

    def test_impute_no_change_when_clean(self, sample_df):
        result = _impute_missing_values(sample_df)
        pd.testing.assert_frame_equal(result, sample_df)

    def test_impute_preserves_non_numeric(self, sample_df):
        sample_df.loc[0, "npl_ratio"] = np.nan
        result = _impute_missing_values(sample_df)
        assert result["bank_id"].tolist() == sample_df["bank_id"].tolist()


# ── Stage 3: Validate ────────────────────────────────────────────────

class TestValidate:
    def test_validate_passes_with_all_features(self, sample_df):
        _validate_features(sample_df)  # should not raise

    def test_validate_raises_on_missing_feature(self, sample_df):
        df = sample_df.drop(columns=["npl_ratio"])
        with pytest.raises(KeyError, match="npl_ratio"):
            _validate_features(df)

    def test_validate_warns_on_missing_sector(self, sample_df, capsys):
        df = sample_df.drop(columns=["sector_loans_energy"])
        _validate_features(df)  # should not raise, just warn
        captured = capsys.readouterr()
        assert "sector_loans_energy" in captured.out


# ── Stage 4: Feature Engineering ─────────────────────────────────────

class TestFeatureEngineering:
    def test_risk_to_profit_ratio_created(self, sample_df):
        result = _create_risk_to_profit_ratio(sample_df)
        assert "risk_to_profit_ratio" in result.columns

    def test_risk_to_profit_ratio_values(self, sample_df):
        result = _create_risk_to_profit_ratio(sample_df)
        # ratio = npl_ratio / ROA, should be positive
        assert (result["risk_to_profit_ratio"] >= 0).all()

    def test_risk_to_profit_ratio_zero_roa(self, sample_df):
        """When ROA=0, ratio should be capped at 1e6."""
        sample_df["return_on_assets"] = 0.0
        result = _create_risk_to_profit_ratio(sample_df)
        assert (result["risk_to_profit_ratio"] == 1e6).all()

    def test_efficiency_ratio_created(self, sample_df):
        result = _create_efficiency_ratio(sample_df)
        assert "efficiency_ratio" in result.columns

    def test_efficiency_ratio_values(self, sample_df):
        result = _create_efficiency_ratio(sample_df)
        assert (result["efficiency_ratio"] <= 10.0).all()

    def test_efficiency_ratio_zero_income(self, sample_df):
        """When operating_income=0, ratio should be capped at 10.0."""
        sample_df["operating_income"] = 0.0
        result = _create_efficiency_ratio(sample_df)
        assert (result["efficiency_ratio"] == 10.0).all()



# ── Stage 5: Expert Rule Engine ──────────────────────────────────────

class TestExpertRules:
    def test_adds_rule_columns(self, sample_df):
        result = evaluate_expert_rules(sample_df)
        assert "rule_violations" in result.columns
        assert "rule_risk_score" in result.columns

    def test_compliant_bank(self, sample_df):
        """A bank with safe values should be Compliant."""
        sample_df["capital_adequacy_ratio"] = 0.15
        sample_df["npl_ratio"] = 0.01
        sample_df["liquidity_coverage_ratio"] = 1.5
        sample_df["loan_to_deposit_ratio"] = 0.8
        sample_df["sector_concentration_hhi"] = 0.1
        sample_df["nsfr"] = 1.2
        sample_df["return_on_assets"] = 0.01
        sample_df["wholesale_dependency_ratio"] = 0.3
        sample_df["top20_borrower_concentration"] = 0.15
        result = evaluate_expert_rules(sample_df)
        assert all(v == ["Compliant"] for v in result["rule_violations"])
        assert (result["rule_risk_score"] == 0).all()

    def test_car_critical_triggered(self, sample_df):
        """CAR < 8% should trigger CAR_CRITICAL."""
        sample_df["capital_adequacy_ratio"] = 0.05  # below 8%
        sample_df["npl_ratio"] = 0.01
        sample_df["liquidity_coverage_ratio"] = 1.5
        sample_df["loan_to_deposit_ratio"] = 0.8
        sample_df["sector_concentration_hhi"] = 0.1
        sample_df["nsfr"] = 1.2
        sample_df["return_on_assets"] = 0.01
        sample_df["wholesale_dependency_ratio"] = 0.3
        sample_df["top20_borrower_concentration"] = 0.15
        result = evaluate_expert_rules(sample_df)
        for violations in result["rule_violations"]:
            assert "CAR_CRITICAL" in violations

    def test_multiple_violations(self, sample_df):
        """A bank violating multiple rules should have score > 1."""
        sample_df["capital_adequacy_ratio"] = 0.05  # CAR_CRITICAL
        sample_df["npl_ratio"] = 0.05               # NPL_WARNING
        sample_df["liquidity_coverage_ratio"] = 0.8  # LCR_CRITICAL
        sample_df["loan_to_deposit_ratio"] = 0.8
        sample_df["sector_concentration_hhi"] = 0.1
        sample_df["nsfr"] = 1.2
        sample_df["return_on_assets"] = 0.01
        sample_df["wholesale_dependency_ratio"] = 0.3
        sample_df["top20_borrower_concentration"] = 0.15
        result = evaluate_expert_rules(sample_df)
        assert (result["rule_risk_score"] >= 3).all()

    def test_does_not_mutate_input(self, sample_df):
        original_cols = set(sample_df.columns)
        evaluate_expert_rules(sample_df)
        assert set(sample_df.columns) == original_cols

    def test_missing_rule_column_skipped(self, sample_df):
        """If a rule column is missing, the rule should be skipped."""
        df = sample_df.drop(columns=["capital_adequacy_ratio"])
        # Should not raise, just skip the rule
        result = evaluate_expert_rules(df)
        for violations in result["rule_violations"]:
            assert "CAR_CRITICAL" not in violations


# ── Stage 6: Scaling ─────────────────────────────────────────────────

class TestScaling:
    def test_scale_feature_group(self, sample_df):
        cols = ["npl_ratio", "loan_growth_rate"]
        df_out, scaler = _scale_feature_group(sample_df.copy(), cols, "test_group")
        # Scaled columns should have ~zero mean
        for col in cols:
            assert abs(df_out[col].mean()) < 1e-10

    def test_scaler_has_correct_attributes(self, sample_df):
        cols = ["npl_ratio", "loan_growth_rate"]
        _, scaler = _scale_feature_group(sample_df.copy(), cols, "test_group")
        assert hasattr(scaler, "mean_")
        assert hasattr(scaler, "scale_")
        assert len(scaler.mean_) == 2


# ── Integration: process_data ────────────────────────────────────────

class TestProcessDataIntegration:
    def test_process_data_with_sample_csv(self, sample_csv):
        df_proc, df_orig, scalers = process_data(sample_csv)
        # Shapes
        assert df_proc.shape[0] == df_orig.shape[0] == 20
        # Scalers for all 7 pillar groups
        assert len(scalers) == len(FEATURE_GROUPS)
        # df_original has rule columns
        assert "rule_violations" in df_orig.columns
        assert "rule_risk_score" in df_orig.columns
        # df_original has engineered features
        assert "risk_to_profit_ratio" in df_orig.columns
        assert "efficiency_ratio" in df_orig.columns

    def test_processed_features_are_scaled(self, sample_csv):
        df_proc, _, _ = process_data(sample_csv)
        for group_name, cols in FEATURE_GROUPS.items():
            for col in cols:
                # Scaled features should have mean ≈ 0
                assert abs(df_proc[col].mean()) < 1e-10, (
                    f"Column '{col}' in group '{group_name}' not properly scaled"
                )

    def test_original_features_unscaled(self, sample_csv):
        df_proc, df_orig, _ = process_data(sample_csv)
        # Original should differ from processed for ML features
        for col in ALL_ML_FEATURES[:3]:
            assert not np.allclose(df_proc[col].values, df_orig[col].values)

    def test_process_real_dataset(self):
        """Smoke test: full pipeline on the actual dataset."""
        if os.path.exists(DATA_PATH):
            df_proc, df_orig, scalers = process_data(DATA_PATH)
            assert len(df_proc) == 200
            assert len(scalers) == len(FEATURE_GROUPS)
