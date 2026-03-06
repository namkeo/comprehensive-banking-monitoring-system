"""
Tests for models/risk_simulators.py – Stress-Testing Engine.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.risk_simulators import (
    _apply_shocks,
    _detect_breaches,
    _validate_inputs,
    _COMPARISON_METRICS,
    _REQUIRED_COLS,
    run_stress_scenario,
)


# ======================================================================
#  Fixtures
# ======================================================================

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Minimal DataFrame with all required columns for stress testing."""
    return pd.DataFrame({
        "bank_id": ["BNK_001", "BNK_002", "BNK_003"],
        "non_performing_loans": [1000.0, 2000.0, 500.0],
        "total_loans": [50000.0, 60000.0, 40000.0],
        "npl_ratio": [0.02, 0.0333, 0.0125],
        "loan_loss_provisions": [800.0, 1500.0, 400.0],
        "tier1_capital": [5000.0, 6000.0, 4000.0],
        "tier2_capital": [1000.0, 1200.0, 800.0],
        "risk_weighted_assets": [55000.0, 65000.0, 45000.0],
        "capital_adequacy_ratio": [0.1091, 0.1108, 0.1067],
        "total_deposits": [80000.0, 90000.0, 70000.0],
        "high_quality_liquid_assets": [15000.0, 18000.0, 12000.0],
        "net_cash_outflows_30d": [10000.0, 12000.0, 8000.0],
        "liquidity_coverage_ratio": [1.5, 1.5, 1.5],
        "total_assets": [100000.0, 120000.0, 90000.0],
        "loan_to_deposit_ratio": [0.625, 0.6667, 0.5714],
        # Extra columns for breach detection
        "sector_concentration_hhi": [0.15, 0.30, 0.20],
        "nsfr": [1.1, 0.95, 1.2],
        "return_on_assets": [0.01, -0.005, 0.02],
        "wholesale_dependency_ratio": [0.3, 0.55, 0.2],
        "top20_borrower_concentration": [0.20, 0.30, 0.15],
    })


# ======================================================================
#  Validation Tests
# ======================================================================

class TestValidation:
    def test_missing_columns_raises_key_error(self):
        df = pd.DataFrame({"bank_id": ["A"], "total_assets": [100]})
        with pytest.raises(KeyError, match="missing from the DataFrame"):
            _validate_inputs(df, 1.5, 0.9, 1.0)

    def test_npl_shock_below_one_raises(self, sample_df):
        with pytest.raises(ValueError, match="npl_shock must be >= 1.0"):
            _validate_inputs(sample_df, 0.8, 0.9, 1.0)

    def test_deposit_shock_zero_raises(self, sample_df):
        with pytest.raises(ValueError, match="deposit_shock must be in"):
            _validate_inputs(sample_df, 1.0, 0.0, 1.0)

    def test_deposit_shock_above_one_raises(self, sample_df):
        with pytest.raises(ValueError, match="deposit_shock must be in"):
            _validate_inputs(sample_df, 1.0, 1.5, 1.0)

    def test_asset_devaluation_zero_raises(self, sample_df):
        with pytest.raises(ValueError, match="asset_devaluation must be in"):
            _validate_inputs(sample_df, 1.0, 1.0, 0.0)

    def test_asset_devaluation_above_one_raises(self, sample_df):
        with pytest.raises(ValueError, match="asset_devaluation must be in"):
            _validate_inputs(sample_df, 1.0, 1.0, 1.5)

    def test_valid_inputs_pass(self, sample_df):
        _validate_inputs(sample_df, 1.5, 0.9, 0.85)  # should not raise


# ======================================================================
#  Shock Mechanics Tests
# ======================================================================

class TestApplyShocks:
    def test_npl_increases(self, sample_df):
        stressed = _apply_shocks(sample_df, npl_shock=2.0, deposit_shock=1.0, asset_devaluation=1.0)
        np.testing.assert_allclose(
            stressed["non_performing_loans"].values,
            sample_df["non_performing_loans"].values * 2.0,
        )

    def test_npl_ratio_recalculated(self, sample_df):
        stressed = _apply_shocks(sample_df, npl_shock=2.0, deposit_shock=1.0, asset_devaluation=1.0)
        expected = sample_df["non_performing_loans"] * 2.0 / sample_df["total_loans"]
        np.testing.assert_allclose(stressed["npl_ratio"].values, expected.values, rtol=1e-6)

    def test_tier1_capital_decreases_with_npl_shock(self, sample_df):
        stressed = _apply_shocks(sample_df, npl_shock=1.5, deposit_shock=1.0, asset_devaluation=1.0)
        incremental = sample_df["non_performing_loans"] * 0.5
        expected = (sample_df["tier1_capital"] - incremental).clip(lower=0)
        np.testing.assert_allclose(stressed["tier1_capital"].values, expected.values)

    def test_car_decreases_with_npl_shock(self, sample_df):
        stressed = _apply_shocks(sample_df, npl_shock=1.5, deposit_shock=1.0, asset_devaluation=1.0)
        assert (stressed["capital_adequacy_ratio"] < sample_df["capital_adequacy_ratio"]).all()

    def test_deposits_decrease_with_shock(self, sample_df):
        stressed = _apply_shocks(sample_df, npl_shock=1.0, deposit_shock=0.8, asset_devaluation=1.0)
        np.testing.assert_allclose(
            stressed["total_deposits"].values,
            sample_df["total_deposits"].values * 0.8,
        )

    def test_lcr_decreases_with_deposit_shock(self, sample_df):
        stressed = _apply_shocks(sample_df, npl_shock=1.0, deposit_shock=0.8, asset_devaluation=1.0)
        assert (stressed["liquidity_coverage_ratio"] < sample_df["liquidity_coverage_ratio"]).all()

    def test_assets_decrease_with_devaluation(self, sample_df):
        stressed = _apply_shocks(sample_df, npl_shock=1.0, deposit_shock=1.0, asset_devaluation=0.85)
        np.testing.assert_allclose(
            stressed["total_assets"].values,
            sample_df["total_assets"].values * 0.85,
        )

    def test_no_shock_returns_baseline(self, sample_df):
        stressed = _apply_shocks(sample_df, npl_shock=1.0, deposit_shock=1.0, asset_devaluation=1.0)
        for col in ["non_performing_loans", "total_deposits", "total_assets"]:
            np.testing.assert_allclose(stressed[col].values, sample_df[col].values)



# ======================================================================
#  Breach Detection Tests
# ======================================================================

class TestBreachDetection:
    def test_breach_df_has_bank_id(self, sample_df):
        result = _detect_breaches(sample_df)
        assert "bank_id" in result.columns

    def test_breach_df_has_n_breaches(self, sample_df):
        result = _detect_breaches(sample_df)
        assert "n_breaches" in result.columns
        assert result["n_breaches"].dtype in [np.int32, np.int64, int]

    def test_breach_columns_are_boolean(self, sample_df):
        result = _detect_breaches(sample_df)
        breach_cols = [c for c in result.columns if c.startswith("breach_")]
        assert len(breach_cols) > 0
        for col in breach_cols:
            assert result[col].dtype == bool

    def test_known_breach_detected(self, sample_df):
        """BNK_002 has npl_ratio=0.0333 > 0.03 threshold -> NPL_WARNING breach."""
        result = _detect_breaches(sample_df)
        bnk2 = result[result["bank_id"] == "BNK_002"].iloc[0]
        assert bnk2["breach_NPL_WARNING"] is True or bnk2["breach_NPL_WARNING"] == True

    def test_nsfr_breach_detected(self, sample_df):
        """BNK_002 has nsfr=0.95 < 1.0 -> NSFR_CRITICAL breach."""
        result = _detect_breaches(sample_df)
        bnk2 = result[result["bank_id"] == "BNK_002"].iloc[0]
        assert bnk2["breach_NSFR_CRITICAL"] is True or bnk2["breach_NSFR_CRITICAL"] == True

    def test_no_car_breach_for_healthy_bank(self, sample_df):
        """All banks have CAR > 0.08 -> no CAR_CRITICAL breach."""
        result = _detect_breaches(sample_df)
        assert not result["breach_CAR_CRITICAL"].any()


# ======================================================================
#  Integration: run_stress_scenario
# ======================================================================

class TestRunStressScenario:
    def test_return_keys(self, sample_df):
        result = run_stress_scenario(sample_df, 1.5, 0.9, 1.0)
        assert set(result.keys()) == {"comparison", "breaches", "summary", "params"}

    def test_comparison_df_columns(self, sample_df):
        result = run_stress_scenario(sample_df, 1.5, 0.9, 1.0)
        comp = result["comparison"]
        assert set(comp.columns) == {"bank_id", "metric", "baseline", "stressed", "delta", "delta_pct"}

    def test_comparison_has_all_banks(self, sample_df):
        result = run_stress_scenario(sample_df, 1.5, 0.9, 1.0)
        comp = result["comparison"]
        assert set(comp["bank_id"].unique()) == {"BNK_001", "BNK_002", "BNK_003"}

    def test_comparison_has_expected_metrics(self, sample_df):
        result = run_stress_scenario(sample_df, 1.5, 0.9, 1.0)
        comp = result["comparison"]
        avail = [m for m in _COMPARISON_METRICS if m in sample_df.columns]
        assert set(comp["metric"].unique()) == set(avail)

    def test_summary_has_required_keys(self, sample_df):
        result = run_stress_scenario(sample_df, 1.5, 0.9, 1.0)
        summary = result["summary"]
        required = [
            "n_banks", "npl_shock", "deposit_shock", "asset_devaluation",
            "baseline_avg_car", "stressed_avg_car",
            "baseline_avg_lcr", "stressed_avg_lcr",
            "baseline_avg_npl", "stressed_avg_npl",
            "baseline_total_breaches", "stressed_total_breaches",
            "banks_with_new_breaches",
        ]
        for key in required:
            assert key in summary, f"Missing summary key: {key}"

    def test_stressed_car_lower_than_baseline(self, sample_df):
        result = run_stress_scenario(sample_df, 2.0, 0.9, 0.85)
        assert result["summary"]["stressed_avg_car"] < result["summary"]["baseline_avg_car"]

    def test_stressed_lcr_lower_than_baseline(self, sample_df):
        result = run_stress_scenario(sample_df, 1.0, 0.8, 1.0)
        assert result["summary"]["stressed_avg_lcr"] < result["summary"]["baseline_avg_lcr"]

    def test_stressed_npl_higher_than_baseline(self, sample_df):
        result = run_stress_scenario(sample_df, 1.5, 1.0, 1.0)
        assert result["summary"]["stressed_avg_npl"] > result["summary"]["baseline_avg_npl"]

    def test_no_shock_deltas_near_zero(self, sample_df):
        result = run_stress_scenario(sample_df, 1.0, 1.0, 1.0)
        comp = result["comparison"]
        assert comp["delta"].abs().max() < 1e-3  # rounding tolerance

    def test_params_returned(self, sample_df):
        result = run_stress_scenario(sample_df, 1.5, 0.9, 0.85)
        assert result["params"] == {
            "npl_shock": 1.5,
            "deposit_shock": 0.9,
            "asset_devaluation": 0.85,
        }

    def test_more_breaches_under_stress(self, sample_df):
        """Severe stress should produce >= baseline breaches."""
        result = run_stress_scenario(sample_df, 3.0, 0.7, 0.7)
        assert result["summary"]["stressed_total_breaches"] >= result["summary"]["baseline_total_breaches"]