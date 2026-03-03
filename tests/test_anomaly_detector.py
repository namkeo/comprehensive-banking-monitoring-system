"""
tests/test_anomaly_detector.py – Unit tests for the anomaly detection engine.
===============================================================================
Tests: BankAnomalyDetector init, pillar consensus, overall ML score,
hybrid status, clustering, anomaly drivers, OBS risk, full pipeline.
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

from config import (
    ALL_ML_FEATURES,
    MODEL_PARAMS,
    RISK_PILLARS,
    SECTOR_LOANS_COLUMNS,
)
from models.anomaly_detector import (
    BankAnomalyDetector,
    PILLAR_KEYS,
    RISK_LABELS,
    _CONSENSUS_MAP,
    _HYBRID_CRITICAL,
    _HYBRID_WARNING,
    _build_model,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _make_sample_df(n: int = 50) -> pd.DataFrame:
    """Create a synthetic DataFrame with all required columns for ML tests.

    Uses n=50 by default (minimum viable for 3 unsupervised models).
    """
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
    for feat in ALL_ML_FEATURES:
        if feat not in data:
            data[feat] = rng.uniform(0, 1, n)
    for col in SECTOR_LOANS_COLUMNS:
        data[col] = rng.uniform(1000, 50000, n)
    # Realistic ranges for key columns
    data["capital_adequacy_ratio"] = rng.uniform(0.05, 0.20, n)
    data["npl_ratio"] = rng.uniform(0.01, 0.06, n)
    data["liquidity_coverage_ratio"] = rng.uniform(0.8, 1.5, n)
    data["loan_to_deposit_ratio"] = rng.uniform(0.7, 1.3, n)
    data["sector_concentration_hhi"] = rng.uniform(0.1, 0.4, n)
    data["return_on_assets"] = rng.uniform(0.005, 0.02, n)
    data["operating_expenses"] = rng.uniform(500, 2000, n)
    data["operating_income"] = rng.uniform(1000, 3000, n)
    return pd.DataFrame(data)


def _make_scaled_df(n: int = 50) -> pd.DataFrame:
    """Create a scaled (standardised) version of the sample DataFrame."""
    df = _make_sample_df(n)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[ALL_ML_FEATURES] = scaler.fit_transform(df[ALL_ML_FEATURES])
    return df


def _prepare_pipeline_data(n: int = 50):
    """Return (df_scaled, df_original) ready for the detector."""
    df_original = _make_sample_df(n)
    df_original["rule_violations"] = [["Compliant"]] * n
    df_original["rule_risk_score"] = 0
    df_scaled = df_original.copy()
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df_scaled[ALL_ML_FEATURES] = scaler.fit_transform(df_scaled[ALL_ML_FEATURES])
    return df_scaled, df_original


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def detector():
    return BankAnomalyDetector()


@pytest.fixture
def pipeline_data():
    return _prepare_pipeline_data(50)


# ── 0. _build_model helper ───────────────────────────────────────────

class TestBuildModel:
    def test_build_isolation_forest(self):
        model = _build_model("IsolationForest")
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_build_lof(self):
        model = _build_model("LocalOutlierFactor")
        assert hasattr(model, "fit")
        assert model.novelty is True

    def test_build_svm(self):
        model = _build_model("OneClassSVM")
        assert hasattr(model, "fit")

    def test_build_unknown_raises(self):
        with pytest.raises((ValueError, KeyError)):
            _build_model("NonExistentModel")


# ── 1. BankAnomalyDetector __init__ ─────────────────────────────────

class TestDetectorInit:
    def test_default_params(self, detector):
        assert detector.contamination == MODEL_PARAMS["contamination"]
        assert detector.n_clusters == MODEL_PARAMS["n_clusters"]
        assert detector.random_state == MODEL_PARAMS["random_state"]

    def test_custom_params(self):
        det = BankAnomalyDetector(contamination=0.1, n_clusters=5, random_state=99)
        assert det.contamination == 0.1
        assert det.n_clusters == 5
        assert det.random_state == 99

    def test_initial_state(self, detector):
        assert detector.pillar_models == {}
        assert detector.kmeans is None
        assert detector.cluster_dna == {}
        assert detector.global_if_model is None
        assert detector.shap_explainer is None
        assert detector.shap_values is None
        # Multi-model XAI attributes
        assert detector.global_lof_model is None
        assert detector.global_svm_model is None
        assert detector.lof_shap_values is None
        assert detector.svm_shap_values is None
        assert detector.permutation_importance_results == {}
        assert detector.lime_explanations == {}


# ── 2. Pillar-by-Pillar Consensus ───────────────────────────────────

class TestPillarConsensus:
    def test_adds_expected_columns(self, detector, pipeline_data):
        df_scaled, _ = pipeline_data
        result = detector._run_pillar_consensus(df_scaled)
        for pillar_name in RISK_PILLARS:
            key = PILLAR_KEYS[pillar_name]
            for suffix in ["IF", "LOF", "SVM", "n_flags", "consensus"]:
                col = f"{key}_{suffix}"
                assert col in result.columns, f"Missing column: {col}"

    def test_consensus_values_in_expected_set(self, detector, pipeline_data):
        df_scaled, _ = pipeline_data
        result = detector._run_pillar_consensus(df_scaled)
        valid_scores = set(_CONSENSUS_MAP.values())  # {0, 33, 66, 100}
        for pillar_name in RISK_PILLARS:
            key = PILLAR_KEYS[pillar_name]
            scores = set(result[f"{key}_consensus"].unique())
            assert scores.issubset(valid_scores), (
                f"Pillar '{pillar_name}' has unexpected scores: {scores - valid_scores}"
            )

    def test_n_flags_range(self, detector, pipeline_data):
        df_scaled, _ = pipeline_data
        result = detector._run_pillar_consensus(df_scaled)
        for pillar_name in RISK_PILLARS:
            key = PILLAR_KEYS[pillar_name]
            flags = result[f"{key}_n_flags"]
            assert flags.min() >= 0
            assert flags.max() <= 3

    def test_predictions_are_minus1_or_1(self, detector, pipeline_data):
        df_scaled, _ = pipeline_data
        result = detector._run_pillar_consensus(df_scaled)
        for pillar_name in RISK_PILLARS:
            key = PILLAR_KEYS[pillar_name]
            for short in ["IF", "LOF", "SVM"]:
                vals = set(result[f"{key}_{short}"].unique())
                assert vals.issubset({-1, 1}), (
                    f"{key}_{short} has unexpected values: {vals}"
                )

    def test_stores_fitted_models(self, detector, pipeline_data):
        df_scaled, _ = pipeline_data
        detector._run_pillar_consensus(df_scaled)
        assert len(detector.pillar_models) == len(RISK_PILLARS)
        for key in PILLAR_KEYS.values():
            assert key in detector.pillar_models
            assert len(detector.pillar_models[key]) == 3



# ── 3. Overall ML Risk Score ────────────────────────────────────────

class TestOverallMLScore:
    def _make_consensus_df(self):
        """Create a DataFrame with pre-set consensus columns."""
        n = 10
        df = pd.DataFrame({"bank_id": [f"B{i}" for i in range(n)]})
        for pillar_name in RISK_PILLARS:
            key = PILLAR_KEYS[pillar_name]
            df[f"{key}_consensus"] = 0  # all normal
        return df

    def test_adds_overall_column(self):
        df = self._make_consensus_df()
        result = BankAnomalyDetector._compute_overall_ml_score(df)
        assert "Overall_ML_Risk_Score" in result.columns

    def test_all_zero_consensus(self):
        df = self._make_consensus_df()
        result = BankAnomalyDetector._compute_overall_ml_score(df)
        assert (result["Overall_ML_Risk_Score"] == 0.0).all()

    def test_all_max_consensus(self):
        df = self._make_consensus_df()
        for pillar_name in RISK_PILLARS:
            key = PILLAR_KEYS[pillar_name]
            df[f"{key}_consensus"] = 100
        result = BankAnomalyDetector._compute_overall_ml_score(df)
        assert (result["Overall_ML_Risk_Score"] == 100.0).all()

    def test_mixed_consensus(self):
        df = self._make_consensus_df()
        keys = list(PILLAR_KEYS.values())
        # Set first pillar to 100, rest to 0
        df[f"{keys[0]}_consensus"] = 100
        result = BankAnomalyDetector._compute_overall_ml_score(df)
        expected = round(100.0 / len(RISK_PILLARS), 2)
        assert result["Overall_ML_Risk_Score"].iloc[0] == pytest.approx(expected)


# ── 4. Hybrid Status ────────────────────────────────────────────────

class TestHybridStatus:
    def _make_ml_df(self, ml_score, rule_score):
        n = 5
        df = pd.DataFrame({
            "Overall_ML_Risk_Score": [ml_score] * n,
            "rule_risk_score": [rule_score] * n,
        })
        return df

    def test_critical_by_ml(self):
        df = self._make_ml_df(ml_score=65, rule_score=0)
        result = BankAnomalyDetector._compute_hybrid_status(df)
        assert (result["Final_Hybrid_Risk_Status"] == "Critical").all()

    def test_critical_by_rules(self):
        df = self._make_ml_df(ml_score=10, rule_score=3)
        result = BankAnomalyDetector._compute_hybrid_status(df)
        assert (result["Final_Hybrid_Risk_Status"] == "Critical").all()

    def test_warning_by_ml(self):
        df = self._make_ml_df(ml_score=35, rule_score=0)
        result = BankAnomalyDetector._compute_hybrid_status(df)
        assert (result["Final_Hybrid_Risk_Status"] == "Warning").all()

    def test_warning_by_rules(self):
        df = self._make_ml_df(ml_score=10, rule_score=1)
        result = BankAnomalyDetector._compute_hybrid_status(df)
        assert (result["Final_Hybrid_Risk_Status"] == "Warning").all()

    def test_normal(self):
        df = self._make_ml_df(ml_score=10, rule_score=0)
        result = BankAnomalyDetector._compute_hybrid_status(df)
        assert (result["Final_Hybrid_Risk_Status"] == "Normal").all()

    def test_is_anomaly_backward_compat(self):
        df = self._make_ml_df(ml_score=65, rule_score=0)
        result = BankAnomalyDetector._compute_hybrid_status(df)
        assert (result["is_anomaly"] == -1).all()

    def test_normal_is_anomaly_1(self):
        df = self._make_ml_df(ml_score=10, rule_score=0)
        result = BankAnomalyDetector._compute_hybrid_status(df)
        assert (result["is_anomaly"] == 1).all()

    def test_anomaly_score_column(self):
        df = self._make_ml_df(ml_score=50, rule_score=0)
        result = BankAnomalyDetector._compute_hybrid_status(df)
        assert "anomaly_score" in result.columns
        # anomaly_score = -(ML_score / 100)
        assert result["anomaly_score"].iloc[0] == pytest.approx(-0.5)

    def test_missing_rule_col_defaults_to_zero(self):
        df = pd.DataFrame({"Overall_ML_Risk_Score": [10, 20, 30]})
        result = BankAnomalyDetector._compute_hybrid_status(df)
        assert "rule_risk_score" in result.columns
        assert (result["rule_risk_score"] == 0).all()



# ── 5. Clustering ───────────────────────────────────────────────────

class TestClustering:
    def test_cluster_labels_assigned(self, detector, pipeline_data):
        df_scaled, df_original = pipeline_data
        # Need consensus columns first
        df_consensus = detector._run_pillar_consensus(df_scaled)
        df_consensus = detector._compute_overall_ml_score(df_consensus)
        df_consensus = detector._compute_hybrid_status(df_consensus)
        result = detector.cluster_banks(df_consensus, df_original)
        assert "cluster_label" in result.columns
        assert "cluster_dna" in result.columns

    def test_cluster_labels_are_valid(self, detector, pipeline_data):
        df_scaled, df_original = pipeline_data
        df_consensus = detector._run_pillar_consensus(df_scaled)
        df_consensus = detector._compute_overall_ml_score(df_consensus)
        df_consensus = detector._compute_hybrid_status(df_consensus)
        result = detector.cluster_banks(df_consensus, df_original)
        valid_labels = set(RISK_LABELS)
        actual_labels = set(result["cluster_label"].unique())
        assert actual_labels.issubset(valid_labels)

    def test_three_clusters(self, detector, pipeline_data):
        df_scaled, df_original = pipeline_data
        df_consensus = detector._run_pillar_consensus(df_scaled)
        df_consensus = detector._compute_overall_ml_score(df_consensus)
        df_consensus = detector._compute_hybrid_status(df_consensus)
        result = detector.cluster_banks(df_consensus, df_original)
        assert detector.kmeans is not None
        assert detector.kmeans.n_clusters == 3

    def test_single_row_fallback(self, detector):
        """With only 1 row, clustering should fallback gracefully."""
        df_scaled, df_original = _prepare_pipeline_data(1)
        result = detector.cluster_banks(df_scaled, df_original)
        assert result["cluster_label"].iloc[0] == "Medium Risk"

    def test_cluster_dna_populated(self, detector, pipeline_data):
        df_scaled, df_original = pipeline_data
        df_consensus = detector._run_pillar_consensus(df_scaled)
        df_consensus = detector._compute_overall_ml_score(df_consensus)
        df_consensus = detector._compute_hybrid_status(df_consensus)
        detector.cluster_banks(df_consensus, df_original)
        assert len(detector.cluster_dna) > 0


# ── 6. OBS Risk Contribution ────────────────────────────────────────

class TestOBSRisk:
    def test_returns_flag_and_zscore(self, detector):
        df = _make_sample_df(20)
        mask = pd.Series(1, index=df.index)
        mask.iloc[:5] = -1  # 5 anomalies
        flag, zscore = detector.compute_obs_risk_contribution(df, mask)
        assert len(flag) == 20
        assert len(zscore) == 20

    def test_no_anomalies_returns_na(self, detector):
        df = _make_sample_df(20)
        mask = pd.Series(1, index=df.index)  # no anomalies
        flag, zscore = detector.compute_obs_risk_contribution(df, mask)
        assert (flag == "N/A").all()
        assert (zscore == 0.0).all()

    def test_missing_obs_column(self, detector):
        df = _make_sample_df(20).drop(columns=["obs_risk_indicator"])
        mask = pd.Series(-1, index=df.index)
        flag, zscore = detector.compute_obs_risk_contribution(df, mask)
        assert (flag == "N/A").all()

    def test_flag_values(self, detector):
        df = _make_sample_df(50)
        mask = pd.Series(-1, index=df.index)  # all anomalies
        flag, _ = detector.compute_obs_risk_contribution(df, mask)
        valid_flags = {"High OBS Risk", "Normal OBS", "N/A"}
        assert set(flag.unique()).issubset(valid_flags)



# ── 7. Anomaly Drivers ──────────────────────────────────────────────

class TestAnomalyDrivers:
    def test_no_anomalies_returns_na(self, detector):
        df = _make_sample_df(20)
        mask = pd.Series(1, index=df.index)  # no anomalies
        drivers, groups = detector.get_anomaly_drivers(df, mask)
        assert (drivers == "N/A").all()
        assert (groups == "N/A").all()

    def test_zscore_fallback(self, detector):
        """Without SHAP, should use z-score fallback."""
        df = _make_sample_df(20)
        mask = pd.Series(1, index=df.index)
        mask.iloc[:3] = -1
        drivers, groups = detector.get_anomaly_drivers(df, mask)
        # Anomalous rows should have a driver feature name
        for idx in range(3):
            assert drivers.iloc[idx] != "N/A"
            assert drivers.iloc[idx] in ALL_ML_FEATURES

    def test_driver_groups_are_valid_pillars(self, detector):
        df = _make_sample_df(20)
        mask = pd.Series(1, index=df.index)
        mask.iloc[:3] = -1
        _, groups = detector.get_anomaly_drivers(df, mask)
        valid_groups = set(RISK_PILLARS.keys()) | {"N/A", "Unknown"}
        for g in groups:
            assert g in valid_groups, f"Unexpected driver group: {g}"


# ── 7b. XAI Methods ───────────────────────────────────────────────

class TestXAIMethods:
    """Tests for compute_multi_model_shap, compute_permutation_importance,
    and compute_local_surrogate."""

    @pytest.fixture(autouse=True)
    def _setup(self, detector, pipeline_data):
        """Fit global models so XAI methods can run."""
        self.detector = detector
        self.df_scaled, self.df_original = pipeline_data
        # Run pillar consensus to populate pillar_models
        df_c = detector._run_pillar_consensus(self.df_scaled)
        df_c = detector._compute_overall_ml_score(df_c)
        df_c = detector._compute_hybrid_status(df_c)
        self.df_consensus = df_c
        # Fit global IF model via SHAP step
        try:
            detector.evaluate_model_with_shap(self.df_scaled)
        except Exception:
            pass

    # --- Multi-Model SHAP ---
    def test_multi_model_shap_returns_dict(self):
        result = self.detector.compute_multi_model_shap(
            self.df_scaled, n_background=10
        )
        assert isinstance(result, dict)
        assert "LOF" in result
        assert "SVM" in result

    def test_multi_model_shap_shapes(self):
        result = self.detector.compute_multi_model_shap(
            self.df_scaled, n_background=10
        )
        n_samples = len(self.df_scaled)
        n_features = len(ALL_ML_FEATURES)
        for key in ["LOF", "SVM"]:
            assert result[key].shape == (n_samples, n_features)

    def test_multi_model_shap_stores_models(self):
        self.detector.compute_multi_model_shap(
            self.df_scaled, n_background=10
        )
        assert self.detector.global_lof_model is not None
        assert self.detector.global_svm_model is not None
        assert self.detector.lof_shap_values is not None
        assert self.detector.svm_shap_values is not None

    # --- Permutation Importance ---
    def test_permutation_importance_returns_dict(self):
        # Ensure global models are fitted
        self.detector.compute_multi_model_shap(
            self.df_scaled, n_background=10
        )
        result = self.detector.compute_permutation_importance(
            self.df_scaled, n_repeats=2
        )
        assert isinstance(result, dict)
        # At least IF should be present (if global_if_model was fitted)
        assert len(result) >= 1

    def test_permutation_importance_keys(self):
        self.detector.compute_multi_model_shap(
            self.df_scaled, n_background=10
        )
        result = self.detector.compute_permutation_importance(
            self.df_scaled, n_repeats=2
        )
        for model_name, pi_data in result.items():
            assert "importances_mean" in pi_data
            assert "importances_std" in pi_data
            assert len(pi_data["importances_mean"]) == len(ALL_ML_FEATURES)
            assert len(pi_data["importances_std"]) == len(ALL_ML_FEATURES)

    def test_permutation_importance_stored(self):
        self.detector.compute_multi_model_shap(
            self.df_scaled, n_background=10
        )
        self.detector.compute_permutation_importance(
            self.df_scaled, n_repeats=2
        )
        assert self.detector.permutation_importance_results != {}

    # --- Local Surrogate ---
    def test_local_surrogate_returns_dict(self):
        self.detector.compute_multi_model_shap(
            self.df_scaled, n_background=10
        )
        result = self.detector.compute_local_surrogate(
            self.df_scaled, bank_indices=[0, 1],
            n_perturbations=50, kernel_width=0.75,
        )
        assert isinstance(result, dict)
        assert len(result) >= 1

    def test_local_surrogate_has_correct_indices(self):
        self.detector.compute_multi_model_shap(
            self.df_scaled, n_background=10
        )
        indices = [0, 2, 4]
        result = self.detector.compute_local_surrogate(
            self.df_scaled, bank_indices=indices,
            n_perturbations=50, kernel_width=0.75,
        )
        for model_name, bank_data in result.items():
            for idx in indices:
                assert idx in bank_data, (
                    f"Missing index {idx} for model {model_name}"
                )

    def test_local_surrogate_feature_coefficients(self):
        self.detector.compute_multi_model_shap(
            self.df_scaled, n_background=10
        )
        result = self.detector.compute_local_surrogate(
            self.df_scaled, bank_indices=[0],
            n_perturbations=50, kernel_width=0.75,
        )
        for model_name, bank_data in result.items():
            coeffs = bank_data[0]
            assert isinstance(coeffs, dict)
            assert len(coeffs) == len(ALL_ML_FEATURES)
            # All values should be finite floats
            for feat, val in coeffs.items():
                assert np.isfinite(val), f"Non-finite coeff for {feat}"

    def test_local_surrogate_stored(self):
        self.detector.compute_multi_model_shap(
            self.df_scaled, n_background=10
        )
        self.detector.compute_local_surrogate(
            self.df_scaled, bank_indices=[0],
            n_perturbations=50, kernel_width=0.75,
        )
        assert self.detector.lime_explanations != {}

    def test_local_surrogate_no_models_empty(self):
        """If no global models are fitted, local surrogate returns empty."""
        fresh_det = BankAnomalyDetector()
        result = fresh_det.compute_local_surrogate(
            self.df_scaled, bank_indices=[0],
            n_perturbations=50, kernel_width=0.75,
        )
        assert result == {}


# ── 8. Full Pipeline Integration ────────────────────────────────────

class TestRunFullAnalysis:
    def test_output_columns(self, detector, pipeline_data):
        df_scaled, df_original = pipeline_data
        result = detector.run_full_analysis(df_scaled, df_original)
        # Per-pillar columns
        for pillar_name in RISK_PILLARS:
            key = PILLAR_KEYS[pillar_name]
            assert f"{key}_consensus" in result.columns
        # Aggregate
        assert "Overall_ML_Risk_Score" in result.columns
        assert "Final_Hybrid_Risk_Status" in result.columns
        # Backward-compatible
        for col in ["is_anomaly", "anomaly_score", "cluster_label",
                     "cluster_dna", "anomaly_driver", "anomaly_driver_group",
                     "obs_risk_flag", "obs_risk_zscore", "shap_top3_drivers"]:
            assert col in result.columns, f"Missing backward-compat column: {col}"

    def test_output_row_count(self, detector, pipeline_data):
        df_scaled, df_original = pipeline_data
        result = detector.run_full_analysis(df_scaled, df_original)
        assert len(result) == len(df_original)

    def test_hybrid_status_values(self, detector, pipeline_data):
        df_scaled, df_original = pipeline_data
        result = detector.run_full_analysis(df_scaled, df_original)
        valid = {"Critical", "Warning", "Normal"}
        assert set(result["Final_Hybrid_Risk_Status"].unique()).issubset(valid)

    def test_cluster_labels_present(self, detector, pipeline_data):
        df_scaled, df_original = pipeline_data
        result = detector.run_full_analysis(df_scaled, df_original)
        assert result["cluster_label"].notna().all()

    def test_original_columns_preserved(self, detector, pipeline_data):
        df_scaled, df_original = pipeline_data
        result = detector.run_full_analysis(df_scaled, df_original)
        # Original identifiers should still be there
        assert "bank_id" in result.columns
        assert "period" in result.columns

    def test_ml_score_range(self, detector, pipeline_data):
        df_scaled, df_original = pipeline_data
        result = detector.run_full_analysis(df_scaled, df_original)
        scores = result["Overall_ML_Risk_Score"]
        assert scores.min() >= 0
        assert scores.max() <= 100
