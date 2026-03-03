"""
tests/test_config.py – Unit tests for config.py
=================================================
Validates the central knowledge base: risk pillars, expert rules,
ML model configs, feature labels, and pipeline parameters.
"""

import pytest

from config import (
    ALL_ML_FEATURES,
    CAPITAL_ADEQUACY_FEATURES,
    CONCENTRATION_RISK_FEATURES,
    CREDIT_RISK_FEATURES,
    EARNINGS_EFFICIENCY_FEATURES,
    EXPERT_RULES,
    EXPOSURE_LIQUIDITY_FEATURES,
    FEATURE_LABELS,
    FINANCIAL_HEALTH_FEATURES,
    FUNDING_STABILITY_FEATURES,
    IDENTIFIERS,
    LIQUIDITY_RISK_FEATURES,
    ML_MODELS_CONFIG,
    MODEL_PARAMS,
    OFF_BALANCE_SHEET_FEATURES,
    RISK_PILLARS,
    RISK_SEVERITY_COLORS,
    SECTOR_LABELS,
    SECTOR_LOANS_COLUMNS,
)


# ── Identifiers ──────────────────────────────────────────────────────

class TestIdentifiers:
    def test_identifiers_count(self):
        assert len(IDENTIFIERS) == 5

    def test_required_identifiers_present(self):
        for col in ["bank_id", "period", "bank_type", "region"]:
            assert col in IDENTIFIERS


# ── Risk Pillars ─────────────────────────────────────────────────────

class TestRiskPillars:
    def test_seven_pillars(self):
        assert len(RISK_PILLARS) == 7

    def test_pillar_names(self):
        expected = {
            "Credit Risk", "Liquidity Risk", "Concentration Risk",
            "Capital Adequacy", "Earnings & Efficiency",
            "Off-Balance Sheet", "Funding Stability",
        }
        assert set(RISK_PILLARS.keys()) == expected

    def test_each_pillar_has_features(self):
        for name, feats in RISK_PILLARS.items():
            assert len(feats) > 0, f"Pillar '{name}' has no features"

    def test_pillar_feature_counts(self):
        assert len(CREDIT_RISK_FEATURES) == 4
        assert len(LIQUIDITY_RISK_FEATURES) == 3
        assert len(CONCENTRATION_RISK_FEATURES) == 3
        assert len(CAPITAL_ADEQUACY_FEATURES) == 3
        assert len(EARNINGS_EFFICIENCY_FEATURES) == 5
        assert len(OFF_BALANCE_SHEET_FEATURES) == 4
        assert len(FUNDING_STABILITY_FEATURES) == 4

    def test_all_ml_features_total_26(self):
        assert len(ALL_ML_FEATURES) == 26

    def test_all_ml_features_matches_pillars(self):
        """ALL_ML_FEATURES should be the flat union of all pillar features."""
        flat = [f for feats in RISK_PILLARS.values() for f in feats]
        assert ALL_ML_FEATURES == flat

    def test_no_duplicate_features(self):
        assert len(ALL_ML_FEATURES) == len(set(ALL_ML_FEATURES))

    def test_sector_loans_columns(self):
        assert len(SECTOR_LOANS_COLUMNS) == 5
        for col in SECTOR_LOANS_COLUMNS:
            assert col.startswith("sector_loans_")


# ── Backward-compatible aliases ──────────────────────────────────────

class TestAliases:
    def test_financial_health_features(self):
        expected = CREDIT_RISK_FEATURES + CAPITAL_ADEQUACY_FEATURES + EARNINGS_EFFICIENCY_FEATURES
        assert FINANCIAL_HEALTH_FEATURES == expected

    def test_exposure_liquidity_features(self):
        expected = OFF_BALANCE_SHEET_FEATURES + LIQUIDITY_RISK_FEATURES + FUNDING_STABILITY_FEATURES
        assert EXPOSURE_LIQUIDITY_FEATURES == expected


# ── Expert Rules ─────────────────────────────────────────────────────

class TestExpertRules:
    def test_nine_rules(self):
        assert len(EXPERT_RULES) == 9

    def test_rule_ids(self):
        expected = {"CAR_CRITICAL", "NPL_WARNING", "LCR_CRITICAL",
                    "LDR_WARNING", "HHI_HIGH_CONCENTRATION",
                    "NSFR_CRITICAL", "ROA_WARNING",
                    "WHOLESALE_DEPENDENCY_WARNING",
                    "TOP20_BORROWER_CONCENTRATION_WARNING"}
        assert set(EXPERT_RULES.keys()) == expected

    def test_rule_structure(self):
        required_keys = {"column", "op", "threshold", "severity", "pillar", "message"}
        for rule_id, rule in EXPERT_RULES.items():
            assert required_keys.issubset(rule.keys()), (
                f"Rule '{rule_id}' missing keys: {required_keys - rule.keys()}"
            )

    def test_rule_operators_valid(self):
        valid_ops = {"lt", "gt", "le", "ge", "eq", "ne"}
        for rule_id, rule in EXPERT_RULES.items():
            assert rule["op"] in valid_ops, f"Rule '{rule_id}' has invalid op '{rule['op']}'"

    def test_rule_severities(self):
        valid = {"critical", "warning"}
        for rule_id, rule in EXPERT_RULES.items():
            assert rule["severity"] in valid

    def test_rule_columns_in_ml_features(self):
        """Every rule column should reference a known ML feature."""
        for rule_id, rule in EXPERT_RULES.items():
            assert rule["column"] in ALL_ML_FEATURES, (
                f"Rule '{rule_id}' references unknown column '{rule['column']}'"
            )

    def test_rule_thresholds_positive(self):
        for rule_id, rule in EXPERT_RULES.items():
            assert isinstance(rule["threshold"], (int, float))

    def test_car_critical_threshold(self):
        assert EXPERT_RULES["CAR_CRITICAL"]["threshold"] == 0.08
        assert EXPERT_RULES["CAR_CRITICAL"]["op"] == "lt"

    def test_lcr_critical_threshold(self):
        assert EXPERT_RULES["LCR_CRITICAL"]["threshold"] == 1.0
        assert EXPERT_RULES["LCR_CRITICAL"]["op"] == "lt"

    def test_nsfr_critical_threshold(self):
        assert EXPERT_RULES["NSFR_CRITICAL"]["threshold"] == 1.0
        assert EXPERT_RULES["NSFR_CRITICAL"]["op"] == "lt"
        assert EXPERT_RULES["NSFR_CRITICAL"]["severity"] == "critical"
        assert EXPERT_RULES["NSFR_CRITICAL"]["pillar"] == "Liquidity Risk"

    def test_roa_warning_threshold(self):
        assert EXPERT_RULES["ROA_WARNING"]["threshold"] == 0.0
        assert EXPERT_RULES["ROA_WARNING"]["op"] == "lt"
        assert EXPERT_RULES["ROA_WARNING"]["severity"] == "warning"
        assert EXPERT_RULES["ROA_WARNING"]["pillar"] == "Earnings & Efficiency"

    def test_wholesale_dependency_warning_threshold(self):
        assert EXPERT_RULES["WHOLESALE_DEPENDENCY_WARNING"]["threshold"] == 0.5
        assert EXPERT_RULES["WHOLESALE_DEPENDENCY_WARNING"]["op"] == "gt"
        assert EXPERT_RULES["WHOLESALE_DEPENDENCY_WARNING"]["severity"] == "warning"
        assert EXPERT_RULES["WHOLESALE_DEPENDENCY_WARNING"]["pillar"] == "Funding Stability"

    def test_top20_borrower_concentration_warning_threshold(self):
        assert EXPERT_RULES["TOP20_BORROWER_CONCENTRATION_WARNING"]["threshold"] == 0.25
        assert EXPERT_RULES["TOP20_BORROWER_CONCENTRATION_WARNING"]["op"] == "gt"
        assert EXPERT_RULES["TOP20_BORROWER_CONCENTRATION_WARNING"]["severity"] == "warning"
        assert EXPERT_RULES["TOP20_BORROWER_CONCENTRATION_WARNING"]["pillar"] == "Concentration Risk"


# ── ML Models Config ─────────────────────────────────────────────────

class TestMLModelsConfig:
    def test_three_models(self):
        assert len(ML_MODELS_CONFIG) == 3

    def test_model_names(self):
        expected = {"IsolationForest", "LocalOutlierFactor", "OneClassSVM"}
        assert set(ML_MODELS_CONFIG.keys()) == expected

    def test_model_structure(self):
        for name, cfg in ML_MODELS_CONFIG.items():
            assert "class" in cfg, f"Model '{name}' missing 'class'"
            assert "params" in cfg, f"Model '{name}' missing 'params'"
            assert "description" in cfg, f"Model '{name}' missing 'description'"

    def test_isolation_forest_params(self):
        params = ML_MODELS_CONFIG["IsolationForest"]["params"]
        assert params["contamination"] == 0.05
        assert params["n_estimators"] == 300
        assert params["random_state"] == 42

    def test_lof_novelty_true(self):
        assert ML_MODELS_CONFIG["LocalOutlierFactor"]["params"]["novelty"] is True


# ── Model Params ─────────────────────────────────────────────────────

class TestModelParams:
    def test_required_keys(self):
        for key in ["contamination", "n_clusters", "random_state",
                     "ensemble_method", "scaling_strategy"]:
            assert key in MODEL_PARAMS

    def test_default_values(self):
        assert MODEL_PARAMS["contamination"] == 0.05
        assert MODEL_PARAMS["n_clusters"] == 3
        assert MODEL_PARAMS["random_state"] == 42
        assert MODEL_PARAMS["ensemble_method"] == "majority_vote"
        assert MODEL_PARAMS["scaling_strategy"] == "per_pillar"


# ── Feature Labels ───────────────────────────────────────────────────

class TestFeatureLabels:
    def test_all_ml_features_have_labels(self):
        for feat in ALL_ML_FEATURES:
            assert feat in FEATURE_LABELS, f"Feature '{feat}' has no label"

    def test_sector_labels(self):
        assert len(SECTOR_LABELS) == 5

    def test_severity_colors(self):
        assert set(RISK_SEVERITY_COLORS.keys()) == {"critical", "warning", "normal"}

