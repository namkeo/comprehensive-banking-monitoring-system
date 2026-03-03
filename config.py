"""
config.py – Comprehensive 360° Banking Monitoring System
=========================================================

Hybrid ML + Rule-based configuration for the BankGuard AI engine.

Architecture
------------
- **8 Risk Pillars** (Feature Groups) – mapped to real dataset columns.
- **Expert Rules Engine** – Basel-aligned absolute thresholds for
  instant regulatory flagging (no ML required).
- **Ensemble ML Config** – three unsupervised anomaly detectors
  (Isolation Forest, Local Outlier Factor, One-Class SVM) whose
  predictions are fused via majority vote.

Author : BankGuard AI Team – Senior Banking Risk Architect
Created: 2026-03-02
"""

from __future__ import annotations

from typing import Any, Dict, List


# ═══════════════════════════════════════════════════════════════════════════
#  1. IDENTIFIERS – non-analytic columns used for indexing / filtering
# ═══════════════════════════════════════════════════════════════════════════

IDENTIFIERS: List[str] = [
    "bank_id",
    "period",
    "bank_type",
    "region",
    "external_credit_rating",
]


# ═══════════════════════════════════════════════════════════════════════════
#  2. EIGHT RISK PILLARS – Feature Groups
# ═══════════════════════════════════════════════════════════════════════════
#
#  Each pillar represents a distinct dimension of banking risk.
#  Columns marked (scaled) should be normalised by total_assets or another
#  denominator during preprocessing before entering ML models.
# ─────────────────────────────────────────────────────────────────────────

# Pillar 1 – Credit Risk
#   Measures asset-quality deterioration and provisioning adequacy.
CREDIT_RISK_FEATURES: List[str] = [
    "npl_ratio",                # Non-performing loans / Total loans
    "loan_growth_rate",         # YoY loan growth (%)
    "provision_coverage_ratio", # Loan-loss reserves / NPLs
    "non_performing_loans",     # Absolute NPL amount (scaled by total_assets)
]

# Pillar 2 – Liquidity Risk
#   Short- and structural-term liquidity buffers.
LIQUIDITY_RISK_FEATURES: List[str] = [
    "liquidity_coverage_ratio", # HQLA / Net cash outflows 30 d (Basel III LCR)
    "nsfr",                     # Available stable funding / Required stable funding
    "net_cash_outflows_30d",    # 30-day stressed net cash outflows (scaled)
]

# Pillar 3 – Concentration Risk
#   Name, sector and geographic concentration.
CONCENTRATION_RISK_FEATURES: List[str] = [
    "sector_concentration_hhi",     # Herfindahl-Hirschman Index on sector loans
    "top20_borrower_concentration", # Top-20 borrower exposure / Total loans
    "geographic_concentration",     # Geographic HHI
]

# Pillar 4 – Capital Adequacy
#   Solvency and loss-absorption capacity (Basel III).
CAPITAL_ADEQUACY_FEATURES: List[str] = [
    "capital_adequacy_ratio", # (Tier1 + Tier2) / RWA
    "tier1_capital",          # Core equity capital (scaled)
    "risk_weighted_assets",   # RWA (scaled)
]

# Pillar 5 – Earnings & Efficiency
#   Profitability trends and cost discipline.
EARNINGS_EFFICIENCY_FEATURES: List[str] = [
    "return_on_assets",    # ROA
    "return_on_equity",    # ROE
    "net_interest_margin", # NIM
    "operating_expenses",  # Total opex (scaled)
    "operating_income",    # Total operating income (scaled)
]

# Pillar 6 – Off-Balance Sheet Exposure
#   Contingent liabilities and derivative leverage.
OFF_BALANCE_SHEET_FEATURES: List[str] = [
    "obs_exposure_total",          # Total OBS notional (scaled)
    "guarantees_issued",           # Outstanding guarantees (scaled)
    "obs_to_assets_ratio",         # OBS / Total assets
    "derivatives_to_assets_ratio", # Derivatives notional / Total assets
]

# Pillar 7 – Funding Stability
#   Structural funding profile and deposit concentration.
FUNDING_STABILITY_FEATURES: List[str] = [
    "wholesale_dependency_ratio", # Wholesale funding / Total liabilities
    "loan_to_deposit_ratio",      # Loans / Deposits
    "top20_depositors_ratio",     # Top-20 depositors / Total deposits
    "deposit_growth_rate",        # YoY deposit growth (%)
]

# Pillar 8 – Sector Loan Allocation (visualisation / DNA profiling only)
SECTOR_LOANS_COLUMNS: List[str] = [
    "sector_loans_energy",
    "sector_loans_real_estate",
    "sector_loans_construction",
    "sector_loans_services",
    "sector_loans_agriculture",
]


# ── Derived convenience aggregates ──────────────────────────────────────

# All pillars that enter the ML feature vector (Pillars 1-7, 26 features)
RISK_PILLARS: Dict[str, List[str]] = {
    "Credit Risk":           CREDIT_RISK_FEATURES,
    "Liquidity Risk":        LIQUIDITY_RISK_FEATURES,
    "Concentration Risk":    CONCENTRATION_RISK_FEATURES,
    "Capital Adequacy":      CAPITAL_ADEQUACY_FEATURES,
    "Earnings & Efficiency": EARNINGS_EFFICIENCY_FEATURES,
    "Off-Balance Sheet":     OFF_BALANCE_SHEET_FEATURES,
    "Funding Stability":     FUNDING_STABILITY_FEATURES,
}

ALL_ML_FEATURES: List[str] = [
    feat
    for pillar_feats in RISK_PILLARS.values()
    for feat in pillar_feats
]

# ── Backward-compatible aliases (consumed by app.py, data_processor.py) ──
FINANCIAL_HEALTH_FEATURES: List[str] = (
    CREDIT_RISK_FEATURES + CAPITAL_ADEQUACY_FEATURES + EARNINGS_EFFICIENCY_FEATURES
)
EXPOSURE_LIQUIDITY_FEATURES: List[str] = (
    OFF_BALANCE_SHEET_FEATURES + LIQUIDITY_RISK_FEATURES + FUNDING_STABILITY_FEATURES
)


# ═══════════════════════════════════════════════════════════════════════════
#  3. EXPERT RULES ENGINE – Basel-aligned regulatory thresholds
# ═══════════════════════════════════════════════════════════════════════════
#
#  Each rule maps to a single column and defines:
#    • column    – dataset column to evaluate
#    • op        – comparison operator ("lt" = less than, "gt" = greater than)
#    • threshold – absolute numeric boundary
#    • severity  – "critical" (immediate action) or "warning" (watch-list)
#    • pillar    – risk pillar the rule belongs to
#    • message   – human-readable alert text
# ─────────────────────────────────────────────────────────────────────────

EXPERT_RULES: Dict[str, Dict[str, Any]] = {
    "CAR_CRITICAL": {
        "column":    "capital_adequacy_ratio",
        "op":        "lt",
        "threshold": 0.08,
        "severity":  "critical",
        "pillar":    "Capital Adequacy",
        "message":   "CAR below 8% minimum – bank is under-capitalised (Basel III Pillar 1 breach).",
    },
    "NPL_WARNING": {
        "column":    "npl_ratio",
        "op":        "gt",
        "threshold": 0.03,
        "severity":  "warning",
        "pillar":    "Credit Risk",
        "message":   "NPL ratio exceeds 3% – elevated credit-quality deterioration.",
    },
    "LCR_CRITICAL": {
        "column":    "liquidity_coverage_ratio",
        "op":        "lt",
        "threshold": 1.0,
        "severity":  "critical",
        "pillar":    "Liquidity Risk",
        "message":   "LCR below 100% – insufficient high-quality liquid assets for 30-day stress.",
    },
    "LDR_WARNING": {
        "column":    "loan_to_deposit_ratio",
        "op":        "gt",
        "threshold": 1.0,
        "severity":  "warning",
        "pillar":    "Funding Stability",
        "message":   "Loan-to-Deposit ratio above 100% – loans exceed deposit base, funding risk.",
    },
    "HHI_HIGH_CONCENTRATION": {
        "column":    "sector_concentration_hhi",
        "op":        "gt",
        "threshold": 0.25,
        "severity":  "warning",
        "pillar":    "Concentration Risk",
        "message":   "Sector HHI above 0.25 – high lending concentration in few sectors.",
    },
    "NSFR_CRITICAL": {
        "column":    "nsfr",
        "op":        "lt",
        "threshold": 1.0,
        "severity":  "critical",
        "pillar":    "Liquidity Risk",
        "message":   "NSFR below 100% – insufficient stable funding relative to required stable funding (Basel III breach).",
    },
    "ROA_WARNING": {
        "column":    "return_on_assets",
        "op":        "lt",
        "threshold": 0.0,
        "severity":  "warning",
        "pillar":    "Earnings & Efficiency",
        "message":   "ROA below 0% – bank is operating at a loss, negative profitability.",
    },
    "WHOLESALE_DEPENDENCY_WARNING": {
        "column":    "wholesale_dependency_ratio",
        "op":        "gt",
        "threshold": 0.5,
        "severity":  "warning",
        "pillar":    "Funding Stability",
        "message":   "Wholesale dependency ratio above 50% – excessive reliance on wholesale funding.",
    },
    "TOP20_BORROWER_CONCENTRATION_WARNING": {
        "column":    "top20_borrower_concentration",
        "op":        "gt",
        "threshold": 0.25,
        "severity":  "warning",
        "pillar":    "Concentration Risk",
        "message":   "Top-20 borrower concentration above 25% – high single-name credit risk.",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
#  4. ML MODELS CONFIG – Ensemble of 3 unsupervised anomaly detectors
# ═══════════════════════════════════════════════════════════════════════════
#
#  Final anomaly label = majority vote across the three detectors.
#  Each entry carries the sklearn class path, instantiation kwargs,
#  and a human-readable description.
# ─────────────────────────────────────────────────────────────────────────

ML_MODELS_CONFIG: Dict[str, Dict[str, Any]] = {
    "IsolationForest": {
        "class": "sklearn.ensemble.IsolationForest",
        "params": {
            "contamination": 0.05,
            "n_estimators":  300,
            "max_features":  0.5,
            "random_state":  42,
            "n_jobs":        -1,
        },
        "description": (
            "Tree-based anomaly detector. Isolates observations by randomly "
            "selecting a feature and a split value; anomalies require fewer "
            "splits → shorter average path length."
        ),
    },
    "LocalOutlierFactor": {
        "class": "sklearn.neighbors.LocalOutlierFactor",
        "params": {
            "n_neighbors":   20,
            "contamination": 0.05,
            "novelty":       True,
            "metric":        "euclidean",
            "n_jobs":        -1,
        },
        "description": (
            "Density-based detector. Compares the local density of a point "
            "to its k-nearest neighbours; points in significantly sparser "
            "regions are flagged as outliers."
        ),
    },
    "OneClassSVM": {
        "class": "sklearn.svm.OneClassSVM",
        "params": {
            "kernel": "rbf",
            "gamma":  "scale",
            "nu":     0.05,
        },
        "description": (
            "Boundary-based detector. Learns a decision boundary around "
            "normal data in kernel-transformed space; points outside the "
            "boundary are classified as anomalies."
        ),
    },
}


# ═══════════════════════════════════════════════════════════════════════════
#  5. GENERAL PIPELINE PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

MODEL_PARAMS: Dict[str, Any] = {
    "contamination":    0.05,            # Expected anomaly rate (shared default)
    "n_clusters":       3,               # K-Means risk clusters (Low / Medium / High)
    "random_state":     42,              # Global reproducibility seed
    "ensemble_method":  "majority_vote", # How to fuse multi-model predictions
    "scaling_strategy": "per_pillar",    # One StandardScaler per risk pillar
}


# ═══════════════════════════════════════════════════════════════════════════
#  6. HUMAN-READABLE LABELS (for dashboard display)
# ═══════════════════════════════════════════════════════════════════════════

FEATURE_LABELS: Dict[str, str] = {
    # Credit Risk
    "npl_ratio":                    "NPL Ratio",
    "loan_growth_rate":             "Loan Growth Rate",
    "provision_coverage_ratio":     "Provision Coverage",
    "non_performing_loans":         "Non-Performing Loans",
    # Liquidity Risk
    "liquidity_coverage_ratio":     "Liquidity Coverage (LCR)",
    "nsfr":                         "NSFR",
    "net_cash_outflows_30d":        "Net Cash Outflows 30d",
    # Concentration Risk
    "sector_concentration_hhi":     "Sector HHI",
    "top20_borrower_concentration": "Top-20 Borrower Conc.",
    "geographic_concentration":     "Geographic Conc.",
    # Capital Adequacy
    "capital_adequacy_ratio":       "Capital Adequacy (CAR)",
    "tier1_capital":                "Tier-1 Capital",
    "risk_weighted_assets":         "Risk-Weighted Assets",
    # Earnings & Efficiency
    "return_on_assets":             "ROA",
    "return_on_equity":             "ROE",
    "net_interest_margin":          "Net Interest Margin",
    "operating_expenses":           "Operating Expenses",
    "operating_income":             "Operating Income",
    # Off-Balance Sheet
    "obs_exposure_total":           "OBS Exposure Total",
    "guarantees_issued":            "Guarantees Issued",
    "obs_to_assets_ratio":          "OBS / Assets",
    "derivatives_to_assets_ratio":  "Derivatives / Assets",
    # Funding Stability
    "wholesale_dependency_ratio":   "Wholesale Dependency",
    "loan_to_deposit_ratio":        "Loan-to-Deposit",
    "top20_depositors_ratio":       "Top-20 Depositors",
    "deposit_growth_rate":          "Deposit Growth Rate",
}

SECTOR_LABELS: Dict[str, str] = {
    "sector_loans_energy":       "Energy",
    "sector_loans_real_estate":  "Real Estate",
    "sector_loans_construction": "Construction",
    "sector_loans_services":     "Services",
    "sector_loans_agriculture":  "Agriculture",
}

RISK_SEVERITY_COLORS: Dict[str, str] = {
    "critical": "#e74c3c",
    "warning":  "#f39c12",
    "normal":   "#2ecc71",
}
