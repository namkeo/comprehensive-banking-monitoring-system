"""
risk_simulators.py – Stress-Testing Engine for BankGuard AI.
=============================================================

Provides macro-prudential stress-testing by applying user-defined shocks
to bank balance-sheet items and re-deriving key prudential ratios.

Shock Parameters
----------------
- ``npl_shock``          : multiplier on ``non_performing_loans``
                           (e.g. 1.5 → NPL increases 50 %).
- ``deposit_shock``      : multiplier on ``total_deposits``
                           (e.g. 0.9 → 10 % deposit run-off).
- ``asset_devaluation``  : multiplier on ``total_assets``
                           (e.g. 0.85 → 15 % asset write-down).

Impact Mechanics
----------------
1. **NPL shock** → ``non_performing_loans`` rises → additional
   ``loan_loss_provisions`` absorb the loss → ``tier1_capital`` is
   reduced by the incremental provision → ``capital_adequacy_ratio``
   is re-derived as ``(tier1 + tier2) / RWA``.
2. **Deposit shock** → ``total_deposits`` falls → ``net_cash_outflows_30d``
   increases proportionally → ``liquidity_coverage_ratio`` is re-derived
   as ``HQLA / net_cash_outflows_30d``.
3. **Asset devaluation** → ``total_assets`` falls → ``risk_weighted_assets``
   is scaled by the same factor → CAR is re-derived.

Breach Detection
----------------
Stressed ratios are compared against the thresholds defined in
``config.EXPERT_RULES`` (CAR < 8 %, LCR < 100 %, NPL > 3 %, etc.).

Author : BankGuard AI Team – Quantitative Risk Analyst
Created: 2026-03-06
"""

from __future__ import annotations

import operator
import os
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# -- Ensure project root is importable ------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import EXPERT_RULES

# Operator look-up (mirrors data_processor)
_OP_MAP = {
    "lt": operator.lt,
    "gt": operator.gt,
    "le": operator.le,
    "ge": operator.ge,
    "eq": operator.eq,
    "ne": operator.ne,
}

# Columns required by the stress engine
_REQUIRED_COLS: List[str] = [
    "bank_id",
    "non_performing_loans",
    "total_loans",
    "npl_ratio",
    "loan_loss_provisions",
    "tier1_capital",
    "tier2_capital",
    "risk_weighted_assets",
    "capital_adequacy_ratio",
    "total_deposits",
    "high_quality_liquid_assets",
    "net_cash_outflows_30d",
    "liquidity_coverage_ratio",
    "total_assets",
    "loan_to_deposit_ratio",
]


# ======================================================================
#  Validation
# ======================================================================

def _validate_inputs(
    df: pd.DataFrame,
    npl_shock: float,
    deposit_shock: float,
    asset_devaluation: float,
) -> None:
    """Raise ``ValueError`` / ``KeyError`` on bad inputs."""
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(
            f"Stress engine requires columns missing from the DataFrame: {missing}"
        )
    if npl_shock < 1.0:
        raise ValueError(
            f"npl_shock must be >= 1.0 (got {npl_shock}). "
            "Use 1.0 for no shock."
        )
    if not 0.0 < deposit_shock <= 1.0:
        raise ValueError(
            f"deposit_shock must be in (0, 1] (got {deposit_shock}). "
            "Use 1.0 for no shock."
        )
    if not 0.0 < asset_devaluation <= 1.0:
        raise ValueError(
            f"asset_devaluation must be in (0, 1] (got {asset_devaluation}). "
            "Use 1.0 for no shock."
        )




# ======================================================================
#  Stress Scenario Engine
# ======================================================================

def _apply_shocks(
    df: pd.DataFrame,
    npl_shock: float,
    deposit_shock: float,
    asset_devaluation: float,
) -> pd.DataFrame:
    """Apply shocks and re-derive prudential ratios.

    Returns a *stressed* copy of *df* with recalculated columns.
    """
    s = df.copy()

    # ---- 1. NPL shock ------------------------------------------------
    s["non_performing_loans"] = s["non_performing_loans"] * npl_shock
    s["npl_ratio"] = np.where(
        s["total_loans"] > 0,
        s["non_performing_loans"] / s["total_loans"],
        s["npl_ratio"],
    )

    # Additional provisions = incremental NPL (assume 100 % provisioning
    # on the incremental amount — conservative).
    incremental_npl = df["non_performing_loans"] * (npl_shock - 1.0)
    s["loan_loss_provisions"] = s["loan_loss_provisions"] + incremental_npl

    # Provisions eat into Tier-1 capital
    s["tier1_capital"] = (s["tier1_capital"] - incremental_npl).clip(lower=0)

    # ---- 2. Asset devaluation ----------------------------------------
    s["total_assets"] = s["total_assets"] * asset_devaluation
    s["risk_weighted_assets"] = s["risk_weighted_assets"] * asset_devaluation

    # ---- Re-derive CAR -----------------------------------------------
    s["capital_adequacy_ratio"] = np.where(
        s["risk_weighted_assets"] > 0,
        (s["tier1_capital"] + s["tier2_capital"]) / s["risk_weighted_assets"],
        0.0,
    )

    # ---- 3. Deposit shock --------------------------------------------
    s["total_deposits"] = s["total_deposits"] * deposit_shock

    # Deposit run-off increases net cash outflows proportionally
    deposit_loss = df["total_deposits"] * (1.0 - deposit_shock)
    s["net_cash_outflows_30d"] = s["net_cash_outflows_30d"] + deposit_loss

    # Re-derive LCR
    s["liquidity_coverage_ratio"] = np.where(
        s["net_cash_outflows_30d"] > 0,
        s["high_quality_liquid_assets"] / s["net_cash_outflows_30d"],
        0.0,
    )

    # Re-derive LDR
    s["loan_to_deposit_ratio"] = np.where(
        s["total_deposits"] > 0,
        s["total_loans"] / s["total_deposits"],
        s["loan_to_deposit_ratio"],
    )

    return s



# ======================================================================
#  Breach Detection
# ======================================================================

def _detect_breaches(df_check: pd.DataFrame) -> pd.DataFrame:
    """Compare ratios against ``EXPERT_RULES`` thresholds.

    Returns a DataFrame with one row per bank and boolean columns
    ``breach_<RULE_ID>`` plus a summary ``n_breaches`` count.
    """
    breach_df = df_check[["bank_id"]].copy()

    for rule_id, rule in EXPERT_RULES.items():
        col = rule["column"]
        if col not in df_check.columns:
            continue
        op_fn = _OP_MAP.get(rule["op"])
        if op_fn is None:
            continue
        breach_df[f"breach_{rule_id}"] = op_fn(
            df_check[col], rule["threshold"]
        )

    breach_cols = [c for c in breach_df.columns if c.startswith("breach_")]
    breach_df["n_breaches"] = breach_df[breach_cols].sum(axis=1).astype(int)
    return breach_df


# ======================================================================
#  Public API
# ======================================================================

# Metrics shown in the Baseline-vs-Stressed comparison
_COMPARISON_METRICS: List[str] = [
    "capital_adequacy_ratio",
    "liquidity_coverage_ratio",
    "npl_ratio",
    "loan_to_deposit_ratio",
    "tier1_capital",
    "non_performing_loans",
    "loan_loss_provisions",
    "total_deposits",
    "total_assets",
    "risk_weighted_assets",
]


def run_stress_scenario(
    df: pd.DataFrame,
    npl_shock: float = 1.5,
    deposit_shock: float = 0.9,
    asset_devaluation: float = 1.0,
) -> Dict[str, Any]:
    """Execute a macro stress-test scenario on the banking dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The *original* (unscaled) DataFrame produced by ``process_data``.
    npl_shock : float
        Multiplier on ``non_performing_loans`` (>= 1.0).
    deposit_shock : float
        Multiplier on ``total_deposits`` (0 < x <= 1.0).
    asset_devaluation : float
        Multiplier on ``total_assets`` (0 < x <= 1.0).

    Returns
    -------
    dict with keys:
        ``"comparison"``  – DataFrame (long format): bank_id, metric,
                            baseline, stressed, delta, delta_pct
        ``"breaches"``    – DataFrame: bank_id, breach_<RULE_ID>..., n_breaches
        ``"summary"``     – dict of aggregate statistics
        ``"params"``      – dict of the shock parameters used
    """
    _validate_inputs(df, npl_shock, deposit_shock, asset_devaluation)

    # --- Apply shocks -------------------------------------------------
    df_stressed = _apply_shocks(df, npl_shock, deposit_shock, asset_devaluation)

    # --- Build comparison table (long format) -------------------------
    avail_metrics = [m for m in _COMPARISON_METRICS if m in df.columns]
    rows: List[Dict[str, Any]] = []

    for idx in range(len(df)):
        bank_id = df.iloc[idx]["bank_id"]
        for metric in avail_metrics:
            b_val = float(df.iloc[idx][metric])
            s_val = float(df_stressed.iloc[idx][metric])
            delta = s_val - b_val
            delta_pct = (delta / b_val * 100) if b_val != 0 else 0.0
            rows.append({
                "bank_id": bank_id,
                "metric": metric,
                "baseline": round(b_val, 6),
                "stressed": round(s_val, 6),
                "delta": round(delta, 6),
                "delta_pct": round(delta_pct, 2),
            })

    comparison_df = pd.DataFrame(rows)

    # --- Breach detection ---------------------------------------------
    breaches_df = _detect_breaches(df_stressed)
    baseline_breaches = _detect_breaches(df)

    # --- Summary statistics -------------------------------------------
    summary = {
        "n_banks": int(df["bank_id"].nunique()),
        "npl_shock": npl_shock,
        "deposit_shock": deposit_shock,
        "asset_devaluation": asset_devaluation,
        "baseline_avg_car": round(float(df["capital_adequacy_ratio"].mean()), 4),
        "stressed_avg_car": round(float(df_stressed["capital_adequacy_ratio"].mean()), 4),
        "baseline_avg_lcr": round(float(df["liquidity_coverage_ratio"].mean()), 4),
        "stressed_avg_lcr": round(float(df_stressed["liquidity_coverage_ratio"].mean()), 4),
        "baseline_avg_npl": round(float(df["npl_ratio"].mean()), 4),
        "stressed_avg_npl": round(float(df_stressed["npl_ratio"].mean()), 4),
        "baseline_total_breaches": int(baseline_breaches["n_breaches"].sum()),
        "stressed_total_breaches": int(breaches_df["n_breaches"].sum()),
        "banks_with_new_breaches": int(
            (breaches_df["n_breaches"] > baseline_breaches["n_breaches"]).sum()
        ),
    }

    return {
        "comparison": comparison_df,
        "breaches": breaches_df,
        "summary": summary,
        "params": {
            "npl_shock": npl_shock,
            "deposit_shock": deposit_shock,
            "asset_devaluation": asset_devaluation,
        },
    }