"""
data_processor.py – Data Ingestion & Transformation Pipeline for BankGuard AI.
==============================================================================

Ingests the **360° config.py** (8 Risk Pillars, Expert Rules, ML Models Config)
and executes a five-stage pipeline:

    1. **Load**       – CSV ingestion with basic validation.
    2. **Impute**     – Median imputation for every numeric column.
    3. **Validate**   – Ensure all 26 ML feature columns exist.
    4. **Engineer**   – Derived features: ``risk_to_profit_ratio``,
                        ``efficiency_ratio``.
    5. **Rule Engine**– Evaluate ``EXPERT_RULES`` thresholds from config.py →
                        ``rule_violations`` (list of triggered rule IDs) and
                        ``rule_risk_score`` (count of violations) per row.
    6. **Scale**      – One ``StandardScaler`` per risk pillar (7 groups,
                        26 features total) for ML input.

Returns
-------
``(df_processed, df_original, scalers)``

Author : BankGuard AI Team – Senior Data Engineer
Created: 2026-03-02
"""

from __future__ import annotations

import operator
import os
import sys
from typing import Any, Dict, IO, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── Ensure project root is importable ─────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import (
    ALL_ML_FEATURES,
    CREDIT_RISK_FEATURES,
    LIQUIDITY_RISK_FEATURES,
    CONCENTRATION_RISK_FEATURES,
    CAPITAL_ADEQUACY_FEATURES,
    EARNINGS_EFFICIENCY_FEATURES,
    OFF_BALANCE_SHEET_FEATURES,
    FUNDING_STABILITY_FEATURES,
    EXPERT_RULES,
    IDENTIFIERS,
    RISK_PILLARS,
    SECTOR_LOANS_COLUMNS,
)

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

DATA_PATH: str = os.path.join(
    _PROJECT_ROOT, "data", "time_series_dataset_enriched_v2.csv"
)

# Scaling groups – one StandardScaler per risk pillar (7 pillars from config)
FEATURE_GROUPS: Dict[str, List[str]] = {
    name.lower().replace(" ", "_").replace("&", "and"): feats
    for name, feats in RISK_PILLARS.items()
}

# Flat list of every ML column across all groups (= config.ALL_ML_FEATURES)
ALL_GROUPED_FEATURES: List[str] = ALL_ML_FEATURES

# Calculated features
RISK_TO_PROFIT_COL: str = "risk_to_profit_ratio"
EFFICIENCY_RATIO_COL: str = "efficiency_ratio"

# Operator look-up for the rule engine
_OP_MAP = {
    "lt": operator.lt,
    "gt": operator.gt,
    "le": operator.le,
    "ge": operator.ge,
    "eq": operator.eq,
    "ne": operator.ne,
}


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 1 – Load
# ═══════════════════════════════════════════════════════════════════════════

def load_data(
    filepath: str = DATA_PATH,
    uploaded_file: Optional[Union[IO, Any]] = None,
) -> pd.DataFrame:
    """Load the raw CSV dataset into a pandas DataFrame.

    Parameters
    ----------
    filepath : str
        Path to a local CSV file (used when *uploaded_file* is ``None``).
    uploaded_file : file-like object, optional
        A file-like object (e.g. from ``st.file_uploader``).  When provided
        the CSV is read directly from this object and *filepath* is ignored.
    """
    try:
        if uploaded_file is not None:
            df: pd.DataFrame = pd.read_csv(uploaded_file)
            source_label = getattr(uploaded_file, "name", "uploaded file")
            if df.empty:
                raise pd.errors.EmptyDataError("Uploaded CSV is empty.")
            print(
                f"[DataProcessor] Loaded {len(df)} rows x "
                f"{len(df.columns)} columns from {source_label}"
            )
            return df

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found at: {filepath}")
        df = pd.read_csv(filepath)
        if df.empty:
            raise pd.errors.EmptyDataError("Loaded CSV is empty.")
        print(
            f"[DataProcessor] Loaded {len(df)} rows x "
            f"{len(df.columns)} columns from "
            f"{os.path.basename(filepath)}"
        )
        return df
    except FileNotFoundError as e:
        print(f"[DataProcessor] ERROR - {e}")
        raise
    except pd.errors.EmptyDataError as e:
        print(f"[DataProcessor] ERROR - {e}")
        raise
    except Exception as e:
        print(f"[DataProcessor] Unexpected error while loading data - {e}")
        raise


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 2 – Impute
# ═══════════════════════════════════════════════════════════════════════════

def _impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Apply **median imputation** to every numerical column."""
    df = df.copy()
    numeric_cols: List[str] = df.select_dtypes(include=[np.number]).columns.tolist()
    n_imputed = 0
    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            n_imputed += 1
            print(
                f"[DataProcessor] Imputed {col} missing values "
                f"with median = {median_val:.6f}"
            )
    if n_imputed == 0:
        print("[DataProcessor] No missing values detected - imputation skipped.")
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 3 – Validate
# ═══════════════════════════════════════════════════════════════════════════

def _validate_features(df: pd.DataFrame) -> None:
    """Ensure all 26 ``config.ALL_ML_FEATURES`` columns exist.

    Raises:
        KeyError: If any ML feature is missing from the DataFrame.
    """
    missing: List[str] = [col for col in ALL_ML_FEATURES if col not in df.columns]
    if missing:
        raise KeyError(
            f"[DataProcessor] The following required ML columns are missing "
            f"from the dataset: {missing}"
        )
    # Warn (don't raise) for optional sector columns
    missing_sector = [c for c in SECTOR_LOANS_COLUMNS if c not in df.columns]
    if missing_sector:
        print(
            f"[DataProcessor] WARNING - Optional sector columns missing: "
            f"{missing_sector}"
        )

    print(f"[DataProcessor] Validation passed - "
        f"all {len(ALL_ML_FEATURES)} ML features present, "
        f"across {len(RISK_PILLARS)} risk pillars."
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 4 – Feature Engineering
# ═══════════════════════════════════════════════════════════════════════════

def _create_risk_to_profit_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Create ``risk_to_profit_ratio = npl_ratio / ROA`` (capped at 1 e6)."""
    roa = df["return_on_assets"].replace(0, np.nan)
    ratio = df["npl_ratio"] / roa
    ratio = ratio.fillna(1e6).clip(upper=1e6)
    df[RISK_TO_PROFIT_COL] = ratio
    print(
        f"[DataProcessor] Created '{RISK_TO_PROFIT_COL}' - "
        f"range [{ratio.min():.4f}, {ratio.max():.4f}]"
    )
    return df


def _create_efficiency_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Create ``efficiency_ratio = operating_expenses / operating_income``.

    Lower is better; values > 1.0 mean the bank spends more than it earns.
    Division-by-zero or ≤ 0 income rows are capped at 10.0.
    """
    income = df["operating_income"].replace(0, np.nan)
    ratio = df["operating_expenses"] / income
    ratio = ratio.fillna(10.0).clip(upper=10.0)
    df[EFFICIENCY_RATIO_COL] = ratio
    print(
        f"[DataProcessor] Created '{EFFICIENCY_RATIO_COL}' - "
        f"range [{ratio.min():.4f}, {ratio.max():.4f}]"
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 5 – Expert Rule Engine
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_expert_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate every row against ``config.EXPERT_RULES``.

    For each bank-period observation the function:
    1. Checks each rule (column, operator, threshold).
    2. Collects the IDs of triggered rules into ``rule_violations``
       (list[str]).  Rows with no violations get ``["Compliant"]``.
    3. Counts violations as ``rule_risk_score`` (int).

    Parameters
    ----------
    df : pd.DataFrame
        The *original* (unscaled) DataFrame with all raw feature columns.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with two new columns:
        ``rule_violations`` and ``rule_risk_score``.
    """
    df = df.copy()

    # Pre-compute boolean masks for each rule (vectorised – fast)
    violation_masks: Dict[str, pd.Series] = {}
    for rule_id, rule in EXPERT_RULES.items():
        col: str = rule["column"]
        if col not in df.columns:
            print(
                f"[DataProcessor] WARNING – Rule '{rule_id}' references "
                f"column '{col}' which is absent; skipping."
            )
            continue
        op_fn = _OP_MAP.get(rule["op"])
        if op_fn is None:
            print(
                f"[DataProcessor] WARNING – Rule '{rule_id}' has unknown "
                f"operator '{rule['op']}'; skipping."
            )
            continue
        threshold: float = rule["threshold"]
        violation_masks[rule_id] = op_fn(df[col], threshold)

    # Build per-row violation lists using the pre-computed masks
    n_rows = len(df)
    violations: List[List[str]] = [[] for _ in range(n_rows)]
    for rule_id, mask in violation_masks.items():
        for idx in mask[mask].index:
            pos = df.index.get_loc(idx)
            violations[pos].append(rule_id)

    # Convert empty lists → ["Compliant"]
    df["rule_violations"] = [
        v if v else ["Compliant"] for v in violations
    ]
    df["rule_risk_score"] = [len(v) if v != ["Compliant"] else 0 for v in df["rule_violations"]]

    # Logging summary
    total_violations = int(df["rule_risk_score"].sum())
    n_non_compliant = int((df["rule_risk_score"] > 0).sum())
    print(
        f"[DataProcessor] Expert Rule Engine -> "
        f"{total_violations} total violations across "
        f"{n_non_compliant}/{n_rows} observations."
    )
    for rule_id, mask in violation_masks.items():
        cnt = int(mask.sum())
        severity = EXPERT_RULES[rule_id]["severity"].upper()
        if cnt > 0:
            print(
                f"[DataProcessor]   Rule {rule_id:24s} [{severity:8s}] "
                f"– triggered {cnt} times"
            )

    return df


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 6 – Scaling
# ═══════════════════════════════════════════════════════════════════════════

def _scale_feature_group(
    df: pd.DataFrame,
    columns: List[str],
    group_name: str,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """Fit a ``StandardScaler`` on *columns* and transform in-place."""
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    print(f"[DataProcessor] Scaled group '{group_name}' - {len(columns)} features")
    return df, scaler


# ═══════════════════════════════════════════════════════════════════════════
#  Public API – End-to-end pipeline
# ═══════════════════════════════════════════════════════════════════════════

def process_data(
    filepath: str = DATA_PATH,
    uploaded_file: Optional[Union[IO, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, StandardScaler]]:
    """End-to-end data processing pipeline.

    Stages
    ------
    1. Load CSV (from *uploaded_file* if provided, else from *filepath*)
    2. Median-impute missing values
    3. Validate all 26 ML feature columns exist
    4. Feature engineering (risk_to_profit_ratio, efficiency_ratio)
    5. Expert Rule Engine → rule_violations, rule_risk_score
    6. StandardScaler per risk pillar (7 groups, 26 features)

    Returns
    -------
    (df_processed, df_original, scalers)
        * ``df_processed`` – scaled features ready for ML models.
        * ``df_original``  – unscaled, enriched with rule engine columns.
        * ``scalers``      – dict[group_name → fitted StandardScaler].
    """
    # Stage 1 – Load
    df_raw: pd.DataFrame = load_data(filepath, uploaded_file=uploaded_file)

    # Stage 2 – Impute
    df_clean: pd.DataFrame = _impute_missing_values(df_raw)

    # Stage 3 – Validate
    _validate_features(df_clean)

    # Stage 4 – Feature Engineering
    df_clean = _create_risk_to_profit_ratio(df_clean)
    df_clean = _create_efficiency_ratio(df_clean)

    # Stage 5 – Expert Rule Engine (runs on unscaled data)
    df_clean = evaluate_expert_rules(df_clean)

    # Snapshot the original (unscaled) with rule columns
    df_original: pd.DataFrame = df_clean.copy()

    # Stage 6 – Scale for ML
    df_processed: pd.DataFrame = df_clean.copy()
    scalers: Dict[str, StandardScaler] = {}

    for group_name, columns in FEATURE_GROUPS.items():
        df_processed, scaler = _scale_feature_group(
            df_processed, columns, group_name
        )
        scalers[group_name] = scaler

    total_scaled = sum(len(c) for c in FEATURE_GROUPS.values())
    print(
        f"[DataProcessor] Scaling complete - "
        f"{len(scalers)} groups, {total_scaled} features total"
    )
    print(
        f"[DataProcessor] Pipeline finished - "
        f"df_processed: {df_processed.shape}, "
        f"df_original: {df_original.shape}"
    )
    return df_processed, df_original, scalers


# ---------------------------------------------------------------------------
#  Convenience: quick sanity-check when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df_proc, df_orig, fitted_scalers = process_data()

    print("\n--- Feature groups & scaler summary ---")
    for gname, scaler in fitted_scalers.items():
        cols = FEATURE_GROUPS[gname]
        print(f"\n  Group: {gname} ({len(cols)} features)")
        print(f"    Columns : {cols}")
        print(f"    Means   : {dict(zip(cols, scaler.mean_))}")
        print(f"    Std devs: {dict(zip(cols, scaler.scale_))}")

    print(f"\n--- Processed (scaled) sample [{len(ALL_GROUPED_FEATURES)} features] ---")
    print(df_proc[ALL_GROUPED_FEATURES].head())

    print(f"\n--- Original (unscaled) sample ---")
    print(df_orig[ALL_GROUPED_FEATURES].head())

    print(f"\n--- Calculated features ---")
    print(df_orig[[RISK_TO_PROFIT_COL, EFFICIENCY_RATIO_COL]].describe())

    print(f"\n--- Expert Rule Engine results ---")
    print(f"  Columns added: rule_violations, rule_risk_score")
    print(f"  Non-compliant: {(df_orig['rule_risk_score'] > 0).sum()} / {len(df_orig)}")
    non_compliant = df_orig[df_orig["rule_risk_score"] > 0]
    if not non_compliant.empty:
        sample_cols = ["bank_id", "period", "rule_violations", "rule_risk_score"]
        sample_cols = [c for c in sample_cols if c in non_compliant.columns]
        print(non_compliant[sample_cols].head(10).to_string(index=False))

