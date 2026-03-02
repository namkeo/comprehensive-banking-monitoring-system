"""
data_processor.py - Data Ingestion & Transformation Pipeline for BankGuard AI.

All feature definitions are imported from ``config.py`` — the single source
of truth for column names across the project:

    - **Financial Health** (9 features) – stability, profitability, coverage
    - **Concentration Risk** (5 features) – borrower / depositor / sector HHI
    - **Exposure & Liquidity** (6 features) – OBS, wholesale, guarantees

A calculated feature ``risk_to_profit_ratio`` (NPL / ROA) is also created.

Author : BankGuard AI Team
Created: 2026-03-02
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── Ensure project root is importable ─────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import (
    ALL_ML_FEATURES,
    CONCENTRATION_RISK_FEATURES,
    EXPOSURE_LIQUIDITY_FEATURES,
    FINANCIAL_HEALTH_FEATURES,
    SECTOR_LOANS_COLUMNS,
)

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

DATA_PATH: str = os.path.join(_PROJECT_ROOT, "data", "time_series_dataset_enriched_v2.csv")

# Scaling groups – each group gets its own StandardScaler
FEATURE_GROUPS: Dict[str, List[str]] = {
    "financial_health": FINANCIAL_HEALTH_FEATURES,
    "concentration_risk": CONCENTRATION_RISK_FEATURES,
    "exposure_liquidity": EXPOSURE_LIQUIDITY_FEATURES,
}

# Flat list of every ML column across all groups (= config.ALL_ML_FEATURES)
ALL_GROUPED_FEATURES: List[str] = ALL_ML_FEATURES

# Calculated feature
RISK_TO_PROFIT_COL: str = "risk_to_profit_ratio"


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def load_data(filepath: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV dataset into a pandas DataFrame."""
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found at: {filepath}")
        df: pd.DataFrame = pd.read_csv(filepath)
        if df.empty:
            raise pd.errors.EmptyDataError("Loaded CSV is empty.")
        print(
            f"[DataProcessor] Loaded {len(df)} rows × "
            f"{len(df.columns)} columns from {filepath}"
        )
        return df
    except FileNotFoundError as e:
        print(f"[DataProcessor] ERROR – {e}")
        raise
    except pd.errors.EmptyDataError as e:
        print(f"[DataProcessor] ERROR – {e}")
        raise
    except Exception as e:
        print(f"[DataProcessor] Unexpected error while loading data – {e}")
        raise


def _impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Apply **median imputation** to every numerical column."""
    df = df.copy()
    numeric_cols: List[str] = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(
                f"[DataProcessor] Imputed {col} missing values "
                f"with median = {median_val:.6f}"
            )
    return df


def _validate_features(df: pd.DataFrame) -> None:
    """Ensure all ``config.ALL_ML_FEATURES`` columns exist.

    Raises:
        KeyError: If any ML feature is missing.
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
        print(f"[DataProcessor] WARNING – Optional sector columns missing: {missing_sector}")


def _create_risk_to_profit_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Create ``risk_to_profit_ratio = npl_ratio / roa`` (capped at 1e6)."""
    roa = df["return_on_assets"].replace(0, np.nan)
    ratio = df["npl_ratio"] / roa
    ratio = ratio.fillna(1e6).clip(upper=1e6)
    df[RISK_TO_PROFIT_COL] = ratio
    print(
        f"[DataProcessor] Created '{RISK_TO_PROFIT_COL}' – "
        f"range [{ratio.min():.4f}, {ratio.max():.4f}]"
    )
    return df


def _scale_feature_group(
    df: pd.DataFrame,
    columns: List[str],
    group_name: str,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """Fit a ``StandardScaler`` on *columns* and transform in-place."""
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    print(f"[DataProcessor] Scaled group '{group_name}' – {len(columns)} features")
    return df, scaler


def process_data(
    filepath: str = DATA_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, StandardScaler]]:
    """End-to-end data processing pipeline.

    Steps: Load → Impute → Validate → Engineer → Scale (3 groups, 20 features).

    Returns:
        (df_processed, df_original, scalers)
    """
    df_raw: pd.DataFrame = load_data(filepath)
    df_clean: pd.DataFrame = _impute_missing_values(df_raw)
    _validate_features(df_clean)
    df_clean = _create_risk_to_profit_ratio(df_clean)

    df_original: pd.DataFrame = df_clean.copy()
    df_processed: pd.DataFrame = df_clean.copy()
    scalers: Dict[str, StandardScaler] = {}

    for group_name, columns in FEATURE_GROUPS.items():
        df_processed, scaler = _scale_feature_group(df_processed, columns, group_name)
        scalers[group_name] = scaler

    total_scaled = sum(len(c) for c in FEATURE_GROUPS.values())
    print(
        f"[DataProcessor] Scaling complete – "
        f"{len(scalers)} groups, {total_scaled} features total"
    )
    print(
        f"[DataProcessor] Pipeline finished – "
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

    print(f"\n--- Calculated feature: {RISK_TO_PROFIT_COL} ---")
    print(df_orig[[RISK_TO_PROFIT_COL]].describe())
