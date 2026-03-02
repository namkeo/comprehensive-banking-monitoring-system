"""
anomaly_detector.py - Core Analytics Engine for BankGuard AI.

All feature definitions are imported from ``config.py``:
    1. **Multivariate Anomaly Detection** – Isolation Forest trained on the
       full ``ALL_ML_FEATURES`` set (20 dimensions: Financial Health 9 +
       Concentration Risk 5 + Exposure & Liquidity 6).
    2. **Enhanced Risk Clustering** – K-Means (k=3) with sector-loan DNA
       profiling per cluster.
    3. **Deep Explainability (XAI)** – per-anomaly driver identification
       across all 3 risk groups **plus** group-level attribution
       (``anomaly_driver_group``) and OBS risk contribution analysis.

Author : BankGuard AI Team
Created: 2026-03-02
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
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
    MODEL_PARAMS,
    SECTOR_LOANS_COLUMNS,
)

# ---------------------------------------------------------------------------
#  Logger setup
# ---------------------------------------------------------------------------

logger = logging.getLogger("BankGuard.AnomalyDetector")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter(
            "[%(name)s] %(asctime)s – %(levelname)s – %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
#  Constants derived from config.py
# ---------------------------------------------------------------------------

# The full feature vector used for Isolation Forest & K-Means
MULTIVARIATE_FEATURES: List[str] = ALL_ML_FEATURES

# Mapping: group label → feature list (for group-level driver attribution)
DRIVER_GROUPS: Dict[str, List[str]] = {
    "Financial Health": FINANCIAL_HEALTH_FEATURES,
    "Concentration Risk": CONCENTRATION_RISK_FEATURES,
    "Exposure & Liquidity": EXPOSURE_LIQUIDITY_FEATURES,
}

# Reverse mapping: feature → group label
_FEATURE_TO_GROUP: Dict[str, str] = {
    feat: group
    for group, feats in DRIVER_GROUPS.items()
    for feat in feats
}

# Pretty labels for sector columns
SECTOR_LABELS: Dict[str, str] = {
    "sector_loans_energy": "Energy",
    "sector_loans_real_estate": "Real Estate",
    "sector_loans_construction": "Construction",
    "sector_loans_services": "Services",
    "sector_loans_agriculture": "Agriculture",
}

# OBS risk column for deep explanation
OBS_RISK_COL: str = "obs_risk_indicator"

# Human-readable risk labels ordered lowest → highest
RISK_LABELS: List[str] = ["Low Risk", "Medium Risk", "High Risk"]


# ═══════════════════════════════════════════════════════════════════════════
#  Main class
# ═══════════════════════════════════════════════════════════════════════════

class BankAnomalyDetector:
    """Multivariate unsupervised analytics engine.

    * **Isolation Forest** on ``ALL_ML_FEATURES`` (20 dims).
    * **K-Means (k=3)** with sector-loan DNA profiling.
    * **Explainability**: per-anomaly feature driver + group attribution +
      OBS risk contribution.

    Attributes:
        contamination, n_clusters, random_state: Hyper-parameters.
        iso_forest: Fitted ``IsolationForest``.
        kmeans: Fitted ``KMeans``.
        cluster_dna: Dict mapping cluster label → sector description.
    """

    def __init__(
        self,
        contamination: float = MODEL_PARAMS["contamination"],
        n_clusters: int = MODEL_PARAMS["n_clusters"],
        random_state: int = MODEL_PARAMS["random_state"],
    ) -> None:
        self.contamination = contamination
        self.n_clusters = n_clusters
        self.random_state = random_state

        self.iso_forest: Optional[IsolationForest] = None
        self.kmeans: Optional[KMeans] = None
        self.cluster_dna: Dict[str, str] = {}

        logger.info(
            "BankAnomalyDetector initialised  "
            "(contamination=%.2f, n_clusters=%d, seed=%d, "
            "multivariate_dim=%d)",
            contamination, n_clusters, random_state,
            len(MULTIVARIATE_FEATURES),
        )

    # ------------------------------------------------------------------
    #  1. Anomaly Detection – Isolation Forest on ALL_ML_FEATURES
    # ------------------------------------------------------------------

    def detect_anomalies(self, df_scaled: pd.DataFrame) -> pd.DataFrame:
        """Run Isolation Forest on the full 20-dim scaled feature vector.

        Returns:
            Copy of *df_scaled* with ``anomaly_score`` and ``is_anomaly``.
        """
        df = df_scaled.copy()
        features = df[MULTIVARIATE_FEATURES]

        if len(df) <= 1:
            logger.warning("Input has %d row(s) – marking all as normal.", len(df))
            df["anomaly_score"] = 0.0
            df["is_anomaly"] = 1
            return df

        logger.info(
            "Training Isolation Forest on %d samples × %d features "
            "(contamination=%.2f) …",
            len(features), features.shape[1], self.contamination,
        )
        logger.info("  Feature vector: %s", ", ".join(MULTIVARIATE_FEATURES))

        self.iso_forest = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=300,
            max_features=min(1.0, 10 / len(MULTIVARIATE_FEATURES)),
            n_jobs=-1,
        )
        self.iso_forest.fit(features)

        df["anomaly_score"] = self.iso_forest.decision_function(features)
        df["is_anomaly"] = self.iso_forest.predict(features)

        n_anom = int((df["is_anomaly"] == -1).sum())
        logger.info(
            "Isolation Forest complete – %d anomalies / %d banks (%.1f%%).",
            n_anom, len(df), 100.0 * n_anom / len(df),
        )
        return df

    # ------------------------------------------------------------------
    #  2. Risk Clustering – K-Means + Sector-Loan DNA
    # ------------------------------------------------------------------

    def cluster_banks(
        self,
        df_scaled: pd.DataFrame,
        df_original: pd.DataFrame,
    ) -> pd.DataFrame:
        """K-Means clustering with data-driven risk labels and sector DNA.

        Returns:
            Copy of *df_scaled* with ``cluster_label`` and ``cluster_dna``.
        """
        df = df_scaled.copy()
        features = df[MULTIVARIATE_FEATURES]

        if len(df) <= 1:
            df["cluster_label"] = "Medium Risk"
            df["cluster_dna"] = "Insufficient data"
            return df

        effective_k = min(self.n_clusters, len(df))
        logger.info(
            "Training K-Means (k=%d) on %d samples × %d features …",
            effective_k, len(features), features.shape[1],
        )

        self.kmeans = KMeans(
            n_clusters=effective_k,
            random_state=self.random_state,
            n_init=10, max_iter=300,
        )
        raw_labels = self.kmeans.fit_predict(features)
        df["_raw_cluster"] = raw_labels

        # ── Risk-label mapping ───────────────────────────────────────
        cluster_stats = (
            df_original[["npl_ratio", "capital_adequacy_ratio"]]
            .assign(_raw_cluster=raw_labels)
            .groupby("_raw_cluster").mean()
        )
        cluster_stats["risk_composite"] = (
            cluster_stats["npl_ratio"] - cluster_stats["capital_adequacy_ratio"]
        )
        sorted_clusters = cluster_stats["risk_composite"].sort_values().index.tolist()
        label_map = {cid: RISK_LABELS[rank] for rank, cid in enumerate(sorted_clusters)}
        df["cluster_label"] = df["_raw_cluster"].map(label_map)

        for lbl in RISK_LABELS[:effective_k]:
            logger.info("  Cluster %-12s : %d banks", lbl, int((df["cluster_label"] == lbl).sum()))

        # ── Sector-loan DNA ──────────────────────────────────────────
        avail_sector = [c for c in SECTOR_LOANS_COLUMNS if c in df_original.columns]
        self.cluster_dna = {}

        if avail_sector:
            sector_df = (
                df_original[avail_sector]
                .assign(_raw_cluster=raw_labels)
                .groupby("_raw_cluster").mean()
            )
            for cid in sorted_clusters:
                risk_label = label_map[cid]
                row = sector_df.loc[cid]
                total = row.sum()
                proportions = (row / total * 100).sort_values(ascending=False) if total > 0 else row.sort_values(ascending=False)

                top_name = SECTOR_LABELS.get(proportions.index[0], proportions.index[0])
                dna = f"High {top_name} Exposure ({proportions.iloc[0]:.1f}%)"
                if len(proportions) > 1 and proportions.iloc[1] > 20:
                    second = SECTOR_LABELS.get(proportions.index[1], proportions.index[1])
                    dna += f" + {second} ({proportions.iloc[1]:.1f}%)"

                self.cluster_dna[risk_label] = dna
                logger.info("  DNA %-12s : %s", risk_label, dna)

            df["cluster_dna"] = df["cluster_label"].map(self.cluster_dna)
        else:
            logger.warning("No sector_loans_* columns – skipping DNA.")
            df["cluster_dna"] = "N/A"
            self.cluster_dna = {lbl: "N/A" for lbl in RISK_LABELS}

        df.drop(columns=["_raw_cluster"], inplace=True)
        logger.info("K-Means + DNA profiling complete.")
        return df

    # ------------------------------------------------------------------
    #  3. Explainability – Anomaly Drivers (feature + group level)
    # ------------------------------------------------------------------

    def get_anomaly_drivers(
        self,
        df_original: pd.DataFrame,
        anomaly_mask: pd.Series,
    ) -> Tuple[pd.Series, pd.Series]:
        """Identify per-anomaly primary driver feature AND the risk group.

        For each anomaly computes absolute z-score deviations across all
        20 ``ALL_ML_FEATURES``. The feature with the largest deviation is
        the ``anomaly_driver``; the group it belongs to (Financial Health,
        Concentration Risk, Exposure & Liquidity) is ``anomaly_driver_group``.

        Returns:
            Tuple of (anomaly_driver, anomaly_driver_group) Series.
        """
        drivers = pd.Series("N/A", index=df_original.index)
        driver_groups = pd.Series("N/A", index=df_original.index)
        bool_mask = (anomaly_mask == -1)

        if not bool_mask.any():
            logger.info("No anomalies – skipping driver analysis.")
            return drivers, driver_groups

        logger.info("Computing anomaly drivers for %d flagged banks …", int(bool_mask.sum()))

        medians = df_original[ALL_ML_FEATURES].median()
        stds = df_original[ALL_ML_FEATURES].std().replace(0, np.nan)

        anomaly_rows = df_original.loc[bool_mask, ALL_ML_FEATURES]
        z_devs = (anomaly_rows - medians).abs() / stds
        top_feature = z_devs.idxmax(axis=1)
        drivers.loc[bool_mask] = top_feature
        driver_groups.loc[bool_mask] = top_feature.map(_FEATURE_TO_GROUP)

        # Log per-group aggregation
        group_counts = driver_groups.loc[bool_mask].value_counts()
        for grp, cnt in group_counts.items():
            logger.info("  Driver group %-22s : %d banks", grp, cnt)

        feature_counts = top_feature.value_counts()
        for feat, cnt in feature_counts.items():
            logger.info("  Driver feature %-30s : %d banks", feat, cnt)

        return drivers, driver_groups

    # ------------------------------------------------------------------
    #  4. OBS Risk Contribution
    # ------------------------------------------------------------------

    def compute_obs_risk_contribution(
        self,
        df_original: pd.DataFrame,
        anomaly_mask: pd.Series,
    ) -> Tuple[pd.Series, pd.Series]:
        """Assess OBS risk for each anomaly via ``obs_risk_indicator``.

        Returns:
            Tuple of (obs_risk_flag, obs_risk_zscore) Series.
        """
        flag = pd.Series("N/A", index=df_original.index)
        zscore = pd.Series(0.0, index=df_original.index)
        bool_mask = (anomaly_mask == -1)

        if OBS_RISK_COL not in df_original.columns:
            logger.warning("Column '%s' not found – skipping OBS analysis.", OBS_RISK_COL)
            return flag, zscore

        if not bool_mask.any():
            return flag, zscore

        logger.info("Computing OBS risk contribution for %d anomalies …", int(bool_mask.sum()))

        obs_col = df_original[OBS_RISK_COL]
        obs_median = float(obs_col.median())
        obs_std = float(obs_col.std())
        obs_p75 = float(obs_col.quantile(0.75))

        if obs_std == 0:
            flag.loc[bool_mask] = "Normal OBS"
            return flag, zscore

        zscore.loc[bool_mask] = (obs_col.loc[bool_mask] - obs_median) / obs_std
        flag.loc[bool_mask & (obs_col > obs_p75)] = "High OBS Risk"
        flag.loc[bool_mask & ~(obs_col > obs_p75)] = "Normal OBS"

        n_high = int((flag == "High OBS Risk").sum())
        n_normal = int((flag == "Normal OBS").sum())
        logger.info(
            "  OBS: %d High, %d Normal (P75=%.4f, median=%.4f)",
            n_high, n_normal, obs_p75, obs_median,
        )
        return flag, zscore

    # ------------------------------------------------------------------
    #  5. Full analysis pipeline
    # ------------------------------------------------------------------

    def run_full_analysis(
        self,
        df_processed: pd.DataFrame,
        df_original: pd.DataFrame,
    ) -> pd.DataFrame:
        """Execute the complete pipeline end-to-end.

        Returns:
            Copy of *df_original* extended with:
            ``anomaly_score``, ``is_anomaly``, ``cluster_label``,
            ``cluster_dna``, ``anomaly_driver``, ``anomaly_driver_group``,
            ``obs_risk_flag``, ``obs_risk_zscore``.
        """
        logger.info("=" * 60)
        logger.info("Starting full BankGuard AI analysis pipeline …")
        logger.info("  ALL_ML_FEATURES : %d dimensions", len(MULTIVARIATE_FEATURES))
        logger.info("=" * 60)

        # Step 1: Anomaly detection
        df_anomaly = self.detect_anomalies(df_processed)

        # Step 2: Clustering + DNA
        df_clustered = self.cluster_banks(df_anomaly, df_original)

        # Step 3: Driver analysis (feature + group)
        anomaly_driver, anomaly_driver_group = self.get_anomaly_drivers(
            df_original, df_clustered["is_anomaly"]
        )

        # Step 4: OBS deep explanation
        obs_flag, obs_zscore = self.compute_obs_risk_contribution(
            df_original, df_clustered["is_anomaly"]
        )

        # Step 5: Integrate
        df_result = df_original.copy()
        df_result["anomaly_score"] = df_clustered["anomaly_score"].values
        df_result["is_anomaly"] = df_clustered["is_anomaly"].values
        df_result["cluster_label"] = df_clustered["cluster_label"].values
        df_result["cluster_dna"] = df_clustered["cluster_dna"].values
        df_result["anomaly_driver"] = anomaly_driver.values
        df_result["anomaly_driver_group"] = anomaly_driver_group.values
        df_result["obs_risk_flag"] = obs_flag.values
        df_result["obs_risk_zscore"] = obs_zscore.values

        n_anom = int((df_result["is_anomaly"] == -1).sum())
        logger.info("-" * 60)
        logger.info("Pipeline complete.  Output: %d × %d", *df_result.shape)
        logger.info("  Anomalies: %d", n_anom)
        for lbl in RISK_LABELS:
            cnt = int((df_result["cluster_label"] == lbl).sum())
            dna = self.cluster_dna.get(lbl, "N/A")
            logger.info("  %-14s : %d banks  [%s]", lbl, cnt, dna)
        logger.info("  High OBS Risk: %d", int((df_result["obs_risk_flag"] == "High OBS Risk").sum()))
        logger.info("=" * 60)

        return df_result


# ═══════════════════════════════════════════════════════════════════════════
#  Standalone sanity-check
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from utils.data_processor import process_data

    df_processed, df_original, scalers = process_data()
    print(f"\nScalers: {list(scalers.keys())}")

    detector = BankAnomalyDetector()
    df_result = detector.run_full_analysis(df_processed, df_original)

    print("\n--- Anomaly summary ---")
    anomalies = df_result[df_result["is_anomaly"] == -1]
    if not anomalies.empty:
        cols = [
            "bank_id", "period", "npl_ratio", "capital_adequacy_ratio",
            "anomaly_score", "cluster_label", "anomaly_driver",
            "anomaly_driver_group", "obs_risk_flag", "obs_risk_zscore",
        ]
        print(anomalies[cols].to_string(index=False))
    else:
        print("No anomalies detected.")

    print("\n--- Cluster DNA ---")
    for lbl, dna in detector.cluster_dna.items():
        print(f"  {lbl:14s} : {dna}")

    print("\n--- Cluster distribution ---")
    print(df_result["cluster_label"].value_counts().to_string())
