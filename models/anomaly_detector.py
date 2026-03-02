"""
anomaly_detector.py - Multi-Algorithm Consensus Scoring Engine for BankGuard AI.
=================================================================================

Architecture (Pillar-by-Pillar Consensus + Hybrid Risk Fusion)
--------------------------------------------------------------

1. **Three Unsupervised Models** - initialised from ``config.ML_MODELS_CONFIG``:
   Isolation Forest, Local Outlier Factor (LOF), One-Class SVM.

2. **Pillar-by-Pillar Execution** - each of the 7 numerical risk pillars
   (Credit, Liquidity, Concentration, Capital, Earnings, OBS, Funding)
   is scored independently by all 3 models on its own scaled feature
   subspace.

3. **Consensus Mechanism** (per pillar, per observation):
   - 3/3 models flag as anomaly  ->  Pillar Consensus Score = 100 (High Risk)
   - 2/3 models flag              ->  Score = 66 (Warning)
   - 1/3 models flag              ->  Score = 33 (Monitor)
   - 0/3 models flag              ->  Score = 0  (Normal)

4. **Aggregate**:
   - ``Overall_ML_Risk_Score``  = mean of 7 pillar consensus scores.
   - ``Final_Hybrid_Risk_Status`` = fusion of ML score + Expert Rule
     ``rule_risk_score`` (from data_processor).

5. **Backward Compatibility** - preserves ``is_anomaly``, ``anomaly_score``,
   ``cluster_label``, ``cluster_dna``, ``anomaly_driver``,
   ``anomaly_driver_group``, ``obs_risk_flag``, ``obs_risk_zscore``
   columns consumed by ``app.py``.

Author : BankGuard AI Team - Lead ML Engineer
Created: 2026-03-02
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# -- Ensure project root is importable ------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import (
    ALL_ML_FEATURES,
    CONCENTRATION_RISK_FEATURES,
    EXPOSURE_LIQUIDITY_FEATURES,
    FINANCIAL_HEALTH_FEATURES,
    ML_MODELS_CONFIG,
    MODEL_PARAMS,
    RISK_PILLARS,
    SECTOR_LOANS_COLUMNS,
)

# --------------------------------------------------------------------------
#  Logger setup
# --------------------------------------------------------------------------

logger = logging.getLogger("BankGuard.AnomalyDetector")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter(
            "[%(name)s] %(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# --------------------------------------------------------------------------
#  Constants derived from config.py
# --------------------------------------------------------------------------

MULTIVARIATE_FEATURES: List[str] = ALL_ML_FEATURES
DRIVER_GROUPS: Dict[str, List[str]] = dict(RISK_PILLARS)

_FEATURE_TO_GROUP: Dict[str, str] = {
    feat: group
    for group, feats in DRIVER_GROUPS.items()
    for feat in feats
}

SECTOR_LABELS: Dict[str, str] = {
    "sector_loans_energy": "Energy",
    "sector_loans_real_estate": "Real Estate",
    "sector_loans_construction": "Construction",
    "sector_loans_services": "Services",
    "sector_loans_agriculture": "Agriculture",
}

OBS_RISK_COL: str = "obs_risk_indicator"
RISK_LABELS: List[str] = ["Low Risk", "Medium Risk", "High Risk"]

# Consensus score mapping: n_flags -> score
_CONSENSUS_MAP: Dict[int, int] = {0: 0, 1: 33, 2: 66, 3: 100}

# Pillar short keys for column naming (e.g. "credit_risk_consensus")
PILLAR_KEYS: Dict[str, str] = {
    "Credit Risk":           "credit_risk",
    "Liquidity Risk":        "liquidity_risk",
    "Concentration Risk":    "concentration_risk",
    "Capital Adequacy":      "capital_adequacy",
    "Earnings & Efficiency": "earnings_efficiency",
    "Off-Balance Sheet":     "obs_exposure",
    "Funding Stability":     "funding_stability",
}

# Hybrid risk thresholds
_HYBRID_CRITICAL = 60  # Overall_ML_Risk_Score >= 60 OR rule_risk_score >= 3
_HYBRID_WARNING  = 30  # Overall_ML_Risk_Score >= 30 OR rule_risk_score >= 1


# ==========================================================================
#  Helper: instantiate an sklearn model from ML_MODELS_CONFIG
# ==========================================================================

def _build_model(name: str) -> Any:
    """Create a fresh sklearn model instance from ``ML_MODELS_CONFIG``."""
    cfg = ML_MODELS_CONFIG[name]
    params = dict(cfg["params"])

    if name == "IsolationForest":
        return IsolationForest(**params)
    elif name == "LocalOutlierFactor":
        return LocalOutlierFactor(**params)
    elif name == "OneClassSVM":
        return OneClassSVM(**params)
    else:
        raise ValueError(f"Unknown model: {name}")


# ==========================================================================
#  Main class
# ==========================================================================

class BankAnomalyDetector:
    """Multi-Algorithm Consensus Scoring Engine.

    For each of the 7 risk pillars the engine runs 3 unsupervised models
    (Isolation Forest, LOF, One-Class SVM) on the pillar's feature
    subspace.  A consensus vote produces a per-pillar score (0/33/66/100).
    An ``Overall_ML_Risk_Score`` is then fused with the Expert Rule
    ``rule_risk_score`` to produce a ``Final_Hybrid_Risk_Status``.

    Backward-compatible with the previous single-Isolation-Forest API
    (``is_anomaly``, ``anomaly_score``, ``cluster_label``, etc.).
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

        # Fitted model storage: {pillar_key: {model_name: fitted_model}}
        self.pillar_models: Dict[str, Dict[str, Any]] = {}
        self.kmeans: Optional[KMeans] = None
        self.cluster_dna: Dict[str, str] = {}

        logger.info(
            "BankAnomalyDetector initialised  "
            "(contamination=%.2f, n_clusters=%d, seed=%d, "
            "pillars=%d, models_per_pillar=%d)",
            contamination, n_clusters, random_state,
            len(RISK_PILLARS), len(ML_MODELS_CONFIG),
        )

    # ------------------------------------------------------------------
    #  1. Pillar-by-Pillar Consensus Scoring
    # ------------------------------------------------------------------

    def _run_pillar_consensus(
        self,
        df_scaled: pd.DataFrame,
    ) -> pd.DataFrame:
        """Run 3 models on each pillar's feature subspace.

        Adds columns per pillar:
          - ``<key>_IF``, ``<key>_LOF``, ``<key>_SVM`` : -1/1 predictions
          - ``<key>_n_flags``     : count of anomaly votes (0-3)
          - ``<key>_consensus``   : consensus score (0/33/66/100)

        Returns the DataFrame (mutated copy).
        """
        df = df_scaled.copy()
        n = len(df)

        for pillar_name, features in RISK_PILLARS.items():
            key = PILLAR_KEYS[pillar_name]
            X = df[features].values

            logger.info(
                "  Pillar [%s] - %d features x %d samples",
                pillar_name, len(features), n,
            )

            predictions: Dict[str, np.ndarray] = {}
            self.pillar_models[key] = {}

            for model_name in ML_MODELS_CONFIG:
                model = _build_model(model_name)
                short = {"IsolationForest": "IF",
                         "LocalOutlierFactor": "LOF",
                         "OneClassSVM": "SVM"}[model_name]
                col = f"{key}_{short}"

                try:
                    if model_name == "LocalOutlierFactor":
                        # LOF with novelty=True: fit then predict
                        model.fit(X)
                        preds = model.predict(X)
                    else:
                        model.fit(X)
                        preds = model.predict(X)

                    predictions[short] = preds
                    df[col] = preds
                    self.pillar_models[key][model_name] = model

                    n_anom = int((preds == -1).sum())
                    logger.info(
                        "    %-20s -> %d anomalies (%.1f%%)",
                        model_name, n_anom, 100.0 * n_anom / n,
                    )
                except Exception as e:
                    logger.warning(
                        "    %-20s FAILED on [%s]: %s",
                        model_name, pillar_name, e,
                    )
                    df[col] = 1  # default to normal
                    predictions[short] = np.ones(n, dtype=int)

            # Consensus: count how many of the 3 models flagged -1
            flag_cols = [f"{key}_{s}" for s in ["IF", "LOF", "SVM"]]
            flag_matrix = df[flag_cols].values  # shape (n, 3), values in {-1, 1}
            n_flags = (flag_matrix == -1).sum(axis=1)
            df[f"{key}_n_flags"] = n_flags
            df[f"{key}_consensus"] = pd.Series(n_flags).map(_CONSENSUS_MAP).values

            # Log consensus distribution
            for score_val in [100, 66, 33, 0]:
                cnt = int((df[f"{key}_consensus"] == score_val).sum())
                if cnt > 0:
                    logger.info(
                        "    Consensus %3d : %d banks", score_val, cnt,
                    )

        return df

    # ------------------------------------------------------------------
    #  2. Aggregate: Overall ML Risk Score
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_overall_ml_score(df: pd.DataFrame) -> pd.DataFrame:
        """``Overall_ML_Risk_Score`` = mean of 7 pillar consensus scores."""
        consensus_cols = [f"{PILLAR_KEYS[p]}_consensus" for p in RISK_PILLARS]
        df["Overall_ML_Risk_Score"] = df[consensus_cols].mean(axis=1).round(2)

        logger.info(
            "Overall ML Risk Score: mean=%.2f, min=%.2f, max=%.2f",
            df["Overall_ML_Risk_Score"].mean(),
            df["Overall_ML_Risk_Score"].min(),
            df["Overall_ML_Risk_Score"].max(),
        )
        return df

    # ------------------------------------------------------------------
    #  3. Hybrid Fusion: ML Score + Expert Rules -> Final Status
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_hybrid_status(df: pd.DataFrame) -> pd.DataFrame:
        """Combine ``Overall_ML_Risk_Score`` with ``rule_risk_score``.

        Status mapping:
          - Critical : ML >= 60  OR  rule_risk_score >= 3
          - Warning  : ML >= 30  OR  rule_risk_score >= 1
          - Normal   : otherwise
        """
        rule_col = "rule_risk_score"
        if rule_col not in df.columns:
            logger.warning(
                "'%s' not found - hybrid status based on ML only.", rule_col,
            )
            df[rule_col] = 0

        ml = df["Overall_ML_Risk_Score"]
        rules = df[rule_col]

        conditions = [
            (ml >= _HYBRID_CRITICAL) | (rules >= 3),
            (ml >= _HYBRID_WARNING) | (rules >= 1),
        ]
        choices = ["Critical", "Warning"]
        df["Final_Hybrid_Risk_Status"] = np.select(conditions, choices, default="Normal")

        # Backward-compatible is_anomaly: Critical -> -1, else 1
        df["is_anomaly"] = np.where(
            df["Final_Hybrid_Risk_Status"] == "Critical", -1, 1
        )
        # Backward-compatible anomaly_score (higher = more anomalous)
        df["anomaly_score"] = -(df["Overall_ML_Risk_Score"] / 100.0)

        for status in ["Critical", "Warning", "Normal"]:
            cnt = int((df["Final_Hybrid_Risk_Status"] == status).sum())
            logger.info("  Hybrid status %-10s : %d banks", status, cnt)

        return df

    # ------------------------------------------------------------------
    #  4. Risk Clustering - K-Means + Sector-Loan DNA
    # ------------------------------------------------------------------

    def cluster_banks(
        self,
        df_scaled: pd.DataFrame,
        df_original: pd.DataFrame,
    ) -> pd.DataFrame:
        """K-Means clustering with data-driven risk labels and sector DNA."""
        df = df_scaled.copy()
        features = df[MULTIVARIATE_FEATURES]

        if len(df) <= 1:
            df["cluster_label"] = "Medium Risk"
            df["cluster_dna"] = "Insufficient data"
            return df

        effective_k = min(self.n_clusters, len(df))
        logger.info(
            "Training K-Means (k=%d) on %d samples x %d features",
            effective_k, len(features), features.shape[1],
        )

        self.kmeans = KMeans(
            n_clusters=effective_k,
            random_state=self.random_state,
            n_init=10, max_iter=300,
        )
        raw_labels = self.kmeans.fit_predict(features)
        df["_raw_cluster"] = raw_labels

        # -- Risk-label mapping ----------------------------------------
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

        # -- Sector-loan DNA -------------------------------------------
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
            logger.warning("No sector_loans_* columns - skipping DNA.")
            df["cluster_dna"] = "N/A"
            self.cluster_dna = {lbl: "N/A" for lbl in RISK_LABELS}

        df.drop(columns=["_raw_cluster"], inplace=True)
        logger.info("K-Means + DNA profiling complete.")
        return df

    # ------------------------------------------------------------------
    #  5. Explainability - Anomaly Drivers (feature + group level)
    # ------------------------------------------------------------------

    def get_anomaly_drivers(
        self,
        df_original: pd.DataFrame,
        anomaly_mask: pd.Series,
    ) -> Tuple[pd.Series, pd.Series]:
        """Identify per-anomaly primary driver feature AND the risk group.

        Uses z-score deviation across all 26 ML features to find the
        single most deviating feature per flagged observation.
        """
        drivers = pd.Series("N/A", index=df_original.index)
        driver_groups = pd.Series("N/A", index=df_original.index)
        bool_mask = (anomaly_mask == -1)

        if not bool_mask.any():
            logger.info("No anomalies - skipping driver analysis.")
            return drivers, driver_groups

        logger.info("Computing anomaly drivers for %d flagged banks", int(bool_mask.sum()))

        medians = df_original[ALL_ML_FEATURES].median()
        stds = df_original[ALL_ML_FEATURES].std().replace(0, np.nan)

        anomaly_rows = df_original.loc[bool_mask, ALL_ML_FEATURES]
        z_devs = (anomaly_rows - medians).abs() / stds
        top_feature = z_devs.idxmax(axis=1)
        drivers.loc[bool_mask] = top_feature
        driver_groups.loc[bool_mask] = top_feature.map(_FEATURE_TO_GROUP)

        group_counts = driver_groups.loc[bool_mask].value_counts()
        for grp, cnt in group_counts.items():
            logger.info("  Driver group %-22s : %d banks", grp, cnt)

        return drivers, driver_groups

    # ------------------------------------------------------------------
    #  6. OBS Risk Contribution
    # ------------------------------------------------------------------

    def compute_obs_risk_contribution(
        self,
        df_original: pd.DataFrame,
        anomaly_mask: pd.Series,
    ) -> Tuple[pd.Series, pd.Series]:
        """Assess OBS risk contribution via ``obs_risk_indicator``."""
        flag = pd.Series("N/A", index=df_original.index)
        zscore = pd.Series(0.0, index=df_original.index)
        bool_mask = (anomaly_mask == -1)

        if OBS_RISK_COL not in df_original.columns:
            logger.warning("Column '%s' not found - skipping OBS analysis.", OBS_RISK_COL)
            return flag, zscore

        if not bool_mask.any():
            return flag, zscore

        logger.info("Computing OBS risk contribution for %d anomalies", int(bool_mask.sum()))

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
    #  7. Full Analysis Pipeline
    # ------------------------------------------------------------------

    def run_full_analysis(
        self,
        df_processed: pd.DataFrame,
        df_original: pd.DataFrame,
    ) -> pd.DataFrame:
        """Execute the complete Multi-Algorithm Consensus pipeline.

        Returns
        -------
        pd.DataFrame
            Copy of *df_original* enriched with:

            Per-pillar (7x):
              ``<key>_IF``, ``<key>_LOF``, ``<key>_SVM``,
              ``<key>_n_flags``, ``<key>_consensus``

            Aggregate:
              ``Overall_ML_Risk_Score``, ``Final_Hybrid_Risk_Status``

            Backward-compatible:
              ``is_anomaly``, ``anomaly_score``,
              ``cluster_label``, ``cluster_dna``,
              ``anomaly_driver``, ``anomaly_driver_group``,
              ``obs_risk_flag``, ``obs_risk_zscore``
        """
        logger.info("=" * 65)
        logger.info("BankGuard AI - Multi-Algorithm Consensus Pipeline")
        logger.info("  Pillars : %d  |  Models/pillar : %d  |  Features : %d",
                     len(RISK_PILLARS), len(ML_MODELS_CONFIG), len(ALL_ML_FEATURES))
        logger.info("=" * 65)

        # Step 1: Pillar-by-Pillar consensus scoring (3 models x 7 pillars)
        logger.info("[Step 1/6] Pillar-by-Pillar Consensus Scoring")
        df_consensus = self._run_pillar_consensus(df_processed)

        # Step 2: Overall ML Risk Score (mean of 7 consensus scores)
        logger.info("[Step 2/6] Aggregate Overall ML Risk Score")
        df_consensus = self._compute_overall_ml_score(df_consensus)

        # Step 3: Hybrid fusion with expert rules
        logger.info("[Step 3/6] Hybrid Fusion (ML + Expert Rules)")
        df_consensus = self._compute_hybrid_status(df_consensus)

        # Step 4: K-Means clustering + sector DNA
        logger.info("[Step 4/6] Risk Clustering + Sector DNA")
        df_clustered = self.cluster_banks(df_consensus, df_original)

        # Step 5: Driver analysis (feature + group level)
        logger.info("[Step 5/6] Explainability - Anomaly Drivers")
        anomaly_driver, anomaly_driver_group = self.get_anomaly_drivers(
            df_original, df_clustered["is_anomaly"]
        )

        # Step 6: OBS risk contribution
        logger.info("[Step 6/6] OBS Risk Contribution")
        obs_flag, obs_zscore = self.compute_obs_risk_contribution(
            df_original, df_clustered["is_anomaly"]
        )

        # ---- Assemble final result on df_original --------------------
        df_result = df_original.copy()

        # Per-pillar columns
        for pillar_name in RISK_PILLARS:
            key = PILLAR_KEYS[pillar_name]
            for suffix in ["IF", "LOF", "SVM", "n_flags", "consensus"]:
                col = f"{key}_{suffix}"
                if col in df_clustered.columns:
                    df_result[col] = df_clustered[col].values

        # Aggregate columns
        df_result["Overall_ML_Risk_Score"] = df_clustered["Overall_ML_Risk_Score"].values
        df_result["Final_Hybrid_Risk_Status"] = df_clustered["Final_Hybrid_Risk_Status"].values

        # Backward-compatible columns
        df_result["anomaly_score"] = df_clustered["anomaly_score"].values
        df_result["is_anomaly"] = df_clustered["is_anomaly"].values
        df_result["cluster_label"] = df_clustered["cluster_label"].values
        df_result["cluster_dna"] = df_clustered["cluster_dna"].values
        df_result["anomaly_driver"] = anomaly_driver.values
        df_result["anomaly_driver_group"] = anomaly_driver_group.values
        df_result["obs_risk_flag"] = obs_flag.values
        df_result["obs_risk_zscore"] = obs_zscore.values

        # ---- Summary log ---------------------------------------------
        n_critical = int((df_result["Final_Hybrid_Risk_Status"] == "Critical").sum())
        n_warning = int((df_result["Final_Hybrid_Risk_Status"] == "Warning").sum())
        n_normal = int((df_result["Final_Hybrid_Risk_Status"] == "Normal").sum())

        logger.info("-" * 65)
        logger.info("Pipeline complete.  Output: %d x %d", *df_result.shape)
        logger.info("  Final Hybrid Status:")
        logger.info("    Critical : %d banks", n_critical)
        logger.info("    Warning  : %d banks", n_warning)
        logger.info("    Normal   : %d banks", n_normal)
        for lbl in RISK_LABELS:
            cnt = int((df_result["cluster_label"] == lbl).sum())
            dna = self.cluster_dna.get(lbl, "N/A")
            logger.info("  %-14s : %d banks  [%s]", lbl, cnt, dna)
        logger.info(
            "  High OBS Risk: %d",
            int((df_result["obs_risk_flag"] == "High OBS Risk").sum()),
        )
        logger.info("=" * 65)

        return df_result


# ==========================================================================
#  Standalone sanity-check
# ==========================================================================

if __name__ == "__main__":
    from utils.data_processor import process_data

    df_processed, df_original, scalers = process_data()
    print(f"\nScalers: {list(scalers.keys())}")

    detector = BankAnomalyDetector()
    df_result = detector.run_full_analysis(df_processed, df_original)

    # ---- Pillar consensus summary ----
    print("\n--- Per-Pillar Consensus Scores (sample) ---")
    consensus_cols = [f"{PILLAR_KEYS[p]}_consensus" for p in RISK_PILLARS]
    id_cols = ["bank_id", "period"] if "bank_id" in df_result.columns else []
    print(df_result[id_cols + consensus_cols].head(10).to_string(index=False))

    # ---- Overall & Hybrid ----
    print("\n--- Overall ML Risk Score ---")
    print(df_result["Overall_ML_Risk_Score"].describe().to_string())

    print("\n--- Final Hybrid Risk Status ---")
    print(df_result["Final_Hybrid_Risk_Status"].value_counts().to_string())

    # ---- Anomalies (Critical) ----
    print("\n--- Critical banks ---")
    critical = df_result[df_result["Final_Hybrid_Risk_Status"] == "Critical"]
    if not critical.empty:
        detail_cols = id_cols + [
            "Overall_ML_Risk_Score", "rule_risk_score",
            "Final_Hybrid_Risk_Status", "cluster_label",
            "anomaly_driver", "anomaly_driver_group",
        ]
        detail_cols = [c for c in detail_cols if c in critical.columns]
        print(critical[detail_cols].head(20).to_string(index=False))
    else:
        print("No Critical banks detected.")

    # ---- Cluster DNA ----
    print("\n--- Cluster DNA ---")
    for lbl, dna in detector.cluster_dna.items():
        print(f"  {lbl:14s} : {dna}")

    print("\n--- Cluster distribution ---")
    print(df_result["cluster_label"].value_counts().to_string())
