# BankGuard AI — Comprehensive 360° Banking Monitoring System

> **Version**: 2.0 · **Architecture**: Hybrid Rule-Based + Multi-Algorithm ML Ensemble  
> **Author**: BankGuard AI Team — KTNN · **Date**: 2026-03-02

---

## 1. Project Overview

**BankGuard AI 360°** is a production-grade monitoring platform designed for supervisory authorities and internal risk teams to continuously assess the financial health of banking institutions. The system ingests periodic regulatory returns (CSV), evaluates each bank-period observation against **two complementary analytical layers** — a deterministic Expert Rule Engine and a stochastic Multi-Algorithm Unsupervised ML Ensemble — and surfaces actionable insights through an interactive Streamlit dashboard.

### Core Objectives

- **Anomaly Detection** — Identify banks exhibiting abnormal risk profiles across 7 distinct risk dimensions using a consensus of three unsupervised ML algorithms (Isolation Forest, LOF, One-Class SVM).
- **Regulatory Compliance Screening** — Instantly flag observations that breach Basel III hard thresholds (e.g., CAR < 8%, LCR < 100%) via a codified Expert Rule Engine.
- **Hybrid Risk Fusion** — Aggregate ML-derived consensus scores with rule-engine violation counts into a unified `Final_Hybrid_Risk_Status` (Critical / Warning / Normal) to minimise both false positives and false negatives.
- **Explainability** — Attribute each critical anomaly to its primary driver feature and risk pillar, enabling targeted supervisory follow-up.
- **Peer Benchmarking** — Provide per-bank 360° profiling with radar charts, sector-loan DNA, and historical trend analysis.

---

## 2. Directory Structure

```
comprehensive banking monitoring system/
│
├── config.py                        # Central knowledge base
├── app.py                           # Streamlit interactive dashboard
├── requirements.txt                 # Python dependencies
│
├── utils/
│   └── data_processor.py            # Data ingestion & transformation pipeline
│
├── models/
│   └── anomaly_detector.py          # Multi-Algorithm Consensus Scoring Engine
│
└── data/
    └── time_series_dataset_enriched_v2.csv   # Input dataset
```

### File-by-File Breakdown

| File | Role |
|---|---|
| **`config.py`** | **The Knowledge Base.** Defines all 8 Risk Pillars and their mapped feature columns (26 ML features across 7 analytical pillars + 1 visualisation-only pillar). Houses the `EXPERT_RULES` dictionary (Basel-aligned thresholds with operator, severity, and pillar mapping) and `ML_MODELS_CONFIG` (sklearn class paths, hyperparameters, and descriptions for the 3 ensemble models). Also contains general pipeline parameters (`contamination`, `n_clusters`, `scaling_strategy`), human-readable `FEATURE_LABELS`, and `SECTOR_LABELS`. |
| **`utils/data_processor.py`** | **Data Ingestion, Rule Evaluation & Feature Scaling.** Implements a 6-stage pipeline: (1) CSV load with validation, (2) median imputation, (3) schema validation against `ALL_ML_FEATURES`, (4) derived feature engineering (`risk_to_profit_ratio`, `efficiency_ratio`), (5) vectorised Expert Rule Engine evaluation producing `rule_violations` and `rule_risk_score` per row, and (6) per-pillar `StandardScaler` fitting (7 independent scalers, one per risk pillar). Returns `(df_processed, df_original, scalers)`. |
| **`models/anomaly_detector.py`** | **The ML Engine.** Contains the `BankAnomalyDetector` class which orchestrates the full consensus pipeline: pillar-by-pillar execution of 3 unsupervised models (IF, LOF, SVM), consensus vote aggregation per pillar, `Overall_ML_Risk_Score` computation, hybrid fusion with expert rule scores, K-Means risk clustering with sector-loan DNA profiling, per-anomaly driver attribution (z-score based), and OBS risk contribution analysis. Exposes `run_full_analysis()` as the single entry point. |
| **`app.py`** | **The Streamlit Interactive Dashboard.** A 4-tab command centre: Executive Summary (KPIs, risk trajectory), Multi-Algorithm Comparison (IF/LOF/SVM agreement heatmaps), Risk Pillar Deep Dive (per-pillar scatter plots with consensus colouring), and 360° Bank Profiler (radar charts, rule violation alerts, funding structure, historical trends). Includes sidebar filters for period, region, bank type, and credit rating. |
| **`requirements.txt`** | **Dependencies.** Specifies pinned versions: `streamlit==1.54.0`, `pandas==2.3.3`, `numpy==2.4.2`, `plotly==6.5.2`, `scikit-learn==1.8.0`. |
| **`data/time_series_dataset_enriched_v2.csv`** | **Input Dataset.** Multi-period banking data with identifier columns (`bank_id`, `period`, `bank_type`, `region`, `external_credit_rating`) and all 26+ numeric features consumed by the ML and rule engines. |

---

## 3. The Hybrid Architecture

BankGuard AI employs a **dual-layered detection architecture** that combines deterministic rules with probabilistic machine learning. This hybrid approach addresses the fundamental limitations of either method in isolation: rules alone cannot capture non-linear, multi-dimensional risk patterns, while ML alone cannot enforce regulatory bright-line thresholds.

```
┌──────────────────────────────────────────────────────────────────┐
│                      RAW BANKING DATA (CSV)                      │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │   Data Processing Pipeline  │
         │  (Impute → Validate → Scale)│
         └──────┬──────────────┬───────┘
                │              │
       ┌────────▼────────┐  ┌─▼──────────────────────────────────┐
       │  LAYER 1        │  │  LAYER 2                            │
       │  Expert Rules   │  │  Unsupervised ML Ensemble           │
       │  (Basel III)    │  │  (IF + LOF + SVM) × 7 Pillars      │
       │                 │  │                                     │
       │  rule_violations│  │  Per-pillar consensus (0/33/66/100) │
       │  rule_risk_score│  │  Overall_ML_Risk_Score              │
       └────────┬────────┘  └─┬───────────────────────────────────┘
                │              │
                └──────┬───────┘
                       ▼
            ┌─────────────────────┐
            │   HYBRID FUSION     │
            │                     │
            │  Final_Hybrid_Risk  │
            │  _Status            │
            │  (Critical/Warning/ │
            │   Normal)           │
            └─────────────────────┘
```

### 3.1 Layer 1 — Expert Rule-Based Engine

**Purpose**: Immediate, deterministic flagging of observations that breach **absolute regulatory thresholds**. No model training is required — this layer enforces compliance constraints directly.

**Implementation** (`data_processor.py → evaluate_expert_rules()`):
- Each rule in `config.EXPERT_RULES` specifies: a target `column`, a comparison `operator` (`lt`, `gt`, etc.), a numeric `threshold`, a `severity` level, and a human-readable `message`.
- Rules are evaluated vectorially (boolean mask per rule across all rows).
- Per-row outputs: `rule_violations` (list of triggered rule IDs or `["Compliant"]`) and `rule_risk_score` (integer count of violations).

**Current Rule Set**:

| Rule ID | Column | Condition | Severity | Rationale |
|---|---|---|---|---|
| `CAR_CRITICAL` | `capital_adequacy_ratio` | < 8% | **Critical** | Basel III Pillar 1 minimum CAR breach — bank is under-capitalised. |
| `NPL_WARNING` | `npl_ratio` | > 3% | Warning | Elevated non-performing loan ratio signals credit-quality deterioration. |
| `LCR_CRITICAL` | `liquidity_coverage_ratio` | < 100% | **Critical** | Basel III LCR breach — insufficient HQLA for 30-day stress. |
| `LDR_WARNING` | `loan_to_deposit_ratio` | > 100% | Warning | Loans exceed the deposit base, indicating structural funding risk. |
| `HHI_HIGH_CONCENTRATION` | `sector_concentration_hhi` | > 0.25 | Warning | High lending concentration in few sectors (HHI > 0.25). |

### 3.2 Layer 2 — Unsupervised Machine Learning Ensemble

**Purpose**: Detect **hidden, non-linear, and multi-dimensional** risk patterns that cannot be captured by static thresholds — e.g., a bank whose individual metrics are borderline compliant but whose *combination* of risk factors is anomalous.

**Key Design Decisions**:

- **Unsupervised** — No labelled "anomaly" data is required. Models learn the structure of "normal" banking behaviour and flag deviations.
- **Per-Pillar Execution** — Each of the 7 risk pillars is scored independently on its own feature subspace, preserving domain interpretability.
- **Multi-Algorithm Consensus** — Three structurally different detectors (tree-based, density-based, boundary-based) vote per pillar, reducing false positives inherent to any single algorithm.
- **Per-Pillar Scaling** — A dedicated `StandardScaler` is fitted per pillar (`scaling_strategy: "per_pillar"`) to prevent feature-scale leakage across risk domains.

---

## 4. Machine Learning Models & Ensemble Logic

### 4.1 Model Specifications

The three unsupervised anomaly detectors are initialised from `config.ML_MODELS_CONFIG` and instantiated per pillar:

| Model | sklearn Class | Detection Paradigm | Key Hyperparameters | Strengths |
|---|---|---|---|---|
| **Isolation Forest** | `sklearn.ensemble.IsolationForest` | **Global, tree-based.** Isolates observations by randomly selecting a feature and a split value; anomalies require fewer random splits → shorter average path length. | `n_estimators=300`, `max_features=0.5`, `contamination=0.05` | Scales well to high dimensions; robust to irrelevant features; fast inference. |
| **Local Outlier Factor (LOF)** | `sklearn.neighbors.LocalOutlierFactor` | **Local, density-based.** Compares the local density of a point to its k-nearest neighbours; points in significantly sparser regions are flagged. | `n_neighbors=20`, `contamination=0.05`, `metric=euclidean`, `novelty=True` | Detects local anomalies invisible to global methods; captures cluster-boundary outliers. |
| **One-Class SVM** | `sklearn.svm.OneClassSVM` | **Boundary-based.** Learns a tight decision boundary around normal data in RBF kernel-transformed space; points outside the boundary are anomalies. | `kernel=rbf`, `gamma=scale`, `nu=0.05` | Effective in high-dimensional spaces; captures complex non-linear decision surfaces. |

### 4.2 Consensus Scoring Mechanism

The consensus mechanism is the core innovation that **reduces false positives** by requiring agreement among structurally diverse algorithms. It operates per risk pillar, per observation:

```
For each bank-period observation, for each of the 7 risk pillars:

  1. Run Isolation Forest   → prediction ∈ {-1 (anomaly), +1 (normal)}
  2. Run LOF                → prediction ∈ {-1, +1}
  3. Run One-Class SVM      → prediction ∈ {-1, +1}

  4. Count anomaly votes:  n_flags = Σ(prediction == -1)   ∈ {0, 1, 2, 3}

  5. Map to Consensus Score:
       n_flags = 3  →  Score = 100  (HIGH RISK — full agreement)
       n_flags = 2  →  Score =  66  (WARNING — majority agreement)
       n_flags = 1  →  Score =  33  (MONITOR — single model flag)
       n_flags = 0  →  Score =   0  (NORMAL — no flags)
```

**Aggregate Score**:

$$\text{Overall\_ML\_Risk\_Score} = \frac{1}{7} \sum_{p=1}^{7} \text{Consensus}_p$$

This yields a value in $[0, 100]$ representing the bank's system-wide ML-derived risk level.

### 4.3 Hybrid Fusion Logic

The final risk status fuses the ML score with the expert rule violation count:

| Condition | `Final_Hybrid_Risk_Status` |
|---|---|
| `Overall_ML_Risk_Score ≥ 60` **OR** `rule_risk_score ≥ 3` | **Critical** |
| `Overall_ML_Risk_Score ≥ 30` **OR** `rule_risk_score ≥ 1` | **Warning** |
| Otherwise | **Normal** |

The `OR` logic ensures that **either** a strong ML signal **or** a regulatory breach is sufficient to escalate status — a deliberate design choice for supervisory conservatism.

### 4.4 Supplementary Analytics

| Component | Description |
|---|---|
| **K-Means Risk Clustering** (`n_clusters=3`) | Groups banks into Low / Medium / High Risk clusters using all 26 ML features. Cluster labels are data-driven: ranked by a composite of `npl_ratio - capital_adequacy_ratio`. |
| **Sector-Loan DNA Profiling** | For each cluster, identifies the dominant sector loan allocation (e.g., "High Real Estate Exposure (35.2%)") from the 5 `sector_loans_*` columns. |
| **Anomaly Driver Attribution** | For each Critical bank, computes z-score deviation across all 26 features and identifies the single most deviating feature and its parent risk pillar. |
| **OBS Risk Contribution** | Evaluates `obs_risk_indicator` for critical banks; flags those above the 75th percentile as "High OBS Risk". |

---

## 5. Risk Pillars (Feature Mapping)

The system evaluates **8 Risk Pillars** spanning 26 ML features (Pillars 1–7) plus 5 sector-allocation features (Pillar 8, visualisation only).

| # | Risk Pillar | Features (Column Names) | Description |
|---|---|---|---|
| 1 | **Credit Risk** | `npl_ratio`, `loan_growth_rate`, `provision_coverage_ratio`, `non_performing_loans` | Asset-quality deterioration and provisioning adequacy. *Example*: `npl_ratio` measures non-performing loans as a share of total loans; `provision_coverage_ratio` captures loan-loss reserves relative to NPLs. |
| 2 | **Liquidity Risk** | `liquidity_coverage_ratio`, `nsfr`, `net_cash_outflows_30d` | Short-term and structural liquidity buffers. *Example*: `liquidity_coverage_ratio` (Basel III LCR) measures HQLA vs. 30-day stressed net cash outflows; `nsfr` captures the Net Stable Funding Ratio. |
| 3 | **Concentration Risk** | `sector_concentration_hhi`, `top20_borrower_concentration`, `geographic_concentration` | Name, sector, and geographic concentration. *Example*: `sector_concentration_hhi` is the Herfindahl-Hirschman Index on sector-level loan distribution; `top20_borrower_concentration` measures Top-20 borrower exposure as a share of total loans. |
| 4 | **Capital Adequacy** | `capital_adequacy_ratio`, `tier1_capital`, `risk_weighted_assets` | Solvency and loss-absorption capacity under Basel III. *Example*: `capital_adequacy_ratio` (CAR) = (Tier 1 + Tier 2 capital) / RWA; `tier1_capital` represents core equity capital. |
| 5 | **Earnings & Efficiency** | `return_on_assets`, `return_on_equity`, `net_interest_margin`, `operating_expenses`, `operating_income` | Profitability trends and cost discipline. *Example*: `return_on_assets` (ROA) measures net income relative to total assets; `net_interest_margin` captures the spread between interest income and expense. |
| 6 | **Off-Balance Sheet** | `obs_exposure_total`, `guarantees_issued`, `obs_to_assets_ratio`, `derivatives_to_assets_ratio` | Contingent liabilities and derivative leverage. *Example*: `obs_to_assets_ratio` quantifies total OBS exposure relative to total assets; `derivatives_to_assets_ratio` captures derivative notional exposure. |
| 7 | **Funding Stability** | `wholesale_dependency_ratio`, `loan_to_deposit_ratio`, `top20_depositors_ratio`, `deposit_growth_rate` | Structural funding profile and deposit concentration. *Example*: `wholesale_dependency_ratio` measures reliance on wholesale vs. deposit funding; `loan_to_deposit_ratio` indicates whether loans exceed the deposit base. |
| 8 | **Sector Loan Allocation** *(visualisation only)* | `sector_loans_energy`, `sector_loans_real_estate`, `sector_loans_construction`, `sector_loans_services`, `sector_loans_agriculture` | Sectoral distribution of the loan book. Used for DNA profiling and pie chart visualisation — **not included in the ML feature vector**. |

**Total ML Features**: 26 (Pillars 1–7) · **Total Visualisation Features**: 5 (Pillar 8)

---

## 6. Dashboard & UI

The Streamlit dashboard (`app.py`) provides a **4-tab command centre** with global sidebar filters (period, region, bank type, credit rating).

### Tab 1 — Executive Summary

- **KPI Cards**: Critical count, Warning count, total rule violations, high-confidence ML anomalies (score ≥ 60), system-wide NPL ratio, average CAR.
- **Hybrid Risk Status Distribution**: Donut chart showing Critical / Warning / Normal proportions.
- **Risk Trajectory**: Dual-axis time-series chart plotting average and max `Overall_ML_Risk_Score` alongside the count of Critical banks per period — functions as an **early-warning trend indicator** for system-wide risk build-up.
- **Risk Cluster Distribution**: Bar chart of K-Means cluster sizes with sector-loan DNA annotations.
- **Anomaly Driver Attribution**: Bar chart showing which risk pillars most frequently drive Critical classifications.

### Tab 2 — Multi-Algorithm Comparison

- **Per-Pillar Consensus Heatmap**: Banks × Pillars matrix colour-coded by consensus score (0 / 33 / 66 / 100). Enables instant identification of which banks are flagged on which dimensions.
- **Model Agreement Matrix**: Per-pillar overlap counts (IF Only, LOF Only, SVM Only, IF+LOF, IF+SVM, LOF+SVM, All 3, None) displayed as a colour-scaled heatmap. Quantifies inter-model agreement and highlights pillars where models disagree.
- **Overall ML Risk Score Distribution**: Histogram with vertical threshold lines at score = 30 (Warning) and 60 (Critical).

### Tab 3 — Risk Pillar Deep Dive

- **Pillar Selector**: Dropdown to select any of the 7 analytical risk pillars.
- **Pillar KPIs**: Average consensus score, count of High Risk (3/3), Warning (2/3), and Monitor (1/3) banks.
- **Feature Scatter Plots**: Pairwise scatter plots of pillar features, with points colour-coded by consensus status (red = 3/3, orange = 2/3, yellow = 1/3, green = Normal).
- **Feature Statistics Table**: Descriptive statistics (count, mean, std, min, 25%, 50%, 75%, max) for each feature in the selected pillar.
- **CSV Export**: Download pillar-specific data for offline analysis.

### Tab 4 — 360° Bank Profiler

- **Bank Selector**: Dropdown to select an individual bank.
- **Info Bar**: Bank ID, region, type, hybrid risk status, ML risk score.
- **Rule Violation Alert Box**: Styled alert listing triggered expert rules (or "Compliant" badge).
- **7-Pillar Consensus Radar Chart**: Normalised radar comparing the selected bank's per-pillar consensus scores against its **peer group** (same `bank_type`) average — enables instant identification of the bank's relative strengths and weaknesses.
- **Per-Pillar Consensus Bar Chart**: Horizontal bar breakdown of all 7 pillar scores for the selected bank.
- **Bank Details Panel**: Cluster label, sector-loan DNA, primary anomaly driver feature and group, OBS risk flag, external credit rating.
- **Funding Structure Pie Charts**: Wholesale vs. deposit funding split; sector loan allocation breakdown.
- **Historical ML Risk Trend**: Time-series chart of the bank's `Overall_ML_Risk_Score` and `rule_risk_score` across all available periods.
- **Full Report CSV Export**: Download the complete enriched dataset.

---

## 7. Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place the dataset
#    Ensure data/time_series_dataset_enriched_v2.csv is present

# 3. Launch the dashboard
streamlit run app.py
```

---

## 8. Configuration Reference

All tunable parameters reside in `config.py`:

| Parameter | Location | Default | Description |
|---|---|---|---|
| `contamination` | `MODEL_PARAMS` / `ML_MODELS_CONFIG` | `0.05` | Expected anomaly rate (5%). Controls IF/LOF threshold and SVM `nu`. |
| `n_clusters` | `MODEL_PARAMS` | `3` | Number of K-Means risk clusters (Low / Medium / High). |
| `random_state` | `MODEL_PARAMS` | `42` | Global seed for reproducibility. |
| `ensemble_method` | `MODEL_PARAMS` | `majority_vote` | Consensus fusion strategy. |
| `scaling_strategy` | `MODEL_PARAMS` | `per_pillar` | One `StandardScaler` per risk pillar (7 independent scalers). |
| `n_estimators` | `ML_MODELS_CONFIG.IsolationForest` | `300` | Number of trees in the Isolation Forest. |
| `max_features` | `ML_MODELS_CONFIG.IsolationForest` | `0.5` | Feature subsampling ratio per tree. |
| `n_neighbors` | `ML_MODELS_CONFIG.LocalOutlierFactor` | `20` | Number of neighbours for LOF density estimation. |
| `kernel` | `ML_MODELS_CONFIG.OneClassSVM` | `rbf` | Kernel function for SVM boundary learning. |

---

*© 2026 BankGuard AI Team — KTNN. All rights reserved.*
