"""
app.py - BankGuard AI 360 Dashboard.
=========================================

4-tab Streamlit command-center leveraging the Multi-Algorithm Consensus
Scoring Engine and Expert Rule Engine:

    Tab 1 - Executive Summary     (KPI cards, Risk Trajectory line chart)
    Tab 2 - Multi-Algo Comparison (IF/LOF/SVM agreement heatmap)
    Tab 3 - Risk Pillar Deep Dive (per-pillar scatter, outlier coloring)
    Tab 4 - 360 Bank Profiler     (radar, rule violations, funding pie)

Author : BankGuard AI Team - Senior Streamlit UI/UX Developer
Created: 2026-03-02
"""

from __future__ import annotations

import io
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    ALL_ML_FEATURES,
    FEATURE_LABELS,
    RISK_PILLARS,
    SECTOR_LABELS,
    SECTOR_LOANS_COLUMNS,
)
from models.anomaly_detector import BankAnomalyDetector, PILLAR_KEYS
from utils.data_processor import process_data

# =====================================================================
#  Global Config & Theme
# =====================================================================

st.set_page_config(
    page_title="BankGuard AI 360 - Banking Monitoring",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Colour palette
HYBRID_COLORS: Dict[str, str] = {
    "Critical": "#e74c3c",
    "Warning":  "#f39c12",
    "Normal":   "#2ecc71",
}
CLUSTER_COLORS: Dict[str, str] = {
    "Low Risk":    "#2ecc71",
    "Medium Risk": "#f39c12",
    "High Risk":   "#e74c3c",
}
PILLAR_COLORS: Dict[str, str] = {
    "Credit Risk":           "#e74c3c",
    "Liquidity Risk":        "#3498db",
    "Concentration Risk":    "#9b59b6",
    "Capital Adequacy":      "#1abc9c",
    "Earnings & Efficiency": "#f39c12",
    "Off-Balance Sheet":     "#e67e22",
    "Funding Stability":     "#2c3e50",
}
ANOMALY_COLOR = "#e74c3c"
NORMAL_COLOR  = "#2ecc71"
PLOTLY_TEMPLATE = "plotly_white"

# =====================================================================
#  Custom CSS
# =====================================================================

st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8f9fc 0%, #eef1f8 100%);
        border: 1px solid #dde2ef;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    div[data-testid="stMetric"] label {
        font-size: 0.82rem;
        color: #5a6a85;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .sidebar-title {
        font-size: 1.65rem;
        font-weight: 700;
        color: #1a2740;
        text-align: center;
        margin-bottom: 4px;
    }
    .sidebar-subtitle {
        font-size: 0.82rem;
        color: #6b7b95;
        text-align: center;
        margin-bottom: 18px;
    }
    .rule-alert-box {
        background: #fde8e8;
        border-left: 5px solid #e74c3c;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
        color: #5a1a1a;
        font-size: 0.92rem;
    }
    .rule-alert-box strong { color: #c0392b; }
    button[data-baseweb="tab"] { font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================================
#  Helpers
# =====================================================================


def _get_latest_snapshot(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the latest-period row per bank_id."""
    if "_period_dt" in frame.columns:
        return frame.sort_values("_period_dt").groupby("bank_id").last().reset_index()
    return frame.groupby("bank_id").last().reset_index()


def _make_radar(
    labels: List[str],
    bank_vals: List[float],
    peer_vals: List[float],
    bank_name: str,
    peer_name: str = "Peer Average",
    color: str = ANOMALY_COLOR,
    height: int = 420,
) -> go.Figure:
    """Build a [0,1]-normalised radar chart: bank vs peer group."""
    all_v = bank_vals + peer_vals
    lo, hi = (min(all_v), max(all_v)) if all_v else (0, 1)
    span = hi - lo if hi != lo else 1
    nb = [(v - lo) / span for v in bank_vals]
    ns = [(v - lo) / span for v in peer_vals]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=nb + [nb[0]], theta=labels + [labels[0]], fill="toself",
        name=bank_name, line=dict(color=color, width=2),
        fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.18)",
        customdata=bank_vals + [bank_vals[0]],
        hovertemplate="<b>%{theta}</b><br>Score: %{customdata:.1f}<extra></extra>",
    ))
    fig.add_trace(go.Scatterpolar(
        r=ns + [ns[0]], theta=labels + [labels[0]], fill="toself",
        name=peer_name, line=dict(color="#3498db", width=2, dash="dash"),
        fillcolor="rgba(52,152,219,0.10)",
        customdata=peer_vals + [peer_vals[0]],
        hovertemplate="<b>%{theta}</b><br>Score: %{customdata:.1f}<extra></extra>",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1.05])),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.22,
                    xanchor="center", x=0.5),
        margin=dict(l=50, r=50, t=30, b=70),
        height=height, template=PLOTLY_TEMPLATE,
    )
    return fig


# =====================================================================
#  Data Loading (cached)
# =====================================================================


@st.cache_data(show_spinner="Loading & processing banking data ...")
def load_and_process():
    """Run the full data pipeline + ML analysis and cache the result."""
    df_processed, df_original, scalers = process_data()
    detector = BankAnomalyDetector()
    df_result = detector.run_full_analysis(df_processed, df_original)
    return df_result, scalers


# =====================================================================
#  Sidebar - Filters
# =====================================================================

with st.sidebar:
    st.markdown(
        '<div style="text-align:center; padding:10px 0 0 0;">'
        '<span style="font-size:3rem;">&#127974;</span>'
        '</div>'
        '<p class="sidebar-title">BankGuard AI 360&#176;</p>'
        '<p class="sidebar-subtitle">Comprehensive Banking Monitoring System</p>'
        '<hr style="border:none; border-top:1px solid #dde2ef; margin:0 0 14px 0;">',
        unsafe_allow_html=True,
    )

    if st.button("Run Analysis", width='stretch', type="primary"):
        load_and_process.clear()
        st.rerun()

    st.markdown("#### Filters")

    df_full, _scalers = load_and_process()
    df_full["_period_dt"] = pd.to_datetime(df_full["period"], format="mixed")

    # Filter options
    all_periods: List[str] = df_full.sort_values("_period_dt")["period"].unique().tolist()
    all_regions: List[str] = sorted(df_full["region"].dropna().unique().tolist())
    all_bank_types: List[str] = sorted(df_full["bank_type"].dropna().unique().tolist())
    all_ratings: List[str] = sorted(
        df_full["external_credit_rating"].dropna().unique().tolist()
    ) if "external_credit_rating" in df_full.columns else []

    selected_periods = st.multiselect(
        "Period", options=all_periods, default=all_periods,
        help="Select one or more reporting periods.",
    )
    selected_regions = st.multiselect(
        "Region", options=all_regions, default=all_regions,
    )
    selected_bank_type = st.multiselect(
        "Bank Type", options=all_bank_types, default=all_bank_types,
    )
    selected_ratings = st.multiselect(
        "Credit Rating", options=all_ratings, default=all_ratings,
    ) if all_ratings else all_ratings

    st.markdown("---")
    st.caption("(c) 2026 BankGuard AI - KTNN")

# =====================================================================
#  Apply Filters
# =====================================================================

mask = (
    df_full["period"].isin(selected_periods)
    & df_full["region"].isin(selected_regions)
    & df_full["bank_type"].isin(selected_bank_type)
)
if selected_ratings and "external_credit_rating" in df_full.columns:
    mask = mask & df_full["external_credit_rating"].isin(selected_ratings)

df: pd.DataFrame = df_full[mask].copy()

if df.empty:
    st.warning("No data matches the current filter selection. Please adjust the sidebar filters.")
    st.stop()

# =====================================================================
#  Header
# =====================================================================

st.markdown(
    "<h1 style='margin-bottom:0'>&#127974; BankGuard AI "
    "<span style='font-size:0.55em;color:#6b7b95;font-weight:400;'>"
    "360&#176; Dashboard</span></h1>",
    unsafe_allow_html=True,
)
st.caption(
    f"Monitoring **{df['bank_id'].nunique()}** banks across "
    f"**{len(selected_periods)}** periods  |  "
    f"{len(df)} observations  |  "
    f"ML features: {len(ALL_ML_FEATURES)} across {len(RISK_PILLARS)} pillars  |  "
    f"Consensus: 3 models x 7 pillars"
)

# =====================================================================
#  Tabs (4)
# =====================================================================

tab_exec, tab_algo, tab_pillar, tab_profiler = st.tabs([
    "Executive Summary",
    "Multi-Algo Comparison",
    "Risk Pillar Deep Dive",
    "360 Bank Profiler",
])

# Consensus column names
CONSENSUS_COLS = [f"{PILLAR_KEYS[p]}_consensus" for p in RISK_PILLARS]

# -----------------------------------------------------------------
#  TAB 1 - Executive Summary
# -----------------------------------------------------------------

with tab_exec:
    st.subheader("Executive Summary")

    # Derive KPI values
    n_critical = int((df["Final_Hybrid_Risk_Status"] == "Critical").sum())
    n_warning  = int((df["Final_Hybrid_Risk_Status"] == "Warning").sum())
    n_normal   = int((df["Final_Hybrid_Risk_Status"] == "Normal").sum())
    avg_ml     = df["Overall_ML_Risk_Score"].mean()

    total_rule_violations = int(df["rule_risk_score"].sum()) if "rule_risk_score" in df.columns else 0
    high_conf_ml = int((df["Overall_ML_Risk_Score"] >= 60).sum())

    system_npl = df["npl_ratio"].mean()
    avg_car    = df["capital_adequacy_ratio"].mean()

    # KPI row
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Critical (Hybrid)", f"{n_critical}", help="ML >= 60 OR rule_risk >= 3")
    k2.metric("Warning (Hybrid)", f"{n_warning}", help="ML >= 30 OR rule_risk >= 1")
    k3.metric("Total Rule Violations", f"{total_rule_violations}")
    k4.metric("High-Conf ML Anomalies", f"{high_conf_ml}", help="Overall_ML_Risk_Score >= 60")
    k5.metric("System NPL Ratio", f"{system_npl:.4%}")
    k6.metric("Avg CAR", f"{avg_car:.2%}")

    st.markdown("")

    # -- Hybrid Status Distribution + Risk Trajectory --
    col_status, col_trajectory = st.columns([2, 3])

    with col_status:
        st.markdown("##### Hybrid Risk Status Distribution")
        status_counts = df["Final_Hybrid_Risk_Status"].value_counts().reindex(
            ["Critical", "Warning", "Normal"]
        ).fillna(0).astype(int)

        fig_pie = px.pie(
            names=status_counts.index, values=status_counts.values,
            color=status_counts.index, color_discrete_map=HYBRID_COLORS,
            hole=0.45, template=PLOTLY_TEMPLATE,
        )
        fig_pie.update_traces(
            textinfo="label+value+percent", textposition="outside",
            pull=[0.06 if x == "Critical" else 0 for x in status_counts.index],
        )
        fig_pie.update_layout(
            showlegend=False,
            margin=dict(l=10, r=10, t=30, b=10), height=400,
        )
        st.plotly_chart(fig_pie, width='stretch')

    with col_trajectory:
        st.markdown("##### Risk Trajectory - Overall ML Risk Score over Time")
        st.caption("Early-warning trend: rising scores indicate system-wide risk build-up.")

        if "_period_dt" in df.columns and df["_period_dt"].nunique() > 1:
            df_traj = (
                df.groupby("period")
                .agg(
                    Avg_ML_Score=("Overall_ML_Risk_Score", "mean"),
                    Max_ML_Score=("Overall_ML_Risk_Score", "max"),
                    N_Critical=("Final_Hybrid_Risk_Status", lambda x: (x == "Critical").sum()),
                    _period_dt=("_period_dt", "first"),
                )
                .sort_values("_period_dt")
                .reset_index()
            )

            fig_traj = go.Figure()
            fig_traj.add_trace(go.Scatter(
                x=df_traj["period"], y=df_traj["Avg_ML_Score"],
                mode="lines+markers", name="Avg ML Score",
                line=dict(color="#3498db", width=3),
                marker=dict(size=8),
                hovertemplate="Period: %{x}<br>Avg Score: %{y:.2f}<extra></extra>",
            ))
            fig_traj.add_trace(go.Scatter(
                x=df_traj["period"], y=df_traj["Max_ML_Score"],
                mode="lines+markers", name="Max ML Score",
                line=dict(color="#e74c3c", width=2, dash="dash"),
                marker=dict(size=6, symbol="diamond"),
                hovertemplate="Period: %{x}<br>Max Score: %{y:.2f}<extra></extra>",
            ))
            fig_traj.add_trace(go.Bar(
                x=df_traj["period"], y=df_traj["N_Critical"],
                name="# Critical Banks", yaxis="y2",
                marker_color="rgba(231,76,60,0.25)",
                hovertemplate="Period: %{x}<br>Critical: %{y}<extra></extra>",
            ))
            fig_traj.update_layout(
                yaxis=dict(title="ML Risk Score", rangemode="tozero"),
                yaxis2=dict(title="# Critical", overlaying="y", side="right",
                            rangemode="tozero", showgrid=False),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="center", x=0.5, title=None),
                template=PLOTLY_TEMPLATE,
                margin=dict(l=20, r=20, t=30, b=20), height=400,
            )
            st.plotly_chart(fig_traj, width='stretch')
        else:
            st.info("Only one period available - trajectory requires 2+ periods.")

    st.markdown("---")

    # -- Cluster Distribution + Anomaly Driver Groups --
    col_cluster, col_drivers = st.columns(2)

    with col_cluster:
        st.markdown("##### Risk Cluster Distribution (K-Means)")
        if "cluster_dna" in df.columns:
            dna_info = df.groupby("cluster_label")["cluster_dna"].first()
            for label, color in CLUSTER_COLORS.items():
                if label in dna_info.index:
                    st.markdown(
                        f'<span style="color:{color};font-weight:600;">{label}</span>'
                        f' - {dna_info[label]}', unsafe_allow_html=True)

        cc = df["cluster_label"].value_counts().reindex(
            CLUSTER_COLORS.keys()
        ).fillna(0).astype(int)
        fig_clust = px.bar(
            x=cc.index, y=cc.values,
            color=cc.index, color_discrete_map=CLUSTER_COLORS,
            template=PLOTLY_TEMPLATE,
            labels={"x": "Cluster", "y": "# Banks"},
        )
        fig_clust.update_layout(
            showlegend=False, height=320,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_clust, width='stretch')

    with col_drivers:
        st.markdown("##### Anomaly Driver Attribution (Critical Banks)")
        anom_df = df[df["is_anomaly"] == -1]
        if not anom_df.empty and "anomaly_driver_group" in anom_df.columns:
            grp_counts = anom_df["anomaly_driver_group"].value_counts()
            fig_drv = px.bar(
                x=grp_counts.index, y=grp_counts.values,
                color=grp_counts.index, color_discrete_map=PILLAR_COLORS,
                template=PLOTLY_TEMPLATE,
                labels={"x": "Risk Pillar", "y": "# Critical Banks"},
            )
            fig_drv.update_layout(
                showlegend=False, height=320,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig_drv, width='stretch')
        else:
            st.info("No critical anomalies in the current filter selection.")


# -----------------------------------------------------------------
#  TAB 2 - Multi-Algo Comparison
# -----------------------------------------------------------------

with tab_algo:
    st.subheader("Multi-Algorithm Objective Comparison")
    st.caption(
        "Agreement/disagreement between Isolation Forest, LOF, and One-Class SVM "
        "across all 7 risk pillars.  Each cell shows the consensus score (0/33/66/100)."
    )

    df_snap = _get_latest_snapshot(df)

    # -- 2A: Pillar Consensus Heatmap (banks x pillars) --
    st.markdown("##### Per-Pillar Consensus Heatmap (Latest Period)")
    st.caption("Rows = banks, Columns = risk pillars. Score: 0 (Normal) / 33 (Monitor) / 66 (Warning) / 100 (High Risk).")

    pillar_names = list(RISK_PILLARS.keys())
    consensus_cols_snap = [f"{PILLAR_KEYS[p]}_consensus" for p in pillar_names]
    avail_consensus = [c for c in consensus_cols_snap if c in df_snap.columns]

    if avail_consensus:
        hm_data = df_snap.set_index("bank_id")[avail_consensus].copy()
        hm_data.columns = [p for p, c in zip(pillar_names, consensus_cols_snap) if c in df_snap.columns]

        fig_hm = px.imshow(
            hm_data.values,
            x=hm_data.columns.tolist(),
            y=hm_data.index.tolist(),
            color_continuous_scale=[
                [0.0, "#d5f5e3"],
                [0.33, "#fef9e7"],
                [0.66, "#fdebd0"],
                [1.0, "#fadbd8"],
            ],
            aspect="auto",
            zmin=0, zmax=100,
            labels=dict(color="Consensus"),
            template=PLOTLY_TEMPLATE,
            text_auto=True,
        )
        fig_hm.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            height=max(350, len(hm_data) * 38),
            xaxis=dict(side="top"),
        )
        st.plotly_chart(fig_hm, width='stretch')
    else:
        st.warning("Consensus columns not found in the dataset.")

    st.markdown("---")

    # -- 2B: Model Agreement Matrix (per pillar) --
    st.markdown("##### Model Agreement Matrix - Per Pillar")
    st.caption(
        "For each pillar: how many banks were flagged by each model, "
        "and the overlap among IF, LOF, SVM."
    )

    agreement_rows = []
    for pillar_name in pillar_names:
        key = PILLAR_KEYS[pillar_name]
        if_col  = f"{key}_IF"
        lof_col = f"{key}_LOF"
        svm_col = f"{key}_SVM"

        if all(c in df_snap.columns for c in [if_col, lof_col, svm_col]):
            if_flag  = (df_snap[if_col]  == -1)
            lof_flag = (df_snap[lof_col] == -1)
            svm_flag = (df_snap[svm_col] == -1)

            agreement_rows.append({
                "Pillar": pillar_name,
                "IF Only":  int((if_flag & ~lof_flag & ~svm_flag).sum()),
                "LOF Only": int((~if_flag & lof_flag & ~svm_flag).sum()),
                "SVM Only": int((~if_flag & ~lof_flag & svm_flag).sum()),
                "IF+LOF":   int((if_flag & lof_flag & ~svm_flag).sum()),
                "IF+SVM":   int((if_flag & ~lof_flag & svm_flag).sum()),
                "LOF+SVM":  int((~if_flag & lof_flag & svm_flag).sum()),
                "All 3":    int((if_flag & lof_flag & svm_flag).sum()),
                "None":     int((~if_flag & ~lof_flag & ~svm_flag).sum()),
            })

    if agreement_rows:
        df_agree = pd.DataFrame(agreement_rows).set_index("Pillar")

        # Styled color heatmap for the overlap counts
        cols_overlap = ["IF Only", "LOF Only", "SVM Only", "IF+LOF", "IF+SVM", "LOF+SVM", "All 3"]
        fig_agree = px.imshow(
            df_agree[cols_overlap].values,
            x=cols_overlap,
            y=df_agree.index.tolist(),
            color_continuous_scale="YlOrRd",
            aspect="auto",
            labels=dict(color="# Banks"),
            template=PLOTLY_TEMPLATE,
            text_auto=True,
        )
        fig_agree.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            height=max(280, len(df_agree) * 42),
            xaxis=dict(side="top"),
        )
        st.plotly_chart(fig_agree, width='stretch')

        # Summary table
        st.markdown("##### Raw Counts")
        st.dataframe(df_agree, width='stretch')
    else:
        st.warning("Model prediction columns (IF/LOF/SVM) not found.")

    st.markdown("---")

    # -- 2C: Overall ML Score Distribution --
    st.markdown("##### Overall ML Risk Score Distribution")
    fig_hist = px.histogram(
        df_snap, x="Overall_ML_Risk_Score", nbins=25,
        color_discrete_sequence=["#3498db"],
        template=PLOTLY_TEMPLATE,
        labels={"Overall_ML_Risk_Score": "Overall ML Risk Score"},
    )
    fig_hist.add_vline(
        x=60, line_dash="dash", line_color="#e74c3c",
        annotation_text="Critical (60)", annotation_position="top right",
    )
    fig_hist.add_vline(
        x=30, line_dash="dot", line_color="#f39c12",
        annotation_text="Warning (30)", annotation_position="top left",
    )
    fig_hist.update_layout(
        margin=dict(l=20, r=20, t=30, b=20), height=350,
    )
    st.plotly_chart(fig_hist, width='stretch')


# -----------------------------------------------------------------
#  TAB 3 - Risk Pillar Deep Dive
# -----------------------------------------------------------------

with tab_pillar:
    st.subheader("Risk Pillar Deep Dive")

    selected_pillar = st.selectbox(
        "Select Risk Pillar",
        options=list(RISK_PILLARS.keys()),
        index=0,
    )

    pillar_features = RISK_PILLARS[selected_pillar]
    pillar_key = PILLAR_KEYS[selected_pillar]
    consensus_col = f"{pillar_key}_consensus"
    n_flags_col = f"{pillar_key}_n_flags"

    avail_feats = [f for f in pillar_features if f in df.columns]
    if not avail_feats:
        st.warning(f"Features for {selected_pillar} not found in the data.")
        st.stop()

    df_snap_pillar = _get_latest_snapshot(df)

    # Outlier coloring
    is_outlier = pd.Series("Normal", index=df_snap_pillar.index)
    if consensus_col in df_snap_pillar.columns:
        is_outlier = df_snap_pillar[consensus_col].apply(
            lambda x: "High Risk (3/3)" if x == 100
            else "Warning (2/3)" if x == 66
            else "Monitor (1/3)" if x == 33
            else "Normal"
        )
    df_snap_pillar["_outlier_status"] = is_outlier

    color_map_outlier = {
        "High Risk (3/3)": "#e74c3c",
        "Warning (2/3)":   "#f39c12",
        "Monitor (1/3)":   "#f1c40f",
        "Normal":          "#2ecc71",
    }

    # KPIs for the selected pillar
    st.markdown(f"##### {selected_pillar} - Pillar Analysis")

    p1, p2, p3, p4 = st.columns(4)
    if consensus_col in df_snap_pillar.columns:
        avg_consensus = df_snap_pillar[consensus_col].mean()
        n_100 = int((df_snap_pillar[consensus_col] == 100).sum())
        n_66  = int((df_snap_pillar[consensus_col] == 66).sum())
        n_33  = int((df_snap_pillar[consensus_col] == 33).sum())
        p1.metric("Avg Consensus Score", f"{avg_consensus:.1f}")
        p2.metric("High Risk (3/3)", f"{n_100}")
        p3.metric("Warning (2/3)", f"{n_66}")
        p4.metric("Monitor (1/3)", f"{n_33}")

    st.markdown("")

    # Dynamic scatter plots for each pair of features in the pillar
    if len(avail_feats) >= 2:
        st.markdown(f"##### Feature Scatter Plots - {selected_pillar}")
        st.caption("Red = 3/3 models agree (anomaly). Orange = 2/3. Yellow = 1/3. Green = Normal.")

        # Primary scatter: first two features
        for i in range(0, len(avail_feats) - 1, 2):
            f1 = avail_feats[i]
            f2 = avail_feats[i + 1] if i + 1 < len(avail_feats) else avail_feats[0]

            fig_sc = px.scatter(
                df_snap_pillar, x=f1, y=f2,
                color="_outlier_status",
                color_discrete_map=color_map_outlier,
                category_orders={"_outlier_status": list(color_map_outlier.keys())},
                hover_data={"bank_id": True, "_outlier_status": True, f1: ":.4f", f2: ":.4f"},
                labels={
                    f1: FEATURE_LABELS.get(f1, f1),
                    f2: FEATURE_LABELS.get(f2, f2),
                    "_outlier_status": "ML Consensus",
                },
                template=PLOTLY_TEMPLATE,
                size_max=14,
            )
            fig_sc.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
            fig_sc.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="center", x=0.5, title=None),
                margin=dict(l=20, r=20, t=30, b=20), height=440,
            )
            st.plotly_chart(fig_sc, width='stretch')

        # If odd number of features, show the last one vs first
        if len(avail_feats) % 2 == 1 and len(avail_feats) > 2:
            f1 = avail_feats[-1]
            f2 = avail_feats[0]
            fig_odd = px.scatter(
                df_snap_pillar, x=f1, y=f2,
                color="_outlier_status",
                color_discrete_map=color_map_outlier,
                category_orders={"_outlier_status": list(color_map_outlier.keys())},
                hover_data={"bank_id": True, f1: ":.4f", f2: ":.4f"},
                labels={
                    f1: FEATURE_LABELS.get(f1, f1),
                    f2: FEATURE_LABELS.get(f2, f2),
                    "_outlier_status": "ML Consensus",
                },
                template=PLOTLY_TEMPLATE,
            )
            fig_odd.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
            fig_odd.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="center", x=0.5, title=None),
                margin=dict(l=20, r=20, t=30, b=20), height=440,
            )
            st.plotly_chart(fig_odd, width='stretch')

    elif len(avail_feats) == 1:
        st.markdown(f"##### Distribution of {FEATURE_LABELS.get(avail_feats[0], avail_feats[0])}")
        fig_single = px.histogram(
            df_snap_pillar, x=avail_feats[0], color="_outlier_status",
            color_discrete_map=color_map_outlier,
            template=PLOTLY_TEMPLATE, nbins=20,
        )
        fig_single.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_single, width='stretch')

    st.markdown("---")

    # Feature statistics table
    st.markdown(f"##### {selected_pillar} - Feature Statistics")
    stats_df = df_snap_pillar[avail_feats].describe().T
    stats_df.insert(0, "Feature", [FEATURE_LABELS.get(f, f) for f in stats_df.index])
    st.dataframe(stats_df, width='stretch')

    # CSV download
    csv_pillar = io.StringIO()
    export_cols = ["bank_id", "period"] + avail_feats
    if consensus_col in df.columns:
        export_cols.append(consensus_col)
    export_cols = [c for c in export_cols if c in df.columns]
    df[export_cols].to_csv(csv_pillar, index=False)
    st.download_button(
        f"Download {selected_pillar} Data (CSV)",
        csv_pillar.getvalue(),
        file_name=f"bankguard_{pillar_key}_data.csv",
        mime="text/csv", width='stretch',
    )


# -----------------------------------------------------------------
#  TAB 4 - 360 Bank Profiler
# -----------------------------------------------------------------

with tab_profiler:
    st.subheader("360 Bank Profiler & Peer Benchmarking")

    bank_ids: List[str] = sorted(df["bank_id"].unique().tolist())
    if not bank_ids:
        st.info("No banks available for the current filter selection.")
        st.stop()

    selected_bank = st.selectbox("Select Bank ID", options=bank_ids, index=0)

    df_bank = df[df["bank_id"] == selected_bank].copy()
    if df_bank.empty:
        st.info(f"No data for **{selected_bank}**.")
        st.stop()

    if "_period_dt" in df_bank.columns:
        df_bank = df_bank.sort_values("_period_dt")
    latest = df_bank.iloc[-1]

    # -- Info bar --
    i1, i2, i3, i4, i5 = st.columns(5)
    i1.metric("Bank ID", selected_bank)
    i2.metric("Region", str(latest.get("region", "N/A")))
    i3.metric("Bank Type", str(latest.get("bank_type", "N/A")))

    hybrid_status = str(latest.get("Final_Hybrid_Risk_Status", "N/A"))
    status_icon = {"Critical": "!", "Warning": "~", "Normal": "OK"}.get(hybrid_status, "")
    i4.metric("Hybrid Status", f"{status_icon} {hybrid_status}")

    ml_score = latest.get("Overall_ML_Risk_Score", 0)
    i5.metric("ML Risk Score", f"{ml_score:.1f} / 100")

    st.markdown("")

    # -- Rule Violations Alert Box --
    rule_violations = latest.get("rule_violations", ["Compliant"])
    if isinstance(rule_violations, str):
        # Handle case where it's stored as string representation
        try:
            import ast
            rule_violations = ast.literal_eval(rule_violations)
        except (ValueError, SyntaxError):
            rule_violations = [rule_violations]

    rule_score = int(latest.get("rule_risk_score", 0))

    if rule_score > 0 and rule_violations != ["Compliant"]:
        violations_html = "".join(
            f"<li><strong>{v}</strong></li>" for v in rule_violations
        )
        st.markdown(
            f'<div class="rule-alert-box">'
            f'<strong>Expert Rule Violations ({rule_score})</strong>'
            f'<ul style="margin:6px 0 0 0; padding-left:18px;">'
            f'{violations_html}'
            f'</ul></div>',
            unsafe_allow_html=True,
        )
    else:
        st.success("**Compliant** - No expert rule violations detected.")

    st.markdown("")

    # -- 4A: Radar Chart - 7 Pillar Consensus Scores vs Peer Group --
    st.markdown("##### 7-Pillar Consensus Radar - Bank vs Peer Group")
    st.caption(
        "Comparing this bank's per-pillar consensus scores against the "
        "average of its **bank_type** peer group."
    )

    bank_type = latest.get("bank_type", None)
    peer_df = df[df["bank_type"] == bank_type] if bank_type else df

    pillar_labels = list(RISK_PILLARS.keys())
    bank_scores = []
    peer_scores = []

    for pname in pillar_labels:
        col = f"{PILLAR_KEYS[pname]}_consensus"
        if col in df.columns:
            bank_scores.append(float(latest.get(col, 0)))
            peer_scores.append(float(peer_df[col].mean()))
        else:
            bank_scores.append(0.0)
            peer_scores.append(0.0)

    fig_radar = _make_radar(
        labels=pillar_labels,
        bank_vals=bank_scores,
        peer_vals=peer_scores,
        bank_name=selected_bank,
        peer_name=f"Avg {bank_type}" if bank_type else "System Average",
        color=ANOMALY_COLOR if hybrid_status == "Critical" else "#f39c12" if hybrid_status == "Warning" else NORMAL_COLOR,
        height=480,
    )
    st.plotly_chart(fig_radar, width='stretch')

    st.markdown("---")

    # -- 4B: Per-Pillar Breakdown Bar --
    col_bar, col_detail = st.columns([3, 2])

    with col_bar:
        st.markdown("##### Per-Pillar Consensus Score Breakdown")
        bar_colors = [
            "#e74c3c" if s >= 66 else "#f39c12" if s >= 33 else "#2ecc71"
            for s in bank_scores
        ]
        fig_pbar = go.Figure(go.Bar(
            x=bank_scores, y=pillar_labels, orientation="h",
            marker=dict(color=bar_colors, line=dict(width=0)),
            text=[f"{s:.0f}" for s in bank_scores],
            textposition="auto",
            hovertemplate="<b>%{y}</b><br>Consensus: %{x:.0f}<extra></extra>",
        ))
        fig_pbar.update_layout(
            xaxis=dict(title="Consensus Score", range=[0, 105]),
            yaxis=dict(autorange="reversed"),
            template=PLOTLY_TEMPLATE,
            margin=dict(l=10, r=20, t=30, b=30), height=380,
        )
        st.plotly_chart(fig_pbar, width='stretch')

    with col_detail:
        st.markdown("##### Bank Details")
        detail_items = {
            "Cluster": latest.get("cluster_label", "N/A"),
            "Cluster DNA": latest.get("cluster_dna", "N/A"),
            "Anomaly Driver": FEATURE_LABELS.get(
                str(latest.get("anomaly_driver", "N/A")),
                str(latest.get("anomaly_driver", "N/A")),
            ),
            "Driver Group": latest.get("anomaly_driver_group", "N/A"),
            "OBS Risk": latest.get("obs_risk_flag", "N/A"),
            "Credit Rating": latest.get("external_credit_rating", "N/A"),
        }
        for label, val in detail_items.items():
            st.markdown(f"**{label}:** {val}")

    st.markdown("---")

    # -- 4C: Funding Structure Pie Charts --
    st.markdown("##### Funding Structure")

    col_fund1, col_fund2 = st.columns(2)

    with col_fund1:
        st.markdown("**Wholesale vs Deposit Funding**")
        wdr = float(latest.get("wholesale_dependency_ratio", 0))
        deposit_share = max(0, 1.0 - wdr)
        fig_fund = px.pie(
            names=["Wholesale Funding", "Deposit Funding"],
            values=[wdr, deposit_share],
            color_discrete_sequence=["#e74c3c", "#2ecc71"],
            hole=0.4, template=PLOTLY_TEMPLATE,
        )
        fig_fund.update_traces(
            textinfo="label+percent",
            textposition="outside",
        )
        fig_fund.update_layout(
            showlegend=False,
            margin=dict(l=10, r=10, t=30, b=10), height=380,
        )
        st.plotly_chart(fig_fund, width='stretch')

    with col_fund2:
        st.markdown("**Sector Loan Allocation**")
        avail_sec = [c for c in SECTOR_LOANS_COLUMNS if c in df.columns]
        if avail_sec:
            sec_vals = [float(latest.get(c, 0)) for c in avail_sec]
            sec_names = [SECTOR_LABELS.get(c, c) for c in avail_sec]
            fig_sec = px.pie(
                names=sec_names, values=sec_vals,
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.4, template=PLOTLY_TEMPLATE,
            )
            fig_sec.update_traces(
                textinfo="label+percent",
                textposition="outside",
            )
            fig_sec.update_layout(
                showlegend=False,
                margin=dict(l=10, r=10, t=30, b=10), height=380,
            )
            st.plotly_chart(fig_sec, width='stretch')
        else:
            st.info("Sector loan columns not available.")

    st.markdown("---")

    # -- 4D: Historical Trend --
    if len(df_bank) > 1:
        st.markdown(f"##### Historical ML Risk Trend - {selected_bank}")
        df_btrend = df_bank[["period", "Overall_ML_Risk_Score"]].copy()
        if "rule_risk_score" in df_bank.columns:
            df_btrend["rule_risk_score"] = df_bank["rule_risk_score"].values

        fig_btrend = go.Figure()
        fig_btrend.add_trace(go.Scatter(
            x=df_btrend["period"], y=df_btrend["Overall_ML_Risk_Score"],
            mode="lines+markers", name="ML Risk Score",
            line=dict(color="#3498db", width=3),
            hovertemplate="Period: %{x}<br>ML Score: %{y:.2f}<extra></extra>",
        ))
        if "rule_risk_score" in df_btrend.columns:
            fig_btrend.add_trace(go.Bar(
                x=df_btrend["period"], y=df_btrend["rule_risk_score"],
                name="Rule Violations", yaxis="y2",
                marker_color="rgba(231,76,60,0.3)",
                hovertemplate="Period: %{x}<br>Violations: %{y}<extra></extra>",
            ))
        fig_btrend.update_layout(
            yaxis=dict(title="ML Risk Score", rangemode="tozero"),
            yaxis2=dict(title="Rule Violations", overlaying="y", side="right",
                        rangemode="tozero", showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="center", x=0.5, title=None),
            template=PLOTLY_TEMPLATE,
            margin=dict(l=20, r=20, t=30, b=20), height=350,
        )
        st.plotly_chart(fig_btrend, width='stretch')

    # Full download
    st.markdown("---")
    full_csv = io.StringIO()
    df.to_csv(full_csv, index=False)
    st.download_button(
        "Download Full Report (CSV)", full_csv.getvalue(),
        file_name="bankguard_full_report.csv", mime="text/csv",
        width='stretch',
    )
