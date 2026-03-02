"""
app.py - BankGuard AI Interactive Dashboard (config.py edition).

5-tab Streamlit command-center importing **all** column definitions from
``config.py``:

    Tab 1 – System Overview   (KPI + scatter risk map + cluster pie)
    Tab 2 – Anomaly Alerts    (table + driver-group attribution)
    Tab 3 – Individual Deep-Dive  (radar per feature GROUP + bar + trend)
    Tab 4 – Sectoral Analysis (SECTOR_LOANS_COLUMNS heatmap/stack)
    Tab 5 – Off-Balance Sheet (EXPOSURE_LIQUIDITY_FEATURES gauge + scatter)

Author : BankGuard AI Team
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
    CONCENTRATION_RISK_FEATURES,
    EXPOSURE_LIQUIDITY_FEATURES,
    FINANCIAL_HEALTH_FEATURES,
    MODEL_PARAMS,
    SECTOR_LOANS_COLUMNS,
)
from models.anomaly_detector import BankAnomalyDetector
from utils.data_processor import process_data

# ═══════════════════════════════════════════════════════════════════════════
#  Global Config & Theme
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="BankGuard AI – Banking Monitoring System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Colour palette
COLOR_MAP: Dict[str, str] = {
    "Low Risk": "#2ecc71",
    "Medium Risk": "#f39c12",
    "High Risk": "#e74c3c",
}
ANOMALY_COLOR = "#e74c3c"
NORMAL_COLOR = "#2ecc71"
PLOTLY_TEMPLATE = "plotly_white"

# ── Human-readable labels for every ML feature ──────────────────────────
FEATURE_LABELS: Dict[str, str] = {
    # Financial Health (9)
    "capital_adequacy_ratio": "Capital Adequacy (CAR)",
    "npl_ratio": "NPL Ratio",
    "liquidity_coverage_ratio": "Liquidity Coverage (LCR)",
    "nsfr": "NSFR",
    "provision_coverage_ratio": "Provision Coverage",
    "loan_to_deposit_ratio": "Loan-to-Deposit",
    "return_on_assets": "ROA",
    "return_on_equity": "ROE",
    "net_interest_margin": "Net Interest Margin",
    # Concentration Risk (5)
    "sector_concentration_hhi": "Sector HHI",
    "top20_borrower_concentration": "Top-20 Borrower",
    "geographic_concentration": "Geographic Conc.",
    "top20_depositors_ratio": "Top-20 Depositors",
    "top5_depositors_ratio": "Top-5 Depositors",
    # Exposure & Liquidity (6)
    "derivatives_to_assets_ratio": "Derivatives / Assets",
    "unused_lines_to_loans_ratio": "Unused Lines / Loans",
    "guarantees_to_loans_ratio": "Guarantees / Loans",
    "obs_to_assets_ratio": "OBS / Assets",
    "wholesale_dependency_ratio": "Wholesale Dependency",
    "liquidity_concentration_risk": "Liquidity Conc. Risk",
}

# Sector-loan pretty labels
SECTOR_LABELS: Dict[str, str] = {
    "sector_loans_energy": "Energy",
    "sector_loans_real_estate": "Real Estate",
    "sector_loans_construction": "Construction",
    "sector_loans_services": "Services",
    "sector_loans_agriculture": "Agriculture",
}

# Feature groups for radar charts (ordered maps)
FEATURE_GROUPS: Dict[str, List[str]] = {
    "Financial Health": FINANCIAL_HEALTH_FEATURES,
    "Concentration Risk": CONCENTRATION_RISK_FEATURES,
    "Exposure & Liquidity": EXPOSURE_LIQUIDITY_FEATURES,
}
GROUP_COLORS: Dict[str, str] = {
    "Financial Health": "#3498db",
    "Concentration Risk": "#9b59b6",
    "Exposure & Liquidity": "#e67e22",
}

# ═══════════════════════════════════════════════════════════════════════════
#  Custom CSS
# ═══════════════════════════════════════════════════════════════════════════

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
    button[data-baseweb="tab"] { font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _get_latest_snapshot(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the latest-period row per bank_id."""
    if "_period_dt" in frame.columns:
        return frame.sort_values("_period_dt").groupby("bank_id").last().reset_index()
    return frame.groupby("bank_id").last().reset_index()


def _make_radar(
    labels: List[str],
    bank_vals: List[float],
    sys_vals: List[float],
    bank_name: str,
    color: str = ANOMALY_COLOR,
    height: int = 400,
) -> go.Figure:
    """Build a normalised [0,1] radar chart comparing *bank* vs *system*."""
    all_v = bank_vals + sys_vals
    lo, hi = (min(all_v), max(all_v)) if all_v else (0, 1)
    span = hi - lo if hi != lo else 1
    nb = [(v - lo) / span for v in bank_vals]
    ns = [(v - lo) / span for v in sys_vals]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=nb + [nb[0]], theta=labels + [labels[0]], fill="toself",
        name=bank_name, line=dict(color=color, width=2),
        fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.15)",
        customdata=bank_vals + [bank_vals[0]],
        hovertemplate="<b>%{theta}</b><br>Value: %{customdata:.6f}<extra></extra>",
    ))
    fig.add_trace(go.Scatterpolar(
        r=ns + [ns[0]], theta=labels + [labels[0]], fill="toself",
        name="System Average", line=dict(color="#3498db", width=2, dash="dash"),
        fillcolor="rgba(52,152,219,0.10)",
        customdata=sys_vals + [sys_vals[0]],
        hovertemplate="<b>%{theta}</b><br>Value: %{customdata:.6f}<extra></extra>",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1.05])),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
        margin=dict(l=50, r=50, t=30, b=70), height=height, template=PLOTLY_TEMPLATE,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  Data Loading (cached)
# ═══════════════════════════════════════════════════════════════════════════


@st.cache_data(show_spinner="Loading & processing banking data …")
def load_and_process():
    """Run the full data pipeline + ML analysis and cache the result."""
    df_processed, df_original, scalers = process_data()
    detector = BankAnomalyDetector()
    df_result = detector.run_full_analysis(df_processed, df_original)
    return df_result, scalers


# ═══════════════════════════════════════════════════════════════════════════
#  Sidebar
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding:10px 0 0 0;">
            <span style="font-size:3rem;">🏦</span>
        </div>
        <p class="sidebar-title">BankGuard AI</p>
        <p class="sidebar-subtitle">Comprehensive Banking Monitoring System</p>
        <hr style="border:none; border-top:1px solid #dde2ef; margin:0 0 14px 0;">
        """,
        unsafe_allow_html=True,
    )

    if st.button("🔄  Run Analysis", width='stretch', type="primary"):
        load_and_process.clear()
        st.rerun()

    st.markdown("#### Filters")

    df_full, _scalers = load_and_process()
    df_full["_period_dt"] = pd.to_datetime(df_full["period"], format="mixed")

    all_periods: List[str] = df_full.sort_values("_period_dt")["period"].unique().tolist()
    all_regions: List[str] = sorted(df_full["region"].dropna().unique().tolist())
    all_bank_types: List[str] = sorted(df_full["bank_type"].dropna().unique().tolist())

    selected_periods = st.multiselect("Period", options=all_periods, default=all_periods,
                                      help="Select one or more reporting periods.")
    selected_regions = st.multiselect("Region", options=all_regions, default=all_regions)
    selected_bank_type = st.multiselect("Bank Type", options=all_bank_types, default=all_bank_types)

    st.markdown("---")
    st.caption("© 2026 BankGuard AI – KTNN")

# ═══════════════════════════════════════════════════════════════════════════
#  Apply Filters
# ═══════════════════════════════════════════════════════════════════════════

df: pd.DataFrame = df_full[
    df_full["period"].isin(selected_periods)
    & df_full["region"].isin(selected_regions)
    & df_full["bank_type"].isin(selected_bank_type)
].copy()

if df.empty:
    st.warning("⚠️  No data matches the current filter selection. Please adjust the sidebar filters.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════
#  Header
# ═══════════════════════════════════════════════════════════════════════════

st.markdown(
    "<h1 style='margin-bottom:0'>🏦 BankGuard AI "
    "<span style='font-size:0.55em;color:#6b7b95;font-weight:400;'>Dashboard</span></h1>",
    unsafe_allow_html=True,
)
st.caption(
    f"Monitoring **{df['bank_id'].nunique()}** banks across "
    f"**{len(selected_periods)}** periods  •  "
    f"{len(df)} observations  •  "
    f"ML features: {len(ALL_ML_FEATURES)}"
)

# ═══════════════════════════════════════════════════════════════════════════
#  Tabs (5)
# ═══════════════════════════════════════════════════════════════════════════

tab_overview, tab_alerts, tab_deepdive, tab_sector, tab_obs = st.tabs([
    "📊  System Overview",
    "🚨  Anomaly Alerts",
    "🔍  Individual Deep-Dive",
    "🏭  Sectoral Analysis",
    "📋  Off-Balance Sheet",
])

# ─────────────────────────────────────────────────────────────────────────
#  TAB 1 – System Overview
# ─────────────────────────────────────────────────────────────────────────

with tab_overview:
    st.subheader("Executive Summary")

    total_assets = df["total_assets"].sum()
    system_npl = df["npl_ratio"].mean()
    n_anomalies = int((df["is_anomaly"] == -1).sum())
    avg_car = df["capital_adequacy_ratio"].mean()
    n_obs_high = int((df["obs_risk_flag"] == "High OBS Risk").sum())

    npl_delta = None
    if "_period_dt" in df.columns and df["_period_dt"].nunique() > 1:
        latest_period = df["_period_dt"].max()
        prev = df[df["_period_dt"] < latest_period]
        if not prev.empty:
            npl_delta = round(system_npl - prev["npl_ratio"].mean(), 6)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Assets Monitored", f"{total_assets:,.0f} B")
    k2.metric("System-wide NPL Ratio", f"{system_npl:.4%}",
              delta=f"{npl_delta:+.4%}" if npl_delta is not None else None,
              delta_color="inverse")
    k3.metric("Critical Anomalies", f"{n_anomalies}")
    k4.metric("Avg. CAR", f"{avg_car:.2%}")
    k5.metric("High OBS Risk", f"{n_obs_high}")

    st.markdown("")

    col_scatter, col_pie = st.columns([3, 2])

    with col_scatter:
        st.markdown("##### Interactive Risk Map")
        fig_scatter = px.scatter(
            df, x="capital_adequacy_ratio", y="npl_ratio",
            color="cluster_label", color_discrete_map=COLOR_MAP,
            size="total_assets", size_max=28,
            hover_data={"bank_id": True, "period": True,
                        "anomaly_score": ":.4f",
                        "capital_adequacy_ratio": ":.4%",
                        "npl_ratio": ":.4%",
                        "total_assets": ":,.0f", "cluster_label": True},
            labels={"capital_adequacy_ratio": "CAR", "npl_ratio": "NPL",
                    "cluster_label": "Risk Cluster", "total_assets": "Total Assets"},
            template=PLOTLY_TEMPLATE,
            category_orders={"cluster_label": list(COLOR_MAP.keys())},
        )
        fig_scatter.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="center", x=0.5, title=None),
            margin=dict(l=20, r=20, t=30, b=20), height=440,
            xaxis_tickformat=".1%", yaxis_tickformat=".2%",
        )
        st.plotly_chart(fig_scatter, width='stretch')

    with col_pie:
        st.markdown("##### Cluster Distribution & DNA")
        if "cluster_dna" in df.columns:
            dna_info = df.groupby("cluster_label")["cluster_dna"].first()
            for label, color in COLOR_MAP.items():
                if label in dna_info.index:
                    st.markdown(
                        f'<span style="color:{color};font-weight:600;">{label}</span>'
                        f' — {dna_info[label]}', unsafe_allow_html=True)

        cc = df["cluster_label"].value_counts().reindex(COLOR_MAP.keys()).fillna(0).astype(int)
        fig_pie = px.pie(names=cc.index, values=cc.values,
                         color=cc.index, color_discrete_map=COLOR_MAP,
                         hole=0.45, template=PLOTLY_TEMPLATE)
        fig_pie.update_traces(textinfo="label+percent", textposition="outside",
                              pull=[0, 0, 0.05])
        fig_pie.update_layout(showlegend=False,
                              margin=dict(l=10, r=10, t=30, b=10), height=440)
        st.plotly_chart(fig_pie, width='stretch')

    # ── Anomaly Driver Group Distribution ────────────────────────────
    if "anomaly_driver_group" in df.columns:
        anom_df = df[df["is_anomaly"] == -1]
        if not anom_df.empty:
            st.markdown("##### Anomaly Driver Group Distribution")
            grp_counts = anom_df["anomaly_driver_group"].value_counts()
            fig_grp = px.bar(
                x=grp_counts.index, y=grp_counts.values,
                color=grp_counts.index,
                color_discrete_map=GROUP_COLORS,
                template=PLOTLY_TEMPLATE,
                labels={"x": "Risk Group", "y": "# Anomalies"},
            )
            fig_grp.update_layout(
                showlegend=False, height=300,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig_grp, width='stretch')

# ─────────────────────────────────────────────────────────────────────────
#  TAB 2 – Anomaly Alerts
# ─────────────────────────────────────────────────────────────────────────

with tab_alerts:
    st.subheader("🚨 Anomaly Alert Centre")

    alert_cols = [
        "bank_id", "period", "npl_ratio", "capital_adequacy_ratio",
        "liquidity_coverage_ratio", "anomaly_score", "is_anomaly",
        "cluster_label", "cluster_dna", "anomaly_driver",
        "anomaly_driver_group", "obs_risk_flag", "obs_risk_zscore",
    ]
    # Only keep columns that exist
    alert_cols = [c for c in alert_cols if c in df.columns]
    df_alerts = df[alert_cols].copy()

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Total Observations", len(df_alerts))
    a2.metric("Flagged Anomalies", int((df_alerts["is_anomaly"] == -1).sum()))
    a3.metric("Anomaly Rate", f"{(df_alerts['is_anomaly'] == -1).mean():.1%}")
    a4.metric("High OBS Risk",
              int((df_alerts["obs_risk_flag"] == "High OBS Risk").sum())
              if "obs_risk_flag" in df_alerts.columns else 0)

    st.markdown("")

    def _highlight_anomaly(row: pd.Series) -> list[str]:
        if row.get("is_anomaly") == -1:
            return ["background-color: #fde8e8; color: #8b1a1a;"] * len(row)
        return [""] * len(row)

    fmt = {
        "npl_ratio": "{:.4%}",
        "capital_adequacy_ratio": "{:.4%}",
        "liquidity_coverage_ratio": "{:.4f}",
        "anomaly_score": "{:.6f}",
        "obs_risk_zscore": "{:.4f}",
    }
    fmt = {k: v for k, v in fmt.items() if k in df_alerts.columns}

    styled = (
        df_alerts.sort_values("anomaly_score", ascending=True)
        .reset_index(drop=True)
        .style.apply(_highlight_anomaly, axis=1)
        .format(fmt)
    )
    st.dataframe(styled, width='stretch', height=480)

    csv_buf = io.StringIO()
    df_alerts.to_csv(csv_buf, index=False)
    st.download_button("📥  Download Anomaly Report (CSV)", csv_buf.getvalue(),
                       file_name="bankguard_anomaly_report.csv", mime="text/csv",
                       width='stretch')

# ─────────────────────────────────────────────────────────────────────────
#  TAB 3 – Individual Deep-Dive (ALL feature groups)
# ─────────────────────────────────────────────────────────────────────────

with tab_deepdive:
    st.subheader("🔍 Individual Bank Analysis")

    bank_ids: List[str] = sorted(df["bank_id"].unique().tolist())
    if not bank_ids:
        st.info("No banks available for the current filter selection.")
        st.stop()

    selected_bank = st.selectbox("Select a Bank", options=bank_ids, index=0)
    df_bank = df[df["bank_id"] == selected_bank].copy()
    if df_bank.empty:
        st.info(f"No data for bank **{selected_bank}** in the current filters.")
        st.stop()

    if "_period_dt" in df_bank.columns:
        df_bank = df_bank.sort_values("_period_dt")
    latest_row = df_bank.iloc[-1]

    # ── Info bar ─────────────────────────────────────────────────────
    i1, i2, i3, i4, i5 = st.columns(5)
    i1.metric("Bank ID", selected_bank)
    i2.metric("Region", latest_row.get("region", "N/A"))
    i3.metric("Cluster", latest_row.get("cluster_label", "N/A"))
    i4.metric("Anomaly Status",
              "⚠️ ANOMALY" if latest_row.get("is_anomaly") == -1 else "✅ Normal")
    obs_f = latest_row.get("obs_risk_flag", "N/A")
    i5.metric("OBS Risk", ("🔴 " + obs_f) if obs_f == "High OBS Risk" else obs_f)

    dna_val = latest_row.get("cluster_dna", "N/A")
    if dna_val and dna_val != "N/A":
        st.info(f"**Sector DNA**: {dna_val}")

    drv = latest_row.get("anomaly_driver", "N/A")
    drv_grp = latest_row.get("anomaly_driver_group", "N/A")
    if drv != "N/A":
        st.warning(f"**Primary Anomaly Driver**: {FEATURE_LABELS.get(drv, drv)}  "
                   f"(Group: *{drv_grp}*)")

    st.markdown("")

    # ── Radar charts per feature group ───────────────────────────────
    st.markdown("##### Peer Comparison – Radar Charts by Risk Group")
    radar_cols = st.columns(len(FEATURE_GROUPS))

    for idx, (group_name, group_feats) in enumerate(FEATURE_GROUPS.items()):
        avail = [f for f in group_feats if f in df.columns]
        if not avail:
            continue
        with radar_cols[idx]:
            st.markdown(f"**{group_name}** ({len(avail)} features)")
            labels = [FEATURE_LABELS.get(f, f) for f in avail]
            bv = [float(latest_row[f]) for f in avail]
            sv = [float(df[f].mean()) for f in avail]
            fig_r = _make_radar(labels, bv, sv, selected_bank,
                                color=GROUP_COLORS.get(group_name, ANOMALY_COLOR),
                                height=380)
            st.plotly_chart(fig_r, width='stretch')

    st.markdown("")

    # ── Risk Factor Breakdown: Horizontal bar (ALL 20 features) ──────
    st.markdown("##### Risk Factor Breakdown – Top Deviations (All 20 Features)")

    avail_ml = [f for f in ALL_ML_FEATURES if f in df.columns]
    medians = df[avail_ml].median()
    stds = df[avail_ml].std().replace(0, np.nan)
    bvals = pd.Series({f: float(latest_row[f]) for f in avail_ml})
    z_devs = ((bvals - medians) / stds).abs().sort_values(ascending=False)
    top_n = min(10, len(z_devs))
    topf = z_devs.head(top_n)

    fig_bar = go.Figure(go.Bar(
        x=topf.values,
        y=[FEATURE_LABELS.get(f, f) for f in topf.index],
        orientation="h",
        marker=dict(
            color=[ANOMALY_COLOR if v > 1.5 else COLOR_MAP["Medium Risk"] if v > 0.75
                   else NORMAL_COLOR for v in topf.values],
            line=dict(width=0),
        ),
        text=[f"{v:.2f}σ" for v in topf.values],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Z-deviation: %{x:.3f}σ<extra></extra>",
    ))
    fig_bar.update_layout(
        xaxis_title="Absolute Z-Score Deviation from Median",
        yaxis=dict(autorange="reversed"),
        template=PLOTLY_TEMPLATE,
        margin=dict(l=10, r=40, t=30, b=40), height=420,
    )
    st.plotly_chart(fig_bar, width='stretch')

    # ── Time-series trend ────────────────────────────────────────────
    if len(df_bank) > 1:
        st.markdown(f"##### Historical Trend – {selected_bank}")
        trend_feats = ["npl_ratio", "capital_adequacy_ratio", "liquidity_coverage_ratio",
                       "net_interest_margin"]
        trend_feats = [f for f in trend_feats if f in df_bank.columns]
        df_trend = df_bank[["period", "_period_dt"] + trend_feats].copy() if "_period_dt" in df_bank.columns else df_bank[["period"] + trend_feats].copy()
        if "_period_dt" in df_trend.columns:
            df_trend = df_trend.sort_values("_period_dt")

        df_m = df_trend.melt(id_vars="period", value_vars=trend_feats,
                             var_name="Metric", value_name="Value")
        df_m["Metric"] = df_m["Metric"].map(FEATURE_LABELS)
        fig_trend = px.line(df_m, x="period", y="Value", color="Metric",
                            markers=True, template=PLOTLY_TEMPLATE,
                            color_discrete_sequence=px.colors.qualitative.Set2)
        fig_trend.update_layout(
            xaxis_title="Period", yaxis_title="Value",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="center", x=0.5, title=None),
            margin=dict(l=20, r=20, t=30, b=20), height=340,
        )
        st.plotly_chart(fig_trend, width='stretch')

    st.markdown("---")
    full_csv = io.StringIO()
    df.to_csv(full_csv, index=False)
    st.download_button("📥  Download Full Report (CSV)", full_csv.getvalue(),
                       file_name="bankguard_full_report.csv", mime="text/csv",
                       width='stretch')

# ─────────────────────────────────────────────────────────────────────────
#  TAB 4 – Sectoral Analysis  (SECTOR_LOANS_COLUMNS)
# ─────────────────────────────────────────────────────────────────────────

with tab_sector:
    st.subheader("🏭 Sectoral Analysis")
    st.caption("Sector lending exposure sourced from `config.SECTOR_LOANS_COLUMNS`.")

    avail_sec = [c for c in SECTOR_LOANS_COLUMNS if c in df.columns]

    if not avail_sec:
        st.info("Sector-loan columns not found in the dataset.")
    else:
        df_snap = _get_latest_snapshot(df)

        # ── 4-A  Stacked bar of sector loans per bank ───────────────
        st.markdown("##### Lending Structure by Bank (Latest Period)")

        df_sec = df_snap[["bank_id"] + avail_sec].copy()
        df_melt = df_sec.melt(id_vars="bank_id", value_vars=avail_sec,
                              var_name="Sector", value_name="Loan Amount")
        df_melt["Sector"] = df_melt["Sector"].map(SECTOR_LABELS)

        fig_stack = px.bar(
            df_melt, x="bank_id", y="Loan Amount", color="Sector",
            barmode="stack", template=PLOTLY_TEMPLATE,
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={"bank_id": "Bank", "Loan Amount": "Loan Amount (B)"},
        )
        fig_stack.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="center", x=0.5, title=None),
            margin=dict(l=20, r=20, t=40, b=20), height=440,
        )
        st.plotly_chart(fig_stack, width='stretch')

        st.markdown("---")

        # ── 4-B  Heatmap of sector proportions ──────────────────────
        st.markdown("##### Sector Proportion Heatmap (% of total loans per bank)")

        df_pct = df_sec.set_index("bank_id")
        totals = df_pct.sum(axis=1)
        df_pct = df_pct.div(totals, axis=0) * 100
        df_pct.columns = [SECTOR_LABELS.get(c, c) for c in df_pct.columns]

        fig_hm = px.imshow(
            df_pct.values,
            x=df_pct.columns.tolist(),
            y=df_pct.index.tolist(),
            color_continuous_scale="YlOrRd",
            aspect="auto",
            labels=dict(color="% of Total"),
            template=PLOTLY_TEMPLATE,
            text_auto=".1f",
        )
        fig_hm.update_layout(
            margin=dict(l=20, r=20, t=30, b=20), height=max(300, len(df_pct) * 40),
        )
        st.plotly_chart(fig_hm, width='stretch')

        st.markdown("---")

        # ── 4-C  Concentration Radar – Bank vs System ───────────────
        st.markdown("##### Concentration Radar – Bank vs System")

        conc_avail = [f for f in CONCENTRATION_RISK_FEATURES if f in df.columns]
        if conc_avail:
            sec_bank_ids = sorted(df["bank_id"].unique().tolist())
            sec_sel = st.selectbox("Select Bank for Concentration Radar",
                                   options=sec_bank_ids, index=0, key="sec_conc_bank")
            df_sb = df[df["bank_id"] == sec_sel]
            if "_period_dt" in df_sb.columns:
                df_sb = df_sb.sort_values("_period_dt")
            sb_latest = df_sb.iloc[-1]

            c_labels = [FEATURE_LABELS.get(f, f) for f in conc_avail]
            c_bank = [float(sb_latest[f]) for f in conc_avail]
            c_sys = [float(df[f].mean()) for f in conc_avail]
            fig_cr = _make_radar(c_labels, c_bank, c_sys, sec_sel,
                                 color=GROUP_COLORS["Concentration Risk"], height=440)
            st.plotly_chart(fig_cr, width='stretch')
        else:
            st.info("Concentration features not available.")

# ─────────────────────────────────────────────────────────────────────────
#  TAB 5 – Off-Balance Sheet  (EXPOSURE_LIQUIDITY_FEATURES)
# ─────────────────────────────────────────────────────────────────────────

with tab_obs:
    st.subheader("📋 Off-Balance Sheet & Exposure Analysis")
    st.caption("Features from `config.EXPOSURE_LIQUIDITY_FEATURES`.")

    exp_avail = [f for f in EXPOSURE_LIQUIDITY_FEATURES if f in df.columns]
    if not exp_avail:
        st.info("Exposure / liquidity features not found in the dataset.")
    else:
        df_snap_obs = _get_latest_snapshot(df)

        # ── 5-A  Summary KPIs ───────────────────────────────────────────
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Avg OBS / Assets", f"{df_snap_obs['obs_to_assets_ratio'].mean():.4f}"
                  if "obs_to_assets_ratio" in df_snap_obs.columns else "N/A")
        e2.metric("Avg Wholesale Dependency", f"{df_snap_obs['wholesale_dependency_ratio'].mean():.2%}"
                  if "wholesale_dependency_ratio" in df_snap_obs.columns else "N/A")
        e3.metric("Avg Derivatives / Assets", f"{df_snap_obs['derivatives_to_assets_ratio'].mean():.4f}"
                  if "derivatives_to_assets_ratio" in df_snap_obs.columns else "N/A")
        e4.metric("High OBS Risk Banks",
                  int((df_snap_obs["obs_risk_flag"] == "High OBS Risk").sum())
                  if "obs_risk_flag" in df_snap_obs.columns else 0)

        st.markdown("")

        # ── 5-B  Exposure Radar – Bank vs System ────────────────────
        st.markdown("##### Exposure & Liquidity Radar – Bank vs System")

        obs_bank_ids = sorted(df["bank_id"].unique().tolist())
        obs_sel = st.selectbox("Select Bank for Exposure Radar",
                               options=obs_bank_ids, index=0, key="obs_radar_bank")
        df_ob = df[df["bank_id"] == obs_sel]
        if "_period_dt" in df_ob.columns:
            df_ob = df_ob.sort_values("_period_dt")
        ob_latest = df_ob.iloc[-1]

        e_labels = [FEATURE_LABELS.get(f, f) for f in exp_avail]
        e_bank = [float(ob_latest[f]) for f in exp_avail]
        e_sys = [float(df[f].mean()) for f in exp_avail]
        fig_er = _make_radar(e_labels, e_bank, e_sys, obs_sel,
                             color=GROUP_COLORS["Exposure & Liquidity"], height=440)
        st.plotly_chart(fig_er, width='stretch')

        st.markdown("---")

        # ── 5-C  Funding Stress Scatter ──────────────────────────────
        st.markdown("##### Funding Stress Test")
        st.caption(
            "Banks in the **upper-left** quadrant rely heavily on wholesale "
            "funding while having low liquidity coverage — a dangerous combination."
        )

        if ("wholesale_dependency_ratio" in df.columns
                and "liquidity_coverage_ratio" in df.columns):
            df_fund = df_snap_obs.copy()
            wdr_med = float(df_fund["wholesale_dependency_ratio"].median())
            lcr_med = float(df_fund["liquidity_coverage_ratio"].median())

            fig_fund = px.scatter(
                df_fund, x="liquidity_coverage_ratio", y="wholesale_dependency_ratio",
                color="cluster_label", color_discrete_map=COLOR_MAP,
                size="total_assets" if "total_assets" in df_fund.columns else None,
                size_max=24,
                hover_data={"bank_id": True,
                            "wholesale_dependency_ratio": ":.2%",
                            "liquidity_coverage_ratio": ":.4f",
                            "cluster_label": True},
                labels={"liquidity_coverage_ratio": "LCR",
                        "wholesale_dependency_ratio": "Wholesale Dependency",
                        "cluster_label": "Risk Cluster"},
                template=PLOTLY_TEMPLATE,
                category_orders={"cluster_label": list(COLOR_MAP.keys())},
            )
            fig_fund.add_hline(y=wdr_med, line_dash="dot", line_color="grey",
                               annotation_text=f"WDR median ({wdr_med:.2%})",
                               annotation_position="top left")
            fig_fund.add_vline(x=lcr_med, line_dash="dot", line_color="grey",
                               annotation_text=f"LCR median ({lcr_med:.4f})",
                               annotation_position="top right")
            fig_fund.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="center", x=0.5, title=None),
                margin=dict(l=20, r=20, t=40, b=20), height=440,
                yaxis_tickformat=".1%",
            )
            st.plotly_chart(fig_fund, width='stretch')
        else:
            st.info("wholesale_dependency_ratio or liquidity_coverage_ratio not available.")

        st.markdown("---")

        # ── 5-D  OBS Gauge per bank ─────────────────────────────────
        st.markdown("##### Off-Balance Sheet Warning – Derivatives Leverage")
        st.caption("Notional value of derivatives relative to total assets.")

        if "derivatives_notional" in df.columns and "total_assets" in df.columns:
            obs_g_ids = sorted(df["bank_id"].unique().tolist())
            obs_g_sel = st.selectbox("Select Bank for OBS Gauge",
                                     options=obs_g_ids, index=0, key="obs_gauge_bank")
            df_g = df[df["bank_id"] == obs_g_sel]
            if "_period_dt" in df_g.columns:
                df_g = df_g.sort_values("_period_dt")
            g_latest = df_g.iloc[-1]

            notional = float(g_latest["derivatives_notional"])
            assets = float(g_latest["total_assets"])
            leverage_pct = (notional / assets * 100) if assets > 0 else 0

            sys_max = float(df["derivatives_notional"].max()
                            / df["total_assets"].mean() * 100)
            gauge_max = max(200, sys_max * 1.2)

            g1, g2 = st.columns([3, 2])

            with g1:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=leverage_pct,
                    number={"suffix": "%", "font": {"size": 40}},
                    delta={"reference": float(
                        df["derivatives_notional"].median()
                        / df["total_assets"].median() * 100),
                        "relative": False, "suffix": " pp vs median"},
                    title={"text": f"{obs_g_sel} – Derivatives / Assets"},
                    gauge={
                        "axis": {"range": [0, gauge_max], "ticksuffix": "%"},
                        "bar": {"color": "#2c3e50"},
                        "steps": [
                            {"range": [0, 50], "color": "#d5f5e3"},
                            {"range": [50, 100], "color": "#fdebd0"},
                            {"range": [100, gauge_max], "color": "#fadbd8"},
                        ],
                        "threshold": {
                            "line": {"color": ANOMALY_COLOR, "width": 4},
                            "thickness": 0.8, "value": leverage_pct,
                        },
                    },
                ))
                fig_gauge.update_layout(height=340,
                                        margin=dict(l=30, r=30, t=60, b=10))
                st.plotly_chart(fig_gauge, width='stretch')

            with g2:
                st.markdown("")
                st.markdown("")
                rl = ("🟢 **Low leverage** – < 50%" if leverage_pct < 50
                      else "🟡 **Moderate leverage** – 50-100%" if leverage_pct < 100
                      else "🔴 **High leverage** – > 100%")
                st.markdown(rl)
                st.metric("Derivatives Notional", f"{notional:,.0f} B")
                st.metric("Total Assets", f"{assets:,.0f} B")
                st.metric("Leverage Ratio", f"{leverage_pct:.1f}%")
        else:
            st.info("derivatives_notional or total_assets not available.")

