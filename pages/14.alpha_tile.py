"""
Alpha Percentile vs BTC — v5 (Simplified Two-Stage)
=====================================================

Simplifications vs v4:
  • No JSON serialisation round-trip in cached functions (fixes FileNotFoundError).
    DataFrames are passed directly; @st.cache_data hashes them by value.
  • QC reduced to a single min_obs gate (R² and t-stat gates removed — they are
    highly correlated with obs count and add little independent information at
    typical crypto universe sizes).
  • Stage 2 scoring reduced from 5 components → 3:
        S1  Historical pct   — expanding rank vs own alpha history   (no lookahead)
        S2  Cross-sec pct    — rank vs peers today
        S3  Momentum         — change in cross-sec pct over N days
  • Component weights dropped; composite = equal-weighted average of S1+S2+S3.
  • Autocorrelation diagnostic retained but moved to QC expander.
  • Total sidebar controls: 5  (was 14+).

Investment logic preserved:
  • Rolling OLS alpha vs BTC with expanding-window historical percentile.
  • Stage 1 OR-gate: hist_pct ≥ threshold OR xsec_pct ≥ threshold.
  • Stage 2 three-component composite, normalised within Stage 1 survivors.
"""

from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

# ─────────────────────────────────────────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Alpha Percentile v5", layout="wide")

st.markdown("""
<style>
[data-testid="metric-container"] {
    background: #0f0f1a;
    border: 1px solid #2a2a40;
    border-radius: 10px;
    padding: 14px 18px;
}
thead tr th { background: #12121f !important; }
section[data-testid="stSidebar"] h3 {
    color: #a78bfa;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 1.2rem;
}
</style>
""", unsafe_allow_html=True)

st.title("📐 Alpha Percentile vs BTC — v5")
st.caption(
    "Stage 1: keep coins above-median on historical OR cross-sectional alpha pct. "
    "Stage 2: rank survivors on 3-component equal-weighted composite."
)

if st.button("🔄 Refresh data"):
    st.cache_data.clear()
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# INPUT DATA
# ─────────────────────────────────────────────────────────────────────────────
df_price = st.session_state.get("price_theme", None)

if df_price is None:
    st.error("`price_theme` not found in session_state. Please load prices first.")
    st.stop()

df_price = df_price.sort_index().copy()

if "BTC" not in df_price.columns:
    st.error("BTC column not found in `price_theme`.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — PARAMETERS  (5 controls total)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Parameters")

    st.subheader("Rolling Alpha Window")
    window = st.select_slider(
        "Estimation window (days)",
        options=[30, 60, 90, 180],
        value=90,
        help="Days used in each rolling OLS regression vs BTC."
    )

    st.subheader("Quality Gate")
    min_obs = st.slider(
        "Min valid observations in window",
        10, window, max(10, int(window * 0.7)), 5,
        help="Minimum non-NaN days required inside the window for the alpha to be recorded."
    )

    st.subheader("Stage 1 — Pre-filter")
    stage1_threshold = st.slider(
        "Percentile threshold (hist OR x-sec ≥)",
        0.0, 1.0, 0.50, 0.05,
        help=(
            "A coin survives Stage 1 if its historical pct OR cross-sec pct "
            "is at or above this value today. 0.50 = above median on either dimension."
        )
    )

    st.subheader("S3 — Momentum Lookback")
    momentum_lookback = st.slider(
        "Δ X-sec pct lookback (days)", 5, 30, 14, 1,
        help="Days over which we measure the change in cross-sectional percentile."
    )

    st.subheader("Display Filter")
    min_score_pct = st.slider(
        "Show coins with composite ≥ (% of max)", 0, 100, 40, 5,
        help="Within Stage 1 survivors, show only coins above this composite score."
    )
    # Max composite = 1.0 (equal-weight average of three 0-1 normalised scores)
    min_score_abs = min_score_pct / 100.0

# ─────────────────────────────────────────────────────────────────────────────
# RETURNS
# ─────────────────────────────────────────────────────────────────────────────
df_ret    = df_price.pct_change().dropna(axis=1, how="all").copy()
btc_ret   = df_ret["BTC"]
all_coins = [c for c in df_ret.columns if c != "BTC"]

# ─────────────────────────────────────────────────────────────────────────────
# ROLLING OLS ALPHA
# NOTE: DataFrames passed directly — no JSON serialisation, no FileNotFoundError.
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Computing rolling OLS alpha…")
def compute_rolling_alpha(
    df_ret: pd.DataFrame,
    btc_ret: pd.Series,
    window: int,
    min_obs_: int,
) -> pd.DataFrame:
    """
    Returns df_alpha: DataFrame[coin → daily alpha×100], NaN where QC fails.
    QC gate: non-NaN observations in window ≥ min_obs_.
    """
    coins = [c for c in df_ret.columns if c != "BTC"]
    n     = len(df_ret)
    alpha_d = {c: [np.nan] * n for c in coins}

    btc_arr = btc_ret.values

    for i in range(window - 1, n):
        btc_w = btc_arr[i - window + 1 : i + 1]
        for coin in coins:
            coin_w = df_ret[coin].iloc[i - window + 1 : i + 1].values
            mask   = (~np.isnan(coin_w)) & (~np.isnan(btc_w))
            if mask.sum() < min_obs_:
                continue
            X = btc_w[mask].reshape(-1, 1)
            y = coin_w[mask]
            model = LinearRegression().fit(X, y)
            alpha_d[coin][i] = model.intercept_ * 100.0

    return pd.DataFrame(alpha_d, index=df_ret.index).round(4)


df_alpha = compute_rolling_alpha(df_ret, btc_ret, window, min_obs)

# ─────────────────────────────────────────────────────────────────────────────
# FULL-UNIVERSE PERCENTILES  (computed before any filter to avoid lookahead)
# ─────────────────────────────────────────────────────────────────────────────
# S1: expanding-window rank vs own history — zero lookahead
df_hist_pct = df_alpha.apply(lambda x: x.expanding().rank(pct=True))

# S2: cross-sectional rank across ALL coins each day
df_xsec_pct = df_alpha.rank(axis=1, pct=True)

# S3: momentum — change in cross-sec pct
df_momentum = df_xsec_pct - df_xsec_pct.shift(momentum_lookback)

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — PRE-FILTER  (evaluated on latest day with valid data)
# ─────────────────────────────────────────────────────────────────────────────
last_hist = df_hist_pct.iloc[-1]
last_xsec = df_xsec_pct.iloc[-1]

stage1_pass = [
    c for c in all_coins
    if not pd.isna(last_hist.get(c, np.nan))
    and not pd.isna(last_xsec.get(c, np.nan))
    and (
        last_hist.get(c, 0) >= stage1_threshold
        or last_xsec.get(c, 0) >= stage1_threshold
    )
]

n_valid_today = int((~last_hist.isna()).sum())
n_stage1      = len(stage1_pass)
n_dropped     = n_valid_today - n_stage1

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — NORMALISE WITHIN STAGE 1 SURVIVORS & SCORE
# Composite = equal-weighted average of three 0-1 normalised scores.
# Normalisation is cross-sectional within Stage 1 universe only.
# ─────────────────────────────────────────────────────────────────────────────
def cs_rank_universe(df: pd.DataFrame, universe: list) -> pd.DataFrame:
    cols = [c for c in universe if c in df.columns]
    return df[cols].rank(axis=1, pct=True)


if stage1_pass:
    n1 = cs_rank_universe(df_hist_pct, stage1_pass)
    n2 = cs_rank_universe(df_xsec_pct, stage1_pass)
    n3 = cs_rank_universe(df_momentum,  stage1_pass)
    df_composite = (n1 + n2 + n3) / 3.0          # 0–1 scale
else:
    n1 = n2 = n3 = df_alpha[[]].copy()
    df_composite  = df_alpha[[]].copy()

# ─────────────────────────────────────────────────────────────────────────────
# TODAY'S SIGNAL TABLE
# ─────────────────────────────────────────────────────────────────────────────
last_s1   = df_hist_pct.iloc[-1]
last_s2   = df_xsec_pct.iloc[-1]
last_s3   = df_momentum.iloc[-1]
last_n1   = n1.iloc[-1] if not n1.empty else pd.Series(dtype=float)
last_n2   = n2.iloc[-1] if not n2.empty else pd.Series(dtype=float)
last_n3   = n3.iloc[-1] if not n3.empty else pd.Series(dtype=float)
last_comp = df_composite.iloc[-1] if not df_composite.empty else pd.Series(dtype=float)

rows = []
for coin in stage1_pass:
    comp = last_comp.get(coin, np.nan)
    if pd.isna(comp):
        continue
    hist_ok = last_s1.get(coin, np.nan) >= stage1_threshold
    xsec_ok = last_s2.get(coin, np.nan) >= stage1_threshold
    gate_str = (
        "Hist + X-sec" if (hist_ok and xsec_ok)
        else ("Hist only" if hist_ok else "X-sec only")
    )
    rows.append({
        "Coin":                             coin,
        "Stage-1 gate":                     gate_str,
        "S1 Hist Pct":                      round(last_s1.get(coin, np.nan), 3),
        "S2 X-sec Pct":                     round(last_s2.get(coin, np.nan), 3),
        f"S3 Δ Pct ({momentum_lookback}d)": round(last_s3.get(coin, np.nan), 3),
        "N1 (hist)":                        round(last_n1.get(coin, np.nan), 3),
        "N2 (x-sec)":                       round(last_n2.get(coin, np.nan), 3),
        "N3 (mom)":                         round(last_n3.get(coin, np.nan), 3),
        "Composite":                        round(comp, 3),
    })

df_signal = (
    pd.DataFrame(rows)
    .sort_values("Composite", ascending=False)
    .reset_index(drop=True)
)
df_signal_filtered = df_signal[df_signal["Composite"] >= min_score_abs].copy()

# ─────────────────────────────────────────────────────────────────────────────
# TOP BANNER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")

n_total_coins = len(all_coins)
n_above_score = len(df_signal_filtered)
top_coin      = df_signal.iloc[0]["Coin"]      if len(df_signal) > 0 else "—"
top_score_val = df_signal.iloc[0]["Composite"] if len(df_signal) > 0 else np.nan
qc_pass_rate  = df_alpha.iloc[-1].notna().sum() / max(n_total_coins, 1)

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Total coins", n_total_coins)
with c2:
    st.metric("Valid alpha today", n_valid_today,
              delta=f"{qc_pass_rate:.0%} pass QC",
              delta_color="normal" if qc_pass_rate > 0.5 else "inverse")
with c3:
    st.metric("Pass Stage 1", n_stage1,
              delta=f"−{n_dropped} dropped", delta_color="off")
with c4:
    st.metric("Above score threshold", n_above_score)
with c5:
    st.metric(
        "Top coin today", top_coin,
        delta=f"composite {top_score_val:.2f}" if not pd.isna(top_score_val) else "—"
    )

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🚦 Stage 1 — Pre-filter Detail")

col_tbl, col_charts = st.columns([1.3, 1])

with col_tbl:
    stage1_rows = []
    for coin in [c for c in all_coins if not pd.isna(last_hist.get(c, np.nan))]:
        h = last_hist.get(coin, np.nan)
        x = last_xsec.get(coin, np.nan)
        hist_ok = h >= stage1_threshold
        xsec_ok = x >= stage1_threshold
        passed  = hist_ok or xsec_ok
        gate    = (
            "Hist + X-sec" if (hist_ok and xsec_ok)
            else ("Hist only" if hist_ok else ("X-sec only" if xsec_ok else "—"))
        )
        stage1_rows.append({
            "Coin": coin,
            "Hist Pct": round(h, 3),
            "X-sec Pct": round(x, 3),
            "Passes": passed,
            "Gate": gate,
        })

    df_s1_detail = (
        pd.DataFrame(stage1_rows)
        .sort_values(["Passes", "Hist Pct"], ascending=[False, False])
        .reset_index(drop=True)
    )

    def _c_pass(val):
        if val is True:  return "color:#4ade80;font-weight:700"
        if val is False: return "color:#f87171"
        return ""

    def _c_pct(val):
        if pd.isna(val): return "color:grey"
        return "color:#4ade80" if val >= stage1_threshold else "color:#f87171"

    st.dataframe(
        df_s1_detail.style
            .applymap(_c_pass, subset=["Passes"])
            .applymap(_c_pct,  subset=["Hist Pct", "X-sec Pct"])
            .format({"Hist Pct": "{:.3f}", "X-sec Pct": "{:.3f}"}, na_rep="—"),
        use_container_width=True, height=360
    )

with col_charts:
    # Bar: how many coins passed via each route
    n_both      = sum(1 for c in stage1_pass if
                      last_hist.get(c, 0) >= stage1_threshold and
                      last_xsec.get(c, 0) >= stage1_threshold)
    n_hist_only = sum(1 for c in stage1_pass if
                      last_hist.get(c, 0) >= stage1_threshold and
                      last_xsec.get(c, 0) < stage1_threshold)
    n_xsec_only = sum(1 for c in stage1_pass if
                      last_hist.get(c, 0) < stage1_threshold and
                      last_xsec.get(c, 0) >= stage1_threshold)

    fig_bar = go.Figure(go.Bar(
        x=["Hist only", "X-sec only", "Both"],
        y=[n_hist_only, n_xsec_only, n_both],
        marker_color=["#60a5fa", "#a78bfa", "#4ade80"],
        text=[n_hist_only, n_xsec_only, n_both],
        textposition="outside"
    ))
    fig_bar.update_layout(
        title=f"Stage 1 pass breakdown (threshold={stage1_threshold:.2f})",
        yaxis_title="# Coins", height=280,
        template="plotly_dark", showlegend=False
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Scatter: Hist pct vs X-sec pct
    pass_set = set(stage1_pass)
    colors_sc = [
        "#4ade80" if c in pass_set else "#f87171"
        for c in df_s1_detail["Coin"]
    ]
    fig_sc = go.Figure(go.Scatter(
        x=df_s1_detail["Hist Pct"], y=df_s1_detail["X-sec Pct"],
        mode="markers+text",
        text=df_s1_detail["Coin"], textposition="top center",
        textfont=dict(size=8),
        marker=dict(color=colors_sc, size=7, opacity=0.8),
        hovertemplate="<b>%{text}</b><br>Hist: %{x:.3f}<br>X-sec: %{y:.3f}<extra></extra>"
    ))
    fig_sc.add_hline(y=stage1_threshold, line_dash="dash", line_color="#facc15",
                     opacity=0.6, annotation_text=f"x-sec thr",
                     annotation_position="right")
    fig_sc.add_vline(x=stage1_threshold, line_dash="dash", line_color="#facc15",
                     opacity=0.6, annotation_text=f"hist thr",
                     annotation_position="top")
    fig_sc.update_layout(
        title="Hist Pct vs X-sec Pct (green = pass Stage 1)",
        xaxis_title="Hist Pct", yaxis_title="X-sec Pct",
        height=320, template="plotly_dark"
    )
    st.plotly_chart(fig_sc, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — COMPOSITE SCORE TABLE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Stage 2 — Composite Score Table")

st.markdown(f"""
| # | Component | Raw metric | What it captures |
|---|---|---|---|
| S1 | **Hist Pct** | Expanding-window rank vs own alpha history | Alpha strong relative to own past |
| S2 | **X-sec Pct** | Rank vs all peers today | Leading the field |
| S3 | **Momentum** | Δ x-sec pct over {momentum_lookback}d | Alpha rank rising |

*N1–N3 are cross-sec ranked within Stage 1 survivors (0–1). Composite = equal-weighted average.*  
*Showing coins with composite ≥ **{min_score_abs:.2f}** ({min_score_pct}% of max).*
""")

if not df_signal_filtered.empty:
    top5 = df_signal_filtered.head(5)["Coin"].tolist()
    st.success(f"🏆 Top coins: **{' · '.join(top5)}**")

    hist_only_top = df_signal_filtered[df_signal_filtered["Stage-1 gate"] == "Hist only"]["Coin"].tolist()
    xsec_only_top = df_signal_filtered[df_signal_filtered["Stage-1 gate"] == "X-sec only"]["Coin"].tolist()
    if hist_only_top:
        st.info(f"📌 Passed via **Hist only** (improving vs own history, not yet leading peers): {', '.join(hist_only_top)}")
    if xsec_only_top:
        st.info(f"📌 Passed via **X-sec only** (leading peers today, not historically elevated): {', '.join(xsec_only_top)}")
else:
    st.warning("No coins above the composite threshold. Lower the threshold or Stage-1 filter.")

norm_cols = ["N1 (hist)", "N2 (x-sec)", "N3 (mom)"]

def _c_norm(v):
    if pd.isna(v): return "color:grey"
    return "color:#4ade80" if v >= 0.7 else ("color:#facc15" if v >= 0.4 else "color:#f87171")

def _c_comp(v):
    if pd.isna(v): return ""
    return (
        "color:#4ade80;font-weight:700" if v >= 0.70 else
        ("color:#facc15;font-weight:600" if v >= 0.45 else "color:#f87171")
    )

def _c_gate(v):
    if v == "Hist + X-sec": return "color:#4ade80"
    if v in ("Hist only", "X-sec only"): return "color:#facc15"
    return ""

def highlight_row(row):
    if row["Composite"] >= 0.70: return ["background-color:#14532d"] * len(row)
    if row["Composite"] >= 0.50: return ["background-color:#1e3a5f"] * len(row)
    return [""] * len(row)

fmt = {
    "S1 Hist Pct": "{:.3f}", "S2 X-sec Pct": "{:.3f}",
    f"S3 Δ Pct ({momentum_lookback}d)": "{:.3f}",
    "N1 (hist)": "{:.3f}", "N2 (x-sec)": "{:.3f}", "N3 (mom)": "{:.3f}",
    "Composite": "{:.3f}",
}

tab_full, tab_raw, tab_norm_tab = st.tabs([
    "🎯 Full table", "📋 Raw components", "🔢 Normalised scores"
])

with tab_full:
    st.dataframe(
        df_signal_filtered.style
            .apply(highlight_row, axis=1)
            .applymap(_c_norm, subset=norm_cols)
            .applymap(_c_comp, subset=["Composite"])
            .applymap(_c_gate, subset=["Stage-1 gate"])
            .format(fmt, na_rep="—"),
        use_container_width=True, height=500
    )

with tab_raw:
    raw_cols = ["Coin", "Stage-1 gate", "S1 Hist Pct", "S2 X-sec Pct",
                f"S3 Δ Pct ({momentum_lookback}d)", "Composite"]
    st.dataframe(
        df_signal_filtered[raw_cols].style
            .apply(highlight_row, axis=1)
            .applymap(_c_comp, subset=["Composite"])
            .applymap(_c_gate, subset=["Stage-1 gate"])
            .format({c: "{:.3f}" for c in raw_cols if c not in ("Coin", "Stage-1 gate")},
                    na_rep="—"),
        use_container_width=True, height=500
    )

with tab_norm_tab:
    n_cols = ["Coin", "Stage-1 gate", "N1 (hist)", "N2 (x-sec)", "N3 (mom)", "Composite"]
    st.dataframe(
        df_signal_filtered[n_cols].style
            .apply(highlight_row, axis=1)
            .applymap(_c_norm, subset=norm_cols)
            .applymap(_c_comp, subset=["Composite"])
            .applymap(_c_gate, subset=["Stage-1 gate"])
            .format({c: "{:.3f}" for c in n_cols if c not in ("Coin", "Stage-1 gate")},
                    na_rep="—"),
        use_container_width=True, height=500
    )

# ─────────────────────────────────────────────────────────────────────────────
# SCORE DISTRIBUTION + RADAR
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📈 Score Distribution & Component Profile")

col_dist, col_radar = st.columns([1.6, 1])

with col_dist:
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=df_signal["Composite"].dropna().values,
        nbinsx=20, marker_color="#6366f1", opacity=0.55,
        name="Stage 1 universe"
    ))
    if not df_signal_filtered.empty:
        fig_dist.add_trace(go.Histogram(
            x=df_signal_filtered["Composite"].dropna().values,
            nbinsx=20, marker_color="#4ade80", opacity=0.75,
            name=f"Above {min_score_pct}% threshold"
        ))
    fig_dist.add_vline(
        x=min_score_abs, line_dash="dash", line_color="#facc15",
        annotation_text=f"threshold={min_score_abs:.2f}",
        annotation_position="top right"
    )
    fig_dist.update_layout(
        title="Composite Score Distribution — Stage 1 Universe",
        xaxis_title="Composite (0–1)", yaxis_title="# Coins",
        barmode="overlay", height=300, template="plotly_dark",
        legend=dict(orientation="h", y=-0.35)
    )
    st.plotly_chart(fig_dist, use_container_width=True)

with col_radar:
    top3 = df_signal_filtered.head(3)
    if not top3.empty:
        cats = ["Hist Pct", "X-sec Pct", "Momentum"]
        fig_radar = go.Figure()
        for _, row in top3.iterrows():
            vals = [row["N1 (hist)"], row["N2 (x-sec)"], row["N3 (mom)"]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=cats + [cats[0]],
                fill="toself", name=row["Coin"], opacity=0.7
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Top 3 — component profile",
            height=300, template="plotly_dark",
            legend=dict(orientation="h", y=-0.25)
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.info("No coins above threshold.")

# ─────────────────────────────────────────────────────────────────────────────
# COIN INSPECTOR
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔍 Coin Inspector")

available_coins = list(df_alpha.columns)
default_coins = (
    df_signal_filtered.head(2)["Coin"].tolist()
    or (["BNB"] if "BNB" in available_coins else available_coins[:1])
)

coins_to_plot = st.multiselect(
    "Select coin(s) to inspect",
    options=available_coins,
    default=default_coins
)

if coins_to_plot:
    # Chart 1: Raw rolling alpha
    fig_a = go.Figure()
    for coin in coins_to_plot:
        if coin in df_alpha.columns:
            s = df_alpha[coin].dropna()
            fig_a.add_trace(go.Scatter(
                x=s.index, y=s.values, mode="lines", name=coin,
                hovertemplate=f"<b>{coin}</b>: %{{y:.4f}}<extra></extra>"
            ))
    fig_a.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.3)
    fig_a.update_layout(
        title=f"Rolling {window}-Day Alpha vs BTC (×100)",
        xaxis_title="Date", yaxis_title="Alpha (×100)",
        height=280, template="plotly_dark",
        legend=dict(orientation="h", y=-0.30)
    )
    st.plotly_chart(fig_a, use_container_width=True)

    # Chart 2: Three components over time
    fig_comp = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=[
            "S1 — Historical pct (expanding, no lookahead)",
            "S2 — Cross-sec pct (full universe)",
            f"S3 — Momentum Δ x-sec pct ({momentum_lookback}d)",
        ],
        vertical_spacing=0.07
    )
    series_map = [
        (df_hist_pct,  1, "#60a5fa"),
        (df_xsec_pct,  2, "#a78bfa"),
        (df_momentum,  3, "#34d399"),
    ]
    for df_, row_, colour_ in series_map:
        for coin in coins_to_plot:
            if coin not in df_.columns:
                continue
            s = df_[coin].dropna()
            fig_comp.add_trace(go.Scatter(
                x=s.index, y=s.values, mode="lines",
                name=coin, line=dict(color=colour_, width=1.5),
                showlegend=(row_ == 1),
                hovertemplate=f"<b>{coin}</b>: %{{y:.3f}}<extra></extra>"
            ), row=row_, col=1)

    for r_ in [1, 2]:
        fig_comp.add_hline(
            y=stage1_threshold, line_dash="dash", line_color="#facc15",
            opacity=0.5, annotation_text=f"Stage-1 thr={stage1_threshold:.2f}",
            annotation_position="top right", row=r_, col=1
        )
        fig_comp.add_hline(y=0.5, line_dash="dot", line_color="white",
                           opacity=0.15, row=r_, col=1)
    fig_comp.add_hline(y=0, line_dash="dot", line_color="white",
                       opacity=0.15, row=3, col=1)

    fig_comp.update_layout(
        height=600, template="plotly_dark",
        legend=dict(orientation="h", y=-0.05),
        title_text="Three Components Over Time"
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # Chart 3: Composite over time
    fig_cs = go.Figure()
    for coin in coins_to_plot:
        if not df_composite.empty and coin in df_composite.columns:
            s = df_composite[coin].dropna()
            fig_cs.add_trace(go.Scatter(
                x=s.index, y=s.values, mode="lines", name=coin,
                hovertemplate=f"<b>{coin}</b> composite: %{{y:.3f}}<extra></extra>"
            ))
    fig_cs.add_hline(
        y=min_score_abs, line_dash="dash", line_color="#facc15", opacity=0.7,
        annotation_text=f"threshold={min_score_abs:.2f}",
        annotation_position="top right"
    )
    fig_cs.update_layout(
        title="Rolling Composite Score Over Time (Stage 1 universe)",
        xaxis_title="Date", yaxis_title="Composite (0–1)",
        height=260, template="plotly_dark",
        legend=dict(orientation="h", y=-0.30)
    )
    st.plotly_chart(fig_cs, use_container_width=True)

else:
    st.info("Select at least one coin above to view charts.")

# ─────────────────────────────────────────────────────────────────────────────
# QC DIAGNOSTICS (collapsed by default)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("🔬 QC Diagnostics", expanded=False):

    # Autocorrelation of cross-sec pct (diagnostic only, no sidebar param)
    AUTOCORR_LAG = 7
    autocorr_vals = {}
    for coin in df_xsec_pct.columns:
        s = df_xsec_pct[coin].dropna()
        if len(s) >= AUTOCORR_LAG + 20:
            from scipy import stats as scipy_stats
            rho, _ = scipy_stats.spearmanr(
                s.iloc[AUTOCORR_LAG:].values,
                s.iloc[:-AUTOCORR_LAG].values
            )
            autocorr_vals[coin] = round(rho, 3)
        else:
            autocorr_vals[coin] = np.nan
    autocorr_series = pd.Series(autocorr_vals)

    # Per-window QC pass rate
    total_windows = max(len(df_alpha) - window + 1, 1)
    pass_rate = (df_alpha.notna().sum() / total_windows).round(3)

    diag_df = pd.DataFrame({
        "QC Pass Rate":          pass_rate,
        "Alpha today (×100)":    df_alpha.iloc[-1],
        f"Autocorr (lag={AUTOCORR_LAG}d)": autocorr_series,
    }).sort_values("QC Pass Rate", ascending=False)

    def _c_pr(v):
        if pd.isna(v): return "color:grey"
        return "color:#4ade80" if v >= 0.6 else ("color:#facc15" if v >= 0.3 else "color:#f87171")

    def _c_ac(v):
        if pd.isna(v): return "color:grey"
        return "color:#4ade80" if v >= 0.4 else ("color:#facc15" if v >= 0.2 else "color:#f87171")

    st.dataframe(
        diag_df.style
            .applymap(_c_pr, subset=["QC Pass Rate"])
            .applymap(_c_ac, subset=[f"Autocorr (lag={AUTOCORR_LAG}d)"])
            .format({
                "QC Pass Rate":        "{:.1%}",
                "Alpha today (×100)":  "{:.4f}",
                f"Autocorr (lag={AUTOCORR_LAG}d)": "{:.3f}",
            }, na_rep="—"),
        use_container_width=True, height=300
    )

    col_qa, col_qb = st.columns(2)
    with col_qa:
        fig_pr = go.Figure(go.Histogram(
            x=pass_rate.values, nbinsx=20,
            marker_color="#6366f1", opacity=0.85
        ))
        fig_pr.add_vline(x=0.3, line_dash="dash", line_color="#f87171",
                         annotation_text="30%", annotation_position="top right")
        fig_pr.add_vline(x=0.6, line_dash="dash", line_color="#4ade80",
                         annotation_text="60%", annotation_position="top right")
        fig_pr.update_layout(title="QC Pass Rate Distribution",
                             xaxis_title="Pass Rate", yaxis_title="# Coins",
                             height=240, template="plotly_dark")
        st.plotly_chart(fig_pr, use_container_width=True)

    with col_qb:
        fig_ac = go.Figure(go.Histogram(
            x=autocorr_series.dropna().values, nbinsx=20,
            marker_color="#a78bfa", opacity=0.85
        ))
        fig_ac.add_vline(x=0, line_dash="dot", line_color="white", opacity=0.4)
        fig_ac.update_layout(title=f"X-sec Pct Autocorr Distribution (lag={AUTOCORR_LAG}d)",
                             xaxis_title="Spearman rho", yaxis_title="# Coins",
                             height=240, template="plotly_dark")
        st.plotly_chart(fig_ac, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"Window: **{window}d** | Min obs: **{min_obs}** | "
    f"Stage-1 threshold: **{stage1_threshold:.2f}** | "
    f"Momentum lookback: **{momentum_lookback}d** | "
    f"Composite: equal-weighted avg of S1+S2+S3 (0–1) | "
    f"Hist pct: expanding window (no lookahead) | "
    f"Normalisation: x-sec rank within Stage-1 survivors"
)
