import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.linear_model import LinearRegression

# ─────────────────────────────────────────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Alpha Percentile vs BTC", layout="wide")

st.markdown("""
<style>
    /* Tighten metric cards */
    [data-testid="metric-container"] { background:#1e1e2e; border-radius:8px; padding:12px; }
    /* Make expanders cleaner */
    details summary { font-weight:600; }
    /* Diagnostic badge */
    .diag-ok   { color:#4ade80; font-weight:700; }
    .diag-warn { color:#facc15; font-weight:700; }
    .diag-fail { color:#f87171; font-weight:700; }
</style>
""", unsafe_allow_html=True)

st.title("📐 Alpha Percentile vs BTC")
st.caption("Rolling OLS alpha (idiosyncratic) with quality filters, percentile signals, and persistence scoring.")

if st.button("🔄 Refresh data"):
    st.cache_data.clear()
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# INPUT DATA
# ─────────────────────────────────────────────────────────────────────────────
df_price_alpha = st.session_state.get("price_theme", None)

if df_price_alpha is None:
    st.error("`price_theme` not found in session_state. Please load prices into `st.session_state['price_theme']` first.")
    st.stop()

df_price_alpha = df_price_alpha.sort_index().copy()

if "BTC" not in df_price_alpha.columns:
    st.error("BTC column not found in `price_theme`.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — ALL PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Parameters")

    st.subheader("Rolling Window")
    window = st.select_slider(
        "Alpha estimation window (days)",
        options=[30, 60, 90, 180],
        value=90,
        help="Number of days used in each rolling OLS regression. Shorter = faster but noisier."
    )

    st.subheader("Quality Filters")
    min_r2 = st.slider(
        "Min R² to include regression",
        min_value=0.0, max_value=0.5, value=0.10, step=0.01,
        help="Discard windows where BTC explains less than this fraction of coin variance."
    )
    min_beta_tstat = st.slider(
        "Min |beta t-stat| for significance",
        min_value=0.0, max_value=4.0, value=1.5, step=0.1,
        help="Discard windows where the BTC beta is not statistically meaningful."
    )
    min_obs = st.slider(
        "Min valid observations in window",
        min_value=10, max_value=window, value=max(10, int(window * 0.7)), step=5,
        help="Minimum non-NaN days required inside the rolling window."
    )

    st.subheader("Signal 1 — Percentile Momentum")
    momentum_lookback = st.slider(
        "Momentum lookback (days)",
        min_value=5, max_value=30, value=14, step=1,
        help="Period over which to measure change in cross-sec percentile."
    )
    pct_lo = st.slider("Current percentile — lower bound", 0.0, 1.0, 0.35, 0.01)
    pct_hi = st.slider("Current percentile — upper bound", 0.0, 1.0, 0.75, 0.01)
    min_delta = st.slider(
        "Min percentile rise over lookback",
        min_value=0.0, max_value=0.5, value=0.10, step=0.01
    )
    peak_margin_s1 = st.slider(
        "Max distance from 30-day peak (Signal 1)",
        min_value=0.0, max_value=0.5, value=0.10, step=0.01,
        help="Exclude coins within this many pct-points of their 30-day high percentile."
    )

    st.subheader("Signal 2 — Hist vs X-sec Divergence")
    min_divergence = st.slider(
        "Min divergence (x-sec − hist)",
        min_value=-0.5, max_value=0.5, value=0.10, step=0.01,
        help="Coin's cross-sec pct must exceed its hist pct by at least this amount."
    )
    peak_margin_s2 = st.slider(
        "Max distance from 30-day peak (Signal 2)",
        min_value=0.0, max_value=0.5, value=0.10, step=0.01
    )

    st.subheader("Autocorrelation (Persistence)")
    autocorr_lag = st.slider(
        "Autocorrelation lag (days)",
        min_value=3, max_value=21, value=7, step=1,
        help="Lag in days for rank autocorrelation of alpha percentile."
    )

# ─────────────────────────────────────────────────────────────────────────────
# RETURNS
# ─────────────────────────────────────────────────────────────────────────────
df_ret = df_price_alpha.pct_change().dropna(axis=1, how="all").copy()

if "BTC" not in df_ret.columns:
    st.error("BTC column missing after return calculation.")
    st.stop()

btc_ret = df_ret["BTC"]
all_coins = [c for c in df_ret.columns if c != "BTC"]

# ─────────────────────────────────────────────────────────────────────────────
# ALPHA ESTIMATION — WITH QUALITY GATING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Computing rolling alpha...")
def compute_rolling_alpha(df_ret_json, btc_ret_json, window, min_r2, min_beta_tstat, min_obs_):
    """
    For each coin and each day, run OLS(coin ~ BTC) on the rolling window.
    Returns:
        df_alpha      — raw alpha (intercept), NaN where quality filters fail
        df_beta       — OLS beta
        df_r2         — R²
        df_beta_tstat — t-statistic of beta coefficient
        df_pass_rate  — fraction of windows passing QC per coin
    """
    df_ret_ = pd.read_json(df_ret_json)
    btc_ret_ = pd.read_json(btc_ret_json, typ="series")

    coins = [c for c in df_ret_.columns if c != "BTC"]
    n = len(df_ret_)

    alpha_dict     = {c: [np.nan] * n for c in coins}
    beta_dict      = {c: [np.nan] * n for c in coins}
    r2_dict        = {c: [np.nan] * n for c in coins}
    tstat_dict     = {c: [np.nan] * n for c in coins}

    for i in range(n):
        if i < window - 1:
            continue
        btc_w = btc_ret_.iloc[i - window + 1: i + 1].values
        for coin in coins:
            coin_w = df_ret_[coin].iloc[i - window + 1: i + 1].values
            mask = (~np.isnan(coin_w)) & (~np.isnan(btc_w))
            if mask.sum() < min_obs_:
                continue

            X = btc_w[mask].reshape(-1, 1)
            y = coin_w[mask]
            n_obs = mask.sum()

            model = LinearRegression().fit(X, y)
            y_pred = model.predict(X)
            residuals = y - y_pred

            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            # Beta t-stat
            se2 = ss_res / max(n_obs - 2, 1)
            x_var = np.sum((X.flatten() - X.mean()) ** 2)
            beta_se = np.sqrt(se2 / x_var) if x_var > 0 else np.inf
            beta_val = model.coef_[0]
            beta_tstat = beta_val / beta_se if beta_se > 0 else 0.0

            # Quality gate
            if r2 < min_r2 or abs(beta_tstat) < min_beta_tstat:
                continue  # leave as NaN

            alpha_dict[coin][i]  = model.intercept_ * 100   # annualise feel: ×100
            beta_dict[coin][i]   = beta_val
            r2_dict[coin][i]     = r2
            tstat_dict[coin][i]  = beta_tstat

    idx = df_ret_.index
    df_alpha     = pd.DataFrame(alpha_dict, index=idx).round(4)
    df_beta      = pd.DataFrame(beta_dict,  index=idx).round(4)
    df_r2        = pd.DataFrame(r2_dict,    index=idx).round(4)
    df_tstat     = pd.DataFrame(tstat_dict, index=idx).round(4)

    # Pass rate: fraction of windows that passed QC
    valid_counts = df_alpha.notna().sum()
    total_windows = max(n - window + 1, 1)
    df_pass_rate = (valid_counts / total_windows).round(3)

    return (
        df_alpha.to_json(),
        df_beta.to_json(),
        df_r2.to_json(),
        df_tstat.to_json(),
        df_pass_rate.to_json()
    )


alpha_json, beta_json, r2_json, tstat_json, pass_rate_json = compute_rolling_alpha(
    df_ret.to_json(),
    btc_ret.to_json(),
    window, min_r2, min_beta_tstat, min_obs
)

df_alpha   = pd.read_json(alpha_json)
df_beta    = pd.read_json(beta_json)
df_r2      = pd.read_json(r2_json)
df_tstat   = pd.read_json(tstat_json)
pass_rate  = pd.read_json(pass_rate_json, typ="series")

# ─────────────────────────────────────────────────────────────────────────────
# PERCENTILES
# ─────────────────────────────────────────────────────────────────────────────

# Historical: expanding window rank — NO lookahead bias
df_alpha_hist_pct = df_alpha.apply(lambda x: x.expanding().rank(pct=True))

# Cross-sectional: within-day rank across coins
df_alpha_xsec_pct = df_alpha.rank(axis=1, pct=True)

# ─────────────────────────────────────────────────────────────────────────────
# AUTOCORRELATION — RANK PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────
def rolling_rank_autocorr(series, lag, min_periods=20):
    """Compute expanding-window Spearman autocorrelation of a percentile series."""
    result = []
    for i in range(len(series)):
        if i < lag + min_periods:
            result.append(np.nan)
        else:
            s_now  = series.iloc[lag:i+1].values
            s_lag  = series.iloc[:i+1-lag].values
            if len(s_now) < min_periods:
                result.append(np.nan)
            else:
                rho, _ = stats.spearmanr(s_now, s_lag)
                result.append(round(rho, 3))
    return pd.Series(result, index=series.index)

@st.cache_data(show_spinner="Computing rank autocorrelation...")
def compute_autocorr(xsec_json, lag):
    df_x = pd.read_json(xsec_json)
    autocorr_dict = {}
    for coin in df_x.columns:
        s = df_x[coin].dropna()
        if len(s) < lag + 20:
            autocorr_dict[coin] = np.nan
        else:
            rho_series = rolling_rank_autocorr(s, lag)
            autocorr_dict[coin] = rho_series.iloc[-1]  # latest value
    return pd.Series(autocorr_dict).round(3)

autocorr_series = compute_autocorr(df_alpha_xsec_pct.to_json(), autocorr_lag)

# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTICS SECTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔬 Diagnostics")

n_total_coins  = len(all_coins)
n_valid_today  = df_alpha.iloc[-1].notna().sum()
avg_pass_rate  = pass_rate.mean()
n_low_pass     = (pass_rate < 0.3).sum()

diag_col1, diag_col2, diag_col3, diag_col4 = st.columns(4)

with diag_col1:
    st.metric("Total coins", n_total_coins)
with diag_col2:
    colour = "normal" if n_valid_today > n_total_coins * 0.5 else "inverse"
    st.metric("Coins with valid alpha today", n_valid_today,
              delta=f"{n_valid_today/n_total_coins:.0%} pass rate", delta_color=colour)
with diag_col3:
    st.metric("Avg QC pass rate (all history)", f"{avg_pass_rate:.1%}")
with diag_col4:
    st.metric("Coins with <30% pass rate", n_low_pass,
              delta="⚠️ low signal quality" if n_low_pass > 0 else "✅ all ok",
              delta_color="inverse" if n_low_pass > 0 else "normal")

with st.expander("📋 Per-coin QC pass rate & autocorrelation table"):
    diag_df = pd.DataFrame({
        "Pass Rate (QC)": pass_rate,
        f"Alpha (last, ×100)": df_alpha.iloc[-1],
        "R² (last)": df_r2.iloc[-1],
        f"Beta t-stat (last)": df_tstat.iloc[-1],
        f"Rank Autocorr (lag={autocorr_lag}d)": autocorr_series,
    }).sort_values("Pass Rate (QC)", ascending=False)

    def colour_pass(val):
        if pd.isna(val): return "color: grey"
        if val >= 0.6:   return "color: #4ade80"
        if val >= 0.3:   return "color: #facc15"
        return "color: #f87171"

    def colour_autocorr(val):
        if pd.isna(val): return "color: grey"
        if val >= 0.4:   return "color: #4ade80"
        if val >= 0.2:   return "color: #facc15"
        return "color: #f87171"

    styled = (
        diag_df.style
        .applymap(colour_pass, subset=["Pass Rate (QC)"])
        .applymap(colour_autocorr, subset=[f"Rank Autocorr (lag={autocorr_lag}d)"])
        .format({
            "Pass Rate (QC)": "{:.1%}",
            f"Alpha (last, ×100)": "{:.4f}",
            "R² (last)": "{:.3f}",
            f"Beta t-stat (last)": "{:.2f}",
            f"Rank Autocorr (lag={autocorr_lag}d)": "{:.3f}",
        }, na_rep="—")
    )
    st.dataframe(styled, use_container_width=True, height=350)

with st.expander("📊 QC pass rate distribution"):
    fig_diag = go.Figure()
    fig_diag.add_trace(go.Histogram(
        x=pass_rate.values,
        nbinsx=20,
        marker_color="#6366f1",
        opacity=0.8,
        name="Coins"
    ))
    fig_diag.add_vline(x=0.3, line_dash="dash", line_color="#f87171",
                       annotation_text="30% threshold", annotation_position="top right")
    fig_diag.add_vline(x=0.6, line_dash="dash", line_color="#4ade80",
                       annotation_text="60% threshold", annotation_position="top right")
    fig_diag.update_layout(
        title="Distribution of QC Pass Rates Across Coins",
        xaxis_title="Pass Rate", yaxis_title="# Coins",
        height=300, template="plotly_dark"
    )
    st.plotly_chart(fig_diag, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📡 Signal Lists")

# Shared: last-day snapshots
last_xsec = df_alpha_xsec_pct.iloc[-1].dropna()
last_hist  = df_alpha_hist_pct.iloc[-1].dropna()

# 30-day rolling peak of cross-sec percentile
rolling_peak_30 = df_alpha_xsec_pct.rolling(30).max()
last_peak_30    = rolling_peak_30.iloc[-1].dropna()

# ── Signal 1: Alpha Percentile Momentum ──────────────────────────────────────
xsec_delta = df_alpha_xsec_pct - df_alpha_xsec_pct.shift(momentum_lookback)
last_delta  = xsec_delta.iloc[-1].dropna()

coins_s1 = []
for coin in last_xsec.index:
    cur_pct  = last_xsec.get(coin, np.nan)
    delta    = last_delta.get(coin, np.nan)
    peak     = last_peak_30.get(coin, np.nan)
    autocorr = autocorr_series.get(coin, np.nan)

    if any(pd.isna(v) for v in [cur_pct, delta, peak]):
        continue

    dist_from_peak = peak - cur_pct   # > 0 means currently below peak

    cond_range    = pct_lo <= cur_pct <= pct_hi
    cond_rising   = delta >= min_delta
    cond_not_peak = dist_from_peak >= peak_margin_s1

    coins_s1.append({
        "Coin": coin,
        "X-sec Pct": round(cur_pct, 3),
        f"Δ Pct ({momentum_lookback}d)": round(delta, 3),
        "30d Peak": round(peak, 3),
        "Dist from Peak": round(dist_from_peak, 3),
        f"Rank Autocorr": round(autocorr, 3) if not pd.isna(autocorr) else np.nan,
        "✅ In Range": cond_range,
        "✅ Rising": cond_rising,
        "✅ Not Exhausted": cond_not_peak,
        "Signal": cond_range and cond_rising and cond_not_peak,
    })

df_s1 = pd.DataFrame(coins_s1).sort_values(f"Δ Pct ({momentum_lookback}d)", ascending=False)

# ── Signal 2: Historical vs Cross-sectional Divergence ───────────────────────
divergence  = df_alpha_xsec_pct - df_alpha_hist_pct
last_div    = divergence.iloc[-1].dropna()

coins_s2 = []
for coin in last_xsec.index:
    cur_xsec = last_xsec.get(coin, np.nan)
    cur_hist  = last_hist.get(coin, np.nan)
    div_val   = last_div.get(coin, np.nan)
    peak      = last_peak_30.get(coin, np.nan)
    autocorr  = autocorr_series.get(coin, np.nan)

    if any(pd.isna(v) for v in [cur_xsec, cur_hist, div_val, peak]):
        continue

    dist_from_peak = peak - cur_xsec

    cond_div      = div_val >= min_divergence
    cond_not_peak = dist_from_peak >= peak_margin_s2

    coins_s2.append({
        "Coin": coin,
        "X-sec Pct": round(cur_xsec, 3),
        "Hist Pct": round(cur_hist, 3),
        "Divergence (X−H)": round(div_val, 3),
        "30d Peak": round(peak, 3),
        "Dist from Peak": round(dist_from_peak, 3),
        f"Rank Autocorr": round(autocorr, 3) if not pd.isna(autocorr) else np.nan,
        "✅ Diverging": cond_div,
        "✅ Not Exhausted": cond_not_peak,
        "Signal": cond_div and cond_not_peak,
    })

df_s2 = pd.DataFrame(coins_s2).sort_values("Divergence (X−H)", ascending=False)

# ── Display Signal Tables ─────────────────────────────────────────────────────
def style_signal_table(df, signal_col="Signal"):
    def highlight_signal(row):
        if row.get(signal_col, False):
            return ["background-color: #14532d"] * len(row)
        return [""] * len(row)

    bool_cols = [c for c in df.columns if str(c).startswith("✅")]

    def colour_bool(val):
        if val is True:  return "color: #4ade80; font-weight:700"
        if val is False: return "color: #f87171"
        return ""

    styled = df.style.apply(highlight_signal, axis=1)
    if bool_cols:
        styled = styled.applymap(colour_bool, subset=bool_cols)
    return styled

sig_tab1, sig_tab2 = st.tabs([
    "📈 Signal 1 — Alpha Percentile Momentum",
    "🔀 Signal 2 — Hist vs X-sec Divergence"
])

with sig_tab1:
    st.markdown(f"""
    **Logic:** Coins whose cross-sectional alpha percentile is:
    - Currently between **{pct_lo:.0%} – {pct_hi:.0%}** (not exhausted, not bottom half)
    - Rose by ≥ **{min_delta:.0%}** over the last **{momentum_lookback} days**
    - At least **{peak_margin_s1:.0%}** below their 30-day peak (not near recent top)
    
    Rows highlighted **green** pass all three conditions.
    """)
    n_signal_s1 = df_s1["Signal"].sum()
    st.metric("Coins passing Signal 1", int(n_signal_s1))
    st.dataframe(
        style_signal_table(df_s1),
        use_container_width=True, height=420
    )

with sig_tab2:
    st.markdown(f"""
    **Logic:** Coins that are outperforming their own history but not yet re-rated by the cross-section:
    - Cross-sec percentile exceeds historical percentile by ≥ **{min_divergence:.0%}** (early recovery signal)
    - At least **{peak_margin_s2:.0%}** below their 30-day cross-sec peak (not exhausted)
    
    Rows highlighted **green** pass both conditions.
    """)
    n_signal_s2 = df_s2["Signal"].sum()
    st.metric("Coins passing Signal 2", int(n_signal_s2))
    st.dataframe(
        style_signal_table(df_s2),
        use_container_width=True, height=420
    )

# Overlap between signals
s1_set = set(df_s1[df_s1["Signal"]]["Coin"].tolist())
s2_set = set(df_s2[df_s2["Signal"]]["Coin"].tolist())
overlap = s1_set & s2_set
if overlap:
    st.success(f"🎯 **High-conviction overlap** (pass both signals): {', '.join(sorted(overlap))}")

# ─────────────────────────────────────────────────────────────────────────────
# CHART SECTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader(f"📉 Rolling {window}-Day Alpha & Percentile Charts")

available_coins = list(df_alpha.columns)
default_coin = (
    list(s1_set)[0] if s1_set else
    ("BNB" if "BNB" in available_coins else (available_coins[0] if available_coins else None))
)

coins_to_plot = st.multiselect(
    "Select coin(s) to inspect",
    options=available_coins,
    default=[default_coin] if default_coin else [],
)

if coins_to_plot:
    # ── Chart 1: Raw alpha ────────────────────────────────────────────────────
    fig1 = go.Figure()
    for coin in coins_to_plot:
        if coin in df_alpha.columns:
            s = df_alpha[coin].dropna()
            fig1.add_trace(go.Scatter(
                x=s.index, y=s.values, mode="lines", name=coin,
                hovertemplate=f"<b>{coin}</b><br>Date: %{{x}}<br>Alpha: %{{y:.4f}}<extra></extra>"
            ))
    fig1.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.3)
    fig1.update_layout(
        title=f"Rolling {window}-Day Alpha vs BTC (×100, QC-filtered)",
        xaxis_title="Date", yaxis_title="Alpha (×100)",
        height=380, template="plotly_dark", legend=dict(orientation="h", y=-0.2)
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ── Chart 2: Cross-sectional percentile + momentum delta ─────────────────
    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         subplot_titles=["Cross-sec Alpha Percentile", f"Δ Percentile ({momentum_lookback}d)"],
                         vertical_spacing=0.12)
    for coin in coins_to_plot:
        if coin in df_alpha_xsec_pct.columns:
            s = df_alpha_xsec_pct[coin].dropna()
            fig2.add_trace(go.Scatter(
                x=s.index, y=s.values, mode="lines", name=coin,
                hovertemplate=f"<b>{coin}</b> X-sec pct: %{{y:.3f}}<extra></extra>"
            ), row=1, col=1)
            d = xsec_delta[coin].dropna()
            fig2.add_trace(go.Scatter(
                x=d.index, y=d.values, mode="lines", name=f"{coin} Δ",
                showlegend=False,
                hovertemplate=f"<b>{coin}</b> Δ pct: %{{y:.3f}}<extra></extra>"
            ), row=2, col=1)

    # Shading for valid percentile range (Signal 1)
    fig2.add_hrect(y0=pct_lo, y1=pct_hi, fillcolor="rgba(99,102,241,0.12)",
                   line_width=0, row=1, col=1,
                   annotation_text="S1 target range", annotation_position="top left")
    fig2.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.3, row=2, col=1)
    fig2.add_hline(y=min_delta, line_dash="dash", line_color="#6366f1",
                   opacity=0.6, row=2, col=1,
                   annotation_text=f"min Δ={min_delta:.2f}", annotation_position="top right")
    fig2.update_layout(height=550, template="plotly_dark",
                       legend=dict(orientation="h", y=-0.12))
    st.plotly_chart(fig2, use_container_width=True)

    # ── Chart 3: Historical percentile vs cross-sectional (divergence) ────────
    fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         subplot_titles=["Hist vs X-sec Percentile", "Divergence (X-sec − Hist)"],
                         vertical_spacing=0.12)
    for coin in coins_to_plot:
        if coin in df_alpha_hist_pct.columns:
            h = df_alpha_hist_pct[coin].dropna()
            x = df_alpha_xsec_pct[coin].dropna()
            fig3.add_trace(go.Scatter(
                x=h.index, y=h.values, mode="lines", name=f"{coin} Hist",
                line=dict(dash="dot"),
                hovertemplate=f"<b>{coin}</b> Hist pct: %{{y:.3f}}<extra></extra>"
            ), row=1, col=1)
            fig3.add_trace(go.Scatter(
                x=x.index, y=x.values, mode="lines", name=f"{coin} X-sec",
                hovertemplate=f"<b>{coin}</b> X-sec pct: %{{y:.3f}}<extra></extra>"
            ), row=1, col=1)
            div = (divergence[coin]).dropna()
            fig3.add_trace(go.Scatter(
                x=div.index, y=div.values, mode="lines",
                name=f"{coin} Divergence", showlegend=False,
                fill="tozeroy",
                fillcolor="rgba(99,102,241,0.15)",
                hovertemplate=f"<b>{coin}</b> Div: %{{y:.3f}}<extra></extra>"
            ), row=2, col=1)

    fig3.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.3, row=2, col=1)
    fig3.add_hline(y=min_divergence, line_dash="dash", line_color="#6366f1",
                   opacity=0.6, row=2, col=1,
                   annotation_text=f"min div={min_divergence:.2f}", annotation_position="top right")
    fig3.update_layout(height=550, template="plotly_dark",
                       legend=dict(orientation="h", y=-0.12))
    st.plotly_chart(fig3, use_container_width=True)

    # ── Chart 4: R² and beta t-stat over time ─────────────────────────────────
    with st.expander("🔍 Regression quality over time (R² & beta t-stat)"):
        fig4 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             subplot_titles=["R² (rolling window)", "Beta t-stat (rolling window)"],
                             vertical_spacing=0.12)
        for coin in coins_to_plot:
            if coin in df_r2.columns:
                r2s = df_r2[coin].dropna()
                fig4.add_trace(go.Scatter(
                    x=r2s.index, y=r2s.values, mode="lines", name=f"{coin} R²"
                ), row=1, col=1)
                ts = df_tstat[coin].dropna()
                fig4.add_trace(go.Scatter(
                    x=ts.index, y=ts.values, mode="lines", name=f"{coin} t-stat", showlegend=False
                ), row=2, col=1)

        fig4.add_hline(y=min_r2, line_dash="dash", line_color="#f87171",
                       row=1, col=1, annotation_text=f"min R²={min_r2}", annotation_position="top right")
        fig4.add_hline(y=min_beta_tstat, line_dash="dash", line_color="#f87171",
                       row=2, col=1, annotation_text=f"min |t|={min_beta_tstat}", annotation_position="top right")
        fig4.update_layout(height=480, template="plotly_dark",
                           legend=dict(orientation="h", y=-0.12))
        st.plotly_chart(fig4, use_container_width=True)

else:
    st.info("Select at least one coin above to view charts.")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"Window: **{window}d** | Min R²: **{min_r2}** | Min |β t-stat|: **{min_beta_tstat}** | "
    f"Min obs: **{min_obs}** | Autocorr lag: **{autocorr_lag}d** | "
    f"Hist pct uses expanding window (no lookahead)."
)
