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
    [data-testid="metric-container"] { background:#1e1e2e; border-radius:8px; padding:12px; }
    details summary { font-weight:600; }
</style>
""", unsafe_allow_html=True)

st.title("📐 Alpha Percentile vs BTC")
st.caption(
    "Combined signal: percentile momentum (L1) + confirmed divergence (L2) + price gate (L3). "
    "All three layers must pass for a coin to be flagged."
)

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

    # ── OLS window ──────────────────────────────────────────────────────────
    st.subheader("Rolling Window")
    window = st.select_slider(
        "Alpha estimation window (days)",
        options=[30, 60, 90, 180], value=90,
        help="Days used in each rolling OLS regression."
    )

    # ── QC filters ──────────────────────────────────────────────────────────
    st.subheader("Quality Filters")
    min_r2 = st.slider("Min R²", 0.0, 0.5, 0.10, 0.01,
        help="Discard windows where BTC explains less than this fraction of coin variance.")
    min_beta_tstat = st.slider("Min |beta t-stat|", 0.0, 4.0, 1.5, 0.1,
        help="Discard windows where the BTC beta is not statistically meaningful.")
    min_obs = st.slider(
        "Min valid observations in window",
        10, window, max(10, int(window * 0.7)), 5,
        help="Minimum non-NaN days required inside the rolling window."
    )

    # ── L1: Percentile momentum ──────────────────────────────────────────────
    st.subheader("Layer 1 — Percentile Momentum")
    momentum_lookback = st.slider("Momentum lookback (days)", 5, 30, 14, 1,
        help="Period over which to measure change in cross-sec percentile.")
    pct_lo = st.slider("Current percentile — lower bound", 0.0, 1.0, 0.35, 0.01,
        help="Coin must be above this percentile today.")
    pct_hi = st.slider("Current percentile — upper bound", 0.0, 1.0, 0.75, 0.01,
        help="Coin must be below this percentile today (not exhausted).")
    min_delta = st.slider("Min percentile rise over lookback", 0.0, 0.5, 0.10, 0.01,
        help="Minimum improvement in cross-sec percentile over the momentum lookback.")
    peak_margin = st.slider(
        "Min distance from 30-day peak",
        0.0, 0.5, 0.10, 0.01,
        help="Coin must be at least this far below its 30-day cross-sec high. Shared by L1 and L2."
    )

    # ── L2: Confirmed divergence ─────────────────────────────────────────────
    st.subheader("Layer 2 — Confirmed Divergence")
    min_divergence = st.slider("Min divergence level (x-sec − hist)", -0.5, 0.5, 0.10, 0.01,
        help="Cross-sec percentile must exceed historical percentile by at least this amount.")
    div_widening_lookback = st.slider("Divergence widening lookback (days)", 3, 21, 7, 1,
        help="Divergence must have grown over this many days.")
    min_div_widening = st.slider("Min divergence widening", 0.0, 0.3, 0.05, 0.01,
        help="Minimum increase in divergence over the widening lookback.")
    div_persistence_days = st.slider("Min days divergence held positive", 3, 21, 5, 1,
        help="Divergence must have been above the min level for at least this many consecutive days.")

    # ── L3: Price gate ───────────────────────────────────────────────────────
    st.subheader("Layer 3 — Price Confirmation")
    price_ret_window = st.slider("Price return window (days)", 3, 14, 5, 1,
        help="N-day raw return window for price confirmation.")
    min_price_ret = st.slider("Min price return (%)", -10.0, 20.0, 0.0, 0.5,
        help="Coin must have returned at least this much. Set 0 to just require positive price action.")

    # ── Autocorrelation ──────────────────────────────────────────────────────
    st.subheader("Persistence (Autocorrelation)")
    autocorr_lag = st.slider("Autocorrelation lag (days)", 3, 21, 7, 1,
        help="Lag in days for Spearman rank autocorrelation of alpha percentile.")

# ─────────────────────────────────────────────────────────────────────────────
# RETURNS
# ─────────────────────────────────────────────────────────────────────────────
df_ret = df_price_alpha.pct_change().dropna(axis=1, how="all").copy()

if "BTC" not in df_ret.columns:
    st.error("BTC column missing after return calculation.")
    st.stop()

btc_ret   = df_ret["BTC"]
all_coins = [c for c in df_ret.columns if c != "BTC"]

# ─────────────────────────────────────────────────────────────────────────────
# ALPHA ESTIMATION — WITH QUALITY GATING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Computing rolling alpha (OLS with QC filters)...")
def compute_rolling_alpha(df_ret_json, btc_ret_json, window, min_r2, min_beta_tstat, min_obs_):
    df_ret_  = pd.read_json(df_ret_json)
    btc_ret_ = pd.read_json(btc_ret_json, typ="series")

    coins = [c for c in df_ret_.columns if c != "BTC"]
    n     = len(df_ret_)

    alpha_dict = {c: [np.nan] * n for c in coins}
    beta_dict  = {c: [np.nan] * n for c in coins}
    r2_dict    = {c: [np.nan] * n for c in coins}
    tstat_dict = {c: [np.nan] * n for c in coins}

    for i in range(n):
        if i < window - 1:
            continue
        btc_w = btc_ret_.iloc[i - window + 1: i + 1].values
        for coin in coins:
            coin_w = df_ret_[coin].iloc[i - window + 1: i + 1].values
            mask   = (~np.isnan(coin_w)) & (~np.isnan(btc_w))
            if mask.sum() < min_obs_:
                continue

            X      = btc_w[mask].reshape(-1, 1)
            y      = coin_w[mask]
            n_obs  = mask.sum()

            model     = LinearRegression().fit(X, y)
            y_pred    = model.predict(X)
            residuals = y - y_pred

            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            se2       = ss_res / max(n_obs - 2, 1)
            x_var     = np.sum((X.flatten() - X.mean()) ** 2)
            beta_se   = np.sqrt(se2 / x_var) if x_var > 0 else np.inf
            beta_val  = model.coef_[0]
            beta_tstat = beta_val / beta_se if beta_se > 0 else 0.0

            if r2 < min_r2 or abs(beta_tstat) < min_beta_tstat:
                continue

            alpha_dict[coin][i]  = model.intercept_ * 100
            beta_dict[coin][i]   = beta_val
            r2_dict[coin][i]     = r2
            tstat_dict[coin][i]  = beta_tstat

    idx       = df_ret_.index
    df_alpha  = pd.DataFrame(alpha_dict, index=idx).round(4)
    df_beta   = pd.DataFrame(beta_dict,  index=idx).round(4)
    df_r2     = pd.DataFrame(r2_dict,    index=idx).round(4)
    df_tstat  = pd.DataFrame(tstat_dict, index=idx).round(4)

    total_windows = max(n - window + 1, 1)
    pass_rate     = (df_alpha.notna().sum() / total_windows).round(3)

    return (
        df_alpha.to_json(), df_beta.to_json(),
        df_r2.to_json(),    df_tstat.to_json(),
        pass_rate.to_json()
    )


alpha_json, beta_json, r2_json, tstat_json, pass_rate_json = compute_rolling_alpha(
    df_ret.to_json(), btc_ret.to_json(),
    window, min_r2, min_beta_tstat, min_obs
)

df_alpha  = pd.read_json(alpha_json)
df_beta   = pd.read_json(beta_json)
df_r2     = pd.read_json(r2_json)
df_tstat  = pd.read_json(tstat_json)
pass_rate = pd.read_json(pass_rate_json, typ="series")

# ─────────────────────────────────────────────────────────────────────────────
# PERCENTILES
# ─────────────────────────────────────────────────────────────────────────────

# Expanding window — no lookahead bias
df_alpha_hist_pct = df_alpha.apply(lambda x: x.expanding().rank(pct=True))

# Cross-sectional rank within each day
df_alpha_xsec_pct = df_alpha.rank(axis=1, pct=True)

# ─────────────────────────────────────────────────────────────────────────────
# DERIVED SERIES FOR COMBINED SIGNAL
# ─────────────────────────────────────────────────────────────────────────────

# L1: momentum
xsec_delta = df_alpha_xsec_pct - df_alpha_xsec_pct.shift(momentum_lookback)

# L2: divergence (level, widening over N days, persistence)
divergence   = df_alpha_xsec_pct - df_alpha_hist_pct
div_widening = divergence - divergence.shift(div_widening_lookback)
# Count of days in rolling window where divergence was above threshold
# equals div_persistence_days only when ALL days passed
div_pers_count = (divergence >= min_divergence).rolling(div_persistence_days).sum()

# Shared: 30-day rolling peak of cross-sec percentile
rolling_peak_30 = df_alpha_xsec_pct.rolling(30).max()

# L3: price confirmation
price_ret_Nd = df_price_alpha.pct_change(price_ret_window) * 100   # in %

# ─────────────────────────────────────────────────────────────────────────────
# AUTOCORRELATION
# ─────────────────────────────────────────────────────────────────────────────
def _rolling_rank_autocorr(series, lag, min_periods=20):
    result = []
    for i in range(len(series)):
        if i < lag + min_periods:
            result.append(np.nan)
        else:
            s_now = series.iloc[lag: i + 1].values
            s_lag = series.iloc[: i + 1 - lag].values
            if len(s_now) < min_periods:
                result.append(np.nan)
            else:
                rho, _ = stats.spearmanr(s_now, s_lag)
                result.append(round(rho, 3))
    return pd.Series(result, index=series.index)


@st.cache_data(show_spinner="Computing rank autocorrelation...")
def compute_autocorr(xsec_json, lag):
    df_x = pd.read_json(xsec_json)
    out  = {}
    for coin in df_x.columns:
        s = df_x[coin].dropna()
        out[coin] = _rolling_rank_autocorr(s, lag).iloc[-1] if len(s) >= lag + 20 else np.nan
    return pd.Series(out).round(3)


autocorr_series = compute_autocorr(df_alpha_xsec_pct.to_json(), autocorr_lag)

# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔬 Diagnostics")

n_total     = len(all_coins)
n_valid_now = df_alpha.iloc[-1].notna().sum()
avg_pass    = pass_rate.mean()
n_low_pass  = (pass_rate < 0.3).sum()

dc1, dc2, dc3, dc4 = st.columns(4)
with dc1:
    st.metric("Total coins", n_total)
with dc2:
    st.metric("Valid alpha today", n_valid_now,
              delta=f"{n_valid_now / n_total:.0%} pass QC",
              delta_color="normal" if n_valid_now > n_total * 0.5 else "inverse")
with dc3:
    st.metric("Avg QC pass rate", f"{avg_pass:.1%}")
with dc4:
    st.metric("Coins <30% pass rate", n_low_pass,
              delta="⚠️ low signal quality" if n_low_pass > 0 else "✅ all ok",
              delta_color="inverse" if n_low_pass > 0 else "normal")

with st.expander("📋 Per-coin QC table"):
    diag_df = pd.DataFrame({
        "Pass Rate (QC)":                        pass_rate,
        "Alpha today (×100)":                    df_alpha.iloc[-1],
        "R² today":                              df_r2.iloc[-1],
        "Beta t-stat today":                     df_tstat.iloc[-1],
        f"Rank Autocorr (lag={autocorr_lag}d)":  autocorr_series,
    }).sort_values("Pass Rate (QC)", ascending=False)

    def _col_pass(v):
        if pd.isna(v): return "color:grey"
        if v >= 0.6:   return "color:#4ade80"
        if v >= 0.3:   return "color:#facc15"
        return "color:#f87171"

    def _col_ac(v):
        if pd.isna(v): return "color:grey"
        if v >= 0.4:   return "color:#4ade80"
        if v >= 0.2:   return "color:#facc15"
        return "color:#f87171"

    st.dataframe(
        diag_df.style
            .applymap(_col_pass, subset=["Pass Rate (QC)"])
            .applymap(_col_ac,   subset=[f"Rank Autocorr (lag={autocorr_lag}d)"])
            .format({
                "Pass Rate (QC)":                       "{:.1%}",
                "Alpha today (×100)":                   "{:.4f}",
                "R² today":                             "{:.3f}",
                "Beta t-stat today":                    "{:.2f}",
                f"Rank Autocorr (lag={autocorr_lag}d)": "{:.3f}",
            }, na_rep="—"),
        use_container_width=True, height=350
    )

with st.expander("📊 QC pass-rate distribution"):
    fig_diag = go.Figure(go.Histogram(
        x=pass_rate.values, nbinsx=20,
        marker_color="#6366f1", opacity=0.8
    ))
    fig_diag.add_vline(x=0.3, line_dash="dash", line_color="#f87171",
                       annotation_text="30%", annotation_position="top right")
    fig_diag.add_vline(x=0.6, line_dash="dash", line_color="#4ade80",
                       annotation_text="60%", annotation_position="top right")
    fig_diag.update_layout(title="QC Pass Rate Distribution",
                           xaxis_title="Pass Rate", yaxis_title="# Coins",
                           height=280, template="plotly_dark")
    st.plotly_chart(fig_diag, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# COMBINED SIGNAL CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📡 Combined Signal")

st.markdown(f"""
**Three layers must ALL pass** for a coin to be flagged:

| Layer | Sub-conditions |
|---|---|
| **L1 — Momentum** | X-sec pct in **{pct_lo:.0%}–{pct_hi:.0%}** · rose ≥ **{min_delta:.0%}** over {momentum_lookback}d · ≥ **{peak_margin:.0%}** below 30d peak |
| **L2 — Confirmed Divergence** | x-sec−hist ≥ **{min_divergence:.0%}** · widened ≥ **{min_div_widening:.0%}** over {div_widening_lookback}d · held for ≥ **{div_persistence_days}** consecutive days |
| **L3 — Price Gate** | {price_ret_window}d return ≥ **{min_price_ret:.1f}%** |
""")

# ── Today snapshots ──────────────────────────────────────────────────────────
last_xsec       = df_alpha_xsec_pct.iloc[-1]
last_hist       = df_alpha_hist_pct.iloc[-1]
last_peak_30    = rolling_peak_30.iloc[-1]
last_delta      = xsec_delta.iloc[-1]
last_div        = divergence.iloc[-1]
last_div_wide   = div_widening.iloc[-1]
last_div_pers   = div_pers_count.iloc[-1]
last_price_ret  = price_ret_Nd.iloc[-1]

rows = []
for coin in all_coins:
    xsec     = last_xsec.get(coin, np.nan)
    hist     = last_hist.get(coin, np.nan)
    peak     = last_peak_30.get(coin, np.nan)
    delta    = last_delta.get(coin, np.nan)
    div_lvl  = last_div.get(coin, np.nan)
    div_wide = last_div_wide.get(coin, np.nan)
    div_pers = last_div_pers.get(coin, np.nan)
    pret     = last_price_ret.get(coin, np.nan)
    autocorr = autocorr_series.get(coin, np.nan)

    if any(pd.isna(v) for v in [xsec, hist, peak, delta, div_lvl, div_wide, pret]):
        continue

    dist_peak = peak - xsec   # positive = below peak (good)

    # Layer 1 sub-conditions
    l1a = pct_lo <= xsec <= pct_hi
    l1b = delta  >= min_delta
    l1c = dist_peak >= peak_margin

    # Layer 2 sub-conditions
    l2a = div_lvl  >= min_divergence
    l2b = div_wide >= min_div_widening
    l2c = (not pd.isna(div_pers)) and (div_pers >= div_persistence_days)

    # Layer 3
    l3  = pret >= min_price_ret

    l1_pass = l1a and l1b and l1c
    l2_pass = l2a and l2b and l2c
    l3_pass = l3
    signal  = l1_pass and l2_pass and l3_pass

    sub_score = sum([l1a, l1b, l1c, l2a, l2b, l2c, l3])

    rows.append({
        "Coin":                                  coin,
        # L1 values
        "X-sec Pct":                             round(xsec,      3),
        f"Δ Pct ({momentum_lookback}d)":         round(delta,     3),
        "Dist from 30d Peak":                    round(dist_peak, 3),
        # L2 values
        "Divergence":                            round(div_lvl,   3),
        f"Div Widening ({div_widening_lookback}d)": round(div_wide, 3),
        "Div Pers Days":                         int(div_pers) if not pd.isna(div_pers) else np.nan,
        # L3 value
        f"Price Ret ({price_ret_window}d) %":    round(pret,      2),
        # Meta
        f"Autocorr (lag={autocorr_lag}d)":       round(autocorr, 3) if not pd.isna(autocorr) else np.nan,
        "Sub-conds (/ 7)":                       sub_score,
        # Layer flags
        "✅ L1":    l1_pass,
        "✅ L2":    l2_pass,
        "✅ L3":    l3_pass,
        "🎯 Signal": signal,
    })

df_signal = (
    pd.DataFrame(rows)
    .sort_values(["🎯 Signal", "Sub-conds (/ 7)"], ascending=[False, False])
)

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY METRICS
# ─────────────────────────────────────────────────────────────────────────────
n_signal  = int(df_signal["🎯 Signal"].sum())
n_l1l2    = int((df_signal["✅ L1"] & df_signal["✅ L2"] & ~df_signal["✅ L3"]).sum())
n_l1_only = int((df_signal["✅ L1"] & ~df_signal["✅ L2"] & ~df_signal["✅ L3"]).sum())
n_l2_only = int((~df_signal["✅ L1"] & df_signal["✅ L2"] & ~df_signal["✅ L3"]).sum())

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("🎯 Full signal (L1+L2+L3)", n_signal)
with m2:
    st.metric("L1+L2 — awaiting price confirm", n_l1l2)
with m3:
    st.metric("L1 only", n_l1_only)
with m4:
    st.metric("L2 only", n_l2_only)

# Callouts
passing = df_signal[df_signal["🎯 Signal"]]["Coin"].tolist()
if passing:
    st.success(f"🎯 **Full signal coins:** {', '.join(sorted(passing))}")

near_miss = df_signal[
    df_signal["✅ L1"] & df_signal["✅ L2"] & ~df_signal["✅ L3"]
]["Coin"].tolist()
if near_miss:
    st.warning(f"⏳ **Near-miss (L1+L2 pass, awaiting price confirmation):** {', '.join(sorted(near_miss))}")

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL TABLE + BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────
bool_cols = ["✅ L1", "✅ L2", "✅ L3", "🎯 Signal"]

def _bool_colour(val):
    if val is True:  return "color:#4ade80;font-weight:700"
    if val is False: return "color:#f87171"
    return ""

def _score_colour(val):
    if pd.isna(val): return ""
    if val == 7:     return "color:#4ade80;font-weight:700"
    if val >= 5:     return "color:#facc15"
    return "color:#f87171"

def _style_main(df):
    def _row(row):
        if row.get("🎯 Signal", False):
            return ["background-color:#14532d"] * len(row)
        if row.get("✅ L1", False) and row.get("✅ L2", False):
            return ["background-color:#1e3a5f"] * len(row)
        return [""] * len(row)
    return (
        df.style
          .apply(_row, axis=1)
          .applymap(_bool_colour, subset=bool_cols)
    )

tab_main, tab_breakdown = st.tabs(["📋 Signal table", "🔍 Sub-condition breakdown"])

with tab_main:
    st.dataframe(_style_main(df_signal), use_container_width=True, height=500)

with tab_breakdown:
    st.markdown(
        "Each row shows which of the 7 sub-conditions a coin passes individually. "
        "Use this to diagnose why a coin is one condition away from a full signal."
    )

    sub_cond_cols = {
        f"L1a: {pct_lo:.0%}≤pct≤{pct_hi:.0%}": df_signal.apply(
            lambda r: pct_lo <= r["X-sec Pct"] <= pct_hi, axis=1),
        f"L1b: Δ≥{min_delta:.2f}":              df_signal[f"Δ Pct ({momentum_lookback}d)"] >= min_delta,
        f"L1c: Peak margin≥{peak_margin:.2f}":   df_signal["Dist from 30d Peak"] >= peak_margin,
        f"L2a: Div≥{min_divergence:.2f}":        df_signal["Divergence"] >= min_divergence,
        f"L2b: Widen≥{min_div_widening:.2f}":    df_signal[f"Div Widening ({div_widening_lookback}d)"] >= min_div_widening,
        f"L2c: Persist {div_persistence_days}d": df_signal["Div Pers Days"] >= div_persistence_days,
        f"L3: Ret≥{min_price_ret:.1f}%":         df_signal[f"Price Ret ({price_ret_window}d) %"] >= min_price_ret,
    }

    df_bd = df_signal[["Coin", "Sub-conds (/ 7)"]].copy()
    for name, series in sub_cond_cols.items():
        df_bd[name] = series.values

    all_sub = list(sub_cond_cols.keys())

    st.dataframe(
        df_bd.sort_values("Sub-conds (/ 7)", ascending=False)
             .style
             .applymap(_bool_colour, subset=all_sub)
             .applymap(_score_colour, subset=["Sub-conds (/ 7)"]),
        use_container_width=True, height=500
    )

# ─────────────────────────────────────────────────────────────────────────────
# COIN INSPECTOR CHARTS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📉 Coin Inspector")

available_coins = list(df_alpha.columns)
default_coins = (
    df_signal[df_signal["🎯 Signal"]]["Coin"].tolist()[:2]
    or df_signal[df_signal["✅ L1"] & df_signal["✅ L2"]]["Coin"].tolist()[:1]
    or (["BNB"] if "BNB" in available_coins else available_coins[:1])
)

coins_to_plot = st.multiselect(
    "Select coin(s) to inspect",
    options=available_coins,
    default=default_coins,
)

if coins_to_plot:

    # ── Chart 1: Raw alpha ────────────────────────────────────────────────────
    fig1 = go.Figure()
    for coin in coins_to_plot:
        if coin in df_alpha.columns:
            s = df_alpha[coin].dropna()
            fig1.add_trace(go.Scatter(
                x=s.index, y=s.values, mode="lines", name=coin,
                hovertemplate=f"<b>{coin}</b> Alpha: %{{y:.4f}}<extra></extra>"
            ))
    fig1.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.3)
    fig1.update_layout(
        title=f"Rolling {window}-Day Alpha vs BTC (×100, QC-filtered)",
        xaxis_title="Date", yaxis_title="Alpha (×100)",
        height=340, template="plotly_dark",
        legend=dict(orientation="h", y=-0.28)
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ── Chart 2: X-sec pct + 30d peak + delta (L1) ───────────────────────────
    fig2 = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=[
            "Cross-sec Alpha Percentile + 30d Peak",
            f"Δ Percentile ({momentum_lookback}d) — L1 momentum"
        ],
        vertical_spacing=0.12
    )
    for coin in coins_to_plot:
        if coin not in df_alpha_xsec_pct.columns:
            continue
        s  = df_alpha_xsec_pct[coin].dropna()
        pk = rolling_peak_30[coin].dropna()
        d  = xsec_delta[coin].dropna()

        fig2.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines", name=coin,
            hovertemplate=f"<b>{coin}</b> X-sec pct: %{{y:.3f}}<extra></extra>"
        ), row=1, col=1)
        fig2.add_trace(go.Scatter(
            x=pk.index, y=pk.values, mode="lines",
            name=f"{coin} 30d peak", line=dict(dash="dot", width=1),
            showlegend=False, opacity=0.45,
            hovertemplate=f"<b>{coin}</b> 30d peak: %{{y:.3f}}<extra></extra>"
        ), row=1, col=1)
        fig2.add_trace(go.Scatter(
            x=d.index, y=d.values, mode="lines",
            name=f"{coin} Δ pct", showlegend=False,
            hovertemplate=f"<b>{coin}</b> Δ pct: %{{y:.3f}}<extra></extra>"
        ), row=2, col=1)

    fig2.add_hrect(y0=pct_lo, y1=pct_hi,
                   fillcolor="rgba(99,102,241,0.10)", line_width=0, row=1, col=1,
                   annotation_text="L1 range", annotation_position="top left")
    fig2.add_hline(y=0,         line_dash="dot",  line_color="white",   opacity=0.3, row=2, col=1)
    fig2.add_hline(y=min_delta, line_dash="dash", line_color="#6366f1", opacity=0.6, row=2, col=1,
                   annotation_text=f"min Δ={min_delta:.2f}", annotation_position="top right")
    fig2.update_layout(height=520, template="plotly_dark",
                       legend=dict(orientation="h", y=-0.12))
    st.plotly_chart(fig2, use_container_width=True)

    # ── Chart 3: Divergence — level, widening, persistence (L2) ──────────────
    fig3 = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=[
            "Hist vs X-sec Percentile",
            "Divergence level (X-sec − Hist) — L2a",
            f"Divergence widening over {div_widening_lookback}d — L2b confirmation"
        ],
        vertical_spacing=0.10
    )
    for coin in coins_to_plot:
        if coin not in df_alpha_hist_pct.columns:
            continue
        h   = df_alpha_hist_pct[coin].dropna()
        x   = df_alpha_xsec_pct[coin].dropna()
        div = divergence[coin].dropna()
        dw  = div_widening[coin].dropna()

        fig3.add_trace(go.Scatter(
            x=h.index, y=h.values, mode="lines",
            name=f"{coin} Hist", line=dict(dash="dot"),
            hovertemplate=f"<b>{coin}</b> Hist: %{{y:.3f}}<extra></extra>"
        ), row=1, col=1)
        fig3.add_trace(go.Scatter(
            x=x.index, y=x.values, mode="lines", name=f"{coin} X-sec",
            hovertemplate=f"<b>{coin}</b> X-sec: %{{y:.3f}}<extra></extra>"
        ), row=1, col=1)
        fig3.add_trace(go.Scatter(
            x=div.index, y=div.values, mode="lines",
            name=f"{coin} Div", showlegend=False,
            fill="tozeroy", fillcolor="rgba(99,102,241,0.12)",
            hovertemplate=f"<b>{coin}</b> Div: %{{y:.3f}}<extra></extra>"
        ), row=2, col=1)
        fig3.add_trace(go.Scatter(
            x=dw.index, y=dw.values, mode="lines",
            name=f"{coin} Div widening", showlegend=False,
            hovertemplate=f"<b>{coin}</b> Widening: %{{y:.3f}}<extra></extra>"
        ), row=3, col=1)

    for row_, thresh_, label_ in [
        (2, min_divergence,   f"min div={min_divergence:.2f}"),
        (3, min_div_widening, f"min widening={min_div_widening:.2f}"),
    ]:
        fig3.add_hline(y=0,       line_dash="dot",  line_color="white",   opacity=0.3,
                       row=row_, col=1)
        fig3.add_hline(y=thresh_, line_dash="dash", line_color="#6366f1", opacity=0.6,
                       row=row_, col=1,
                       annotation_text=label_, annotation_position="top right")

    fig3.update_layout(height=640, template="plotly_dark",
                       legend=dict(orientation="h", y=-0.07))
    st.plotly_chart(fig3, use_container_width=True)

    # ── Chart 4: Price return gate (L3) ──────────────────────────────────────
    fig4 = go.Figure()
    for coin in coins_to_plot:
        if coin in price_ret_Nd.columns:
            pr = price_ret_Nd[coin].dropna()
            fig4.add_trace(go.Scatter(
                x=pr.index, y=pr.values, mode="lines", name=coin,
                hovertemplate=f"<b>{coin}</b> {price_ret_window}d ret: %{{y:.2f}}%<extra></extra>"
            ))
    fig4.add_hline(y=min_price_ret, line_dash="dash", line_color="#6366f1", opacity=0.7,
                   annotation_text=f"gate={min_price_ret:.1f}%", annotation_position="top right")
    fig4.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.3)
    fig4.update_layout(
        title=f"Rolling {price_ret_window}-Day Price Return % (L3 price gate)",
        xaxis_title="Date", yaxis_title="Return (%)",
        height=320, template="plotly_dark",
        legend=dict(orientation="h", y=-0.32)
    )
    st.plotly_chart(fig4, use_container_width=True)

    # ── Chart 5: R² + beta t-stat (QC view) ──────────────────────────────────
    with st.expander("🔍 Regression quality over time (R² & beta t-stat)"):
        fig5 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             subplot_titles=["R²", "Beta t-stat"],
                             vertical_spacing=0.12)
        for coin in coins_to_plot:
            if coin not in df_r2.columns:
                continue
            fig5.add_trace(go.Scatter(
                x=df_r2[coin].dropna().index,
                y=df_r2[coin].dropna().values,
                mode="lines", name=f"{coin} R²"
            ), row=1, col=1)
            fig5.add_trace(go.Scatter(
                x=df_tstat[coin].dropna().index,
                y=df_tstat[coin].dropna().values,
                mode="lines", name=f"{coin} t-stat", showlegend=False
            ), row=2, col=1)
        fig5.add_hline(y=min_r2, line_dash="dash", line_color="#f87171",
                       row=1, col=1, annotation_text=f"min R²={min_r2}",
                       annotation_position="top right")
        fig5.add_hline(y=min_beta_tstat, line_dash="dash", line_color="#f87171",
                       row=2, col=1, annotation_text=f"min |t|={min_beta_tstat}",
                       annotation_position="top right")
        fig5.update_layout(height=440, template="plotly_dark",
                           legend=dict(orientation="h", y=-0.12))
        st.plotly_chart(fig5, use_container_width=True)

else:
    st.info("Select at least one coin above to view charts.")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"Window: **{window}d** | Min R²: **{min_r2}** | Min |β t-stat|: **{min_beta_tstat}** | "
    f"Min obs: **{min_obs}** | Autocorr lag: **{autocorr_lag}d** | "
    f"Hist pct: expanding window (no lookahead) | "
    f"Price gate: {price_ret_window}d return ≥ {min_price_ret:.1f}%"
)
