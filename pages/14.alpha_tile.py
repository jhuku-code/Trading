"""
Alpha Percentile vs BTC — Two-Stage Edition
============================================

Stage 1 — Pre-filter (from Suggestion 1 logic)
───────────────────────────────────────────────
Keep only coins where, on the latest day:

    hist_pct  ≥  median_threshold   (above-median alpha vs own history)
    OR
    xsec_pct  ≥  median_threshold   (above-median alpha vs peers)

  • Both are expanding-window / cross-sectional ranks so no lookahead.
  • "OR" is intentional: a coin can qualify by being strong on either
    dimension. Requiring BOTH would be too restrictive and would eliminate
    coins that are quietly improving vs their own history but not yet
    leading the field cross-sectionally (and vice versa).
  • median_threshold defaults to 0.50 but is user-adjustable.

Stage 2 — Composite scoring (from Suggestion A / v3)
──────────────────────────────────────────────────────
Among coins that pass Stage 1, score each on five components,
each normalised to 0–1 via cross-sectional rank among the
pre-filtered universe, then combined with user weights:

    S1  Historical pct     — expanding-window rank vs own history
    S2  Cross-sec pct      — rank vs peers on today's alpha
    S3  Momentum (Δ pct)   — change in x-sec pct over N days
    S4  Divergence         — x-sec pct minus hist pct
    S5  Not exhausted      — distance below recent x-sec peak

Quality gating (carried from Code 2):
    R² ≥ min_r2  AND  |beta t-stat| ≥ min_beta_tstat  AND  obs ≥ min_obs

Lookahead fix:
    Historical pct uses expanding().rank(pct=True).
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

# ─────────────────────────────────────────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Alpha Percentile — Two-Stage", layout="wide")

st.markdown("""
<style>
[data-testid="metric-container"] {
    background: #12121f;
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

st.title("📐 Alpha Percentile vs BTC — Two-Stage")
st.caption(
    "Stage 1: keep coins above-median on historical OR cross-sectional alpha pct. "
    "Stage 2: score survivors on 5 weighted components."
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
# SIDEBAR — PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Parameters")

    # ── OLS window ──────────────────────────────────────────────────────────
    st.subheader("Rolling Alpha Window")
    window = st.select_slider(
        "Alpha estimation window (days)",
        options=[30, 60, 90, 180],
        value=90,
        help="Days used in each rolling OLS regression."
    )

    # ── QC filters ──────────────────────────────────────────────────────────
    st.subheader("Regression Quality Control")
    min_r2 = st.slider(
        "Min R²", 0.0, 0.5, 0.10, 0.01,
        help="Reject windows where BTC explains less than this share of coin variance."
    )
    min_beta_tstat = st.slider(
        "Min |beta t-stat|", 0.0, 4.0, 1.5, 0.1,
        help="Reject windows where the BTC-beta is not statistically meaningful."
    )
    min_obs = st.slider(
        "Min valid observations in window",
        10, window, max(10, int(window * 0.7)), 5,
        help="Minimum non-NaN days required inside the rolling window."
    )

    # ── Stage 1 pre-filter ──────────────────────────────────────────────────
    st.subheader("Stage 1 — Pre-filter")
    median_threshold = st.slider(
        "Median threshold (hist OR x-sec pct ≥)",
        0.0, 1.0, 0.50, 0.05,
        help=(
            "A coin survives Stage 1 if its historical pct OR its cross-sec pct "
            "is at or above this value today. 0.50 = above-median on either dimension."
        )
    )

    # ── S3 momentum ─────────────────────────────────────────────────────────
    st.subheader("S3 — Momentum Lookback")
    momentum_lookback = st.slider(
        "Δ X-sec pct lookback (days)", 5, 30, 14, 1,
        help="How far back to measure the change in cross-sectional percentile."
    )

    # ── S5 exhaustion ───────────────────────────────────────────────────────
    st.subheader("S5 — Exhaustion Peak Window")
    peak_window = st.slider(
        "Rolling peak window (days)", 10, 60, 30, 5,
        help="Window for measuring the recent high in x-sec percentile."
    )

    # ── Weights ──────────────────────────────────────────────────────────────
    st.subheader("Component Weights (Stage 2)")
    st.caption("Each weight scales its component's 0–1 normalised score.")

    w1 = st.slider("W1 — Historical pct",   0.0, 2.0, 1.0, 0.1,
                   help="Alpha vs coin's own history.")
    w2 = st.slider("W2 — Cross-sec pct",    0.0, 2.0, 1.0, 0.1,
                   help="Alpha vs all peers today.")
    w3 = st.slider("W3 — Momentum (Δ pct)", 0.0, 2.0, 1.0, 0.1,
                   help="Is the x-sec percentile rising?")
    w4 = st.slider("W4 — Divergence",       0.0, 2.0, 1.0, 0.1,
                   help="x-sec pct beating hist pct — unusual cross-sec breakout.")
    w5 = st.slider("W5 — Not exhausted",    0.0, 2.0, 1.0, 0.1,
                   help="Far below recent peak — room to run.")

    max_possible = w1 + w2 + w3 + w4 + w5
    st.caption(f"Max composite: **{max_possible:.1f}**")

    # ── Display filter ───────────────────────────────────────────────────────
    st.subheader("Display Filter (Stage 2)")
    min_score_pct = st.slider(
        "Show coins with composite ≥ (% of max)", 0, 100, 40, 5,
        help="Within Stage 1 survivors, show only those above this composite threshold."
    )
    min_score_abs = (min_score_pct / 100) * max_possible

    # ── Autocorrelation ──────────────────────────────────────────────────────
    st.subheader("Autocorrelation Diagnostic")
    autocorr_lag = st.slider(
        "Autocorrelation lag (days)", 3, 21, 7, 1,
        help="Spearman rank autocorrelation lag for x-sec percentile."
    )

# ─────────────────────────────────────────────────────────────────────────────
# RETURNS
# ─────────────────────────────────────────────────────────────────────────────
df_ret = df_price.pct_change().dropna(axis=1, how="all").copy()

if "BTC" not in df_ret.columns:
    st.error("BTC column missing after return calculation.")
    st.stop()

btc_ret   = df_ret["BTC"]
all_coins = [c for c in df_ret.columns if c != "BTC"]

# ─────────────────────────────────────────────────────────────────────────────
# ROLLING OLS ALPHA — WITH QC GATING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Computing rolling OLS alpha with QC filters…")
def compute_rolling_alpha(df_ret_json, btc_ret_json, window, min_r2, min_beta_tstat, min_obs_):
    df_r  = pd.read_json(df_ret_json)
    btc_r = pd.read_json(btc_ret_json, typ="series")
    coins = [c for c in df_r.columns if c != "BTC"]
    n     = len(df_r)

    alpha_d = {c: [np.nan] * n for c in coins}
    r2_d    = {c: [np.nan] * n for c in coins}
    tstat_d = {c: [np.nan] * n for c in coins}

    for i in range(n):
        if i < window - 1:
            continue
        btc_w = btc_r.iloc[i - window + 1: i + 1].values
        for coin in coins:
            coin_w = df_r[coin].iloc[i - window + 1: i + 1].values
            mask   = (~np.isnan(coin_w)) & (~np.isnan(btc_w))
            n_obs  = mask.sum()
            if n_obs < min_obs_:
                continue

            X = btc_w[mask].reshape(-1, 1)
            y = coin_w[mask]

            model     = LinearRegression().fit(X, y)
            y_pred    = model.predict(X)
            residuals = y - y_pred

            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            se2     = ss_res / max(n_obs - 2, 1)
            x_var   = np.sum((X.flatten() - X.mean()) ** 2)
            beta_se = np.sqrt(se2 / x_var) if x_var > 0 else np.inf
            b_tstat = model.coef_[0] / beta_se if beta_se > 0 else 0.0

            if r2 < min_r2 or abs(b_tstat) < min_beta_tstat:
                continue

            alpha_d[coin][i] = model.intercept_ * 100
            r2_d[coin][i]    = round(r2, 4)
            tstat_d[coin][i] = round(b_tstat, 4)

    idx      = df_r.index
    df_alpha = pd.DataFrame(alpha_d, index=idx).round(4)
    df_r2_   = pd.DataFrame(r2_d,    index=idx).round(4)
    df_ts_   = pd.DataFrame(tstat_d, index=idx).round(4)

    total_w   = max(n - window + 1, 1)
    pass_rate = (df_alpha.notna().sum() / total_w).round(3)

    return df_alpha.to_json(), df_r2_.to_json(), df_ts_.to_json(), pass_rate.to_json()


alpha_json, r2_json, tstat_json, pass_rate_json = compute_rolling_alpha(
    df_ret.to_json(), btc_ret.to_json(),
    window, min_r2, min_beta_tstat, min_obs
)

df_alpha  = pd.read_json(alpha_json)
df_r2     = pd.read_json(r2_json)
df_tstat  = pd.read_json(tstat_json)
pass_rate = pd.read_json(pass_rate_json, typ="series")

# ─────────────────────────────────────────────────────────────────────────────
# FULL-UNIVERSE PERCENTILES  (computed over ALL coins, before any filter)
# ─────────────────────────────────────────────────────────────────────────────

# S1: Historical — expanding window per coin, no lookahead
df_hist_pct = df_alpha.apply(lambda x: x.expanding().rank(pct=True))

# S2: Cross-sectional — rank within each day across ALL coins
df_xsec_pct = df_alpha.rank(axis=1, pct=True)

# Derived series (also over full universe for correct relative ranking)
df_momentum   = df_xsec_pct - df_xsec_pct.shift(momentum_lookback)
df_divergence = df_xsec_pct - df_hist_pct
df_peak       = df_xsec_pct.rolling(peak_window).max()
df_dist_peak  = df_peak - df_xsec_pct   # high = far below peak = not exhausted

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — PRE-FILTER
# ─────────────────────────────────────────────────────────────────────────────
# Evaluate on the latest day with valid data
last_hist = df_hist_pct.iloc[-1]
last_xsec = df_xsec_pct.iloc[-1]

# A coin passes if it is above (or at) the threshold on EITHER dimension
stage1_pass = last_hist.index[
    (last_hist >= median_threshold) | (last_xsec >= median_threshold)
].tolist()

# Coins that had NaN on either metric are excluded
stage1_pass = [
    c for c in stage1_pass
    if not pd.isna(last_hist.get(c, np.nan))
    and not pd.isna(last_xsec.get(c, np.nan))
]

n_total_valid = int((~last_hist.isna()).sum())
n_stage1      = len(stage1_pass)
n_dropped     = n_total_valid - n_stage1

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — NORMALISE WITHIN THE FILTERED UNIVERSE & SCORE
# ─────────────────────────────────────────────────────────────────────────────
# Normalisation is done cross-sectionally within Stage 1 survivors only.
# This keeps scores meaningful: a 0.8 on S1 means top 20% *among coins
# that already passed Stage 1*, not among all coins.

def cs_rank_universe(df, universe):
    """Cross-sec rank each row, restricted to the given universe of columns."""
    df_u = df[universe] if all(c in df.columns for c in universe) else df
    return df_u.rank(axis=1, pct=True)


if stage1_pass:
    n1 = cs_rank_universe(df_hist_pct,   stage1_pass)
    n2 = cs_rank_universe(df_xsec_pct,   stage1_pass)
    n3 = cs_rank_universe(df_momentum,   stage1_pass)
    n4 = cs_rank_universe(df_divergence, stage1_pass)
    n5 = cs_rank_universe(df_dist_peak,  stage1_pass)

    df_composite = w1*n1 + w2*n2 + w3*n3 + w4*n4 + w5*n5
else:
    n1 = n2 = n3 = n4 = n5 = df_alpha[[]].copy()
    df_composite = df_alpha[[]].copy()

# ─────────────────────────────────────────────────────────────────────────────
# AUTOCORRELATION (diagnostic, full universe)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Computing rank autocorrelation…")
def compute_autocorr(xsec_json, lag):
    from scipy import stats as scipy_stats
    df_x = pd.read_json(xsec_json)
    out  = {}
    for coin in df_x.columns:
        s = df_x[coin].dropna()
        if len(s) < lag + 20:
            out[coin] = np.nan
            continue
        rho, _ = scipy_stats.spearmanr(s.iloc[lag:].values, s.iloc[:-lag].values)
        out[coin] = round(rho, 3)
    return pd.Series(out).round(3)


autocorr_series = compute_autocorr(df_xsec_pct.to_json(), autocorr_lag)

# ─────────────────────────────────────────────────────────────────────────────
# TODAY'S SNAPSHOT FOR SIGNAL TABLE
# ─────────────────────────────────────────────────────────────────────────────
last_s1   = df_hist_pct.iloc[-1]
last_s2   = df_xsec_pct.iloc[-1]
last_s3   = df_momentum.iloc[-1]
last_s4   = df_divergence.iloc[-1]
last_s5   = df_dist_peak.iloc[-1]
last_n1   = n1.iloc[-1] if not n1.empty else pd.Series(dtype=float)
last_n2   = n2.iloc[-1] if not n2.empty else pd.Series(dtype=float)
last_n3   = n3.iloc[-1] if not n3.empty else pd.Series(dtype=float)
last_n4   = n4.iloc[-1] if not n4.empty else pd.Series(dtype=float)
last_n5   = n5.iloc[-1] if not n5.empty else pd.Series(dtype=float)
last_comp = df_composite.iloc[-1] if not df_composite.empty else pd.Series(dtype=float)

rows = []
for coin in stage1_pass:
    comp = last_comp.get(coin, np.nan)
    if pd.isna(comp):
        continue

    # Which Stage-1 condition(s) fired
    hist_ok = last_s1.get(coin, np.nan) >= median_threshold
    xsec_ok = last_s2.get(coin, np.nan) >= median_threshold
    gate_str = (
        "Hist + X-sec" if (hist_ok and xsec_ok) else
        ("Hist only"   if hist_ok else "X-sec only")
    )

    rows.append({
        "Coin":                              coin,
        "Stage-1 gate":                      gate_str,
        # Raw component values (full-universe percentiles)
        "S1 Hist Pct":                       round(last_s1.get(coin, np.nan), 3),
        "S2 X-sec Pct":                      round(last_s2.get(coin, np.nan), 3),
        f"S3 Δ Pct ({momentum_lookback}d)":  round(last_s3.get(coin, np.nan), 3),
        "S4 Divergence":                     round(last_s4.get(coin, np.nan), 3),
        f"S5 Dist Peak ({peak_window}d)":    round(last_s5.get(coin, np.nan), 3),
        # Normalised scores (within Stage-1 universe)
        "N1": round(last_n1.get(coin, np.nan), 3),
        "N2": round(last_n2.get(coin, np.nan), 3),
        "N3": round(last_n3.get(coin, np.nan), 3),
        "N4": round(last_n4.get(coin, np.nan), 3),
        "N5": round(last_n5.get(coin, np.nan), 3),
        # Composite
        "Composite":   round(comp, 3),
        "Composite %": round(comp / max_possible, 3) if max_possible > 0 else np.nan,
        # Autocorr
        f"Autocorr (lag={autocorr_lag}d)": (
            round(autocorr_series.get(coin, np.nan), 3)
            if not pd.isna(autocorr_series.get(coin, np.nan)) else np.nan
        ),
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
n_valid_today = int(df_alpha.iloc[-1].notna().sum())
n_above_score = len(df_signal_filtered)
top_coin      = df_signal.iloc[0]["Coin"]      if len(df_signal) > 0 else "—"
top_score_val = df_signal.iloc[0]["Composite"] if len(df_signal) > 0 else np.nan

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    st.metric("Total coins", n_total_coins)
with c2:
    st.metric("Valid alpha today", n_valid_today,
              delta=f"{n_valid_today/n_total_coins:.0%} pass QC",
              delta_color="normal" if n_valid_today > n_total_coins * 0.5 else "inverse")
with c3:
    st.metric("Pass Stage 1 filter", n_stage1,
              delta=f"−{n_dropped} dropped",
              delta_color="off")
with c4:
    st.metric("Above score threshold", n_above_score)
with c5:
    st.metric("Avg QC pass rate", f"{pass_rate.mean():.1%}")
with c6:
    st.metric(
        "Top coin today", top_coin,
        delta=f"{top_score_val:.2f} / {max_possible:.1f}" if not pd.isna(top_score_val) else "—"
    )

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🚦 Stage 1 — Pre-filter Detail")

col_info, col_venn = st.columns([1.4, 1])

with col_info:
    # Show exactly which coins passed and why
    stage1_rows = []
    for coin in [c for c in all_coins if not pd.isna(last_hist.get(c, np.nan))]:
        h = last_hist.get(coin, np.nan)
        x = last_xsec.get(coin, np.nan)
        hist_ok = h >= median_threshold
        xsec_ok = x >= median_threshold
        passed  = hist_ok or xsec_ok
        gate    = (
            "Hist + X-sec" if (hist_ok and xsec_ok) else
            ("Hist only"   if hist_ok else
             ("X-sec only"  if xsec_ok else "—"))
        )
        stage1_rows.append({
            "Coin": coin,
            "Hist Pct": round(h, 3),
            "X-sec Pct": round(x, 3),
            "Passes Stage 1": passed,
            "Gate": gate,
        })

    df_stage1 = (
        pd.DataFrame(stage1_rows)
        .sort_values(["Passes Stage 1", "Hist Pct"], ascending=[False, False])
        .reset_index(drop=True)
    )

    def _c_pass_s1(val):
        if val is True:  return "color:#4ade80;font-weight:700"
        if val is False: return "color:#f87171"
        return ""

    def _c_pct_s1(val):
        if pd.isna(val): return "color:grey"
        return "color:#4ade80" if val >= median_threshold else "color:#f87171"

    st.dataframe(
        df_stage1.style
            .applymap(_c_pass_s1, subset=["Passes Stage 1"])
            .applymap(_c_pct_s1,  subset=["Hist Pct", "X-sec Pct"])
            .format({"Hist Pct": "{:.3f}", "X-sec Pct": "{:.3f}"}, na_rep="—"),
        use_container_width=True, height=340
    )

with col_venn:
    # Venn-like bar showing how many passed via each route
    n_both      = sum(1 for c in stage1_pass if
                      last_hist.get(c, 0) >= median_threshold and
                      last_xsec.get(c, 0) >= median_threshold)
    n_hist_only = sum(1 for c in stage1_pass if
                      last_hist.get(c, 0) >= median_threshold and
                      last_xsec.get(c, 0) < median_threshold)
    n_xsec_only = sum(1 for c in stage1_pass if
                      last_hist.get(c, 0) < median_threshold and
                      last_xsec.get(c, 0) >= median_threshold)

    fig_venn = go.Figure(go.Bar(
        x=["Hist only", "X-sec only", "Both"],
        y=[n_hist_only, n_xsec_only, n_both],
        marker_color=["#60a5fa", "#a78bfa", "#4ade80"],
        text=[n_hist_only, n_xsec_only, n_both],
        textposition="outside"
    ))
    fig_venn.update_layout(
        title=f"Stage 1 pass breakdown (threshold={median_threshold:.2f})",
        yaxis_title="# Coins",
        height=320, template="plotly_dark",
        showlegend=False
    )
    st.plotly_chart(fig_venn, use_container_width=True)

    # Scatter: Hist pct vs X-sec pct, coloured by pass/fail
    pass_set = set(stage1_pass)
    colors_scatter = [
        "#4ade80" if c in pass_set else "#f87171"
        for c in df_stage1["Coin"]
    ]
    fig_scatter = go.Figure(go.Scatter(
        x=df_stage1["Hist Pct"],
        y=df_stage1["X-sec Pct"],
        mode="markers+text",
        text=df_stage1["Coin"],
        textposition="top center",
        textfont=dict(size=8),
        marker=dict(color=colors_scatter, size=7, opacity=0.8),
        hovertemplate="<b>%{text}</b><br>Hist: %{x:.3f}<br>X-sec: %{y:.3f}<extra></extra>"
    ))
    fig_scatter.add_hline(y=median_threshold, line_dash="dash",
                          line_color="#facc15", opacity=0.6,
                          annotation_text=f"x-sec={median_threshold:.2f}",
                          annotation_position="right")
    fig_scatter.add_vline(x=median_threshold, line_dash="dash",
                          line_color="#facc15", opacity=0.6,
                          annotation_text=f"hist={median_threshold:.2f}",
                          annotation_position="top")
    fig_scatter.update_layout(
        title="Hist Pct vs X-sec Pct (green = pass)",
        xaxis_title="Hist Pct", yaxis_title="X-sec Pct",
        height=350, template="plotly_dark"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — COMPOSITE SCORE TABLE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Stage 2 — Composite Score Table")

st.markdown(f"""
| Component | Raw metric | What it captures | Weight |
|---|---|---|---|
| **S1 Hist Pct** | Expanding-window rank vs own history | Good alpha relative to own past | **{w1}** |
| **S2 X-sec Pct** | Rank vs all peers today | Winning the room | **{w2}** |
| **S3 Momentum** | Δ x-sec pct over {momentum_lookback}d | Alpha rank is rising | **{w3}** |
| **S4 Divergence** | X-sec pct minus hist pct | Unusually strong vs peers relative to own baseline | **{w4}** |
| **S5 Not Exhausted** | Distance below {peak_window}d x-sec peak | Room left to run | **{w5}** |

*Normalised scores (N1–N5) are cross-sec ranked within the Stage 1 universe.*  
Max composite = **{max_possible:.1f}** &nbsp;|&nbsp; Showing coins ≥ **{min_score_abs:.2f}** ({min_score_pct}% of max)
""")

if not df_signal_filtered.empty:
    top5 = df_signal_filtered.head(5)["Coin"].tolist()
    st.success(f"🏆 Top coins: **{' · '.join(top5)}**")

    # Also flag coins that passed on just one gate
    hist_only_top = df_signal_filtered[df_signal_filtered["Stage-1 gate"] == "Hist only"]["Coin"].tolist()
    xsec_only_top = df_signal_filtered[df_signal_filtered["Stage-1 gate"] == "X-sec only"]["Coin"].tolist()
    if hist_only_top:
        st.info(f"📌 Passed via **Hist only** (improving vs own history, not yet leading peers): {', '.join(hist_only_top)}")
    if xsec_only_top:
        st.info(f"📌 Passed via **X-sec only** (leading peers but not unusually strong vs own history): {', '.join(xsec_only_top)}")
else:
    st.warning("No coins above the composite threshold in the Stage 1 universe. Lower the threshold or adjust weights.")

score_cols = ["N1", "N2", "N3", "N4", "N5"]

def _colour_norm(v):
    if pd.isna(v): return "color:grey"
    return "color:#4ade80" if v >= 0.7 else ("color:#facc15" if v >= 0.4 else "color:#f87171")

def _colour_comp(v):
    if pd.isna(v) or max_possible == 0: return ""
    frac = v / max_possible
    return (
        "color:#4ade80;font-weight:700" if frac >= 0.70 else
        ("color:#facc15;font-weight:600" if frac >= 0.45 else "color:#f87171")
    )

def _colour_gate(v):
    if v == "Hist + X-sec": return "color:#4ade80"
    if v in ("Hist only", "X-sec only"): return "color:#facc15"
    return ""

def _colour_ac(v):
    if pd.isna(v): return "color:grey"
    return "color:#4ade80" if v >= 0.4 else ("color:#facc15" if v >= 0.2 else "color:#f87171")

def highlight_top(row):
    frac = row["Composite"] / max_possible if max_possible > 0 else 0
    if frac >= 0.70: return ["background-color:#14532d"] * len(row)
    if frac >= 0.50: return ["background-color:#1e3a5f"] * len(row)
    return [""] * len(row)

fmt = {
    "S1 Hist Pct": "{:.3f}", "S2 X-sec Pct": "{:.3f}",
    f"S3 Δ Pct ({momentum_lookback}d)": "{:.3f}",
    "S4 Divergence": "{:.3f}",
    f"S5 Dist Peak ({peak_window}d)": "{:.3f}",
    "N1": "{:.3f}", "N2": "{:.3f}", "N3": "{:.3f}",
    "N4": "{:.3f}", "N5": "{:.3f}",
    "Composite": "{:.3f}", "Composite %": "{:.1%}",
    f"Autocorr (lag={autocorr_lag}d)": "{:.3f}",
}

tab_full, tab_raw, tab_norm = st.tabs([
    "🎯 Full table", "📋 Raw components", "🔢 Normalised scores"
])

with tab_full:
    st.dataframe(
        df_signal_filtered.style
            .apply(highlight_top, axis=1)
            .applymap(_colour_norm,  subset=score_cols)
            .applymap(_colour_comp,  subset=["Composite"])
            .applymap(_colour_gate,  subset=["Stage-1 gate"])
            .applymap(_colour_ac,    subset=[f"Autocorr (lag={autocorr_lag}d)"])
            .format(fmt, na_rep="—"),
        use_container_width=True, height=500
    )

with tab_raw:
    raw_cols = ["Coin", "Stage-1 gate", "S1 Hist Pct", "S2 X-sec Pct",
                f"S3 Δ Pct ({momentum_lookback}d)", "S4 Divergence",
                f"S5 Dist Peak ({peak_window}d)", "Composite"]
    st.dataframe(
        df_signal_filtered[raw_cols].style
            .apply(highlight_top, axis=1)
            .applymap(_colour_comp, subset=["Composite"])
            .applymap(_colour_gate, subset=["Stage-1 gate"])
            .format({c: "{:.3f}" for c in raw_cols if c not in ("Coin","Stage-1 gate")},
                    na_rep="—"),
        use_container_width=True, height=500
    )

with tab_norm:
    norm_cols = ["Coin", "Stage-1 gate", "N1", "N2", "N3", "N4", "N5",
                 "Composite", "Composite %"]
    st.dataframe(
        df_signal_filtered[norm_cols].style
            .apply(highlight_top, axis=1)
            .applymap(_colour_norm, subset=score_cols)
            .applymap(_colour_comp, subset=["Composite"])
            .applymap(_colour_gate, subset=["Stage-1 gate"])
            .format({
                "N1": "{:.3f}", "N2": "{:.3f}", "N3": "{:.3f}",
                "N4": "{:.3f}", "N5": "{:.3f}",
                "Composite": "{:.3f}", "Composite %": "{:.1%}",
            }, na_rep="—"),
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
    # Background: all Stage 1 coins
    fig_dist.add_trace(go.Histogram(
        x=df_signal["Composite"].dropna().values,
        nbinsx=25, marker_color="#6366f1", opacity=0.55,
        name="Stage 1 universe"
    ))
    # Overlay: coins above threshold
    if not df_signal_filtered.empty:
        fig_dist.add_trace(go.Histogram(
            x=df_signal_filtered["Composite"].dropna().values,
            nbinsx=25, marker_color="#4ade80", opacity=0.75,
            name=f"Above threshold ({min_score_pct}%)"
        ))
    fig_dist.add_vline(
        x=min_score_abs, line_dash="dash", line_color="#facc15",
        annotation_text=f"threshold={min_score_abs:.2f}",
        annotation_position="top right"
    )
    fig_dist.update_layout(
        title="Composite Score Distribution — Stage 1 Universe",
        xaxis_title="Composite Score", yaxis_title="# Coins",
        barmode="overlay", height=300, template="plotly_dark",
        legend=dict(orientation="h", y=-0.35)
    )
    st.plotly_chart(fig_dist, use_container_width=True)

with col_radar:
    top3 = df_signal_filtered.head(3)
    if not top3.empty:
        categories = ["Hist Pct", "X-sec Pct", "Momentum", "Divergence", "Not Exhausted"]
        fig_radar  = go.Figure()
        for _, row in top3.iterrows():
            vals = [row["N1"], row["N2"], row["N3"], row["N4"], row["N5"]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=categories + [categories[0]],
                fill="toself", name=row["Coin"], opacity=0.7
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Top 3 coins — component profile",
            height=300, template="plotly_dark",
            legend=dict(orientation="h", y=-0.25)
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.info("No coins above threshold.")

# ─────────────────────────────────────────────────────────────────────────────
# QC DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("🔬 QC Diagnostics", expanded=False):
    diag_df = pd.DataFrame({
        "Pass Rate (QC)":                       pass_rate,
        "Alpha today (×100)":                   df_alpha.iloc[-1],
        "R² today":                             df_r2.iloc[-1],
        "Beta t-stat today":                    df_tstat.iloc[-1],
        f"Autocorr (lag={autocorr_lag}d)":      autocorr_series,
    }).sort_values("Pass Rate (QC)", ascending=False)

    def _c_pass(v):
        if pd.isna(v): return "color:grey"
        return "color:#4ade80" if v >= 0.6 else ("color:#facc15" if v >= 0.3 else "color:#f87171")

    def _c_ac_diag(v):
        if pd.isna(v): return "color:grey"
        return "color:#4ade80" if v >= 0.4 else ("color:#facc15" if v >= 0.2 else "color:#f87171")

    st.dataframe(
        diag_df.style
            .applymap(_c_pass,    subset=["Pass Rate (QC)"])
            .applymap(_c_ac_diag, subset=[f"Autocorr (lag={autocorr_lag}d)"])
            .format({
                "Pass Rate (QC)":                  "{:.1%}",
                "Alpha today (×100)":              "{:.4f}",
                "R² today":                        "{:.3f}",
                "Beta t-stat today":               "{:.2f}",
                f"Autocorr (lag={autocorr_lag}d)": "{:.3f}",
            }, na_rep="—"),
        use_container_width=True, height=320
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
        fig_ac.update_layout(title=f"Autocorr Distribution (lag={autocorr_lag}d)",
                             xaxis_title="Spearman rho", yaxis_title="# Coins",
                             height=240, template="plotly_dark")
        st.plotly_chart(fig_ac, use_container_width=True)

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

    # Chart 1: Raw alpha
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
        title=f"Rolling {window}-Day Alpha vs BTC (QC-filtered, ×100)",
        xaxis_title="Date", yaxis_title="Alpha (×100)",
        height=300, template="plotly_dark",
        legend=dict(orientation="h", y=-0.30)
    )
    st.plotly_chart(fig_a, use_container_width=True)

    # Chart 2: Five components over time
    fig_comp = make_subplots(
        rows=5, cols=1, shared_xaxes=True,
        subplot_titles=[
            "S1 — Historical pct (expanding, no lookahead)",
            "S2 — Cross-sec pct (full universe)",
            f"S3 — Momentum Δ x-sec pct ({momentum_lookback}d)",
            "S4 — Divergence (x-sec − hist)",
            f"S5 — Distance from {peak_window}d peak (not exhausted)",
        ],
        vertical_spacing=0.04
    )
    series_map = [
        (df_hist_pct,   1, "#60a5fa"),
        (df_xsec_pct,   2, "#a78bfa"),
        (df_momentum,   3, "#34d399"),
        (df_divergence, 4, "#f472b6"),
        (df_dist_peak,  5, "#fbbf24"),
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

    # Threshold lines
    for r_ in [1, 2]:
        fig_comp.add_hline(y=median_threshold, line_dash="dash",
                           line_color="#facc15", opacity=0.5,
                           annotation_text=f"Stage-1 thr={median_threshold:.2f}",
                           annotation_position="top right",
                           row=r_, col=1)
        fig_comp.add_hline(y=0.5, line_dash="dot", line_color="white",
                           opacity=0.2, row=r_, col=1)
    for r_ in [3, 4, 5]:
        fig_comp.add_hline(y=0, line_dash="dot", line_color="white",
                           opacity=0.2, row=r_, col=1)

    fig_comp.update_layout(
        height=900, template="plotly_dark",
        legend=dict(orientation="h", y=-0.04),
        title_text="Five Components Over Time"
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # Chart 3: Composite score over time
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
        title="Rolling Composite Score Over Time (within Stage 1 universe)",
        xaxis_title="Date", yaxis_title="Composite",
        height=300, template="plotly_dark",
        legend=dict(orientation="h", y=-0.30)
    )
    st.plotly_chart(fig_cs, use_container_width=True)

    # Chart 4: R² + beta t-stat
    with st.expander("🔍 Regression quality over time"):
        fig_qc = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               subplot_titles=["R²", "Beta t-stat"],
                               vertical_spacing=0.12)
        for coin in coins_to_plot:
            if coin in df_r2.columns:
                fig_qc.add_trace(go.Scatter(
                    x=df_r2[coin].dropna().index,
                    y=df_r2[coin].dropna().values,
                    mode="lines", name=f"{coin} R²"
                ), row=1, col=1)
                fig_qc.add_trace(go.Scatter(
                    x=df_tstat[coin].dropna().index,
                    y=df_tstat[coin].dropna().values,
                    mode="lines", name=f"{coin} t-stat", showlegend=False
                ), row=2, col=1)
        fig_qc.add_hline(y=min_r2, line_dash="dash", line_color="#f87171",
                         row=1, col=1,
                         annotation_text=f"min R²={min_r2}",
                         annotation_position="top right")
        fig_qc.add_hline(y=min_beta_tstat, line_dash="dash", line_color="#f87171",
                         row=2, col=1,
                         annotation_text=f"min |t|={min_beta_tstat}",
                         annotation_position="top right")
        fig_qc.update_layout(height=400, template="plotly_dark",
                             legend=dict(orientation="h", y=-0.12))
        st.plotly_chart(fig_qc, use_container_width=True)

else:
    st.info("Select at least one coin above to view charts.")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"Window: **{window}d** | Min R²: **{min_r2}** | Min |β t-stat|: **{min_beta_tstat}** | "
    f"Min obs: **{min_obs}** | Stage-1 threshold: **{median_threshold:.2f}** | "
    f"Momentum lookback: **{momentum_lookback}d** | Peak window: **{peak_window}d** | "
    f"Autocorr lag: **{autocorr_lag}d** | "
    f"Weights: W1={w1} W2={w2} W3={w3} W4={w4} W5={w5} | "
    f"Hist pct: expanding window (no lookahead) | "
    f"Normalisation: cross-sec rank within Stage-1 universe"
)
