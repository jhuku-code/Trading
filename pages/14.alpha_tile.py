"""
Alpha Percentile vs BTC — Composite Score Edition
===================================================
Signal logic (Suggestion A):
  For each coin, compute 5 component scores (each 0–1), then combine into
  a weighted composite (0–5 scale).

  Components
  ──────────
  S1  Historical percentile   — expanding-window rank of today's alpha vs own history
  S2  Cross-sectional pct     — rank of today's alpha vs all peers
  S3  Δ X-sec pct (momentum)  — change in cross-sec pct over N days (clipped 0–1)
  S4  Divergence              — x-sec pct minus hist pct, normalised to 0–1
  S5  Distance from 30d peak  — how far below the 30-day x-sec high (inverted exhaustion)

  Weights are user-configurable via the sidebar.

Quality gating (from Code 2):
  Each rolling OLS window is accepted only if:
    • R² ≥ min_r2
    • |beta t-stat| ≥ min_beta_tstat
    • valid observations ≥ min_obs

Lookahead fix:
  Historical pct uses expanding().rank(pct=True) — no future data leaks in.
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
st.set_page_config(page_title="Alpha Percentile — Composite Score", layout="wide")

st.markdown("""
<style>
/* ── dark card ─────────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: #12121f;
    border: 1px solid #2a2a40;
    border-radius: 10px;
    padding: 14px 18px;
}
/* ── table header ───────────────────────────────────────────────────────── */
thead tr th { background: #12121f !important; }
/* ── sidebar section headers ───────────────────────────────────────────── */
section[data-testid="stSidebar"] h3 {
    color: #a78bfa;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 1.2rem;
}
/* ── score badge colours ─────────────────────────────────────────────────*/
.score-hi  { color: #4ade80; font-weight: 700; }
.score-mid { color: #facc15; font-weight: 600; }
.score-lo  { color: #f87171; }
</style>
""", unsafe_allow_html=True)

st.title("📐 Alpha Percentile vs BTC — Composite Score")
st.caption(
    "Five components, each scored 0–1, combined into a weighted composite (max 5). "
    "No hard gates — every coin gets a score; use the threshold slider to filter."
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

    # ── Component parameters ────────────────────────────────────────────────
    st.subheader("S3 — Momentum Lookback")
    momentum_lookback = st.slider(
        "Δ X-sec pct lookback (days)", 5, 30, 14, 1,
        help="How far back to measure the change in cross-sectional percentile."
    )

    st.subheader("S5 — Exhaustion Peak Window")
    peak_window = st.slider(
        "Rolling peak window (days)", 10, 60, 30, 5,
        help="Window for measuring the recent high in x-sec percentile (exhaustion gauge)."
    )

    # ── Weights ──────────────────────────────────────────────────────────────
    st.subheader("Component Weights")
    st.caption("Each weight scales its component's 0–1 score. Max composite = sum of all weights.")

    w1 = st.slider("W1 — Historical pct",    0.0, 2.0, 1.0, 0.1,
                   help="How good is today's alpha vs this coin's own history?")
    w2 = st.slider("W2 — Cross-sec pct",     0.0, 2.0, 1.0, 0.1,
                   help="How good is today's alpha vs all peers?")
    w3 = st.slider("W3 — Momentum (Δ pct)",  0.0, 2.0, 1.0, 0.1,
                   help="Is the cross-sec percentile improving?")
    w4 = st.slider("W4 — Divergence",        0.0, 2.0, 1.0, 0.1,
                   help="Is x-sec pct beating hist pct? (unusual outperformance vs own baseline)")
    w5 = st.slider("W5 — Not exhausted",     0.0, 2.0, 1.0, 0.1,
                   help="Is the coin still away from its recent peak? (0 = right at peak, 1 = far below)")

    max_possible = w1 + w2 + w3 + w4 + w5
    st.caption(f"Max composite score with current weights: **{max_possible:.1f}**")

    # ── Display filter ───────────────────────────────────────────────────────
    st.subheader("Display Filter")
    min_score_pct = st.slider(
        "Show coins with composite ≥ (% of max)", 0, 100, 40, 5,
        help="Filter the signal table to coins above this fraction of the max possible score."
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
    df_r   = pd.read_json(df_ret_json)
    btc_r  = pd.read_json(btc_ret_json, typ="series")
    coins  = [c for c in df_r.columns if c != "BTC"]
    n      = len(df_r)

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

            se2      = ss_res / max(n_obs - 2, 1)
            x_var    = np.sum((X.flatten() - X.mean()) ** 2)
            beta_se  = np.sqrt(se2 / x_var) if x_var > 0 else np.inf
            beta_val = model.coef_[0]
            b_tstat  = beta_val / beta_se if beta_se > 0 else 0.0

            if r2 < min_r2 or abs(b_tstat) < min_beta_tstat:
                continue

            alpha_d[coin][i] = model.intercept_ * 100
            r2_d[coin][i]    = round(r2, 4)
            tstat_d[coin][i] = round(b_tstat, 4)

    idx      = df_r.index
    df_alpha = pd.DataFrame(alpha_d, index=idx).round(4)
    df_r2    = pd.DataFrame(r2_d,    index=idx).round(4)
    df_ts    = pd.DataFrame(tstat_d, index=idx).round(4)

    total_w   = max(n - window + 1, 1)
    pass_rate = (df_alpha.notna().sum() / total_w).round(3)

    return df_alpha.to_json(), df_r2.to_json(), df_ts.to_json(), pass_rate.to_json()


alpha_json, r2_json, tstat_json, pass_rate_json = compute_rolling_alpha(
    df_ret.to_json(), btc_ret.to_json(),
    window, min_r2, min_beta_tstat, min_obs
)

df_alpha  = pd.read_json(alpha_json)
df_r2     = pd.read_json(r2_json)
df_tstat  = pd.read_json(tstat_json)
pass_rate = pd.read_json(pass_rate_json, typ="series")

# ─────────────────────────────────────────────────────────────────────────────
# PERCENTILES
# ─────────────────────────────────────────────────────────────────────────────

# S1: Historical — expanding window, no lookahead
df_hist_pct = df_alpha.apply(lambda x: x.expanding().rank(pct=True))

# S2: Cross-sectional — rank within each day across coins
df_xsec_pct = df_alpha.rank(axis=1, pct=True)

# ─────────────────────────────────────────────────────────────────────────────
# DERIVED SERIES
# ─────────────────────────────────────────────────────────────────────────────

# S3: Momentum — change in x-sec pct over lookback
df_momentum = df_xsec_pct - df_xsec_pct.shift(momentum_lookback)

# S4: Divergence — x-sec pct minus hist pct
df_divergence = df_xsec_pct - df_hist_pct

# S5: Distance from rolling peak (exhaustion) — positive = below peak (not exhausted)
df_peak      = df_xsec_pct.rolling(peak_window).max()
df_dist_peak = df_peak - df_xsec_pct   # high value = far below peak = NOT exhausted

# ─────────────────────────────────────────────────────────────────────────────
# NORMALISE COMPONENTS TO 0–1 USING CROSS-SECTIONAL RANK ON EACH DAY
# ─────────────────────────────────────────────────────────────────────────────
# Each component is percentile-ranked cross-sectionally so that the score
# reflects relative standing among peers, not raw magnitudes.
# This makes the weights directly comparable.

def cs_rank(df):
    """Cross-sectional rank each row to [0, 1]."""
    return df.rank(axis=1, pct=True)


n1 = cs_rank(df_hist_pct)     # S1 — already a pct, but re-rank for consistency
n2 = cs_rank(df_xsec_pct)     # S2
n3 = cs_rank(df_momentum)     # S3
n4 = cs_rank(df_divergence)   # S4
n5 = cs_rank(df_dist_peak)    # S5  (high rank = far from peak = not exhausted)

# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE SCORE
# ─────────────────────────────────────────────────────────────────────────────
df_composite = (
    w1 * n1 +
    w2 * n2 +
    w3 * n3 +
    w4 * n4 +
    w5 * n5
)

# ─────────────────────────────────────────────────────────────────────────────
# AUTOCORRELATION OF X-SEC PCT (diagnostic)
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
        s_now = s.iloc[lag:].values
        s_lag = s.iloc[:-lag].values
        rho, _ = scipy_stats.spearmanr(s_now, s_lag)
        out[coin] = round(rho, 3)
    return pd.Series(out).round(3)


autocorr_series = compute_autocorr(df_xsec_pct.to_json(), autocorr_lag)

# ─────────────────────────────────────────────────────────────────────────────
# TODAY'S SNAPSHOT
# ─────────────────────────────────────────────────────────────────────────────
last_s1        = df_hist_pct.iloc[-1]
last_s2        = df_xsec_pct.iloc[-1]
last_s3        = df_momentum.iloc[-1]
last_s4        = df_divergence.iloc[-1]
last_s5        = df_dist_peak.iloc[-1]
last_n1        = n1.iloc[-1]
last_n2        = n2.iloc[-1]
last_n3        = n3.iloc[-1]
last_n4        = n4.iloc[-1]
last_n5        = n5.iloc[-1]
last_composite = df_composite.iloc[-1]

# ─────────────────────────────────────────────────────────────────────────────
# BUILD SIGNAL TABLE
# ─────────────────────────────────────────────────────────────────────────────
rows = []
for coin in all_coins:
    s1 = last_s1.get(coin, np.nan)
    s2 = last_s2.get(coin, np.nan)
    s3 = last_s3.get(coin, np.nan)
    s4 = last_s4.get(coin, np.nan)
    s5 = last_s5.get(coin, np.nan)

    nn1 = last_n1.get(coin, np.nan)
    nn2 = last_n2.get(coin, np.nan)
    nn3 = last_n3.get(coin, np.nan)
    nn4 = last_n4.get(coin, np.nan)
    nn5 = last_n5.get(coin, np.nan)
    comp = last_composite.get(coin, np.nan)

    if pd.isna(comp):
        continue

    ac = autocorr_series.get(coin, np.nan)

    rows.append({
        "Coin":                            coin,
        # Raw component values
        "S1 Hist Pct":                     round(s1,   3),
        "S2 X-sec Pct":                    round(s2,   3),
        f"S3 Δ Pct ({momentum_lookback}d)": round(s3,   3),
        "S4 Divergence":                   round(s4,   3),
        f"S5 Dist Peak ({peak_window}d)":  round(s5,   3),
        # Normalised scores (0–1 cs-rank)
        "N1":                              round(nn1,  3),
        "N2":                              round(nn2,  3),
        "N3":                              round(nn3,  3),
        "N4":                              round(nn4,  3),
        "N5":                              round(nn5,  3),
        # Composite
        "Composite":                       round(comp, 3),
        "Composite %":                     round(comp / max_possible, 3) if max_possible > 0 else np.nan,
        # Autocorrelation
        f"Autocorr (lag={autocorr_lag}d)": round(ac, 3) if not pd.isna(ac) else np.nan,
    })

df_signal = (
    pd.DataFrame(rows)
    .sort_values("Composite", ascending=False)
    .reset_index(drop=True)
)

df_signal_filtered = df_signal[df_signal["Composite"] >= min_score_abs].copy()

# ─────────────────────────────────────────────────────────────────────────────
# TOP METRICS BANNER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")

n_total      = len(all_coins)
n_valid      = df_alpha.iloc[-1].notna().sum()
n_above_thr  = len(df_signal_filtered)
avg_pass     = pass_rate.mean()
top_coin     = df_signal.iloc[0]["Coin"] if len(df_signal) > 0 else "—"
top_score    = df_signal.iloc[0]["Composite"] if len(df_signal) > 0 else np.nan

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Total coins", n_total)
with c2:
    st.metric("Valid alpha today", n_valid,
              delta=f"{n_valid/n_total:.0%} pass QC",
              delta_color="normal" if n_valid > n_total * 0.5 else "inverse")
with c3:
    st.metric("Coins above threshold", n_above_thr)
with c4:
    st.metric("Avg QC pass rate", f"{avg_pass:.1%}")
with c5:
    st.metric(
        "Top coin today",
        top_coin,
        delta=f"{top_score:.2f} / {max_possible:.1f}" if not pd.isna(top_score) else "—"
    )

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

    def _c_ac(v):
        if pd.isna(v): return "color:grey"
        return "color:#4ade80" if v >= 0.4 else ("color:#facc15" if v >= 0.2 else "color:#f87171")

    st.dataframe(
        diag_df.style
            .applymap(_c_pass, subset=["Pass Rate (QC)"])
            .applymap(_c_ac,   subset=[f"Autocorr (lag={autocorr_lag}d)"])
            .format({
                "Pass Rate (QC)":                  "{:.1%}",
                "Alpha today (×100)":              "{:.4f}",
                "R² today":                        "{:.3f}",
                "Beta t-stat today":               "{:.2f}",
                f"Autocorr (lag={autocorr_lag}d)": "{:.3f}",
            }, na_rep="—"),
        use_container_width=True, height=340
    )

    col_qca, col_qcb = st.columns(2)
    with col_qca:
        fig_pr = go.Figure(go.Histogram(
            x=pass_rate.values, nbinsx=20,
            marker_color="#6366f1", opacity=0.85
        ))
        fig_pr.add_vline(x=0.3, line_dash="dash", line_color="#f87171",
                         annotation_text="30%", annotation_position="top right")
        fig_pr.add_vline(x=0.6, line_dash="dash", line_color="#4ade80",
                         annotation_text="60%", annotation_position="top right")
        fig_pr.update_layout(
            title="QC Pass Rate Distribution",
            xaxis_title="Pass Rate", yaxis_title="# Coins",
            height=260, template="plotly_dark"
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    with col_qcb:
        fig_ac = go.Figure(go.Histogram(
            x=autocorr_series.dropna().values, nbinsx=20,
            marker_color="#a78bfa", opacity=0.85
        ))
        fig_ac.add_vline(x=0, line_dash="dot", line_color="white", opacity=0.4)
        fig_ac.update_layout(
            title=f"Autocorr Distribution (lag={autocorr_lag}d)",
            xaxis_title="Spearman rho", yaxis_title="# Coins",
            height=260, template="plotly_dark"
        )
        st.plotly_chart(fig_ac, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE SCORE TABLE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Composite Score Table")

st.markdown(f"""
| Component | Raw metric | What it captures | Weight |
|---|---|---|---|
| **S1 Hist Pct** | Expanding-window rank of today's alpha vs coin's own history | Good alpha vs own history | **{w1}** |
| **S2 X-sec Pct** | Rank of today's alpha vs all peers | Winning the room today | **{w2}** |
| **S3 Momentum** | Δ x-sec pct over {momentum_lookback}d | Alpha rank is rising | **{w3}** |
| **S4 Divergence** | X-sec pct minus hist pct | Unusually good vs peers relative to own baseline | **{w4}** |
| **S5 Not Exhausted** | Distance below {peak_window}d x-sec peak | Room to run (not crowded/peaked) | **{w5}** |

*Each component is normalised to 0–1 via cross-sectional rank before weighting.*
Max possible composite = **{max_possible:.1f}** &nbsp;|&nbsp; Showing coins with composite ≥ **{min_score_abs:.2f}** ({min_score_pct}% of max)
""")

# Callout for top coins
if not df_signal_filtered.empty:
    top5 = df_signal_filtered.head(5)["Coin"].tolist()
    st.success(f"🏆 Top coins: **{' · '.join(top5)}**")

# Colour helpers for the table
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

def _colour_ac(v):
    if pd.isna(v): return "color:grey"
    return "color:#4ade80" if v >= 0.4 else ("color:#facc15" if v >= 0.2 else "color:#f87171")

def highlight_top(row):
    frac = row["Composite"] / max_possible if max_possible > 0 else 0
    if frac >= 0.70:
        return ["background-color:#14532d"] * len(row)
    if frac >= 0.50:
        return ["background-color:#1e3a5f"] * len(row)
    return [""] * len(row)


tab_full, tab_raw, tab_norm = st.tabs([
    "🎯 Full table (raw + scores)",
    "📋 Raw components only",
    "🔢 Normalised scores only"
])

with tab_full:
    st.dataframe(
        df_signal_filtered.style
            .apply(highlight_top, axis=1)
            .applymap(_colour_norm,  subset=score_cols)
            .applymap(_colour_comp,  subset=["Composite"])
            .applymap(_colour_ac,    subset=[f"Autocorr (lag={autocorr_lag}d)"])
            .format({
                "S1 Hist Pct":                      "{:.3f}",
                "S2 X-sec Pct":                     "{:.3f}",
                f"S3 Δ Pct ({momentum_lookback}d)": "{:.3f}",
                "S4 Divergence":                    "{:.3f}",
                f"S5 Dist Peak ({peak_window}d)":   "{:.3f}",
                "N1": "{:.3f}", "N2": "{:.3f}", "N3": "{:.3f}",
                "N4": "{:.3f}", "N5": "{:.3f}",
                "Composite":   "{:.3f}",
                "Composite %": "{:.1%}",
                f"Autocorr (lag={autocorr_lag}d)": "{:.3f}",
            }, na_rep="—"),
        use_container_width=True, height=500
    )

with tab_raw:
    raw_cols = ["Coin", "S1 Hist Pct", "S2 X-sec Pct",
                f"S3 Δ Pct ({momentum_lookback}d)", "S4 Divergence",
                f"S5 Dist Peak ({peak_window}d)", "Composite"]
    st.dataframe(
        df_signal_filtered[raw_cols].style
            .apply(highlight_top, axis=1)
            .applymap(_colour_comp, subset=["Composite"])
            .format({c: "{:.3f}" for c in raw_cols if c != "Coin"}, na_rep="—"),
        use_container_width=True, height=500
    )

with tab_norm:
    norm_cols = ["Coin", "N1", "N2", "N3", "N4", "N5", "Composite", "Composite %"]
    st.dataframe(
        df_signal_filtered[norm_cols].style
            .apply(highlight_top, axis=1)
            .applymap(_colour_norm, subset=["N1","N2","N3","N4","N5"])
            .applymap(_colour_comp, subset=["Composite"])
            .format({
                "N1": "{:.3f}", "N2": "{:.3f}", "N3": "{:.3f}",
                "N4": "{:.3f}", "N5": "{:.3f}",
                "Composite":   "{:.3f}",
                "Composite %": "{:.1%}",
            }, na_rep="—"),
        use_container_width=True, height=500
    )

# ─────────────────────────────────────────────────────────────────────────────
# SCORE DISTRIBUTION CHART
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📈 Score Distribution & Component Breakdown")

col_dist, col_radar = st.columns([1.6, 1])

with col_dist:
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=df_signal["Composite"].dropna().values,
        nbinsx=30,
        marker_color="#6366f1",
        opacity=0.85,
        name="All coins"
    ))
    fig_dist.add_vline(
        x=min_score_abs,
        line_dash="dash", line_color="#facc15",
        annotation_text=f"threshold={min_score_abs:.2f}",
        annotation_position="top right"
    )
    if not df_signal_filtered.empty:
        fig_dist.add_vline(
            x=df_signal.iloc[0]["Composite"],
            line_dash="dot", line_color="#4ade80",
            annotation_text=f"top: {top_coin}",
            annotation_position="top left"
        )
    fig_dist.update_layout(
        title="Composite Score Distribution (all coins today)",
        xaxis_title="Composite Score",
        yaxis_title="# Coins",
        height=300,
        template="plotly_dark"
    )
    st.plotly_chart(fig_dist, use_container_width=True)

with col_radar:
    # Show radar for top 3 coins
    top3 = df_signal_filtered.head(3)
    if not top3.empty:
        categories = ["Hist Pct", "X-sec Pct", "Momentum", "Divergence", "Not Exhausted"]
        fig_radar  = go.Figure()
        for _, row in top3.iterrows():
            vals = [row["N1"], row["N2"], row["N3"], row["N4"], row["N5"]]
            vals_closed = vals + [vals[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals_closed,
                theta=categories + [categories[0]],
                fill="toself",
                name=row["Coin"],
                opacity=0.7
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Top 3 coins — component profile",
            height=300,
            template="plotly_dark",
            legend=dict(orientation="h", y=-0.25)
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.info("No coins above threshold — lower the threshold to see radar.")

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

    # ── Chart 1: Raw alpha ────────────────────────────────────────────────────
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
        height=320, template="plotly_dark",
        legend=dict(orientation="h", y=-0.30)
    )
    st.plotly_chart(fig_a, use_container_width=True)

    # ── Chart 2: Five components over time ───────────────────────────────────
    fig_comp = make_subplots(
        rows=5, cols=1, shared_xaxes=True,
        subplot_titles=[
            "S1 — Historical pct (expanding)",
            "S2 — Cross-sec pct",
            f"S3 — Momentum (Δ x-sec pct, {momentum_lookback}d)",
            "S4 — Divergence (x-sec − hist)",
            f"S5 — Distance from {peak_window}d peak (not exhausted)"
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

    # Reference lines
    for row_ in [1, 2]:
        fig_comp.add_hline(y=0.5, line_dash="dot", line_color="white",
                           opacity=0.25, row=row_, col=1)
    for row_ in [3, 4, 5]:
        fig_comp.add_hline(y=0, line_dash="dot", line_color="white",
                           opacity=0.25, row=row_, col=1)

    fig_comp.update_layout(
        height=900, template="plotly_dark",
        legend=dict(orientation="h", y=-0.04),
        title_text="Five Components Over Time"
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # ── Chart 3: Composite score over time ───────────────────────────────────
    fig_cs = go.Figure()
    for coin in coins_to_plot:
        if coin in df_composite.columns:
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
        title="Rolling Composite Score Over Time",
        xaxis_title="Date", yaxis_title="Composite Score",
        height=320, template="plotly_dark",
        legend=dict(orientation="h", y=-0.30)
    )
    st.plotly_chart(fig_cs, use_container_width=True)

    # ── Chart 4: R² + beta t-stat ─────────────────────────────────────────────
    with st.expander("🔍 Regression quality over time (R² & beta t-stat)"):
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
        fig_qc.update_layout(
            height=420, template="plotly_dark",
            legend=dict(orientation="h", y=-0.12)
        )
        st.plotly_chart(fig_qc, use_container_width=True)

else:
    st.info("Select at least one coin above to view charts.")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"Window: **{window}d** | Min R²: **{min_r2}** | Min |β t-stat|: **{min_beta_tstat}** | "
    f"Min obs: **{min_obs}** | Momentum lookback: **{momentum_lookback}d** | "
    f"Peak window: **{peak_window}d** | Autocorr lag: **{autocorr_lag}d** | "
    f"Hist pct: expanding window (no lookahead) | "
    f"Weights: W1={w1} W2={w2} W3={w3} W4={w4} W5={w5}"
)
