import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="Trend Following (BTC-Relative)", layout="wide")
st.title("📈 Trend Following — Coin / BTC Pairs")

if st.button("🔄 Refresh data"):
    st.cache_data.clear()
    st.rerun()

# ─────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────
df_raw = st.session_state.get("price_theme", None)

if df_raw is None:
    st.error(
        "`price_theme` not found in session_state. "
        "Please load prices into `st.session_state['price_theme']` first."
    )
    st.stop()

if "BTC" not in df_raw.columns:
    st.error("BTC column not found in `price_theme`. BTC is required as the base asset.")
    st.stop()

df_raw = df_raw.sort_index().copy()

# ─────────────────────────────────────────────────────────
# DETECT BAR FREQUENCY
# ─────────────────────────────────────────────────────────
def detect_bar_freq(index: pd.DatetimeIndex):
    """
    Returns (bars_per_day, label) by inspecting median gap between rows.
    Works for daily, 4H, 1H, 30m, 15m data.
    """
    if len(index) < 3:
        return 1, "1D"
    diffs = pd.Series(index).diff().dropna()
    median_min = diffs.median().total_seconds() / 60
    if median_min >= 1400:
        return 1,   "Daily"
    elif median_min >= 700:
        return 2,   "12H"
    elif median_min >= 350:
        return 4,   "4H"
    elif median_min >= 170:
        return 8,   "3H"
    elif median_min >= 55:
        return 24,  "1H"
    elif median_min >= 25:
        return 48,  "30m"
    elif median_min >= 13:
        return 96,  "15m"
    else:
        return 1,   "Unknown"

bars_per_day, freq_label = detect_bar_freq(df_raw.index)

# ─────────────────────────────────────────────────────────
# SIDEBAR — USER-DRIVEN HORIZON PARAMETERS
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Parameters")
    st.caption(f"Detected bar frequency: **{freq_label}** ({bars_per_day} bars/day)")

    st.subheader("Investment Horizons (days)")
    short_days  = st.slider("Short horizon (days)",  min_value=1,  max_value=30,  value=7,  step=1)
    medium_days = st.slider("Medium horizon (days)", min_value=5,  max_value=90,  value=30, step=1)
    long_days   = st.slider("Long horizon (days)",   min_value=10, max_value=365, value=90, step=1)

    st.subheader("Signal Parameters (days)")
    aema_days = st.slider("Adaptive EMA period (days)",   min_value=3,  max_value=60, value=15, step=1)
    jma_days  = st.slider("JMA / FATL period (days)",     min_value=3,  max_value=60, value=15, step=1)
    nw_days   = st.slider("Nadaraya-Watson window (days)", min_value=5,  max_value=90, value=20, step=1)
    nw_r      = st.slider("NW bandwidth (r)",              min_value=1.0, max_value=200.0, value=48.0, step=1.0)
    jma_phase = st.slider("JMA phase",                    min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    st.subheader("Signal Weights")
    w_aema = st.slider("Adaptive EMA weight", 0.0, 1.0, 1.0, 0.1)
    w_jma  = st.slider("JMA/FATL weight",     0.0, 1.0, 1.0, 0.1)
    w_nw   = st.slider("Nadaraya-Watson weight", 0.0, 1.0, 1.0, 0.1)

    st.subheader("Filter")
    min_composite = st.slider(
        "Min composite score to show",
        min_value=-3.0, max_value=3.0, value=0.0, step=0.5,
        help="Negative = show sell signals too. 3 = strongest buys only."
    )

# Convert day-based params to bars
def days_to_bars(d):
    return max(2, int(round(d * bars_per_day)))

short_bars  = days_to_bars(short_days)
medium_bars = days_to_bars(medium_days)
long_bars   = days_to_bars(long_days)
aema_bars   = days_to_bars(aema_days)
jma_bars    = days_to_bars(jma_days)
nw_bars     = days_to_bars(nw_days)

# ─────────────────────────────────────────────────────────
# COMPUTE COIN / BTC RATIO PRICES
# ─────────────────────────────────────────────────────────
btc = df_raw["BTC"].replace(0, np.nan)
alt_cols = [c for c in df_raw.columns if c != "BTC"]
df_btc_rel = df_raw[alt_cols].div(btc, axis=0)  # coin/BTC ratio for every alt

# Drop columns with insufficient data (need at least nw_bars + 20 rows)
min_rows = max(aema_bars, jma_bars, nw_bars) + 50
valid_cols = [c for c in df_btc_rel.columns if df_btc_rel[c].dropna().shape[0] >= min_rows]
df_btc_rel = df_btc_rel[valid_cols]

# ─────────────────────────────────────────────────────────
# SIGNAL FUNCTIONS
# ─────────────────────────────────────────────────────────
def adaptive_ema(series: pd.Series, period: int) -> pd.Series:
    """Kaufman-style Adaptive EMA. Returns smoothed series."""
    vals = series.values.copy().astype(float)
    result = vals.copy()
    noise = 0.0
    for i in range(period, len(vals)):
        if np.isnan(vals[i]) or np.isnan(vals[i - period]):
            result[i] = result[i - 1] if not np.isnan(result[i - 1]) else np.nan
            continue
        sig = abs(vals[i] - vals[i - period])
        noise += abs(vals[i] - vals[i - 1]) - abs(vals[i] - vals[i - period])
        noise_val = max(abs(noise), 1e-10)
        er = sig / noise_val
        fast_sc = 2.0 / (2.0 + 1)
        slow_sc = 2.0 / (30.0 + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        result[i] = result[i - 1] + sc * (vals[i] - result[i - 1])
    return pd.Series(result, index=series.index)


def jfatl_hybrid(series: pd.Series, fatl_len: int, jma_len: int, phase: float) -> pd.Series:
    """FATL (triangular MA) smoothed with a JMA-like weighted combination."""
    fatl = series.rolling(fatl_len, min_periods=fatl_len // 2).mean()
    e = 0.5 * (phase + 1)
    wma1 = fatl.ewm(span=jma_len, adjust=False).mean()
    wma2 = fatl.ewm(span=max(jma_len // 2, 2), adjust=False).mean()
    return wma1 * e + wma2 * (1.0 - e)


def nadaraya_watson(series: pd.Series, h: int, r: float) -> pd.Series:
    """
    Rational-quadratic kernel Nadaraya-Watson smoother.
    Uses only past + current values (causal, no look-ahead).
    """
    n = len(series)
    vals = series.values.astype(float)
    smoothed = np.full(n, np.nan)
    for t in range(h, n):
        window_start = max(0, t - h * 3)   # limit window for speed
        indices = np.arange(window_start, t + 1)
        distances = t - indices
        weights = (1.0 + distances**2 / (h**2 * 2.0 * r)) ** (-r)
        v = vals[window_start : t + 1]
        mask = ~np.isnan(v)
        if mask.sum() < 2:
            continue
        smoothed[t] = np.sum(v[mask] * weights[mask]) / np.sum(weights[mask])
    return pd.Series(smoothed, index=series.index)


def signal_direction(smooth: pd.Series) -> int:
    """
    Returns +1 if smoother is trending up (last vs prev), -1 down, 0 flat.
    Uses last two valid values.
    """
    valid = smooth.dropna()
    if len(valid) < 2:
        return 0
    delta = valid.iloc[-1] - valid.iloc[-2]
    if delta > 0:
        return 1
    elif delta < 0:
        return -1
    return 0


def percentile_rank(series: pd.Series) -> float:
    """
    Returns where the last value sits in the full historical range [0, 100].
    0 = at all-time low, 100 = at all-time high in the dataset.
    """
    valid = series.dropna()
    if len(valid) < 2:
        return np.nan
    lo, hi = valid.min(), valid.max()
    if hi == lo:
        return 50.0
    return (valid.iloc[-1] - lo) / (hi - lo) * 100.0


# ─────────────────────────────────────────────────────────
# COMPUTE SIGNALS FOR ALL COINS
# ─────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner="Computing signals…")
def compute_signals(df_rel: pd.DataFrame,
                    aema_bars: int, jma_bars: int, nw_bars: int, nw_r: float,
                    jma_phase: float,
                    w_aema: float, w_jma: float, w_nw: float,
                    short_bars: int, medium_bars: int, long_bars: int) -> pd.DataFrame:
    rows = []
    total_w = (w_aema + w_jma + w_nw) or 1.0

    for coin in df_rel.columns:
        s = df_rel[coin].dropna()
        if len(s) < max(aema_bars, jma_bars, nw_bars) + 10:
            continue

        # --- Three smoothers ---
        aema_line  = adaptive_ema(s, aema_bars)
        jfatl_line = jfatl_hybrid(s, jma_bars, jma_bars, jma_phase)
        nw_line    = nadaraya_watson(s, nw_bars, nw_r)

        # --- Raw direction signals (-1, 0, +1) ---
        sig_aema  = signal_direction(aema_line)
        sig_jfatl = signal_direction(jfatl_line)
        sig_nw    = signal_direction(nw_line)

        # --- Weighted composite ---
        composite = (sig_aema * w_aema + sig_jfatl * w_jma + sig_nw * w_nw) / total_w

        # --- Horizon returns ---
        last_val = s.iloc[-1]
        ret_short  = (last_val / s.iloc[-short_bars]  - 1) * 100 if len(s) > short_bars  else np.nan
        ret_medium = (last_val / s.iloc[-medium_bars] - 1) * 100 if len(s) > medium_bars else np.nan
        ret_long   = (last_val / s.iloc[-long_bars]   - 1) * 100 if len(s) > long_bars   else np.nan

        # --- Percentile rank in full history ---
        pct_rank = percentile_rank(s)

        rows.append({
            "Coin": coin,
            "Composite": round(composite, 3),
            "AEMA Signal": sig_aema,
            "JMA/FATL Signal": sig_jfatl,
            "NW Signal": sig_nw,
            f"{short_days}D Return (vs BTC %)": round(ret_short, 2)  if not np.isnan(ret_short)  else np.nan,
            f"{medium_days}D Return (vs BTC %)": round(ret_medium, 2) if not np.isnan(ret_medium) else np.nan,
            f"{long_days}D Return (vs BTC %)": round(ret_long, 2)   if not np.isnan(ret_long)   else np.nan,
            "Historical Rank (%)": round(pct_rank, 1),
            "Current Ratio": round(last_val, 6),
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.sort_values("Composite", ascending=False).reset_index(drop=True)
    return result


df_signals = compute_signals(
    df_btc_rel,
    aema_bars, jma_bars, nw_bars, nw_r,
    jma_phase,
    w_aema, w_jma, w_nw,
    short_bars, medium_bars, long_bars,
)

# ─────────────────────────────────────────────────────────
# SIGNAL LABEL HELPERS
# ─────────────────────────────────────────────────────────
def composite_label(score):
    if score >= 0.9:   return "🟢 Strong Buy"
    elif score >= 0.4: return "🟡 Weak Buy"
    elif score <= -0.9: return "🔴 Strong Sell"
    elif score <= -0.4: return "🟠 Weak Sell"
    else:               return "⚪ Neutral"

def pct_rank_label(pct, composite):
    """Contextualise the historical rank relative to signal direction."""
    if np.isnan(pct):
        return "N/A"
    if composite > 0:
        # Buy signal — how far has the rally come?
        if pct >= 80:   return f"{pct:.0f}% — Extended (near highs)"
        elif pct >= 50: return f"{pct:.0f}% — Mid-range"
        else:           return f"{pct:.0f}% — Early (room to run)"
    elif composite < 0:
        # Sell signal — how far has the selloff come?
        if pct <= 20:   return f"{pct:.0f}% — Extended (near lows)"
        elif pct <= 50: return f"{pct:.0f}% — Mid-range"
        else:           return f"{pct:.0f}% — Early selloff"
    return f"{pct:.0f}%"

# ─────────────────────────────────────────────────────────
# FILTER & DISPLAY
# ─────────────────────────────────────────────────────────
st.markdown(f"**Data frequency detected:** `{freq_label}` · "
            f"Periods — Short: `{short_days}d` ({short_bars} bars), "
            f"Medium: `{medium_days}d` ({medium_bars} bars), "
            f"Long: `{long_days}d` ({long_bars} bars)")

if df_signals.empty:
    st.warning("No signals computed. Check that price_theme has sufficient history.")
    st.stop()

df_filtered = df_signals[df_signals["Composite"] >= min_composite].copy()
df_filtered.insert(1, "Signal", df_filtered["Composite"].apply(composite_label))
df_filtered["Historical Rank"] = df_filtered.apply(
    lambda r: pct_rank_label(r["Historical Rank (%)"], r["Composite"]), axis=1
)

# ─────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────
tab_buy, tab_sell, tab_all = st.tabs(["🟢 Buy Signals", "🔴 Sell Signals", "📊 Full Ranking"])

def render_signal_table(df_sub: pd.DataFrame, signal_type: str):
    if df_sub.empty:
        st.info(f"No {signal_type} signals with current parameters.")
        return

    display_cols = [
        "Coin", "Signal", "Composite",
        "AEMA Signal", "JMA/FATL Signal", "NW Signal",
        f"{short_days}D Return (vs BTC %)",
        f"{medium_days}D Return (vs BTC %)",
        f"{long_days}D Return (vs BTC %)",
        "Historical Rank",
        "Historical Rank (%)",
    ]
    display_cols = [c for c in display_cols if c in df_sub.columns]

    styled = df_sub[display_cols].style.background_gradient(
        subset=["Composite"],
        cmap="RdYlGn",
        vmin=-1, vmax=1,
    ).background_gradient(
        subset=["Historical Rank (%)"],
        cmap="RdYlGn",
        vmin=0, vmax=100,
    ).format({
        "Composite": "{:.2f}",
        "Historical Rank (%)": "{:.1f}%",
        f"{short_days}D Return (vs BTC %)": "{:+.2f}%",
        f"{medium_days}D Return (vs BTC %)": "{:+.2f}%",
        f"{long_days}D Return (vs BTC %)": "{:+.2f}%",
    })

    st.dataframe(styled, use_container_width=True, height=min(600, 60 + len(df_sub) * 38))

with tab_buy:
    st.subheader(f"Buy Signals  (Composite > 0)")
    buys = df_filtered[df_filtered["Composite"] > 0]
    st.caption(
        "**Historical Rank** shows where the Coin/BTC ratio sits in its full history. "
        "High rank (near 100%) on a buy signal means the rally is extended; "
        "low rank means early-stage and more room to run."
    )
    render_signal_table(buys, "buy")

with tab_sell:
    st.subheader(f"Sell Signals  (Composite < 0)")
    sells = df_filtered[df_filtered["Composite"] < 0]
    st.caption(
        "**Historical Rank** shows where the Coin/BTC ratio sits in its full history. "
        "Low rank (near 0%) on a sell signal means the selloff is extended; "
        "high rank means early-stage and more to fall."
    )
    render_signal_table(sells, "sell")

with tab_all:
    st.subheader("Full Universe Ranking (by Composite Score)")
    render_signal_table(df_filtered, "")

# ─────────────────────────────────────────────────────────
# SIGNAL DISTRIBUTION CHART
# ─────────────────────────────────────────────────────────
st.subheader("Signal Distribution")
col1, col2 = st.columns(2)

with col1:
    counts = {
        "Strong Buy (≥0.9)":   (df_signals["Composite"] >= 0.9).sum(),
        "Weak Buy (0–0.9)":    ((df_signals["Composite"] > 0) & (df_signals["Composite"] < 0.9)).sum(),
        "Neutral":              (df_signals["Composite"] == 0).sum(),
        "Weak Sell (-0.9–0)":  ((df_signals["Composite"] < 0) & (df_signals["Composite"] > -0.9)).sum(),
        "Strong Sell (≤-0.9)": (df_signals["Composite"] <= -0.9).sum(),
    }
    fig_dist = go.Figure(go.Bar(
        x=list(counts.keys()),
        y=list(counts.values()),
        marker_color=["#22c55e", "#86efac", "#94a3b8", "#fca5a5", "#ef4444"],
        text=list(counts.values()),
        textposition="auto",
    ))
    fig_dist.update_layout(
        title="Signal Count by Category",
        height=350,
        margin=dict(t=40, b=40),
        xaxis_tickangle=-20,
    )
    st.plotly_chart(fig_dist, use_container_width=True)

with col2:
    # Historical rank scatter for buy signals
    buys_all = df_signals[df_signals["Composite"] > 0].copy()
    if not buys_all.empty:
        fig_rank = go.Figure(go.Scatter(
            x=buys_all["Composite"],
            y=buys_all["Historical Rank (%)"],
            mode="markers+text",
            text=buys_all["Coin"],
            textposition="top center",
            marker=dict(
                color=buys_all["Historical Rank (%)"],
                colorscale="RdYlGn_r",
                size=10,
                colorbar=dict(title="Rank %"),
                cmin=0, cmax=100,
            ),
        ))
        fig_rank.update_layout(
            title="Buy Signals: Signal Strength vs Historical Rank",
            xaxis_title="Composite Score (signal strength)",
            yaxis_title="Historical Rank (%) — higher = closer to all-time high",
            height=350,
            margin=dict(t=40, b=40),
        )
        fig_rank.add_hline(y=80, line_dash="dash", line_color="red",
                           annotation_text="Extended (80%)", annotation_position="right")
        fig_rank.add_hline(y=50, line_dash="dot", line_color="gray",
                           annotation_text="Mid-range (50%)", annotation_position="right")
        st.plotly_chart(fig_rank, use_container_width=True)
    else:
        st.info("No buy signals to scatter-plot.")

# ─────────────────────────────────────────────────────────
# INDIVIDUAL COIN CHART (optional deep-dive)
# ─────────────────────────────────────────────────────────
st.subheader("🔍 Coin / BTC Chart")
coin_choices = df_signals["Coin"].tolist()
selected_coin = st.selectbox("Select coin to chart (Coin/BTC ratio)", coin_choices)

if selected_coin and selected_coin in df_btc_rel.columns:
    s = df_btc_rel[selected_coin].dropna()

    aema_line  = adaptive_ema(s, aema_bars)
    jfatl_line = jfatl_hybrid(s, jma_bars, jma_bars, jma_phase)
    nw_line    = nadaraya_watson(s, nw_bars, nw_r)

    # Limit chart to last 500 bars for performance
    plot_n = min(500, len(s))
    s_p      = s.iloc[-plot_n:]
    aema_p   = aema_line.iloc[-plot_n:]
    jfatl_p  = jfatl_line.iloc[-plot_n:]
    nw_p     = nw_line.iloc[-plot_n:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s_p.index, y=s_p.values, name="Coin/BTC", line=dict(color="#94a3b8", width=1)))
    fig.add_trace(go.Scatter(x=aema_p.index, y=aema_p.values, name="Adaptive EMA", line=dict(color="#3b82f6", width=1.5)))
    fig.add_trace(go.Scatter(x=jfatl_p.index, y=jfatl_p.values, name="JMA/FATL", line=dict(color="#f59e0b", width=1.5)))
    fig.add_trace(go.Scatter(x=nw_p.index, y=nw_p.values, name="Nadaraya-Watson", line=dict(color="#22c55e", width=1.5, dash="dot")))

    row_data = df_signals[df_signals["Coin"] == selected_coin]
    if not row_data.empty:
        composite = row_data["Composite"].iloc[0]
        pct = row_data["Historical Rank (%)"].iloc[0]
        fig.update_layout(
            title=f"{selected_coin}/BTC  |  Composite: {composite:+.2f}  |  Historical Rank: {pct:.1f}%",
        )

    fig.update_layout(
        height=420,
        margin=dict(t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────
# METHODOLOGY NOTES
# ─────────────────────────────────────────────────────────
with st.expander("📖 Methodology"):
    st.markdown(f"""
**Data**: All coins are expressed as `Coin / BTC` price ratios. Signals identify coins gaining vs Bitcoin, not just vs USD.

**Bar frequency**: Auto-detected as **{freq_label}** ({bars_per_day} bars/day). All period inputs (days) are converted to bars accordingly.

**Three signals** (each scores −1, 0, or +1):
1. **Adaptive EMA** — Kaufman-style efficiency-ratio EMA. Direction = slope of last two values.
2. **JMA / FATL Hybrid** — Triangular MA (FATL) followed by dual EWM smoothing with phase weighting. Direction = slope.
3. **Nadaraya-Watson** — Rational-quadratic kernel regression, causal window. Direction = slope.

**Composite score** = weighted average of the three signals. Range [−1, +1].
- ≥ +0.9 → Strong Buy (all three agree)
- ≥ +0.4 → Weak Buy
- ≤ −0.9 → Strong Sell
- ≤ −0.4 → Weak Sell

**Historical Rank (%)**: Where the current Coin/BTC ratio sits between its all-time low (0%) and all-time high (100%) *within the loaded dataset*.  
- **Buy signals**: High rank = rally already extended. Low rank = early stage.  
- **Sell signals**: Low rank = selloff already extended. High rank = early stage.

**Horizon returns** are also expressed in Coin/BTC terms (relative to BTC), not raw USD.
    """)
