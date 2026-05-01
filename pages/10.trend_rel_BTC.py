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

# ─────────────────────────────────────────────────────────
# BACKTESTING ANALYSIS
# ─────────────────────────────────────────────────────────
st.markdown("---")
st.header("🧪 Signal Backtesting Analysis")
st.caption(
    "Historical signal performance is evaluated **in-sample** using a rolling look-back. "
    "For each historical bar, the composite signal is re-computed on data available up to that point, "
    "then forward returns (vs BTC) are measured `fwd_bars` bars ahead."
)

# ─── Sidebar additions ───────────────────────────────────
with st.sidebar:
    st.subheader("Backtest Parameters")
    fwd_days = st.slider(
        "Forward return window (days)", min_value=1, max_value=30, value=10, step=1,
        help="How many bars ahead to measure outcome returns."
    )
    bt_top_n = st.slider(
        "Top-N coins per signal bucket", min_value=1, max_value=10, value=5, step=1,
        help="Aggregate the N strongest buy/sell signals at each bar."
    )
    bt_min_history = st.slider(
        "Min history required (days)", min_value=30, max_value=180, value=60, step=10,
        help="Minimum bars of history needed before a bar is included in backtest."
    )

fwd_bars   = days_to_bars(fwd_days)
bt_min_bars = days_to_bars(bt_min_history)


# ─── Rolling signal reconstruction ───────────────────────
@st.cache_data(ttl=300, show_spinner="Running backtest…")
def run_backtest(
    df_rel: pd.DataFrame,
    aema_bars: int, jma_bars: int, nw_bars: int, nw_r: float,
    jma_phase: float,
    w_aema: float, w_jma: float, w_nw: float,
    fwd_bars: int,
    bt_min_bars: int,
) -> pd.DataFrame:
    """
    For every coin and every historical bar t (where enough history exists),
    compute the composite signal using data up to t, then record the
    forward excess return vs BTC over the next fwd_bars bars.

    Returns a long-form DataFrame with columns:
        date, coin, composite, signal_label, fwd_return
    """
    total_w = (w_aema + w_jma + w_nw) or 1.0
    records = []

    for coin in df_rel.columns:
        s = df_rel[coin].dropna()
        n = len(s)
        if n < bt_min_bars + fwd_bars + max(aema_bars, jma_bars, nw_bars):
            continue

        # Pre-compute smoothers on the full series (causal — no look-ahead)
        aema_full  = adaptive_ema(s, aema_bars)
        jfatl_full = jfatl_hybrid(s, jma_bars, jma_bars, jma_phase)
        nw_full    = nadaraya_watson(s, nw_bars, nw_r)

        vals = s.values
        idx  = s.index

        start = bt_min_bars + max(aema_bars, jma_bars, nw_bars)

        for t in range(start, n - fwd_bars):
            # Signal at bar t (only uses data up to t — causal)
            def _dir(series_vals, t_):
                v1 = series_vals[t_]
                v0 = series_vals[t_ - 1]
                if np.isnan(v1) or np.isnan(v0):
                    return 0
                if v1 > v0:   return  1
                if v1 < v0:   return -1
                return 0

            sig_a = _dir(aema_full.values,  t)
            sig_j = _dir(jfatl_full.values, t)
            sig_n = _dir(nw_full.values,    t)

            composite = (sig_a * w_aema + sig_j * w_jma + sig_n * w_nw) / total_w

            # Forward return in coin/BTC ratio terms
            fwd_ret = (vals[t + fwd_bars] / vals[t] - 1) * 100 if vals[t] != 0 else np.nan

            if np.isnan(fwd_ret):
                continue

            # Categorise signal
            if composite >= 0.9:    label = "Strong Buy"
            elif composite >= 0.4:  label = "Weak Buy"
            elif composite <= -0.9: label = "Strong Sell"
            elif composite <= -0.4: label = "Weak Sell"
            else:                   label = "Neutral"

            records.append({
                "date":         idx[t],
                "coin":         coin,
                "composite":    composite,
                "signal_label": label,
                "fwd_return":   fwd_ret,
            })

    return pd.DataFrame(records)


with st.spinner("Running backtest (this may take a moment on large universes)…"):
    df_bt = run_backtest(
        df_btc_rel,
        aema_bars, jma_bars, nw_bars, nw_r,
        jma_phase,
        w_aema, w_jma, w_nw,
        fwd_bars,
        bt_min_bars,
    )

if df_bt.empty:
    st.warning("Not enough history to run backtest. Try reducing 'Min history required' or the forward window.")
    st.stop()

# ─── Summary stats ────────────────────────────────────────
SIGNAL_ORDER = ["Strong Buy", "Weak Buy", "Neutral", "Weak Sell", "Strong Sell"]
SIGNAL_COLORS = {
    "Strong Buy":  "#22c55e",
    "Weak Buy":    "#86efac",
    "Neutral":     "#94a3b8",
    "Weak Sell":   "#fca5a5",
    "Strong Sell": "#ef4444",
}

def win_rate(series: pd.Series) -> float:
    """% of observations where fwd_return > 0."""
    valid = series.dropna()
    if len(valid) == 0:
        return np.nan
    return (valid > 0).sum() / len(valid) * 100


summary = (
    df_bt.groupby("signal_label")["fwd_return"]
    .agg(
        Count="count",
        Avg_Return="mean",
        Median_Return="median",
        Std_Return="std",
        Win_Rate=win_rate,
        P10=lambda x: np.percentile(x.dropna(), 10),
        P90=lambda x: np.percentile(x.dropna(), 90),
    )
    .reset_index()
    .rename(columns={"signal_label": "Signal"})
)
summary["Sharpe_proxy"] = summary["Avg_Return"] / summary["Std_Return"].replace(0, np.nan)
summary = summary.set_index("Signal").reindex(
    [s for s in SIGNAL_ORDER if s in summary["Signal"].values or s in summary.index]
).reset_index()


tab_bt_summary, tab_bt_dist, tab_bt_cum, tab_bt_coin = st.tabs([
    "📋 Summary Stats",
    "📊 Return Distributions",
    "📈 Cumulative Return",
    "🔍 Coin-Level Detail",
])


# ── Tab 1: Summary Stats ──────────────────────────────────
with tab_bt_summary:
    st.subheader(f"Signal Performance — {fwd_days}d Forward Return (vs BTC)")

    # Metric cards
    cols = st.columns(len(summary))
    for i, row in summary.iterrows():
        sig = row["Signal"]
        with cols[i]:
            color = SIGNAL_COLORS.get(sig, "#94a3b8")
            st.markdown(
                f"""
                <div style="border-left:4px solid {color};padding:8px 12px;
                            background:rgba(0,0,0,0.03);border-radius:4px;margin-bottom:8px;">
                    <b style="color:{color}">{sig}</b><br>
                    <small>n={int(row['Count']):,}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.metric("Avg Excess Ret", f"{row['Avg_Return']:+.2f}%")
            st.metric("Win Rate",       f"{row['Win_Rate']:.1f}%")
            st.metric("Median",         f"{row['Median_Return']:+.2f}%")
            st.metric("Sharpe proxy",   f"{row['Sharpe_proxy']:+.2f}" if not np.isnan(row['Sharpe_proxy']) else "—")

    st.markdown("---")

    # Full table
    st.dataframe(
        summary.style.format({
            "Avg_Return":     "{:+.2f}%",
            "Median_Return":  "{:+.2f}%",
            "Std_Return":     "{:.2f}%",
            "Win_Rate":       "{:.1f}%",
            "P10":            "{:+.2f}%",
            "P90":            "{:+.2f}%",
            "Sharpe_proxy":   "{:+.2f}",
        }).background_gradient(subset=["Avg_Return", "Win_Rate"], cmap="RdYlGn"),
        use_container_width=True,
    )

    st.caption(
        "**Avg Excess Return** = average (coin/BTC) % change over the forward window, aggregated across all signal occurrences. "
        "**Win Rate** = % of signals where coin outperformed BTC. "
        "**Sharpe proxy** = Avg / Std (unnormalised, directional quality measure). "
        "**P10/P90** = 10th and 90th percentile of forward returns."
    )


# ── Tab 2: Return Distributions ──────────────────────────
with tab_bt_dist:
    st.subheader("Forward Return Distributions by Signal Type")

    fig_dist_bt = go.Figure()
    for sig in SIGNAL_ORDER:
        grp = df_bt[df_bt["signal_label"] == sig]["fwd_return"].dropna()
        if grp.empty:
            continue
        fig_dist_bt.add_trace(go.Violin(
            x=[sig] * len(grp),
            y=grp.values,
            name=sig,
            box_visible=True,
            meanline_visible=True,
            line_color=SIGNAL_COLORS.get(sig, "#94a3b8"),
            fillcolor=SIGNAL_COLORS.get(sig, "#94a3b8"),
            opacity=0.6,
            points="outliers",
        ))

    fig_dist_bt.add_hline(y=0, line_dash="dash", line_color="white", line_width=1)
    fig_dist_bt.update_layout(
        title=f"Distribution of {fwd_days}d Forward Returns (vs BTC) by Signal",
        yaxis_title="Forward Return (%)",
        xaxis_title="Signal Type",
        height=450,
        showlegend=False,
        violinmode="group",
        margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig_dist_bt, use_container_width=True)

    # Average return bar chart with error bars
    fig_avg = go.Figure()
    for _, row in summary.iterrows():
        sig = row["Signal"]
        fig_avg.add_trace(go.Bar(
            x=[sig],
            y=[row["Avg_Return"]],
            error_y=dict(type="data", array=[row["Std_Return"] / np.sqrt(max(row["Count"], 1))], visible=True),
            marker_color=SIGNAL_COLORS.get(sig, "#94a3b8"),
            name=sig,
            text=[f"{row['Avg_Return']:+.2f}%"],
            textposition="outside",
        ))

    fig_avg.add_hline(y=0, line_dash="solid", line_color="white", line_width=1)
    fig_avg.update_layout(
        title=f"Mean {fwd_days}d Forward Return ± SE by Signal",
        yaxis_title="Mean Forward Return (%)",
        height=380,
        showlegend=False,
        margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig_avg, use_container_width=True)


# ── Tab 3: Cumulative Return ──────────────────────────────
with tab_bt_cum:
    st.subheader("Cumulative Excess Return — Top-N Signal Strategy")
    st.caption(
        f"Each bar: take the **top {bt_top_n} strongest buy signals** and **bottom {bt_top_n} strongest sell signals** "
        f"(by composite score). Equal-weight their forward returns. "
        "Long-only and long-short equity curves are shown."
    )

    # Build daily (by date) aggregate returns for top-N longs and top-N shorts
    def aggregate_top_n(group: pd.DataFrame, n: int, direction: str) -> float:
        """Average fwd_return of top-N by |composite| in the given direction."""
        if direction == "buy":
            top = group[group["composite"] > 0].nlargest(n, "composite")
        else:
            top = group[group["composite"] < 0].nsmallest(n, "composite")
        if top.empty:
            return np.nan
        return top["fwd_return"].mean()

    daily_long  = df_bt.groupby("date").apply(aggregate_top_n, n=bt_top_n, direction="buy")
    daily_short = df_bt.groupby("date").apply(aggregate_top_n, n=bt_top_n, direction="sell")

    daily_long  = daily_long.sort_index().dropna()
    daily_short = daily_short.sort_index().dropna()

    # Align
    all_dates = daily_long.index.union(daily_short.index)
    daily_long  = daily_long.reindex(all_dates)
    daily_short = daily_short.reindex(all_dates)

    # Long-short: long the buys, short the sells (short = expect negative fwd_return, so negate)
    long_short = daily_long.fillna(0) - daily_short.fillna(0)

    # Cumulative returns (arithmetic, in % points)
    cum_long       = daily_long.cumsum()
    cum_short      = (-daily_short).cumsum()   # short leg profit = negative of avg sell return
    cum_long_short = long_short.cumsum()

    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=cum_long.index, y=cum_long.values,
        name=f"Long top-{bt_top_n} buys",
        line=dict(color="#22c55e", width=2),
        mode="lines",
    ))
    fig_cum.add_trace(go.Scatter(
        x=cum_short.index, y=cum_short.values,
        name=f"Short top-{bt_top_n} sells",
        line=dict(color="#ef4444", width=2),
        mode="lines",
    ))
    fig_cum.add_trace(go.Scatter(
        x=cum_long_short.index, y=cum_long_short.values,
        name="Long-Short combined",
        line=dict(color="#f59e0b", width=2.5, dash="dot"),
        mode="lines",
    ))
    fig_cum.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)

    fig_cum.update_layout(
        title=f"Cumulative Excess Return (vs BTC) — Top-{bt_top_n} Signal Strategy",
        yaxis_title="Cumulative Return (% pts, vs BTC)",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=60, b=40),
        hovermode="x unified",
    )
    st.plotly_chart(fig_cum, use_container_width=True)

    # Rolling win-rate over time
    roll_win = (daily_long > 0).rolling(20).mean() * 100
    fig_wr = go.Figure(go.Scatter(
        x=roll_win.index, y=roll_win.values,
        fill="tozeroy",
        line=dict(color="#3b82f6", width=1.5),
        name="Rolling 20-bar Win Rate (long leg)",
    ))
    fig_wr.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50% break-even")
    fig_wr.update_layout(
        title="Rolling 20-bar Win Rate — Long Leg (top-N buy signals)",
        yaxis_title="Win Rate (%)",
        yaxis_range=[0, 100],
        height=300,
        margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig_wr, use_container_width=True)


# ── Tab 4: Coin-Level Detail ──────────────────────────────
with tab_bt_coin:
    st.subheader("Per-Coin Signal Performance")

    coin_bt_summary = (
        df_bt[df_bt["signal_label"].isin(["Strong Buy", "Weak Buy"])]
        .groupby("coin")["fwd_return"]
        .agg(
            Count="count",
            Avg_Return="mean",
            Win_Rate=win_rate,
            Std_Return="std",
        )
        .reset_index()
        .rename(columns={"coin": "Coin"})
        .sort_values("Avg_Return", ascending=False)
    )
    coin_bt_summary["Sharpe"] = coin_bt_summary["Avg_Return"] / coin_bt_summary["Std_Return"].replace(0, np.nan)

    if not coin_bt_summary.empty:
        fig_coin_perf = go.Figure(go.Bar(
            x=coin_bt_summary["Coin"],
            y=coin_bt_summary["Avg_Return"],
            marker=dict(
                color=coin_bt_summary["Avg_Return"],
                colorscale="RdYlGn",
                cmin=-5, cmax=5,
                colorbar=dict(title="Avg Ret %"),
            ),
            text=coin_bt_summary["Avg_Return"].apply(lambda x: f"{x:+.2f}%"),
            textposition="outside",
        ))
        fig_coin_perf.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_coin_perf.update_layout(
            title=f"Avg {fwd_days}d Forward Return (vs BTC) per Coin — Buy Signals Only",
            yaxis_title="Avg Forward Return (%)",
            height=420,
            xaxis_tickangle=-45,
            margin=dict(t=50, b=80),
        )
        st.plotly_chart(fig_coin_perf, use_container_width=True)

        st.dataframe(
            coin_bt_summary.style.format({
                "Avg_Return": "{:+.2f}%",
                "Win_Rate":   "{:.1f}%",
                "Std_Return": "{:.2f}%",
                "Sharpe":     "{:+.2f}",
            }).background_gradient(subset=["Avg_Return", "Win_Rate"], cmap="RdYlGn"),
            use_container_width=True,
        )
    else:
        st.info("No buy signal history found for coin-level analysis.")
