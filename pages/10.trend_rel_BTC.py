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
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Parameters")
    st.caption(f"Detected bar frequency: **{freq_label}** ({bars_per_day} bars/day)")

    st.subheader("Investment Horizons (days)")
    short_days  = st.slider("Short horizon (days)",  min_value=1,  max_value=30,  value=7,  step=1)
    medium_days = st.slider("Medium horizon (days)", min_value=5,  max_value=90,  value=30, step=1)
    long_days   = st.slider("Long horizon (days)",   min_value=10, max_value=365, value=90, step=1)

    st.subheader("Signal Parameters (days)")
    aema_days = st.slider("Adaptive EMA period (days)",    min_value=3,  max_value=60,  value=15, step=1)
    jma_days  = st.slider("JMA / FATL period (days)",      min_value=3,  max_value=60,  value=15, step=1)
    nw_days   = st.slider("Nadaraya-Watson window (days)", min_value=5,  max_value=90,  value=20, step=1)
    nw_r      = st.slider("NW bandwidth (r)",              min_value=1.0, max_value=200.0, value=48.0, step=1.0)
    jma_phase = st.slider("JMA phase",                    min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    st.subheader("Signal Weights")
    w_aema = st.slider("Adaptive EMA weight",    0.0, 1.0, 1.0, 0.1)
    w_jma  = st.slider("JMA/FATL weight",        0.0, 1.0, 1.0, 0.1)
    w_nw   = st.slider("Nadaraya-Watson weight", 0.0, 1.0, 1.0, 0.1)

    st.subheader("Filter")
    min_composite = st.slider(
        "Min composite score to show",
        min_value=-3.0, max_value=3.0, value=0.0, step=0.5,
        help="Negative = show sell signals too. 3 = strongest buys only."
    )

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

def days_to_bars(d):
    return max(2, int(round(d * bars_per_day)))

short_bars  = days_to_bars(short_days)
medium_bars = days_to_bars(medium_days)
long_bars   = days_to_bars(long_days)
aema_bars   = days_to_bars(aema_days)
jma_bars    = days_to_bars(jma_days)
nw_bars     = days_to_bars(nw_days)
fwd_bars    = days_to_bars(fwd_days)
bt_min_bars = days_to_bars(bt_min_history)

# ─────────────────────────────────────────────────────────
# COMPUTE COIN / BTC RATIO PRICES
# ─────────────────────────────────────────────────────────
btc = df_raw["BTC"].replace(0, np.nan)
alt_cols = [c for c in df_raw.columns if c != "BTC"]
df_btc_rel = df_raw[alt_cols].div(btc, axis=0)

min_rows = max(aema_bars, jma_bars, nw_bars) + 50
valid_cols = [c for c in df_btc_rel.columns if df_btc_rel[c].dropna().shape[0] >= min_rows]
df_btc_rel = df_btc_rel[valid_cols]

# ─────────────────────────────────────────────────────────
# SIGNAL FUNCTIONS
# ─────────────────────────────────────────────────────────
def adaptive_ema(series: pd.Series, period: int) -> pd.Series:
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
    fatl = series.rolling(fatl_len, min_periods=fatl_len // 2).mean()
    e = 0.5 * (phase + 1)
    wma1 = fatl.ewm(span=jma_len, adjust=False).mean()
    wma2 = fatl.ewm(span=max(jma_len // 2, 2), adjust=False).mean()
    return wma1 * e + wma2 * (1.0 - e)


def nadaraya_watson(series: pd.Series, h: int, r: float) -> pd.Series:
    n = len(series)
    vals = series.values.astype(float)
    smoothed = np.full(n, np.nan)
    for t in range(h, n):
        window_start = max(0, t - h * 3)
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
    valid = smooth.dropna()
    if len(valid) < 2:
        return 0
    delta = valid.iloc[-1] - valid.iloc[-2]
    if delta > 0:   return  1
    elif delta < 0: return -1
    return 0


def percentile_rank(series: pd.Series) -> float:
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
                    short_bars: int, medium_bars: int, long_bars: int,
                    short_days: int, medium_days: int, long_days: int) -> pd.DataFrame:
    rows = []
    total_w = (w_aema + w_jma + w_nw) or 1.0

    for coin in df_rel.columns:
        s = df_rel[coin].dropna()
        if len(s) < max(aema_bars, jma_bars, nw_bars) + 10:
            continue

        aema_line  = adaptive_ema(s, aema_bars)
        jfatl_line = jfatl_hybrid(s, jma_bars, jma_bars, jma_phase)
        nw_line    = nadaraya_watson(s, nw_bars, nw_r)

        sig_aema  = signal_direction(aema_line)
        sig_jfatl = signal_direction(jfatl_line)
        sig_nw    = signal_direction(nw_line)

        composite = (sig_aema * w_aema + sig_jfatl * w_jma + sig_nw * w_nw) / total_w

        last_val   = s.iloc[-1]
        ret_short  = (last_val / s.iloc[-short_bars]  - 1) * 100 if len(s) > short_bars  else np.nan
        ret_medium = (last_val / s.iloc[-medium_bars] - 1) * 100 if len(s) > medium_bars else np.nan
        ret_long   = (last_val / s.iloc[-long_bars]   - 1) * 100 if len(s) > long_bars   else np.nan

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
    short_days, medium_days, long_days,
)

# ─────────────────────────────────────────────────────────
# SIGNAL LABEL HELPERS
# ─────────────────────────────────────────────────────────
def composite_label(score):
    if score >= 0.9:    return "🟢 Strong Buy"
    elif score >= 0.4:  return "🟡 Weak Buy"
    elif score <= -0.9: return "🔴 Strong Sell"
    elif score <= -0.4: return "🟠 Weak Sell"
    else:               return "⚪ Neutral"

def pct_rank_label(pct, composite):
    if np.isnan(pct):
        return "N/A"
    if composite > 0:
        if pct >= 80:   return f"{pct:.0f}% — Extended (near highs)"
        elif pct >= 50: return f"{pct:.0f}% — Mid-range"
        else:           return f"{pct:.0f}% — Early (room to run)"
    elif composite < 0:
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
# TABS — SIGNAL TABLES
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
    st.subheader("Buy Signals  (Composite > 0)")
    buys = df_filtered[df_filtered["Composite"] > 0]
    st.caption(
        "**Historical Rank** shows where the Coin/BTC ratio sits in its full history. "
        "High rank (near 100%) on a buy signal means the rally is extended; "
        "low rank means early-stage and more room to run."
    )
    render_signal_table(buys, "buy")

with tab_sell:
    st.subheader("Sell Signals  (Composite < 0)")
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
# INDIVIDUAL COIN CHART
# ─────────────────────────────────────────────────────────
st.subheader("🔍 Coin / BTC Chart")
coin_choices = df_signals["Coin"].tolist()
selected_coin = st.selectbox("Select coin to chart (Coin/BTC ratio)", coin_choices)

if selected_coin and selected_coin in df_btc_rel.columns:
    s = df_btc_rel[selected_coin].dropna()

    aema_line  = adaptive_ema(s, aema_bars)
    jfatl_line = jfatl_hybrid(s, jma_bars, jma_bars, jma_phase)
    nw_line    = nadaraya_watson(s, nw_bars, nw_r)

    plot_n   = min(500, len(s))
    s_p      = s.iloc[-plot_n:]
    aema_p   = aema_line.iloc[-plot_n:]
    jfatl_p  = jfatl_line.iloc[-plot_n:]
    nw_p     = nw_line.iloc[-plot_n:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s_p.index, y=s_p.values,      name="Coin/BTC",          line=dict(color="#94a3b8", width=1)))
    fig.add_trace(go.Scatter(x=aema_p.index, y=aema_p.values, name="Adaptive EMA",      line=dict(color="#3b82f6", width=1.5)))
    fig.add_trace(go.Scatter(x=jfatl_p.index, y=jfatl_p.values, name="JMA/FATL",       line=dict(color="#f59e0b", width=1.5)))
    fig.add_trace(go.Scatter(x=nw_p.index, y=nw_p.values,    name="Nadaraya-Watson",   line=dict(color="#22c55e", width=1.5, dash="dot")))

    row_data = df_signals[df_signals["Coin"] == selected_coin]
    if not row_data.empty:
        composite = row_data["Composite"].iloc[0]
        pct = row_data["Historical Rank (%)"].iloc[0]
        fig.update_layout(title=f"{selected_coin}/BTC  |  Composite: {composite:+.2f}  |  Historical Rank: {pct:.1f}%")

    fig.update_layout(
        height=420,
        margin=dict(t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────
# BACKTESTING ANALYSIS
# ─────────────────────────────────────────────────────────
st.markdown("---")
st.header("🧪 Signal Backtesting Analysis")
st.caption(
    "Signals are reconstructed historically using only data available at each bar (causal — no look-ahead). "
    "Forward returns are measured in Coin/BTC terms over the selected forward window."
)

SIGNAL_ORDER  = ["Strong Buy", "Weak Buy", "Neutral", "Weak Sell", "Strong Sell"]
SIGNAL_COLORS = {
    "Strong Buy":  "#22c55e",
    "Weak Buy":    "#86efac",
    "Neutral":     "#94a3b8",
    "Weak Sell":   "#fca5a5",
    "Strong Sell": "#ef4444",
}

# ─────────────────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner="Running backtest…")
def run_backtest(
    df_rel: pd.DataFrame,
    aema_bars: int, jma_bars: int, nw_bars: int, nw_r: float,
    jma_phase: float,
    w_aema: float, w_jma: float, w_nw: float,
    fwd_bars: int,
    bt_min_bars: int,
) -> pd.DataFrame:
    total_w = (w_aema + w_jma + w_nw) or 1.0
    records = []

    for coin in df_rel.columns:
        s = df_rel[coin].dropna()
        n = len(s)
        if n < bt_min_bars + fwd_bars + max(aema_bars, jma_bars, nw_bars):
            continue

        aema_full  = adaptive_ema(s, aema_bars)
        jfatl_full = jfatl_hybrid(s, jma_bars, jma_bars, jma_phase)
        nw_full    = nadaraya_watson(s, nw_bars, nw_r)

        aema_v  = aema_full.values
        jfatl_v = jfatl_full.values
        nw_v    = nw_full.values
        vals    = s.values
        idx     = s.index

        start = bt_min_bars + max(aema_bars, jma_bars, nw_bars)

        for t in range(start, n - fwd_bars):
            def _dir(arr, t_):
                v1, v0 = arr[t_], arr[t_ - 1]
                if np.isnan(v1) or np.isnan(v0): return 0
                return 1 if v1 > v0 else (-1 if v1 < v0 else 0)

            sig_a = _dir(aema_v,  t)
            sig_j = _dir(jfatl_v, t)
            sig_n = _dir(nw_v,    t)

            composite = (sig_a * w_aema + sig_j * w_jma + sig_n * w_nw) / total_w

            fwd_ret = (vals[t + fwd_bars] / vals[t] - 1) * 100 if vals[t] != 0 else np.nan
            if np.isnan(fwd_ret):
                continue

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


with st.spinner("Running backtest…"):
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

# ─────────────────────────────────────────────────────────
# SUMMARY STATS HELPER
# ─────────────────────────────────────────────────────────
def win_rate(series: pd.Series) -> float:
    valid = series.dropna()
    if len(valid) == 0: return np.nan
    return (valid > 0).sum() / len(valid) * 100

def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("signal_label")["fwd_return"]
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
    present = [s for s in SIGNAL_ORDER if s in summary["Signal"].values]
    return summary.set_index("Signal").reindex(present).reset_index()

summary_all = build_summary(df_bt)

# ─────────────────────────────────────────────────────────
# BACKTEST TABS
# ─────────────────────────────────────────────────────────
(
    tab_bt_summary,
    tab_bt_strong_buy,
    tab_bt_weak_buy,
    tab_bt_weak_sell,
    tab_bt_strong_sell,
    tab_bt_dist,
    tab_bt_cum,
    tab_bt_coin,
) = st.tabs([
    "📋 Summary",
    "🟢 Strong Buy",
    "🟡 Weak Buy",
    "🟠 Weak Sell",
    "🔴 Strong Sell",
    "📊 Distributions",
    "📈 Cumulative",
    "🔍 Per-Coin",
])

# ── helpers ──────────────────────────────────────────────
def render_summary_cards(summary_df: pd.DataFrame):
    cols = st.columns(len(summary_df))
    for i, row in summary_df.iterrows():
        sig   = row["Signal"]
        color = SIGNAL_COLORS.get(sig, "#94a3b8")
        with cols[i]:
            st.markdown(
                f"""<div style="border-left:4px solid {color};padding:8px 12px;
                    background:rgba(0,0,0,0.03);border-radius:4px;margin-bottom:8px;">
                    <b style="color:{color}">{sig}</b><br>
                    <small>n={int(row['Count']):,}</small></div>""",
                unsafe_allow_html=True,
            )
            st.metric("Avg Excess Ret", f"{row['Avg_Return']:+.2f}%")
            st.metric("Win Rate",       f"{row['Win_Rate']:.1f}%")
            st.metric("Median",         f"{row['Median_Return']:+.2f}%")
            sharpe_str = f"{row['Sharpe_proxy']:+.2f}" if not np.isnan(row['Sharpe_proxy']) else "—"
            st.metric("Sharpe proxy",   sharpe_str)


def render_signal_detail(signal_name: str, color: str):
    """Full drill-down for a single signal bucket."""
    df_sig = df_bt[df_bt["signal_label"] == signal_name].copy()

    if df_sig.empty:
        st.info(f"No historical observations for **{signal_name}** with current parameters.")
        return

    n_obs   = len(df_sig)
    avg_ret = df_sig["fwd_return"].mean()
    med_ret = df_sig["fwd_return"].median()
    wr      = win_rate(df_sig["fwd_return"])
    std_ret = df_sig["fwd_return"].std()
    sharpe  = avg_ret / std_ret if std_ret > 0 else np.nan
    p10     = np.percentile(df_sig["fwd_return"].dropna(), 10)
    p90     = np.percentile(df_sig["fwd_return"].dropna(), 90)

    # Metric cards
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Observations",  f"{n_obs:,}")
    m2.metric("Avg Return",    f"{avg_ret:+.2f}%")
    m3.metric("Median Return", f"{med_ret:+.2f}%")
    m4.metric("Win Rate",      f"{wr:.1f}%")
    m5.metric("Sharpe proxy",  f"{sharpe:+.2f}" if not np.isnan(sharpe) else "—")
    m6.metric("P10 / P90",     f"{p10:+.1f}% / {p90:+.1f}%")

    st.markdown("---")
    c1, c2 = st.columns(2)

    # Return histogram
    with c1:
        fig_hist = go.Figure(go.Histogram(
            x=df_sig["fwd_return"],
            nbinsx=50,
            marker_color=color,
            opacity=0.75,
            name=signal_name,
        ))
        fig_hist.add_vline(x=0,       line_dash="dash", line_color="white",  line_width=1)
        fig_hist.add_vline(x=avg_ret, line_dash="dot",  line_color="yellow", line_width=1.5,
                           annotation_text=f"Mean {avg_ret:+.2f}%", annotation_position="top right")
        fig_hist.update_layout(
            title=f"{signal_name} — Forward Return Distribution",
            xaxis_title=f"{fwd_days}d Forward Return (vs BTC %)",
            yaxis_title="Count",
            height=350,
            margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Win rate by coin
    with c2:
        coin_stats = (
            df_sig.groupby("coin")["fwd_return"]
            .agg(Count="count", Avg="mean", WR=win_rate)
            .reset_index()
            .sort_values("Avg", ascending=False)
        )
        if not coin_stats.empty:
            fig_coin_wr = go.Figure(go.Bar(
                x=coin_stats["coin"],
                y=coin_stats["WR"],
                marker=dict(
                    color=coin_stats["WR"],
                    colorscale="RdYlGn",
                    cmin=30, cmax=70,
                ),
                text=coin_stats["WR"].apply(lambda x: f"{x:.0f}%"),
                textposition="outside",
            ))
            fig_coin_wr.add_hline(y=50, line_dash="dash", line_color="gray",
                                  annotation_text="50% breakeven")
            fig_coin_wr.update_layout(
                title=f"{signal_name} — Win Rate per Coin",
                yaxis_title="Win Rate (%)",
                yaxis_range=[0, 105],
                height=350,
                xaxis_tickangle=-45,
                margin=dict(t=50, b=80),
            )
            st.plotly_chart(fig_coin_wr, use_container_width=True)

    # Avg return per coin bar chart
    if not coin_stats.empty:
        fig_coin_ret = go.Figure(go.Bar(
            x=coin_stats["coin"],
            y=coin_stats["Avg"],
            marker=dict(
                color=coin_stats["Avg"],
                colorscale="RdYlGn",
                cmin=-5, cmax=5,
            ),
            text=coin_stats["Avg"].apply(lambda x: f"{x:+.2f}%"),
            textposition="outside",
        ))
        fig_coin_ret.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_coin_ret.update_layout(
            title=f"{signal_name} — Avg {fwd_days}d Return per Coin (vs BTC)",
            yaxis_title="Avg Forward Return (%)",
            height=380,
            xaxis_tickangle=-45,
            margin=dict(t=50, b=80),
        )
        st.plotly_chart(fig_coin_ret, use_container_width=True)

    # Return over time (rolling average)
    df_sig_sorted = df_sig.sort_values("date")
    roll_avg = df_sig_sorted.groupby("date")["fwd_return"].mean().rolling(20).mean()
    if not roll_avg.dropna().empty:
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(
            x=roll_avg.index,
            y=roll_avg.values,
            fill="tozeroy",
            line=dict(color=color, width=1.5),
            name="Rolling 20-bar avg return",
        ))
        fig_time.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_time.update_layout(
            title=f"{signal_name} — Rolling Avg Forward Return Over Time",
            yaxis_title="Avg Return (%)",
            height=300,
            margin=dict(t=50, b=40),
            hovermode="x unified",
        )
        st.plotly_chart(fig_time, use_container_width=True)

    # Raw data table
    with st.expander(f"📄 Raw {signal_name} observations ({n_obs:,} rows)"):
        st.dataframe(
            df_sig[["date", "coin", "composite", "fwd_return"]]
            .sort_values("date", ascending=False)
            .style.format({
                "composite":  "{:+.3f}",
                "fwd_return": "{:+.2f}%",
            }).background_gradient(subset=["fwd_return"], cmap="RdYlGn", vmin=-10, vmax=10),
            use_container_width=True,
            height=400,
        )


# ── Tab: Summary ─────────────────────────────────────────
with tab_bt_summary:
    st.subheader(f"Signal Performance — {fwd_days}d Forward Return (vs BTC)")
    render_summary_cards(summary_all)
    st.markdown("---")
    st.dataframe(
        summary_all.style.format({
            "Avg_Return":    "{:+.2f}%",
            "Median_Return": "{:+.2f}%",
            "Std_Return":    "{:.2f}%",
            "Win_Rate":      "{:.1f}%",
            "P10":           "{:+.2f}%",
            "P90":           "{:+.2f}%",
            "Sharpe_proxy":  "{:+.2f}",
        }).background_gradient(subset=["Avg_Return", "Win_Rate"], cmap="RdYlGn"),
        use_container_width=True,
    )
    st.caption(
        "**Avg Excess Return** = mean Coin/BTC % change over the forward window across all signal occurrences. "
        "**Win Rate** = % of signals where coin outperformed BTC. "
        "**Sharpe proxy** = Avg / Std (directional quality measure). "
        "**P10/P90** = 10th/90th percentile of forward returns."
    )

# ── Tab: Strong Buy ──────────────────────────────────────
with tab_bt_strong_buy:
    st.subheader(f"🟢 Strong Buy — Deep Dive  (composite ≥ 0.9)")
    st.caption("All three smoothers agree bullishly on Coin/BTC. Highest-conviction long signal.")
    render_signal_detail("Strong Buy", SIGNAL_COLORS["Strong Buy"])

# ── Tab: Weak Buy ────────────────────────────────────────
with tab_bt_weak_buy:
    st.subheader(f"🟡 Weak Buy — Deep Dive  (0.4 ≤ composite < 0.9)")
    st.caption("Majority but not all smoothers agree bullishly. Lower conviction — useful to compare vs Strong Buy.")
    render_signal_detail("Weak Buy", SIGNAL_COLORS["Weak Buy"])

# ── Tab: Weak Sell ───────────────────────────────────────
with tab_bt_weak_sell:
    st.subheader(f"🟠 Weak Sell — Deep Dive  (−0.9 < composite ≤ −0.4)")
    st.caption("Majority but not all smoothers agree bearishly on Coin/BTC. Lower conviction short/avoid signal.")
    render_signal_detail("Weak Sell", SIGNAL_COLORS["Weak Sell"])

# ── Tab: Strong Sell ─────────────────────────────────────
with tab_bt_strong_sell:
    st.subheader(f"🔴 Strong Sell — Deep Dive  (composite ≤ −0.9)")
    st.caption("All three smoothers agree bearishly on Coin/BTC. Highest-conviction underperform signal.")
    render_signal_detail("Strong Sell", SIGNAL_COLORS["Strong Sell"])

# ── Tab: Distributions ───────────────────────────────────
with tab_bt_dist:
    st.subheader("Forward Return Distributions — All Signal Types")

    fig_violin = go.Figure()
    for sig in SIGNAL_ORDER:
        grp = df_bt[df_bt["signal_label"] == sig]["fwd_return"].dropna()
        if grp.empty:
            continue
        fig_violin.add_trace(go.Violin(
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
    fig_violin.add_hline(y=0, line_dash="dash", line_color="white", line_width=1)
    fig_violin.update_layout(
        title=f"Distribution of {fwd_days}d Forward Returns (vs BTC) by Signal",
        yaxis_title="Forward Return (%)",
        height=450,
        showlegend=False,
        margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig_violin, use_container_width=True)

    # Mean ± SE bar chart
    fig_avg = go.Figure()
    for _, row in summary_all.iterrows():
        sig = row["Signal"]
        se  = row["Std_Return"] / np.sqrt(max(row["Count"], 1))
        fig_avg.add_trace(go.Bar(
            x=[sig],
            y=[row["Avg_Return"]],
            error_y=dict(type="data", array=[se], visible=True),
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

    # Win rate by signal
    fig_wr_bar = go.Figure(go.Bar(
        x=summary_all["Signal"],
        y=summary_all["Win_Rate"],
        marker_color=[SIGNAL_COLORS.get(s, "#94a3b8") for s in summary_all["Signal"]],
        text=summary_all["Win_Rate"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside",
    ))
    fig_wr_bar.add_hline(y=50, line_dash="dash", line_color="gray",
                         annotation_text="50% breakeven", annotation_position="right")
    fig_wr_bar.update_layout(
        title="Win Rate by Signal Type",
        yaxis_title="Win Rate (%)",
        yaxis_range=[0, 105],
        height=340,
        showlegend=False,
        margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig_wr_bar, use_container_width=True)


# ── Tab: Cumulative ──────────────────────────────────────
with tab_bt_cum:
    st.subheader("Cumulative Excess Return — Signal Strategies")
    st.caption(
        f"Each bar: take the top-{bt_top_n} strongest signals in each bucket, equal-weight their forward returns. "
        "Returns are vs BTC (excess return)."
    )

    def agg_top_n(group: pd.DataFrame, n: int, direction: str) -> float:
        if direction == "buy":
            top = group[group["composite"] > 0].nlargest(n, "composite")
        else:
            top = group[group["composite"] < 0].nsmallest(n, "composite")
        return top["fwd_return"].mean() if not top.empty else np.nan

    def agg_label(group: pd.DataFrame, label: str) -> float:
        sub = group[group["signal_label"] == label]
        return sub["fwd_return"].mean() if not sub.empty else np.nan

    daily_strong_buy  = df_bt.groupby("date").apply(agg_label, label="Strong Buy").sort_index()
    daily_weak_buy    = df_bt.groupby("date").apply(agg_label, label="Weak Buy").sort_index()
    daily_weak_sell   = df_bt.groupby("date").apply(agg_label, label="Weak Sell").sort_index()
    daily_strong_sell = df_bt.groupby("date").apply(agg_label, label="Strong Sell").sort_index()

    cum_strong_buy  = daily_strong_buy.dropna().cumsum()
    cum_weak_buy    = daily_weak_buy.dropna().cumsum()
    cum_weak_sell   = (-daily_weak_sell).dropna().cumsum()
    cum_strong_sell = (-daily_strong_sell).dropna().cumsum()

    # Combined long-short (strong signals only)
    all_dates = daily_strong_buy.index.union(daily_strong_sell.index)
    ls_strong = (daily_strong_buy.reindex(all_dates).fillna(0)
                 - daily_strong_sell.reindex(all_dates).fillna(0))
    cum_ls_strong = ls_strong.cumsum()

    # Combined long-short (weak signals)
    all_dates_w = daily_weak_buy.index.union(daily_weak_sell.index)
    ls_weak = (daily_weak_buy.reindex(all_dates_w).fillna(0)
               - daily_weak_sell.reindex(all_dates_w).fillna(0))
    cum_ls_weak = ls_weak.cumsum()

    fig_cum = go.Figure()
    for series, name, color, dash in [
        (cum_strong_buy,  "Long Strong Buy",      "#22c55e", "solid"),
        (cum_weak_buy,    "Long Weak Buy",         "#86efac", "dot"),
        (cum_strong_sell, "Short Strong Sell",     "#ef4444", "solid"),
        (cum_weak_sell,   "Short Weak Sell",       "#fca5a5", "dot"),
        (cum_ls_strong,   "L/S Strong (Buy-Sell)", "#f59e0b", "dashdot"),
        (cum_ls_weak,     "L/S Weak (Buy-Sell)",   "#a78bfa", "dashdot"),
    ]:
        if not series.dropna().empty:
            fig_cum.add_trace(go.Scatter(
                x=series.index, y=series.values,
                name=name,
                line=dict(color=color, width=2, dash=dash),
                mode="lines",
            ))

    fig_cum.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    fig_cum.update_layout(
        title="Cumulative Excess Return (vs BTC) — Signal Strategies",
        yaxis_title="Cumulative Return (% pts, vs BTC)",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=80, b=40),
        hovermode="x unified",
    )
    st.plotly_chart(fig_cum, use_container_width=True)

    # Rolling win-rate comparison
    st.subheader("Rolling 20-bar Win Rate Comparison")
    fig_roll_wr = go.Figure()
    for series, name, color in [
        (daily_strong_buy,  "Strong Buy",  "#22c55e"),
        (daily_weak_buy,    "Weak Buy",    "#86efac"),
        (daily_strong_sell, "Strong Sell", "#ef4444"),
        (daily_weak_sell,   "Weak Sell",   "#fca5a5"),
    ]:
        roll = (series > 0).rolling(20).mean() * 100
        if not roll.dropna().empty:
            fig_roll_wr.add_trace(go.Scatter(
                x=roll.index, y=roll.values,
                name=name,
                line=dict(color=color, width=1.5),
                mode="lines",
            ))
    fig_roll_wr.add_hline(y=50, line_dash="dash", line_color="gray",
                          annotation_text="50% breakeven")
    fig_roll_wr.update_layout(
        yaxis_title="Win Rate (%)",
        yaxis_range=[0, 100],
        height=320,
        margin=dict(t=30, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_roll_wr, use_container_width=True)


# ── Tab: Per-Coin ─────────────────────────────────────────
with tab_bt_coin:
    st.subheader("Per-Coin Performance — Buy vs Sell Signal Comparison")

    signal_selector = st.selectbox(
        "Filter by signal type",
        ["Strong Buy", "Weak Buy", "Weak Sell", "Strong Sell", "All Buy Signals", "All Sell Signals"],
        index=0,
    )

    if signal_selector == "All Buy Signals":
        df_coin_sub = df_bt[df_bt["signal_label"].isin(["Strong Buy", "Weak Buy"])]
    elif signal_selector == "All Sell Signals":
        df_coin_sub = df_bt[df_bt["signal_label"].isin(["Strong Sell", "Weak Sell"])]
    else:
        df_coin_sub = df_bt[df_bt["signal_label"] == signal_selector]

    coin_stats = (
        df_coin_sub.groupby("coin")["fwd_return"]
        .agg(Count="count", Avg_Return="mean", Win_Rate=win_rate, Std_Return="std")
        .reset_index()
        .rename(columns={"coin": "Coin"})
        .sort_values("Avg_Return", ascending=False)
    )
    coin_stats["Sharpe"] = coin_stats["Avg_Return"] / coin_stats["Std_Return"].replace(0, np.nan)

    if not coin_stats.empty:
        color = SIGNAL_COLORS.get(signal_selector.replace("All Buy Signals", "Strong Buy")
                                                  .replace("All Sell Signals", "Strong Sell"), "#94a3b8")
        fig_coin_bar = go.Figure(go.Bar(
            x=coin_stats["Coin"],
            y=coin_stats["Avg_Return"],
            marker=dict(
                color=coin_stats["Avg_Return"],
                colorscale="RdYlGn",
                cmin=-5, cmax=5,
                colorbar=dict(title="Avg Ret %"),
            ),
            text=coin_stats["Avg_Return"].apply(lambda x: f"{x:+.2f}%"),
            textposition="outside",
        ))
        fig_coin_bar.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_coin_bar.update_layout(
            title=f"Avg {fwd_days}d Forward Return (vs BTC) — {signal_selector}",
            yaxis_title="Avg Forward Return (%)",
            height=420,
            xaxis_tickangle=-45,
            margin=dict(t=50, b=80),
        )
        st.plotly_chart(fig_coin_bar, use_container_width=True)

        st.dataframe(
            coin_stats.style.format({
                "Avg_Return": "{:+.2f}%",
                "Win_Rate":   "{:.1f}%",
                "Std_Return": "{:.2f}%",
                "Sharpe":     "{:+.2f}",
            }).background_gradient(subset=["Avg_Return", "Win_Rate"], cmap="RdYlGn"),
            use_container_width=True,
        )
    else:
        st.info(f"No observations found for **{signal_selector}**.")


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

**Historical Rank (%)**: Where the current Coin/BTC ratio sits between its all-time low (0%) and all-time high (100%) within the loaded dataset.

**Backtesting**: Signals are reconstructed at every historical bar using only data available up to that point (causal — no look-ahead). Forward returns are measured `{fwd_days}d` ahead in Coin/BTC terms.
    """)
