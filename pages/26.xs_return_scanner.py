# pages/excess_return_scanner.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Excess Return Scanner", layout="wide")
st.title("Excess Return Scanner")

# ─────────────────────────────────────────────
# CACHED: PER-COIN EXCESS CUMULATIVE RETURNS
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def compute_excess_cumreturns(price_theme: pd.DataFrame, theme_values_aligned, cache_token):
    """
    For each coin:
        excess_return[t] = coin_return[t] − theme_median_return[t]
    Cumsum → additive excess cumulative return in %-pts.
    """
    df_prices = price_theme.copy()
    if not isinstance(df_prices.index, pd.DatetimeIndex):
        df_prices.index = pd.to_datetime(df_prices.index)
    df_prices = df_prices.sort_index()

    returns = df_prices.pct_change().dropna() * 100

    if isinstance(theme_values_aligned, (list, tuple, np.ndarray)):
        coin_themes = pd.Series(list(theme_values_aligned), index=price_theme.columns)
    elif isinstance(theme_values_aligned, pd.Series):
        coin_themes = theme_values_aligned.reindex(price_theme.columns).fillna("Unknown")
    else:
        coin_themes = pd.Series(["Unknown"] * len(price_theme.columns), index=price_theme.columns)

    theme_median_ts = {}
    for theme in coin_themes.unique():
        coins_in = coin_themes[coin_themes == theme].index.tolist()
        theme_median_ts[theme] = returns[coins_in].median(axis=1)
    theme_median_df = pd.DataFrame(theme_median_ts)

    theme_median_per_coin = pd.DataFrame(
        {coin: theme_median_df[coin_themes[coin]] for coin in returns.columns},
        index=returns.index,
    )

    excess_cum = (returns - theme_median_per_coin).cumsum()
    return excess_cum, coin_themes


# ─────────────────────────────────────────────
# IMPROVED WEIGHTED SCORING CLASSIFIER
# ─────────────────────────────────────────────

def _linear_r2(y: np.ndarray) -> float:
    """R² of a simple linear fit — measures how clean/sustained the trend is."""
    if len(y) < 3:
        return 0.0
    x = np.arange(len(y), dtype=float)
    try:
        p       = np.polyfit(x, y, 1)
        y_hat   = np.polyval(p, x)
        ss_res  = np.sum((y - y_hat) ** 2)
        ss_tot  = np.sum((y - y.mean()) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    except Exception:
        return 0.0


def classify_curvature(
    series: pd.Series,
    n_bars: int,
    flatness_thresh: float = 0.05,
    min_score: int = 3,
):
    """
    Five-criterion weighted scoring classifier.
    Returns (label, score, detail_dict)
    """
    s = series.dropna().tail(n_bars)
    if len(s) < 9:
        return "flat", 0, {}

    y    = s.values.astype(float)
    n    = len(y)
    yr   = np.ptp(y)
    thr  = flatness_thresh * yr if yr > 0 else flatness_thresh

    detail = {}
    score  = 0

    # ── C1: robust net direction ─────────────────────────────────
    tenth   = max(2, n // 10)
    med_end = np.median(y[-tenth:])
    med_beg = np.median(y[:tenth])
    net     = med_end - med_beg
    c1 = 1 if net > thr else (-1 if net < -thr else 0)
    score += c1
    detail["C1_net_direction"] = c1

    # ── C2: majority-vote right-side quadratic ───────────────────
    votes_up = votes_dn = 0
    for frac in (0.20, 0.30, 0.40):
        rn  = max(5, int(n * frac))
        yr_ = y[-rn:]
        xr  = np.arange(len(yr_), dtype=float)
        xn  = xr / xr[-1] if xr[-1] != 0 else xr
        try:
            a, b, _ = np.polyfit(xn, yr_, 2)
            if (a >  thr) and (b >  0): votes_up += 1
            if (a < -thr) and (b <  0): votes_dn += 1
        except Exception:
            pass
    c2 = 2 if votes_up >= 2 else (-2 if votes_dn >= 2 else 0)
    score += c2
    detail["C2_right_quadratic"] = c2

    # ── C3: second-half slope sign ───────────────────────────────
    mid = n // 2
    xh  = np.arange(n - mid, dtype=float)
    try:
        slope_second = np.polyfit(xh, y[mid:], 1)[0]
        c3 = 1 if slope_second > thr else (-1 if slope_second < -thr else 0)
    except Exception:
        c3 = 0
    score += c3
    detail["C3_second_half_slope"] = c3

    # ── C4: recent momentum ──────────────────────────────────────
    fifth        = max(3, n // 5)
    recent_mean  = y[-fifth:].mean()
    prior_mean   = y[-2 * fifth:-fifth].mean()
    diff         = recent_mean - prior_mean
    c4 = 1 if diff > thr else (-1 if diff < -thr else 0)
    score += c4
    detail["C4_recent_momentum"] = c4

    # ── C5: trend quality via linear R² ─────────────────────────
    r2    = _linear_r2(y)
    c5 = 0
    if r2 > 0.5:
        try:
            lin_slope = np.polyfit(np.arange(n, dtype=float), y, 1)[0]
            c5 = 1 if lin_slope > thr else (-1 if lin_slope < -thr else 0)
        except Exception:
            c5 = 0
    score += c5
    detail["C5_trend_quality_R2"] = c5
    detail["_r2"] = round(r2, 3)

    # ── Decision ─────────────────────────────────────────────────
    if   score >=  min_score: label = "convex"
    elif score <= -min_score: label = "concave"
    else:                     label = "flat"

    return label, score, detail


# ─────────────────────────────────────────────
# SESSION STATE GUARD
# ─────────────────────────────────────────────

price_theme         = st.session_state.get("price_theme",         None)
theme_values        = st.session_state.get("theme_values",        None)
price_theme_version = st.session_state.get("price_theme_version", None)
price_timeframe     = st.session_state.get("price_timeframe",     None)

if price_theme_version is not None:
    st.caption(f"Source data version: {price_theme_version}")
if price_timeframe:
    st.markdown(f"**Timeframe:** `{price_timeframe}`")

if price_theme is None or (isinstance(price_theme, pd.DataFrame) and price_theme.empty):
    st.warning("No price data found in session state (`price_theme`). Load data first.")
    st.stop()
if theme_values is None:
    st.warning("No theme mapping found in session state (`theme_values`). Load data first.")
    st.stop()

local_refresh = st.button("Refresh computations (no network fetch)")
cache_token   = (price_theme_version, bool(local_refresh))

excess_cum, coin_themes = compute_excess_cumreturns(price_theme, theme_values, cache_token)

# ═════════════════════════════════════════════════════════════════
# SECTION 1 — Excess Cumulative Return Charts (by theme)
# ═════════════════════════════════════════════════════════════════

st.subheader("Coin Excess Returns vs. Theme Average")
st.caption(
    "Each coin's return minus its theme median return, cumulated over time. "
    "Positive = outperforming theme; Negative = underperforming."
)

excess_n_bars = st.number_input(
    "Number of bars to display",
    min_value=5, max_value=len(excess_cum), value=min(90, len(excess_cum)), step=1,
    key="excess_n_bars",
)

excess_display = excess_cum.tail(int(excess_n_bars)).copy()

for theme in sorted(coin_themes.unique()):
    coins = [c for c in coin_themes[coin_themes == theme].index if c in excess_display.columns]
    if not coins:
        continue
    with st.expander(f"Theme: {theme}  ({len(coins)} coins)", expanded=False):
        n_cols   = min(4, len(coins))
        rows     = (len(coins) + n_cols - 1) // n_cols
        fig_grid = make_subplots(
            rows=rows, cols=n_cols,
            subplot_titles=coins,
            shared_xaxes=False, shared_yaxes=False,
            vertical_spacing=0.12, horizontal_spacing=0.06,
        )
        for idx, coin in enumerate(coins):
            r, c   = divmod(idx, n_cols)
            s_data = excess_display[coin].dropna()
            fig_grid.add_trace(
                go.Scatter(x=s_data.index, y=s_data.values,
                           mode="lines", name=coin, showlegend=False),
                row=r + 1, col=c + 1,
            )
        fig_grid.update_layout(height=280 * rows, hovermode="x unified",
                               margin=dict(t=40, b=20))
        st.plotly_chart(fig_grid, use_container_width=True)

# ═════════════════════════════════════════════════════════════════
# SECTION 2 — Convexity Scanner → Buy / Sell Table
# ═════════════════════════════════════════════════════════════════

st.markdown("---")
st.subheader("Convexity Scanner — Buy / Sell Signals")

col_a, col_b, col_c = st.columns(3)
with col_a:
    scan_n_bars = st.number_input(
        "Bars to scan",
        min_value=9, max_value=len(excess_cum), value=min(90, len(excess_cum)), step=1,
        key="scan_n_bars",
    )
with col_b:
    flatness_thresh = st.slider(
        "Flatness threshold",
        min_value=0.0, max_value=0.5, value=0.05, step=0.01,
        key="flatness_thresh",
    )
with col_c:
    min_score = st.slider(
        "Min score to signal",
        min_value=1, max_value=6, value=3, step=1,
        key="min_score",
    )

# Classify every coin
raw_results = {
    coin: classify_curvature(excess_cum[coin], int(scan_n_bars), flatness_thresh, min_score)
    for coin in excess_cum.columns
}
classifications = {coin: label  for coin, (label, _, _)  in raw_results.items()}
scores          = {coin: sc     for coin, (_, sc, _)     in raw_results.items()}
details         = {coin: det    for coin, (_, _, det)    in raw_results.items()}

buy_by_theme:  dict[str, list[str]] = {}
sell_by_theme: dict[str, list[str]] = {}
for coin, label in classifications.items():
    theme = coin_themes.get(coin, "Unknown")
    if   label == "convex":  buy_by_theme.setdefault(theme,  []).append(coin)
    elif label == "concave": sell_by_theme.setdefault(theme, []).append(coin)

table_rows = []
for theme in sorted(coin_themes.unique()):
    buys  = ", ".join(sorted(buy_by_theme.get(theme,  [])))
    sells = ", ".join(sorted(sell_by_theme.get(theme, [])))
    if buys or sells:
        table_rows.append({"Theme": theme, "Buy (Convex ↑)": buys, "Sell (Concave ↓)": sells})

if not table_rows:
    st.info("No signals detected. Try lowering the min score or flatness threshold.")
else:
    signal_df = pd.DataFrame(table_rows).set_index("Theme")

    def highlight_buysell(df):
        styled = pd.DataFrame("", index=df.index, columns=df.columns)
        styled["Buy (Convex ↑)"]   = df["Buy (Convex ↑)"].apply(
            lambda v: "background-color: #0d3b1e; color: #4ade80;" if v else ""
        )
        styled["Sell (Concave ↓)"] = df["Sell (Concave ↓)"].apply(
            lambda v: "background-color: #3b0d0d; color: #f87171;" if v else ""
        )
        return styled

    st.dataframe(signal_df.style.apply(highlight_buysell, axis=None),
                 use_container_width=True)

    total_buy  = sum(len(v) for v in buy_by_theme.values())
    total_sell = sum(len(v) for v in sell_by_theme.values())
    total_flat = sum(1 for v in classifications.values() if v == "flat")
    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 Buy signals",    total_buy)
    c2.metric("🔴 Sell signals",   total_sell)
    c3.metric("⬜ Flat (ignored)", total_flat)

# ── Score debug table ─────────────────────────────────────────────
with st.expander("Show raw scores & criterion breakdown for all coins", expanded=False):
    score_rows = []
    for coin in sorted(excess_cum.columns):
        det = details.get(coin, {})
        score_rows.append({
            "Coin":    coin,
            "Theme":   coin_themes.get(coin, "Unknown"),
            "Signal":  classifications[coin].upper(),
            "Score":   scores[coin],
            "C1 net":  det.get("C1_net_direction",    0),
            "C2 quad": det.get("C2_right_quadratic",  0),
            "C3 slope":det.get("C3_second_half_slope",0),
            "C4 mom":  det.get("C4_recent_momentum",  0),
            "C5 R²":   det.get("C5_trend_quality_R2", 0),
            "R²":      det.get("_r2",                 0.0),
        })
    score_df = pd.DataFrame(score_rows).set_index("Coin")

    def colour_signal(val):
        if val == "CONVEX":  return "color: #4ade80"
        if val == "CONCAVE": return "color: #f87171"
        return "color: #94a3b8"

    def colour_score(val):
        try:
            v = float(val)
            if v > 0: return "color: #4ade80"
            if v < 0: return "color: #f87171"
        except Exception:
            pass
        return "color: #94a3b8"

    # CRITICAL FIX: Changed .applymap() to .map() for Pandas 2.x+
    st.dataframe(
        score_df.style
            .map(colour_signal, subset=["Signal"])
            .map(colour_score,  subset=["Score", "C1 net", "C2 quad", 
                                        "C3 slope", "C4 mom", "C5 R²"]),
        use_container_width=True,
    )

# ═════════════════════════════════════════════════════════════════
# SECTION 3 — Single-Coin Excess Cumulative Return Lookup
# ═════════════════════════════════════════════════════════════════

st.markdown("---")
st.subheader("Single-Coin Excess Cumulative Return Lookup")

all_coins = sorted(excess_cum.columns.tolist())

col_coin, col_period = st.columns([2, 1])
with col_coin:
    selected_coin = st.selectbox("Select coin", all_coins, key="single_coin_select")
with col_period:
    single_period = st.number_input(
        "Period (bars)",
        min_value=5, max_value=len(excess_cum), value=min(90, len(excess_cum)), step=1,
        key="single_period",
    )

if selected_coin:
    series      = excess_cum[selected_coin].dropna().tail(int(single_period))
    theme_label = coin_themes.get(selected_coin, "Unknown")
    shape_label = classifications.get(selected_coin, "flat")
    coin_score  = scores.get(selected_coin, 0)
    coin_r2      = details.get(selected_coin, {}).get("_r2", 0.0)

    color_map  = {"convex": "#4ade80", "concave": "#f87171", "flat": "#94a3b8"}
    line_color = color_map[shape_label]
    fill_color = line_color + "26"

    fig_coin = go.Figure()
    fig_coin.add_trace(
        go.Scatter(
            x=series.index, y=series.values,
            mode="lines",
            line=dict(color=line_color, width=2),
            name=selected_coin,
            fill="tozeroy",
            fillcolor=fill_color,
        )
    )
    fig_coin.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig_coin.update_layout(
        title=(
            f"{selected_coin}  |  Theme: {theme_label}  |  "
            f"Signal: {shape_label.upper()}  |  Score: {coin_score} / 6  |  R²: {coin_r2:.2f}"
        ),
        xaxis_title="Date",
        yaxis_title="Cumulative Excess Return (%-pts vs theme median)",
        hovermode="x unified",
        height=420,
    )
    st.plotly_chart(fig_coin, use_container_width=True)

st.success("Scanner Refreshed")
