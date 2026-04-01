# pages/excess_return_scanner.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Excess Return Scanner", layout="wide")
st.title("Excess Return Scanner")

# ─────────────────────────────────────────────
# Cached: per-coin excess cumulative returns
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def compute_excess_cumreturns(price_theme: pd.DataFrame, theme_values_aligned, cache_token):
    df_prices = price_theme.copy()
    if not isinstance(df_prices.index, pd.DatetimeIndex):
        df_prices.index = pd.to_datetime(df_prices.index)
    df_prices = df_prices.sort_index()

    returns = df_prices.pct_change().dropna() * 100  # (n_dates, n_coins)

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

    excess_returns = returns - theme_median_per_coin
    excess_cum     = excess_returns.cumsum()

    return excess_cum, coin_themes


# ─────────────────────────────────────────────
# Improved convexity classifier
# ─────────────────────────────────────────────

def classify_curvature(series: pd.Series, n_bars: int, flatness_thresh: float = 0.05):
    """
    Multi-criteria curvature classifier.

    A single global quadratic fit over the full window is unreliable:
    a chart like EIGEN (big dip in the middle, slight uptick at the
    tail) can produce a positive 'a' coefficient while still being an
    overall loser; ADA (strong early rise then collapse) can also
    appear convex globally despite falling hard recently.

    Instead, three criteria must ALL agree before a non-flat label
    is assigned:

    Criterion 1 — NET DIRECTION (full window)
        net_change = last value − first value
        Must be positive for Buy, negative for Sell.
        Eliminates EIGEN (net negative over the window).

    Criterion 2 — RIGHT-SIDE QUADRATIC (last ⌈n/3⌉ bars only)
        Fit y = ax² + bx + c to the most-recent third of the window.
        For Buy:  a > thresh  AND  b > 0
                  (curving upward AND currently sloping upward)
        For Sell: a < −thresh AND  b < 0
                  (curving downward AND currently sloping downward)
        Requiring both a and b to agree in sign prevents a momentary
        uptick at the tail of a falling chart (EIGEN) from qualifying
        as convex, and prevents a big early rise (ADA) from carrying
        through as convex once the right side is falling.

    Criterion 3 — HALF-WINDOW SLOPE COMPARISON
        Compare linear trend of the second half vs. the first half.
        For Buy:  slope_second > slope_first   (accelerating up)
        For Sell: slope_second < slope_first   (accelerating down)
        ETHFI / BTC rise faster in their second half → Buy.
        ADA rises in the first half then falls → slope_second much
        lower than slope_first → correctly excluded from Buy and
        flagged as Sell.

    flatness_thresh is applied as a fraction of the full y-range so
    the dead-band scales with the magnitude of the series.
    """
    s = series.dropna().tail(n_bars)
    if len(s) < 9:
        return "flat"

    y_full  = s.values.astype(float)
    y_range = np.ptp(y_full)
    thresh  = flatness_thresh * y_range if y_range > 0 else flatness_thresh

    # ── Criterion 1: net direction ───────────────────────────────
    net_change = y_full[-1] - y_full[0]
    if abs(net_change) < thresh:
        return "flat"
    net_up = net_change > 0

    # ── Criterion 2: right-side quadratic (last third) ───────────
    right_n = max(5, len(s) // 3)
    y_right = y_full[-right_n:]
    x_right = np.arange(len(y_right), dtype=float)
    x_norm  = x_right / x_right[-1] if x_right[-1] != 0 else x_right

    try:
        a, b, _ = np.polyfit(x_norm, y_right, 2)
    except Exception:
        return "flat"

    right_convex  = (a >  thresh) and (b >  0)
    right_concave = (a < -thresh) and (b <  0)

    # ── Criterion 3: second-half slope vs. first-half slope ──────
    mid = len(y_full) // 2
    x_h = np.arange(mid, dtype=float)

    try:
        slope_first  = np.polyfit(x_h, y_full[:mid], 1)[0]
        end_idx      = min(mid * 2, len(y_full))
        slope_second = np.polyfit(x_h[:end_idx - mid], y_full[mid:end_idx], 1)[0]
    except Exception:
        return "flat"

    accelerating_up   = slope_second > slope_first
    accelerating_down = slope_second < slope_first

    # ── Final decision: all three must agree ─────────────────────
    if net_up and right_convex and accelerating_up:
        return "convex"
    if (not net_up) and right_concave and accelerating_down:
        return "concave"
    return "flat"


# ─────────────────────────────────────────────
# Session state guard
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
    "Each coin's return minus its theme median return, cumulated over time (additive, %-pts). "
    "Positive = outperforming theme;  Negative = underperforming."
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
            series = excess_display[coin].dropna()
            fig_grid.add_trace(
                go.Scatter(
                    x=series.index, y=series.values,
                    mode="lines", name=coin, showlegend=False,
                ),
                row=r + 1, col=c + 1,
            )
        fig_grid.update_layout(
            height=280 * rows, hovermode="x unified",
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig_grid, use_container_width=True)

# ═════════════════════════════════════════════════════════════════
# SECTION 2 — Convexity Scanner → Buy / Sell Table
# ═════════════════════════════════════════════════════════════════

st.markdown("---")
st.subheader("Convexity Scanner — Buy / Sell Signals")
st.caption(
    "**Buy (Convex ↑):** net positive over the window  +  right side curving & sloping upward  "
    "+  second half accelerating vs. first half.  \n"
    "**Sell (Concave ↓):** net negative over the window  +  right side curving & sloping downward  "
    "+  second half decelerating vs. first half.  \n"
    "All three criteria must agree — reduces false signals from mid-window spikes/dips."
)

col_a, col_b, _ = st.columns(3)
with col_a:
    scan_n_bars = st.number_input(
        "Bars to scan (right-most window)",
        min_value=9, max_value=len(excess_cum), value=min(90, len(excess_cum)), step=1,
        key="scan_n_bars",
    )
with col_b:
    flatness_thresh = st.slider(
        "Flatness threshold  (0 = very sensitive · 0.5 = strict)",
        min_value=0.0, max_value=0.5, value=0.05, step=0.01,
        key="flatness_thresh",
        help=(
            "Fraction of the full-window y-range used as dead-band. "
            "Increase to require stronger moves before signalling."
        ),
    )

classifications = {
    coin: classify_curvature(excess_cum[coin], int(scan_n_bars), flatness_thresh)
    for coin in excess_cum.columns
}

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
    st.info("No convex/concave signals detected. Try reducing the flatness threshold.")
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

    st.dataframe(
        signal_df.style.apply(highlight_buysell, axis=None),
        use_container_width=True,
    )

    total_buy  = sum(len(v) for v in buy_by_theme.values())
    total_sell = sum(len(v) for v in sell_by_theme.values())
    total_flat = sum(1 for v in classifications.values() if v == "flat")
    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 Buy signals",    total_buy)
    c2.metric("🔴 Sell signals",   total_sell)
    c3.metric("⬜ Flat (ignored)", total_flat)

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
        title=f"{selected_coin}  |  Theme: {theme_label}  |  Signal: {shape_label.upper()}",
        xaxis_title="Date",
        yaxis_title="Cumulative Excess Return (%-pts vs theme median)",
        hovermode="x unified",
        height=420,
    )
    st.plotly_chart(fig_coin, use_container_width=True)

st.success("Done")
