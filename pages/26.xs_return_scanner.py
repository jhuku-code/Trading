# pages/excess_return_scanner.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Excess Return Scanner", layout="wide")
st.title("Excess Return Scanner")

# ─────────────────────────────────────────────
# Cached computations
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def compute_excess_cumreturns(price_theme: pd.DataFrame, theme_values_aligned, cache_token):
    """
    For each coin compute:
        excess_return[t] = coin_return[t] - theme_median_return[t]
    Then cumsum the excess returns (additive, in %-pts).

    Returns
    -------
    excess_cum  : pd.DataFrame  shape (n_dates, n_coins)
    coin_themes : pd.Series     coin -> theme mapping
    """
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

    # theme median return at each date  →  (n_dates, n_themes)
    theme_median_ts = {}
    for theme in coin_themes.unique():
        coins_in = coin_themes[coin_themes == theme].index.tolist()
        theme_median_ts[theme] = returns[coins_in].median(axis=1)
    theme_median_df = pd.DataFrame(theme_median_ts)

    # broadcast theme medians to coin level
    theme_median_per_coin = pd.DataFrame(
        {coin: theme_median_df[coin_themes[coin]] for coin in returns.columns},
        index=returns.index,
    )

    excess_returns = returns - theme_median_per_coin   # (n_dates, n_coins)
    excess_cum     = excess_returns.cumsum()           # additive cumsum in %-pts

    return excess_cum, coin_themes


# ─────────────────────────────────────────────
# Convexity classifier
# ─────────────────────────────────────────────

def classify_curvature(series: pd.Series, n_bars: int, flatness_thresh: float = 0.05):
    """
    Fit quadratic y = a*x^2 + b*x + c to the last `n_bars` of `series`.
    `a` is normalised by the y-range so the threshold is dimensionless.

    Returns 'convex', 'concave', or 'flat'.
    """
    s = series.dropna().tail(n_bars)
    if len(s) < 5:
        return "flat"
    x = np.arange(len(s), dtype=float)
    y = s.values.astype(float)
    x_norm = x / x[-1] if x[-1] != 0 else x
    try:
        coeffs = np.polyfit(x_norm, y, 2)
    except Exception:
        return "flat"
    a      = coeffs[0]
    y_range = np.ptp(y)
    thresh  = flatness_thresh * y_range if y_range > 0 else flatness_thresh
    if   a >  thresh: return "convex"
    elif a < -thresh: return "concave"
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
        n_cols = min(4, len(coins))
        rows   = (len(coins) + n_cols - 1) // n_cols
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
    "Fits a quadratic to the right-most N bars of each coin's excess-cumulative-return chart. "
    "Convex (∪) → gaining → **Buy**.  Concave (∩) → losing → **Sell**.  Flat → ignored."
)

col_a, col_b, _ = st.columns(3)
with col_a:
    scan_n_bars = st.number_input(
        "Bars to scan (right-most window)",
        min_value=5, max_value=len(excess_cum), value=min(90, len(excess_cum)), step=1,
        key="scan_n_bars",
    )
with col_b:
    flatness_thresh = st.slider(
        "Flatness threshold  (0 = very sensitive · 0.5 = strict)",
        min_value=0.0, max_value=0.5, value=0.05, step=0.01,
        key="flatness_thresh",
        help="Fraction of the y-range used as dead-band for 'flat' classification.",
    )

# Classify every coin
classifications = {
    coin: classify_curvature(excess_cum[coin], int(scan_n_bars), flatness_thresh)
    for coin in excess_cum.columns
}

buy_by_theme:  dict[str, list[str]] = {}
sell_by_theme: dict[str, list[str]] = {}
for coin, label in classifications.items():
    theme = coin_themes.get(coin, "Unknown")
    if   label == "convex":  buy_by_theme.setdefault(theme, []).append(coin)
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
    fill_color = line_color + "26"   # 15 % opacity hex suffix

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
