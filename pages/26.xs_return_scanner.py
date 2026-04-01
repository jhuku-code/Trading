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
    """
    For each coin compute:
        excess_return[t] = coin_return[t] - theme_median_return[t]
    Then cumsum the excess returns (additive, in %-pts).

    Returns
    -------
    excess_cum  : pd.DataFrame  (n_dates × n_coins)
    coin_themes : pd.Series     coin -> theme
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

    excess_returns = returns - theme_median_per_coin
    excess_cum     = excess_returns.cumsum()
    return excess_cum, coin_themes


# ─────────────────────────────────────────────
# Weighted scoring classifier
# ─────────────────────────────────────────────

def classify_curvature(
    series: pd.Series,
    n_bars: int,
    flatness_thresh: float = 0.05,
    min_score: int = 3,
):
    """
    Weighted scoring classifier — returns 'convex', 'concave', or 'flat'.

    Four criteria are evaluated; each contributes a signed score toward
    Buy (+) or Sell (-).  A coin is labelled Buy if total score >= +min_score,
    Sell if total score <= -min_score, otherwise Flat.

    Criteria and weights
    ────────────────────
    C1  Net direction  (weight 1)
        last − first of the full window.
        Positive → +1, Negative → −1, near-zero → 0.
        Catches EIGEN (net loser) and ADA (collapsed since peak).

    C2  Right-side quadratic  (weight 2)  ← most informative, double weight
        Quadratic fit on the most-recent ⌈n/3⌉ bars.
        For upward: a > thresh AND b > 0  → +2
        For downward: a < −thresh AND b < 0  → −2
        Requiring a and b to agree in sign stops a tail-uptick on an
        otherwise falling chart (EIGEN) from scoring positive here.

    C3  Half-window slope acceleration  (weight 1)
        Linear slope of second half vs. first half.
        slope_second > slope_first → +1 (accelerating up, like ETHFI/BTC)
        slope_second < slope_first → −1 (decelerating/reversing, like ADA)

    C4  Recent momentum  (weight 1)
        Mean of last 20 % of bars vs. mean of the preceding 20 %.
        recent_mean > prior_mean → +1
        recent_mean < prior_mean → −1
        Short-range, orthogonal to the quadratic math — catches coins
        just beginning to turn before the full-window metrics react.

    Max score = 1 + 2 + 1 + 1 = 5
    Default min_score = 3, so possible qualifying combos:
        C2 alone (2) is NOT enough — prevents noisy tail-fits triggering
        C1+C2 (3) — net positive + right-side curving up → Buy
        C2+C3 (3) — right-side curving up + accelerating → Buy
        C2+C4 (3) — right-side curving up + recent momentum → Buy
        C1+C3+C4 (3) — no strong quadratic but all directional agree → Buy
        Any combo ≥ 3 → Buy / Sell

    flatness_thresh is scaled by the full-window y-range so the
    dead-band is dimensionless across all magnitudes.
    """
    s = series.dropna().tail(n_bars)
    if len(s) < 9:
        return "flat", 0

    y    = s.values.astype(float)
    yr   = np.ptp(y)
    thr  = flatness_thresh * yr if yr > 0 else flatness_thresh

    score = 0

    # ── C1: net direction (weight 1) ─────────────────────────────
    net = y[-1] - y[0]
    if   net >  thr: score += 1
    elif net < -thr: score -= 1

    # ── C2: right-side quadratic on last third (weight 2) ────────
    right_n = max(5, len(s) // 3)
    yr_r    = y[-right_n:]
    xr      = np.arange(len(yr_r), dtype=float)
    xr_n    = xr / xr[-1] if xr[-1] != 0 else xr
    try:
        a, b, _ = np.polyfit(xr_n, yr_r, 2)
        if   (a >  thr) and (b >  0): score += 2
        elif (a < -thr) and (b <  0): score -= 2
    except Exception:
        pass

    # ── C3: half-window slope acceleration (weight 1) ────────────
    mid = len(y) // 2
    xh  = np.arange(mid, dtype=float)
    try:
        s1 = np.polyfit(xh, y[:mid], 1)[0]
        s2 = np.polyfit(xh[:len(y[mid:mid * 2])], y[mid:mid * 2], 1)[0]
        if   s2 > s1: score += 1
        elif s2 < s1: score -= 1
    except Exception:
        pass

    # ── C4: recent momentum — last 20 % vs prior 20 % (weight 1) ─
    fifth = max(3, len(y) // 5)
    recent_mean = y[-fifth:].mean()
    prior_mean  = y[-2 * fifth:-fifth].mean()
    if   recent_mean > prior_mean: score += 1
    elif recent_mean < prior_mean: score -= 1

    # ── Decision ─────────────────────────────────────────────────
    if   score >=  min_score: return "convex",  score
    elif score <= -min_score: return "concave", score
    return "flat", score


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
    "Scores each coin across four criteria (max 5 pts). "
    "**Buy** = score ≥ threshold (convex/gaining).  "
    "**Sell** = score ≤ −threshold (concave/losing).  "
    "Adjust the sliders to control sensitivity."
)

col_a, col_b, col_c = st.columns(3)
with col_a:
    scan_n_bars = st.number_input(
        "Bars to scan (right-most window)",
        min_value=9, max_value=len(excess_cum), value=min(90, len(excess_cum)), step=1,
        key="scan_n_bars",
    )
with col_b:
    flatness_thresh = st.slider(
        "Flatness threshold  (0 = sensitive · 0.5 = strict)",
        min_value=0.0, max_value=0.5, value=0.05, step=0.01,
        key="flatness_thresh",
        help="Fraction of y-range used as dead-band before any criterion fires.",
    )
with col_c:
    min_score = st.slider(
        "Min score to signal  (max possible = 5)",
        min_value=1, max_value=5, value=3, step=1,
        key="min_score",
        help=(
            "3 = at least two criteria agree (recommended). "
            "2 = C2 alone can trigger. "
            "4 = very strict. "
            "5 = all criteria must agree."
        ),
    )

# Classify every coin
raw_results = {
    coin: classify_curvature(excess_cum[coin], int(scan_n_bars), flatness_thresh, min_score)
    for coin in excess_cum.columns
}
classifications = {coin: label for coin, (label, _) in raw_results.items()}
scores          = {coin: sc    for coin, (_, sc)    in raw_results.items()}

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

# ── Score debug table ─────────────────────────────────────────────
with st.expander("Show raw scores for all coins", expanded=False):
    score_rows = []
    for coin in sorted(excess_cum.columns):
        score_rows.append({
            "Coin":   coin,
            "Theme":  coin_themes.get(coin, "Unknown"),
            "Score":  scores[coin],
            "Signal": classifications[coin].upper(),
        })
    score_df = pd.DataFrame(score_rows).set_index("Coin")

    def colour_signal(val):
        if val == "CONVEX":  return "color: #4ade80"
        if val == "CONCAVE": return "color: #f87171"
        return "color: #94a3b8"

    st.dataframe(
        score_df.style.applymap(colour_signal, subset=["Signal"]),
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
            f"Signal: {shape_label.upper()}  |  Score: {coin_score} / 5"
        ),
        xaxis_title="Date",
        yaxis_title="Cumulative Excess Return (%-pts vs theme median)",
        hovermode="x unified",
        height=420,
    )
    st.plotly_chart(fig_coin, use_container_width=True)

st.success("Done")
