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
# SCORING LOGIC
# ─────────────────────────────────────────────

def _linear_r2(y: np.ndarray) -> float:
    if len(y) < 3: return 0.0
    x = np.arange(len(y), dtype=float)
    try:
        p       = np.polyfit(x, y, 1)
        y_hat   = np.polyval(p, x)
        ss_res  = np.sum((y - y_hat) ** 2)
        ss_tot  = np.sum((y - y.mean()) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    except: return 0.0

def classify_curvature(series: pd.Series, n_bars: int, flatness_thresh: float = 0.05, min_score: int = 3):
    s = series.dropna().tail(n_bars)
    if len(s) < 9: return "flat", 0, {}

    y = s.values.astype(float)
    n = len(y)
    yr = np.ptp(y)
    thr = flatness_thresh * yr if yr > 0 else flatness_thresh
    detail, score = {}, 0

    # C1: Net direction
    tenth = max(2, n // 10)
    net = np.median(y[-tenth:]) - np.median(y[:tenth])
    c1 = 1 if net > thr else (-1 if net < -thr else 0)
    score += c1; detail["C1_net_direction"] = c1

    # C2: Quadratic votes
    v_up = v_dn = 0
    for f in (0.2, 0.3, 0.4):
        rn = max(5, int(n * f))
        yr_ = y[-rn:]; xr = np.arange(len(yr_))
        xn = xr / xr[-1] if xr[-1] != 0 else xr
        try:
            a, b, _ = np.polyfit(xn, yr_, 2)
            if a > thr and b > 0: v_up += 1
            if a < -thr and b < 0: v_dn += 1
        except: pass
    c2 = 2 if v_up >= 2 else (-2 if v_dn >= 2 else 0)
    score += c2; detail["C2_right_quadratic"] = c2

    # C3: Second half slope
    mid = n // 2
    try:
        slope = np.polyfit(np.arange(n-mid), y[mid:], 1)[0]
        c3 = 1 if slope > thr else (-1 if slope < -thr else 0)
    except: c3 = 0
    score += c3; detail["C3_second_half_slope"] = c3

    # C4: Momentum
    fifth = max(3, n // 5)
    diff = y[-fifth:].mean() - y[-2*fifth:-fifth].mean()
    c4 = 1 if diff > thr else (-1 if diff < -thr else 0)
    score += c4; detail["C4_recent_momentum"] = c4

    # C5: R2
    r2 = _linear_r2(y)
    c5 = 0
    if r2 > 0.5:
        try:
            l_slope = np.polyfit(np.arange(n), y, 1)[0]
            c5 = 1 if l_slope > thr else (-1 if l_slope < -thr else 0)
        except: pass
    score += c5; detail["C5_trend_quality_R2"] = c5; detail["_r2"] = round(r2, 3)

    if score >= min_score: label = "convex"
    elif score <= -min_score: label = "concave"
    else: label = "flat"
    return label, score, detail

# ─────────────────────────────────────────────
# DATA & CONTROLS
# ─────────────────────────────────────────────

price_theme = st.session_state.get("price_theme")
theme_values = st.session_state.get("theme_values")

if price_theme is None or theme_values is None:
    st.warning("Data not found in session state. Load source data first.")
    st.stop()

local_refresh = st.button("Refresh computations")
excess_cum, coin_themes = compute_excess_cumreturns(price_theme, theme_values, (st.session_state.get("price_theme_version"), local_refresh))

# ═════════════════════════════════════════════════════════════════
# SECTION 1 — GRID CHARTS
# ═════════════════════════════════════════════════════════════════
st.subheader("Coin Excess Returns vs. Theme Average")
excess_n_bars = st.number_input("Display bars", 5, len(excess_cum), min(90, len(excess_cum)))

# Fetch the tail of the data and make a copy
excess_display = excess_cum.tail(int(excess_n_bars)).copy()

# Rebase all columns to start at 100 based on their first valid value in the display window
excess_display = excess_display.apply(
    lambda col: col - col.loc[col.first_valid_index()] + 100 if col.first_valid_index() is not None else col
)

for theme in sorted(coin_themes.unique()):
    coins = [c for c in coin_themes[coin_themes == theme].index if c in excess_display.columns]
    if not coins: continue
    with st.expander(f"Theme: {theme} ({len(coins)} coins)"):
        n_cols = min(4, len(coins))
        rows = (len(coins) + n_cols - 1) // n_cols
        fig = make_subplots(rows=rows, cols=n_cols, subplot_titles=coins, vertical_spacing=0.1)
        for i, coin in enumerate(coins):
            r, c = divmod(i, n_cols)
            fig.add_trace(go.Scatter(x=excess_display.index, y=excess_display[coin], mode="lines", showlegend=False), row=r+1, col=c+1)
        fig.update_layout(height=280*rows, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

# ═════════════════════════════════════════════════════════════════
# SECTION 2 — SCANNER TABLE (FIXED STYLING)
# ═════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("Convexity Scanner")
c_a, c_b, c_c = st.columns(3)
scan_n = c_a.number_input("Scan bars", 9, len(excess_cum), min(90, len(excess_cum)))
f_thr = c_b.slider("Flatness", 0.0, 0.5, 0.05)
m_scr = c_c.slider("Min Score", 1, 6, 3)

raw_results = {c: classify_curvature(excess_cum[c], int(scan_n), f_thr, m_scr) for c in excess_cum.columns}
classifications = {c: res[0] for c, res in raw_results.items()}
scores = {c: res[1] for c, res in raw_results.items()}
details = {c: res[2] for c, res in raw_results.items()}

# Build table
table_data = []
for theme in sorted(coin_themes.unique()):
    buys = [c for c, l in classifications.items() if l == "convex" and coin_themes[c] == theme]
    sells = [c for c, l in classifications.items() if l == "concave" and coin_themes[c] == theme]
    if buys or sells:
        table_data.append({"Theme": theme, "Buy (Convex ↑)": ", ".join(buys), "Sell (Concave ↓)": ", ".join(sells)})

if table_data:
    df_sig = pd.DataFrame(table_data).set_index("Theme")
    
    # Fix: Robust styling function that matches dataframe dimensions perfectly
    def style_buy_sell(df):
        style_df = pd.DataFrame('', index=df.index, columns=df.columns)
        if "Buy (Convex ↑)" in df.columns:
            style_df["Buy (Convex ↑)"] = "background-color: #0d3b1e; color: #4ade80"
        if "Sell (Concave ↓)" in df.columns:
            style_df["Sell (Concave ↓)"] = "background-color: #3b0d0d; color: #f87171"
        return style_df

    st.dataframe(df_sig.style.apply(style_buy_sell, axis=None), use_container_width=True)

# Debug expansion
with st.expander("Raw Scores Breakdown"):
    score_rows = [{"Coin": c, "Signal": classifications[c].upper(), "Score": scores[c], "R2": details[c].get("_r2", 0)} for c in sorted(excess_cum.columns)]
    sdf = pd.DataFrame(score_rows).set_index("Coin")
    # Using .map instead of .applymap for Pandas 2.x/3.x
    st.dataframe(sdf.style.map(lambda v: "color: #4ade80" if v == "CONVEX" else ("color: #f87171" if v == "CONCAVE" else ""), subset=["Signal"]), use_container_width=True)

# ═════════════════════════════════════════════════════════════════
# SECTION 3 — SINGLE LOOKUP
# ═════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("Single-Coin Lookup")
sel_coin = st.selectbox("Select coin", sorted(excess_cum.columns))
sel_period = st.number_input("Period", 5, len(excess_cum), min(90, len(excess_cum)), key="single_p")

if sel_coin:
    s_data = excess_cum[sel_coin].dropna().tail(int(sel_period))
    label = classifications.get(sel_coin, "flat")
    
    color_map = {
        "convex":  {"line": "#4ade80", "fill": "rgba(74, 222, 128, 0.15)"},
        "concave": {"line": "#f87171", "fill": "rgba(248, 113, 113, 0.15)"},
        "flat":    {"line": "#94a3b8", "fill": "rgba(148, 163, 184, 0.15)"}
    }
    
    cfg = color_map.get(label, color_map["flat"])
    
    fig_c = go.Figure()
    fig_c.add_trace(go.Scatter(
        x=s_data.index, y=s_data.values,
        mode="lines",
        line=dict(color=cfg["line"], width=2),
        fill="tozeroy",
        fillcolor=cfg["fill"],
        name=sel_coin
    ))
    fig_c.update_layout(
        title=f"{sel_coin} | Signal: {label.upper()} | Score: {scores.get(sel_coin)}",
        hovermode="x unified", height=400
    )
    st.plotly_chart(fig_c, use_container_width=True)
