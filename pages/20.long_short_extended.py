# pages/Long_Short_Extended.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from itertools import combinations

st.set_page_config(page_title="Long / Short Extended", layout="wide")
st.title("📊 Long / Short Extended Scanner")

# =========================
# 🔁 SESSION DATA
# =========================
if "price_theme" not in st.session_state:
    st.error("Run Themes Tracker first")
    st.stop()

if "ticker_to_theme" not in st.session_state:
    st.error("ticker_to_theme missing")
    st.stop()

price_df = st.session_state["price_theme"].copy()
ticker_to_theme = st.session_state["ticker_to_theme"]

price_df.columns = price_df.columns.str.upper()

# =========================
# 🧠 BUILD THEME MAP
# =========================
theme_map_df = pd.DataFrame({
    "coin": list(ticker_to_theme.keys()),
    "theme": list(ticker_to_theme.values())
})

themes = sorted(theme_map_df["theme"].unique())

# =========================
# ⚙️ PARAMETERS
# =========================
st.sidebar.header("Filter Parameters")
ma_window = st.sidebar.slider("Spread MA Window", 10, 100, 30)
trend_ma_window = st.sidebar.slider("Long Leg Trend MA Window", 5, 100, 10)

# =========================
# 🔍 CORE FUNCTION (FULL HISTORY)
# =========================
def analyze_pair(df, c1, c2, ma_window):
    sub = df[[c1, c2]].dropna()

    if len(sub) < ma_window + 5:
        return None

    spread = np.log(sub[c1]) - np.log(sub[c2])

    # ✅ FULL HISTORY MEAN & STD
    mean = spread.mean()
    std = spread.std()

    if std == 0 or np.isnan(std):
        return None

    current = spread.iloc[-1]

    ma_series = spread.rolling(ma_window).mean()
    ma = ma_series.iloc[-1]

    if np.isnan(ma):
        return None

    return current, mean, std, ma

# =========================
# 🚀 CACHED SCANNER
# =========================
@st.cache_data
def run_scanner(price_df, theme_map_df, ma_window, trend_ma_window):

    mr_dict = {}
    trend_dict = {}

    # Pre-calculate the trend moving average for all coins to optimize the loop
    trend_ma_df = price_df.rolling(window=trend_ma_window).mean()
    is_uptrend = price_df.iloc[-1] > trend_ma_df.iloc[-1]

    for theme in sorted(theme_map_df["theme"].unique()):

        coins = theme_map_df[theme_map_df["theme"] == theme]["coin"].tolist()
        coins = [c for c in coins if c in price_df.columns]

        mr_pairs = []
        trend_pairs = []

        for c1, c2 in combinations(coins, 2):

            # =====================
            # ABSOLUTE TREND FILTER 
            # Long candidate (c1) must be > its own moving average
            # =====================
            if not is_uptrend[c1]:
                continue

            res = analyze_pair(price_df, c1, c2, ma_window)
            if res is None:
                continue

            current, mean, std, ma = res

            # =====================
            # 1️⃣ MEAN REVERSION
            # mean < spread < mean + 1σ
            # =====================
            if (current > mean) and (current < mean + std):
                pair = f"{c1}/{c2}"
                mr_pairs.append(pair)

            # =====================
            # 2️⃣ TREND (STRICT)
            # spread > mean AND spread > MA
            # =====================
            if (current > mean) and (current > ma):
                pair = f"{c1}/{c2}"
                trend_pairs.append(pair)

        if mr_pairs:
            mr_dict[theme] = "; ".join(sorted(set(mr_pairs)))

        if trend_pairs:
            trend_dict[theme] = "; ".join(sorted(set(trend_pairs)))

    mr_df = pd.DataFrame.from_dict(mr_dict, orient="index", columns=["Pairs"])
    trend_df = pd.DataFrame.from_dict(trend_dict, orient="index", columns=["Pairs"])

    mr_df.index.name = "Theme"
    trend_df.index.name = "Theme"

    return mr_df, trend_df

# =========================
# 🧮 RUN SCAN
# =========================
mr_df, trend_df = run_scanner(price_df, theme_map_df, ma_window, trend_ma_window)

# =========================
# 📊 TABLE OUTPUT
# =========================
st.subheader("📉 Mean Reversion (Mean → +1σ)")

if not mr_df.empty:
    st.dataframe(mr_df, use_container_width=True)
else:
    st.info("No signals")

st.subheader("📈 Trend (Spread > Mean AND > MA)")

if not trend_df.empty:
    st.dataframe(trend_df, use_container_width=True)
else:
    st.info("No signals")

# =========================
# 📊 INTERACTIVE CHART
# =========================
st.markdown("---")
st.subheader("📊 Pair Explorer")

all_coins = sorted(price_df.columns.tolist())

col1, col2 = st.columns(2)
with col1:
    long_coin = st.selectbox("Long Coin", all_coins)
with col2:
    short_coin = st.selectbox("Short Coin", all_coins, index=1)

df = price_df[[long_coin, short_coin]].dropna().copy()

df['spread'] = np.log(df[long_coin]) - np.log(df[short_coin])

# ✅ FULL HISTORY MEAN & STD
mean = df['spread'].mean()
std = df['spread'].std()

df['mean'] = mean
df['upper1'] = mean + std
df['lower1'] = mean - std
df['upper2'] = mean + 2 * std
df['lower2'] = mean - 2 * std

df['ma'] = df['spread'].rolling(ma_window).mean()

# =========================
# 📈 PLOT
# =========================
fig = go.Figure()

fig.add_trace(go.Scatter(x=df.index, y=df['spread'], name="Spread"))

fig.add_trace(go.Scatter(x=df.index, y=df['mean'], name="Mean", line=dict(dash="dash")))

fig.add_trace(go.Scatter(x=df.index, y=df['upper1'], name="+1σ", line=dict(dash="dot")))
fig.add_trace(go.Scatter(x=df.index, y=df['lower1'], name="-1σ", line=dict(dash="dot")))

fig.add_trace(go.Scatter(x=df.index, y=df['upper2'], name="+2σ", line=dict(dash="dash")))
fig.add_trace(go.Scatter(x=df.index, y=df['lower2'], name="-2σ", line=dict(dash="dash")))

fig.add_trace(go.Scatter(x=df.index, y=df['ma'], name="MA"))

fig.update_layout(height=500, hovermode="x unified")

st.plotly_chart(fig, use_container_width=True)
