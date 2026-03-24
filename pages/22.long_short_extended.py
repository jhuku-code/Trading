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
lookback = st.sidebar.slider("Lookback", 30, 200, 90)
ma_window = st.sidebar.slider("MA Window", 10, 100, 30)

# =========================
# 🔍 CORE FUNCTION
# =========================
def analyze_pair(df, c1, c2, lookback, ma_window):
    sub = df[[c1, c2]].dropna()

    if len(sub) < lookback:
        return None

    sub = sub.tail(lookback)

    spread = np.log(sub[c1]) - np.log(sub[c2])

    mean = spread.mean()
    std = spread.std()

    if std == 0:
        return None

    current = spread.iloc[-1]
    z = (current - mean) / std

    ma = spread.rolling(ma_window).mean().iloc[-1]

    return current, mean, std, z, ma

# =========================
# 🚀 CACHED SCANNER
# =========================
@st.cache_data
def run_scanner(price_df, theme_map_df, lookback, ma_window):

    mr_dict = {}
    trend_dict = {}

    for theme in sorted(theme_map_df["theme"].unique()):

        coins = theme_map_df[theme_map_df["theme"] == theme]["coin"].tolist()
        coins = [c for c in coins if c in price_df.columns]

        mr_pairs = []
        trend_pairs = []

        for c1, c2 in combinations(coins, 2):

            res = analyze_pair(price_df, c1, c2, lookback, ma_window)
            if res is None:
                continue

            current, mean, std, z, ma = res

            # =====================
            # 1️⃣ MEAN REVERSION
            # =====================
            if 1 < abs(z) < 2:
                if z > 0:
                    pair = f"{c2}/{c1}"
                else:
                    pair = f"{c1}/{c2}"

                mr_pairs.append(pair)

            # =====================
            # 2️⃣ TREND (FIXED)
            # =====================
            if current > mean and current > ma:
                pair = f"{c1}/{c2}"
                trend_pairs.append(pair)

        # Join into single row
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
mr_df, trend_df = run_scanner(price_df, theme_map_df, lookback, ma_window)

# =========================
# 📊 TABLE OUTPUT
# =========================
st.subheader("📉 Mean Reversion (1σ–2σ)")

if not mr_df.empty:
    st.dataframe(mr_df, use_container_width=True)
else:
    st.info("No signals")

st.subheader("📈 Trend (Above Mean & MA)")

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
