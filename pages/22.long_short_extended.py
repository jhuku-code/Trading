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
lookback = st.sidebar.slider("Lookback (for stats)", 30, 200, 90)
ma_window = st.sidebar.slider("MA Window", 10, 100, 30)

# =========================
# 🔍 PAIR SCANNER FUNCTION
# =========================
def analyze_pair(df, c1, c2):
    sub = df[[c1, c2]].dropna()

    if len(sub) < lookback:
        return None

    sub = sub.tail(lookback)

    spread = np.log(sub[c1]) - np.log(sub[c2])

    mean = spread.mean()
    std = spread.std()

    z = (spread.iloc[-1] - mean) / std if std != 0 else 0

    ma = spread.rolling(ma_window).mean().iloc[-1]

    return {
        "z": z,
        "spread": spread.iloc[-1],
        "mean": mean,
        "ma": ma
    }

# =========================
# 🧮 SCAN ALL THEMES
# =========================
mr_results = []
trend_results = []

for theme in themes:

    coins = theme_map_df[theme_map_df["theme"] == theme]["coin"].tolist()
    coins = [c for c in coins if c in price_df.columns]

    for c1, c2 in combinations(coins, 2):

        res = analyze_pair(price_df, c1, c2)
        if res is None:
            continue

        z = res["z"]
        spread = res["spread"]
        mean = res["mean"]
        ma = res["ma"]

        # =====================
        # 1️⃣ MEAN REVERSION
        # =====================
        if 1 < abs(z) < 2:
            if z > 0:
                pair = f"{c2}/{c1}"   # short c1, long c2
            else:
                pair = f"{c1}/{c2}"

            mr_results.append({
                "Theme": theme,
                "Pair": pair
            })

        # =====================
        # 2️⃣ TREND
        # =====================
        if spread > mean and spread > ma:
            pair = f"{c1}/{c2}"
            trend_results.append({
                "Theme": theme,
                "Pair": pair
            })

# =========================
# 📊 OUTPUT TABLES
# =========================
st.subheader("📉 Mean Reversion Opportunities (1σ–2σ)")

if mr_results:
    mr_df = pd.DataFrame(mr_results)
    st.dataframe(mr_df, use_container_width=True)
else:
    st.info("No signals found")

st.subheader("📈 Trend Opportunities (Above Mean & MA)")

if trend_results:
    trend_df = pd.DataFrame(trend_results)
    st.dataframe(trend_df, use_container_width=True)
else:
    st.info("No signals found")

# =========================
# 📊 INTERACTIVE CHART
# =========================
st.markdown("---")
st.subheader("📊 Pair Explorer")

all_coins = sorted(price_df.columns.tolist())

long_coin = st.selectbox("Long Coin", all_coins)
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

fig.add_trace(go.Scatter(x=df.index, y=df['ma'], name="MA", line=dict(color="white")))

fig.update_layout(height=500, hovermode="x unified")

st.plotly_chart(fig, use_container_width=True)
