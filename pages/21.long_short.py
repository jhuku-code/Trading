# 21.long_short_ideas.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Long / Short Ideas", layout="wide")

st.title("📊 Long / Short Idea Generator")

# =========================
# 🔁 LOAD DATA FROM SESSION
# =========================

if "price_theme" not in st.session_state:
    st.error("❌ price_theme not found. Run Themes Tracker page first.")
    st.stop()

if "ticker_to_theme" not in st.session_state:
    st.error("❌ ticker_to_theme not found.")
    st.stop()

if "beta_df" not in st.session_state:
    st.error("❌ beta_df not found. Run Clustering Vol page first.")
    st.stop()

price_df = st.session_state["price_theme"].copy()
ticker_to_theme = st.session_state["ticker_to_theme"]
beta_df = st.session_state["beta_df"].copy()

# Normalize
price_df.columns = price_df.columns.str.upper()
beta_df['coin'] = beta_df['coin'].str.upper()

# =========================
# 🧠 BUILD THEME MAP DF
# =========================

theme_map_df = pd.DataFrame({
    "coin": list(ticker_to_theme.keys()),
    "theme": list(ticker_to_theme.values())
})

# =========================
# 🎛️ SIDEBAR
# =========================

st.sidebar.header("⚙️ Trade Setup")

all_coins = sorted(price_df.columns.tolist())

# ✅ AUTOCOMPLETE
long_coin = st.sidebar.selectbox(
    "Select Long Coin",
    all_coins
)

# Get theme
theme = theme_map_df.loc[
    theme_map_df['coin'] == long_coin, 'theme'
].values[0]

# Same theme coins
same_theme_coins = theme_map_df[
    theme_map_df['theme'] == theme
]['coin'].tolist()

same_theme_coins = [c for c in same_theme_coins if c != long_coin and c in all_coins]

if len(same_theme_coins) == 0:
    st.warning("No same-theme coins available.")
    st.stop()

short_coin = st.sidebar.selectbox(
    "Select Short Coin",
    sorted(same_theme_coins)
)

window = st.sidebar.slider("Rolling Window", 30, 180, 90)
capital = st.sidebar.number_input("Capital ($)", value=10000)

# =========================
# 📊 DATA PREP
# =========================

df = price_df[[long_coin, short_coin]].dropna().copy()

# Log spread
df['spread'] = np.log(df[long_coin]) - np.log(df[short_coin])

# Rolling stats
df['mean'] = df['spread'].rolling(window).mean()
df['std'] = df['spread'].rolling(window).std()

df['upper'] = df['mean'] + 2 * df['std']
df['lower'] = df['mean'] - 2 * df['std']

# Z-score
df['zscore'] = (df['spread'] - df['mean']) / df['std']

df = df.dropna()

if df.empty:
    st.warning("Not enough data.")
    st.stop()

# =========================
# 📈 CHART
# =========================

st.subheader(f"{long_coin} vs {short_coin} (Spread)")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df.index, y=df['spread'],
    name="Spread"
))

fig.add_trace(go.Scatter(
    x=df.index, y=df['mean'],
    name="Mean",
    line=dict(dash="dash")
))

fig.add_trace(go.Scatter(
    x=df.index, y=df['upper'],
    name="+2 STD",
    line=dict(dash="dash")
))

fig.add_trace(go.Scatter(
    x=df.index, y=df['lower'],
    name="-2 STD",
    line=dict(dash="dash")
))

st.plotly_chart(fig, use_container_width=True)

# =========================
# 📊 Z-SCORE
# =========================

current_z = df['zscore'].iloc[-1]

st.subheader("Z-Score Signal")

col1, col2 = st.columns(2)

col1.metric("Z-Score", round(current_z, 2))
col2.metric("Window", window)

if current_z > 2:
    st.error("Overbought → SHORT spread")

elif current_z < -2:
    st.success("Oversold → LONG spread")

else:
    st.info("Neutral")

# =========================
# ⚖️ BETA HEDGE
# =========================

try:
    beta_long = beta_df.loc[beta_df['coin'] == long_coin, 'beta'].values[0]
    beta_short = beta_df.loc[beta_df['coin'] == short_coin, 'beta'].values[0]
except:
    st.error("Missing beta data")
    st.stop()

hedge_ratio = beta_long / beta_short if beta_short != 0 else np.nan

short_pos = capital / (1 + hedge_ratio)
long_pos = capital - short_pos

# =========================
# 📌 OUTPUT
# =========================

st.subheader("Trade Setup")

col1, col2 = st.columns(2)

with col1:
    st.write(f"Theme: **{theme}**")
    st.write(f"{long_coin} Beta: {beta_long:.2f}")
    st.write(f"{short_coin} Beta: {beta_short:.2f}")

with col2:
    st.write(f"Long {long_coin}: ${long_pos:.2f}")
    st.write(f"Short {short_coin}: ${short_pos:.2f}")
    st.write(f"Hedge Ratio: {hedge_ratio:.2f}")

st.markdown("---")

st.write("""
### 🧠 Interpretation
- Z > 2 → spread expensive → short long coin  
- Z < -2 → spread cheap → long long coin  
- Beta neutral sizing removes market direction risk  
""")
