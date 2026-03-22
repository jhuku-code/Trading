# 3.long_short_ideas.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Long / Short Ideas", layout="wide")

st.title("📊 Long / Short Idea Generator")

# =========================
# 🔁 LOAD SHARED DATA
# =========================
# Expecting these to be stored from previous pages

if "price_df" not in st.session_state:
    st.error("❌ price_df not found. Run 'Themes Tracker' page first.")
    st.stop()

if "theme_map_df" not in st.session_state:
    st.error("❌ theme_map_df not found.")
    st.stop()

if "beta_df" not in st.session_state:
    st.error("❌ beta_df not found. Run 'Clustering Vol' page first.")
    st.stop()

price_df = st.session_state["price_df"]
theme_map_df = st.session_state["theme_map_df"]
beta_df = st.session_state["beta_df"]

# Ensure uppercase consistency
price_df.columns = price_df.columns.str.upper()
theme_map_df['coin'] = theme_map_df['coin'].str.upper()
beta_df['coin'] = beta_df['coin'].str.upper()

# =========================
# 🎛️ SIDEBAR INPUTS
# =========================

st.sidebar.header("⚙️ Trade Setup")

# ✅ AUTOCOMPLETE (Dropdown instead of text input)
all_coins = sorted(price_df.columns.tolist())

long_coin = st.sidebar.selectbox(
    "Select Long Coin",
    all_coins
)

# Get theme
try:
    theme = theme_map_df.loc[
        theme_map_df['coin'] == long_coin, 'theme'
    ].values[0]
except:
    st.error(f"Theme not found for {long_coin}")
    st.stop()

# Filter same-theme coins
same_theme_coins = theme_map_df[
    theme_map_df['theme'] == theme
]['coin'].tolist()

same_theme_coins = [c for c in same_theme_coins if c in all_coins and c != long_coin]

if len(same_theme_coins) == 0:
    st.warning("No alternative coins in same theme.")
    st.stop()

short_coin = st.sidebar.selectbox(
    "Select Short Coin (Same Theme)",
    sorted(same_theme_coins)
)

# Rolling window
window = st.sidebar.slider("Rolling Window (days)", 30, 180, 90)

# Capital
capital = st.sidebar.number_input("Total Capital ($)", value=10000)

# =========================
# 📊 DATA PREP
# =========================

df = price_df[[long_coin, short_coin]].dropna().copy()

# Use log spread (more stable)
df['ratio'] = np.log(df[long_coin]) - np.log(df[short_coin])

# Rolling stats
df['mean'] = df['ratio'].rolling(window).mean()
df['std'] = df['ratio'].rolling(window).std()

df['upper'] = df['mean'] + 2 * df['std']
df['lower'] = df['mean'] - 2 * df['std']

# Z-score
df['zscore'] = (df['ratio'] - df['mean']) / df['std']

df = df.dropna()

if df.empty:
    st.warning("Not enough data after rolling window.")
    st.stop()

# =========================
# 📈 CHART
# =========================

st.subheader(f"📉 Spread: {long_coin} / {short_coin}")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df.index, y=df['ratio'],
    name='Spread',
    line=dict(width=2)
))

fig.add_trace(go.Scatter(
    x=df.index, y=df['mean'],
    name='Mean',
    line=dict(dash='dash')
))

fig.add_trace(go.Scatter(
    x=df.index, y=df['upper'],
    name='+2 STD',
    line=dict(dash='dash')
))

fig.add_trace(go.Scatter(
    x=df.index, y=df['lower'],
    name='-2 STD',
    line=dict(dash='dash')
))

fig.update_layout(
    height=500,
    margin=dict(l=10, r=10, t=40, b=10)
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# 📊 Z-SCORE SIGNAL
# =========================

current_z = df['zscore'].iloc[-1]

st.subheader("📊 Z-Score Signal")

col1, col2 = st.columns(2)

col1.metric("Current Z-Score", round(current_z, 2))
col2.metric("Rolling Window", f"{window} days")

if current_z > 2:
    st.error("⚠️ Overbought Spread → SHORT Long Coin / LONG Short Coin")

elif current_z < -2:
    st.success("✅ Oversold Spread → LONG Long Coin / SHORT Short Coin")

else:
    st.info("Neutral Zone")

# =========================
# ⚖️ BETA + POSITION SIZING
# =========================

try:
    beta_long = beta_df.loc[beta_df['coin'] == long_coin, 'beta'].values[0]
    beta_short = beta_df.loc[beta_df['coin'] == short_coin, 'beta'].values[0]
except:
    st.error("Beta not found for one of the coins.")
    st.stop()

hedge_ratio = beta_long / beta_short if beta_short != 0 else np.nan

short_position = capital / (1 + hedge_ratio)
long_position = capital - short_position

# =========================
# 📌 OUTPUT
# =========================

st.subheader("📌 Trade Setup")

col1, col2 = st.columns(2)

with col1:
    st.write(f"**Theme:** {theme}")
    st.write("### Betas")
    st.write(f"{long_coin}: {beta_long:.2f}")
    st.write(f"{short_coin}: {beta_short:.2f}")

with col2:
    st.write("### Position Sizing (Delta Neutral)")
    st.write(f"Long {long_coin}: ${long_position:.2f}")
    st.write(f"Short {short_coin}: ${short_position:.2f}")
    st.write(f"Hedge Ratio (L/S): {hedge_ratio:.2f}")

# =========================
# 🧠 EXTRA INSIGHT
# =========================

st.subheader("🧠 Interpretation")

st.write("""
- Z-score measures how stretched the spread is relative to history  
- +2 → statistically expensive → mean reversion expected  
- -2 → statistically cheap → bounce expected  

This is a **market-neutral relative value trade**, not a directional bet.
""")
