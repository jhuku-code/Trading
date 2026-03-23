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

if "coin_btc_betas" not in st.session_state:
    st.error("❌ coin_btc_betas not found. Run Clustering Vol page first.")
    st.stop()

price_df = st.session_state["price_theme"].copy()
ticker_to_theme = st.session_state["ticker_to_theme"]
beta_df = st.session_state["coin_btc_betas"].copy()

# Normalize
price_df.columns = price_df.columns.str.upper()
beta_df['Coin'] = beta_df['Coin'].str.upper()

# =========================
# 🧠 BUILD THEME MAP
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

long_coin = st.sidebar.selectbox("Select Long Coin", all_coins)

theme = theme_map_df.loc[
    theme_map_df['coin'] == long_coin, 'theme'
].values[0]

same_theme = theme_map_df[
    theme_map_df['theme'] == theme
]['coin'].tolist()

same_theme = [c for c in same_theme if c != long_coin and c in all_coins]

if not same_theme:
    st.warning("No same-theme coins available.")
    st.stop()

short_coin = st.sidebar.selectbox("Select Short Coin", sorted(same_theme))

capital = st.sidebar.number_input("Capital ($)", value=10000)

# =========================
# 📊 DATA PREP
# =========================

df = price_df[[long_coin, short_coin]].dropna().copy()

# Log spread
df['spread'] = np.log(df[long_coin]) - np.log(df[short_coin])

# Static mean & std
mean = df['spread'].mean()
std = df['spread'].std()

df['mean'] = mean
df['upper'] = mean + 2 * std
df['lower'] = mean - 2 * std

# Z-score
df['zscore'] = (df['spread'] - mean) / std

# =========================
# 📈 SPREAD CHART
# =========================

st.subheader(f"📉 Spread: {long_coin} vs {short_coin}")

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

fig.update_layout(height=450)

st.plotly_chart(fig, use_container_width=True)

# =========================
# 📈 NORMALIZED PRICE CHART
# =========================

st.subheader(f"📊 Normalized Price (Base = 100)")

norm_df = df[[long_coin, short_coin]].copy()

# Normalize both to 100
norm_df[long_coin] = norm_df[long_coin] / norm_df[long_coin].iloc[0] * 100
norm_df[short_coin] = norm_df[short_coin] / norm_df[short_coin].iloc[0] * 100

fig_norm = go.Figure()

fig_norm.add_trace(go.Scatter(
    x=norm_df.index,
    y=norm_df[long_coin],
    name=f"{long_coin} (Long)"
))

fig_norm.add_trace(go.Scatter(
    x=norm_df.index,
    y=norm_df[short_coin],
    name=f"{short_coin} (Short)"
))

fig_norm.update_layout(height=400)

st.plotly_chart(fig_norm, use_container_width=True)

# =========================
# 📊 Z-SCORE SIGNAL
# =========================

current_z = df['zscore'].iloc[-1]

st.subheader("📊 Z-Score Signal")

col1, col2 = st.columns(2)

col1.metric("Z-Score", round(current_z, 2))
col2.metric("Mean", round(mean, 4))

if current_z > 2:
    st.error("⚠️ Overbought → SHORT spread")

elif current_z < -2:
    st.success("✅ Oversold → LONG spread")

else:
    st.info("Neutral")

# =========================
# ⚖️ BETA HEDGE
# =========================

try:
    beta_long = beta_df.loc[
        beta_df['Coin'] == long_coin, 'BTC_Beta'
    ].values[0]

    beta_short = beta_df.loc[
        beta_df['Coin'] == short_coin, 'BTC_Beta'
    ].values[0]

except:
    st.error("Missing beta data")
    st.stop()

hedge_ratio = beta_long / beta_short if beta_short != 0 else np.nan

short_pos = capital / (1 + hedge_ratio)
long_pos = capital - short_pos

# =========================
# 📌 OUTPUT
# =========================

st.subheader("📌 Trade Setup")

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
### 🧠 Interpretation (Static Mean Model)
- Mean = long-term equilibrium level  
- +2σ → statistically expensive → short spread  
- -2σ → statistically cheap → long spread  

This assumes **mean reversion to a fixed anchor**, not a drifting regime.
""")
