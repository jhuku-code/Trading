import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Liquidation Pressure Index", layout="wide")
st.title("Liquidation Pressure Index Dashboard")

# ---------------------------------------------------------
# CHECK REQUIRED DATASETS
# ---------------------------------------------------------

required_keys = [
    "liq_long_data",
    "liq_short_data",
    "oi_data",
    "funding_data"
]

missing = [k for k in required_keys if k not in st.session_state]

if missing:

    st.error(
        "Required datasets not loaded.\n\n"
        "Please run these pages first:\n"
        "- Liquidations page\n"
        "- Open Interest page\n"
        "- Funding Rate page"
    )

    st.stop()

# ---------------------------------------------------------
# LOAD DATA FROM SESSION STATE
# ---------------------------------------------------------

liq_long = st.session_state["liq_long_data"]
liq_short = st.session_state["liq_short_data"]
oi_df = st.session_state["oi_data"]
funding_df = st.session_state["funding_data"]

# ---------------------------------------------------------
# USER CONTROLS
# ---------------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    z_window = st.slider("Z-score window", 50, 400, 200)

with col2:
    smoothing = st.slider("Indicator smoothing", 1, 48, 12)

with col3:
    btc_weight = st.slider("BTC weight in Market LPI", 0.0, 1.0, 0.7)

# ---------------------------------------------------------
# ALIGN SYMBOL UNIVERSE
# ---------------------------------------------------------

common_symbols = (
    set(liq_long.columns)
    & set(liq_short.columns)
    & set(oi_df.columns)
    & set(funding_df.columns)
)

common_symbols = list(common_symbols)

if len(common_symbols) == 0:

    st.error("No common symbols across datasets.")
    st.stop()

liq_long = liq_long[common_symbols]
liq_short = liq_short[common_symbols]
oi_df = oi_df[common_symbols]
funding_df = funding_df[common_symbols]

st.write("Symbols used for LPI:", len(common_symbols))

# ---------------------------------------------------------
# BUILD LPI COMPONENTS
# ---------------------------------------------------------

liq_signal = np.log((liq_long + 1) / (liq_short + 1))

oi_velocity = oi_df.pct_change(6)

funding_accel = funding_df.diff()

# ---------------------------------------------------------
# Z-SCORES
# ---------------------------------------------------------

liq_z = (
    liq_signal - liq_signal.rolling(z_window).mean()
) / liq_signal.rolling(z_window).std()

oi_z = (
    oi_velocity - oi_velocity.rolling(z_window).mean()
) / oi_velocity.rolling(z_window).std()

fund_z = (
    funding_accel - funding_accel.rolling(z_window).mean()
) / funding_accel.rolling(z_window).std()

# ---------------------------------------------------------
# FINAL LPI
# ---------------------------------------------------------

lpi_df = 0.4 * liq_z + 0.35 * oi_z + 0.25 * fund_z

lpi_df = lpi_df.rolling(smoothing).mean()

# ---------------------------------------------------------
# MARKET LPI (BTC + ETH)
# ---------------------------------------------------------

btc_cols = [c for c in lpi_df.columns if "BTC" in c]
eth_cols = [c for c in lpi_df.columns if "ETH" in c]

btc_lpi = lpi_df[btc_cols].mean(axis=1)
eth_lpi = lpi_df[eth_cols].mean(axis=1)

market_lpi = btc_weight * btc_lpi + (1 - btc_weight) * eth_lpi

# ---------------------------------------------------------
# MARKET LPI CHART
# ---------------------------------------------------------

st.subheader("BTC + ETH Weighted Market LPI")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=market_lpi.index,
        y=market_lpi,
        name="Market LPI"
    )
)

fig.update_layout(
    height=500,
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# TOP / BOTTOM COINS
# ---------------------------------------------------------

st.subheader("Top / Bottom Coins by Latest LPI")

latest = lpi_df.iloc[-1].dropna().sort_values(ascending=False)

col1, col2 = st.columns(2)

with col1:

    st.markdown("### Highest LPI (Long Squeeze Risk)")

    st.dataframe(latest.head(20))

with col2:

    st.markdown("### Lowest LPI (Short Squeeze Risk)")

    st.dataframe(latest.tail(20))

# ---------------------------------------------------------
# COIN LPI TIME SERIES
# ---------------------------------------------------------

st.subheader("Coin LPI Time Series")

coin = st.selectbox("Select coin", lpi_df.columns)

fig2 = go.Figure()

fig2.add_trace(
    go.Scatter(
        x=lpi_df.index,
        y=lpi_df[coin],
        name="LPI"
    )
)

fig2.update_layout(
    height=500,
    hovermode="x unified"
)

st.plotly_chart(fig2, use_container_width=True)
