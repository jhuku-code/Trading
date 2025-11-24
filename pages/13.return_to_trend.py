import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Return-to-Trend Signals", layout="wide")
st.title("Return-to-Trend (Percentile-Based)")

# ---------------------------------------------------------
# REFRESH BUTTON
# ---------------------------------------------------------
if st.button("🔄 Refresh data"):
    st.rerun()

# ---------------------------------------------------------
# INPUT DATA FROM SESSION STATE
# ---------------------------------------------------------
df_h = st.session_state.get("price_theme", None)

if df_h is None:
    st.error(
        "`price_theme` not found in session_state. "
        "Please load prices into `st.session_state['price_theme']` first."
    )
    st.stop()

# Ensure index is sorted and we work on a copy
df_h = df_h.sort_index().copy()

# Use the full universe (ignore filtered_tickers_h)
df_short_h = df_h.copy()

# ---------------------------------------------------------
# PARAMETER CONTROLS
# ---------------------------------------------------------
st.subheader("Strategy Parameters")

col1, col2 = st.columns(2)
with col1:
    ema_period = st.number_input(
        "EMA period",
        min_value=5,
        max_value=300,
        value=50,
        step=5,
        help="Period for EMA smoothing.",
    )
    slope_window = st.number_input(
        "Slope window",
        min_value=5,
        max_value=300,
        value=30,
        step=5,
        help="Window size (in bars) to compute EMA slope.",
    )

with col2:
    dev_pct_thresh = st.slider(
        "Deviation percentile threshold",
        min_value=0.0,
        max_value=0.5,
        value=0.3,
        step=0.01,
        help="Bottom X% for deviation. Example: 0.3 = bottom 30% (pullback).",
    )
    slope_pct_thresh = st.slider(
        "Slope percentile threshold",
        min_value=0.5,
        max_value=1.0,
        value=0.7,
        step=0.01,
        help="Top X% for slope. Example: 0.7 = top 30% uptrend strength.",
    )

# ---------------------------------------------------------
# STRATEGY FUNCTION
# ---------------------------------------------------------
def return_to_trend_signals_percentile(
    df_prices,
    ema_period=50,
    slope_window=20,
    dev_pct_thresh=0.2,
    slope_pct_thresh=0.8,
):
    """
    Adaptive Return-to-Trend Strategy using percentile ranks across coins.

    Parameters:
    - df_prices: Price DataFrame with datetime index and coin symbols as columns
    - ema_period: Period for EMA smoothing
    - slope_window: Window to compute EMA slope
    - dev_pct_thresh: Percentile threshold for deviation (e.g., 0.2 = bottom 20%)
    - slope_pct_thresh: Percentile threshold for slope (e.g., 0.8 = top 20%)

    Returns:
    - signals: DataFrame with 1 (buy), -1 (short), 0 (no trade)
    - ema: EMA DataFrame
    - ema_slope: Slope of EMA
    - deviation: % deviation from EMA
    - dev_rank, slope_rank: percentile ranks used for filtering
    """
    # Step 1: Compute EMA
    ema = df_prices.ewm(span=ema_period, adjust=False).mean()

    # Step 2: Compute EMA Slope
    ema_slope = ema.diff(slope_window) / slope_window

    # Step 3: Compute % Deviation
    deviation = (df_prices - ema) / ema

    # Step 4: Compute cross-sectional percentile ranks
    dev_rank = deviation.rank(axis=1, pct=True)
    slope_rank = ema_slope.rank(axis=1, pct=True)

    # Step 5: Signals
    signals = pd.DataFrame(0, index=df_prices.index, columns=df_prices.columns)

    # Buy: pulled back but strong uptrend
    buy_condition = (dev_rank < dev_pct_thresh) & (slope_rank > slope_pct_thresh)

    # Sell: extended but strong downtrend
    short_condition = (dev_rank > (1 - dev_pct_thresh)) & (
        slope_rank < (1 - slope_pct_thresh)
    )

    signals[buy_condition] = 1
    signals[short_condition] = -1

    return signals, ema, ema_slope, deviation, dev_rank, slope_rank


# ---------------------------------------------------------
# RUN STRATEGY
# ---------------------------------------------------------
signals, ema, ema_slope, deviation, dev_rank, slope_rank = (
    return_to_trend_signals_percentile(
        df_short_h,
        ema_period=int(ema_period),
        slope_window=int(slope_window),
        dev_pct_thresh=float(dev_pct_thresh),
        slope_pct_thresh=float(slope_pct_thresh),
    )
)

# Get last row signals
last_row = signals.iloc[-1]
buy_signals = last_row[last_row == 1].index.tolist()
sell_signals = last_row[last_row == -1].index.tolist()

# ---------------------------------------------------------
# DISPLAY SIGNALS
# ---------------------------------------------------------
st.subheader("Latest Signals (Last Bar)")

col_buy, col_sell = st.columns(2)

with col_buy:
    st.markdown("**Buy Signals**")
    if buy_signals:
        st.write(buy_signals)
    else:
        st.write("No buy signals.")

with col_sell:
    st.markdown("**Sell Signals**")
    if sell_signals:
        st.write(sell_signals)
    else:
        st.write("No sell signals.")

# Optional: show full last-row signal map as a quick overview
with st.expander("Show full signal vector for last bar (all coins)"):
    st.dataframe(last_row.to_frame("signal").T, use_container_width=True)
