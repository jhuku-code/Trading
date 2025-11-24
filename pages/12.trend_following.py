import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Trend Following", layout="wide")
st.title("Trend Following (Filtered Momentum Theme)")

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
    st.error("`price_theme` not found in session_state. Please load prices into `st.session_state['price_theme']` first.")
    st.stop()

# You should already have these from earlier steps in the app
# filtered_tickers_h, buy_list, short_list must be created before this block
if "filtered_tickers_h" not in st.session_state:
    st.error("`filtered_tickers_h` not found in session_state. Please compute and store it first.")
    st.stop()

filtered_tickers_h = st.session_state["filtered_tickers_h"]

# Optional: allow user to see the filtered tickers
with st.expander("Show filtered tickers used for trend following"):
    st.write(filtered_tickers_h)

# ---------------------------------------------------------
# KEEP ONLY FILTERED COINS
# ---------------------------------------------------------
df_short_h = df_h[filtered_tickers_h].copy()

# ---------------------------------------------------------
# PARAMETERS (kept for reference, even if not directly used below)
# ---------------------------------------------------------
base_length = 15
nw_start = 150  # 25 * 6 times/day
ss_wma_period = 90  # 15 days * 6 times/day
aema_period = 6 * base_length  # 15 days * 6 times/day
fatl_length = 6 * base_length  # 15 days * 6 times/day
jma_length = 6 * base_length  # 15 days * 6 times/day
nw_window = 6 * base_length  # 14 * 6 times/day
nw_r = 48.0  # 8 * 6 times/day
phase = 0.5
hold_days = 16

# ---------------------------------------------------------
# SMOOTHING FUNCTIONS (if you need them elsewhere in this page)
# ---------------------------------------------------------
def adaptive_ema(series, period):
    ema = series.copy()
    noise = np.zeros_like(series.values)
    for i in range(period, len(series)):
        sig = abs(series.iloc[i] - series.iloc[i - period])
        noise[i] = (
            noise[i - 1]
            + abs(series.iloc[i] - series.iloc[i - 1])
            - abs(series.iloc[i] - series.iloc[i - period])
        )
        noise_val = noise[i] if noise[i] != 0 else 1
        efratio = sig / noise_val
        slow_end = period * 5
        fast_end = max(period / 2.0, 1)
        avg_period = ((sig / noise_val) * (slow_end - fast_end)) + fast_end
        alpha = 2.0 / (1.0 + avg_period)
        ema.iloc[i] = ema.iloc[i - 1] + alpha * (series.iloc[i] - ema.iloc[i - 1])
    return ema


def jfatl(series, fatl_len, jma_len, phase):
    fatl = series.rolling(fatl_len).mean()
    e = 0.5 * (phase + 1)
    wma1 = fatl.rolling(jma_len).mean()
    wma2 = fatl.rolling(jma_len // 2).mean()
    return wma1 * e + wma2 * (1 - e)


def nadaraya_watson_vectorized(series, h, r, start_regression_at_bar):
    n = len(series)
    smoothed = np.full(n, np.nan)

    for t in range(start_regression_at_bar, n):
        indices = np.arange(0, t)
        distances = t - indices
        weights = (1 + (distances ** 2 / ((h ** 2) * 2 * r))) ** (-r)
        values = series.values[:t]
        smoothed[t] = np.sum(values * weights) / np.sum(weights)

    return pd.Series(smoothed, index=series.index)

# ---------------------------------------------------------
# BUY / SHORT LISTS
# ---------------------------------------------------------
# Expect these to be defined earlier in the script; you can also
# store them in session_state if you prefer.
if "buy_list" in st.session_state:
    buy_list = st.session_state["buy_list"]
elif "buy_list" in locals():
    buy_list = buy_list
else:
    buy_list = []

if "short_list" in st.session_state:
    short_list = st.session_state["short_list"]
elif "short_list" in locals():
    short_list = short_list
else:
    short_list = []

st.subheader("Signal Lists")

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Buy List**")
    st.write(buy_list if buy_list else "No tickers in buy list.")

with col_b:
    st.markdown("**Short / Sell List**")
    st.write(short_list if short_list else "No tickers in short list.")

# ---------------------------------------------------------
# RETURNS CALCULATION
# ---------------------------------------------------------
# 7-day and 30-day returns assuming 4H data: 7d=42 bars, 30d=180 bars
returns_7d = df_short_h.pct_change(42, fill_method=None).iloc[-1] * 100
returns_30d = df_short_h.pct_change(180, fill_method=None).iloc[-1] * 100

def get_return_table_df(ticker_list, label):
    rows = []
    for ticker in ticker_list:
        if ticker in returns_7d.index and ticker in returns_30d.index:
            rows.append(
                {
                    f"{label} Ticker": ticker,
                    "7-Day Return (%)": round(returns_7d[ticker], 2),
                    "30-Day Return (%)": round(returns_30d[ticker], 2),
                }
            )
    # Always add BTC as benchmark if available
    if "BTC" in returns_7d.index and "BTC" in returns_30d.index:
        rows.append(
            {
                f"{label} Ticker": "BTC",
                "7-Day Return (%)": round(returns_7d["BTC"], 2),
                "30-Day Return (%)": round(returns_30d["BTC"], 2),
            }
        )
    return pd.DataFrame(rows)

buy_return_table = get_return_table_df(buy_list, "Buy")
short_return_table = get_return_table_df(short_list, "Short")

# ---------------------------------------------------------
# DISPLAY RETURN TABLES
# ---------------------------------------------------------
st.subheader("Return Tables")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Buy Return Table**")
    if buy_return_table.empty:
        st.write("No return data for buy list.")
    else:
        st.dataframe(buy_return_table, use_container_width=True)

with col2:
    st.markdown("**Short Return Table**")
    if short_return_table.empty:
        st.write("No return data for short list.")
    else:
        st.dataframe(short_return_table, use_container_width=True)
