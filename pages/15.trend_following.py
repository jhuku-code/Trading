import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Trend Following System", layout="wide")
st.title("📈 Trend Following (Filtered & Ranked)")

# ---------------------------------------------------------
# SIDEBAR CONTROLS & TIMEFRAME HANDLING
# ---------------------------------------------------------
st.sidebar.header("Settings")
timeframe = st.sidebar.radio("Data Timeframe", ["Daily", "4-Hour"])

# Set multipliers based on timeframe
if timeframe == "4-Hour":
    tf_mult = 6  # 6 bars per day
elif timeframe == "Daily":
    tf_mult = 1  # 1 bar per day

# Base parameters dynamically adjusted by timeframe
base_length = 15
aema_period = tf_mult * base_length
fatl_length = tf_mult * base_length
jma_length = tf_mult * base_length
nw_r = 8.0 * tf_mult 
phase = 0.5

# Lookbacks for returns
lookback_7d = 7 * tf_mult
lookback_30d = 30 * tf_mult

if st.sidebar.button("🔄 Refresh Data"):
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

# Ensure index is sorted, drop completely empty columns, and work on a copy
df_h = df_h.sort_index().dropna(axis=1, how='all').copy()

# ---------------------------------------------------------
# SMOOTHING FUNCTIONS
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

def nadaraya_watson_optimized(series, h, r):
    """
    Optimized to only calculate the last value to prevent Streamlit from freezing.
    Returns the most recent smoothed value.
    """
    n = len(series)
    if n == 0: return np.nan
    
    # We only need the very last smoothed point for the signal
    t = n - 1
    indices = np.arange(0, t)
    distances = t - indices
    weights = (1 + (distances ** 2 / ((h ** 2) * 2 * r))) ** (-r)
    values = series.values[:t]
    
    if np.sum(weights) == 0:
        return np.nan
        
    return np.sum(values * weights) / np.sum(weights)

# ---------------------------------------------------------
# SIGNAL GENERATION & RANKING ENGINE
# ---------------------------------------------------------
st.write("Processing signals... This may take a moment depending on the universe size.")

buy_list_data = []
sell_list_data = []

# Process only the last 200 bars (adjusted for timeframe) to save compute time
calculation_window = max(200, lookback_30d + aema_period + 10)

for ticker in df_h.columns:
    series = df_h[ticker].dropna()
    
    if len(series) < calculation_window:
        continue  # Skip if not enough data
        
    # Slice the series for faster processing
    series_sliced = series.iloc[-calculation_window:]
    
    # 1. Calculate Indicators
    aema = adaptive_ema(series_sliced, aema_period).iloc[-1]
    jtl = jfatl(series_sliced, fatl_length, jma_length, phase).iloc[-1]
    nw = nadaraya_watson_optimized(series_sliced, h=base_length, r=nw_r)
    
    current_price = series_sliced.iloc[-1]
    
    # 2. Calculate Returns (Momentum & Overextension)
    ret_7d = (current_price / series_sliced.iloc[-lookback_7d] - 1) * 100 if len(series_sliced) >= lookback_7d else 0
    ret_30d = (current_price / series_sliced.iloc[-lookback_30d] - 1) * 100 if len(series_sliced) >= lookback_30d else 0
    
    # 3. Evaluate "2 out of 3" Conditions
    # Booleans evaluate to 1 (True) or 0 (False)
    bullish_conditions = (current_price > aema) + (current_price > jtl) + (current_price > nw)
    bearish_conditions = (current_price < aema) + (current_price < jtl) + (current_price < nw)
    
    # 4. Generate Signals & Rank
    if bullish_conditions >= 2:
        # BUY RANKING: Reward 7d momentum, but heavily penalize if 30d return is massive
        # e.g., A coin up 10% this week, but up 200% this month gets a lower score.
        buy_score = ret_7d - (ret_30d * 0.4) 
        
        buy_list_data.append({
            "Ticker": ticker,
            "Price": round(current_price, 4),
            "7D Return (%)": round(ret_7d, 2),
            "30D Return (%)": round(ret_30d, 2),
            "Rank Score": round(buy_score, 2),
            "Indicators Met": bullish_conditions
        })
        
    elif bearish_conditions >= 2:
        # SELL RANKING: Reward 7d drops, but penalize if it's already dumped 90% over 30 days
        sell_score = -ret_7d + (ret_30d * 0.4)
        
        sell_list_data.append({
            "Ticker": ticker,
            "Price": round(current_price, 4),
            "7D Return (%)": round(ret_7d, 2),
            "30D Return (%)": round(ret_30d, 2),
            "Rank Score": round(sell_score, 2),
            "Indicators Met": bearish_conditions
        })

# Convert to DataFrames and Sort by Rank Score (Descending)
df_buys = pd.DataFrame(buy_list_data)
if not df_buys.empty:
    df_buys = df_buys.sort_values(by="Rank Score", ascending=False).reset_index(drop=True)

df_sells = pd.DataFrame(sell_list_data)
if not df_sells.empty:
    df_sells = df_sells.sort_values(by="Rank Score", ascending=False).reset_index(drop=True)

# ---------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------
st.subheader("Signal & Return Tables")

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"**🟢 Buy Signals (Best Ranked at Top)**")
    if df_buys.empty:
        st.info("No buy signals met the criteria.")
    else:
        st.dataframe(df_buys, use_container_width=True)

with col2:
    st.markdown(f"**🔴 Sell / Short Signals (Best Ranked at Top)**")
    if df_sells.empty:
        st.info("No sell signals met the criteria.")
    else:
        st.dataframe(df_sells, use_container_width=True)
