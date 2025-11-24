import time
import numpy as np
import pandas as pd
import streamlit as st

st.title("Price Range Breakout")

# ---------------------------------------------------------
# INPUT DATA FROM SESSION STATE
# ---------------------------------------------------------
price_d = st.session_state.get("price_theme", None)

if price_d is None or price_d.empty:
    st.error("price_theme not found in session_state or is empty. Please load prices first on the main page.")
    st.stop()

# ---------------------------------------------------------
# REFRESH BUTTON
# ---------------------------------------------------------
refresh_col, _ = st.columns([1, 5])
with refresh_col:
    if st.button("🔄 Refresh data"):
        # Clear cached computations and rerun
        st.cache_data.clear()
        st.rerun()

# ---------------------------------------------------------
# SUPPORT / RESISTANCE DETECTION
# ---------------------------------------------------------
@st.cache_data(show_spinner=True)
def detect_sr_channels_coins_only(df, 
                                  prd=5,
                                  loopback=200,
                                  min_strength=1,
                                  channel_w_start=0.1,
                                  channel_w_max=0.3,
                                  proximity_start=0.02,
                                  proximity_max=0.1,
                                  top_n=15):

    def detect_once(df, prd, loopback, min_strength, channel_w, proximity):
        results = { "near_support": [], "near_resistance": [],
                    "broke_support": [], "broke_resistance": [] }

        for coin in df.columns:
            prices = df[coin].dropna()
            if len(prices) < loopback + prd + 2:
                continue

            closes = prices.values
            highs, lows = closes, closes

            # Step 1: pivots
            pivots = []
            for i in range(prd, len(closes) - prd):
                is_high = all(highs[i] >= highs[i - k] for k in range(1, prd+1)) and \
                          all(highs[i] >= highs[i + k] for k in range(1, prd+1))
                is_low = all(lows[i] <= lows[i - k] for k in range(1, prd+1)) and \
                         all(lows[i] <= lows[i + k] for k in range(1, prd+1))
                if is_high or is_low:
                    pivots.append((i, closes[i]))

            if not pivots:
                continue

            # Step 2: channels
            max_width = (closes[-loopback:].max() - closes[-loopback:].min()) * channel_w
            channels = []
            for _, val in pivots:
                lo, hi, count = val, val, 0
                for _, val2 in pivots:
                    if abs(val2 - val) <= max_width:
                        lo, hi = min(lo, val2), max(hi, val2)
                        count += 1
                if count >= min_strength:
                    channels.append([lo, hi, count])

            if not channels:
                continue

            # Step 3: score by loopback
            for ch in channels:
                lo, hi, s = ch
                for k in range(1, loopback+1):
                    if len(closes)-k < 0:
                        break
                    if (lows[-k] <= hi and highs[-k] >= lo):
                        ch[2] += 1

            channels = sorted(channels, key=lambda x: x[2], reverse=True)[:3]

            # Step 4: classify
            last, prev = closes[-1], closes[-2]
            support = [c for c in channels if c[0] <= last]
            resistance = [c for c in channels if c[1] >= last]

            if support:
                lo, hi, s = max(support, key=lambda x: x[2])
                if abs(last - lo)/last <= proximity:
                    results["near_support"].append((coin, s))
                if prev >= lo and last < lo:
                    results["broke_support"].append((coin, s))

            if resistance:
                lo, hi, s = max(resistance, key=lambda x: x[2])
                if abs(last - hi)/last <= proximity:
                    results["near_resistance"].append((coin, s))
                if prev <= hi and last > hi:
                    results["broke_resistance"].append((coin, s))

        for k in results:
            results[k] = sorted(results[k], key=lambda x: x[1], reverse=True)

        return results

    # Adaptive loop
    channel_w, proximity = channel_w_start, proximity_start
    while channel_w <= channel_w_max and proximity <= proximity_max:
        res = detect_once(df, prd, loopback, min_strength, channel_w, proximity)

        assigned = set()
        unique_res = {k: [] for k in res}

        # Priority: breakouts first, then near levels
        for key in ["broke_support", "broke_resistance", "near_support", "near_resistance"]:
            for coin, strength in res[key]:
                if coin not in assigned:
                    unique_res[key].append((coin, strength))
                    assigned.add(coin)

        # Limit to top_n and keep only coin names
        for k in unique_res:
            unique_res[k] = [c for c, _ in unique_res[k][:top_n]]

        if any(len(unique_res[k]) > 0 for k in unique_res):
            return unique_res

        # Relax thresholds
        channel_w *= 1.5
        proximity *= 1.5

    # Return last unique_res even if empty
    return unique_res

# ---------------------------------------------------------
# RUN DETECTION
# ---------------------------------------------------------
signals = detect_sr_channels_coins_only(price_d)

# ---------------------------------------------------------
# DISPLAY RESULTS AS TABLES
# ---------------------------------------------------------
st.subheader("Support / Resistance Signals")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Near Support**")
    near_support_df = pd.DataFrame({"Coin": signals.get("near_support", [])})
    st.dataframe(near_support_df, use_container_width=True)

    st.markdown("**Broke Support**")
    broke_support_df = pd.DataFrame({"Coin": signals.get("broke_support", [])})
    st.dataframe(broke_support_df, use_container_width=True)

with col2:
    st.markdown("**Near Resistance**")
    near_resistance_df = pd.DataFrame({"Coin": signals.get("near_resistance", [])})
    st.dataframe(near_resistance_df, use_container_width=True)

    st.markdown("**Broke Resistance**")
    broke_resistance_df = pd.DataFrame({"Coin": signals.get("broke_resistance", [])})
    st.dataframe(broke_resistance_df, use_container_width=True)
