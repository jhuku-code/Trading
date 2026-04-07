import time
from pathlib import Path
from datetime import datetime

import ccxt
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ---------------- Page Config ----------------
st.set_page_config(page_title="Alpha RRG Dashboard", layout="wide")
st.title("Theme Excess Return (Alpha) RRG")

# ---------------- Sidebar ----------------
st.sidebar.header("Settings")

default_excel_relpath = Path("Input-Files") / "Themes_mapping.xlsx"
st.sidebar.write(f"Using: `{default_excel_relpath}`")

timeframe = st.sidebar.selectbox("Timeframe", ["1d", "4h", "1h"], index=0)
# NOTE: Increased max limit to 500 to support the 180-day trail + smoothing window
limit = st.sidebar.number_input("OHLCV limit", value=250, min_value=50, max_value=2000)
sleep_seconds = st.sidebar.number_input("Sleep (s)", value=0.2)

# ---------------- Session State ----------------
if "final_df" not in st.session_state:
    st.session_state.final_df = None
if "price_theme" not in st.session_state:
    st.session_state.price_theme = None

# ---------------- Helpers ----------------
@st.cache_data
def read_theme_excel(path):
    return pd.read_excel(path, engine="openpyxl")

@st.cache_resource
def get_exchange():
    ex = ccxt.kucoin()
    ex.load_markets()
    return ex

def fetch_prices(exchange, symbols, timeframe, limit):
    frames = []
    for sym in symbols:
        try:
            ohlcv = exchange.fetch_ohlcv(f"{sym}/USDT", timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=["ts","o","h","l","c","v"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms")
            frames.append(df.set_index("ts")["c"].rename(sym))
            time.sleep(sleep_seconds)
        except Exception as e:
            st.error(f"Error fetching {sym}: {e}")
            continue
    return pd.concat(frames, axis=1)

# ---------------- RRG Math (Excess Return Based) ----------------
def calculate_excess_rrg(price_df, coins, window=14):
    """
    X-axis: RS-Ratio (Smoothed Cumulative Excess Return)
    Y-axis: RS-Momentum (Rate of Change of RS-Ratio)
    """
    # 1. Periodic Returns
    returns_df = price_df[coins].pct_change().dropna()
    
    # 2. Benchmark (Theme Average Return)
    benchmark_ret = returns_df.mean(axis=1)
    
    rrg_results = {}
    
    for coin in coins:
        # 3. Excess Return (Alpha)
        excess_ret = returns_df[coin] - benchmark_ret
        
        # 4. Cumulative Alpha Index (Base 100)
        alpha_index = (1 + excess_ret).cumprod() * 100
        
        # 5. RS-Ratio (Trend) 
        # Normalized against its own 2*window moving average
        short_ma = alpha_index.rolling(window=window).mean()
        long_ma = alpha_index.rolling(window=window * 2).mean()
        rs_ratio = (short_ma / long_ma) * 100
        
        # 6. RS-Momentum (Acceleration)
        # ROC of the RS-Ratio
        rs_mom = (rs_ratio.pct_change(periods=max(1, window // 2)) * 100) + 100
        
        rrg_results[coin] = pd.DataFrame({'x': rs_ratio, 'y': rs_mom}).dropna()
        
    return rrg_results

# ---------------- Fetch Logic ----------------
if st.sidebar.button("🔄 Fetch Data"):
    df_map = read_theme_excel(default_excel_relpath)
    symbols = df_map["Symbol"].str.upper().tolist()
    mapping = dict(zip(symbols, df_map["Theme"].tolist()))
    
    ex = get_exchange()
    with st.spinner("Fetching prices and computing Alpha..."):
        prices = fetch_prices(ex, symbols, timeframe, limit)
        st.session_state["price_theme"] = prices
        st.session_state["ticker_to_theme"] = mapping
        st.session_state.last_fetch = datetime.utcnow()
        st.success("Data Updated!")

# ---------------- RRG Visualization Section ----------------
if st.session_state["price_theme"] is not None:
    prices = st.session_state["price_theme"]
    mapping = st.session_state["ticker_to_theme"]

    st.divider()
    st.header("Relative Rotation Graph (Excess Returns)")
    
    # --- UI Controls ---
    c1, c2, c3 = st.columns([1, 1, 2])
    
    with c1:
        themes = sorted(list(set(mapping.values())))
        sel_theme = st.selectbox("Select Theme", themes)
        theme_coins = [k for k, v in mapping.items() if v == sel_theme and k in prices.columns]
        
    with c2:
        # User requested up to 180 days trail
        trail_len = st.slider("Trail Length (Periods)", 5, 180, 20)
        smooth_win = st.number_input("Smoothing (Window)", 5, 40, 14)

    with c3:
        active_coins = st.multiselect("Coins to Plot", theme_coins, default=theme_coins)

    if len(active_coins) < 2:
        st.warning("Please select at least 2 coins to calculate a relative average.")
    else:
        results = calculate_excess_rrg(prices, active_coins, window=smooth_win)
        
        fig = go.Figure()

        # Quadrant Colors & Lines
        fig.add_hline(y=100, line_dash="dot", line_color="gray", opacity=0.5)
        fig.add_vline(x=100, line_dash="dot", line_color="gray", opacity=0.5)

        for coin in active_coins:
            df = results[coin].tail(trail_len)
            if df.empty: continue
            
            # 1. The Trail (Line)
            fig.add_trace(go.Scatter(
                x=df['x'], y=df['y'],
                mode='lines',
                line=dict(width=2),
                hoverinfo='skip',
                showlegend=False,
                name=f"{coin} trail"
            ))
            
            # 2. The Directional Arrow (Latest Point)
            # We use 'triangle-up' and calculate the angle of movement
            # Or simply use the 'arrow' symbol with angleref
            fig.add_trace(go.Scatter(
                x=[df['x'].iloc[-1]],
                y=[df['y'].iloc[-1]],
                mode='markers+text',
                marker=dict(
                    symbol='arrow', 
                    size=15, 
                    angleref='previous', # Points arrow in direction of movement
                    line=dict(width=1, color='white')
                ),
                text=[coin],
                textposition="top center",
                name=coin,
                hovertemplate=f"<b>{coin}</b><br>RS-Ratio: %{{x:.2f}}<br>RS-Mom: %{{y:.2f}}"
            ))

        # Annotate Quadrants
        quad_attr = dict(showarrow=False, font=dict(size=14, color="rgba(255,255,255,0.5)"))
        fig.add_annotation(x=102, y=102, text="LEADING", **quad_attr)
        fig.add_annotation(x=102, y=98, text="WEAKENING", **quad_attr)
        fig.add_annotation(x=98, y=98, text="LAGGING", **quad_attr)
        fig.add_annotation(x=98, y=102, text="IMPROVING", **quad_attr)

        fig.update_layout(
            height=800,
            xaxis_title="Relative Strength Ratio (Trend)",
            yaxis_title="Relative Strength Momentum (Momentum)",
            template="plotly_dark",
            xaxis=dict(range=[min(95, df['x'].min() if 'df' in locals() else 95), 
                             max(105, df['x'].max() if 'df' in locals() else 105)]),
            yaxis=dict(range=[min(95, df['y'].min() if 'df' in locals() else 95), 
                             max(105, df['y'].max() if 'df' in locals() else 105)])
        )

        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Click 'Fetch Data' in the sidebar to load the price history.")
