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

# ---------------- RRG Math (EMA + Z-Score Standardization) ----------------
def calculate_excess_rrg(price_df, coins, window=14):
    returns_df = price_df[coins].pct_change().dropna()
    
    # Intra-theme benchmark: average return of the selected group
    benchmark_ret = returns_df.mean(axis=1)
    
    rrg_results = {}
    for coin in coins:
        # 1. Relative Price (Asset cumulated return / Benchmark cumulated return)
        rel_price = (1 + returns_df[coin]).cumprod() / (1 + benchmark_ret).cumprod()
        
        # 2. RS-Ratio: EMA smoothed relative price + Z-Score Normalization
        rs_ratio_raw = rel_price.ewm(span=window).mean()
        rs_ratio_mean = rs_ratio_raw.rolling(window=window, min_periods=1).mean()
        rs_ratio_std = rs_ratio_raw.rolling(window=window, min_periods=1).std().replace(0, 0.0001)
        
        # Scale factor (e.g., 5) keeps the visualization tight around the 100 center
        rs_ratio = ((rs_ratio_raw - rs_ratio_mean) / rs_ratio_std * 5) + 100
        
        # 3. RS-Momentum: Rate of change of RS-Ratio + Z-Score Normalization
        roc_period = max(1, window // 2)
        rs_mom_raw = rs_ratio_raw.pct_change(periods=roc_period).ewm(span=roc_period).mean()
        rs_mom_mean = rs_mom_raw.rolling(window=window, min_periods=1).mean()
        rs_mom_std = rs_mom_raw.rolling(window=window, min_periods=1).std().replace(0, 0.0001)
        
        rs_mom = ((rs_mom_raw - rs_mom_mean) / rs_mom_std * 5) + 100
        
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
    st.header("Relative Rotation Graph (Intra-Theme Outperformance)")
    
    c1, c2, c3 = st.columns([1, 1, 2])
    
    with c1:
        themes = sorted(list(set(mapping.values())))
        sel_theme = st.selectbox("Select Theme", themes)
        theme_coins = [k for k, v in mapping.items() if v == sel_theme and k in prices.columns]
        
    with c2:
        trail_len = st.slider("Trail Length (Periods)", 5, 180, 20)
        smooth_win = st.number_input("EMA Smoothing Window", 5, 40, 14)

    with c3:
        active_coins = st.multiselect("Coins to Plot", theme_coins, default=theme_coins)

    if len(active_coins) < 2:
        st.warning("Please select at least 2 coins to calculate a relative average.")
    else:
        results = calculate_excess_rrg(prices, active_coins, window=smooth_win)
        fig = go.Figure()

        # Explicit 100-lines for X and Y axes
        fig.add_hline(y=100, line_width=3, line_dash="solid", line_color="rgba(255, 255, 255, 0.4)", layer="below")
        fig.add_vline(x=100, line_width=3, line_dash="solid", line_color="rgba(255, 255, 255, 0.4)", layer="below")

        for coin in active_coins:
            if coin not in results:
                continue
                
            df = results[coin].tail(trail_len)
            if len(df) < 2: continue
            
            x_vals = df['x'].tolist()
            y_vals = df['y'].tolist()

            # 1. The Trail
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode='lines',
                line=dict(width=2),
                showlegend=False,
                name=coin
            ))
            
            # 2. Entry / Exit Marker Logic
            entries_x, entries_y = [], []
            exits_x, exits_y = [], []
            
            for i in range(1, len(x_vals)):
                prev_x, prev_y = x_vals[i-1], y_vals[i-1]
                curr_x, curr_y = x_vals[i], y_vals[i]
                
                # Entry: Crossing into 'Improving' (Y goes > 100 while X <= 100) 
                # OR Crossing into 'Leading' (X goes > 100 while Y >= 100)
                if (prev_y < 100 <= curr_y and curr_x <= 100) or (prev_x < 100 <= curr_x and curr_y >= 100):
                    entries_x.append(curr_x)
                    entries_y.append(curr_y)
                    
                # Exit: Crossing into 'Weakening' (Y goes < 100 while X >= 100)
                # OR Crossing into 'Lagging' (X goes < 100 while Y <= 100)
                if (prev_y > 100 >= curr_y and curr_x >= 100) or (prev_x > 100 >= curr_x and curr_y <= 100):
                    exits_x.append(curr_x)
                    exits_y.append(curr_y)
            
            # Plot Entries
            if entries_x:
                fig.add_trace(go.Scatter(
                    x=entries_x, y=entries_y,
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=12, color='#00FF00', line=dict(width=1, color='white')),
                    name=f'{coin} Entry',
                    showlegend=False,
                    hovertemplate="<b>Entry Signal</b>"
                ))
            
            # Plot Exits
            if exits_x:
                fig.add_trace(go.Scatter(
                    x=exits_x, y=exits_y,
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=12, color='#FF0000', line=dict(width=1, color='white')),
                    name=f'{coin} Exit',
                    showlegend=False,
                    hovertemplate="<b>Exit Signal</b>"
                ))

            # 3. The Latest Data Point
            fig.add_trace(go.Scatter(
                x=[x_vals[-1]], y=[y_vals[-1]],
                mode='markers+text',
                marker=dict(size=8),
                text=[coin],
                textposition="top center",
                name=coin,
                hovertemplate=f"<b>{coin}</b><br>RS-Ratio: %{{x:.2f}}<br>RS-Mom: %{{y:.2f}}"
            ))

            # 4. Directional Arrow Annotation
            fig.add_annotation(
                x=x_vals[-1], y=y_vals[-1],
                ax=x_vals[-2], ay=y_vals[-2],
                xref="x", yref="y",
                axref="x", ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor="rgba(255, 255, 255, 0.8)"
            )

        # Annotate Quadrants
        quad_attr = dict(showarrow=False, font=dict(size=16, color="rgba(255,255,255,0.3)"))
        
        # Calculate dynamic positions for quadrant text based on layout limits
        fig.add_annotation(x=105, y=105, text="LEADING", **quad_attr)
        fig.add_annotation(x=105, y=95, text="WEAKENING", **quad_attr)
        fig.add_annotation(x=95, y=95, text="LAGGING", **quad_attr)
        fig.add_annotation(x=95, y=105, text="IMPROVING", **quad_attr)

        # Ensure aspect ratio is exactly 1:1 and limits are properly buffered
        fig.update_layout(
            height=800,
            xaxis_title="Relative Strength Ratio (Trend)",
            yaxis_title="Relative Strength Momentum (Momentum)",
            template="plotly_dark",
            xaxis=dict(range=[90, 110]),
            yaxis=dict(range=[90, 110], scaleanchor="x", scaleratio=1),
            hovermode="closest"
        )

        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Click 'Fetch Data' in the sidebar to load the price history.")
