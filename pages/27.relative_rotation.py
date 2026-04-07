import time
from pathlib import Path
from datetime import datetime

import ccxt
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ---------------- Page Config ----------------
st.set_page_config(page_title="Theme Alpha RRG", layout="wide")
st.title("Theme Excess Return Rotation (Alpha RRG)")

# ... [Keep your existing Sidebar, Fetch Logic, and compute_returns as is] ...

# ---------------- Advanced RRG Calculation (Excess Return Based) ----------------

def calculate_excess_rrg(price_df, coins, window=14):
    """
    Calculates RRG coordinates based on Excess Returns.
    X-axis: RS-Ratio (Smoothed Cumulative Excess Return)
    Y-axis: RS-Momentum (Rate of Change of the Ratio)
    """
    # 1. Calculate percentage returns
    returns_df = price_df[coins].pct_change().dropna()
    
    # 2. Define Theme Benchmark (Mean return of selected coins)
    benchmark_return = returns_df.mean(axis=1)
    
    rrg_data = {}
    
    for coin in coins:
        # 3. Calculate Daily Excess Return
        excess_return = returns_df[coin] - benchmark_return
        
        # 4. Create Cumulative Excess Index (Equity Curve of Alpha)
        # We start at 100 for normalization
        excess_index = (1 + excess_return).cumprod() * 100
        
        # 5. RS-Ratio: Double smoothed trend of the index
        # We normalize by the average index value to keep it centered near 100
        short_avg = excess_index.rolling(window=window).mean()
        rs_ratio = (short_avg / short_avg.rolling(window=window*2).mean()) * 100
        
        # 6. RS-Momentum: Rate of change of the RS-Ratio
        rs_momentum = (rs_ratio.pct_change(periods=window//2) * 100) + 100
        
        rrg_data[coin] = pd.DataFrame({
            'x': rs_ratio,
            'y': rs_momentum
        }).dropna()
        
    return rrg_data

# ---------------- Streamlit UI Logic ----------------

if "price_theme" not in st.session_state:
    st.info("Please fetch data from the sidebar to generate the RRG.")
else:
    prices = st.session_state["price_theme"]
    mapping = st.session_state["ticker_to_theme"]
    
    st.divider()
    st.subheader("Relative Rotation Graph: Alpha vs. Theme Average")
    st.write("This chart uses **Excess Returns** to strip out market noise, showing which coins are genuinely outperforming the sector average.")

    # --- Controls ---
    c1, c2, c3 = st.columns([1, 1, 2])
    
    with c1:
        unique_themes = sorted(list(set(mapping.values())))
        theme_choice = st.selectbox("Select Theme", unique_themes)
        
    theme_coins = [k for k, v in mapping.items() if v == theme_choice and k in prices.columns]
    
    with c3:
        selected_subset = st.multiselect("Active Coins", theme_coins, default=theme_coins)
        
    with c2:
        trail_size = st.slider("Trail Length", 5, 40, 15)
        smoothing = st.number_input("Smoothing Window", 5, 30, 14)

    if len(selected_subset) < 2:
        st.warning("Select at least 2 coins to compute a relative average.")
    else:
        # Compute the RRG coordinates
        rrg_results = calculate_excess_rrg(prices, selected_subset, window=smoothing)
        
        fig = go.Figure()

        # Define colors for quadrants
        # Note: Plotly's shapes or annotations can label quadrants
        
        for coin in selected_subset:
            df = rrg_results[coin].tail(trail_size)
            if df.empty: continue
            
            # Draw the tail (history)
            fig.add_trace(go.Scatter(
                x=df['x'], y=df['y'],
                mode='lines',
                line=dict(width=2),
                hoverinfo='skip',
                showlegend=False,
                opacity=0.4
            ))
            
            # Draw the head (current)
            fig.add_trace(go.Scatter(
                x=[df['x'].iloc[-1]],
                y=[df['y'].iloc[-1]],
                mode='markers+text',
                marker=dict(size=12, symbol='circle'),
                text=[coin],
                textposition="top center",
                name=coin,
                hovertemplate=f"<b>{coin}</b><br>Trend (RS-Ratio): %{{x:.2f}}<br>Momentum: %{{y:.2f}}"
            ))

        # Layout styling to create the "Four Quadrants"
        fig.update_layout(
            height=700,
            xaxis_title="Relative Strength Ratio (Trend)",
            yaxis_title="Relative Strength Momentum",
            template="plotly_dark",
            shapes=[
                # Horizontal Center Line
                dict(type="line", x0=90, x1=110, y0=100, y1=100, line=dict(color="rgba(255,255,255,0.2)", dash="dot")),
                # Vertical Center Line
                dict(type="line", x0=100, x1=100, y0=90, y1=110, line=dict(color="rgba(255,255,255,0.2)", dash="dot")),
            ],
            annotations=[
                dict(x=105, y=105, text="<b>LEADING</b>", showarrow=False, font=dict(color="#00ff00")),
                dict(x=105, y=95, text="<b>WEAKENING</b>", showarrow=False, font=dict(color="#ffa500")),
                dict(x=95, y=95, text="<b>LAGGING</b>", showarrow=False, font=dict(color="#ff0000")),
                dict(x=95, y=105, text="<b>IMPROVING</b>", showarrow=False, font=dict(color="#00bfff")),
            ]
        )
        
        # Force the plot to stay somewhat centered around 100, 100
        fig.update_xaxes(range=[95, 105])
        fig.update_yaxes(range=[95, 105])

        st.plotly_chart(fig, use_container_width=True)
