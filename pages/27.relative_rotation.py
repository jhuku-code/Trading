import time
from pathlib import Path
from datetime import datetime

import ccxt
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ---------------- Page Config ----------------
st.set_page_config(page_title="Theme Returns Dashboard", layout="wide")
st.title("Theme Returns Dashboard")

# ... [KEEP ALL YOUR EXISTING SIDEBAR, HELPERS, AND FETCH LOGIC UNCHANGED] ...
# (The code below assumes your existing session_state variables are populated)

# [START OF ADDITION] ---------------------------------------------------------
# ---------------- RRG Calculation Logic ----------------

def calculate_rrg(price_df, coins, window=14):
    """
    Calculates simplified RS-Ratio and RS-Momentum for RRG.
    Benchmark is the mean price of the selected coins.
    """
    # 1. Define Benchmark (Mean of selected coins)
    benchmark = price_df[coins].mean(axis=1)
    
    rrg_data = {}
    
    for coin in coins:
        # 2. Relative Strength (RS)
        rs = price_df[coin] / benchmark
        
        # 3. RS-Ratio (Trend of RS)
        # We use a rolling mean and normalize around 100
        rs_ratio = rs.rolling(window=window).mean()
        rs_ratio_norm = (rs_ratio / rs_ratio.mean()) * 100
        
        # 4. RS-Momentum (Rate of change of RS-Ratio)
        # Difference between current ratio and previous, normalized around 100
        rs_mom = rs_ratio.pct_change(periods=window//2) + 1
        rs_mom_norm = (rs_mom / rs_mom.mean()) * 100
        
        rrg_data[coin] = pd.DataFrame({
            'x': rs_ratio_norm,
            'y': rs_mom_norm
        })
        
    return rrg_data

# ---------------- RRG Section ----------------
st.divider()
st.header("Relative Rotation Graph (RRG)")

if "price_theme" not in st.session_state:
    st.warning("Please 'Fetch Data' first to enable RRG.")
else:
    prices = st.session_state["price_theme"]
    mapping = st.session_state["ticker_to_theme"]
    
    # --- RRG Controls ---
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        unique_themes = sorted(list(set(mapping.values())))
        selected_theme = st.selectbox("Select Theme for RRG", unique_themes)
        
    # Filter coins for this theme
    theme_coins = [coin for coin, theme in mapping.items() if theme == selected_theme and coin in prices.columns]
    
    with col3:
        selected_coins = st.multiselect(
            "Select/Unselect Coins", 
            options=theme_coins, 
            default=theme_coins
        )
        
    with col2:
        history_len = st.slider("History Trail (Periods)", min_value=5, max_value=50, value=15)

    if len(selected_coins) < 2:
        st.error("Select at least 2 coins to generate a relative benchmark.")
    else:
        # Calculate RRG Values
        rrg_results = calculate_rrg(prices, selected_coins)
        
        # --- Plotly Visualization ---
        fig = go.Figure()

        # Add Background Quadrant Shapes/Colors
        # Leading (Top Right), Weakening (Bottom Right), Lagging (Bottom Left), Improving (Top Left)
        # Quadrant logic centered at 100, 100
        
        for coin in selected_coins:
            df_plot = rrg_results[coin].tail(history_len).dropna()
            
            if df_plot.empty: continue
            
            # Add the trail line
            fig.add_trace(go.Scatter(
                x=df_plot['x'], y=df_plot['y'],
                mode='lines',
                name=coin,
                line=dict(width=1.5),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Add the head (current position)
            fig.add_trace(go.Scatter(
                x=[df_plot['x'].iloc[-1]], 
                y=[df_plot['y'].iloc[-1]],
                mode='markers+text',
                name=coin,
                text=[coin],
                textposition="top center",
                marker=dict(size=10, symbol='arrow', angleref='previous'),
                hovertemplate=f"<b>{coin}</b><br>RS-Ratio: %{{x:.2f}}<br>RS-Mom: %{{y:.2f}}"
            ))

        # Update Layout for RRG Look
        fig.update_layout(
            height=700,
            xaxis_title="RS-Ratio (Trend)",
            yaxis_title="RS-Momentum (Momentum)",
            shapes=[
                # Vertical Line
                dict(type="line", x0=100, x1=100, y0=min([95, fig.layout.yaxis.range[0] if fig.layout.yaxis.range else 95]), 
                     y1=max([105, fig.layout.yaxis.range[1] if fig.layout.yaxis.range else 105]), line=dict(color="gray", dash="dash")),
                # Horizontal Line
                dict(type="line", x0=min([95, fig.layout.xaxis.range[0] if fig.layout.xaxis.range else 95]), 
                     x1=max([105, fig.layout.xaxis.range[1] if fig.layout.xaxis.range else 105]), y0=100, y1=100, line=dict(color="gray", dash="dash")),
            ],
            annotations=[
                dict(x=101, y=101, text="LEADING", showarrow=False, font=dict(color="green", size=16)),
                dict(x=101, y=99, text="WEAKENING", showarrow=False, font=dict(color="orange", size=16)),
                dict(x=99, y=99, text="LAGGING", showarrow=False, font=dict(color="red", size=16)),
                dict(x=99, y=101, text="IMPROVING", showarrow=False, font=dict(color="blue", size=16)),
            ]
        )
        
        # Center the axis around 100
        fig.update_xaxes(zeroline=False)
        fig.update_yaxes(zeroline=False)

        st.plotly_chart(fig, use_container_width=True)
# [END OF ADDITION] -----------------------------------------------------------
