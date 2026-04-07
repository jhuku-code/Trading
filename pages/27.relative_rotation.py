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
st.title("Theme Excess Return Rotation (Alpha RRG)")

# ... [KEEP YOUR EXISTING SIDEBAR & FETCH LOGIC] ...
# Ensure 'limit' in the sidebar is set to at least 250 to support 180d trails + smoothing

# ---------------- Advanced RRG Calculation ----------------

def calculate_excess_rrg(price_df, coins, window=14):
    """
    Calculates RRG coordinates based on Cumulative Excess Returns.
    X-axis (RS-Ratio): Smoothed Trend of Alpha
    Y-axis (RS-Momentum): Rate of Change of Alpha
    """
    # 1. Calculate percentage returns
    returns_df = price_df[coins].pct_change().dropna()
    
    # 2. Benchmark = Average Return of the selected group
    benchmark_return = returns_df.mean(axis=1)
    
    rrg_data = {}
    for coin in coins:
        # 3. Excess Return (Alpha)
        excess_return = returns_df[coin] - benchmark_return
        
        # 4. Cumulative Index (Starting at 100)
        excess_index = (1 + excess_return).cumprod() * 100
        
        # 5. RS-Ratio: The trend component (Standardized)
        # Using a double-window approach to center around 100
        base_ma = excess_index.rolling(window=window).mean()
        long_ma = excess_index.rolling(window=window*2).mean()
        rs_ratio = (base_ma / long_ma) * 100
        
        # 6. RS-Momentum: The velocity component
        rs_momentum = (rs_ratio.pct_change(periods=window//2) * 100) + 100
        
        rrg_data[coin] = pd.DataFrame({'x': rs_ratio, 'y': rs_momentum}).dropna()
        
    return rrg_data

# ---------------- RRG Visualization Section ----------------

if "price_theme" not in st.session_state:
    st.info("Please fetch data from the sidebar to generate the RRG.")
else:
    prices = st.session_state["price_theme"]
    mapping = st.session_state["ticker_to_theme"]
    
    st.divider()
    
    # --- UI Controls ---
    c1, c2, c3 = st.columns([1, 1, 2])
    
    with c1:
        unique_themes = sorted(list(set(mapping.values())))
        theme_choice = st.selectbox("Select Theme", unique_themes)
        
    theme_coins = [k for k, v in mapping.items() if v == theme_choice and k in prices.columns]
    
    with c3:
        selected_subset = st.multiselect("Filter Coins", theme_coins, default=theme_coins)
        
    with c2:
        # Updated to 180 days as requested
        trail_len = st.slider("History Trail (Days)", 5, 180, 30)
        smoothing = st.number_input("Smoothing (Period)", 5, 40, 14)

    if len(selected_subset) < 2:
        st.warning("Select at least 2 coins to compute relative performance.")
    else:
        results = calculate_excess_rrg(prices, selected_subset, window=smoothing)
        
        fig = go.Figure()

        # Quadrant Background Colors (Optional but helpful for visual cues)
        fig.add_hrect(y0=100, y1=110, x0=100, x1=110, fillcolor="green", opacity=0.05, layer="below")
        fig.add_hrect(y0=90, y1=100, x0=100, x1=110, fillcolor="orange", opacity=0.05, layer="below")
        fig.add_hrect(y0=90, y1=100, x0=90, x1=100, fillcolor="red", opacity=0.05, layer="below")
        fig.add_hrect(y0=100, y1=110, x0=90, x1=100, fillcolor="blue", opacity=0.05, layer="below")

        for coin in selected_subset:
            df = results[coin].tail(trail_len)
            if len(df) < 2: continue
            
            # 1. The Tail (History)
            fig.add_trace(go.Scatter(
                x=df['x'], y=df['y'],
                mode='lines',
                line=dict(width=1.5),
                hoverinfo='skip',
                showlegend=False,
                opacity=0.5
            ))
            
            # 2. The Head (Arrow indicating direction)
            # We use the last point and point it away from the second-to-last point
            fig.add_trace(go.Scatter(
                x=[df['x'].iloc[-1]],
                y=[df['y'].iloc[-1]],
                mode='markers+text',
                marker=dict(
                    size=12, 
                    symbol='arrow', 
                    angleref='previous', # Automatically aligns arrow to the line path
                    standoff=5
                ),
                text=[coin],
                textposition="top center",
                name=coin,
                hovertemplate=f"<b>{coin}</b><br>Ratio: %{{x:.2f}}<br>Mom: %{{y:.2f}}"
            ))

        # Center lines and labels
        fig.update_layout(
            height=800,
            template="plotly_dark",
            xaxis=dict(title="Relative Strength (Trend)", gridcolor="gray", zeroline=False),
            yaxis=dict(title="Relative Momentum (Velocity)", gridcolor="gray", zeroline=False),
            shapes=[
                dict(type="line", x0=100, x1=100, y0=df['y'].min() if not df.empty else 90, 
                     y1=df['y'].max() if not df.empty else 110, line=dict(color="white", width=1)),
                dict(type="line", x0=df['x'].min() if not df.empty else 90, 
                     x1=df['x'].max() if not df.empty else 110, y0=100, y1=100, line=dict(color="white", width=1)),
            ],
            annotations=[
                dict(x=101, y=101, text="LEADING", showarrow=False, font=dict(color="lightgreen", size=14)),
                dict(x=101, y=99, text="WEAKENING", showarrow=False, font=dict(color="orange", size=14)),
                dict(x=99, y=99, text="LAGGING", showarrow=False, font=dict(color="red", size=14)),
                dict(x=99, y=101, text="IMPROVING", showarrow=False, font=dict(color="skyblue", size=14)),
            ]
        )

        st.plotly_chart(fig, use_container_width=True)
