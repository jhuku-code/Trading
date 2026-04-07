"""
Volatility Breakout Signals — v4 (The Coiled Spring)
======================================================
Key improvements for Close-Only Data:
-------------------------------------
1.  Close-Only Vol Proxy: Uses the rolling standard deviation of log returns 
    so high-priced and low-priced coins are perfectly normalized.
2.  The Squeeze Gate: Requires short-term volatility to have been LOWER than 
    long-term volatility within the last week (a contraction period).
3.  The Expansion Gate: Requires short-term volatility today to violently 
    expand past the long-term baseline (the release).
4.  Donchian Proxy: Requires today's close to be >= the highest close of the 
    previous 20 days.
5.  Ranking: Ranks purely by the "Expansion Multiplier" to surface the most 
    explosive breakouts.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from typing import Dict, Optional

# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Volatility Breakout: Coiled Spring", layout="wide")
st.title("Volatility Breakout · The Coiled Spring")

# ---------------------------------------------------------
# INPUT DATA FROM SESSION STATE
# ---------------------------------------------------------
df_h: Optional[pd.DataFrame] = st.session_state.get("price_theme", None)

if df_h is None or df_h.empty:
    st.error(
        "price_theme not found in st.session_state or is empty.\n\n"
        "Please go to the Themes Tracker page, click **Refresh / Fetch Data**, "
        "and then come back to this page."
    )
    st.stop()

# ---------------------------------------------------------
# SIDEBAR PARAMETERS
# ---------------------------------------------------------
st.sidebar.header("Parameters")

with st.sidebar.expander("Lookback Windows", expanded=True):
    donchian_window = st.number_input(
        "Donchian Breakout Window (bars)",
        min_value=5, max_value=100, value=20, step=5,
        help="Price must hit a new high over this many previous bars.",
    )
    vol_long_window = st.number_input(
        "Long-Term Volatility (bars)",
        min_value=20, max_value=200, value=90, step=10,
        help="Baseline for normal market volatility.",
    )
    vol_short_window = st.number_input(
        "Short-Term Volatility (bars)",
        min_value=5, max_value=50, value=14, step=1,
        help="Recent volatility window.",
    )
    squeeze_lookback = st.number_input(
        "Squeeze Lookback (bars)",
        min_value=1, max_value=20, value=5, step=1,
        help="How many days back we look to confirm a volatility contraction.",
    )

with st.sidebar.expander("Threshold Gates", expanded=True):
    expansion_ratio = st.number_input(
        "Vol Expansion Ratio",
        min_value=1.1, max_value=3.0, value=1.5, step=0.1,
        help="Short-term vol must be this many times larger than long-term vol today.",
    )

with st.sidebar.expander("Ranking & Display", expanded=True):
    top_n_total = st.number_input(
        "Compute top N candidates",
        min_value=5, max_value=100, value=20, step=5,
    )
    rank_start = st.number_input(
        "Display rank start (1-indexed)",
        min_value=1, max_value=int(top_n_total), value=1, step=1,
    )
    rank_end = st.number_input(
        "Display rank end (1-indexed)",
        min_value=int(rank_start), max_value=int(top_n_total),
        value=min(10, int(top_n_total)), step=1,
    )

st.sidebar.markdown("---")
refresh = st.sidebar.button("🔄 Refresh / Recompute", use_container_width=True)

if "vol_spring_results" not in st.session_state:
    st.session_state["vol_spring_results"] = None

# ---------------------------------------------------------
# CLASS DEFINITION
# ---------------------------------------------------------
class CoiledSpringBreakout:
    def __init__(
        self,
        donchian_window: int = 20,
        vol_short_window: int = 14,
        vol_long_window: int = 90,
        expansion_ratio: float = 1.5,
        squeeze_lookback: int = 5,
    ):
        self.donchian_window = donchian_window
        self.vol_short = vol_short_window
        self.vol_long = vol_long_window
        self.expansion_ratio = expansion_ratio
        self.squeeze_lookback = squeeze_lookback

    def generate_signals(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        # 1. Volatility Proxy (Rolling Std Dev of Log Returns)
        returns = np.log(df / df.shift(1))
        short_vol = returns.rolling(window=self.vol_short).std()
        long_vol = returns.rolling(window=self.vol_long).std()

        # 2. Squeeze Condition: Short vol was less than long vol recently
        is_squeezed = (short_vol < long_vol).astype(int)
        recent_squeeze = is_squeezed.rolling(window=self.squeeze_lookback).max() == 1

        # 3. Expansion Gate: Short vol is rapidly expanding today
        vol_expansion = short_vol > (long_vol * self.expansion_ratio)

        # 4. Donchian Proxy: Today's close >= highest close of previous N days
        rolling_high_close = df.shift(1).rolling(window=self.donchian_window).max()
        donchian_breakout = df >= rolling_high_close

        # 5. Master Signal
        raw_signal = (recent_squeeze & vol_expansion & donchian_breakout).astype(int)

        return {
            "short_vol": short_vol,
            "long_vol": long_vol,
            "is_squeezed": is_squeezed,
            "donchian_breakout": donchian_breakout,
            "raw_signal": raw_signal
        }

    def get_top_signals(self, signals_dict: Dict, top_n: int = 20) -> pd.DataFrame:
        ts = signals_dict["raw_signal"].index[-1]
        
        signal_today = signals_dict["raw_signal"].loc[ts]
        short_vol = signals_dict["short_vol"].loc[ts].dropna()
        long_vol = signals_dict["long_vol"].loc[ts].dropna()
        
        eligible = signal_today[signal_today == 1].index.tolist()
        
        if not eligible:
            return pd.DataFrame()

        expansion_multiplier = (short_vol.reindex(eligible) / long_vol.reindex(eligible))

        df_out = pd.DataFrame({
            "coin": eligible,
            "expansion_multiplier": expansion_multiplier.values,
            "short_term_vol": short_vol.reindex(eligible).values,
            "long_term_vol": long_vol.reindex(eligible).values,
        })

        # Rank by the most explosive volatility expansion
        df_out = df_out.sort_values("expansion_multiplier", ascending=False)
        return df_out.head(top_n).reset_index(drop=True)

    def diagnose_signals(self, signals_dict: Dict) -> Dict:
        ts = signals_dict["raw_signal"].index[-1]
        raw = signals_dict["raw_signal"].loc[ts]
        breakout = signals_dict["donchian_breakout"].loc[ts]
        squeezed = signals_dict["is_squeezed"].loc[ts]
        
        return {
            "timestamp": str(ts),
            "total_coins": int(len(raw)),
            "coins_hitting_new_highs": int(breakout.sum()),
            "coins_currently_squeezed": int(squeezed.sum()),
            "coins_triggering_full_signal": int(raw.sum()),
            "expansion_ratio_required": float(self.expansion_ratio),
        }

# ---------------------------------------------------------
# MAIN COMPUTATION
# ---------------------------------------------------------
if refresh or st.session_state["vol_spring_results"] is None:
    with st.spinner("Hunting for coiled springs…"):
        spring = CoiledSpringBreakout(
            donchian_window=int(donchian_window),
            vol_short_window=int(vol_short_window),
            vol_long_window=int(vol_long_window),
            expansion_ratio=float(expansion_ratio),
            squeeze_lookback=int(squeeze_lookback),
        )

        signals_dict = spring.generate_signals(df_h)
        base_top = spring.get_top_signals(signals_dict, top_n=int(top_n_total))
        diagnostics = spring.diagnose_signals(signals_dict)

        st.session_state["vol_spring_results"] = {
            "signals_dict": signals_dict,
            "base_top": base_top,
            "diagnostics": diagnostics,
            "spring": spring,
        }

results = st.session_state["vol_spring_results"]

if results is None:
    st.info("Click **Refresh / Recompute** in the sidebar to compute signals.")
    st.stop()

base_top: pd.DataFrame = results["base_top"]
diagnostics: Dict = results["diagnostics"]
signals_dict: Dict = results["signals_dict"]

# ---------------------------------------------------------
# DISPLAY — SELECTED RANK SLICE
# ---------------------------------------------------------
st.subheader(f"Explosive Breakouts — Rank {int(rank_start)}–{int(rank_end)}")

if base_top is None or base_top.empty:
    st.warning(
        "No coiled spring breakouts detected today. "
        "The market might be too choppy, or no assets have built up enough compression. "
        "Try lowering the Vol Expansion Ratio."
    )
else:
    display_df = base_top.iloc[int(rank_start) - 1 : int(rank_end)].copy()
    display_df.insert(0, "rank", range(int(rank_start), int(rank_start) + len(display_df)))

    # Format the dataframe cleanly
    styled = display_df.style.format({
        "expansion_multiplier": "{:.2f}x",
        "short_term_vol": "{:.4f}",
        "long_term_vol": "{:.4f}"
    })
    st.dataframe(styled, use_container_width=True)

    # ---------------------------------------------------------
    # CHART — Expansion Multiplier Bar Chart
    # ---------------------------------------------------------
    if not display_df.empty:
        fig = go.Figure(
            go.Bar(
                x=display_df["coin"],
                y=display_df["expansion_multiplier"],
                marker_color="#d62728", # A strong red to indicate an explosive move
                text=[f"{v:.2f}x" for v in display_df["expansion_multiplier"]],
                textposition="outside",
            )
        )
        
        # Add a line showing the minimum required expansion ratio
        fig.add_hline(
            y=expansion_ratio, 
            line_dash="dot", 
            annotation_text="Required Expansion Gate", 
            annotation_position="bottom right"
        )

        fig.update_layout(
            title="Volatility Expansion Multiplier by Coin",
            xaxis_title="Coin",
            yaxis_title="Short Vol / Long Vol",
            height=350,
            margin=dict(t=50, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# DISPLAY — Diagnostics
# ---------------------------------------------------------
with st.expander("Under the Hood: Market Diagnostics", expanded=False):
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Coins Scanned", diagnostics["total_coins"])
    col2.metric(f"{donchian_window}-Day Highs", diagnostics["coins_hitting_new_highs"])
    col3.metric("Currently Squeezed", diagnostics["coins_currently_squeezed"])
    
    st.write("---")
    st.json(diagnostics)
