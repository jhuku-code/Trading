"""
Volatility Breakout Signals — v3 (Streamlined Vol-Momentum + CS Gate)
======================================================================
Key improvements over v2:
--------------------------
1.  Simplified Breakout Logic: Replaced complex "vol_skew" with a direct momentum
    check (window return > 0). If volatility is expanding and price is up, it's bullish.
2.  Dual-Gate Filtering: Cross-sectional Z-score is now an explicit gate (filter)
    rather than a blended composite weight. This removes the "beta" of market rallies.
3.  Removed Fragile Math: Dropped _robust_clip_std and ratio normalizations, 
    making the script significantly faster and statistically sound.
4.  Ranking: Candidates are ranked cleanly by the magnitude of their 
    historical volatility expansion (hist_z_score).
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from typing import Dict, Optional

# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Volatility Breakout v3", layout="wide")
st.title("Volatility Breakout Signals · v3 (Streamlined)")

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
    historical_window = st.number_input(
        "Historical window (bars)",
        min_value=10, max_value=365, value=90, step=5,
        help="Lookback for historical vol baseline and price MA.",
    )
    volatility_window = st.number_input(
        "Volatility window (bars)",
        min_value=5, max_value=96, value=14, step=1,
        help="Rolling window for current vol.",
    )
    min_signal_bars = st.number_input(
        "Signal persistence (bars)",
        min_value=1, max_value=20, value=2, step=1,
        help="Signal must be active for this many consecutive bars.",
    )

with st.sidebar.expander("Threshold Gates", expanded=True):
    hist_z_threshold = st.number_input(
        "Historical Vol Z-Score Gate",
        min_value=0.1, max_value=5.0, value=1.5, step=0.1,
        help="Gate 1: Asset's volatility must be high relative to its own history.",
    )
    cs_z_threshold = st.number_input(
        "Cross-Sectional Z-Score Gate",
        min_value=-1.0, max_value=5.0, value=1.0, step=0.1,
        help="Gate 2: Asset's volatility expansion must be higher than the market average.",
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

if "vol_brkout_v3_results" not in st.session_state:
    st.session_state["vol_brkout_v3_results"] = None

# ---------------------------------------------------------
# CLASS DEFINITION
# ---------------------------------------------------------
class VolatilityBreakoutSignal:
    """
    Volatility breakout signal generator — Streamlined.
    Requires Vol expansion (Time-series AND Cross-sectional) + Positive Momentum.
    """

    def __init__(
        self,
        historical_window: int = 90,
        volatility_window: int = 14,
        hist_z_threshold: float = 1.5,
        cs_z_threshold: float = 1.0,
        min_signal_bars: int = 2,
    ):
        self.historical_window = historical_window
        self.volatility_window = volatility_window
        self.hist_z_threshold = hist_z_threshold
        self.cs_z_threshold = cs_z_threshold
        self.min_signal_bars = min_signal_bars

    def generate_signals(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        # 1. Base Volatility
        returns = np.log(df / df.shift(1))
        vol = returns.rolling(window=self.volatility_window).std()

        # 2. Time-Series Z-Score (Relative to own history)
        ts_mean = vol.rolling(window=self.historical_window).mean()
        ts_std = vol.rolling(window=self.historical_window).std().replace(0, np.nan)
        hist_z = (vol - ts_mean) / ts_std

        # 3. Cross-Sectional Z-Score (Relative to peers)
        cs_mean = vol.mean(axis=1)
        cs_std = vol.std(axis=1).replace(0, np.nan)
        cs_z = vol.sub(cs_mean, axis=0).div(cs_std, axis=0)

        # 4. Filters: Momentum & Trend
        window_return = df / df.shift(self.volatility_window) - 1
        price_above_ma = df > df.rolling(window=self.historical_window).mean()

        # 5. Funnel Logic: All conditions must be met
        raw_signal = (
            (hist_z > self.hist_z_threshold) & 
            (cs_z > self.cs_z_threshold) & 
            (window_return > 0) & 
            price_above_ma
        ).astype(int)

        # 6. Persistence Filter
        persistent_signal = (
            raw_signal.rolling(window=self.min_signal_bars).min().fillna(0).astype(int)
        )

        return {
            "hist_z": hist_z,
            "cs_z": cs_z,
            "window_return": window_return,
            "raw_signal": raw_signal,
            "persistent_signal": persistent_signal
        }

    def get_top_signals(self, signals_dict: Dict, top_n: int = 20) -> pd.DataFrame:
        ts = signals_dict["persistent_signal"].index[-1]

        persistent = signals_dict["persistent_signal"].loc[ts]
        hist_z = signals_dict["hist_z"].loc[ts].dropna()
        cs_z = signals_dict["cs_z"].loc[ts].dropna()
        window_ret = signals_dict["window_return"].loc[ts].dropna()

        eligible = persistent[persistent == 1].index.tolist()

        if not eligible:
            return pd.DataFrame()

        df_out = pd.DataFrame({
            "coin": eligible,
            "hist_vol_zscore": hist_z.reindex(eligible).values,
            "cs_vol_zscore": cs_z.reindex(eligible).values,
            "window_return_pct": window_ret.reindex(eligible).values * 100,
        })

        # Rank by absolute magnitude of historical breakout
        df_out = df_out.sort_values("hist_vol_zscore", ascending=False)
        return df_out.head(top_n).reset_index(drop=True)

    def diagnose_signals(self, signals_dict: Dict) -> Dict:
        ts = signals_dict["persistent_signal"].index[-1]
        
        hist_z = signals_dict["hist_z"].loc[ts].dropna()
        cs_z = signals_dict["cs_z"].loc[ts].dropna()
        raw = signals_dict["raw_signal"].loc[ts]
        persistent = signals_dict["persistent_signal"].loc[ts]

        return {
            "timestamp": str(ts),
            "total_coins": int(len(hist_z)),
            "coins_raw_signal": int((raw == 1).sum()),
            "coins_persistent_signal": int((persistent == 1).sum()),
            "hist_z_threshold": float(self.hist_z_threshold),
            "cs_z_threshold": float(self.cs_z_threshold),
            "hist_z_max": float(hist_z.max()) if len(hist_z) else np.nan,
            "cs_z_max": float(cs_z.max()) if len(cs_z) else np.nan,
        }

# ---------------------------------------------------------
# MAIN COMPUTATION
# ---------------------------------------------------------
if refresh or st.session_state["vol_brkout_v3_results"] is None:
    with st.spinner("Computing signals…"):
        vb = VolatilityBreakoutSignal(
            historical_window=int(historical_window),
            volatility_window=int(volatility_window),
            hist_z_threshold=float(hist_z_threshold),
            cs_z_threshold=float(cs_z_threshold),
            min_signal_bars=int(min_signal_bars),
        )

        signals_dict = vb.generate_signals(df_h)
        base_top = vb.get_top_signals(signals_dict, top_n=int(top_n_total))
        diagnostics = vb.diagnose_signals(signals_dict)

        st.session_state["vol_brkout_v3_results"] = {
            "signals_dict": signals_dict,
            "base_top": base_top,
            "diagnostics": diagnostics,
            "vb": vb,
        }

results = st.session_state["vol_brkout_v3_results"]

if results is None:
    st.info("Click **Refresh / Recompute** in the sidebar to compute signals.")
    st.stop()

base_top: pd.DataFrame = results["base_top"]
diagnostics: Dict = results["diagnostics"]
signals_dict: Dict = results["signals_dict"]

# ---------------------------------------------------------
# DISPLAY — SELECTED RANK SLICE
# ---------------------------------------------------------
st.subheader(f"Breakout Candidates — Rank {int(rank_start)}–{int(rank_end)}")

if base_top is None or base_top.empty:
    st.warning(
        "No breakout signals under current parameters. "
        "Try lowering the historical or cross-sectional Z-score thresholds."
    )
else:
    display_df = base_top.iloc[int(rank_start) - 1 : int(rank_end)].copy()
    display_df.insert(0, "rank", range(int(rank_start), int(rank_start) + len(display_df)))

    # Format the dataframe cleanly
    styled = display_df.style.format({
        "hist_vol_zscore": "{:.2f}",
        "cs_vol_zscore": "{:.2f}",
        "window_return_pct": "{:.2f}%"
    })
    st.dataframe(styled, use_container_width=True)

    # ---------------------------------------------------------
    # CHART — Historical Z-Score Bar Chart
    # ---------------------------------------------------------
    if not display_df.empty:
        fig = go.Figure(
            go.Bar(
                x=display_df["coin"],
                y=display_df["hist_vol_zscore"],
                marker_color="#1f77b4",
                text=display_df["hist_vol_zscore"].round(2),
                textposition="outside",
            )
        )
        fig.update_layout(
            title="Historical Volatility Z-Score by Coin",
            xaxis_title="Coin",
            yaxis_title="Hist Z-Score",
            height=350,
            margin=dict(t=50, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# DISPLAY — Diagnostics & Distributions
# ---------------------------------------------------------
with st.expander("Diagnostics & Distributions", expanded=False):
    st.json(diagnostics)
    
    ts_last = signals_dict["hist_z"].index[-1]
    hist_z_row = signals_dict["hist_z"].loc[ts_last].dropna()
    cs_z_row = signals_dict["cs_z"].loc[ts_last].dropna()

    col1, col2 = st.columns(2)

    with col1:
        fig_hz = go.Figure()
        fig_hz.add_trace(go.Histogram(x=hist_z_row.values, nbinsx=30, name="Hist Z"))
        fig_hz.add_vline(
            x=float(hist_z_threshold),
            line_dash="dash", line_color="red",
            annotation_text=f"Gate 1: {hist_z_threshold}",
            annotation_position="top right",
        )
        fig_hz.update_layout(
            title="Historical Z-Score Distribution",
            height=300, margin=dict(t=40, b=30, l=20, r=20),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_hz, use_container_width=True)

    with col2:
        fig_cs = go.Figure()
        fig_cs.add_trace(go.Histogram(x=cs_z_row.values, nbinsx=30, name="CS Z"))
        fig_cs.add_vline(
            x=float(cs_z_threshold),
            line_dash="dash", line_color="orange",
            annotation_text=f"Gate 2: {cs_z_threshold}",
            annotation_position="top right",
        )
        fig_cs.update_layout(
            title="Cross-Sectional Z-Score Distribution",
            height=300, margin=dict(t=40, b=30, l=20, r=20),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_cs, use_container_width=True)
