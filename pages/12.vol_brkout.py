"""
Volatility Breakout Signals — v2
=================================
Key improvements over v1
--------------------------
1.  Breakout metric split into two components:
        • total_vol_zscore  — confirms a genuine vol-regime shift
        • vol_skew          — ranks breakout quality (upside vs downside proportion)
    Ranking uses vol_skew only among confirmed breakout candidates.

2.  Price momentum gate — coin must be trading above its rolling MA
    over the same historical window before it can appear in results.

3.  Signal persistence filter — signal must be active for at least
    `min_signal_bars` consecutive bars to filter single-bar noise.

4.  Minimum non-zero return guard — upside/downside std is computed only
    when enough non-zero observations exist; otherwise returns NaN.

5.  Volatility window raised (default 14) so clipped-return std is stable.

6.  Rank-start / rank-end are sidebar parameters (replaces hard-coded 11–20).

7.  Both-signal gate replaced: only `total_vol_zscore > historical_threshold`
    is required (strict gate). Cross-sectional z-score is used as a secondary
    ranking boost, not a hard gate — fixes altcoin bias.

8.  Diagnostic panel expanded with distribution plots.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Volatility Breakout v2", layout="wide")
st.title("Volatility Breakout Signals · v2")


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
        key="hw",
    )
    volatility_window = st.number_input(
        "Volatility window (bars)",
        min_value=5, max_value=96, value=14, step=1,
        help="Rolling window for current vol. Min 5 recommended for stable std.",
        key="vw",
    )
    min_nonzero_returns = st.number_input(
        "Min non-zero returns (in vol window)",
        min_value=2, max_value=20, value=5, step=1,
        help="Upside/downside std is set to NaN if fewer non-zero bars exist.",
        key="mnr",
    )
    min_signal_bars = st.number_input(
        "Signal persistence (bars)",
        min_value=1, max_value=20, value=2, step=1,
        help="Signal must be active for this many consecutive bars.",
        key="msb",
    )

with st.sidebar.expander("Thresholds", expanded=True):
    historical_threshold = st.number_input(
        "Historical vol z-score threshold",
        min_value=0.1, max_value=5.0, value=1.5, step=0.1,
        help="Hard gate: coin must exceed this threshold to be a candidate.",
        key="ht",
    )
    cs_boost_weight = st.number_input(
        "Cross-sectional boost weight",
        min_value=0.0, max_value=1.0, value=0.3, step=0.05,
        help=(
            "Weight applied to cross-sectional z-score when computing the "
            "composite ranking score. 0 = ignore cross-sectional signal entirely."
        ),
        key="csbw",
    )

with st.sidebar.expander("Ranking & Display", expanded=True):
    top_n_total = st.number_input(
        "Compute top N candidates",
        min_value=5, max_value=100, value=20, step=5,
        key="tnt",
    )
    rank_start = st.number_input(
        "Display rank start (1-indexed)",
        min_value=1, max_value=int(top_n_total), value=1, step=1,
        key="rs",
    )
    rank_end = st.number_input(
        "Display rank end (1-indexed)",
        min_value=int(rank_start), max_value=int(top_n_total),
        value=min(10, int(top_n_total)), step=1,
        key="re",
    )
    require_price_above_ma = st.checkbox(
        "Require price above MA (momentum gate)",
        value=True,
        help="Only coins trading above their rolling MA are eligible.",
        key="rpam",
    )

st.sidebar.markdown("---")
refresh = st.sidebar.button("🔄 Refresh / Recompute", use_container_width=True)

if "vol_brkout_v2_results" not in st.session_state:
    st.session_state["vol_brkout_v2_results"] = None


# ---------------------------------------------------------
# HELPER: robust clipped-return std
# ---------------------------------------------------------
def _robust_clip_std(series: pd.Series, min_nonzero: int) -> float:
    """
    Std of a clipped return series, requiring at least `min_nonzero`
    non-zero observations. Returns NaN otherwise.
    """
    nonzero = series[series != 0]
    if len(nonzero) < min_nonzero:
        return np.nan
    return float(nonzero.std())


# ---------------------------------------------------------
# CLASS DEFINITION
# ---------------------------------------------------------
class VolatilityBreakoutSignal:
    """
    Volatility breakout signal generator — v2.

    Separates the breakout confirmation (total vol z-score) from
    the breakout quality ranking (vol skew = upside proportion).
    """

    def __init__(
        self,
        historical_window: int = 90,
        volatility_window: int = 14,
        historical_threshold: float = 1.5,
        cs_boost_weight: float = 0.3,
        min_nonzero_returns: int = 5,
        min_signal_bars: int = 2,
    ):
        self.historical_window = historical_window
        self.volatility_window = volatility_window
        self.historical_threshold = historical_threshold
        self.cs_boost_weight = cs_boost_weight
        self.min_nonzero_returns = min_nonzero_returns
        self.min_signal_bars = min_signal_bars

    # ------------------------------------------------------------------
    # STEP 1 — Returns
    # ------------------------------------------------------------------
    def _calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        return np.log(df / df.shift(1)).dropna()

    # ------------------------------------------------------------------
    # STEP 2 — Total rolling vol (symmetric — for breakout detection)
    # ------------------------------------------------------------------
    def _calculate_total_vol(self, returns: pd.DataFrame) -> pd.DataFrame:
        return returns.rolling(window=self.volatility_window).std()

    # ------------------------------------------------------------------
    # STEP 3 — Vol skew (upside proportion — for breakout quality ranking)
    # ------------------------------------------------------------------
    def _calculate_vol_skew(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        vol_skew = upside_std / (upside_std + downside_std + ε)
        Range [0, 1].  → 1 = all upside vol, 0.5 = balanced, 0 = all downside.
        Computed with the robust clipped-std helper.
        """
        vw = self.volatility_window
        mn = self.min_nonzero_returns
        eps = 1e-8

        pos = returns.clip(lower=0)
        neg = returns.clip(upper=0).abs()

        upside_vol = pos.rolling(window=vw).apply(
            lambda s: _robust_clip_std(s, mn), raw=False
        )
        downside_vol = neg.rolling(window=vw).apply(
            lambda s: _robust_clip_std(s, mn), raw=False
        )

        vol_skew = upside_vol / (upside_vol + downside_vol + eps)
        return vol_skew

    # ------------------------------------------------------------------
    # STEP 4 — Historical z-score of total vol
    # ------------------------------------------------------------------
    def _historical_zscore(self, total_vol: pd.DataFrame) -> pd.DataFrame:
        roll_mean = total_vol.rolling(window=self.historical_window).mean()
        roll_std = total_vol.rolling(window=self.historical_window).std().replace(0, np.nan)
        return (total_vol - roll_mean) / roll_std

    # ------------------------------------------------------------------
    # STEP 5 — Cross-sectional z-score of total vol (secondary boost only)
    # ------------------------------------------------------------------
    def _cross_sectional_zscore(self, total_vol: pd.DataFrame) -> pd.DataFrame:
        cs_mean = total_vol.mean(axis=1)
        cs_std = total_vol.std(axis=1).replace(0, np.nan)
        return total_vol.sub(cs_mean, axis=0).div(cs_std, axis=0)

    # ------------------------------------------------------------------
    # STEP 6 — Price above MA gate
    # ------------------------------------------------------------------
    def _price_above_ma(self, df: pd.DataFrame) -> pd.Series:
        """
        Returns a boolean Series (per coin) indicating whether the latest
        closing price is above the rolling MA over the historical window.
        """
        ma = df.rolling(window=self.historical_window).mean()
        latest_price = df.iloc[-1]
        latest_ma = ma.iloc[-1]
        return latest_price > latest_ma

    # ------------------------------------------------------------------
    # STEP 7 — Composite ranking score
    # ------------------------------------------------------------------
    def _composite_score(
        self,
        hist_z: pd.DataFrame,
        cs_z: pd.DataFrame,
        vol_skew: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        composite = (1 − cs_boost_weight) × vol_skew_zscore
                    + cs_boost_weight × cs_z_norm

        vol_skew is z-scored historically so it's on a comparable scale.
        cs_z is kept raw (already cross-sectional).
        """
        vskew_mean = vol_skew.rolling(window=self.historical_window).mean()
        vskew_std = vol_skew.rolling(window=self.historical_window).std().replace(0, np.nan)
        vol_skew_z = (vol_skew - vskew_mean) / vskew_std

        w_cs = self.cs_boost_weight
        w_vs = 1.0 - w_cs
        composite = w_vs * vol_skew_z + w_cs * cs_z
        return composite

    # ------------------------------------------------------------------
    # PUBLIC: generate_signals
    # ------------------------------------------------------------------
    def generate_signals(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        returns = self._calculate_returns(df)
        total_vol = self._calculate_total_vol(returns)
        vol_skew = self._calculate_vol_skew(returns)

        hist_z = self._historical_zscore(total_vol)
        cs_z = self._cross_sectional_zscore(total_vol)
        composite = self._composite_score(hist_z, cs_z, vol_skew)

        # Hard gate: vol breakout confirmed
        raw_signal = (hist_z > self.historical_threshold).astype(int)

        # Persistence filter: must be active for min_signal_bars consecutive bars
        persistent_signal = (
            raw_signal
            .rolling(window=self.min_signal_bars)
            .min()
            .fillna(0)
            .astype(int)
        )

        return {
            "returns": returns,
            "total_vol": total_vol,
            "vol_skew": vol_skew,
            "hist_z": hist_z,
            "cs_z": cs_z,
            "composite": composite,
            "raw_signal": raw_signal,
            "persistent_signal": persistent_signal,
            "price_above_ma": self._price_above_ma(df),
        }

    # ------------------------------------------------------------------
    # PUBLIC: get_top_signals
    # ------------------------------------------------------------------
    def get_top_signals(
        self,
        signals_dict: Dict,
        top_n: int = 20,
        require_price_above_ma: bool = True,
    ) -> pd.DataFrame:
        """
        Returns a ranked DataFrame of breakout candidates.

        Eligibility:
          • persistent_signal == 1  (vol breakout confirmed + persistence)
          • price_above_ma == True  (optional momentum gate)

        Ranked by: composite score (descending).
        """
        ts = signals_dict["persistent_signal"].index[-1]

        persistent = signals_dict["persistent_signal"].loc[ts]
        composite = signals_dict["composite"].loc[ts].dropna()
        hist_z = signals_dict["hist_z"].loc[ts]
        cs_z = signals_dict["cs_z"].loc[ts]
        vol_skew = signals_dict["vol_skew"].loc[ts]
        price_above_ma = signals_dict["price_above_ma"]

        # Eligible coins: signal active
        eligible = persistent[persistent == 1].index.tolist()

        # Momentum gate
        if require_price_above_ma:
            eligible = [c for c in eligible if price_above_ma.get(c, False)]

        # Must have composite score
        eligible = [c for c in eligible if c in composite.index]

        if not eligible:
            return pd.DataFrame()

        df_out = pd.DataFrame({
            "coin": eligible,
            "composite_score": composite.reindex(eligible).values,
            "hist_vol_zscore": hist_z.reindex(eligible).values,
            "cs_vol_zscore": cs_z.reindex(eligible).values,
            "vol_skew": vol_skew.reindex(eligible).values,
            "price_above_ma": price_above_ma.reindex(eligible).values
                              if hasattr(price_above_ma, "reindex")
                              else [price_above_ma.get(c, np.nan) for c in eligible],
        })

        df_out = df_out.sort_values("composite_score", ascending=False)
        return df_out.head(top_n).reset_index(drop=True)

    # ------------------------------------------------------------------
    # PUBLIC: diagnose_signals
    # ------------------------------------------------------------------
    def diagnose_signals(self, signals_dict: Dict) -> Dict:
        ts = signals_dict["persistent_signal"].index[-1]

        hist_z = signals_dict["hist_z"].loc[ts].dropna()
        cs_z = signals_dict["cs_z"].loc[ts].dropna()
        vol_skew = signals_dict["vol_skew"].loc[ts].dropna()
        raw = signals_dict["raw_signal"].loc[ts]
        persistent = signals_dict["persistent_signal"].loc[ts]
        price_above_ma = signals_dict["price_above_ma"]

        n_raw = int((raw == 1).sum())
        n_persistent = int((persistent == 1).sum())
        n_ma_pass = int(price_above_ma.sum()) if hasattr(price_above_ma, "sum") else "N/A"

        return {
            "timestamp": str(ts),
            "total_coins": int(len(hist_z)),
            "coins_raw_signal": n_raw,
            "coins_persistent_signal": n_persistent,
            "coins_price_above_ma": n_ma_pass,
            "historical_threshold": float(self.historical_threshold),
            "min_signal_bars": int(self.min_signal_bars),
            "hist_z_max": float(hist_z.max()) if len(hist_z) else np.nan,
            "hist_z_mean": float(hist_z.mean()) if len(hist_z) else np.nan,
            "hist_z_std": float(hist_z.std()) if len(hist_z) else np.nan,
            "cs_z_max": float(cs_z.max()) if len(cs_z) else np.nan,
            "cs_z_mean": float(cs_z.mean()) if len(cs_z) else np.nan,
            "vol_skew_mean": float(vol_skew.mean()) if len(vol_skew) else np.nan,
            "vol_skew_median": float(vol_skew.median()) if len(vol_skew) else np.nan,
        }


# ---------------------------------------------------------
# MAIN COMPUTATION
# ---------------------------------------------------------
if refresh or st.session_state["vol_brkout_v2_results"] is None:
    with st.spinner("Computing signals…"):
        vb = VolatilityBreakoutSignal(
            historical_window=int(historical_window),
            volatility_window=int(volatility_window),
            historical_threshold=float(historical_threshold),
            cs_boost_weight=float(cs_boost_weight),
            min_nonzero_returns=int(min_nonzero_returns),
            min_signal_bars=int(min_signal_bars),
        )

        signals_dict = vb.generate_signals(df_h)

        base_top = vb.get_top_signals(
            signals_dict,
            top_n=int(top_n_total),
            require_price_above_ma=bool(require_price_above_ma),
        )

        diagnostics = vb.diagnose_signals(signals_dict)

        st.session_state["vol_brkout_v2_results"] = {
            "signals_dict": signals_dict,
            "base_top": base_top,
            "diagnostics": diagnostics,
            "vb": vb,
        }

results = st.session_state["vol_brkout_v2_results"]

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
        "Try lowering the historical z-score threshold or the signal persistence bars."
    )
else:
    # Rank is 1-indexed in the DataFrame; iloc uses 0-based
    display_df = base_top.iloc[int(rank_start) - 1 : int(rank_end)].copy()
    display_df.insert(0, "rank", range(int(rank_start), int(rank_start) + len(display_df)))

    # Colour-code vol_skew: green when skewed upward, red when balanced/downward
    def _color_skew(val):
        try:
            v = float(val)
            if v >= 0.65:
                return "background-color: #1a5c2a; color: white"
            elif v >= 0.55:
                return "background-color: #2e7d32; color: white"
            elif v >= 0.45:
                return ""
            else:
                return "background-color: #7b1a1a; color: white"
        except Exception:
            return ""

    styled = display_df.style.applymap(_color_skew, subset=["vol_skew"])
    st.dataframe(styled, use_container_width=True)

    # ---------------------------------------------------------
    # CHART — Composite score bar chart for displayed coins
    # ---------------------------------------------------------
    if not display_df.empty:
        fig = go.Figure(
            go.Bar(
                x=display_df["coin"],
                y=display_df["composite_score"],
                marker_color=[
                    "#2e7d32" if v >= 0.55 else "#c62828"
                    for v in display_df["vol_skew"].fillna(0.5)
                ],
                text=display_df["composite_score"].round(2),
                textposition="outside",
            )
        )
        fig.update_layout(
            title="Composite Breakout Score (selected rank slice)",
            xaxis_title="Coin",
            yaxis_title="Composite Score",
            height=350,
            margin=dict(t=50, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------
# DISPLAY — Full base ranking table
# ---------------------------------------------------------
with st.expander("Full base ranking (all computed candidates)", expanded=False):
    if base_top is not None and not base_top.empty:
        full = base_top.copy()
        full.insert(0, "rank", range(1, len(full) + 1))
        st.dataframe(full, use_container_width=True)
    else:
        st.write("No candidates to show.")


# ---------------------------------------------------------
# DISPLAY — Distribution plots (diagnostic)
# ---------------------------------------------------------
with st.expander("Signal distribution charts", expanded=False):
    ts_last = signals_dict["hist_z"].index[-1]
    hist_z_row = signals_dict["hist_z"].loc[ts_last].dropna()
    cs_z_row = signals_dict["cs_z"].loc[ts_last].dropna()
    vol_skew_row = signals_dict["vol_skew"].loc[ts_last].dropna()

    col1, col2, col3 = st.columns(3)

    with col1:
        fig_hz = go.Figure()
        fig_hz.add_trace(go.Histogram(x=hist_z_row.values, nbinsx=20, name="Hist Z"))
        fig_hz.add_vline(
            x=float(historical_threshold),
            line_dash="dash", line_color="red",
            annotation_text=f"threshold={historical_threshold}",
            annotation_position="top right",
        )
        fig_hz.update_layout(
            title="Historical Vol Z-score",
            height=280, margin=dict(t=40, b=30, l=20, r=20),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_hz, use_container_width=True)

    with col2:
        fig_cs = go.Figure()
        fig_cs.add_trace(go.Histogram(x=cs_z_row.values, nbinsx=20, name="CS Z"))
        fig_cs.update_layout(
            title="Cross-Sectional Vol Z-score",
            height=280, margin=dict(t=40, b=30, l=20, r=20),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_cs, use_container_width=True)

    with col3:
        fig_sk = go.Figure()
        fig_sk.add_trace(go.Histogram(x=vol_skew_row.values, nbinsx=20, name="Vol Skew"))
        fig_sk.add_vline(
            x=0.5, line_dash="dash", line_color="gray",
            annotation_text="balanced=0.5",
        )
        fig_sk.update_layout(
            title="Vol Skew (upside proportion)",
            height=280, margin=dict(t=40, b=30, l=20, r=20),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_sk, use_container_width=True)


# ---------------------------------------------------------
# DISPLAY — Diagnostics JSON
# ---------------------------------------------------------
with st.expander("Diagnostics", expanded=False):
    st.json(diagnostics)
