"""
Volatility Breakout Signals — v5 (Composite Weighted Score)
=============================================================
Approach:
---------
Instead of hard binary gates that often produce zero signals, each component
(vol contraction, vol expansion, donchian proximity, momentum) is converted
into a continuous 0-1 score. These are combined into a single weighted composite
score per coin. Coins are ranked by this score within each theme, and the top N
per theme are displayed in a unified table.

Score Components:
-----------------
1.  Squeeze Score   : How compressed was recent vol relative to baseline?
                      Higher = more coiled.
2.  Expansion Score : How much has short-term vol expanded vs long-term?
                      Higher = more explosive.
3.  Donchian Score  : How close is today's close to the rolling N-day high?
                      1.0 = at/above the high, tapers off below.
4.  Momentum Score  : Positive recent return bias (5-day log return > 0).
                      Continuous: maps return quantile → 0-1.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from typing import Dict, Optional

# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Volatility Breakout: Composite Score", layout="wide")
st.title("Volatility Breakout · Composite Score Ranker")
st.caption(
    "Coins are ranked within each theme by a weighted composite of four continuous "
    "volatility and momentum factors — no hard gates, always produces rankings."
)

# ---------------------------------------------------------
# INPUT DATA FROM SESSION STATE
# ---------------------------------------------------------
df_h: Optional[pd.DataFrame] = st.session_state.get("price_theme", None)
theme_map: Optional[pd.Series] = st.session_state.get("coin_theme_map", None)

if df_h is None or df_h.empty:
    st.error(
        "price_theme not found in st.session_state or is empty.\n\n"
        "Please go to the Themes Tracker page, click **Refresh / Fetch Data**, "
        "and then come back to this page."
    )
    st.stop()

# Build theme map: expects either a Series (coin → theme) or a DataFrame with
# columns ['coin', 'theme']. Fall back to a single "All" bucket if unavailable.
if theme_map is None:
    theme_col = st.session_state.get("df_meta", None)
    if theme_col is not None and "theme" in theme_col.columns and "coin" in theme_col.columns:
        theme_map = theme_col.set_index("coin")["theme"]
    else:
        # Assign every coin to a single theme so the page still works
        theme_map = pd.Series("All", index=df_h.columns, name="theme")

# Align theme_map to the columns in df_h
theme_map = theme_map.reindex(df_h.columns).fillna("Unknown")

# ---------------------------------------------------------
# SIDEBAR PARAMETERS
# ---------------------------------------------------------
st.sidebar.header("Parameters")

with st.sidebar.expander("Lookback Windows", expanded=True):
    donchian_window = st.number_input(
        "Donchian Window (bars)",
        min_value=5, max_value=100, value=20, step=5,
        help="Rolling high used to score proximity to new high.",
    )
    vol_long_window = st.number_input(
        "Long-Term Vol Window (bars)",
        min_value=20, max_value=200, value=90, step=10,
        help="Baseline for normal market volatility.",
    )
    vol_short_window = st.number_input(
        "Short-Term Vol Window (bars)",
        min_value=5, max_value=50, value=14, step=1,
        help="Recent volatility window.",
    )
    squeeze_lookback = st.number_input(
        "Squeeze Lookback (bars)",
        min_value=1, max_value=20, value=7, step=1,
        help="Window over which to measure the minimum short/long vol ratio (contraction depth).",
    )
    momentum_window = st.number_input(
        "Momentum Window (bars)",
        min_value=2, max_value=30, value=5, step=1,
        help="Log return lookback for the momentum component.",
    )

with st.sidebar.expander("Component Weights", expanded=True):
    w_squeeze = st.slider(
        "Squeeze Score weight",
        min_value=0.0, max_value=1.0, value=0.25, step=0.05,
        help="Weight for how coiled the coin was recently.",
    )
    w_expansion = st.slider(
        "Expansion Score weight",
        min_value=0.0, max_value=1.0, value=0.35, step=0.05,
        help="Weight for current vol expansion burst.",
    )
    w_donchian = st.slider(
        "Donchian Score weight",
        min_value=0.0, max_value=1.0, value=0.25, step=0.05,
        help="Weight for proximity to N-day high.",
    )
    w_momentum = st.slider(
        "Momentum Score weight",
        min_value=0.0, max_value=1.0, value=0.15, step=0.05,
        help="Weight for recent directional momentum.",
    )

    total_w = w_squeeze + w_expansion + w_donchian + w_momentum
    if abs(total_w) < 1e-9:
        st.error("All weights are zero — please set at least one weight > 0.")
        st.stop()

    st.caption(f"Weights sum to **{total_w:.2f}** (auto-normalised internally).")

with st.sidebar.expander("Output Settings", expanded=True):
    top_n_per_theme = st.number_input(
        "Top N per theme",
        min_value=1, max_value=20, value=3, step=1,
        help="Number of top coins to show for each theme.",
    )

st.sidebar.markdown("---")
refresh = st.sidebar.button("🔄 Refresh / Recompute", use_container_width=True)

if "vol_v5_results" not in st.session_state:
    st.session_state["vol_v5_results"] = None


# ---------------------------------------------------------
# SCORING ENGINE
# ---------------------------------------------------------
class CompositeVolScorer:
    """
    Converts raw price history into per-coin composite scores using four
    continuous, normalised components — no hard binary gates.
    """

    def __init__(
        self,
        donchian_window: int = 20,
        vol_short_window: int = 14,
        vol_long_window: int = 90,
        squeeze_lookback: int = 7,
        momentum_window: int = 5,
        weights: Dict[str, float] = None,
    ):
        self.donchian_window = donchian_window
        self.vol_short = vol_short_window
        self.vol_long = vol_long_window
        self.squeeze_lookback = squeeze_lookback
        self.momentum_window = momentum_window
        self.weights = weights or {
            "squeeze": 0.25,
            "expansion": 0.35,
            "donchian": 0.25,
            "momentum": 0.15,
        }
        # Normalise weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _rank_norm(series: pd.Series) -> pd.Series:
        """Cross-sectional rank normalisation → [0, 1]. NaNs remain NaN."""
        ranked = series.rank(pct=True, na_option="keep")
        return ranked

    # ------------------------------------------------------------------
    # score computation
    # ------------------------------------------------------------------
    def compute_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a DataFrame indexed by coin with columns:
            squeeze_raw, expansion_raw, donchian_raw, momentum_raw,
            squeeze_score, expansion_score, donchian_score, momentum_score,
            composite_score
        """
        log_ret = np.log(df / df.shift(1))
        short_vol = log_ret.rolling(self.vol_short).std()
        long_vol  = log_ret.rolling(self.vol_long).std()

        ts = df.index[-1]

        # ── 1. Squeeze raw ─────────────────────────────────────────────
        # Minimum (short/long) ratio over the squeeze lookback window — lower
        # ratio = deeper contraction = more coiled.  We invert so higher = better.
        vol_ratio = short_vol / long_vol.replace(0, np.nan)
        min_ratio_recent = vol_ratio.rolling(self.squeeze_lookback).min()
        squeeze_raw = 1.0 - min_ratio_recent.loc[ts].clip(0, 2) / 2.0

        # ── 2. Expansion raw ───────────────────────────────────────────
        # Current short/long ratio — higher = more explosive right now.
        expansion_raw = vol_ratio.loc[ts].clip(0, 5) / 5.0

        # ── 3. Donchian proximity raw ──────────────────────────────────
        # How close is today's close to the N-day rolling high (using prior bars).
        rolling_high = df.shift(1).rolling(self.donchian_window).max()
        high_today   = rolling_high.loc[ts].replace(0, np.nan)
        close_today  = df.loc[ts]
        # ratio >= 1 means at/above the high
        donchian_ratio = (close_today / high_today).clip(0, 1.1)
        donchian_raw   = donchian_ratio.clip(0, 1.0)  # cap at 1.0

        # ── 4. Momentum raw ────────────────────────────────────────────
        # N-day log return; map via cross-sectional percentile → 0-1
        momentum_ret = log_ret.rolling(self.momentum_window).sum().loc[ts]
        momentum_raw = momentum_ret  # will be rank-normalised below

        # ── combine into one DataFrame ─────────────────────────────────
        scores_df = pd.DataFrame({
            "squeeze_raw":    squeeze_raw,
            "expansion_raw":  expansion_raw,
            "donchian_raw":   donchian_raw,
            "momentum_raw":   momentum_ret,
        }).dropna(how="all")

        # Cross-sectional rank normalisation for each component
        scores_df["squeeze_score"]   = self._rank_norm(scores_df["squeeze_raw"])
        scores_df["expansion_score"] = self._rank_norm(scores_df["expansion_raw"])
        scores_df["donchian_score"]  = self._rank_norm(scores_df["donchian_raw"])
        scores_df["momentum_score"]  = self._rank_norm(scores_df["momentum_raw"])

        # Weighted composite
        scores_df["composite_score"] = (
            self.weights["squeeze"]   * scores_df["squeeze_score"]
            + self.weights["expansion"] * scores_df["expansion_score"]
            + self.weights["donchian"]  * scores_df["donchian_score"]
            + self.weights["momentum"]  * scores_df["momentum_score"]
        )

        # Attach raw vol info for display
        scores_df["short_vol"] = short_vol.loc[ts].reindex(scores_df.index)
        scores_df["long_vol"]  = long_vol.loc[ts].reindex(scores_df.index)
        scores_df["vol_expansion_ratio"] = vol_ratio.loc[ts].reindex(scores_df.index)

        return scores_df.reset_index().rename(columns={"index": "coin"})

    def top_n_per_theme(
        self,
        scores_df: pd.DataFrame,
        theme_map: pd.Series,
        top_n: int = 3,
    ) -> pd.DataFrame:
        """
        Join theme labels, rank within each theme, return top N per theme.
        """
        scores_df = scores_df.copy()
        scores_df["theme"] = scores_df["coin"].map(theme_map)
        scores_df["theme_rank"] = (
            scores_df
            .groupby("theme")["composite_score"]
            .rank(ascending=False, method="first")
            .astype(int)
        )
        top = scores_df[scores_df["theme_rank"] <= top_n].copy()
        top = top.sort_values(["theme", "theme_rank"]).reset_index(drop=True)
        return top


# ---------------------------------------------------------
# MAIN COMPUTATION
# ---------------------------------------------------------
def run_computation():
    weights = {
        "squeeze":   float(w_squeeze),
        "expansion": float(w_expansion),
        "donchian":  float(w_donchian),
        "momentum":  float(w_momentum),
    }
    scorer = CompositeVolScorer(
        donchian_window=int(donchian_window),
        vol_short_window=int(vol_short_window),
        vol_long_window=int(vol_long_window),
        squeeze_lookback=int(squeeze_lookback),
        momentum_window=int(momentum_window),
        weights=weights,
    )
    scores_df = scorer.compute_scores(df_h)
    top_df    = scorer.top_n_per_theme(scores_df, theme_map, top_n=int(top_n_per_theme))
    return scorer, scores_df, top_df


if refresh or st.session_state["vol_v5_results"] is None:
    with st.spinner("Computing composite volatility scores…"):
        scorer, scores_df, top_df = run_computation()
        st.session_state["vol_v5_results"] = {
            "scorer":    scorer,
            "scores_df": scores_df,
            "top_df":    top_df,
        }

results   = st.session_state["vol_v5_results"]
scorer    = results["scorer"]
scores_df = results["scores_df"]
top_df    = results["top_df"]

# ---------------------------------------------------------
# DISPLAY — TOP N PER THEME TABLE
# ---------------------------------------------------------
st.subheader(f"Top {int(top_n_per_theme)} Coins per Theme · Composite Score")

if top_df.empty:
    st.warning("No scores could be computed. Check that price_theme has sufficient history.")
else:
    display_cols = [
        "theme", "theme_rank", "coin",
        "composite_score",
        "squeeze_score", "expansion_score", "donchian_score", "momentum_score",
        "vol_expansion_ratio", "short_vol", "long_vol",
    ]
    # Only keep columns that actually exist (safety for partial data)
    display_cols = [c for c in display_cols if c in top_df.columns]
    display_df = top_df[display_cols].copy()

    # Rename for readability
    display_df = display_df.rename(columns={
        "theme_rank":          "Rank",
        "theme":               "Theme",
        "coin":                "Coin",
        "composite_score":     "Composite",
        "squeeze_score":       "Squeeze",
        "expansion_score":     "Expansion",
        "donchian_score":      "Donchian",
        "momentum_score":      "Momentum",
        "vol_expansion_ratio": "Vol Ratio",
        "short_vol":           "Short Vol",
        "long_vol":            "Long Vol",
    })

    fmt = {
        "Composite":  "{:.3f}",
        "Squeeze":    "{:.3f}",
        "Expansion":  "{:.3f}",
        "Donchian":   "{:.3f}",
        "Momentum":   "{:.3f}",
        "Vol Ratio":  "{:.2f}x",
        "Short Vol":  "{:.4f}",
        "Long Vol":   "{:.4f}",
    }
    fmt = {k: v for k, v in fmt.items() if k in display_df.columns}

    st.dataframe(
        display_df.style
            .format(fmt)
            .background_gradient(subset=["Composite"], cmap="YlOrRd"),
        use_container_width=True,
        height=min(60 + len(display_df) * 35, 800),
    )

# ---------------------------------------------------------
# CHART — Top N per theme: composite score bar chart
# ---------------------------------------------------------
if not top_df.empty:
    st.subheader("Composite Score by Coin · Coloured by Theme")

    # Assign a colour per theme
    themes      = top_df["theme"].unique().tolist()
    palette     = [
        "#e45756", "#4c78a8", "#f28e2b", "#72b7b2", "#54a24b",
        "#b279a2", "#ff9da6", "#9d755d", "#bab0ac", "#eeca3b",
    ]
    theme_color = {t: palette[i % len(palette)] for i, t in enumerate(themes)}

    fig = go.Figure()
    for theme in themes:
        sub = top_df[top_df["theme"] == theme].sort_values("composite_score", ascending=False)
        fig.add_trace(go.Bar(
            name=theme,
            x=sub["coin"],
            y=sub["composite_score"],
            marker_color=theme_color[theme],
            text=[f"{v:.3f}" for v in sub["composite_score"]],
            textposition="outside",
        ))

    fig.update_layout(
        barmode="group",
        xaxis_title="Coin",
        yaxis_title="Composite Score",
        yaxis_range=[0, 1.05],
        height=420,
        margin=dict(t=40, b=40),
        legend_title="Theme",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# DISPLAY — Full Universe Scores (debug)
# ---------------------------------------------------------
with st.expander("Full Universe Scores (all coins, sorted by composite)", expanded=False):
    full_display = scores_df.copy()
    full_display["theme"] = full_display["coin"].map(theme_map)
    full_display = full_display.sort_values("composite_score", ascending=False).reset_index(drop=True)
    full_display.insert(0, "rank", range(1, len(full_display) + 1))

    cols_to_show = [
        "rank", "theme", "coin", "composite_score",
        "squeeze_score", "expansion_score", "donchian_score", "momentum_score",
        "vol_expansion_ratio",
    ]
    cols_to_show = [c for c in cols_to_show if c in full_display.columns]

    st.dataframe(
        full_display[cols_to_show].style.format({
            "composite_score":  "{:.3f}",
            "squeeze_score":    "{:.3f}",
            "expansion_score":  "{:.3f}",
            "donchian_score":   "{:.3f}",
            "momentum_score":   "{:.3f}",
            "vol_expansion_ratio": "{:.2f}x",
        }).background_gradient(subset=["composite_score"], cmap="YlOrRd"),
        use_container_width=True,
        height=400,
    )

# ---------------------------------------------------------
# DISPLAY — Weight Summary
# ---------------------------------------------------------
with st.expander("Active Weight Configuration", expanded=False):
    w_df = pd.DataFrame([
        {"Component": "Squeeze (coil depth)",    "Raw Weight": w_squeeze,   "Normalised": scorer.weights["squeeze"]},
        {"Component": "Expansion (vol burst)",   "Raw Weight": w_expansion, "Normalised": scorer.weights["expansion"]},
        {"Component": "Donchian (price at high)","Raw Weight": w_donchian,  "Normalised": scorer.weights["donchian"]},
        {"Component": "Momentum (recent return)","Raw Weight": w_momentum,  "Normalised": scorer.weights["momentum"]},
    ])
    st.dataframe(
        w_df.style.format({"Raw Weight": "{:.2f}", "Normalised": "{:.3f}"}),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Normalised = raw weight ÷ sum of all raw weights. "
        "The composite score is the weighted average of four cross-sectionally rank-normalised (0–1) components."
    )
