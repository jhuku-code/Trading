import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from typing import Dict, Optional

# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------

st.set_page_config(
    page_title="Volatility Breakout: Composite Score",
    layout="wide"
)

st.title("Volatility Breakout · Composite Score Ranker")

st.caption(
    "Coins are ranked within each theme by a weighted composite "
    "of volatility compression, expansion, breakout proximity, "
    "and momentum."
)

# ---------------------------------------------------------
# INPUT DATA
# ---------------------------------------------------------

df_h: Optional[pd.DataFrame] = st.session_state.get(
    "price_theme",
    None
)

theme_map: Optional[pd.Series] = st.session_state.get(
    "coin_theme_map",
    None
)

if df_h is None or df_h.empty:
    st.error(
        "price_theme not found in st.session_state."
    )
    st.stop()

# ---------------------------------------------------------
# THEME FALLBACK
# ---------------------------------------------------------

if theme_map is None:

    theme_col = st.session_state.get(
        "df_meta",
        None
    )

    if (
        theme_col is not None
        and "theme" in theme_col.columns
        and "coin" in theme_col.columns
    ):

        theme_map = (
            theme_col
            .set_index("coin")["theme"]
        )

    else:
        theme_map = pd.Series(
            "All",
            index=df_h.columns,
            name="theme"
        )

theme_map = (
    theme_map
    .reindex(df_h.columns)
    .fillna("Unknown")
)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.header("Parameters")

donchian_window = st.sidebar.number_input(
    "Donchian Window",
    min_value=5,
    max_value=100,
    value=20,
    step=5,
)

vol_long_window = st.sidebar.number_input(
    "Long Vol Window",
    min_value=20,
    max_value=200,
    value=90,
    step=10,
)

vol_short_window = st.sidebar.number_input(
    "Short Vol Window",
    min_value=5,
    max_value=50,
    value=14,
    step=1,
)

squeeze_lookback = st.sidebar.number_input(
    "Squeeze Lookback",
    min_value=1,
    max_value=20,
    value=7,
    step=1,
)

momentum_window = st.sidebar.number_input(
    "Momentum Window",
    min_value=2,
    max_value=30,
    value=5,
    step=1,
)

# ---------------------------------------------------------
# WEIGHTS
# ---------------------------------------------------------

st.sidebar.subheader("Weights")

w_squeeze = st.sidebar.slider(
    "Squeeze",
    0.0,
    1.0,
    0.25,
    0.05,
)

w_expansion = st.sidebar.slider(
    "Expansion",
    0.0,
    1.0,
    0.35,
    0.05,
)

w_donchian = st.sidebar.slider(
    "Donchian",
    0.0,
    1.0,
    0.25,
    0.05,
)

w_momentum = st.sidebar.slider(
    "Momentum",
    0.0,
    1.0,
    0.15,
    0.05,
)

top_n_per_theme = st.sidebar.number_input(
    "Top N per Theme",
    min_value=1,
    max_value=20,
    value=3,
    step=1,
)

# ---------------------------------------------------------
# SCORER CLASS
# ---------------------------------------------------------

class CompositeVolScorer:

    def __init__(
        self,
        donchian_window=20,
        vol_short_window=14,
        vol_long_window=90,
        squeeze_lookback=7,
        momentum_window=5,
        weights=None,
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

        total = sum(self.weights.values())

        self.weights = {
            k: v / total
            for k, v in self.weights.items()
        }

    # -----------------------------------------------------
    # Helpers
    # -----------------------------------------------------

    @staticmethod
    def _rank_norm(series):

        if series.dropna().empty:
            return pd.Series(
                np.nan,
                index=series.index
            )

        return series.rank(
            pct=True,
            na_option="keep"
        )

    @staticmethod
    def _safe_divide(a, b):

        out = a / b.replace(0, np.nan)

        out = out.replace(
            [np.inf, -np.inf],
            np.nan
        )

        return out

    # -----------------------------------------------------
    # Main Score Computation
    # -----------------------------------------------------

    def compute_scores(self, df):

        if df.empty:
            return pd.DataFrame()

        # Clean price data
        df = df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        # Remove fully empty coins
        df = df.dropna(
            axis=1,
            how="all"
        )

        if df.empty:
            return pd.DataFrame()

        # -------------------------------------------------
        # Returns
        # -------------------------------------------------

        log_ret = np.log(df / df.shift(1))

        log_ret = log_ret.replace(
            [np.inf, -np.inf],
            np.nan
        )

        # -------------------------------------------------
        # Rolling Volatility
        # -------------------------------------------------

        short_vol = (
            log_ret
            .rolling(
                self.vol_short,
                min_periods=self.vol_short
            )
            .std()
        )

        long_vol = (
            log_ret
            .rolling(
                self.vol_long,
                min_periods=self.vol_long
            )
            .std()
        )

        ts = df.index[-1]

        # -------------------------------------------------
        # Vol Ratio
        # -------------------------------------------------

        vol_ratio = self._safe_divide(
            short_vol,
            long_vol
        )

        # -------------------------------------------------
        # Squeeze
        # -------------------------------------------------

        min_ratio_recent = (
            vol_ratio
            .rolling(
                self.squeeze_lookback,
                min_periods=1
            )
            .min()
        )

        squeeze_raw = (
            1
            - min_ratio_recent.loc[ts]
            .clip(0, 2)
            / 2
        )

        # -------------------------------------------------
        # Expansion
        # -------------------------------------------------

        expansion_raw = (
            vol_ratio.loc[ts]
            .clip(0, 5)
            / 5
        )

        # -------------------------------------------------
        # Donchian
        # -------------------------------------------------

        rolling_high = (
            df.shift(1)
            .rolling(
                self.donchian_window,
                min_periods=self.donchian_window
            )
            .max()
        )

        high_today = rolling_high.loc[ts]
        close_today = df.loc[ts]

        donchian_ratio = self._safe_divide(
            close_today,
            high_today
        )

        donchian_raw = donchian_ratio.clip(0, 1)

        # -------------------------------------------------
        # Momentum
        # -------------------------------------------------

        momentum_ret = (
            log_ret
            .rolling(
                self.momentum_window,
                min_periods=self.momentum_window
            )
            .sum()
            .loc[ts]
        )

        # -------------------------------------------------
        # Build Score DF
        # -------------------------------------------------

        scores_df = pd.DataFrame({

            "squeeze_raw": squeeze_raw,
            "expansion_raw": expansion_raw,
            "donchian_raw": donchian_raw,
            "momentum_raw": momentum_ret,

        })

        scores_df = scores_df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        # -------------------------------------------------
        # Rank Normalize
        # -------------------------------------------------

        scores_df["squeeze_score"] = self._rank_norm(
            scores_df["squeeze_raw"]
        )

        scores_df["expansion_score"] = self._rank_norm(
            scores_df["expansion_raw"]
        )

        scores_df["donchian_score"] = self._rank_norm(
            scores_df["donchian_raw"]
        )

        scores_df["momentum_score"] = self._rank_norm(
            scores_df["momentum_raw"]
        )

        # -------------------------------------------------
        # Drop incomplete rows
        # -------------------------------------------------

        required_cols = [
            "squeeze_score",
            "expansion_score",
            "donchian_score",
            "momentum_score",
        ]

        scores_df = scores_df.dropna(
            subset=required_cols
        )

        if scores_df.empty:
            return pd.DataFrame()

        # -------------------------------------------------
        # Composite
        # -------------------------------------------------

        scores_df["composite_score"] = (

            self.weights["squeeze"]
            * scores_df["squeeze_score"]

            + self.weights["expansion"]
            * scores_df["expansion_score"]

            + self.weights["donchian"]
            * scores_df["donchian_score"]

            + self.weights["momentum"]
            * scores_df["momentum_score"]

        )

        # -------------------------------------------------
        # Diagnostics
        # -------------------------------------------------

        scores_df["short_vol"] = short_vol.loc[ts]

        scores_df["long_vol"] = long_vol.loc[ts]

        scores_df["vol_expansion_ratio"] = (
            vol_ratio.loc[ts]
        )

        # Final cleanup
        scores_df = scores_df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        scores_df = scores_df.dropna(
            subset=["composite_score"]
        )

        scores_df = (
            scores_df
            .reset_index()
            .rename(columns={"index": "coin"})
        )

        return scores_df

    # -----------------------------------------------------
    # Theme Ranking
    # -----------------------------------------------------

    def top_n_per_theme(
        self,
        scores_df,
        theme_map,
        top_n=3,
    ):

        if scores_df.empty:
            return pd.DataFrame()

        scores_df = scores_df.copy()

        scores_df = scores_df.dropna(
            subset=["composite_score"]
        )

        scores_df["theme"] = (
            scores_df["coin"]
            .map(theme_map)
            .fillna("Unknown")
        )

        scores_df["theme_rank"] = (

            scores_df
            .groupby("theme")["composite_score"]
            .rank(
                ascending=False,
                method="first"
            )
            .astype("Int64")

        )

        top_df = (
            scores_df[
                scores_df["theme_rank"] <= top_n
            ]
            .sort_values(
                ["theme", "theme_rank"]
            )
            .reset_index(drop=True)
        )

        return top_df


# ---------------------------------------------------------
# RUN COMPUTATION
# ---------------------------------------------------------

weights = {
    "squeeze": float(w_squeeze),
    "expansion": float(w_expansion),
    "donchian": float(w_donchian),
    "momentum": float(w_momentum),
}

scorer = CompositeVolScorer(

    donchian_window=int(donchian_window),

    vol_short_window=int(vol_short_window),

    vol_long_window=int(vol_long_window),

    squeeze_lookback=int(squeeze_lookback),

    momentum_window=int(momentum_window),

    weights=weights,
)

with st.spinner("Computing scores..."):

    scores_df = scorer.compute_scores(df_h)

    top_df = scorer.top_n_per_theme(
        scores_df,
        theme_map,
        top_n=int(top_n_per_theme),
    )

# ---------------------------------------------------------
# DISPLAY
# ---------------------------------------------------------

st.subheader("Top Coins per Theme")

if top_df.empty:

    st.warning(
        "No valid scores computed."
    )

else:

    display_cols = [

        "theme",
        "theme_rank",
        "coin",
        "composite_score",

        "squeeze_score",
        "expansion_score",
        "donchian_score",
        "momentum_score",

        "vol_expansion_ratio",

    ]

    display_cols = [
        c for c in display_cols
        if c in top_df.columns
    ]

    st.dataframe(

        top_df[display_cols]
        .style
        .format({

            "composite_score": "{:.3f}",
            "squeeze_score": "{:.3f}",
            "expansion_score": "{:.3f}",
            "donchian_score": "{:.3f}",
            "momentum_score": "{:.3f}",
            "vol_expansion_ratio": "{:.2f}x",

        })
        .background_gradient(
            subset=["composite_score"],
            cmap="YlOrRd"
        ),

        use_container_width=True,

    )

# ---------------------------------------------------------
# CHART
# ---------------------------------------------------------

if not top_df.empty:

    fig = go.Figure()

    for theme in top_df["theme"].unique():

        sub = (
            top_df[
                top_df["theme"] == theme
            ]
            .sort_values(
                "composite_score",
                ascending=False
            )
        )

        fig.add_trace(

            go.Bar(

                name=theme,

                x=sub["coin"],

                y=sub["composite_score"],

                text=[
                    f"{v:.3f}"
                    for v in sub["composite_score"]
                ],

                textposition="outside",

            )

        )

    fig.update_layout(

        barmode="group",

        xaxis_title="Coin",

        yaxis_title="Composite Score",

        yaxis_range=[0, 1.05],

        height=500,

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
