import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Trend of Trend", layout="wide")
st.title("Trend of Trend (Multi-Coin Ranking)")

# ---------------------------------------------------------
# REFRESH BUTTON
# ---------------------------------------------------------
if st.button("🔄 Refresh data"):
    st.rerun()

# ---------------------------------------------------------
# INPUT DATA FROM SESSION STATE
# ---------------------------------------------------------
df_short_h = st.session_state.get("price_theme", None)

if df_short_h is None:
    st.error(
        "`price_theme` not found in session_state. "
        "Please load prices into `st.session_state['price_theme']` first."
    )
    st.stop()

# Clean basic structure
df_short_h = df_short_h.sort_index().copy()

# ---------------------------------------------------------
# PARAMETER CONTROLS
# ---------------------------------------------------------
st.subheader("Strategy Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    top_n = st.number_input(
        "Top N coins (long/short)",
        min_value=1,
        max_value=100,
        value=15,
        step=1,
        help="Number of coins to long and short based on tot_score ranking.",
    )
    trend_span = st.number_input(
        "Trend span (EMA on log price)",
        min_value=5,
        max_value=300,
        value=55,
        step=5,
        help="Span for EMA of log prices to estimate primary trend.",
    )

with col2:
    slope_smooth = st.number_input(
        "Slope smoothing span",
        min_value=1,
        max_value=100,
        value=5,
        step=1,
        help="EWMA span for smoothing slope of the trend.",
    )
    accel_smooth = st.number_input(
        "Acceleration smoothing span",
        min_value=1,
        max_value=100,
        value=5,
        step=1,
        help="EWMA span for smoothing acceleration of the trend.",
    )

with col3:
    vol_lookback = st.number_input(
        "Volatility lookback (bars)",
        min_value=20,
        max_value=500,
        value=120,
        step=10,
        help="Rolling window used to estimate volatility for z-scoring.",
    )

# ---------------------------------------------------------
# CORE FUNCTIONS
# ---------------------------------------------------------
def trend_of_trend_signals(
    df_prices: pd.DataFrame,
    trend_span: int = 55,
    slope_smooth: int = 5,
    accel_smooth: int = 5,
    vol_lookback: int = 120,
):
    """
    Compute trend-of-trend components and continuous scores.
    """
    df = df_prices.copy().astype(float)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    df = df.fillna(method="ffill")

    lp = np.log(df)

    # Primary trend
    ema = lp.ewm(span=trend_span, min_periods=trend_span, adjust=False).mean()
    slope = ema.diff().ewm(span=slope_smooth, adjust=False).mean()
    accel = slope.diff().ewm(span=accel_smooth, adjust=False).mean()

    # Vol normalization
    logret = lp.diff()
    vol = logret.rolling(
        vol_lookback,
        min_periods=max(20, vol_lookback // 3),
    ).std()

    slope_z = slope / vol
    accel_z = accel / vol

    # Composite score (slope + accel)
    tot_score = 0.6 * slope_z + 0.4 * accel_z

    return {
        "slope_z": slope_z,
        "accel_z": accel_z,
        "tot_score": tot_score,
    }


def last_bar_entries_topN(signal_dict, top_n: int = 15):
    """
    Select top N and bottom N coins by tot_score at the last bar.

    Parameters:
      top_n : number of coins to long/short

    Returns:
      timestamp, dict with 'long_entry' and 'short_entry' coin lists
    """
    tot_score = signal_dict["tot_score"]
    last_time = tot_score.index[-1]
    last_scores = tot_score.loc[last_time].dropna()

    if last_scores.empty:
        return last_time, {"long_entry": [], "short_entry": []}

    # Rank by score
    sorted_scores = last_scores.sort_values(ascending=False)

    long_entry = sorted_scores.head(top_n).index.tolist()
    short_entry = sorted_scores.tail(top_n).index.tolist()

    return last_time, {"long_entry": long_entry, "short_entry": short_entry}


# ---------------------------------------------------------
# RUN STRATEGY
# ---------------------------------------------------------
signals = trend_of_trend_signals(
    df_short_h,
    trend_span=int(trend_span),
    slope_smooth=int(slope_smooth),
    accel_smooth=int(accel_smooth),
    vol_lookback=int(vol_lookback),
)

ts, entries = last_bar_entries_topN(signals, top_n=int(top_n))
long_entry = entries["long_entry"]
short_entry = entries["short_entry"]

# ---------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------
st.subheader(f"Entries at last bar: {ts}")

col_long, col_short = st.columns(2)

with col_long:
    st.markdown("### Long Entry")
    if long_entry:
        st.write(long_entry)
    else:
        st.write("No long entries for the latest bar.")

with col_short:
    st.markdown("### Short Entry")
    if short_entry:
        st.write(short_entry)
    else:
        st.write("No short entries for the latest bar.")

# Optional: show full tot_score snapshot at last bar
with st.expander("Show full tot_score at last bar"):
    last_scores = signals["tot_score"].loc[ts].dropna().sort_values(ascending=False)
    st.dataframe(last_scores.to_frame("tot_score").T, use_container_width=True)
