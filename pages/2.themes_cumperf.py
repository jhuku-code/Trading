# pages/1_theme_cumperf.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Themes TS Cum. Perf", layout="wide")
st.title("Themes — Time Series Cumulative Performance (Period-based view)")

# -------------------------
# Cached computation
# -------------------------
@st.cache_data(show_spinner=False)
def compute_theme_cumulative(price_theme: pd.DataFrame, theme_values_aligned, cache_token):
    df_prices = price_theme.copy()
    if not isinstance(df_prices.index, pd.DatetimeIndex):
        df_prices.index = pd.to_datetime(df_prices.index)

    returns = df_prices.pct_change().dropna() * 100
    if returns.empty:
        return pd.DataFrame(), pd.DataFrame()

    returns_T = returns.T

    if isinstance(theme_values_aligned, (list, tuple, np.ndarray)):
        theme_series = pd.Series(list(theme_values_aligned), index=returns_T.index)
    elif isinstance(theme_values_aligned, pd.Series):
        theme_series = theme_values_aligned.reindex(returns_T.index).fillna("Unknown")
    else:
        theme_series = pd.Series(["Unknown"] * len(returns_T.index), index=returns_T.index)

    returns_T = returns_T.copy()
    returns_T["Theme"] = theme_series

    date_columns = [c for c in returns_T.columns if c != "Theme"]
    try:
        date_dt = pd.to_datetime(date_columns)
        sorted_pairs = sorted(zip(date_dt, date_columns))
        sorted_date_cols = [pair[1] for pair in sorted_pairs]
    except Exception:
        sorted_date_cols = date_columns

    themes = returns_T["Theme"].unique().tolist()
    theme_medians = {}
    for theme in themes:
        rows = returns_T[returns_T["Theme"] == theme]
        median_series = rows[sorted_date_cols].median(axis=0, skipna=True)
        df_theme = pd.DataFrame([median_series.values], index=[theme], columns=sorted_date_cols)
        theme_medians[theme] = df_theme

    consolidated_medians = pd.concat(theme_medians.values(), axis=0)
    consolidated_medians = consolidated_medians.apply(pd.to_numeric, errors="coerce")

    cumulative = pd.DataFrame(index=consolidated_medians.index, columns=consolidated_medians.columns)
    for theme in consolidated_medians.index:
        series = consolidated_medians.loc[theme].astype(float).fillna(0)
        cumulative.loc[theme] = (1 + series / 100).cumprod() * 100

    try:
        cumulative.columns = pd.to_datetime(cumulative.columns)
        consolidated_medians.columns = pd.to_datetime(consolidated_medians.columns)
    except Exception:
        pass

    return cumulative, consolidated_medians


# -------------------------
# Inputs
# -------------------------
price_theme = st.session_state.get("price_theme", None)
theme_values = st.session_state.get("theme_values", None)
price_timeframe = st.session_state.get("price_timeframe", None)
price_theme_version = st.session_state.get("price_theme_version", None)

if price_theme_version is not None:
    st.caption(f"Source data version: {price_theme_version}")

col_left, col_right = st.columns([1, 4])
with col_left:
    local_refresh = st.button("Refresh computations (no network fetch)")
with col_right:
    st.write("Use the Period controls below to select the displayed window.")

if price_theme is None or (isinstance(price_theme, pd.DataFrame) and price_theme.empty):
    st.stop()

if theme_values is None:
    st.stop()

tf_label = price_timeframe if price_timeframe else "unknown"
st.markdown(f"**Timeframe:** `{tf_label}`")

# -------------------------
# Compute
# -------------------------
cache_token = (price_theme_version, bool(local_refresh))
cumulative_returns, consolidated_medians = compute_theme_cumulative(price_theme, theme_values, cache_token)

if cumulative_returns.empty:
    st.stop()

cum_T_full = cumulative_returns.T.copy()
cum_T_full.index = pd.to_datetime(cum_T_full.index)

# -------------------------
# Period controls (UPDATED)
# -------------------------
period_options = ["1 period", "10 periods", "30 periods", "60 periods", "90 periods", "180 periods", "All"]

if "view_option" not in st.session_state:
    st.session_state.view_option = "30 periods"

btn_cols = st.columns([1, 1, 1, 1, 1, 1])
labels = ["1 period", "10 periods", "30 periods", "60 periods", "90 periods", "180 periods", "All"]

for c, label in zip(btn_cols, labels):
    if c.button(label.split()[0] + ("p" if "period" in label else ""), key=f"btn_{label}"):
        st.session_state.view_option = label
        st.experimental_rerun()

view_option = st.selectbox(
    "Display window",
    options=period_options,
    index=period_options.index(st.session_state.view_option),
)
st.session_state.view_option = view_option

# -------------------------
# Slice + rebase
# -------------------------
def slice_and_rebase_by_periods(cum_df, view):
    if view == "All":
        return cum_df.copy(), 0, len(cum_df)

    n = int(view.split()[0])
    if len(cum_df) <= n:
        return cum_df.copy(), 0, len(cum_df)

    df_slice = cum_df.tail(n).copy()
    rebased = df_slice.divide(df_slice.iloc[0]).mul(100)
    return rebased, 0, n

view_df, _, _ = slice_and_rebase_by_periods(cum_T_full, view_option)

# -------------------------
# Theme plot
# -------------------------
plot_df = view_df.reset_index().melt(id_vars="index", var_name="Theme", value_name="Cumulative")
plot_df.rename(columns={"index": "Date"}, inplace=True)

fig = px.line(plot_df, x="Date", y="Cumulative", color="Theme")
fig.update_layout(hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Coin-level chart (UPDATED)
# -------------------------
st.markdown("---")
st.subheader("Per-theme — coin-level cumulative performance")

theme_series_full = pd.Series(theme_values, index=price_theme.columns)
available_themes = sorted(theme_series_full.unique())

selected_theme = st.selectbox("Select theme", available_themes)

tickers_for_theme = [t for t, th in zip(price_theme.columns, theme_values) if th == selected_theme]

selected_tickers = st.multiselect("Tickers", tickers_for_theme, default=tickers_for_theme)

if selected_tickers:
    price_sub = price_theme[selected_tickers]
    price_sub.index = pd.to_datetime(price_sub.index)

    coin_returns = price_sub.pct_change().dropna()
    coin_cum = (1 + coin_returns).cumprod() * 100

    coin_view_df, _, _ = slice_and_rebase_by_periods(coin_cum, view_option)

    # ✅ ADD AVERAGE LINE
    coin_view_df["_average"] = coin_view_df.mean(axis=1)

    coin_plot_df = coin_view_df.reset_index().melt(id_vars="index", var_name="Ticker", value_name="Cumulative")
    coin_plot_df.rename(columns={"index": "Date"}, inplace=True)

    fig2 = px.line(coin_plot_df, x="Date", y="Cumulative", color="Ticker")

    # ✅ IMPROVED LEGEND FILTERING
    fig2.update_layout(
        hovermode="x unified",
        legend=dict(
            itemclick="toggle",
            itemdoubleclick="toggleothers"
        )
    )

    st.plotly_chart(fig2, use_container_width=True)

st.success("Done")
