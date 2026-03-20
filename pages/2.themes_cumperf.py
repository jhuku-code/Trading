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
        df_prices.index = pd.to_datetime(df_prices.index, errors="coerce")

    returns = df_prices.pct_change().dropna() * 100
    if returns.empty:
        return pd.DataFrame(), pd.DataFrame()

    returns_T = returns.T

    # Align theme mapping
    try:
        theme_series = pd.Series(list(theme_values_aligned), index=returns_T.index)
    except Exception:
        theme_series = pd.Series(["Unknown"] * len(returns_T.index), index=returns_T.index)

    returns_T["Theme"] = theme_series

    # Sort dates
    date_cols = [c for c in returns_T.columns if c != "Theme"]
    try:
        date_cols_sorted = sorted(date_cols, key=lambda x: pd.to_datetime(x))
    except Exception:
        date_cols_sorted = date_cols

    # Median per theme
    theme_medians = {}
    for theme in returns_T["Theme"].unique():
        rows = returns_T[returns_T["Theme"] == theme]
        med = rows[date_cols_sorted].median(axis=0)
        theme_medians[theme] = pd.DataFrame([med.values], index=[theme], columns=date_cols_sorted)

    consolidated = pd.concat(theme_medians.values(), axis=0)
    consolidated = consolidated.apply(pd.to_numeric, errors="coerce")

    # Cumulative
    cumulative = (1 + consolidated.fillna(0) / 100).cumprod(axis=1) * 100

    try:
        cumulative.columns = pd.to_datetime(cumulative.columns)
    except Exception:
        pass

    return cumulative, consolidated


# -------------------------
# Inputs
# -------------------------
price_theme = st.session_state.get("price_theme")
theme_values = st.session_state.get("theme_values")
price_timeframe = st.session_state.get("price_timeframe")
price_theme_version = st.session_state.get("price_theme_version")

if price_theme is None or price_theme.empty:
    st.warning("No price data found.")
    st.stop()

if theme_values is None:
    st.warning("No theme mapping found.")
    st.stop()

# -------------------------
# Compute
# -------------------------
cumulative_returns, consolidated_medians = compute_theme_cumulative(
    price_theme, theme_values, price_theme_version
)

if cumulative_returns.empty:
    st.warning("Not enough data.")
    st.stop()

cum_T_full = cumulative_returns.T.copy()

# Ensure datetime index
cum_T_full.index = pd.to_datetime(cum_T_full.index, errors="coerce")

# -------------------------
# View controls (FIXED)
# -------------------------
period_options = [
    "1 period",
    "10 periods",
    "30 periods",
    "60 periods",
    "90 periods",
    "180 periods",
    "All"
]

if "view_option" not in st.session_state:
    st.session_state.view_option = "30 periods"

# Safe index selection
default_index = period_options.index(st.session_state.view_option) \
    if st.session_state.view_option in period_options else 2

view_option = st.selectbox(
    "Display window",
    options=period_options,
    index=default_index
)

st.session_state.view_option = view_option


# -------------------------
# Slice function (SAFE)
# -------------------------
def slice_and_rebase(df, view):

    if view == "All":
        return df.copy()

    try:
        n = int(view.split()[0])
    except Exception:
        return df.copy()

    if len(df) <= n:
        return df.copy()

    df_slice = df.tail(n).copy()

    base = df_slice.iloc[0].replace(0, np.nan)
    rebased = df_slice.divide(base) * 100

    return rebased


view_df = slice_and_rebase(cum_T_full, view_option)

# -------------------------
# Theme plot (FIXED melt)
# -------------------------
plot_df = view_df.copy()
plot_df["Date"] = plot_df.index

plot_df = plot_df.melt(id_vars="Date", var_name="Theme", value_name="Cumulative")

fig = px.line(plot_df, x="Date", y="Cumulative", color="Theme")

fig.update_layout(hovermode="x unified")

st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Coin-level section
# -------------------------
st.markdown("---")
st.subheader("Per-theme — coin-level cumulative performance")

theme_series_full = pd.Series(theme_values, index=price_theme.columns)

available_themes = sorted(theme_series_full.dropna().astype(str).unique())

selected_theme = st.selectbox("Select theme", available_themes)

tickers = [
    t for t, th in zip(price_theme.columns, theme_values)
    if str(th) == str(selected_theme)
]

selected_tickers = st.multiselect(
    "Select tickers",
    options=tickers,
    default=tickers
)

if selected_tickers:

    price_sub = price_theme[selected_tickers].copy()
    price_sub.index = pd.to_datetime(price_sub.index, errors="coerce")

    returns = price_sub.pct_change().dropna()

    if not returns.empty:

        coin_cum = (1 + returns).cumprod() * 100

        # ✅ ADD AVERAGE
        coin_cum["_average"] = coin_cum.mean(axis=1)

        coin_view = slice_and_rebase(coin_cum, view_option)

        coin_plot = coin_view.copy()
        coin_plot["Date"] = coin_plot.index

        coin_plot = coin_plot.melt(id_vars="Date", var_name="Ticker", value_name="Cumulative")

        fig2 = px.line(coin_plot, x="Date", y="Cumulative", color="Ticker")

        # ✅ STYLE AVERAGE
        for trace in fig2.data:
            if trace.name == "_average":
                trace.update(line=dict(width=3, dash="dash"))

        # ✅ LEGEND FILTERING
        fig2.update_layout(
            hovermode="x unified",
            legend=dict(
                itemclick="toggle",
                itemdoubleclick="toggleothers"
            )
        )

        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.warning("Not enough data for selected tickers.")
```
