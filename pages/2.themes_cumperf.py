# pages/1_theme_cumperf.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import StringIO

st.set_page_config(page_title="Themes TS Cum. Perf", layout="wide")
st.title("Themes — Time Series Cumulative Performance")

# ---------------------------------------------------------------------
# Helper: compute medians per theme and cumulative returns.
# We include price_theme_version as a function input so st.cache_data will
# invalidate automatically when the source data version changes.
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_theme_cumulative(price_theme: pd.DataFrame, theme_values_aligned, price_theme_version):
    """
    price_theme: wide prices DataFrame (index = timestamps, columns = tickers)
    theme_values_aligned: list or Series aligned to price_theme.columns giving Theme for each ticker
    price_theme_version: numeric/string to allow cache invalidation when upstream data changes
    Returns:
      - cumulative_returns: DataFrame indexed by Theme (rows) with datetime-like columns (dates)
      - consolidated_medians: same shape but median returns (period returns, not cumulative)
      - theme_index: list of theme names (order = cumulative_returns.index)
    """
    # Defensive copies
    df_prices = price_theme.copy()
    # Ensure datetime index
    if not isinstance(df_prices.index, pd.DatetimeIndex):
        df_prices.index = pd.to_datetime(df_prices.index)

    # Compute percent returns (period over period) in percent units
    returns = df_prices.pct_change().dropna() * 100  # columns = tickers, index = dates

    if returns.empty:
        # Return empty placeholders
        return pd.DataFrame(), pd.DataFrame(), []

    # Transpose so rows = tickers, columns = dates
    returns_T = returns.T

    # Align theme_values_aligned to returns_T.index
    # Accept list or Series. If list, assume same order as price_theme.columns
    if isinstance(theme_values_aligned, (list, tuple, np.ndarray)):
        theme_series = pd.Series(list(theme_values_aligned), index=returns_T.index)
    elif isinstance(theme_values_aligned, pd.Series):
        # reindex to returns_T.index, filling unknowns
        theme_series = theme_values_aligned.reindex(returns_T.index).fillna("Unknown")
    else:
        # fallback: unknown theme for all
        theme_series = pd.Series(["Unknown"] * len(returns_T.index), index=returns_T.index)

    returns_T = returns_T.copy()
    returns_T["Theme"] = theme_series

    # date columns (strings or datetimes) excluding 'Theme'
    date_columns = [c for c in returns_T.columns if c != "Theme"]
    # Ensure date order is ascending by converting columns to datetimes if possible
    try:
        date_dt = pd.to_datetime(date_columns)
        sorted_pairs = sorted(zip(date_dt, date_columns))
        sorted_date_cols = [pair[1] for pair in sorted_pairs]
    except Exception:
        sorted_date_cols = date_columns

    # Build per-theme median series (one-row per theme)
    themes = returns_T["Theme"].unique().tolist()
    theme_medians = {}
    for theme in themes:
        rows = returns_T[returns_T["Theme"] == theme]
        # Compute median across tickers for each date column
        median_series = rows[sorted_date_cols].median(axis=0, skipna=True)
        df_theme = pd.DataFrame([median_series.values], index=[theme], columns=sorted_date_cols)
        theme_medians[theme] = df_theme

    consolidated_medians = pd.concat(theme_medians.values(), axis=0)
    # Ensure numeric
    consolidated_medians = consolidated_medians.apply(pd.to_numeric, errors="coerce")

    # Calculate cumulative returns base 100 from percent returns
    cumulative = pd.DataFrame(index=consolidated_medians.index, columns=consolidated_medians.columns)
    for theme in consolidated_medians.index:
        series = consolidated_medians.loc[theme].astype(float).fillna(0)  # treat NaN as 0% change
        cumulative.loc[theme] = (1 + series / 100).cumprod() * 100

    # Try to convert column labels to datetimes (useful for plotting)
    try:
        cumulative.columns = pd.to_datetime(cumulative.columns)
        consolidated_medians.columns = pd.to_datetime(consolidated_medians.columns)
    except Exception:
        pass

    return cumulative, consolidated_medians, consolidated_medians.index.tolist()

# ---------------------------------------------------------------------
# Page logic
# ---------------------------------------------------------------------
# Read session_state produced by app.py
price_theme = st.session_state.get("price_theme", None)
theme_values = st.session_state.get("theme_values", None)            # list aligned to columns
theme_list = st.session_state.get("theme_list", None)                # column tickers order
price_timeframe = st.session_state.get("price_timeframe", None)      # friendly label e.g. 'daily'
price_theme_version = st.session_state.get("price_theme_version", None)

# Info about upstream data version
if price_theme_version is not None:
    st.caption(f"Source data version: {price_theme_version}")

# manual refresh for local computations (does not trigger network fetch)
col1, col2 = st.columns([1, 4])
with col1:
    local_refresh = st.button("Refresh computations (no network fetch)")
with col2:
    st.write("Click the button to recompute charts from current `price_theme` in session state.")

# If nothing present, show guidance
if price_theme is None or (isinstance(price_theme, pd.DataFrame) and price_theme.empty):
    st.info("No price data found in session state. Please go to the data / Fetch page and click **Refresh / Fetch Data** to populate `price_theme` and theme mappings.")
    st.stop()

# Validate theme_values
if theme_values is None:
    st.warning("No `theme_values` found in session state. The median-by-theme calculation requires a mapping from ticker -> Theme. Please ensure your fetch page stores `st.session_state['theme_values']`.")
    st.stop()

# If user clicked local refresh, we want to re-run cached computation.
# We can pass price_theme_version (which changes on upstream refresh) and also the local_refresh boolean
# into the cached function to invalidate.
cache_token = (price_theme_version, bool(local_refresh))

# Compute cumulative and consolidated medians
cumulative_returns, consolidated_medians, theme_index = compute_theme_cumulative(price_theme, theme_values, cache_token)

if cumulative_returns.empty:
    st.info("No returns could be computed (not enough price bars). Fetch more historical bars and try again.")
    st.stop()

# Transpose for plotting convenience: index = date, columns = themes
cum_T = cumulative_returns.T.copy()
# Add Date column for exporting
cum_T_export = cum_T.reset_index().rename(columns={"index": "Date"})
# Convert Date to datetime if possible
try:
    cum_T_export["Date"] = pd.to_datetime(cum_T_export["Date"])
except Exception:
    pass

# UI: timeframe label
tf_label = price_timeframe if price_timeframe is not None else "unknown"
st.markdown(f"**Timeframe:** `{tf_label}` — cumulative series base 100 (median across theme members).")

# Theme selection
available_themes = list(cumulative_returns.index)
# Default to all themes unless many
default_sel = available_themes if len(available_themes) <= 8 else available_themes[:8]
selected = st.multiselect("Select themes to display", options=available_themes, default=default_sel)

if not selected:
    st.warning("Please select at least one theme to plot.")
    st.stop()

# Prepare plot DataFrame: melt cum_T (index=date)
plot_df = cum_T.reset_index().melt(id_vars="index", value_vars=selected, var_name="Theme", value_name="Cumulative")
plot_df = plot_df.rename(columns={"index": "Date"})
# convert Date
try:
    plot_df["Date"] = pd.to_datetime(plot_df["Date"])
except Exception:
    pass

# Plotly interactive chart
fig = px.line(plot_df, x="Date", y="Cumulative", color="Theme",
              title=f"Themes cumulative returns (base 100) — {tf_label}",
              labels={"Cumulative": "Cumulative (base 100)", "Date": "Date"})

fig.update_layout(
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=60, b=40, l=40, r=20)
)

# Add a rangeslider & basic range selector (7d/30d/90d/all)
# Works even for hourly data because it uses date axis.
fig.update_xaxes(
    rangeslider=dict(visible=True),
    rangeselector=dict(
        buttons=list([
            dict(count=7, label="7d", step="day", stepmode="backward"),
            dict(count=30, label="30d", step="day", stepmode="backward"),
            dict(count=90, label="90d", step="day", stepmode="backward"),
            dict(step="all")
        ])
    )
)

st.plotly_chart(fig, use_container_width=True)

# Show median returns (non-cumulative) preview in expander
with st.expander("Show median returns (per-period medians used to create cumulative series)"):
    # consolidated_medians: index = theme, columns = dates
    # Show last 30 columns for readability if many
    df_preview = consolidated_medians.copy()
    if df_preview.shape[1] > 30:
        st.info("Large number of date columns — showing the most recent 30 columns.")
        df_preview = df_preview.iloc[:, -30:]
    st.dataframe(df_preview.style.format(precision=3), height=300)

# Download CSV of cumulative returns (dates as rows)
csv_buf = cum_T_export.to_csv(index=False)
st.download_button("Download cumulative returns CSV", csv_buf, file_name="themes_cumulative_returns.csv", mime="text/csv")

# Also show price_theme (optional) and mapping
with st.expander("Show underlying price_theme (wide) and theme mapping"):
    st.markdown("**price_theme (wide)** — index = timestamps, columns = tickers")
    st.dataframe(price_theme)
    st.markdown("**Theme mapping (aligned to price_theme.columns)**")
    try:
        mapping_df = pd.DataFrame({
            "Ticker": list(price_theme.columns),
            "Theme": list(theme_values)
        })
        st.dataframe(mapping_df)
    except Exception:
        st.write("Could not display mapping cleanly (unexpected format).")

st.success("Chart ready — use selectors above to customize the view.")
