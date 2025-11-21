# pages/1_theme_cumperf.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Themes TS Cum. Perf", layout="wide")
st.title("Themes — Time Series Cumulative Performance (Period-based view)")

# ---------------------------------------------------------------------
# Cached compute: produce cumulative (base-100) series per theme.
# cache_token ensures re-run when upstream data version or local refresh toggles.
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_theme_cumulative(price_theme: pd.DataFrame, theme_values_aligned, cache_token):
    """
    price_theme: wide prices DataFrame (index = timestamps, columns = tickers)
    theme_values_aligned: list/Series aligned to price_theme.columns giving Theme for each ticker
    cache_token: any hashable token to invalidate cache on upstream changes
    Returns:
      - cumulative: DataFrame indexed by theme, columns are timestamps (ascending)
      - medians: DataFrame (per-period median returns in percent), same shape
    """
    df_prices = price_theme.copy()
    if not isinstance(df_prices.index, pd.DatetimeIndex):
        df_prices.index = pd.to_datetime(df_prices.index)

    # percent returns (period over period) in percent units
    returns = df_prices.pct_change().dropna() * 100
    if returns.empty:
        return pd.DataFrame(), pd.DataFrame()

    returns_T = returns.T

    # Align theme_values_aligned to returns_T.index
    if isinstance(theme_values_aligned, (list, tuple, np.ndarray)):
        theme_series = pd.Series(list(theme_values_aligned), index=returns_T.index)
    elif isinstance(theme_values_aligned, pd.Series):
        theme_series = theme_values_aligned.reindex(returns_T.index).fillna("Unknown")
    else:
        theme_series = pd.Series(["Unknown"] * len(returns_T.index), index=returns_T.index)

    returns_T = returns_T.copy()
    returns_T["Theme"] = theme_series

    date_columns = [c for c in returns_T.columns if c != "Theme"]
    # sort date columns in ascending order (keep original labels)
    try:
        date_dt = pd.to_datetime(date_columns)
        sorted_pairs = sorted(zip(date_dt, date_columns))
        sorted_date_cols = [pair[1] for pair in sorted_pairs]
    except Exception:
        sorted_date_cols = date_columns

    # Build per-theme median series
    themes = returns_T["Theme"].unique().tolist()
    theme_medians = {}
    for theme in themes:
        rows = returns_T[returns_T["Theme"] == theme]
        median_series = rows[sorted_date_cols].median(axis=0, skipna=True)
        df_theme = pd.DataFrame([median_series.values], index=[theme], columns=sorted_date_cols)
        theme_medians[theme] = df_theme

    consolidated_medians = pd.concat(theme_medians.values(), axis=0)
    consolidated_medians = consolidated_medians.apply(pd.to_numeric, errors="coerce")

    # cumulative returns base 100
    cumulative = pd.DataFrame(index=consolidated_medians.index, columns=consolidated_medians.columns)
    for theme in consolidated_medians.index:
        series = consolidated_medians.loc[theme].astype(float).fillna(0)
        cumulative.loc[theme] = (1 + series / 100).cumprod() * 100

    # try convert column labels to datetimes for plotting convenience
    try:
        cumulative.columns = pd.to_datetime(cumulative.columns)
        consolidated_medians.columns = pd.to_datetime(consolidated_medians.columns)
    except Exception:
        pass

    return cumulative, consolidated_medians

# ---------------------------------------------------------------------
# Page logic
# ---------------------------------------------------------------------
price_theme = st.session_state.get("price_theme", None)
theme_values = st.session_state.get("theme_values", None)
price_timeframe = st.session_state.get("price_timeframe", None)  # 'daily', '4hourly', '1hourly'
price_theme_version = st.session_state.get("price_theme_version", None)

if price_theme_version is not None:
    st.caption(f"Source data version: {price_theme_version}")

col1, col2 = st.columns([1, 4])
with col1:
    local_refresh = st.button("Refresh computations (no network fetch)")
with col2:
    st.write("Click to recompute charts from current `price_theme` in session state. Use the selector below to pick the number of periods to view (period = data timeframe).")

if price_theme is None or (isinstance(price_theme, pd.DataFrame) and price_theme.empty):
    st.info("No price data in session state. Go to the fetch page and click 'Refresh / Fetch Data' to populate price data.")
    st.stop()

if theme_values is None:
    st.warning("No `theme_values` found in session state. Make sure the fetch page stores the theme mapping.")
    st.stop()

# Describe what a 'period' means (based on price_timeframe)
tf_label = price_timeframe if price_timeframe is not None else "unknown"
if tf_label == "daily":
    period_desc = "1 period = 1 day"
elif tf_label == "4hourly":
    period_desc = "1 period = 4 hours"
elif tf_label == "1hourly":
    period_desc = "1 period = 1 hour"
else:
    period_desc = f"1 period = data timeframe ({tf_label})"

st.markdown(f"**Timeframe:** `{tf_label}` — {period_desc}")

# compute (cached) cumulative series
cache_token = (price_theme_version, bool(local_refresh))
cumulative_returns, consolidated_medians = compute_theme_cumulative(price_theme, theme_values, cache_token)

if cumulative_returns.empty:
    st.info("Not enough data to compute returns. Fetch more bars and retry.")
    st.stop()

# transpose for plotting: index = dates (ascending), columns = themes
cum_T_full = cumulative_returns.T.copy()
# ensure datetime index if possible
try:
    cum_T_full.index = pd.to_datetime(cum_T_full.index)
except Exception:
    pass

# Build dynamic period-based view options
period_options = ["1 period", "30 periods", "60 periods", "180 periods", "All"]
view_option = st.selectbox("Display window (periods): choose # of most recent periods to show and rebase to 100",
                           options=period_options, index=1)

# slice by number of periods (rows) and rebase
def slice_and_rebase_by_periods(cum_df: pd.DataFrame, view: str):
    """
    cum_df: DataFrame indexed by datetime (rows) and columns = themes
    view: 'N period(s)' or 'All'
    Returns: rebased_df (index=date ascending), used_start_idx (int)
    """
    if view == "All":
        return cum_df.copy(), 0

    try:
        n = int(view.split()[0])
    except Exception:
        n = None

    if n is None or n <= 0:
        return cum_df.copy(), 0

    total_rows = len(cum_df)
    # if fewer rows than requested, fallback but warn upstream
    if total_rows == 0:
        return cum_df.copy(), 0
    if total_rows <= n:
        # not enough rows: return full df (no strict slicing)
        return cum_df.copy(), max(0, total_rows - n)

    # take last n rows
    df_slice = cum_df.tail(n).copy()
    # rebase to 100 at first row of slice
    first_vals = df_slice.iloc[0]
    # treat zeros/NaN by temporarily setting to NaN to avoid div by 0
    safe_first = first_vals.replace({0: np.nan})
    rebased = df_slice.divide(safe_first, axis=1) * 100
    # for columns where rebasing failed (NaN or inf), fallback to original slice
    for col in rebased.columns:
        if not np.isfinite(rebased[col].iloc[0]):
            rebased[col] = df_slice[col]
    return rebased, total_rows - n

view_df, start_idx_hint = slice_and_rebase_by_periods(cum_T_full, view_option)

# Inform if requested more periods than available
if view_option != "All":
    requested = int(view_option.split()[0])
    available = len(cum_T_full)
    if available < requested:
        st.warning(f"Requested {requested} periods but only {available} periods available — showing full history (no strict slice).")

# Prepare plot_df for Plotly: reset_index -> melt
plot_df = view_df.reset_index().melt(id_vars=view_df.index.name or "index", value_vars=view_df.columns,
                                    var_name="Theme", value_name="Cumulative")
# Normalize date column name
date_col_name = view_df.index.name or "Date"
if (view_df.index.name is None) or (view_df.index.name == "index"):
    plot_df = plot_df.rename(columns={"index": "Date"})
else:
    plot_df = plot_df.rename(columns={view_df.index.name: "Date"})

# convert Date to datetime if possible
try:
    plot_df["Date"] = pd.to_datetime(plot_df["Date"])
except Exception:
    pass

# Plotly interactive chart
fig = px.line(plot_df, x="Date", y="Cumulative", color="Theme",
              title=f"Themes cumulative returns (rebased to 100 over selected periods) — {tf_label}",
              labels={"Cumulative": "Cumulative (base 100)", "Date": "Date"})

fig.update_layout(
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=60, b=40, l=40, r=20)
)

# rangeslider and selector: still helpful for navigation, but note rebasing is server-side
fig.update_xaxes(
    rangeslider=dict(visible=True),
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1p", step="day", stepmode="backward"),   # purely navigational labels
            dict(count=30, label="30p", step="day", stepmode="backward"),
            dict(count=60, label="60p", step="day", stepmode="backward"),
            dict(count=180, label="180p", step="day", stepmode="backward"),
            dict(step="all")
        ])
    )
)

st.plotly_chart(fig, use_container_width=True)

# Show median returns (non-cumulative) preview
with st.expander("Show per-period median returns used to compute cumulative series"):
    df_preview = consolidated_medians.copy()
    if df_preview.shape[1] > 30:
        st.info("Large number of date columns — showing the most recent 30 columns.")
        df_preview = df_preview.iloc[:, -30:]
    st.dataframe(df_preview.style.format(precision=3), height=300)

# CSV download of displayed (sliced & rebased) cumulative returns
export_df = view_df.reset_index().rename(columns={view_df.index.name or "index": "Date"})
try:
    export_df["Date"] = pd.to_datetime(export_df["Date"])
except Exception:
    pass
csv_buf = export_df.to_csv(index=False)
st.download_button("Download displayed cumulative returns CSV", csv_buf, file_name="themes_cumulative_returns_view.csv", mime="text/csv")

# Show underlying price_theme and mapping
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

st.success("Chart ready — use the selectors above to change the period window and rebase.")
