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
    """
    Returns:
      cumulative: DataFrame indexed by theme, columns = timestamps (ascending)
      medians: DataFrame indexed by theme, columns = timestamps (period medians in percent)
    cache_token is used to invalidate when upstream changes.
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

    # Determine date columns (exclude Theme) and keep them in ascending order
    date_columns = [c for c in returns_T.columns if c != "Theme"]
    try:
        date_dt = pd.to_datetime(date_columns)
        sorted_pairs = sorted(zip(date_dt, date_columns))
        sorted_date_cols = [pair[1] for pair in sorted_pairs]
    except Exception:
        sorted_date_cols = date_columns

    # Build median series per theme (one-row per theme)
    themes = returns_T["Theme"].unique().tolist()
    theme_medians = {}
    for theme in themes:
        rows = returns_T[returns_T["Theme"] == theme]
        median_series = rows[sorted_date_cols].median(axis=0, skipna=True)
        df_theme = pd.DataFrame([median_series.values], index=[theme], columns=sorted_date_cols)
        theme_medians[theme] = df_theme

    consolidated_medians = pd.concat(theme_medians.values(), axis=0)
    consolidated_medians = consolidated_medians.apply(pd.to_numeric, errors="coerce")

    # Cumulative returns from percent medians (base 100)
    cumulative = pd.DataFrame(index=consolidated_medians.index, columns=consolidated_medians.columns)
    for theme in consolidated_medians.index:
        series = consolidated_medians.loc[theme].astype(float).fillna(0)  # treat NaN as 0%
        cumulative.loc[theme] = (1 + series / 100).cumprod() * 100

    # Attempt to convert columns to datetimes for plotting
    try:
        cumulative.columns = pd.to_datetime(cumulative.columns)
        consolidated_medians.columns = pd.to_datetime(consolidated_medians.columns)
    except Exception:
        pass

    return cumulative, consolidated_medians

# -------------------------
# Page inputs & state
# -------------------------
price_theme = st.session_state.get("price_theme", None)            # wide prices DF
theme_values = st.session_state.get("theme_values", None)          # list aligned to columns
price_timeframe = st.session_state.get("price_timeframe", None)    # 'daily' / '4hourly' / '1hourly'
price_theme_version = st.session_state.get("price_theme_version", None)

# Show source version if present
if price_theme_version is not None:
    st.caption(f"Source data version: {price_theme_version}")

# local controls
col_left, col_right = st.columns([1, 4])
with col_left:
    local_refresh = st.button("Refresh computations (no network fetch)")
with col_right:
    st.write("Use the Period controls below (Streamlit buttons/selectbox) to select the displayed window. The selected window is rebased to 100.")

# Validate presence of data
if price_theme is None or (isinstance(price_theme, pd.DataFrame) and price_theme.empty):
    st.info("No `price_theme` found in session state. Go to the fetch page and click 'Refresh / Fetch Data' to populate data.")
    st.stop()

if theme_values is None:
    st.warning("No `theme_values` mapping found in session state. Please ensure the fetch page stores it.")
    st.stop()

# Describe period meaning based on timeframe
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

# -------------------------
# Compute cumulative + medians (cached)
# -------------------------
cache_token = (price_theme_version, bool(local_refresh))
cumulative_returns, consolidated_medians = compute_theme_cumulative(price_theme, theme_values, cache_token)

if cumulative_returns.empty:
    st.info("Not enough data to compute returns. Fetch more bars and try again.")
    st.stop()

# transpose for plotting: rows = dates (ascending), columns = themes
cum_T_full = cumulative_returns.T.copy()
# ensure index is datetime if possible
try:
    cum_T_full.index = pd.to_datetime(cum_T_full.index)
except Exception:
    pass

# -------------------------
# Streamlit-based view controls (only these control the view/range)
# -------------------------
# Available options (period counts)
period_options = ["1 period", "30 periods", "60 periods", "180 periods", "All"]

# Use session_state to keep selection persistent and allow quick-button updates
if "view_option" not in st.session_state:
    st.session_state.view_option = "30 periods"

# Quick buttons row: 1p, 30p, 60p, 180p, All
btn_cols = st.columns([1, 1, 1, 1, 1])
labels = ["1 period", "30 periods", "60 periods", "180 periods", "All"]
for c, label in zip(btn_cols, labels):
    if c.button(label.split()[0] + ("p" if "period" in label else ""), key=f"btn_{label}"):
        st.session_state.view_option = label
        # re-run so the selection takes effect immediately
        st.experimental_rerun()

# Primary selectbox (keeps selection and acts as canonical control)
view_option = st.selectbox("Display window (this rebases selected window to start at 100)", options=period_options, index=period_options.index(st.session_state.view_option) if st.session_state.view_option in period_options else 1, key="select_view_option")
# Keep session state aligned
st.session_state.view_option = view_option

# -------------------------
# Slice by number of periods and rebase (server-side)
# -------------------------
def slice_and_rebase_by_periods(cum_df: pd.DataFrame, view: str):
    """
    cum_df: DataFrame indexed by datetime (ascending), columns = themes
    view: 'N period(s)' or 'All'
    Returns: rebased_df (index datetime asc), used_start_idx (int), used_count (int)
    """
    if view == "All":
        return cum_df.copy(), 0, len(cum_df)

    try:
        n = int(view.split()[0])
    except Exception:
        n = None

    if n is None or n <= 0:
        return cum_df.copy(), 0, len(cum_df)

    total_rows = len(cum_df)
    if total_rows == 0:
        return cum_df.copy(), 0, 0

    if total_rows <= n:
        # not enough rows: return full df (no strict slice)
        return cum_df.copy(), 0, total_rows

    df_slice = cum_df.tail(n).copy()
    # Rebase: divide by first row value and multiply by 100
    first_vals = df_slice.iloc[0]
    safe_first = first_vals.replace({0: np.nan})
    rebased = df_slice.divide(safe_first, axis=1) * 100
    # fallback for invalid columns
    for col in rebased.columns:
        if not np.isfinite(rebased[col].iloc[0]):
            rebased[col] = df_slice[col]
    return rebased, total_rows - n, n

view_df, start_idx, used_count = slice_and_rebase_by_periods(cum_T_full, view_option)

# Inform user if requested more periods than available
if view_option != "All":
    requested = int(view_option.split()[0])
    available = len(cum_T_full)
    if available < requested:
        st.warning(f"Requested {requested} periods but only {available} periods available — showing full history (no strict slice).")

# -------------------------
# Plotting (Plotly) — no builtin range selector or rangeslider
# -------------------------
plot_df = view_df.reset_index().melt(id_vars=view_df.index.name or "index", value_vars=view_df.columns, var_name="Theme", value_name="Cumulative")
if (view_df.index.name is None) or (view_df.index.name == "index"):
    plot_df = plot_df.rename(columns={"index": "Date"})
else:
    plot_df = plot_df.rename(columns={view_df.index.name: "Date"})
try:
    plot_df["Date"] = pd.to_datetime(plot_df["Date"])
except Exception:
    pass

fig = px.line(plot_df, x="Date", y="Cumulative", color="Theme",
              title=f"Themes cumulative returns (rebased to 100 over selected periods) — {tf_label}",
              labels={"Cumulative": "Cumulative (base 100)", "Date": "Date"})

fig.update_layout(
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=60, b=40, l=40, r=20)
)

# display Plotly chart (no plotly-built range selector)
st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Extras: medians preview, CSV download, mapping preview
# -------------------------
with st.expander("Show per-period median returns used to compute cumulative series"):
    df_preview = consolidated_medians.copy()
    if df_preview.shape[1] > 30:
        st.info("Large number of date columns — showing the most recent 30 columns.")
        df_preview = df_preview.iloc[:, -30:]
    st.dataframe(df_preview.style.format(precision=3), height=300)

# CSV download of the displayed (sliced & rebased) cumulative returns
export_df = view_df.reset_index().rename(columns={view_df.index.name or "index": "Date"})
try:
    export_df["Date"] = pd.to_datetime(export_df["Date"])
except Exception:
    pass
csv_buf = export_df.to_csv(index=False)
st.download_button("Download displayed cumulative returns CSV", csv_buf, file_name="themes_cumulative_returns_view.csv", mime="text/csv")

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

st.success("Chart ready — use Streamlit controls above to change the view window (periods) and rebase.")
