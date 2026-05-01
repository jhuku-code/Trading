import time
import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from dateutil.relativedelta import relativedelta

# ---------------------------------------------------------
# STREAMLIT PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Liquidations Z-Score / Share Monitor (Enhanced)", layout="wide")
st.title("Liquidations Monitor (Z-Score + Share vs Total) — Enhanced")

# ---------------------------------------------------------
# INPUTS / CONTROLS
# ---------------------------------------------------------
top_n = st.number_input(
    "Number of symbols to show in Top/Bottom lists",
    min_value=5,
    max_value=100,
    value=15,
    step=5,
)

lookback_months = st.number_input(
    "Lookback window (months)",
    min_value=1,
    max_value=12,
    value=3,
    step=1,
)

interval = st.selectbox(
    "Interval / Granularity",
    options=[
        "1min",
        "5min",
        "15min",
        "30min",
        "1hour",
        "2hour",
        "4hour",
        "6hour",
        "12hour",
        "daily",
    ],
    index=6,  # default "4hour"
)

convert_usd = st.checkbox("Convert liquidation values to USD", value=True)
units_label = "USD" if convert_usd else "native units"

status_placeholder = st.empty()

# ---------------------------------------------------------
# CONFIG: API KEY & BASE URL
# (we don't call the API until user presses Refresh)
# ---------------------------------------------------------
API_KEY = st.secrets.get("API_KEY")
BASE_URL = "https://api.coinalyze.net/v1"

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


@st.cache_data(ttl=3600)
def load_perps_list_from_github(url: str) -> pd.DataFrame:
    """Load perps list (cached)."""
    return pd.read_excel(url)


@st.cache_data(ttl=1800)
def fetch_liquidations(symbols, months_back: int, interval: str, api_key: str, convert_usd_flag: bool):
    """
    Fetch liquidation history from Coinalyze.
    Returns two DataFrames (liquidations_long_wide, liquidations_short_wide)
    indexed by timestamp, columns are symbols.
    """
    liq_url = f"{BASE_URL}/liquidation-history"

    from_timestamp = int((datetime.now() - relativedelta(months=months_back)).timestamp())
    to_timestamp = int(datetime.now().timestamp())

    all_rows = []

    for batch in chunked(symbols, 20):
        symbols_param = ",".join(batch)
        params = {
            "symbols": symbols_param,
            "interval": interval,
            "from": from_timestamp,
            "to": to_timestamp,
            "convert_to_usd": "true" if convert_usd_flag else "false",
            "api_key": api_key,
        }

        while True:
            resp = requests.get(liq_url, params=params)
            if resp.status_code == 200:
                data = resp.json()
                if not isinstance(data, list):
                    # unexpected format
                    st.warning("Unexpected response format for a batch; skipping.")
                    break

                for entry in data:
                    symbol = entry.get("symbol")
                    history = entry.get("history", [])
                    if not symbol or not history:
                        continue

                    df = pd.DataFrame(history)
                    df["time"] = pd.to_datetime(df["t"], unit="s")
                    df.rename(columns={"l": "liq_long", "s": "liq_short"}, inplace=True)
                    df = df[["time", "liq_long", "liq_short"]]
                    df["symbol"] = symbol
                    all_rows.append(df)

                time.sleep(1)  # polite pause
                break

            elif resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                try:
                    retry_after = int(retry_after) if retry_after is not None else 30
                except ValueError:
                    retry_after = 30
                st.warning(f"429 Too Many Requests for batch starting with {batch[0]} — sleeping {retry_after}s")
                time.sleep(retry_after)

            else:
                st.error(f"API error for batch starting {batch[0]}: {resp.status_code} - {resp.text}")
                break

    if not all_rows:
        raise Exception("No data received for any symbol.")

    df_all = pd.concat(all_rows, ignore_index=True)

    liq_long_wide = df_all.pivot_table(index="time", columns="symbol", values="liq_long").sort_index()
    liq_short_wide = df_all.pivot_table(index="time", columns="symbol", values="liq_short").sort_index()

    # ensure numeric and sorted index
    liq_long_wide = liq_long_wide.apply(pd.to_numeric, errors="coerce").sort_index()
    liq_short_wide = liq_short_wide.apply(pd.to_numeric, errors="coerce").sort_index()

    return liq_long_wide, liq_short_wide


# ---------------------------------------------------------
# COMPUTE PIPELINE (run on Refresh)
# ---------------------------------------------------------
def compute_all(run_clear_cache: bool = True):
    try:
        if run_clear_cache:
            # clear cached fetch results to force fresh API calls
            st.cache_data.clear()

        status_placeholder.info("Loading perps list from GitHub...")
        GITHUB_PERPS_URL = "https://raw.githubusercontent.com/jhuku-code/Trading/main/Input-Files/perps_list.xlsx"
        df_perps = load_perps_list_from_github(GITHUB_PERPS_URL)

        if "Symbol" not in df_perps.columns:
            st.session_state["compute_error"] = "Column 'Symbol' not found in perps_list.xlsx"
            return

        usdt_perp_symbols = df_perps["Symbol"].dropna().astype(str).tolist()
        st.session_state["perps_loaded_count"] = len(usdt_perp_symbols)

        status_placeholder.info("Fetching liquidation history from Coinalyze API...")
        liq_long_data, liq_short_data = fetch_liquidations(
            usdt_perp_symbols, lookback_months, interval, API_KEY, convert_usd
        )

        if liq_long_data.empty or liq_short_data.empty:
            st.session_state["compute_error"] = "Empty dataframes returned from fetch."
            return

        # Ensure numeric and consistent
        liq_long_data = liq_long_data.apply(pd.to_numeric, errors="coerce").sort_index()
        liq_short_data = liq_short_data.apply(pd.to_numeric, errors="coerce").sort_index()

        st.session_state["liq_long_data"] = liq_long_data
        st.session_state["liq_short_data"] = liq_short_data

        # Z-scores (avoid zero-std)
        long_mean = liq_long_data.mean(numeric_only=True)
        long_std = liq_long_data.std(numeric_only=True).replace(0, np.nan)
        short_mean = liq_short_data.mean(numeric_only=True)
        short_std = liq_short_data.std(numeric_only=True).replace(0, np.nan)

        long_zscores = (liq_long_data - long_mean) / long_std
        short_zscores = (liq_short_data - short_mean) / short_std

        latest_long_z = long_zscores.iloc[-1].dropna()
        latest_short_z = short_zscores.iloc[-1].dropna()

        st.session_state["latest_long_z"] = latest_long_z
        st.session_state["latest_short_z"] = latest_short_z

        # Totals and shares
        rolling_window = 30

        total_long = liq_long_data.sum(axis=1)
        total_short = liq_short_data.sum(axis=1)

        st.session_state["total_long_series"] = total_long.fillna(0)
        st.session_state["total_short_series"] = total_short.fillna(0)

        total_long_safe = total_long.replace(0, np.nan)
        total_short_safe = total_short.replace(0, np.nan)

        ratio_long = liq_long_data.div(total_long_safe, axis=0)
        ratio_short = liq_short_data.div(total_short_safe, axis=0)

        long_ratio_ma = ratio_long.rolling(window=rolling_window, min_periods=rolling_window // 2).mean()
        short_ratio_ma = ratio_short.rolling(window=rolling_window, min_periods=rolling_window // 2).mean()

        latest_long_ratio = ratio_long.iloc[-1].dropna()
        latest_short_ratio = ratio_short.iloc[-1].dropna()
        latest_long_ratio_ma = long_ratio_ma.iloc[-1].dropna()
        latest_short_ratio_ma = short_ratio_ma.iloc[-1].dropna()

        long_excess = latest_long_ratio - latest_long_ratio_ma.reindex_like(latest_long_ratio)
        short_excess = latest_short_ratio - latest_short_ratio_ma.reindex_like(latest_short_ratio)

        long_excess_clean = long_excess.dropna()
        short_excess_clean = short_excess.dropna()

        sorted_long_excess = long_excess_clean.sort_values(ascending=False)
        sorted_short_excess = short_excess_clean.sort_values(ascending=False)

        sorted_long_z = latest_long_z.dropna().sort_values(ascending=False)
        sorted_short_z = latest_short_z.dropna().sort_values(ascending=False)

        # raw per-timestamp L/S ratio (wide df)
        raw_ratio = liq_long_data.div(liq_short_data.replace(0, np.nan))
        raw_ratio = raw_ratio.replace([np.inf, -np.inf], np.nan).sort_index()

        latest_raw_ratio = raw_ratio.iloc[-1].dropna()
        sorted_raw_ratio_desc = latest_raw_ratio.sort_values(ascending=False)
        sorted_raw_ratio_asc = latest_raw_ratio.sort_values(ascending=True)

        # Save to session
        st.session_state.update({
            "ratio_long": ratio_long,
            "ratio_short": ratio_short,
            "latest_long_ratio": latest_long_ratio,
            "latest_short_ratio": latest_short_ratio,
            "latest_long_ratio_ma": latest_long_ratio_ma,
            "latest_short_ratio_ma": latest_short_ratio_ma,
            "long_excess": long_excess,
            "short_excess": short_excess,
            "sorted_long_excess": sorted_long_excess,
            "sorted_short_excess": sorted_short_excess,
            "sorted_long_z": sorted_long_z,
            "sorted_short_z": sorted_short_z,
            "raw_ratio": raw_ratio,
            "latest_raw_ratio": latest_raw_ratio,
            "sorted_raw_ratio_desc": sorted_raw_ratio_desc,
            "sorted_raw_ratio_asc": sorted_raw_ratio_asc,
            "data_loaded": True,
        })

        st.session_state.pop("compute_error", None)
        status_placeholder.success("Data loaded and calculations complete.")
    except Exception as e:
        st.session_state["compute_error"] = str(e)
        st.session_state["data_loaded"] = False
        status_placeholder.error(f"Error during compute: {e}")


# ---------------------------------------------------------
# REFRESH BUTTON
# ---------------------------------------------------------
col_top = st.columns([1, 5])[0]
with col_top:
    if st.button("🔄 Refresh Data (fetch & compute)"):
        compute_all(run_clear_cache=True)


# ---------------------------------------------------------
# SHOW MESSAGE IF DATA NOT LOADED
# ---------------------------------------------------------
if not st.session_state.get("data_loaded", False):
    if "compute_error" in st.session_state:
        st.error(f"Last run error: {st.session_state['compute_error']}")
    status_placeholder.info("No computed data in session. Click 'Refresh Data' to fetch and run calculations.")
    if st.session_state.get("perps_loaded_count") is not None:
        st.write(f"Perps list last loaded has {st.session_state['perps_loaded_count']} symbols.")
    st.stop()


# ---------------------------------------------------------
# LOAD FROM SESSION
# ---------------------------------------------------------
liq_long_data = st.session_state["liq_long_data"]
liq_short_data = st.session_state["liq_short_data"]
latest_long_z = st.session_state["latest_long_z"]
latest_short_z = st.session_state["latest_short_z"]

ratio_long = st.session_state["ratio_long"]
ratio_short = st.session_state["ratio_short"]
latest_long_ratio = st.session_state["latest_long_ratio"]
latest_short_ratio = st.session_state["latest_short_ratio"]
latest_long_ratio_ma = st.session_state["latest_long_ratio_ma"]
latest_short_ratio_ma = st.session_state["latest_short_ratio_ma"]
long_excess = st.session_state["long_excess"]
short_excess = st.session_state["short_excess"]
sorted_long_excess = st.session_state["sorted_long_excess"]
sorted_short_excess = st.session_state["sorted_short_excess"]
sorted_long_z = st.session_state["sorted_long_z"]
sorted_short_z = st.session_state["sorted_short_z"]

raw_ratio = st.session_state["raw_ratio"]
latest_raw_ratio = st.session_state["latest_raw_ratio"]
sorted_raw_ratio_desc = st.session_state["sorted_raw_ratio_desc"]
sorted_raw_ratio_asc = st.session_state["sorted_raw_ratio_asc"]

total_long_series = st.session_state.get("total_long_series")
total_short_series = st.session_state.get("total_short_series")

st.subheader("Liquidations Data")
st.write("Data shape (long, short):", liq_long_data.shape, liq_short_data.shape)
st.write("Latest timestamp in data:", liq_long_data.index.max())

# ---------------------------------------------------------
# TIME SERIES CHARTS (all line charts)
# ---------------------------------------------------------
st.subheader("Time Series Charts")
colA, colB = st.columns([2, 2])

with colA:
    st.markdown("### Total long vs short liquidations (sum across symbols)")
    df_totals = pd.DataFrame({
        f"Total Long ({units_label})": total_long_series,
        f"Total Short ({units_label})": total_short_series,
    })
    df_totals = df_totals.dropna(how="all")
    if df_totals.empty:
        st.info("No totals time series to display.")
    else:
        st.line_chart(df_totals)

with colB:
    st.markdown("### Select symbol: Long & Short liquidations time series")
    symbols_list = sorted(liq_long_data.columns.tolist())
    selected_symbol = st.selectbox("Symbol for time series (searchable)", options=symbols_list)

    if selected_symbol:
        df_symbol = pd.DataFrame({
            f"{selected_symbol} - Long ({units_label})": liq_long_data[selected_symbol],
            f"{selected_symbol} - Short ({units_label})": liq_short_data[selected_symbol],
        })
        df_symbol = df_symbol.dropna(how="all")
        if df_symbol.empty:
            st.info(f"No historical long/short liquidation time series available for {selected_symbol}.")
        else:
            st.line_chart(df_symbol)

# ---------------------------------------------------------
# Long-to-Short Ratio: Top/Bottom & Time Series (fixed)
# ---------------------------------------------------------
st.subheader("Long-to-Short Ratio (raw values)")
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"### Top {top_n} symbols by latest Long/Short ratio (highest)")
    df_top_ratio = pd.DataFrame({
        "Symbol": sorted_raw_ratio_desc.head(top_n).index,
        "Latest Ratio (L/S)": sorted_raw_ratio_desc.head(top_n).values,
    }).set_index("Symbol")
    st.dataframe(df_top_ratio.style.format({"Latest Ratio (L/S)": "{:.3f}"}))

    st.markdown(f"### Bottom {top_n} symbols by latest Long/Short ratio (lowest)")
    df_bottom_ratio = pd.DataFrame({
        "Symbol": sorted_raw_ratio_asc.head(top_n).index,
        "Latest Ratio (L/S)": sorted_raw_ratio_asc.head(top_n).values,
    }).set_index("Symbol")
    st.dataframe(df_bottom_ratio.style.format({"Latest Ratio (L/S)": "{:.3f}"}))

with col2:
    st.markdown("### Select symbol: Long/Short ratio time series")

    selected_symbol_ratio = st.selectbox(
        "Symbol for ratio time series (searchable)",
        options=symbols_list,
        key="ratio_sym",
    )

    # smoothing / fallback options
    smoothing_method = st.selectbox("Plot method", options=["rolling_mean", "rolling_sum", "raw"], index=0)
    rolling_win = st.slider("Rolling window (periods) for smoothing / rolling-sum", min_value=1, max_value=200, value=5)

    if selected_symbol_ratio:
        # compute raw per-timestamp ratio from full series
        num = liq_long_data[selected_symbol_ratio].astype(float)
        den = liq_short_data[selected_symbol_ratio].astype(float).replace(0, np.nan)

        ratio_ts = num.div(den)  # per-timestamp raw ratio (may be very sparse)
        ratio_ts = ratio_ts.replace([np.inf, -np.inf], np.nan)

        # Diagnostics: fraction NaN
        total_points = len(ratio_ts)
        n_nan = ratio_ts.isna().sum()
        pct_nan = 100.0 * n_nan / total_points if total_points > 0 else 100.0

        st.markdown(f"**Ratio sparsity:** {n_nan}/{total_points} timestamps are NaN ({pct_nan:.1f}%).")

        # compute rolling mean and rolling sum ratio as alternatives
        ratio_rolling_mean = ratio_ts.rolling(window=rolling_win, min_periods=1).mean()

        # For rolling-sum ratio: sum(long)/sum(short) over rolling window (more robust when denom often zero)
        num_roll_sum = num.rolling(window=rolling_win, min_periods=1).sum()
        den_roll_sum = den.rolling(window=rolling_win, min_periods=1).sum().replace(0, np.nan)
        ratio_rolling_sum = num_roll_sum.div(den_roll_sum).replace([np.inf, -np.inf], np.nan)

        # Build DataFrame to plot
        if smoothing_method == "raw":
            df_ratio_plot = pd.DataFrame({f"{selected_symbol_ratio} - Raw L/S Ratio": ratio_ts})
        elif smoothing_method == "rolling_mean":
            df_ratio_plot = pd.DataFrame({
                f"{selected_symbol_ratio} - Raw L/S Ratio": ratio_ts,
                f"{selected_symbol_ratio} - RollingMean({rolling_win})": ratio_rolling_mean,
            })
        else:  # rolling_sum
            df_ratio_plot = pd.DataFrame({
                f"{selected_symbol_ratio} - RollingSumRatio({rolling_win})": ratio_rolling_sum,
                f"{selected_symbol_ratio} - Raw L/S Ratio": ratio_ts,
            })

        df_ratio_plot = df_ratio_plot.dropna(how="all")

        if df_ratio_plot.empty:
            st.info(
                "No historical ratio points available to plot for this symbol. "
                "This usually means the short-side liquidation series is zero or missing for most timestamps. "
                "Try 'rolling_sum' plot method or increase lookback window."
            )
        else:
            st.line_chart(df_ratio_plot)

# ---------------------------------------------------------
# Top / Bottom Z-score & Share tables (fixed)
# ---------------------------------------------------------
st.subheader("Top / Bottom Symbols (existing tables)")
tab_z, tab_share, tab_search = st.tabs(["By Z-Score (Long / Short)", "By Share vs 30-Period Avg", "Search by Symbol"])

# TAB 1: Z-SCORES
with tab_z:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### Long Liquidations - Top {top_n} by Z-Score")
        top_long_z_symbols = sorted_long_z.head(top_n).index if not sorted_long_z.empty else []
        df_top_long_z = pd.DataFrame(
            {"Symbol": top_long_z_symbols, "Long Z-Score": latest_long_z.reindex(top_long_z_symbols).values}
        ).set_index("Symbol")
        st.dataframe(df_top_long_z.style.format({"Long Z-Score": "{:.2f}"}))

        st.markdown(f"### Long Liquidations - Bottom {top_n} by Z-Score")
        bottom_long_z_symbols = sorted_long_z.tail(top_n).index if not sorted_long_z.empty else []
        df_bottom_long_z = pd.DataFrame(
            {"Symbol": bottom_long_z_symbols, "Long Z-Score": latest_long_z.reindex(bottom_long_z_symbols).values}
        ).set_index("Symbol")
        st.dataframe(df_bottom_long_z.style.format({"Long Z-Score": "{:.2f}"}))

    with col2:
        st.markdown(f"### Short Liquidations - Top {top_n} by Z-Score")
        top_short_z_symbols = sorted_short_z.head(top_n).index if not sorted_short_z.empty else []
        df_top_short_z = pd.DataFrame(
            {"Symbol": top_short_z_symbols, "Short Z-Score": latest_short_z.reindex(top_short_z_symbols).values}
        ).set_index("Symbol")
        st.dataframe(df_top_short_z.style.format({"Short Z-Score": "{:.2f}"}))

        st.markdown(f"### Short Liquidations - Bottom {top_n} by Z-Score")
        bottom_short_z_symbols = sorted_short_z.tail(top_n).index if not sorted_short_z.empty else []
        df_bottom_short_z = pd.DataFrame(
            {"Symbol": bottom_short_z_symbols, "Short Z-Score": latest_short_z.reindex(bottom_short_z_symbols).values}
        ).set_index("Symbol")
        st.dataframe(df_bottom_short_z.style.format({"Short Z-Score": "{:.2f}"}))

# TAB 2: SHARE VS 30-PERIOD AVG
with tab_share:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### Long Liquidations - Top {top_n} Excess Share vs 30-Period Avg")
        top_long_excess_symbols = sorted_long_excess.head(top_n).index if not sorted_long_excess.empty else []
        df_top_long_excess = pd.DataFrame({
            "Symbol": top_long_excess_symbols,
            "Latest Share (%)": (latest_long_ratio.reindex(top_long_excess_symbols) * 100.0).values,
            "30-Period Avg Share (%)": (latest_long_ratio_ma.reindex(top_long_excess_symbols) * 100.0).values,
            "Excess (% pts)": (long_excess.reindex(top_long_excess_symbols) * 100.0).values,
        }).set_index("Symbol")
        st.dataframe(df_top_long_excess.style.format({
            "Latest Share (%)": "{:.3f}",
            "30-Period Avg Share (%)": "{:.3f}",
            "Excess (% pts)": "{:.3f}",
        }))

        st.markdown(f"### Long Liquidations - Bottom {top_n} Excess Share vs 30-Period Avg")
        bottom_long_excess_symbols = sorted_long_excess.tail(top_n).index if not sorted_long_excess.empty else []
        df_bottom_long_excess = pd.DataFrame({
            "Symbol": bottom_long_excess_symbols,
            "Latest Share (%)": (latest_long_ratio.reindex(bottom_long_excess_symbols) * 100.0).values,
            "30-Period Avg Share (%)": (latest_long_ratio_ma.reindex(bottom_long_excess_symbols) * 100.0).values,
            "Excess (% pts)": (long_excess.reindex(bottom_long_excess_symbols) * 100.0).values,
        }).set_index("Symbol")
        st.dataframe(df_bottom_long_excess.style.format({
            "Latest Share (%)": "{:.3f}",
            "30-Period Avg Share (%)": "{:.3f}",
            "Excess (% pts)": "{:.3f}",
        }))

    with col2:
        st.markdown(f"### Short Liquidations - Top {top_n} Excess Share vs 30-Period Avg")
        top_short_excess_symbols = sorted_short_excess.head(top_n).index if not sorted_short_excess.empty else []
        df_top_short_excess = pd.DataFrame({
            "Symbol": top_short_excess_symbols,
            "Latest Share (%)": (latest_short_ratio.reindex(top_short_excess_symbols) * 100.0).values,
            "30-Period Avg Share (%)": (latest_short_ratio_ma.reindex(top_short_excess_symbols) * 100.0).values,
            "Excess (% pts)": (short_excess.reindex(top_short_excess_symbols) * 100.0).values,
        }).set_index("Symbol")
        st.dataframe(df_top_short_excess.style.format({
            "Latest Share (%)": "{:.3f}",
            "30-Period Avg Share (%)": "{:.3f}",
            "Excess (% pts)": "{:.3f}",
        }))

        st.markdown(f"### Short Liquidations - Bottom {top_n} Excess Share vs 30-Period Avg")
        bottom_short_excess_symbols = sorted_short_excess.tail(top_n).index if not sorted_short_excess.empty else []
        df_bottom_short_excess = pd.DataFrame({
            "Symbol": bottom_short_excess_symbols,
            "Latest Share (%)": (latest_short_ratio.reindex(bottom_short_excess_symbols) * 100.0).values,
            "30-Period Avg Share (%)": (latest_short_ratio_ma.reindex(bottom_short_excess_symbols) * 100.0).values,
            "Excess (% pts)": (short_excess.reindex(bottom_short_excess_symbols) * 100.0).values,
        }).set_index("Symbol")
        st.dataframe(df_bottom_short_excess.style.format({
            "Latest Share (%)": "{:.3f}",
            "30-Period Avg Share (%)": "{:.3f}",
            "Excess (% pts)": "{:.3f}",
        }))

# TAB 3: SEARCH BY SYMBOL
with tab_search:
    st.markdown("### Search Long / Short Metrics by Symbol")
    search_symbol = st.text_input("Enter symbol (exact as in perps list, e.g. BTCUSDT_PERP.A)", value="").strip()

    if search_symbol:
        symbol = search_symbol.upper()
        if symbol not in liq_long_data.columns:
            st.error(f"Symbol '{symbol}' not found in data.")
        else:
            def safe(x):
                return np.nan if pd.isna(x) else x

            metrics = [
                "Long Share (%)",
                "Short Share (%)",
                "Long 30-Period Avg Share (%)",
                "Short 30-Period Avg Share (%)",
                "Long Excess (% pts)",
                "Short Excess (% pts)",
                "Long Z-Score",
                "Short Z-Score",
                "Latest Raw L/S Ratio",
            ]
            values = [
                safe(latest_long_ratio.get(symbol, np.nan) * 100.0),
                safe(latest_short_ratio.get(symbol, np.nan) * 100.0),
                safe(latest_long_ratio_ma.get(symbol, np.nan) * 100.0),
                safe(latest_short_ratio_ma.get(symbol, np.nan) * 100.0),
                safe(long_excess.get(symbol, np.nan) * 100.0),
                safe(short_excess.get(symbol, np.nan) * 100.0),
                safe(latest_long_z.get(symbol, np.nan)),
                safe(latest_short_z.get(symbol, np.nan)),
                safe(latest_raw_ratio.get(symbol, np.nan)),
            ]

            df_search = pd.DataFrame({"Metric": metrics, "Value": values}).set_index("Metric")

            def fmt_value(val, metric):
                if pd.isna(val):
                    return "N/A"
                if "Share" in metric or "Excess" in metric:
                    return f"{val:.3f}"
                elif "Ratio" in metric:
                    return f"{val:.3f}"
                else:
                    return f"{val:.2f}"

            df_search_display = df_search.copy()
            df_search_display["Value"] = [fmt_value(v, m) for m, v in zip(df_search_display.index, df_search_display["Value"])]
            st.dataframe(df_search_display)

            rank_lines = []
            if symbol in sorted_long_excess.index:
                long_ex_rank = sorted_long_excess.index.get_loc(symbol) + 1
                rank_lines.append(f"- Long Excess Share: rank {long_ex_rank} of {len(sorted_long_excess)} (1 = highest positive excess)")
            if symbol in sorted_short_excess.index:
                short_ex_rank = sorted_short_excess.index.get_loc(symbol) + 1
                rank_lines.append(f"- Short Excess Share: rank {short_ex_rank} of {len(sorted_short_excess)} (1 = highest positive excess)")

            if rank_lines:
                st.markdown("#### Approximate Ranks")
                for line in rank_lines:
                    st.write(line)

            with st.expander("Show full latest snapshot (all symbols)"):
                snapshot_df = pd.DataFrame({
                    "Long Share (%)": latest_long_ratio * 100.0,
                    "Short Share (%)": latest_short_ratio * 100.0,
                    "Long 30-Period Avg Share (%)": latest_long_ratio_ma * 100.0,
                    "Short 30-Period Avg Share (%)": latest_short_ratio_ma * 100.0,
                    "Long Excess (% pts)": long_excess * 100.0,
                    "Short Excess (% pts)": short_excess * 100.0,
                    "Long Z-Score": latest_long_z,
                    "Short Z-Score": latest_short_z,
                    "Latest Raw L/S Ratio": latest_raw_ratio,
                })
                st.dataframe(snapshot_df.sort_values("Long Excess (% pts)", ascending=False).style.format({
                    "Long Share (%)": "{:.3f}",
                    "Short Share (%)": "{:.3f}",
                    "Long 30-Period Avg Share (%)": "{:.3f}",
                    "Short 30-Period Avg Share (%)": "{:.3f}",
                    "Long Excess (% pts)": "{:.3f}",
                    "Short Excess (% pts)": "{:.3f}",
                    "Long Z-Score": "{:.2f}",
                    "Short Z-Score": "{:.2f}",
                    "Latest Raw L/S Ratio": "{:.3f}",
                }))

# End of app
