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
st.set_page_config(page_title="Liquidations Z-Score / Share Monitor", layout="wide")
st.title("Liquidations Monitor (Z-Score + Share vs Total)")

# ---------------------------------------------------------
# INPUTS / CONTROLS
# ---------------------------------------------------------
# Number of symbols in top/bottom lists
top_n = st.number_input(
    "Number of symbols to show in Top/Bottom lists",
    min_value=5,
    max_value=100,
    value=15,
    step=5,
)

# Lookback months (default 3 months)
lookback_months = st.number_input(
    "Lookback window (months)",
    min_value=1,
    max_value=12,
    value=3,
    step=1,
)

# Interval selection
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

# Toggle: convert liquidation values to USD or keep in native units
convert_usd = st.checkbox("Convert liquidation values to USD", value=True)
units_label = "USD" if convert_usd else "native units"

# Small status area
status_placeholder = st.empty()

# ---------------------------------------------------------
# CONFIG: API KEY & BASE URL
# (note: we won't access API until Refresh is pressed)
# ---------------------------------------------------------
API_KEY = st.secrets.get("API_KEY")
BASE_URL = "https://api.coinalyze.net/v1"

# ---------------------------------------------------------
# HELPERS / CACHED FUNCTIONS (these only *define* work)
# ---------------------------------------------------------
def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


@st.cache_data(ttl=3600)
def load_perps_list_from_github(url: str) -> pd.DataFrame:
    """Cached small helper to load perps list from GitHub raw URL."""
    return pd.read_excel(url)


@st.cache_data(ttl=1800)
def fetch_liquidations(symbols, months_back: int, interval: str, api_key: str, convert_usd_flag: bool):
    """
    Returns:
        liq_long_wide: DataFrame (index=time, columns=symbol, values=long liquidations)
        liq_short_wide: DataFrame (index=time, columns=symbol, values=short liquidations)
    """
    liq_url = f"{BASE_URL}/liquidation-history"

    from_timestamp = int((datetime.now() - relativedelta(months=months_back)).timestamp())
    to_timestamp = int(datetime.now().timestamp())

    all_rows = []

    for batch in chunked(symbols, 20):  # max 20 per request
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
                    # Unexpected format; skip this batch
                    break

                for entry in data:
                    symbol = entry.get("symbol")
                    history = entry.get("history", [])
                    if not symbol or not history:
                        continue

                    df = pd.DataFrame(history)
                    # t = timestamp, l = long liqs, s = short liqs
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

                st.warning(
                    f"429 Too Many Requests for batch starting with {batch[0]}... "
                    f"sleeping {retry_after} seconds before retrying"
                )
                time.sleep(retry_after)

            else:
                st.error(
                    f"Error for batch starting with {batch[0]}: "
                    f"{resp.status_code} - {resp.text}"
                )
                break

    if not all_rows:
        raise Exception("No data received for any symbol.")

    df_long = pd.concat(all_rows, ignore_index=True)

    liq_long_wide = (
        df_long.pivot_table(index="time", columns="symbol", values="liq_long")
        .sort_index()
    )
    liq_short_wide = (
        df_long.pivot_table(index="time", columns="symbol", values="liq_short")
        .sort_index()
    )

    return liq_long_wide, liq_short_wide


# ---------------------------------------------------------
# COMPUTE PIPELINE (runs only when Refresh pressed)
# ---------------------------------------------------------
def compute_all(run_clear_cache: bool = True):
    """
    Fetch perps list, liquidations and perform calculations.
    Stores results in st.session_state for later display.
    """
    try:
        # Optionally clear caches to force fresh API calls
        if run_clear_cache:
            st.cache_data.clear()

        status_placeholder.info("Loading perps list from GitHub...")
        GITHUB_PERPS_URL = (
            "https://raw.githubusercontent.com/jhuku-code/Trading/main/Input-Files/perps_list.xlsx"
        )
        df_perps = load_perps_list_from_github(GITHUB_PERPS_URL)

        if "Symbol" not in df_perps.columns:
            st.session_state["compute_error"] = "Column 'Symbol' not found in perps_list.xlsx"
            return

        usdt_perp_symbols = df_perps["Symbol"].dropna().astype(str).tolist()
        st.session_state["perps_loaded_count"] = len(usdt_perp_symbols)

        # Fetch liquidations (this is the heavy part)
        status_placeholder.info("Fetching liquidation history from Coinalyze API...")
        liq_long_data, liq_short_data = fetch_liquidations(
            usdt_perp_symbols, lookback_months, interval, API_KEY, convert_usd
        )

        # Basic checks
        if liq_long_data.empty or liq_short_data.empty:
            st.session_state["compute_error"] = "Empty dataframes returned from fetch."
            return

        # Save raw to session_state
        st.session_state["liq_long_data"] = liq_long_data
        st.session_state["liq_short_data"] = liq_short_data

        # Z-scores
        long_zscores = (liq_long_data - liq_long_data.mean(numeric_only=True)) / liq_long_data.std(
            numeric_only=True
        )
        short_zscores = (liq_short_data - liq_short_data.mean(numeric_only=True)) / liq_short_data.std(
            numeric_only=True
        )
        latest_long_z = long_zscores.iloc[-1]
        latest_short_z = short_zscores.iloc[-1]

        st.session_state["latest_long_z"] = latest_long_z
        st.session_state["latest_short_z"] = latest_short_z

        # Share of total & excess vs trailing average
        rolling_window = 30

        total_long = liq_long_data.sum(axis=1)
        total_short = liq_short_data.sum(axis=1)
        total_long_safe = total_long.replace(0, np.nan)
        total_short_safe = total_short.replace(0, np.nan)

        ratio_long = liq_long_data.div(total_long_safe, axis=0)
        ratio_short = liq_short_data.div(total_short_safe, axis=0)

        long_ratio_ma = ratio_long.rolling(window=rolling_window, min_periods=rolling_window // 2).mean()
        short_ratio_ma = ratio_short.rolling(window=rolling_window, min_periods=rolling_window // 2).mean()

        latest_long_ratio = ratio_long.iloc[-1]
        latest_short_ratio = ratio_short.iloc[-1]
        latest_long_ratio_ma = long_ratio_ma.iloc[-1]
        latest_short_ratio_ma = short_ratio_ma.iloc[-1]

        long_excess = latest_long_ratio - latest_long_ratio_ma
        short_excess = latest_short_ratio - latest_short_ratio_ma

        long_excess_clean = long_excess.dropna()
        short_excess_clean = short_excess.dropna()

        sorted_long_excess = long_excess_clean.sort_values(ascending=False)
        sorted_short_excess = short_excess_clean.sort_values(ascending=False)

        sorted_long_z = latest_long_z.sort_values(ascending=False)
        sorted_short_z = latest_short_z.sort_values(ascending=False)

        # Put computed objects in session_state
        st.session_state["ratio_long"] = ratio_long
        st.session_state["ratio_short"] = ratio_short
        st.session_state["latest_long_ratio"] = latest_long_ratio
        st.session_state["latest_short_ratio"] = latest_short_ratio
        st.session_state["latest_long_ratio_ma"] = latest_long_ratio_ma
        st.session_state["latest_short_ratio_ma"] = latest_short_ratio_ma
        st.session_state["long_excess"] = long_excess
        st.session_state["short_excess"] = short_excess
        st.session_state["sorted_long_excess"] = sorted_long_excess
        st.session_state["sorted_short_excess"] = sorted_short_excess
        st.session_state["sorted_long_z"] = sorted_long_z
        st.session_state["sorted_short_z"] = sorted_short_z

        st.session_state["data_loaded"] = True
        st.session_state.pop("compute_error", None)
        status_placeholder.success("Data loaded and calculations complete.")
    except Exception as e:
        st.session_state["compute_error"] = str(e)
        st.session_state["data_loaded"] = False
        status_placeholder.error(f"Error during compute: {e}")


# ---------------------------------------------------------
# REFRESH BUTTON (user triggers everything explicitly)
# ---------------------------------------------------------
col_top = st.columns([1, 5])[0]
with col_top:
    if st.button("🔄 Refresh Data (fetch & compute)"):
        # run_clear_cache=True will clear cache_data before compute (fresh pull).
        # Change to False if you want to keep cached fetch results.
        compute_all(run_clear_cache=True)

# ---------------------------------------------------------
# SHOW MESSAGE IF DATA NOT LOADED
# ---------------------------------------------------------
if not st.session_state.get("data_loaded", False):
    if "compute_error" in st.session_state:
        st.error(f"Last run error: {st.session_state['compute_error']}")
    status_placeholder.info("No computed data in session. Click 'Refresh Data' to fetch and run calculations.")
    # offer to show perps count if loaded earlier
    if st.session_state.get("perps_loaded_count") is not None:
        st.write(f"Perps list last loaded has {st.session_state['perps_loaded_count']} symbols.")
    # stop here — heavy code will not run until user clicks Refresh
    st.stop()

# ---------------------------------------------------------
# If we reach here, data is available in session_state -> show results
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

st.subheader("Liquidations Data")
st.write("Data shape (long, short):", liq_long_data.shape, liq_short_data.shape)
st.write("Latest timestamp in data:", liq_long_data.index.max())

# ---------------------------------------------------------
# BUILD TOP / BOTTOM SYMBOL LISTS (same logic)
# ---------------------------------------------------------
top_long_excess_symbols = sorted_long_excess.head(top_n).index
bottom_long_excess_symbols = sorted_long_excess.tail(top_n).index

top_short_excess_symbols = sorted_short_excess.head(top_n).index
bottom_short_excess_symbols = sorted_short_excess.tail(top_n).index

top_long_z_symbols = sorted_long_z.head(top_n).index
bottom_long_z_symbols = sorted_long_z.tail(top_n).index

top_short_z_symbols = sorted_short_z.head(top_n).index
bottom_short_z_symbols = sorted_short_z.tail(top_n).index

# ---------------------------------------------------------
# DISPLAY RESULTS (same as before)
# ---------------------------------------------------------
st.subheader("Top / Bottom Symbols")

tab_z, tab_share, tab_search = st.tabs(
    ["By Z-Score (Long / Short)", "By Share vs 30-Period Avg", "Search by Symbol"]
)

# ---------- TAB 1: Z-SCORES ----------
with tab_z:
    col1, col2 = st.columns(2)

    # LONG
    with col1:
        st.markdown(f"### Long Liquidations - Top {top_n} by Z-Score")
        df_top_long_z = (
            pd.DataFrame(
                {"Symbol": top_long_z_symbols, "Long Z-Score": latest_long_z[top_long_z_symbols].values}
            )
            .set_index("Symbol")
        )
        st.dataframe(df_top_long_z.style.format({"Long Z-Score": "{:.2f}"}))

        st.markdown(f"### Long Liquidations - Bottom {top_n} by Z-Score")
        df_bottom_long_z = (
            pd.DataFrame(
                {"Symbol": bottom_long_z_symbols, "Long Z-Score": latest_long_z[bottom_long_z_symbols].values}
            )
            .set_index("Symbol")
        )
        st.dataframe(df_bottom_long_z.style.format({"Long Z-Score": "{:.2f}"}))

    # SHORT
    with col2:
        st.markdown(f"### Short Liquidations - Top {top_n} by Z-Score")
        df_top_short_z = (
            pd.DataFrame(
                {"Symbol": top_short_z_symbols, "Short Z-Score": latest_short_z[top_short_z_symbols].values}
            )
            .set_index("Symbol")
        )
        st.dataframe(df_top_short_z.style.format({"Short Z-Score": "{:.2f}"}))

        st.markdown(f"### Short Liquidations - Bottom {top_n} by Z-Score")
        df_bottom_short_z = (
            pd.DataFrame(
                {"Symbol": bottom_short_z_symbols, "Short Z-Score": latest_short_z[bottom_short_z_symbols].values}
            )
            .set_index("Symbol")
        )
        st.dataframe(df_bottom_short_z.style.format({"Short Z-Score": "{:.2f}"}))

# ---------- TAB 2: SHARE VS 30-PERIOD AVERAGE ----------
with tab_share:
    col1, col2 = st.columns(2)

    # LONG SIDE
    with col1:
        st.markdown(f"### Long Liquidations - Top {top_n} Excess Share vs 30-Period Avg")
        df_top_long_excess = pd.DataFrame(
            {
                "Symbol": top_long_excess_symbols,
                "Latest Share (%)": (latest_long_ratio[top_long_excess_symbols] * 100.0).values,
                "30-Period Avg Share (%)": (latest_long_ratio_ma[top_long_excess_symbols] * 100.0).values,
                "Excess (% pts)": (long_excess[top_long_excess_symbols] * 100.0).values,
            }
        ).set_index("Symbol")
        st.dataframe(
            df_top_long_excess.style.format(
                {
                    "Latest Share (%)": "{:.3f}",
                    "30-Period Avg Share (%)": "{:.3f}",
                    "Excess (% pts)": "{:.3f}",
                }
            )
        )

        st.markdown(f"### Long Liquidations - Bottom {top_n} Excess Share vs 30-Period Avg")
        df_bottom_long_excess = pd.DataFrame(
            {
                "Symbol": bottom_long_excess_symbols,
                "Latest Share (%)": (latest_long_ratio[bottom_long_excess_symbols] * 100.0).values,
                "30-Period Avg Share (%)": (latest_long_ratio_ma[bottom_long_excess_symbols] * 100.0).values,
                "Excess (% pts)": (long_excess[bottom_long_excess_symbols] * 100.0).values,
            }
        ).set_index("Symbol")
        st.dataframe(
            df_bottom_long_excess.style.format(
                {
                    "Latest Share (%)": "{:.3f}",
                    "30-Period Avg Share (%)": "{:.3f}",
                    "Excess (% pts)": "{:.3f}",
                }
            )
        )

    # SHORT SIDE
    with col2:
        st.markdown(f"### Short Liquidations - Top {top_n} Excess Share vs 30-Period Avg")
        df_top_short_excess = pd.DataFrame(
            {
                "Symbol": top_short_excess_symbols,
                "Latest Share (%)": (latest_short_ratio[top_short_excess_symbols] * 100.0).values,
                "30-Period Avg Share (%)": (latest_short_ratio_ma[top_short_excess_symbols] * 100.0).values,
                "Excess (% pts)": (short_excess[top_short_excess_symbols] * 100.0).values,
            }
        ).set_index("Symbol")
        st.dataframe(
            df_top_short_excess.style.format(
                {
                    "Latest Share (%)": "{:.3f}",
                    "30-Period Avg Share (%)": "{:.3f}",
                    "Excess (% pts)": "{:.3f}",
                }
            )
        )

        st.markdown(f"### Short Liquidations - Bottom {top_n} Excess Share vs 30-Period Avg")
        df_bottom_short_excess = pd.DataFrame(
            {
                "Symbol": bottom_short_excess_symbols,
                "Latest Share (%)": (latest_short_ratio[bottom_short_excess_symbols] * 100.0).values,
                "30-Period Avg Share (%)": (latest_short_ratio_ma[bottom_short_excess_symbols] * 100.0).values,
                "Excess (% pts)": (short_excess[bottom_short_excess_symbols] * 100.0).values,
            }
        ).set_index("Symbol")
        st.dataframe(
            df_bottom_short_excess.style.format(
                {
                    "Latest Share (%)": "{:.3f}",
                    "30-Period Avg Share (%)": "{:.3f}",
                    "Excess (% pts)": "{:.3f}",
                }
            )
        )

# ---------- TAB 3: SEARCH BY SYMBOL ----------
with tab_search:
    st.markdown("### Search Long / Short Metrics by Symbol")

    search_symbol = st.text_input(
        "Enter symbol (exact as in perps list, e.g. BTCUSDT_PERP.A)",
        value="",
    ).strip()

    if search_symbol:
        symbol = search_symbol.upper()

        if symbol not in liq_long_data.columns:
            st.error(f"Symbol '{symbol}' not found in data.")
        else:
            # Handle possible NaNs safely
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
            ]

            df_search = pd.DataFrame({"Metric": metrics, "Value": values}).set_index("Metric")

            def fmt_value(val, metric):
                if pd.isna(val):
                    return "N/A"
                if "Share" in metric or "Excess" in metric:
                    return f"{val:.3f}"
                else:
                    return f"{val:.2f}"

            df_search_display = df_search.copy()
            df_search_display["Value"] = [
                fmt_value(v, m) for m, v in zip(df_search_display.index, df_search_display["Value"])
            ]
            st.dataframe(df_search_display)

            # Ranks based on excess share (where available)
            rank_lines = []
            if symbol in sorted_long_excess.index:
                long_ex_rank = sorted_long_excess.index.get_loc(symbol) + 1
                rank_lines.append(
                    f"- Long Excess Share: rank {long_ex_rank} of {len(sorted_long_excess)} (1 = highest positive excess)"
                )
            if symbol in sorted_short_excess.index:
                short_ex_rank = sorted_short_excess.index.get_loc(symbol) + 1
                rank_lines.append(
                    f"- Short Excess Share: rank {short_ex_rank} of {len(sorted_short_excess)} (1 = highest positive excess)"
                )

            if rank_lines:
                st.markdown("#### Approximate Ranks")
                for line in rank_lines:
                    st.write(line)

# Optional: Show full latest snapshot
with st.expander("Show full latest snapshot (all symbols)"):
    snapshot_df = pd.DataFrame(
        {
            "Long Share (%)": latest_long_ratio * 100.0,
            "Short Share (%)": latest_short_ratio * 100.0,
            "Long 30-Period Avg Share (%)": latest_long_ratio_ma * 100.0,
            "Short 30-Period Avg Share (%)": latest_short_ratio_ma * 100.0,
            "Long Excess (% pts)": long_excess * 100.0,
            "Short Excess (% pts)": short_excess * 100.0,
            "Long Z-Score": latest_long_z,
            "Short Z-Score": latest_short_z,
        }
    )
    st.dataframe(
        snapshot_df.sort_values("Long Excess (% pts)", ascending=False).style.format(
            {
                "Long Share (%)": "{:.3f}",
                "Short Share (%)": "{:.3f}",
                "Long 30-Period Avg Share (%)": "{:.3f}",
                "Short 30-Period Avg Share (%)": "{:.3f}",
                "Long Excess (% pts)": "{:.3f}",
                "Short Excess (% pts)": "{:.3f}",
                "Long Z-Score": "{:.2f}",
                "Short Z-Score": "{:.2f}",
            }
        )
    )
