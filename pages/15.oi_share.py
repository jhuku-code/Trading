import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import io
import streamlit as st

# ---------------------------------------------------------
# STREAMLIT PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="OI Market Share Dashboard", layout="wide")
st.title("Open Interest Market Share Dashboard")

# ---------------------------------------------------------
# CONSTANTS / CONFIG
# ---------------------------------------------------------
API_KEY = st.secrets.get("API_KEY", "xxxxxx")
BASE_URL = "https://api.coinalyze.net/v1"

# GitHub repo info – CHANGE THESE TO YOURS
GITHUB_USER = "jhuku-code"
GITHUB_REPO = "Trading"
BRANCH = "main"  # or "master" or another branch name

# Build raw GitHub URL for /Input-Files/perps_list.xlsx
GITHUB_EXCEL_URL = (
    f"https://raw.githubusercontent.com/{GITHUB_USER}/"
    f"{GITHUB_REPO}/{BRANCH}/Input-Files/perps_list.xlsx"
)

# ---------------------------------------------------------
# SIDEBAR SETTINGS
# ---------------------------------------------------------
st.sidebar.header("Settings")

st.sidebar.markdown("**Perps list file (on GitHub):**")
st.sidebar.code("/Input-Files/perps_list.xlsx", language="text")

st.sidebar.markdown("**Repo:**")
st.sidebar.write(f"{GITHUB_USER}/{GITHUB_REPO} ({BRANCH})")

# Lookback window in months
months_back = st.sidebar.number_input(
    "Lookback window (months)",
    min_value=1,
    max_value=12,
    value=3,
    step=1,
)

# Number of symbols in Top/Bottom list (instead of fixed 15)
top_bottom_n = st.sidebar.number_input(
    "Number of symbols to show in Top/Bottom lists",
    min_value=5,
    max_value=100,
    value=15,
    step=5,
)

# Sidebar short-circuit cache-clear button (manual)
if st.sidebar.button("Clear cached API data"):
    st.cache_data.clear()
    st.sidebar.success("Cache cleared. Click 'Run analysis' to fetch fresh data.")

# ---------------------------------------------------------
# PAGE-LEVEL CONTROL: do not auto-run heavy code
# ---------------------------------------------------------
if "run_analysis" not in st.session_state:
    st.session_state["run_analysis"] = False

col_run, col_force = st.columns([1, 1])

with col_run:
    if st.button("Run analysis / Refresh calculations"):
        st.session_state["run_analysis"] = True

with col_force:
    if st.button("Force refresh (clear cache & run)"):
        st.cache_data.clear()
        st.session_state["run_analysis"] = True

# Optional: a 'clear results' button to hide results without clearing cache
if st.button("Clear results (hide)"):
    st.session_state["run_analysis"] = False

st.write("Tip: The page won't fetch or compute OI until you click **Run analysis**.")

# ---------------------------------------------------------
# LOAD SYMBOL LIST FROM GITHUB /Input-Files/perps_list.xlsx
# ---------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_symbols_from_github(url: str):
    """
    Download perps_list.xlsx from GitHub (in /Input-Files folder)
    and return the list of symbols from the 'Symbol' column.
    """
    resp = requests.get(url)
    resp.raise_for_status()  # will raise if not 200
    # Use BytesIO to be robust
    df = pd.read_excel(io.BytesIO(resp.content))
    if "Symbol" not in df.columns:
        raise ValueError("Excel file must contain a 'Symbol' column.")
    return df["Symbol"].dropna().tolist()

try:
    usdt_perp_symbols = load_symbols_from_github(GITHUB_EXCEL_URL)
except Exception as e:
    st.error(f"Error loading perps_list.xlsx from GitHub: {e}")
    st.stop()

if not usdt_perp_symbols:
    st.error("No symbols found in perps_list.xlsx on GitHub.")
    st.stop()

st.info(f"Loaded {len(usdt_perp_symbols)} perp symbols from GitHub (symbols list loaded; data fetch is manual).")

# ---------------------------------------------------------
# HELPER: chunked
# ---------------------------------------------------------
def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]

# ---------------------------------------------------------
# FETCH OPEN INTEREST DATA (CACHED) - only called when run
# ---------------------------------------------------------
@st.cache_data(show_spinner=True)
def fetch_oi_data(symbols, months_back, api_key):
    oi_url = f"{BASE_URL}/open-interest-history"

    from_timestamp = int(
        (datetime.now() - relativedelta(months=int(months_back))).timestamp()
    )
    to_timestamp = int(datetime.now().timestamp())

    all_rows = []

    for batch in chunked(symbols, 20):  # max 20 per request
        symbols_param = ",".join(batch)
        params = {
            "symbols": symbols_param,
            "interval": "4hour",
            "from": from_timestamp,
            "to": to_timestamp,
            "api_key": api_key,
        }

        while True:
            resp = requests.get(oi_url, params=params)

            # If OK, process and break out of retry loop
            if resp.status_code == 200:
                data = resp.json()
                if not isinstance(data, list):
                    # Log unexpected format and skip this batch
                    st.warning(
                        f"Unexpected response format for batch starting {batch[0]}: {data}"
                    )
                    break

                for entry in data:
                    symbol = entry.get("symbol")
                    history = entry.get("history", [])
                    if not symbol or not history:
                        st.info(f"No history for {symbol}")
                        continue

                    df = pd.DataFrame(history)
                    df["time"] = pd.to_datetime(df["t"], unit="s")
                    df.rename(columns={"o": "open_interest"}, inplace=True)
                    df = df[["time", "open_interest"]]
                    df["symbol"] = symbol
                    all_rows.append(df)

                # small pause to be gentle with API
                time.sleep(1)
                break

            # If rate limited, respect Retry-After header and retry
            elif resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                try:
                    retry_after = int(retry_after) if retry_after is not None else 30
                except ValueError:
                    retry_after = 30

                st.warning(
                    f"429 Too Many Requests for batch {batch[:3]}... "
                    f"sleeping {retry_after} seconds before retrying"
                )
                time.sleep(retry_after)

            # Other errors: log and skip this batch
            else:
                st.error(
                    f"Error for batch {batch[:3]}...: {resp.status_code} - {resp.text}"
                )
                break

    if not all_rows:
        raise Exception("No data received for any symbol.")

    df_long = pd.concat(all_rows, ignore_index=True)

    # Pivot → wide: rows = time, columns = symbol, values = open_interest
    df_wide = df_long.pivot_table(
        index="time",
        columns="symbol",
        values="open_interest",
    ).sort_index()

    return df_wide

# ---------------------------------------------------------
# RUN ANALYSIS (only when user requests it)
# ---------------------------------------------------------
if st.session_state["run_analysis"]:
    with st.spinner("Fetching open interest data from Coinalyze and running analysis..."):
        try:
            oi_data = fetch_oi_data(usdt_perp_symbols, months_back, API_KEY)
        except Exception as e:
            st.error(f"Error fetching OI data: {e}")
            st.stop()

    st.success("OI data loaded successfully.")

    # ----------------------------------------------------------
    # 1. Total OI values over time + line chart
    # ----------------------------------------------------------
    st.subheader("Total Open Interest Over Time")
    total_oi = oi_data.sum(axis=1)
    st.line_chart(total_oi)

    # ----------------------------------------------------------
    # 2. Market share dataframe (oi_share_df)
    # ----------------------------------------------------------
    oi_share_df = oi_data.div(oi_data.sum(axis=1), axis=0) * 100
    oi_share_df = oi_share_df.round(3)

    # ---------------------- NEW: Single-symbol OI share chart ----------------------
    # Let user pick a symbol (dropdown that can be typed into). Show its series (percent).
    st.subheader("Open Interest Market Share — Single Symbol View")  # <-- ADDED
    # Provide a helpful instruction
    st.markdown("Select a symbol to view its open interest market share (percent) time series. You can start typing to filter the list.")  # <-- ADDED

    # Ensure the symbols list matches columns in oi_share_df (in case some symbols had no data)
    available_symbols = list(oi_share_df.columns)
    if not available_symbols:
        st.warning("No symbols available to plot (oi_share_df is empty).")
    else:
        # Use selectbox — supports typing to filter (autofill-like)
        default_idx = 0
        try:
            # prefer to preselect a sensible default if present (e.g., first symbol in your list)
            default_symbol = available_symbols[default_idx]
        except Exception:
            default_symbol = available_symbols[0]

        selected_symbol = st.selectbox(
            "Pick a symbol:",
            options=available_symbols,
            index=available_symbols.index(default_symbol) if default_symbol in available_symbols else 0,
        )  # <-- ADDED

        # Plot the percent time series for the selected symbol
        # Convert to Series with datetime index (oi_share_df is already datetime-indexed)
        symbol_series = oi_share_df[selected_symbol].dropna()
        if symbol_series.empty:
            st.info(f"No market share data available for {selected_symbol}.")
        else:
            st.line_chart(symbol_series)  # percentage values (rounded to 3 dp) plotted over time
    # -------------------- END NEW BLOCK --------------------

    # ----------------------------------------------------------
    # 3. Compute market_share_avg (excluding last day)
    # ----------------------------------------------------------
    oi_share_df.index = pd.to_datetime(oi_share_df.index)
    temp = oi_share_df.copy()
    temp["date"] = temp.index.date

    last_day = temp["date"].max()
    market_share_avg = {}

    for symbol in oi_share_df.columns:
        filtered = temp[temp["date"] != last_day][symbol]
        market_share_avg[symbol] = filtered.mean()

    market_share_avg = pd.Series(market_share_avg)

    # ----------------------------------------------------------
    # 4. Create oi_diff dataframe
    # ----------------------------------------------------------
    last_market_share = oi_share_df.iloc[-1]
    oi_diff = last_market_share - market_share_avg

    oi_diff_df = pd.DataFrame(
        {
            "symbol": oi_diff.index,
            "diff": oi_diff.values,
            "last_market_share": last_market_share.values,
            "avg_market_share": market_share_avg.values,
        }
    ).sort_values("diff", ascending=False).reset_index(drop=True)

    # ----------------------------------------------------------
    # 5. Top N and Bottom N symbols by difference
    # ----------------------------------------------------------
    st.subheader(
        f"Top {int(top_bottom_n)} and Bottom {int(top_bottom_n)} Symbols by OI Market Share Change"
    )

    col1, col2 = st.columns(2)

    top_n = oi_diff_df.nlargest(int(top_bottom_n), "diff")
    bottom_n = oi_diff_df.nsmallest(int(top_bottom_n), "diff")

    with col1:
        st.markdown(f"### Top {int(top_bottom_n)} (Increase in OI Share)")
        st.dataframe(top_n, use_container_width=True)

    with col2:
        st.markdown(f"### Bottom {int(top_bottom_n)} (Decrease in OI Share)")
        st.dataframe(bottom_n, use_container_width=True)

    # Optional: show the full diff dataframe
    with st.expander("Show full OI market share difference table"):
        st.dataframe(oi_diff_df, use_container_width=True)

else:
    st.info("Analysis not run. Click 'Run analysis / Refresh calculations' to fetch data and compute results.")
