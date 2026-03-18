import time
import requests
import pandas as pd
import numpy as np
import io
import streamlit as st
from datetime import datetime
from dateutil.relativedelta import relativedelta

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

GITHUB_USER = "jhuku-code"
GITHUB_REPO = "Trading"
BRANCH = "main"

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

months_back = st.sidebar.number_input(
    "Lookback window (months)",
    min_value=1,
    max_value=12,
    value=3,
    step=1,
)

top_bottom_n = st.sidebar.number_input(
    "Number of symbols to show in Top/Bottom lists",
    min_value=5,
    max_value=100,
    value=15,
    step=5,
)

# ✅ NEW: Interval Selectbox
interval = st.sidebar.selectbox(
    "Select interval granularity",
    options=["1min", "1hour", "4hour", "6hour", "12hour", "daily"],
    index=2,
)

# Cache clear
if st.sidebar.button("Clear cached API data"):
    st.cache_data.clear()
    st.sidebar.success("Cache cleared. Click 'Run analysis' to fetch fresh data.")

# ---------------------------------------------------------
# PAGE CONTROL
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

if st.button("Clear results (hide)"):
    st.session_state["run_analysis"] = False

st.write("Tip: The page won't fetch or compute OI until you click **Run analysis**.")
st.info(f"Using interval: {interval}")

# ---------------------------------------------------------
# LOAD SYMBOLS
# ---------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_symbols_from_github(url: str):
    resp = requests.get(url)
    resp.raise_for_status()
    df = pd.read_excel(io.BytesIO(resp.content))
    if "Symbol" not in df.columns:
        raise ValueError("Excel file must contain a 'Symbol' column.")
    return df["Symbol"].dropna().astype(str).tolist()

try:
    usdt_perp_symbols = load_symbols_from_github(GITHUB_EXCEL_URL)
except Exception as e:
    st.error(f"Error loading perps_list.xlsx from GitHub: {e}")
    st.stop()

if not usdt_perp_symbols:
    st.error("No symbols found in perps_list.xlsx on GitHub.")
    st.stop()

st.info(f"Loaded {len(usdt_perp_symbols)} perp symbols.")

# ---------------------------------------------------------
# HELPER
# ---------------------------------------------------------
def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]

# ---------------------------------------------------------
# FETCH DATA
# ---------------------------------------------------------
@st.cache_data(show_spinner=True)
def fetch_oi_data(symbols, months_back, api_key, interval):
    oi_url = f"{BASE_URL}/open-interest-history"

    from_timestamp = int((datetime.now() - relativedelta(months=int(months_back))).timestamp())
    to_timestamp = int(datetime.now().timestamp())

    all_rows = []

    for batch in chunked(symbols, 20):
        params = {
            "symbols": ",".join(batch),
            "interval": interval,  # ✅ dynamic
            "from": from_timestamp,
            "to": to_timestamp,
            "api_key": api_key,
        }

        while True:
            resp = requests.get(oi_url, params=params)

            if resp.status_code == 200:
                data = resp.json()

                for entry in data:
                    symbol = entry.get("symbol")
                    history = entry.get("history", [])

                    if not symbol or not history:
                        continue

                    df = pd.DataFrame(history)
                    df["time"] = pd.to_datetime(df["t"], unit="s")
                    df.rename(columns={"o": "open_interest"}, inplace=True)
                    df = df[["time", "open_interest"]]
                    df["symbol"] = symbol
                    all_rows.append(df)

                time.sleep(1)
                break

            elif resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 30))
                st.warning(f"Rate limited. Sleeping {retry_after}s")
                time.sleep(retry_after)

            else:
                st.error(f"Error: {resp.status_code}")
                break

    if not all_rows:
        raise Exception("No data received")

    df_long = pd.concat(all_rows, ignore_index=True)

    df_wide = (
        df_long.pivot_table(index="time", columns="symbol", values="open_interest")
        .sort_index()
    )

    cols_intersect = [c for c in symbols if c in df_wide.columns]
    df_wide = df_wide.loc[:, cols_intersect]

    return df_wide

# ---------------------------------------------------------
# RUN ANALYSIS
# ---------------------------------------------------------
if st.session_state["run_analysis"]:
    with st.spinner("Fetching data..."):
        try:
            oi_data = fetch_oi_data(
                usdt_perp_symbols,
                months_back,
                API_KEY,
                interval,  # ✅ passed here
            )

            st.session_state["oi_data"] = oi_data
            st.session_state["oi_last_update"] = datetime.now()

        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    st.success("OI data loaded successfully.")

    # TOTAL OI
    st.subheader("Total Open Interest Over Time")
    total_oi = oi_data.sum(axis=1)
    st.line_chart(total_oi)

    # MARKET SHARE
    oi_hist = oi_data.copy()
    total_oi_hist = oi_hist.sum(axis=1)

    oi_share_df = oi_hist.div(total_oi_hist, axis=0) * 100
    oi_share_df = oi_share_df.round(3)

    # SINGLE SYMBOL VIEW
    st.subheader("Single Symbol Market Share")
    selected_symbol = st.selectbox("Pick a symbol:", oi_hist.columns)

    symbol_series = (oi_hist[selected_symbol] / total_oi_hist) * 100
    st.line_chart(symbol_series)

    # AVG SHARE
    temp = oi_share_df.copy()
    temp["date"] = temp.index.date
    last_day = temp["date"].max()

    market_share_avg = {
        s: temp[temp["date"] != last_day][s].mean()
        for s in oi_share_df.columns
    }
    market_share_avg = pd.Series(market_share_avg)

    # DIFF
    last_market_share = oi_share_df.iloc[-1]
    oi_diff = last_market_share - market_share_avg

    oi_diff_df = pd.DataFrame({
        "symbol": oi_diff.index,
        "diff": oi_diff.values,
        "last_market_share": last_market_share.values,
        "avg_market_share": market_share_avg.values,
    }).sort_values("diff", ascending=False)

    # TOP / BOTTOM
    st.subheader(f"Top & Bottom {int(top_bottom_n)}")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Top Gainers")
        st.dataframe(oi_diff_df.head(int(top_bottom_n)))

    with col2:
        st.write("Top Losers")
        st.dataframe(oi_diff_df.tail(int(top_bottom_n)))

else:
    st.info("Click 'Run analysis' to start.")
