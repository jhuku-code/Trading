import time
import requests
import pandas as pd
import numpy as np
import io
import streamlit as st
from datetime import datetime
from dateutil.relativedelta import relativedelta

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="OI Market Share Dashboard", layout="wide")
st.title("Open Interest Market Share Dashboard")

# ---------------------------------------------------------
# CONSTANTS
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
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("Settings")

months_back = st.sidebar.number_input("Lookback (months)", 1, 12, 3)
top_bottom_n = st.sidebar.number_input("Top/Bottom N", 5, 100, 15)

interval = st.sidebar.selectbox(
    "Interval",
    ["1min", "1hour", "4hour", "6hour", "12hour", "daily"],
    index=2,
)

# ---------------------------------------------------------
# BUTTONS
# ---------------------------------------------------------
col1, col2 = st.columns(2)

run_clicked = False

with col1:
    if st.button("Run analysis"):
        run_clicked = True

with col2:
    if st.button("Force refresh"):
        st.cache_data.clear()
        st.session_state.pop("oi_data", None)
        run_clicked = True

if st.sidebar.button("Clear cache"):
    st.cache_data.clear()

# ---------------------------------------------------------
# LOAD SYMBOLS
# ---------------------------------------------------------
@st.cache_data
def load_symbols(url):
    resp = requests.get(url)
    resp.raise_for_status()
    df = pd.read_excel(io.BytesIO(resp.content))
    return df["Symbol"].dropna().astype(str).tolist()

symbols = load_symbols(GITHUB_EXCEL_URL)

# ---------------------------------------------------------
# HELPER
# ---------------------------------------------------------
def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

# ---------------------------------------------------------
# FETCH
# ---------------------------------------------------------
@st.cache_data
def fetch_oi(symbols, months_back, interval, api_key):

    url = f"{BASE_URL}/open-interest-history"

    from_ts = int((datetime.now() - relativedelta(months=months_back)).timestamp())
    to_ts = int(datetime.now().timestamp())

    all_data = []

    for batch in chunked(symbols, 20):

        params = {
            "symbols": ",".join(batch),
            "interval": interval,
            "from": from_ts,
            "to": to_ts,
            "api_key": api_key,
        }

        while True:
            resp = requests.get(url, params=params)

            if resp.status_code == 200:
                data = resp.json()

                for entry in data:
                    symbol = entry.get("symbol")
                    history = entry.get("history", [])

                    if not history:
                        continue

                    df = pd.DataFrame(history)
                    df["time"] = pd.to_datetime(df["t"], unit="s")
                    df["symbol"] = symbol
                    df.rename(columns={"o": "oi"}, inplace=True)

                    all_data.append(df[["time", "symbol", "oi"]])

                time.sleep(1)
                break

            elif resp.status_code == 429:
                retry_raw = resp.headers.get("Retry-After", 30)
                try:
                    retry = float(retry_raw)
                except:
                    retry = 30
                time.sleep(retry)

            else:
                break

    if not all_data:
        raise Exception("No data fetched")

    df = pd.concat(all_data)

    return df.pivot(index="time", columns="symbol", values="oi").sort_index()

# ---------------------------------------------------------
# DATA LOAD LOGIC (KEY FIX)
# ---------------------------------------------------------
if run_clicked:
    with st.spinner("Fetching data..."):
        st.session_state["oi_data"] = fetch_oi(
            symbols, months_back, interval, API_KEY
        )
        st.session_state["last_params"] = (months_back, interval)

# ---------------------------------------------------------
# USE EXISTING DATA IF AVAILABLE
# ---------------------------------------------------------
if "oi_data" not in st.session_state:
    st.info("Click Run analysis to fetch data")
    st.stop()

# OPTIONAL: detect parameter change
if "last_params" in st.session_state:
    if st.session_state["last_params"] != (months_back, interval):
        st.warning("Settings changed — click Run to refresh data")

oi_data = st.session_state["oi_data"]

# ---------------------------------------------------------
# ANALYSIS
# ---------------------------------------------------------
st.success("Using cached data")

# TOTAL OI
st.subheader("Total OI")
total_oi = oi_data.sum(axis=1)
st.line_chart(total_oi)

# MARKET SHARE
total = oi_data.sum(axis=1)
share = oi_data.div(total, axis=0) * 100

# SINGLE SYMBOL
st.subheader("Single Symbol Market Share")
sym = st.selectbox("Select symbol", oi_data.columns)

series = (oi_data[sym] / total) * 100
st.line_chart(series)

# AVG
temp = share.copy()
temp["date"] = temp.index.date
last_day = temp["date"].max()

avg = {
    c: temp[temp["date"] != last_day][c].mean()
    for c in share.columns
}
avg = pd.Series(avg)

# DIFF
last = share.iloc[-1]
diff = last - avg

df = pd.DataFrame({
    "symbol": diff.index,
    "diff": diff.values,
    "last": last.values,
    "avg": avg.values
}).sort_values("diff", ascending=False)

# TOP / BOTTOM
st.subheader("Top / Bottom Movers")

col1, col2 = st.columns(2)

with col1:
    st.write("Top")
    st.dataframe(df.head(top_bottom_n))

with col2:
    st.write("Bottom")
    st.dataframe(df.tail(top_bottom_n))
