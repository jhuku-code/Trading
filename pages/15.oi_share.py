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
    "Number of symbols to show",
    min_value=5,
    max_value=100,
    value=15,
    step=5,
)

# ✅ Interval selector
interval = st.sidebar.selectbox(
    "Select interval granularity",
    ["1min", "1hour", "4hour", "6hour", "12hour", "daily"],
    index=2,
)

# Cache clear
if st.sidebar.button("Clear cached API data"):
    st.cache_data.clear()
    st.sidebar.success("Cache cleared")

# ---------------------------------------------------------
# PAGE CONTROL
# ---------------------------------------------------------
if "run_analysis" not in st.session_state:
    st.session_state["run_analysis"] = False

col1, col2 = st.columns(2)

with col1:
    if st.button("Run analysis"):
        st.session_state["run_analysis"] = True

with col2:
    if st.button("Force refresh"):
        st.cache_data.clear()
        st.session_state["run_analysis"] = True

if st.button("Clear results"):
    st.session_state["run_analysis"] = False

st.info(f"Selected interval: {interval}")

# ---------------------------------------------------------
# LOAD SYMBOLS
# ---------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_symbols(url):
    resp = requests.get(url)
    resp.raise_for_status()
    df = pd.read_excel(io.BytesIO(resp.content))
    return df["Symbol"].dropna().astype(str).tolist()

try:
    symbols = load_symbols(GITHUB_EXCEL_URL)
except Exception as e:
    st.error(f"Error loading symbols: {e}")
    st.stop()

# ---------------------------------------------------------
# HELPER
# ---------------------------------------------------------
def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

# ---------------------------------------------------------
# FETCH DATA
# ---------------------------------------------------------
@st.cache_data(show_spinner=True)
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

            # ---------------- SUCCESS ----------------
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

                time.sleep(1)  # polite pause
                break

            # ---------------- RATE LIMIT FIX ----------------
            elif resp.status_code == 429:
                retry_raw = resp.headers.get("Retry-After", 30)

                try:
                    retry = float(retry_raw)  # ✅ handles "55.496"
                except:
                    retry = 30

                st.warning(f"Rate limited. Sleeping {retry:.2f}s")
                time.sleep(retry)

            # ---------------- ERROR ----------------
            else:
                st.error(f"Error {resp.status_code}: {resp.text}")
                break

    if not all_data:
        raise Exception("No data fetched")

    df = pd.concat(all_data)

    df_wide = (
        df.pivot(index="time", columns="symbol", values="oi")
        .sort_index()
    )

    return df_wide

# ---------------------------------------------------------
# RUN ANALYSIS
# ---------------------------------------------------------
if st.session_state["run_analysis"]:

    with st.spinner("Fetching data..."):

        try:
            oi_data = fetch_oi(symbols, months_back, interval, API_KEY)
        except Exception as e:
            st.error(e)
            st.stop()

    st.success("Data loaded")

    # ---------------- TOTAL OI ----------------
    st.subheader("Total OI")
    total_oi = oi_data.sum(axis=1)
    st.line_chart(total_oi)

    # ---------------- MARKET SHARE ----------------
    total = oi_data.sum(axis=1)
    share = oi_data.div(total, axis=0) * 100

    # ---------------- SINGLE SYMBOL ----------------
    st.subheader("Single Symbol Market Share")
    sym = st.selectbox("Select symbol", oi_data.columns)

    series = (oi_data[sym] / total) * 100
    st.line_chart(series)

    # ---------------- AVERAGE ----------------
    temp = share.copy()
    temp["date"] = temp.index.date
    last_day = temp["date"].max()

    avg = {
        c: temp[temp["date"] != last_day][c].mean()
        for c in share.columns
    }
    avg = pd.Series(avg)

    # ---------------- DIFF ----------------
    last = share.iloc[-1]
    diff = last - avg

    df = pd.DataFrame({
        "symbol": diff.index,
        "diff": diff.values,
        "last": last.values,
        "avg": avg.values
    }).sort_values("diff", ascending=False)

    # ---------------- TOP / BOTTOM ----------------
    st.subheader("Top / Bottom Movers")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Top")
        st.dataframe(df.head(top_bottom_n))

    with col2:
        st.write("Bottom")
        st.dataframe(df.tail(top_bottom_n))

else:
    st.info("Click Run analysis to start")
