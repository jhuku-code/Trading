import time
import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from dateutil.relativedelta import relativedelta

# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Funding Rate Z-Score Monitor", layout="wide")
st.title("Funding Rate Z-Score Monitor")

# ---------------------------------------------------------
# SIDEBAR CONTROLS (UPDATED)
# ---------------------------------------------------------
st.sidebar.header("Settings")

top_n = st.sidebar.number_input("Top/Bottom N", 5, 100, 15)
lookback_months = st.sidebar.number_input("Lookback (months)", 1, 12, 3)

# ✅ NEW INTERVAL SELECTOR
interval = st.sidebar.selectbox(
    "Interval",
    ["1min", "1hour", "4hour", "6hour", "12hour", "daily"],
    index=2,
)

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
API_KEY = st.secrets.get("API_KEY", "")
BASE_URL = "https://api.coinalyze.net/v1"

if not API_KEY:
    st.error("API_KEY not found in Streamlit secrets")
    st.stop()

# ---------------------------------------------------------
# LOAD SYMBOLS
# ---------------------------------------------------------
GITHUB_PERPS_URL = "https://raw.githubusercontent.com/jhuku-code/Trading/main/Input-Files/perps_list.xlsx"

@st.cache_data(ttl=3600)
def load_perps_list(url):
    return pd.read_excel(url)

df_perps = load_perps_list(GITHUB_PERPS_URL)
symbols = df_perps["Symbol"].dropna().astype(str).tolist()

st.write(f"Total symbols loaded: {len(symbols)}")

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

# ---------------------------------------------------------
# FETCH FUNDING DATA (UPDATED)
# ---------------------------------------------------------
@st.cache_data(ttl=1800)
def fetch_funding_rates(symbols, months_back, interval):

    url = f"{BASE_URL}/funding-rate-history"

    from_ts = int((datetime.now() - relativedelta(months=months_back)).timestamp())
    to_ts = int(datetime.now().timestamp())

    rows = []

    for batch in chunked(symbols, 20):

        params = {
            "symbols": ",".join(batch),
            "interval": interval,  # ✅ dynamic
            "from": from_ts,
            "to": to_ts,
            "api_key": API_KEY,
        }

        while True:
            r = requests.get(url, params=params)

            if r.status_code == 200:
                data = r.json()

                for entry in data:
                    sym = entry.get("symbol")
                    hist = entry.get("history", [])

                    if not hist:
                        continue

                    df = pd.DataFrame(hist)
                    df["time"] = pd.to_datetime(df["t"], unit="s")
                    df.rename(columns={"o": "funding_rate"}, inplace=True)
                    df["symbol"] = sym

                    rows.append(df[["time", "symbol", "funding_rate"]])

                time.sleep(1)
                break

            elif r.status_code == 429:
                retry_raw = r.headers.get("Retry-After", 30)
                try:
                    retry = float(retry_raw)  # ✅ FIX
                except:
                    retry = 30
                time.sleep(retry)

            else:
                break

    if not rows:
        raise Exception("No data fetched")

    df = pd.concat(rows)

    df_wide = df.pivot(index="time", columns="symbol", values="funding_rate").sort_index()

    return df_wide * 100.0

# ---------------------------------------------------------
# BUTTONS
# ---------------------------------------------------------
col1, col2 = st.columns(2)

run_clicked = False

with col1:
    if st.button("🔄 Refresh Data"):
        run_clicked = True

with col2:
    if st.button("Force Refresh"):
        st.cache_data.clear()
        st.session_state.pop("funding_data", None)
        run_clicked = True

# ---------------------------------------------------------
# DATA FETCH LOGIC (PERSISTENCE FIX)
# ---------------------------------------------------------
params = (lookback_months, interval)

if run_clicked:
    with st.spinner("Fetching funding data..."):
        df = fetch_funding_rates(symbols, lookback_months, interval)

        st.session_state["funding_data"] = df
        st.session_state["funding_params"] = params
        st.session_state["funding_last_update"] = datetime.now()

# ---------------------------------------------------------
# USE EXISTING DATA
# ---------------------------------------------------------
if "funding_data" not in st.session_state:
    st.info("Click Refresh Data to load")
    st.stop()

if st.session_state.get("funding_params") != params:
    st.warning("Settings changed — click Refresh to update data")

fr_data = st.session_state["funding_data"]

st.success("Using cached data")

st.write("Data shape:", fr_data.shape)
st.write("Latest timestamp:", fr_data.index.max())
st.write("Last fetched:", st.session_state.get("funding_last_update"))

# ---------------------------------------------------------
# Z-SCORE CALC
# ---------------------------------------------------------
fr_z = (fr_data - fr_data.mean()) / fr_data.std()

latest_z = fr_z.iloc[-1]
latest_rates = fr_data.iloc[-1]

sorted_z = latest_z.sort_values(ascending=False)

top_symbols = sorted_z.head(top_n).index
bottom_symbols = sorted_z.tail(top_n).index

# ---------------------------------------------------------
# DISPLAY
# ---------------------------------------------------------
st.subheader("Top / Bottom Symbols")

tab1, tab2 = st.tabs(["Z-Score", "Funding Rates"])

with tab1:

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Top Z-Score")
        st.dataframe(pd.DataFrame({
            "ZScore": latest_z[top_symbols],
            "Funding (%)": latest_rates[top_symbols]
        }).style.format("{:.4f}"))

    with col2:
        st.markdown("### Bottom Z-Score")
        st.dataframe(pd.DataFrame({
            "ZScore": latest_z[bottom_symbols],
            "Funding (%)": latest_rates[bottom_symbols]
        }).style.format("{:.4f}"))

with tab2:

    sorted_rates = latest_rates.sort_values(ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Highest Funding")
        st.dataframe(sorted_rates.head(top_n).to_frame("Funding (%)").style.format("{:.4f}"))

    with col2:
        st.markdown("### Lowest Funding")
        st.dataframe(sorted_rates.tail(top_n).to_frame("Funding (%)").style.format("{:.4f}"))

# ---------------------------------------------------------
# TIME SERIES
# ---------------------------------------------------------
st.subheader("Funding Rate Time Series")

sym = st.selectbox("Select symbol", fr_data.columns)

series = fr_data[sym].dropna()
ma30 = series.rolling(30).mean()

st.line_chart(pd.DataFrame({
    "Funding (%)": series,
    "MA30": ma30
}))
