import time
import requests
import pandas as pd
import numpy as np
import io
import streamlit as st
from datetime import datetime
from dateutil.relativedelta import relativedelta

# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Open Interest Dashboard", layout="wide")
st.title("Open Interest Market Share Dashboard")

API_KEY = st.secrets.get("API_KEY")
BASE_URL = "https://api.coinalyze.net/v1"

GITHUB_URL = "https://raw.githubusercontent.com/jhuku-code/Trading/main/Input-Files/perps_list.xlsx"

# ---------------------------------------------------------
# USER INPUTS
# ---------------------------------------------------------
months_back = st.sidebar.number_input(
    "Lookback window (months)", 1, 12, 3
)

top_n = st.sidebar.number_input(
    "Top / Bottom symbols", 5, 100, 15
)

# ---------------------------------------------------------
# LOAD SYMBOLS
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def load_symbols(url):
    resp = requests.get(url)
    resp.raise_for_status()

    df = pd.read_excel(io.BytesIO(resp.content))

    if "Symbol" not in df.columns:
        raise ValueError("Symbol column missing")

    return df["Symbol"].dropna().astype(str).tolist()

symbols = load_symbols(GITHUB_URL)

st.write(f"Total symbols loaded: {len(symbols)}")

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

# ---------------------------------------------------------
# FETCH OPEN INTEREST
# ---------------------------------------------------------
@st.cache_data(ttl=1800)
def fetch_oi(symbols, months_back):

    url = f"{BASE_URL}/open-interest-history"

    from_ts = int((datetime.now() - relativedelta(months=months_back)).timestamp())
    to_ts = int(datetime.now().timestamp())

    rows = []

    for batch in chunked(symbols, 20):

        params = {
            "symbols": ",".join(batch),
            "interval": "4hour",
            "from": from_ts,
            "to": to_ts,
            "api_key": API_KEY
        }

        resp = requests.get(url, params=params)

        if resp.status_code != 200:
            continue

        data = resp.json()

        for entry in data:

            symbol = entry.get("symbol")
            history = entry.get("history", [])

            if not history:
                continue

            df = pd.DataFrame(history)

            df["time"] = pd.to_datetime(df["t"], unit="s")

            df.rename(columns={"o": "oi"}, inplace=True)

            df = df[["time", "oi"]]
            df["symbol"] = symbol

            rows.append(df)

        time.sleep(1)

    if not rows:
        raise Exception("No OI data returned")

    df = pd.concat(rows)

    df = df.pivot_table(
        index="time",
        columns="symbol",
        values="oi"
    ).sort_index()

    return df

# ---------------------------------------------------------
# LOAD DATA FROM SESSION OR FETCH
# ---------------------------------------------------------
if "oi_data" in st.session_state:

    oi_data = st.session_state["oi_data"]

else:

    if st.button("Load Open Interest Data"):

        with st.spinner("Downloading OI data..."):

            oi_data = fetch_oi(symbols, months_back)

            st.session_state["oi_data"] = oi_data
            st.session_state["oi_last_update"] = datetime.now()

    else:

        st.info("Click 'Load Open Interest Data'")
        st.stop()

# ---------------------------------------------------------
# DATA SUMMARY
# ---------------------------------------------------------
st.subheader("Dataset Info")

st.write("Data shape:", oi_data.shape)
st.write("Latest timestamp:", oi_data.index.max())

if "oi_last_update" in st.session_state:
    st.write("Last fetched:", st.session_state["oi_last_update"])

# ---------------------------------------------------------
# TOTAL OPEN INTEREST
# ---------------------------------------------------------
st.subheader("Total Open Interest")

total_oi = oi_data.sum(axis=1)

st.line_chart(total_oi)

# ---------------------------------------------------------
# MARKET SHARE
# ---------------------------------------------------------
total_oi_safe = total_oi.replace(0, np.nan)

oi_share = oi_data.div(total_oi_safe, axis=0) * 100

latest_share = oi_share.iloc[-1]

avg_share = oi_share.iloc[:-6].mean()

share_diff = latest_share - avg_share

share_table = pd.DataFrame({
    "Latest Share (%)": latest_share,
    "Average Share (%)": avg_share,
    "Difference": share_diff
}).sort_values("Difference", ascending=False)

# ---------------------------------------------------------
# TOP / BOTTOM TABLES
# ---------------------------------------------------------
st.subheader("Market Share Change")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Top Increase")
    st.dataframe(share_table.head(top_n))

with col2:
    st.markdown("### Top Decrease")
    st.dataframe(share_table.tail(top_n))

# ---------------------------------------------------------
# SYMBOL MARKET SHARE CHART
# ---------------------------------------------------------
st.subheader("Symbol Market Share Time Series")

symbol = st.selectbox(
    "Select symbol",
    sorted(oi_data.columns)
)

series = oi_share[symbol].dropna()

if series.empty:
    st.warning(f"No market share data for {symbol}")
else:

    chart_df = pd.DataFrame({
        f"{symbol} Market Share (%)": series
    })

    st.line_chart(chart_df)

    st.write(
        f"Latest market share for {symbol}: **{series.iloc[-1]:.3f}%**"
    )

# ---------------------------------------------------------
# OPTIONAL FULL TABLE
# ---------------------------------------------------------
with st.expander("Show full market share table"):

    st.dataframe(
        oi_share.iloc[-1]
        .sort_values(ascending=False)
        .to_frame("Market Share (%)")
    )
