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
st.set_page_config(page_title="Funding Rate Z-Score Monitor", layout="wide")
st.title("Funding Rate Z-Score Monitor")

# ---------------------------------------------------------
# INPUTS / CONTROLS
# ---------------------------------------------------------
top_n = st.number_input(
    "Number of symbols to show in Top/Bottom lists",
    min_value=5,
    max_value=100,
    value=15,
    step=5
)

lookback_months = st.number_input(
    "Lookback window (months)",
    min_value=1,
    max_value=12,
    value=3,
    step=1
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
# READ PERPS LIST
# ---------------------------------------------------------
st.subheader("Perpetual Symbols Universe")

GITHUB_PERPS_URL = "https://raw.githubusercontent.com/jhuku-code/Trading/main/Input-Files/perps_list.xlsx"

@st.cache_data(ttl=3600)
def load_perps_list(url):
    return pd.read_excel(url)

try:
    df_perps = load_perps_list(GITHUB_PERPS_URL)
except Exception as e:
    st.error(f"Error loading perps list: {e}")
    st.stop()

if "Symbol" not in df_perps.columns:
    st.error("Symbol column missing in perps list")
    st.stop()

usdt_perp_symbols = df_perps["Symbol"].dropna().astype(str).tolist()

st.write(f"Total symbols loaded: {len(usdt_perp_symbols)}")

# ---------------------------------------------------------
# HELPER
# ---------------------------------------------------------
def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]

# ---------------------------------------------------------
# FETCH FUNDING DATA
# ---------------------------------------------------------
@st.cache_data(ttl=1800)
def fetch_funding_rates(symbols, months_back, api_key):

    fr_url = f"{BASE_URL}/funding-rate-history"

    from_timestamp = int((datetime.now() - relativedelta(months=months_back)).timestamp())
    to_timestamp = int(datetime.now().timestamp())

    all_rows = []

    for batch in chunked(symbols, 20):

        symbols_param = ",".join(batch)

        params = {
            "symbols": symbols_param,
            "interval": "4hour",
            "from": from_timestamp,
            "to": to_timestamp,
            "api_key": api_key,
        }

        while True:

            resp = requests.get(fr_url, params=params)

            if resp.status_code == 200:

                data = resp.json()

                if not isinstance(data, list):
                    break

                for entry in data:

                    symbol = entry.get("symbol")
                    history = entry.get("history", [])

                    if not symbol or not history:
                        continue

                    df = pd.DataFrame(history)

                    df["time"] = pd.to_datetime(df["t"], unit="s")

                    df.rename(columns={"o": "funding_rate"}, inplace=True)

                    df = df[["time", "funding_rate"]]

                    df["symbol"] = symbol

                    all_rows.append(df)

                time.sleep(1)
                break

            elif resp.status_code == 429:

                retry_after = resp.headers.get("Retry-After", 30)

                try:
                    retry_after = int(retry_after)
                except:
                    retry_after = 30

                st.warning(f"Rate limit hit. Sleeping {retry_after}s")

                time.sleep(retry_after)

            else:

                st.error(f"API error: {resp.status_code}")
                break

    if not all_rows:
        raise Exception("No funding data returned")

    df_long = pd.concat(all_rows, ignore_index=True)

    df_wide = df_long.pivot_table(
        index="time",
        columns="symbol",
        values="funding_rate"
    ).sort_index()

    df_wide = df_wide * 100.0

    return df_wide

# ---------------------------------------------------------
# LOAD OR FETCH DATA
# ---------------------------------------------------------
def get_funding_data(symbols, months_back, api_key):

    if "funding_data" in st.session_state and st.session_state["funding_data"] is not None:
        return st.session_state["funding_data"]

    with st.spinner("Fetching funding rates..."):

        df = fetch_funding_rates(symbols, months_back, api_key)

        st.session_state["funding_data"] = df
        st.session_state["funding_last_update"] = datetime.now()
        st.session_state["funding_symbols"] = list(df.columns)

        return df

# ---------------------------------------------------------
# REFRESH BUTTON
# ---------------------------------------------------------
if st.button("🔄 Refresh Data"):

    st.cache_data.clear()

    st.session_state.pop("funding_data", None)
    st.session_state.pop("funding_last_update", None)
    st.session_state.pop("funding_symbols", None)

    st.success("Cache cleared. Data will reload.")

# ---------------------------------------------------------
# GET DATA
# ---------------------------------------------------------
st.subheader("Funding Rate Data")

fr_data = get_funding_data(usdt_perp_symbols, lookback_months, API_KEY)

if fr_data is None or fr_data.empty:
    st.warning("No funding data available")
    st.stop()

st.write("Data shape:", fr_data.shape)
st.write("Latest timestamp:", fr_data.index.max())

if "funding_last_update" in st.session_state:
    st.write("Last fetched:", st.session_state["funding_last_update"])

# ---------------------------------------------------------
# Z-SCORES
# ---------------------------------------------------------
fr_zscores = (fr_data - fr_data.mean()) / fr_data.std()

latest_z = fr_zscores.iloc[-1]
latest_rates = fr_data.iloc[-1]

sorted_z = latest_z.sort_values(ascending=False)

top_symbols = sorted_z.head(top_n).index
bottom_symbols = sorted_z.tail(top_n).index

# ---------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------
st.subheader("Top / Bottom Symbols")

tab1, tab2 = st.tabs(["Z-Score Ranking", "Raw Funding Rates"])

# ---------------- TAB 1 ----------------
with tab1:

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Top Z-Score")

        df_top = pd.DataFrame({
            "Symbol": top_symbols,
            "ZScore": latest_z[top_symbols],
            "Funding Rate (%)": latest_rates[top_symbols]
        }).set_index("Symbol")

        st.dataframe(df_top.style.format({
            "ZScore":"{:.2f}",
            "Funding Rate (%)":"{:.4f}"
        }))

    with col2:

        st.markdown("### Bottom Z-Score")

        df_bottom = pd.DataFrame({
            "Symbol": bottom_symbols,
            "ZScore": latest_z[bottom_symbols],
            "Funding Rate (%)": latest_rates[bottom_symbols]
        }).set_index("Symbol")

        st.dataframe(df_bottom.style.format({
            "ZScore":"{:.2f}",
            "Funding Rate (%)":"{:.4f}"
        }))

# ---------------- TAB 2 ----------------
with tab2:

    col1, col2 = st.columns(2)

    sorted_rates = latest_rates.sort_values(ascending=False)

    with col1:

        st.markdown("### Highest Funding")

        st.dataframe(
            sorted_rates.head(top_n).to_frame("Funding Rate (%)")
            .style.format("{:.4f}")
        )

    with col2:

        st.markdown("### Lowest Funding")

        st.dataframe(
            sorted_rates.tail(top_n).to_frame("Funding Rate (%)")
            .style.format("{:.4f}")
        )

# ---------------------------------------------------------
# SYMBOL FUNDING CHART
# ---------------------------------------------------------
st.subheader("Funding Rate Time Series")

symbol = st.selectbox("Select symbol", fr_data.columns)

series = fr_data[symbol].dropna()

ma30 = series.rolling(30).mean()

df_plot = pd.DataFrame({
    "Funding Rate (%)": series,
    "MA30": ma30
})

st.line_chart(df_plot)
