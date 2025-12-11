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
# Number of symbols in top/bottom lists (instead of fixed 15/20)
top_n = st.number_input(
    "Number of symbols to show in Top/Bottom lists",
    min_value=5,
    max_value=100,
    value=15,
    step=5,
    key="top_n",
)

# Lookback months (optional UI; default 3 months as your code)
lookback_months = st.number_input(
    "Lookback window (months)",
    min_value=1,
    max_value=12,
    value=3,
    step=1,
    key="lookback_months",
)

# ---------------------------------------------------------
# CONFIG: API KEY & BASE URL
# ---------------------------------------------------------
# Read API key securely from Streamlit secrets (.streamlit/secrets.toml)
API_KEY = st.secrets.get("API_KEY", "")
BASE_URL = "https://api.coinalyze.net/v1"

if not API_KEY:
    st.error("API_KEY not found in Streamlit secrets. Please add it to .streamlit/secrets.toml")
    st.stop()

# ---------------------------------------------------------
# READ PERPS LIST FROM GITHUB
# ---------------------------------------------------------
st.subheader("Perpetual Symbols Universe")
GITHUB_PERPS_URL = (
    "https://raw.githubusercontent.com/jhuku-code/Trading/main/Input-Files/perps_list.xlsx"
)

@st.cache_data(ttl=3600)
def load_perps_list(url: str) -> pd.DataFrame:
    return pd.read_excel(url)

try:
    df_perps = load_perps_list(GITHUB_PERPS_URL)
    st.write("Loaded symbols from GitHub `Input-Files/perps_list.xlsx`")
except Exception as e:
    st.error(f"Error loading perps_list.xlsx from GitHub: {e}")
    st.stop()

if "Symbol" not in df_perps.columns:
    st.error("Column 'Symbol' not found in perps_list.xlsx")
    st.stop()

usdt_perp_symbols = df_perps["Symbol"].dropna().astype(str).tolist()
st.write(f"Total symbols loaded: {len(usdt_perp_symbols)}")

# ---------------------------------------------------------
# HELPER: CHUNK LIST
# ---------------------------------------------------------
def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]

# ---------------------------------------------------------
# FETCH FUNDING RATE HISTORY (CACHED)
# ---------------------------------------------------------
@st.cache_data(ttl=1800)
def fetch_funding_rates_cached(symbols, months_back: int, api_key: str) -> pd.DataFrame:
    """This function is cached by streamlit so repeated calls with same args use cache."""
    fr_url = f"{BASE_URL}/funding-rate-history"

    from_timestamp = int((datetime.now() - relativedelta(months=months_back)).timestamp())
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

    df_wide = (
        df_long.pivot_table(index="time", columns="symbol", values="funding_rate")
        .sort_index()
    )

    # convert to percentage
    df_wide = df_wide * 100.0
    return df_wide

# ---------------------------------------------------------
# SESSION-STATE CONTROLLED FETCH
# ---------------------------------------------------------
def get_data_from_session_or_fetch(symbols, months_back, api_key):
    """Return DataFrame from session_state if present, otherwise fetch and store it."""
    if "fr_data" in st.session_state and st.session_state.get("fr_data") is not None:
        return st.session_state["fr_data"]

    # not present -> fetch now (first load)
    with st.spinner("Fetching funding rate history from Coinalyze API..."):
        try:
            df = fetch_funding_rates_cached(symbols, months_back, api_key)
            st.session_state["fr_data"] = df
            st.session_state["fr_last_update"] = datetime.now()
            return df
        except Exception as e:
            st.error(f"Error fetching funding rate data: {e}")
            st.session_state["fr_data"] = None
            st.stop()

# Refresh button: clear cache & session data then fetch again
if st.button("🔄 Refresh Data"):
    # Clear streamlit cache for the fetch function
    try:
        st.cache_data.clear()
    except Exception:
        # older streamlit compatibility, ignore if not available
        pass

    # Remove session stored data so new fetch occurs
    st.session_state.pop("fr_data", None)
    st.session_state.pop("fr_last_update", None)

    # Immediately fetch and store new data (gives user immediate feedback)
    with st.spinner("Refreshing funding rate data (this may take a while)..."):
        try:
            fr_data = fetch_funding_rates_cached(usdt_perp_symbols, lookback_months, API_KEY)
            st.session_state["fr_data"] = fr_data
            st.session_state["fr_last_update"] = datetime.now()
            st.success("Refresh complete.")
        except Exception as e:
            st.error(f"Refresh failed: {e}")
            st.session_state["fr_data"] = None

# ---------------------------------------------------------
# MAIN DATA FETCH + CALCULATIONS (use cached session value)
# ---------------------------------------------------------
st.subheader("Funding Rate Data")
fr_data = get_data_from_session_or_fetch(usdt_perp_symbols, lookback_months, API_KEY)

if fr_data is None or fr_data.empty:
    st.warning("No funding rate data available.")
    st.stop()

st.write("Data shape:", fr_data.shape)
st.write("Latest timestamp in data:", fr_data.index.max())
if "fr_last_update" in st.session_state:
    st.write("Last fetched:", st.session_state["fr_last_update"].strftime("%Y-%m-%d %H:%M:%S"))

# ---------------------------------------------------------
# Z-SCORES
# ---------------------------------------------------------
fr_zscores = (fr_data - fr_data.mean(numeric_only=True)) / fr_data.std(numeric_only=True)

latest_zscores = fr_zscores.iloc[-1]
latest_rates = fr_data.iloc[-1]

# Sort Z-scores
sorted_zscores = latest_zscores.sort_values(ascending=False)

top_symbols_z = sorted_zscores.head(top_n).index
bottom_symbols_z = sorted_zscores.tail(top_n).index

top_z_list = [(symbol, float(latest_zscores[symbol]), float(latest_rates[symbol])) for symbol in top_symbols_z]
bottom_z_list = [(symbol, float(latest_zscores[symbol]), float(latest_rates[symbol])) for symbol in bottom_symbols_z]

# ---------------------------------------------------------
# RAW FUNDING RATE RANKING
# ---------------------------------------------------------
sorted_rates = latest_rates.sort_values(ascending=False)

top_rates_series = sorted_rates.head(top_n)
bottom_rates_series = sorted_rates.tail(top_n)

top_rates_list = list(top_rates_series.items())
bottom_rates_list = list(bottom_rates_series.items())

# ---------------------------------------------------------
# DISPLAY RESULTS IN STREAMLIT
# ---------------------------------------------------------
st.subheader("Top / Bottom Symbols")

tab1, tab2 = st.tabs(["By Z-Score Deviation", "By Raw Funding Rate"])

# ---------- TAB 1: Z-SCORES ----------
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### Top {top_n} by Z-Score (High Positive Deviation)")
        df_top_z = pd.DataFrame(
            top_z_list,
            columns=["Symbol", "Z-Score", "Funding Rate (%)"],
        ).set_index("Symbol")
        st.dataframe(df_top_z.style.format({"Z-Score": "{:.2f}", "Funding Rate (%)": "{:.4f}"}))

    with col2:
        st.markdown(f"### Bottom {top_n} by Z-Score (High Negative Deviation)")
        df_bottom_z = pd.DataFrame(
            bottom_z_list,
            columns=["Symbol", "Z-Score", "Funding Rate (%)"],
        ).set_index("Symbol")
        st.dataframe(df_bottom_z.style.format({"Z-Score": "{:.2f}", "Funding Rate (%)": "{:.4f}"}))

# ---------- TAB 2: RAW FUNDING RATES ----------
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### Top {top_n} Funding Rates (Highest)")
        df_top_rates = pd.DataFrame(
            top_rates_list,
            columns=["Symbol", "Funding Rate (%)"],
        ).set_index("Symbol")
        st.dataframe(df_top_rates.style.format({"Funding Rate (%)": "{:.4f}"}))

    with col2:
        st.markdown(f"### Bottom {top_n} Funding Rates (Lowest / Most Negative)")
        df_bottom_rates = pd.DataFrame(
            bottom_rates_list,
            columns=["Symbol", "Funding Rate (%)"],
        ).set_index("Symbol")
        st.dataframe(df_bottom_rates.style.format({"Funding Rate (%)": "{:.4f}"}))

# Optional: Show the full latest funding rate snapshot
with st.expander("Show full latest funding rate snapshot (all symbols)"):
    st.dataframe(latest_rates.to_frame("Funding Rate (%)").sort_values("Funding Rate (%)", ascending=False))
