# app.py

import time
from typing import List, Dict

import requests
import pandas as pd
import streamlit as st
from scipy.stats import zscore

# -----------------------------
# Streamlit page configuration
# -----------------------------
st.set_page_config(
    page_title="Binance Funding Rates Dashboard",
    layout="wide"
)

st.title("📊 Binance Funding Rates Dashboard")


# -----------------------------
# Binance funding rate collector
# -----------------------------
class BinanceFundingRateCollector:
    def __init__(self):
        self.base_url = "https://data.binance.com"
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'Mozilla/5.0'
        })

    def get_funding_rate_history(
        self, symbol: str, start_time: int, end_time: int, limit: int = 1000
    ) -> List[Dict]:
        url = f"{self.base_url}/fapi/v1/fundingRate"
        params = {
            "symbol": symbol,
            "startTime": start_time,
            "endTime": end_time,
            "limit": limit
        }

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.warning(f"Error fetching data for {symbol}: {e}")
            return []

    def get_all_funding_rates(self, symbol: str, days_back: int = 30) -> List[Dict]:
        end_time = int(time.time() * 1000)
        start_time = end_time - (days_back * 24 * 60 * 60 * 1000)

        all_data = []
        current_start = start_time

        while current_start < end_time:
            batch_end = min(current_start + (333 * 24 * 60 * 60 * 1000), end_time)
            batch_data = self.get_funding_rate_history(symbol, current_start, batch_end, 1000)

            if not batch_data:
                break

            all_data.extend(batch_data)

            if len(batch_data) < 1000:
                break

            current_start = batch_data[-1]['fundingTime'] + 1
            time.sleep(0.1)

        return all_data

    def process_funding_data(self, raw_data: List[Dict]) -> pd.DataFrame:
        if not raw_data:
            return pd.DataFrame()

        df = pd.DataFrame(raw_data)
        df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
        df['fundingRate'] = df['fundingRate'].astype(float)
        df = df.sort_values('fundingTime').reset_index(drop=True)
        df['fundingRatePercent'] = df['fundingRate'] * 100
        df['annualizedRate'] = df['fundingRate'] * 365 * 3
        df['annualizedRatePercent'] = df['annualizedRate'] * 100

        return df

    def get_active_futures_symbols(self) -> List[str]:
        """
        Try to get active USDT perpetual symbols from Binance.
        May fail with 451 in some regions.
        """
        url = f"{self.base_url}/fapi/v1/exchangeInfo"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()

            symbols = []
            for symbol_info in data['symbols']:
                if (
                    symbol_info['status'] == 'TRADING'
                    and symbol_info['contractType'] == 'PERPETUAL'
                    and symbol_info['symbol'].endswith('USDT')
                ):
                    symbols.append(symbol_info['symbol'])

            return sorted(symbols)
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 451:
                st.warning(
                    "Binance API returned **451 (Unavailable for legal reasons)**. "
                    "This usually means futures data is restricted in your region. "
                    "Use the *Custom symbol list* mode instead."
                )
            else:
                st.error(f"HTTP error fetching symbols: {e}")
            return []
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching symbols: {e}")
            return []


# -----------------------------
# Core analysis functions
# -----------------------------
def analyze_annualized_rate_percent(
    symbols: List[str],
    days_back: int = 90,
    outlier_method: str = "iqr"
):
    collector = BinanceFundingRateCollector()

    if not symbols:
        st.error("No symbols provided. Please add symbols in the sidebar.")
        return None, None, None, None

    all_data = []
    progress = st.progress(0)
    for i, symbol in enumerate(symbols):
        try:
            raw_data = collector.get_all_funding_rates(symbol, days_back)
            if raw_data:
                df = collector.process_funding_data(raw_data)
                df['symbol'] = symbol
                all_data.append(df)
        except Exception as e:
            st.write(f"Skipping {symbol}: {e}")

        progress.progress((i + 1) / len(symbols))

    if not all_data:
        st.error("No data collected from Binance for the given symbols.")
        return None, None, None, None

    combined_df = pd.concat(all_data, ignore_index=True)

    # Outlier removal
    if outlier_method == "iqr":
        q1 = combined_df['annualizedRatePercent'].quantile(0.25)
        q3 = combined_df['annualizedRatePercent'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        filtered_df = combined_df[
            (combined_df['annualizedRatePercent'] >= lower_bound)
            & (combined_df['annualizedRatePercent'] <= upper_bound)
        ]
    else:
        filtered_df = combined_df.copy()

    # Latest value per symbol
    latest_df = (
        filtered_df.sort_values("fundingTime")
        .groupby("symbol")
        .tail(1)
        .set_index("symbol")
    )

    # Z-score on latest annualizedRatePercent cross-section
    latest_df['zscore_annualized'] = zscore(latest_df['annualizedRatePercent'])

    top_10 = latest_df.sort_values("zscore_annualized", ascending=False).head(10)
    bottom_10 = latest_df.sort_values("zscore_annualized", ascending=True).head(10)

    return filtered_df, latest_df, top_10, bottom_10


def reshape_funding_data_by_hour(filtered_data: pd.DataFrame) -> pd.DataFrame:
    if filtered_data is None or filtered_data.empty:
        raise ValueError("Input DataFrame is empty.")

    filtered_data = filtered_data.copy()
    filtered_data['fundingTime'] = pd.to_datetime(filtered_data['fundingTime'])
    filtered_data['fundingHour'] = filtered_data['fundingTime'].dt.floor('H')

    grouped = (
        filtered_data
        .groupby(['fundingHour', 'symbol'])['annualizedRatePercent']
        .mean()
        .reset_index()
    )

    pivot_df = grouped.pivot(index='fundingHour', columns='symbol', values='annualizedRatePercent')
    pivot_df = pivot_df.sort_index().sort_index(axis=1)
    return pivot_df


def compute_latest_zscores_and_highlight(df_hourly_annualized: pd.DataFrame):
    def col_z(col: pd.Series) -> pd.Series:
        if col.notna().sum() < 2:
            return pd.Series([pd.NA] * len(col), index=col.index)
        z = zscore(col, nan_policy='omit')
        return pd.Series(z, index=col.index)

    zscore_df = df_hourly_annualized.apply(col_z, axis=0)

    latest_zscores = zscore_df.ffill().iloc[-1]

    high_threshold = 0.9
    low_threshold = -0.9

    top_coins = latest_zscores[latest_zscores > high_threshold].sort_values(ascending=False)
    bottom_coins = latest_zscores[latest_zscores < low_threshold].sort_values()

    return latest_zscores, top_coins, bottom_coins


def run_full_analysis(symbols: List[str], days_back: int, outlier_method: str):
    filtered_data, latest_data, top_10_z, bottom_10_z = analyze_annualized_rate_percent(
        symbols=symbols,
        days_back=days_back,
        outlier_method=outlier_method
    )

    if filtered_data is None or latest_data is None:
        return None

    hourly_annualized_df = reshape_funding_data_by_hour(filtered_data)
    latest_z, top_z, bottom_z = compute_latest_zscores_and_highlight(hourly_annualized_df)

    return {
        "filtered_data": filtered_data,
        "latest_data": latest_data,
        "top_10_z": top_10_z,
        "bottom_10_z": bottom_10_z,
        "hourly_annualized_df": hourly_annualized_df,
        "latest_z": latest_z,
        "top_z": top_z,
        "bottom_z": bottom_z,
    }


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Settings")

days_back = st.sidebar.slider(
    "Days back",
    min_value=7,
    max_value=180,
    value=90,
    step=1,
    help="How many days of historical funding data to fetch."
)

outlier_method = st.sidebar.selectbox(
    "Outlier method",
    options=["iqr", "none"],
    index=0,
    help="Method for removing outliers on annualized funding rates."
)

symbol_source = st.sidebar.radio(
    "Symbol source",
    options=["Custom list", "Binance API"],
    index=0,
    help="Use a custom list to avoid Binance geo restrictions."
)

default_symbols_text = "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT"

custom_symbols_text = st.sidebar.text_area(
    "Custom futures symbols (comma-separated)",
    value=default_symbols_text,
    help="Example: BTCUSDT,ETHUSDT,SOLUSDT"
)

refresh_button = st.sidebar.button("🔄 Refresh data")


# -----------------------------
# Determine symbol list
# -----------------------------
collector_for_symbols = BinanceFundingRateCollector()

if symbol_source == "Binance API":
    api_symbols = collector_for_symbols.get_active_futures_symbols()
    if not api_symbols:
        st.warning(
            "Could not fetch symbols from Binance. "
            "Falling back to custom symbol list."
        )
        symbols = [s.strip().upper() for s in custom_symbols_text.split(",") if s.strip()]
    else:
        symbols = api_symbols
else:
    symbols = [s.strip().upper() for s in custom_symbols_text.split(",") if s.strip()]

# -----------------------------
# Data loading with refresh
# -----------------------------
if "funding_results" not in st.session_state or refresh_button:
    with st.spinner("Fetching funding rate data from Binance..."):
        results = run_full_analysis(symbols, days_back, outlier_method)
        st.session_state["funding_results"] = results

results = st.session_state.get("funding_results", None)

if not results:
    st.stop()

latest_data = results["latest_data"]
top_10_z = results["top_10_z"]
bottom_10_z = results["bottom_10_z"]
latest_z = results["latest_z"]
top_z = results["top_z"]
bottom_z = results["bottom_z"]

all_symbols = sorted(latest_data.index.unique())


# -----------------------------
# Global symbol filter
# -----------------------------
st.subheader("Filter")

selected_symbols = st.multiselect(
    "Filter by symbol (leave empty for all)",
    options=all_symbols,
    default=[]
)


def filter_indexed_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if not selected_symbols:
        return df
    return df[df.index.isin(selected_symbols)]


def filter_series(s: pd.Series) -> pd.Series:
    if s is None or s.empty:
        return s
    if not selected_symbols:
        return s
    return s[s.index.isin(selected_symbols)]


# -----------------------------
# Section 1: latest_df, top_10, bottom_10
# -----------------------------
st.markdown("## Latest Annualized Funding Rates (Per Symbol)")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Latest per symbol (`latest_df`)**")
    st.dataframe(filter_indexed_df(latest_data))

with col2:
    st.markdown("**Top 10 by z-score (`top_10`)**")
    st.dataframe(filter_indexed_df(top_10_z))

with col3:
    st.markdown("**Bottom 10 by z-score (`bottom_10`)**")
    st.dataframe(filter_indexed_df(bottom_10_z))


# -----------------------------
# Section 2: latest_zscores, top_coins, bottom_coins
# -----------------------------
st.markdown("## Latest Z-Scores vs Each Coin's Own History")

col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("**Latest z-scores (`latest_zscores`)**")
    latest_z_filtered = filter_series(latest_z)
    st.dataframe(latest_z_filtered.to_frame(name="zscore_latest"))

with col5:
    st.markdown("**Top coins (`top_coins`)**")
    top_z_filtered = filter_series(top_z)
    st.dataframe(top_z_filtered.to_frame(name="zscore_latest"))

with col6:
    st.markdown("**Bottom coins (`bottom_coins`)**")
    bottom_z_filtered = filter_series(bottom_z)
    st.dataframe(bottom_z_filtered.to_frame(name="zscore_latest"))

