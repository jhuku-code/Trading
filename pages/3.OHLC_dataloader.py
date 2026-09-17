"""
================================================================================
OHLC MultiIndex Data Loader — Binance USDT-M Futures
================================================================================
Fetches OHLC candle data for all coins listed in Themes_mapping.xlsx from
Binance USDT-margined perpetual futures (binanceusdm) via ccxt.

Output: a single DataFrame with MultiIndex columns (symbol, field) where
field ∈ {o, h, l, c}, stored in st.session_state["ohlc_multi"].
Also derives a flat close-price DataFrame in st.session_state["close_prices"].

Exchange notes:
  - Market format: {SYM}/USDT:USDT (perpetual linear swap)
  - Max candles per request: 1500
  - Rate limit: handled via configurable sleep between calls
================================================================================
"""

import time
from pathlib import Path
from datetime import datetime

import ccxt
import pandas as pd
import streamlit as st


# =================== Page Configuration ===================
st.set_page_config(page_title="OHLC Data (MultiIndex)", layout="wide")
st.title("OHLC MultiIndex Data Loader")


# =================== Sidebar ===================
st.sidebar.header("Settings")

default_excel_relpath = Path("Input-Files") / "Themes_mapping.xlsx"
st.sidebar.write(f"Using: `{default_excel_relpath}`")

timeframe = st.sidebar.selectbox("Timeframe", ["1d", "4h", "1h", "30m", "15m"], index=0)
limit = st.sidebar.number_input("OHLC limit", value=90, min_value=10, max_value=1500,
                                 help="Max candles per coin. Binance futures supports up to 1500.")
sleep_seconds = st.sidebar.number_input("Sleep (s)", value=0.20, step=0.05,
                                         help="Delay between API calls to respect rate limits")


# =================== Session State Init ===================
if "ohlc_multi" not in st.session_state:
    st.session_state["ohlc_multi"] = None

if "close_prices" not in st.session_state:
    st.session_state["close_prices"] = None

if "last_fetch" not in st.session_state:
    st.session_state["last_fetch"] = None


# =================== Helpers ===================

@st.cache_data
def read_theme_excel(path):
    """Load the coin-to-theme mapping spreadsheet."""
    return pd.read_excel(path, engine="openpyxl")


@st.cache_resource
def get_exchange():
    """Initialize Binance USDT-M futures exchange and load market metadata."""
    ex = ccxt.binanceusdm()
    ex.load_markets()
    return ex


def fetch_ohlc_multiindex(exchange, symbols, timeframe, limit):
    """
    Fetch OHLC candles for each symbol from Binance USDT-M futures.

    Returns a single DataFrame with:
      - DatetimeIndex (UTC timestamps)
      - MultiIndex columns: (symbol, field) where field ∈ {o, h, l, c}

    Coins that fail to fetch (unlisted, delisted, API errors) are logged
    as warnings and skipped — the rest proceed normally.
    """
    frames = []
    fetched = 0
    failed = 0

    for sym in symbols:
        # Binance USDT-M perpetual market format: SYM/USDT:USDT
        market_id = f"{sym}/USDT:USDT"

        try:
            ohlcv = exchange.fetch_ohlcv(market_id, timeframe=timeframe, limit=limit)

            df = pd.DataFrame(ohlcv, columns=["ts", "o", "h", "l", "c", "v"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms")
            df = df.set_index("ts")

            # Keep OHLC only (drop volume)
            df = df[["o", "h", "l", "c"]]

            # MultiIndex columns → (symbol, field)
            df.columns = pd.MultiIndex.from_product([[sym], df.columns])

            frames.append(df)
            fetched += 1

            time.sleep(sleep_seconds)

        except Exception as e:
            st.warning(f"⚠️ {sym}: {e}")
            failed += 1
            continue

    if not frames:
        st.error("No data fetched for any symbol. Check your Themes_mapping.xlsx entries "
                 "against available Binance futures listings.")
        return pd.DataFrame()

    # Outer join on timestamps — coins with fewer bars get NaN-filled
    final_df = pd.concat(frames, axis=1).sort_index()

    st.toast(f"✅ Fetched {fetched} coins ({failed} failed)")

    return final_df


# =================== Fetch Button ===================
if st.sidebar.button("🔄 Fetch OHLC Data"):

    df_map = read_theme_excel(default_excel_relpath)
    symbols = df_map["Symbol"].str.upper().tolist()

    ex = get_exchange()

    with st.spinner(f"Fetching {len(symbols)} coins from Binance USDT-M futures..."):
        ohlc_df = fetch_ohlc_multiindex(ex, symbols, timeframe, limit)

    if not ohlc_df.empty:
        # -------- Store in session state --------
        st.session_state["ohlc_multi"] = ohlc_df

        # Derived close-only DataFrame (for pages that expect a flat close matrix)
        st.session_state["close_prices"] = ohlc_df.xs("c", level=1, axis=1)

        st.session_state["last_fetch"] = datetime.utcnow()
        st.session_state["timeframe"] = timeframe

        st.success("OHLC data loaded successfully!")


# =================== Display ===================
if st.session_state["ohlc_multi"] is None:
    st.info("Click '🔄 Fetch OHLC Data' to load data.")
else:
    df = st.session_state["ohlc_multi"]

    st.subheader("Dataset Info")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Coins", len(df.columns.get_level_values(0).unique()))

    st.write("Time range:", df.index.min(), "→", df.index.max())

    st.subheader("Column Structure (MultiIndex)")
    st.write(df.columns[:12])

    st.subheader("Preview (OHLC Data)")
    st.dataframe(df.tail(20), use_container_width=True)

    st.subheader("Derived Close Prices (for compatibility)")
    st.dataframe(st.session_state["close_prices"].tail(10), use_container_width=True)

    st.caption(f"Last updated: {st.session_state['last_fetch']}")
