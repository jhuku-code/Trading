# ohlc_data_page.py

import time
from pathlib import Path
from datetime import datetime

import ccxt
import pandas as pd
import streamlit as st


# ---------------- Page Config ----------------
st.set_page_config(page_title="OHLC Data (MultiIndex)", layout="wide")
st.title("OHLC MultiIndex Data Loader")


# ---------------- Sidebar ----------------
st.sidebar.header("Settings")

default_excel_relpath = Path("Input-Files") / "Themes_mapping.xlsx"
st.sidebar.write(f"Using: `{default_excel_relpath}`")

timeframe = st.sidebar.selectbox("Timeframe", ["1d", "4h", "1h"], index=0)
limit = st.sidebar.number_input("OHLC limit", value=90, min_value=10, max_value=2000)
sleep_seconds = st.sidebar.number_input("Sleep (s)", value=0.2)


# ---------------- Session Init ----------------
if "ohlc_multi" not in st.session_state:
    st.session_state["ohlc_multi"] = None

if "close_prices" not in st.session_state:
    st.session_state["close_prices"] = None

if "last_fetch" not in st.session_state:
    st.session_state["last_fetch"] = None


# ---------------- Helpers ----------------
@st.cache_data
def read_theme_excel(path):
    return pd.read_excel(path, engine="openpyxl")


@st.cache_resource
def get_exchange():
    ex = ccxt.kucoin()
    ex.load_markets()
    return ex


def fetch_ohlc_multiindex(exchange, symbols, timeframe, limit):
    frames = []

    for sym in symbols:
        try:
            ohlcv = exchange.fetch_ohlcv(f"{sym}/USDT", timeframe=timeframe, limit=limit)

            df = pd.DataFrame(
                ohlcv,
                columns=["ts", "o", "h", "l", "c", "v"]
            )

            df["ts"] = pd.to_datetime(df["ts"], unit="ms")
            df = df.set_index("ts")

            # Keep OHLC only
            df = df[["o", "h", "l", "c"]]

            # MultiIndex columns → (symbol, field)
            df.columns = pd.MultiIndex.from_product([[sym], df.columns])

            frames.append(df)

            time.sleep(sleep_seconds)

        except Exception as e:
            st.warning(f"Error fetching {sym}: {e}")
            continue

    final_df = pd.concat(frames, axis=1).sort_index()

    return final_df


# ---------------- Fetch Button ----------------
if st.sidebar.button("🔄 Fetch OHLC Data"):

    df_map = read_theme_excel(default_excel_relpath)
    symbols = df_map["Symbol"].str.upper().tolist()

    ex = get_exchange()

    with st.spinner("Fetching OHLC data..."):
        ohlc_df = fetch_ohlc_multiindex(ex, symbols, timeframe, limit)

    # -------- Store in session --------
    st.session_state["ohlc_multi"] = ohlc_df

    # Derived close-only (for reuse with your old logic)
    st.session_state["close_prices"] = ohlc_df.xs("c", level=1, axis=1)

    st.session_state["last_fetch"] = datetime.utcnow()
    st.session_state["timeframe"] = timeframe

    st.success("OHLC data loaded successfully!")


# ---------------- Display ----------------
if st.session_state["ohlc_multi"] is None:
    st.info("Click 'Fetch OHLC Data' to load data")
else:
    df = st.session_state["ohlc_multi"]

    st.subheader("Dataset Info")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Coins", len(df.columns.levels[0]))

    st.write("Time range:", df.index.min(), "→", df.index.max())

    st.subheader("Column Structure (MultiIndex)")
    st.write(df.columns[:12])

    st.subheader("Preview (OHLC Data)")
    st.dataframe(df.tail(20), use_container_width=True)

    st.subheader("Derived Close Prices (for compatibility)")
    st.dataframe(st.session_state["close_prices"].tail(10), use_container_width=True)

    st.caption(f"Last updated: {st.session_state['last_fetch']}")
