"""
================================================================================
OHLC MultiIndex Data Loader — Bybit Public REST API (no ccxt)
================================================================================
Fetches OHLC candle data for all coins listed in Themes_mapping.xlsx from
Bybit's v5 public kline API for linear perpetual futures.

Why direct REST instead of ccxt:
  ccxt's Bybit driver calls load_markets() which fetches spot + inverse +
  linear + option metadata in multiple requests. Streamlit Cloud's shared
  IPs consistently trigger Bybit's rate limiter on these metadata calls.
  The public kline endpoint needs no auth and no market metadata — just
  the symbol string and interval.

Output: a single DataFrame with MultiIndex columns (symbol, field) where
field ∈ {o, h, l, c}, stored in st.session_state["ohlc_multi"].
Also derives a flat close-price DataFrame in st.session_state["close_prices"].

Pagination:
  Bybit caps at 1000 candles per request. For limits > 1000 we paginate
  backwards using the `end` timestamp parameter in chunks of 1000.

API reference:
  https://bybit-exchange.github.io/docs/v5/market/kline
  Endpoint: GET https://api.bybit.com/v5/market/kline
  Public, no auth required.
================================================================================
"""

import time
from pathlib import Path
from datetime import datetime

import requests
import pandas as pd
import streamlit as st


# =================== Page Configuration ===================
st.set_page_config(page_title="OHLC Data (MultiIndex)", layout="wide")
st.title("OHLC MultiIndex Data Loader")


# =================== Constants ===================
BYBIT_KLINE_URL = "https://api.bybit.com/v5/market/kline"
BYBIT_MAX_CANDLES = 1000  # per-request ceiling

# Timeframe display → Bybit API interval mapping
# Bybit intervals: 1,3,5,15,30,60,120,240,360,720,D,W,M
TF_TO_BYBIT = {
    "1d":  "D",
    "4h":  "240",
    "1h":  "60",
    "30m": "30",
    "15m": "15",
}

# Timeframe → milliseconds for pagination offset calculations
TF_TO_MS = {
    "1d":  86_400_000,
    "4h":  14_400_000,
    "1h":   3_600_000,
    "30m":  1_800_000,
    "15m":    900_000,
}


# =================== Sidebar ===================
st.sidebar.header("Settings")

default_excel_relpath = Path("Input-Files") / "Themes_mapping.xlsx"
st.sidebar.write(f"Using: `{default_excel_relpath}`")

timeframe = st.sidebar.selectbox("Timeframe", ["1d", "4h", "1h", "30m", "15m"], index=0)
limit = st.sidebar.number_input("OHLC limit", value=90, min_value=10, max_value=2000,
                                 help="Total candles per coin. Pagination handles limits > 1000 automatically.")
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


def bybit_fetch_klines(symbol: str, interval: str, limit: int = 1000,
                        end: int = None) -> list:
    """
    Call Bybit's v5 public kline endpoint for a single symbol.

    Args:
        symbol:   Bybit market symbol, e.g. "BTCUSDT"
        interval: Bybit interval string, e.g. "30", "60", "D"
        limit:    Number of candles (max 1000)
        end:      End timestamp in milliseconds (None = latest)

    Returns:
        List of [timestamp_ms, open, high, low, close] (oldest first).
        Bybit returns newest-first, so we reverse here for consistency.

    Raises on HTTP errors after 3 retries with backoff.
    """
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "limit": min(limit, BYBIT_MAX_CANDLES),
    }
    if end is not None:
        params["end"] = end

    # Retry with backoff for transient rate-limit / server errors
    for attempt in range(3):
        resp = requests.get(BYBIT_KLINE_URL, params=params, timeout=15)

        if resp.status_code == 200:
            data = resp.json()
            if data.get("retCode") == 0 and data.get("result", {}).get("list"):
                # Bybit returns: [ts, o, h, l, c, volume, turnover] newest-first
                raw = data["result"]["list"]
                # Extract [ts, o, h, l, c], convert strings to float, reverse to oldest-first
                candles = [
                    [int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4])]
                    for r in raw
                ]
                candles.reverse()
                return candles
            else:
                # Valid response but no data (delisted / unlisted symbol)
                return []

        elif resp.status_code == 429:
            # Rate limited — backoff and retry
            wait = 2 * (attempt + 1)
            time.sleep(wait)
        else:
            # Other HTTP error — backoff and retry
            time.sleep(1)

    return []  # all retries exhausted


def fetch_ohlcv_paginated(symbol: str, timeframe: str, total_limit: int,
                           sleep_s: float) -> list:
    """
    Fetch up to `total_limit` candles for a single symbol, paginating
    backwards in chunks of 1000.

    Strategy:
      1) First call: no `end` → returns the most recent 1000 candles.
      2) Use the earliest timestamp from that chunk as the new `end`
         for the next (older) batch.
      3) Repeat until we have enough bars or the API returns nothing.
      4) Deduplicate, sort oldest-first, trim to total_limit (keeping latest).

    Returns list of [ts_ms, o, h, l, c], oldest first.
    """
    interval = TF_TO_BYBIT[timeframe]
    bybit_symbol = f"{symbol}USDT"  # Bybit format: no slash, no colon
    all_candles = []
    remaining = total_limit

    # -- First fetch: latest candles --
    chunk_size = min(remaining, BYBIT_MAX_CANDLES)
    candles = bybit_fetch_klines(bybit_symbol, interval, limit=chunk_size)

    if not candles:
        return []

    all_candles.extend(candles)
    remaining -= len(candles)

    # -- Paginate backwards if we need more bars --
    while remaining > 0 and len(candles) >= chunk_size:
        # Earliest timestamp in current batch → use as `end` for next batch
        # Subtract 1ms so we don't re-fetch the boundary candle
        end_ts = candles[0][0] - 1

        chunk_size = min(remaining, BYBIT_MAX_CANDLES)

        time.sleep(sleep_s)

        candles = bybit_fetch_klines(bybit_symbol, interval,
                                      limit=chunk_size, end=end_ts)

        if not candles:
            break

        all_candles.extend(candles)
        remaining -= len(candles)

    # -- Deduplicate by timestamp, sort oldest-first, trim --
    seen = {}
    for c in all_candles:
        seen[c[0]] = c

    sorted_candles = sorted(seen.values(), key=lambda x: x[0])

    # Keep the most recent `total_limit` candles
    if len(sorted_candles) > total_limit:
        sorted_candles = sorted_candles[-total_limit:]

    return sorted_candles


def fetch_ohlc_multiindex(symbols: list, timeframe: str, limit: int) -> pd.DataFrame:
    """
    Fetch OHLC candles for each symbol from Bybit linear perpetuals.

    Returns a single DataFrame with:
      - DatetimeIndex (UTC timestamps)
      - MultiIndex columns: (symbol, field) where field ∈ {o, h, l, c}

    Coins that fail to fetch (unlisted, delisted, no data) are logged
    as warnings and skipped — the rest proceed normally.
    """
    frames = []
    fetched = 0
    failed = 0
    failed_symbols = []
    progress_bar = st.progress(0, text="Starting...")

    for i, sym in enumerate(symbols):
        progress_bar.progress((i + 1) / len(symbols),
                              text=f"Fetching {sym} ({i + 1}/{len(symbols)})")

        try:
            ohlcv = fetch_ohlcv_paginated(sym, timeframe, limit, sleep_seconds)

            if not ohlcv:
                failed += 1
                failed_symbols.append(sym)
                continue

            df = pd.DataFrame(ohlcv, columns=["ts", "o", "h", "l", "c"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms")
            df = df.set_index("ts")

            # MultiIndex columns → (symbol, field)
            df.columns = pd.MultiIndex.from_product([[sym], df.columns])

            frames.append(df)
            fetched += 1

            time.sleep(sleep_seconds)

        except Exception as e:
            st.warning(f"⚠️ {sym}: {e}")
            failed += 1
            failed_symbols.append(sym)
            continue

    progress_bar.empty()

    if not frames:
        st.error("No data fetched for any symbol. Check your Themes_mapping.xlsx entries "
                 "against available Bybit perpetual listings.")
        return pd.DataFrame()

    # Outer join on timestamps — coins with fewer bars get NaN-filled
    final_df = pd.concat(frames, axis=1).sort_index()

    st.toast(f"✅ Fetched {fetched} coins, {failed} failed | {final_df.shape[0]} bars")

    # Show failed symbols in a collapsed expander if any
    if failed_symbols:
        with st.expander(f"⚠️ {failed} symbol(s) returned no data"):
            st.write(", ".join(sorted(failed_symbols)))

    return final_df


# =================== Fetch Button ===================
if st.sidebar.button("🔄 Fetch OHLC Data"):

    df_map = read_theme_excel(default_excel_relpath)
    symbols = df_map["Symbol"].str.upper().tolist()

    with st.spinner(f"Fetching {len(symbols)} coins from Bybit linear perpetuals..."):
        ohlc_df = fetch_ohlc_multiindex(symbols, timeframe, limit)

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
