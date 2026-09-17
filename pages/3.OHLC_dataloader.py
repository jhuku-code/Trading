"""
================================================================================
OHLC MultiIndex Data Loader — Bybit USDT-M Linear Perpetuals
================================================================================
Fetches OHLC candle data for all coins listed in Themes_mapping.xlsx from
Bybit linear perpetual futures via ccxt.

Output: a single DataFrame with MultiIndex columns (symbol, field) where
field ∈ {o, h, l, c}, stored in st.session_state["ohlc_multi"].
Also derives a flat close-price DataFrame in st.session_state["close_prices"].

Pagination:
  Bybit caps at 1000 candles per request. For limits > 1000 we paginate
  backwards from the latest candle in batches of 1000, stitching the
  results together. This lets us reliably pull 1500+ bars per coin.

Rate limiting:
  - ccxt's built-in rate limiter is enabled (100ms between requests)
  - load_markets() retries with exponential backoff (Streamlit Cloud
    shared IPs are frequently throttled by Bybit)
  - Configurable sleep between per-coin fetches

Exchange notes:
  - Market format: {SYM}/USDT:USDT (linear perpetual swap)
  - Max candles per request: 1000
  - No US-IP geo-blocking (works on Streamlit Cloud)
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


# =================== Constants ===================
BYBIT_MAX_CANDLES = 1000  # Bybit's per-request ceiling

# Timeframe → milliseconds lookup for pagination offset calculations
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
sleep_seconds = st.sidebar.number_input("Sleep (s)", value=0.30, step=0.05,
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
    """
    Initialize Bybit exchange with built-in rate limiting.
    Retries load_markets() with exponential backoff because Streamlit Cloud's
    shared IPs frequently trigger Bybit's rate limiter on the initial
    metadata fetch.
    """
    ex = ccxt.bybit({
        'enableRateLimit': True,  # ccxt auto-throttles to respect Bybit's limits
        'rateLimit': 100,         # minimum ms between requests (~10 req/sec)
    })

    # Retry load_markets with backoff — 1s, 2s, 4s between attempts
    for attempt in range(3):
        try:
            ex.load_markets()
            return ex
        except (ccxt.RateLimitExceeded, ccxt.ExchangeNotAvailable) as e:
            wait = 2 ** attempt
            st.toast(f"⏳ Rate limited on load_markets(), retrying in {wait}s...")
            time.sleep(wait)

    # Final attempt — let it raise if still failing
    ex.load_markets()
    return ex


def fetch_ohlcv_paginated(exchange, market_id, timeframe, total_limit, sleep_s):
    """
    Fetch up to `total_limit` candles for a single market, paginating
    backwards in chunks of BYBIT_MAX_CANDLES (1000).

    Strategy:
      1) First call: no `since` → returns the most recent chunk.
      2) Use the earliest timestamp from that chunk to calculate `since`
         for the next (older) batch, stepping back by chunk_size * tf_ms.
      3) Repeat until we have enough bars or the exchange returns nothing.
      4) Sort ascending by timestamp, deduplicate, and trim to total_limit.

    Returns a list of [timestamp_ms, o, h, l, c, v] lists, oldest first.
    """
    tf_ms = TF_TO_MS[timeframe]  # milliseconds per candle
    all_candles = []
    remaining = total_limit

    # -- First fetch: latest candles (no `since`) --
    chunk_size = min(remaining, BYBIT_MAX_CANDLES)
    candles = exchange.fetch_ohlcv(market_id, timeframe=timeframe, limit=chunk_size)

    if not candles:
        return []

    all_candles.extend(candles)
    remaining -= len(candles)

    # -- Paginate backwards if we need more bars --
    while remaining > 0 and len(candles) == chunk_size:
        # Earliest timestamp in current batch → step back by one full chunk
        earliest_ts = candles[0][0]
        since = earliest_ts - (min(remaining, BYBIT_MAX_CANDLES) * tf_ms)

        chunk_size = min(remaining, BYBIT_MAX_CANDLES)

        time.sleep(sleep_s)  # rate-limit pause between paginated calls

        candles = exchange.fetch_ohlcv(market_id, timeframe=timeframe,
                                        since=since, limit=chunk_size)

        if not candles:
            break  # no more history available

        all_candles.extend(candles)
        remaining -= len(candles)

    # -- Deduplicate by timestamp, sort oldest-first, trim to requested limit --
    seen = {}
    for c in all_candles:
        seen[c[0]] = c  # last-write-wins dedup on timestamp

    sorted_candles = sorted(seen.values(), key=lambda x: x[0])

    # Keep the most recent `total_limit` candles
    if len(sorted_candles) > total_limit:
        sorted_candles = sorted_candles[-total_limit:]

    return sorted_candles


def fetch_ohlc_multiindex(exchange, symbols, timeframe, limit):
    """
    Fetch OHLC candles for each symbol from Bybit linear perpetuals.

    Returns a single DataFrame with:
      - DatetimeIndex (UTC timestamps)
      - MultiIndex columns: (symbol, field) where field ∈ {o, h, l, c}

    Coins that fail to fetch (unlisted, delisted, API errors) are logged
    as warnings and skipped — the rest proceed normally.
    """
    frames = []
    fetched = 0
    failed = 0
    progress_bar = st.progress(0, text="Starting...")

    for i, sym in enumerate(symbols):
        # Bybit linear perpetual market format: SYM/USDT:USDT
        market_id = f"{sym}/USDT:USDT"
        progress_bar.progress((i + 1) / len(symbols),
                              text=f"Fetching {sym} ({i + 1}/{len(symbols)})")

        try:
            # Use paginated fetch to handle limits > 1000
            ohlcv = fetch_ohlcv_paginated(exchange, market_id, timeframe, limit, sleep_seconds)

            if not ohlcv:
                st.warning(f"⚠️ {sym}: no data returned")
                failed += 1
                continue

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

        except ccxt.RateLimitExceeded:
            # If rate-limited mid-fetch, wait and retry once for this symbol
            st.toast(f"⏳ Rate limited on {sym}, waiting 3s and retrying...")
            time.sleep(3)
            try:
                ohlcv = fetch_ohlcv_paginated(exchange, market_id, timeframe, limit, sleep_seconds)
                if ohlcv:
                    df = pd.DataFrame(ohlcv, columns=["ts", "o", "h", "l", "c", "v"])
                    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
                    df = df.set_index("ts")
                    df = df[["o", "h", "l", "c"]]
                    df.columns = pd.MultiIndex.from_product([[sym], df.columns])
                    frames.append(df)
                    fetched += 1
                else:
                    failed += 1
            except Exception:
                st.warning(f"⚠️ {sym}: failed after rate-limit retry")
                failed += 1

        except Exception as e:
            st.warning(f"⚠️ {sym}: {e}")
            failed += 1
            continue

    progress_bar.empty()  # clean up progress bar after completion

    if not frames:
        st.error("No data fetched for any symbol. Check your Themes_mapping.xlsx entries "
                 "against available Bybit perpetual listings.")
        return pd.DataFrame()

    # Outer join on timestamps — coins with fewer bars get NaN-filled
    final_df = pd.concat(frames, axis=1).sort_index()

    st.toast(f"✅ Fetched {fetched} coins, {failed} failed | {final_df.shape[0]} bars")

    return final_df


# =================== Fetch Button ===================
if st.sidebar.button("🔄 Fetch OHLC Data"):

    df_map = read_theme_excel(default_excel_relpath)
    symbols = df_map["Symbol"].str.upper().tolist()

    ex = get_exchange()

    with st.spinner(f"Fetching {len(symbols)} coins from Bybit linear perpetuals..."):
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
