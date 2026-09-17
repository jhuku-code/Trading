"""
================================================================================
OHLC MultiIndex Data Loader — Bybit Public REST API (Diagnostic Build)
================================================================================
Hardcoded test symbols for debugging. Verbose error logging on every API call
to diagnose connectivity/format issues from Streamlit Cloud.

Once working, swap TEST_SYMBOLS back to the Excel-based loader.
================================================================================
"""

import time
from datetime import datetime

import requests
import pandas as pd
import streamlit as st


# =================== Page Configuration ===================
st.set_page_config(page_title="OHLC Data (MultiIndex)", layout="wide")
st.title("OHLC MultiIndex Data Loader")


# =================== Hardcoded Test Symbols ===================
# Bybit linear perpetual format: just append USDT (no slash, no colon)
# These are high-liquidity pairs guaranteed to exist on Bybit futures
TEST_SYMBOLS = ["BTC", "ETH", "SOL", "XRP", "DOGE"]


# =================== Constants ===================
BYBIT_KLINE_URL = "https://api.bybit.com/v5/market/kline"
BYBIT_MAX_CANDLES = 1000

# Timeframe display → Bybit API interval mapping
TF_TO_BYBIT = {
    "1d":  "D",
    "4h":  "240",
    "1h":  "60",
    "30m": "30",
    "15m": "15",
}

# Timeframe → milliseconds for pagination
TF_TO_MS = {
    "1d":  86_400_000,
    "4h":  14_400_000,
    "1h":   3_600_000,
    "30m":  1_800_000,
    "15m":    900_000,
}


# =================== Sidebar ===================
st.sidebar.header("Settings")

st.sidebar.write(f"**Test symbols:** {', '.join(TEST_SYMBOLS)}")

timeframe = st.sidebar.selectbox("Timeframe", ["1d", "4h", "1h", "30m", "15m"], index=0)
limit = st.sidebar.number_input("OHLC limit", value=90, min_value=10, max_value=2000,
                                 help="Total candles per coin. Pagination handles limits > 1000.")
sleep_seconds = st.sidebar.number_input("Sleep (s)", value=0.30, step=0.05)


# =================== Session State Init ===================
if "ohlc_multi" not in st.session_state:
    st.session_state["ohlc_multi"] = None

if "close_prices" not in st.session_state:
    st.session_state["close_prices"] = None

if "last_fetch" not in st.session_state:
    st.session_state["last_fetch"] = None


# =================== API Functions ===================

def bybit_fetch_klines(symbol: str, interval: str, limit: int = 1000,
                        end: int = None) -> list:
    """
    Call Bybit's v5 public kline endpoint for a single symbol.
    Includes verbose diagnostic logging for every failure mode.

    Returns list of [ts_ms, o, h, l, c] oldest first, or [] on failure.
    """
    # Bybit expects symbol like "BTCUSDT" for linear perps
    bybit_symbol = f"{symbol}USDT"

    params = {
        "category": "linear",
        "symbol": bybit_symbol,
        "interval": interval,
        "limit": min(limit, BYBIT_MAX_CANDLES),
    }
    if end is not None:
        params["end"] = end

    last_status = None
    last_body = None

    for attempt in range(3):
        try:
            resp = requests.get(BYBIT_KLINE_URL, params=params, timeout=15)
            last_status = resp.status_code
            last_body = resp.text[:300]  # capture response for diagnostics

            if resp.status_code == 200:
                data = resp.json()

                if data.get("retCode") == 0 and data.get("result", {}).get("list"):
                    # Success — parse candles
                    raw = data["result"]["list"]
                    candles = [
                        [int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4])]
                        for r in raw
                    ]
                    candles.reverse()  # Bybit returns newest-first; we want oldest-first
                    return candles

                else:
                    # HTTP 200 but Bybit returned an error or empty data
                    st.warning(
                        f"⚠️ **{bybit_symbol}**: Bybit returned retCode={data.get('retCode')}, "
                        f"retMsg='{data.get('retMsg', 'n/a')}'"
                    )
                    return []

            elif resp.status_code == 429:
                # Rate limited — backoff and retry
                wait = 2 * (attempt + 1)
                st.toast(f"⏳ {bybit_symbol}: rate limited, waiting {wait}s (attempt {attempt+1}/3)")
                time.sleep(wait)

            else:
                # Other HTTP error
                time.sleep(1)

        except requests.exceptions.ConnectionError as e:
            last_status = "ConnectionError"
            last_body = str(e)[:300]
            time.sleep(1)

        except requests.exceptions.Timeout:
            last_status = "Timeout"
            last_body = "Request timed out after 15s"
            time.sleep(1)

        except Exception as e:
            last_status = type(e).__name__
            last_body = str(e)[:300]
            time.sleep(1)

    # All retries exhausted — show exactly what happened
    st.warning(
        f"⚠️ **{bybit_symbol}**: failed after 3 retries — "
        f"status={last_status}, response: `{last_body}`"
    )
    return []


def fetch_ohlcv_paginated(symbol: str, timeframe: str, total_limit: int,
                           sleep_s: float) -> list:
    """
    Fetch up to `total_limit` candles for a single symbol, paginating
    backwards in chunks of 1000.

    Returns list of [ts_ms, o, h, l, c], oldest first.
    """
    interval = TF_TO_BYBIT[timeframe]
    all_candles = []
    remaining = total_limit

    # -- First fetch: latest candles --
    chunk_size = min(remaining, BYBIT_MAX_CANDLES)
    candles = bybit_fetch_klines(symbol, interval, limit=chunk_size)

    if not candles:
        return []

    all_candles.extend(candles)
    remaining -= len(candles)

    # -- Paginate backwards if we need more bars --
    while remaining > 0 and len(candles) >= chunk_size:
        # Earliest timestamp in current batch → use as `end` for next batch
        end_ts = candles[0][0] - 1  # -1ms to avoid boundary overlap

        chunk_size = min(remaining, BYBIT_MAX_CANDLES)

        time.sleep(sleep_s)

        candles = bybit_fetch_klines(symbol, interval, limit=chunk_size, end=end_ts)

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
    """
    frames = []
    fetched = 0
    failed = 0
    failed_symbols = []

    for i, sym in enumerate(symbols):
        st.write(f"**[{i+1}/{len(symbols)}]** Fetching `{sym}USDT` ...")

        ohlcv = fetch_ohlcv_paginated(sym, timeframe, limit, sleep_seconds)

        if not ohlcv:
            st.error(f"❌ {sym}: no data returned")
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
        st.write(f"✅ {sym}: {len(ohlcv)} bars fetched")

        time.sleep(sleep_seconds)

    if not frames:
        st.error("No data fetched for any symbol. See warnings above for details.")
        return pd.DataFrame()

    # Outer join on timestamps — coins with fewer bars get NaN-filled
    final_df = pd.concat(frames, axis=1).sort_index()

    st.success(f"✅ Done — {fetched} coins fetched, {failed} failed, {final_df.shape[0]} total bars")

    return final_df


# =================== Fetch Button ===================
if st.sidebar.button("🔄 Fetch OHLC Data"):

    symbols = [s.upper().strip() for s in TEST_SYMBOLS]

    with st.spinner(f"Fetching {len(symbols)} test coins from Bybit..."):
        ohlc_df = fetch_ohlc_multiindex(symbols, timeframe, limit)

    if not ohlc_df.empty:
        st.session_state["ohlc_multi"] = ohlc_df
        st.session_state["close_prices"] = ohlc_df.xs("c", level=1, axis=1)
        st.session_state["last_fetch"] = datetime.utcnow()
        st.session_state["timeframe"] = timeframe


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
