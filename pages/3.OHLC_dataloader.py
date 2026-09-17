"""
================================================================================
OHLC MultiIndex Data Loader — KuCoin Public REST API (no ccxt)
================================================================================
Fetches OHLC candle data for all coins listed in Themes_mapping.xlsx from
KuCoin's public kline API for spot markets.

Why KuCoin:
  Both Binance and Bybit geo-block API access from Streamlit Cloud's
  US-based servers (403 via CloudFront). KuCoin is the only major
  exchange confirmed to work from Streamlit Cloud.

Why direct REST instead of ccxt:
  Avoids ccxt's load_markets() overhead and potential rate-limit issues.
  The public kline endpoint needs no auth — just symbol and interval.

Output: a single DataFrame with MultiIndex columns (symbol, field) where
field ∈ {o, h, l, c}, stored in st.session_state["ohlc_multi"].
Also derives a flat close-price DataFrame in st.session_state["close_prices"].

Pagination:
  KuCoin returns max 1500 candles per request. For limits > 1500 we
  paginate using startAt/endAt timestamp windows.

API reference:
  https://www.kucoin.com/docs/rest/spot-trading/market-data/get-klines
  Endpoint: GET https://api.kucoin.com/api/v1/market/candles
  Public, no auth required.
  Response: [[timestamp_s, open, close, high, low, volume, turnover], ...]
  Note: KuCoin returns NEWEST first and uses seconds (not ms).
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
KUCOIN_KLINE_URL = "https://api.kucoin.com/api/v1/market/candles"
KUCOIN_MAX_CANDLES = 1500  # per-request ceiling

# Timeframe display → KuCoin API type mapping
# KuCoin types: 1min,3min,5min,15min,30min,1hour,2hour,4hour,6hour,8hour,12hour,1day,1week
TF_TO_KUCOIN = {
    "1d":  "1day",
    "4h":  "4hour",
    "1h":  "1hour",
    "30m": "30min",
    "15m": "15min",
}

# Timeframe → seconds for pagination offset calculations
TF_TO_SECS = {
    "1d":  86_400,
    "4h":  14_400,
    "1h":   3_600,
    "30m":  1_800,
    "15m":    900,
}


# =================== Sidebar ===================
st.sidebar.header("Settings")

default_excel_relpath = Path("Input-Files") / "Themes_mapping.xlsx"
st.sidebar.write(f"Using: `{default_excel_relpath}`")

timeframe = st.sidebar.selectbox("Timeframe", ["1d", "4h", "1h", "30m", "15m"], index=0)
limit = st.sidebar.number_input("OHLC limit", value=90, min_value=10, max_value=3000,
                                 help="Total candles per coin. Pagination handles limits > 1500 automatically.")
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


def kucoin_fetch_klines(symbol: str, kc_type: str, start_at: int = None,
                         end_at: int = None) -> list:
    """
    Call KuCoin's public kline endpoint for a single symbol.

    Args:
        symbol:   KuCoin market symbol, e.g. "BTC-USDT"
        kc_type:  KuCoin interval type, e.g. "30min", "1day"
        start_at: Start timestamp in SECONDS (inclusive). None = no lower bound.
        end_at:   End timestamp in SECONDS (inclusive). None = latest.

    Returns:
        List of [timestamp_s, open, high, low, close] (oldest first).
        KuCoin returns [ts, open, CLOSE, HIGH, LOW, vol, turnover] newest-first,
        so we reorder columns and reverse.

    KuCoin quirk: column order is ts, open, CLOSE, high, low (not OHLC).
    """
    params = {
        "type": kc_type,
        "symbol": symbol,
    }
    if start_at is not None:
        params["startAt"] = start_at
    if end_at is not None:
        params["endAt"] = end_at

    last_status = None
    last_body = None

    for attempt in range(3):
        try:
            resp = requests.get(KUCOIN_KLINE_URL, params=params, timeout=15)
            last_status = resp.status_code
            last_body = resp.text[:300]

            if resp.status_code == 200:
                data = resp.json()

                if data.get("code") == "200000" and data.get("data"):
                    raw = data["data"]
                    # KuCoin column order: [ts, open, close, high, low, volume, turnover]
                    # We want: [ts, open, high, low, close] → remap columns
                    candles = [
                        [int(r[0]), float(r[1]), float(r[3]), float(r[4]), float(r[2])]
                        for r in raw
                    ]
                    # KuCoin returns newest-first; reverse to oldest-first
                    candles.reverse()
                    return candles

                elif data.get("code") == "200000" and not data.get("data"):
                    # Valid response but no data — symbol likely not listed
                    return []

                else:
                    st.warning(
                        f"⚠️ **{symbol}**: KuCoin error code={data.get('code')}, "
                        f"msg='{data.get('msg', 'n/a')}'"
                    )
                    return []

            elif resp.status_code == 429:
                wait = 2 * (attempt + 1)
                time.sleep(wait)

            else:
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

    # All retries exhausted
    st.warning(
        f"⚠️ **{symbol}**: failed after 3 retries — "
        f"status={last_status}, response: `{last_body}`"
    )
    return []


def fetch_ohlcv_paginated(symbol: str, timeframe: str, total_limit: int,
                           sleep_s: float) -> list:
    """
    Fetch up to `total_limit` candles for a single symbol, paginating
    backwards in chunks of KUCOIN_MAX_CANDLES (1500).

    Strategy:
      1) Calculate startAt/endAt window: endAt = now, startAt = now - (chunk * tf_secs).
      2) Fetch the chunk.
      3) Slide the window backwards for the next (older) batch.
      4) Repeat until we have enough bars or the API returns nothing.
      5) Deduplicate, sort oldest-first, trim to total_limit (keeping latest).

    Returns list of [ts_s, o, h, l, c], oldest first.
    """
    kc_type = TF_TO_KUCOIN[timeframe]
    tf_secs = TF_TO_SECS[timeframe]
    kc_symbol = f"{symbol}-USDT"  # KuCoin format: dash-separated
    all_candles = []
    remaining = total_limit

    # Current time as end boundary
    end_at = int(time.time())

    while remaining > 0:
        chunk_size = min(remaining, KUCOIN_MAX_CANDLES)
        # Calculate start timestamp for this chunk
        start_at = end_at - (chunk_size * tf_secs)

        candles = kucoin_fetch_klines(kc_symbol, kc_type,
                                       start_at=start_at, end_at=end_at)

        if not candles:
            break

        all_candles.extend(candles)
        remaining -= len(candles)

        # Slide window backwards: new end = oldest timestamp in this batch - 1 second
        end_at = candles[0][0] - 1

        # If we got fewer candles than the window allows, no more history exists
        if len(candles) < chunk_size // 2:
            break

        time.sleep(sleep_s)

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
    Fetch OHLC candles for each symbol from KuCoin spot markets.

    Returns a single DataFrame with:
      - DatetimeIndex (UTC timestamps)
      - MultiIndex columns: (symbol, field) where field ∈ {o, h, l, c}

    Coins that fail to fetch (unlisted, no data) are logged as warnings
    and skipped — the rest proceed normally.
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
            # KuCoin timestamps are in seconds — convert to datetime
            df["ts"] = pd.to_datetime(df["ts"], unit="s")
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
                 "against available KuCoin spot listings.")
        return pd.DataFrame()

    # Outer join on timestamps — coins with fewer bars get NaN-filled
    final_df = pd.concat(frames, axis=1).sort_index()

    st.toast(f"✅ Fetched {fetched} coins, {failed} failed | {final_df.shape[0]} bars")

    # Show failed symbols in a collapsed expander if any
    if failed_symbols:
        with st.expander(f"⚠️ {failed} symbol(s) returned no data (not listed on KuCoin spot)"):
            st.write(", ".join(sorted(failed_symbols)))

    return final_df


# =================== Fetch Button ===================
if st.sidebar.button("🔄 Fetch OHLC Data"):

    df_map = read_theme_excel(default_excel_relpath)
    symbols = df_map["Symbol"].str.upper().tolist()

    with st.spinner(f"Fetching {len(symbols)} coins from KuCoin spot..."):
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
