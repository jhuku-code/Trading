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
st.set_page_config(page_title="Liquidations Z-Score / Share Monitor", layout="wide")
st.title("Liquidations Monitor (Z-Score + Share vs Total)")

# =========================================================
# SECTION 1 — CONTROL EXECUTION (NO AUTO-RUN)
# =========================================================

# Initialize session flag
if "run_analysis" not in st.session_state:
    st.session_state["run_analysis"] = False

colA, colB, colC = st.columns([1, 1, 1])

with colA:
    if st.button("▶️ Run Analysis / Refresh Calculations"):
        st.session_state["run_analysis"] = True

with colB:
    if st.button("♻️ Force Refresh (Clear Cache + Run)"):
        st.cache_data.clear()
        st.session_state["run_analysis"] = True

with colC:
    if st.button("🧹 Clear Results (Hide)"):
        st.session_state["run_analysis"] = False

st.write("---")
st.info("Nothing will run until you click **Run Analysis / Refresh Calculations**.")

# =========================================================
# SECTION 2 — INPUTS / CONTROLS
# =========================================================

if st.session_state["run_analysis"]:
    # Number of symbols in top/bottom lists
    top_n = st.number_input(
        "Number of symbols to show in Top/Bottom lists",
        min_value=5,
        max_value=100,
        value=15,
        step=5,
    )

    # Lookback months
    lookback_months = st.number_input(
        "Lookback window (months)",
        min_value=1,
        max_value=12,
        value=3,
        step=1,
    )

    # Interval selection
    interval = st.selectbox(
        "Interval / Granularity",
        options=[
            "1min",
            "5min",
            "15min",
            "30min",
            "1hour",
            "2hour",
            "4hour",
            "6hour",
            "12hour",
            "daily",
        ],
        index=6,
    )

    convert_usd = st.checkbox("Convert liquidation values to USD", value=True)
    units_label = "USD" if convert_usd else "native units"

# =========================================================
# SECTION 3 — LOAD PERPS LIST
# =========================================================

API_KEY = st.secrets["API_KEY"]
BASE_URL = "https://api.coinalyze.net/v1"

# GitHub perpetual symbols list
GITHUB_PERPS_URL = (
    "https://raw.githubusercontent.com/jhuku-code/Trading/main/Input-Files/perps_list.xlsx"
)

@st.cache_data(ttl=3600)
def load_perps_list(url: str) -> pd.DataFrame:
    return pd.read_excel(url)

if st.session_state["run_analysis"]:
    try:
        df_perps = load_perps_list(GITHUB_PERPS_URL)
        st.success("Loaded symbols from GitHub `Input-Files/perps_list.xlsx`")
    except Exception as e:
        st.error(f"Error loading perps_list.xlsx from GitHub: {e}")
        st.stop()

    if "Symbol" not in df_perps.columns:
        st.error("Column 'Symbol' not found in perps_list.xlsx")
        st.stop()

    usdt_perp_symbols = df_perps["Symbol"].dropna().astype(str).tolist()
    st.write(f"Total symbols loaded: {len(usdt_perp_symbols)}")

# =========================================================
# SECTION 4 — FETCH LIQUIDATION HISTORY
# =========================================================

def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]

@st.cache_data(ttl=1800)
def fetch_liquidations(symbols, months_back: int, interval: str, api_key: str, convert_usd_flag: bool):
    """
    Returns:
        liq_long_wide, liq_short_wide
    """
    liq_url = f"{BASE_URL}/liquidation-history"

    from_timestamp = int((datetime.now() - relativedelta(months=months_back)).timestamp())
    to_timestamp = int(datetime.now().timestamp())

    all_rows = []

    for batch in chunked(symbols, 20):
        params = {
            "symbols": ",".join(batch),
            "interval": interval,
            "from": from_timestamp,
            "to": to_timestamp,
            "convert_to_usd": "true" if convert_usd_flag else "false",
            "api_key": api_key,
        }

        while True:
            resp = requests.get(liq_url, params=params)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    for entry in data:
                        symbol = entry.get("symbol")
                        hist = entry.get("history", [])
                        if not symbol or not hist:
                            continue
                        df = pd.DataFrame(hist)
                        df["time"] = pd.to_datetime(df["t"], unit="s")
                        df.rename(columns={"l": "liq_long", "s": "liq_short"}, inplace=True)
                        df = df[["time", "liq_long", "liq_short"]]
                        df["symbol"] = symbol
                        all_rows.append(df)
                time.sleep(1)
                break

            elif resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", "30"))
                st.warning(f"429 rate-limited. Waiting {retry_after}s …")
                time.sleep(retry_after)
            else:
                st.error(f"Error fetching batch starting {batch[0]}: {resp.text}")
                break

    if not all_rows:
        raise Exception("No liquidation data received.")

    df_long = pd.concat(all_rows, ignore_index=True)

    liq_long_wide = df_long.pivot_table(index="time", columns="symbol", values="liq_long").sort_index()
    liq_short_wide = df_long.pivot_table(index="time", columns="symbol", values="liq_short").sort_index()

    return liq_long_wide, liq_short_wide

# =========================================================
# SECTION 5 — RUN DATA PIPELINE ONLY WHEN BUTTON CLICKED
# =========================================================

if st.session_state["run_analysis"]:

    st.subheader("Fetching Liquidation Data")

    with st.spinner("Calling Coinalyze API…"):
        try:
            liq_long_data, liq_short_data = fetch_liquidations(
                usdt_perp_symbols, lookback_months, interval, API_KEY, convert_usd
            )
        except Exception as e:
            st.error(f"Error fetching liquidation data: {e}")
            st.stop()

    st.success("Liquidation data loaded successfully.")
    st.write("Data shape:", liq_long_data.shape, liq_short_data.shape)
    st.write("Latest timestamp:", liq_long_data.index.max())

    # =====================================================
    # All original calculations follow exactly as before
    # =====================================================

    # ---------- Z-SCORES ----------
    long_zscores = (liq_long_data - liq_long_data.mean()) / liq_long_data.std()
    short_zscores = (liq_short_data - liq_short_data.mean()) / liq_short_data.std()

    latest_long_z = long_zscores.iloc[-1]
    latest_short_z = short_zscores.iloc[-1]

    # ---------- SHARE / EXCESS ----------
    total_long = liq_long_data.sum(axis=1).replace(0, np.nan)
    total_short = liq_short_data.sum(axis=1).replace(0, np.nan)

    ratio_long = liq_long_data.div(total_long, axis=0)
    ratio_short = liq_short_data.div(total_short, axis=0)

    long_ratio_ma = ratio_long.rolling(30, min_periods=15).mean()
    short_ratio_ma = ratio_short.rolling(30, min_periods=15).mean()

    latest_long_ratio = ratio_long.iloc[-1]
    latest_short_ratio = ratio_short.iloc[-1]
    latest_long_ratio_ma = long_ratio_ma.iloc[-1]
    latest_short_ratio_ma = short_ratio_ma.iloc[-1]

    long_excess = (latest_long_ratio - latest_long_ratio_ma).dropna()
    short_excess = (latest_short_ratio - latest_short_ratio_ma).dropna()

    sorted_long_excess = long_excess.sort_values(ascending=False)
    sorted_short_excess = short_excess.sort_values(ascending=False)

    # ---------- DISPLAY TABS ----------
    st.subheader("Analysis Outputs")

    tab_z, tab_share, tab_search = st.tabs(
        ["By Z-Score", "By Share vs 30-Period Avg", "Search Symbol"]
    )

    # ------------------------------------------------------
    # TAB 1: Z-SCORES
    # ------------------------------------------------------
    with tab_z:
        st.write("Top & Bottom Z-score values")

        st.write("### Long Liquidations")
        st.dataframe(latest_long_z.sort_values(ascending=False).head(top_n))

        st.write("### Short Liquidations")
        st.dataframe(latest_short_z.sort_values(ascending=False).head(top_n))

    # ------------------------------------------------------
    # TAB 2: SHARE vs AVERAGE
    # ------------------------------------------------------
    with tab_share:
        st.write("### Top Excess (Long)")
        st.dataframe(sorted_long_excess.head(top_n))

        st.write("### Bottom Excess (Long)")
        st.dataframe(sorted_long_excess.tail(top_n))

        st.write("### Top Excess (Short)")
        st.dataframe(sorted_short_excess.head(top_n))

        st.write("### Bottom Excess (Short)")
        st.dataframe(sorted_short_excess.tail(top_n))

    # ------------------------------------------------------
    # TAB 3: Search
    # ------------------------------------------------------
    with tab_search:
        symbol = st.text_input("Enter symbol")
        if symbol:
            symbol = symbol.upper()
            if symbol not in liq_long_data.columns:
                st.error("Symbol not found.")
            else:
                st.write("Z-Score (Long):", latest_long_z[symbol])
                st.write("Z-Score (Short):", latest_short_z[symbol])

                st.write("Share (Long):", latest_long_ratio[symbol])
                st.write("Share Avg (Long):", latest_long_ratio_ma[symbol])

                st.write("Share (Short):", lat
