# app.py

import time
from pathlib import Path
from datetime import datetime

import ccxt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Theme Returns Dashboard", layout="wide")
st.title("Theme Returns Dashboard")

# ---------------- Sidebar ----------------
st.sidebar.header("Settings")

default_excel_relpath = Path("Input-Files") / "Themes_mapping.xlsx"
st.sidebar.write(f"Using: `{default_excel_relpath}`")

timeframe = st.sidebar.selectbox("Timeframe", ["1d", "4h", "1h"], index=0)
limit = st.sidebar.number_input("OHLCV limit", value=90, min_value=2, max_value=2000)
sleep_seconds = st.sidebar.number_input("Sleep (s)", value=0.2)

# ---------------- Session ----------------
for key in ["final_df", "last_fetch"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ---------------- Helpers ----------------
@st.cache_data
def read_theme_excel(path):
    return pd.read_excel(path, engine="openpyxl")

@st.cache_resource
def get_exchange():
    ex = ccxt.kucoin()
    ex.load_markets()
    return ex

def fetch_prices(exchange, symbols, timeframe, limit):
    frames = []
    for sym in symbols:
        try:
            ohlcv = exchange.fetch_ohlcv(f"{sym}/USDT", timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=["ts","o","h","l","c","v"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms")
            frames.append(df.set_index("ts")["c"].rename(sym))
            time.sleep(sleep_seconds)
        except:
            continue
    return pd.concat(frames, axis=1)

def compute_returns(price_df):
    lookbacks = [1,3,5,10,15,30,60]
    results = {}
    for lb in lookbacks:
        results[f"{lb}d"] = (price_df.iloc[-1] - price_df.iloc[-1-lb]) / price_df.iloc[-1-lb]
    return pd.DataFrame(results)

# ---------------- Styling ----------------
def style_excess(df, top_n, bottom_n):
    styled = pd.DataFrame('', index=df.index, columns=df.columns)
    excess_cols = [c for c in df.columns if c.endswith("_Excess")]

    for col in excess_cols:
        s = df[col].dropna()
        if s.empty:
            continue

        top_idx = s.nlargest(top_n).index
        bot_idx = s.nsmallest(bottom_n).index

        for i in top_idx:
            styled.loc[i, col] = "font-weight: bold; color: white;"
        for i in bot_idx:
            styled.loc[i, col] = "font-weight: bold; color: white;"

    return styled

def apply_gradient(df):
    excess_cols = [c for c in df.columns if c.endswith("_Excess")]
    numeric_cols = df.select_dtypes(include="number").columns

    return (
        df.style
        .background_gradient(cmap="RdYlGn", subset=excess_cols)
        .set_properties(**{"color": "white"})  # ✅ WHITE TEXT
        .format({col: "{:.2f}" for col in numeric_cols})
    )

# ---------------- Fetch ----------------
if st.sidebar.button("🔄 Fetch Data"):
    df_map = read_theme_excel(default_excel_relpath)

    symbols = df_map["Symbol"].str.upper().tolist()
    themes = df_map["Theme"].tolist()
    mapping = dict(zip(symbols, themes))

    ex = get_exchange()
    with st.spinner("Fetching prices..."):
        prices = fetch_prices(ex, symbols, timeframe, limit)

    returns = compute_returns(prices)
    returns["Theme"] = returns.index.map(mapping)

    # Theme avg
    theme_avg = returns.groupby("Theme").mean()

    # Excess vs theme median
    theme_median = returns.groupby("Theme").transform("median")
    excess = returns.drop(columns="Theme") - theme_median.drop(columns="Theme")
    excess.columns = [c + "_Excess" for c in excess.columns]

    final = pd.concat([returns.drop(columns="Theme"), excess], axis=1)
    final["Theme"] = returns["Theme"]
    final["Coin"] = final.index

    # Theme-level excess vs global
    global_avg = returns.drop(columns="Theme").mean()
    theme_excess = theme_avg - global_avg
    theme_excess.columns = [c + "_Excess" for c in theme_excess.columns]

    theme_final = pd.concat([theme_avg, theme_excess], axis=1)
    theme_final["Coin"] = theme_final.index + "_average"
    theme_final["Theme"] = theme_final.index

    final = pd.concat([final, theme_final], ignore_index=True)

    # Convert to %
    final = final * 100

    # Round numeric columns
    numeric_cols = final.select_dtypes(include="number").columns
    final[numeric_cols] = final[numeric_cols].round(2)

    st.session_state.final_df = final
    st.session_state.last_fetch = datetime.utcnow()

# ---------------- Display ----------------
if st.session_state.final_df is None:
    st.info("Click Fetch Data")
else:
    df = st.session_state.final_df.copy()

    avg = df[df["Coin"].str.endswith("_average")]
    coins = df[~df["Coin"].str.endswith("_average")]

    st.subheader("Theme Averages")

    styled_avg = apply_gradient(avg).apply(
        style_excess, top_n=3, bottom_n=3, axis=None
    )
    st.dataframe(styled_avg, use_container_width=True)

    st.subheader("Coins")

    styled_coins = apply_gradient(coins).apply(
        style_excess, top_n=15, bottom_n=15, axis=None
    )
    st.dataframe(styled_coins, use_container_width=True)

st.success("Ready.")
