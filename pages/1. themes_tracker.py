# app.py
import time
from pathlib import Path

import ccxt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Theme Returns Dashboard", layout="wide")

st.title("Theme Returns Dashboard")

# --- Sidebar controls ---
st.sidebar.header("Settings")

# Date inputs (kept for future use; not used directly in current return calc)
start_dt = st.sidebar.date_input("Start date", value=pd.to_datetime("2024-12-31"))
end_dt = st.sidebar.date_input("End date", value=pd.to_datetime("2025-11-10"))

# Excel path (fixed relative folder)
default_excel_relpath = Path("Input-Files") / "Themes_mapping.xlsx"
st.sidebar.markdown("**Excel file (read-only, relative path)**")
st.sidebar.write(f"Using: `{default_excel_relpath}`")

timeframe = st.sidebar.selectbox("Timeframe", options=["1d", "4h", "1h"], index=0)
limit = st.sidebar.number_input("OHLCV limit (most recent bars)", value=90, min_value=2, max_value=1000)
sleep_seconds = st.sidebar.number_input("Sleep between calls (s)", value=0.2, min_value=0.0, step=0.05)

show_price_theme = st.sidebar.checkbox("Show price_theme (wide prices)", value=False)
show_returns_df = st.sidebar.checkbox("Show returns_df (raw returns)", value=False)

# --- Helpers with caching ---
@st.cache_data(show_spinner=False)
def read_theme_excel(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    # Normalize expected column names
    expected_cols = [c.lower() for c in df.columns]
    # If user has 'Symbol' and 'Theme' capitalized names (as in original), it's fine
    return df

@st.cache_resource
def get_exchange():
    # Using kucoin as in user's version. Change to ccxt.binance() if you prefer.
    exchange = ccxt.kucoin()
    # If needed, you can set api keys: exchange.apiKey = '...' etc.
    try:
        _ = exchange.load_markets()
    except Exception:
        # sometimes load_markets fails due to connectivity; that's okay - we'll attempt fetch_ohlcv per symbol later
        pass
    return exchange

def fetch_price_theme(exchange, theme_list, timeframe="1d", limit=90, sleep_seconds=0.2):
    frames = []
    binance_pairs = [f"{sym}/USDT" for sym in theme_list]

    # Attempt to get available pairs from exchange if load_markets works
    try:
        markets = exchange.load_markets()
        available_pairs = set(markets.keys())
    except Exception:
        available_pairs = set()

    valid_pairs = [pair for pair in binance_pairs if (not available_pairs) or (pair in available_pairs)]
    # If available_pairs was empty (load failed), try to fetch each pair anyway and handle exceptions
    if not valid_pairs:
        # If no markets info, attempt to query each pair — we'll still catch failures per-pair
        valid_pairs = binance_pairs

    progress = st.progress(0)
    total = len(valid_pairs)
    i = 0
    for pair in valid_pairs:
        i += 1
        try:
            base = pair.split('/')[0]
            # fetch_ohlcv expects the pair as exchange market id; if pair not found this will raise
            ohlcv = exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)
            df_ohlc = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_ohlc['timestamp'] = pd.to_datetime(df_ohlc['timestamp'], unit='ms')
            df_ohlc = df_ohlc[['timestamp', 'close']].rename(columns={'close': base})
            frames.append(df_ohlc.set_index('timestamp'))
        except Exception as e:
            # show which pairs failed (non-blocking)
            st.sidebar.write(f"⚠️ Failed fetch for {pair}: {e}")
        finally:
            # update progress
            progress.progress(min(i / total, 1.0))
            time.sleep(sleep_seconds)

    if not frames:
        return pd.DataFrame()  # empty

    price_theme = pd.concat(frames, axis=1).sort_index()
    # drop columns that are entirely NaN (coins with no data)
    price_theme = price_theme.dropna(axis=1, how='all')
    return price_theme

def compute_final_df(price_theme, theme_list, themes):
    # safety check
    if price_theme.empty:
        return pd.DataFrame()

    # Lookbacks
    lookbacks = [1, 3, 5, 10, 15, 30, 60]

    target_date = price_theme.index[-1]

    results = {}
    for lb in lookbacks:
        current = price_theme.loc[target_date]
        # target_date - lb days
        past_date = target_date - pd.Timedelta(days=lb)
        if past_date in price_theme.index:
            past = price_theme.loc[past_date]
        else:
            # use the most recent available date before target (fallback to iloc)
            # here using .iloc[-lb] like original: that picks lb rows before end
            # but guard if there aren't enough rows
            if len(price_theme) > lb:
                past = price_theme.iloc[-(lb + 1)]  # one row before the most recent lb days
            else:
                past = price_theme.iloc[0]
        returns = (current - past) / past
        results[f'{lb}d'] = returns

    returns_df = pd.DataFrame(results)
    returns_df.index.name = 'Coin'

    # Map theme (expect theme_list aligned with original symbols)
    ticker_to_theme = dict(zip(theme_list, themes))
    remaining_tickers = returns_df.T.columns.to_list()

    theme_values = []
    for t in remaining_tickers:
        # t may be like 'BTC' or 'btc' depending on how pair names were provided
        theme_values.append(ticker_to_theme.get(t, ticker_to_theme.get(t.upper(), "Unknown")))

    returns_df['Theme'] = theme_values
    returns_df['Coin'] = returns_df.index

    # Remove duplicates
    returns_df = returns_df.drop_duplicates(subset=['Coin', 'Theme'])

    lookback_cols = [f'{lb}d' for lb in [1,3,5,10,15,30,60]]

    df_long = returns_df.melt(id_vars=['Coin', 'Theme'], value_vars=lookback_cols,
                              var_name='Period', value_name='Return')

    df_long['Theme_Avg'] = df_long.groupby(['Theme', 'Period'])['Return'].transform('median')
    df_long['Excess_Return'] = df_long['Return'] - df_long['Theme_Avg']

    returns_wide = df_long.pivot(index=['Coin', 'Theme'], columns='Period', values='Return')
    excess_wide = df_long.pivot(index=['Coin', 'Theme'], columns='Period', values='Excess_Return')

    returns_wide.columns = [f"{col}" for col in returns_wide.columns]
    excess_wide.columns = [f"{col}_Excess" for col in excess_wide.columns]

    final_df = pd.concat([returns_wide, excess_wide], axis=1)

    final_df = final_df * 100
    final_df = final_df.round(2)

    # Reorder columns
    sorted_cols = []
    for col in lookback_cols:
        sorted_cols.append(col)
        sorted_cols.append(f"{col}_Excess")
    # Some columns may be missing if data was missing; filter accordingly
    sorted_cols = [c for c in sorted_cols if c in final_df.columns]
    final_df = final_df[sorted_cols]

    final_df = final_df.reset_index()
    final_df = final_df.sort_values(by=['Theme', 'Coin'])

    # Theme-wise average returns and excess vs global average
    returns_df_for_avg = returns_df.copy()
    # Ensure lookback col names match the format in returns_df (like '1d' not '1d'…)
    # returns_df originally had columns like '1d','3d', etc.
    numeric_cols = [c for c in lookback_cols if c in returns_df_for_avg.columns]
    theme_avg_returns = returns_df_for_avg.groupby('Theme')[numeric_cols].mean() * 100
    theme_avg_returns = theme_avg_returns.round(2)
    global_avg = returns_df_for_avg[numeric_cols].mean() * 100
    global_avg = global_avg.round(2)

    theme_excess = theme_avg_returns.subtract(global_avg, axis=1)
    theme_excess.columns = [f"{col}_Excess" for col in theme_excess.columns]

    theme_combined = pd.concat([theme_avg_returns, theme_excess], axis=1)
    theme_combined['Coin'] = theme_combined.index + '_average'
    theme_combined = theme_combined.reset_index()  # 'Theme' becomes a column

    # Reorder columns to match final_df if possible
    cols_to_use = ['Coin', 'Theme'] + sorted_cols
    theme_combined = theme_combined[[c for c in cols_to_use if c in theme_combined.columns]]

    final_df = pd.concat([final_df, theme_combined], ignore_index=True)
    final_df = final_df.sort_values(by=['Theme', 'Coin']).reset_index(drop=True)

    return final_df

# --- Main flow ---
excel_path = default_excel_relpath
if not excel_path.exists():
    st.error(f"Excel file not found at: {excel_path}. Please upload Themes_mapping.xlsx into Input-Files/")
    st.stop()

with st.spinner("Reading theme Excel..."):
    df_themes = read_theme_excel(str(excel_path))

# Ensure expected columns exist
expected_required_cols = {"Symbol", "Theme"}
cols = set(df_themes.columns)
if not expected_required_cols.issubset(cols):
    st.warning(f"Expected columns {expected_required_cols} not found in Excel. Found columns: {list(cols)}")
    # Try to be forgiving: attempt to find close matches
    # But proceed only if 'Symbol' and 'Theme' can be located in any case
    if 'symbol' in (c.lower() for c in cols) and 'theme' in (c.lower() for c in cols):
        # normalize
        df_themes.columns = [c.strip().title() for c in df_themes.columns]
    else:
        st.stop()

theme_list = df_themes['Symbol'].tolist()
themes = df_themes['Theme'].tolist()

exchange = get_exchange()

# Fetch prices
st.subheader("Price fetch")
with st.spinner("Fetching OHLCV for coins..."):
    price_theme = fetch_price_theme(exchange, theme_list, timeframe=timeframe, limit=limit, sleep_seconds=sleep_seconds)

if price_theme.empty:
    st.error("No price data fetched for the provided symbols/pairs. Check the exchange and symbol names.")
    st.stop()

if show_price_theme:
    st.subheader("price_theme (wide)")
    st.dataframe(price_theme)

# Compute final_df
with st.spinner("Computing returns and excess returns..."):
    final_df = compute_final_df(price_theme, theme_list, themes)

if final_df.empty:
    st.error("final_df is empty — likely due to insufficient price data.")
else:
    st.subheader("final_df (returns + excess returns)")
    st.dataframe(final_df)

# Optional: show returns_df intermediate
if show_returns_df:
    st.subheader("Raw returns_df")
    # Recompute returns_df quickly (same logic as inside compute_final_df)
    # For brevity, show the returns for the latest target date
    lookbacks = [1,3,5,10,15,30,60]
    target_date = price_theme.index[-1]
    results = {}
    for lb in lookbacks:
        current = price_theme.loc[target_date]
        past_date = target_date - pd.Timedelta(days=lb)
        if past_date in price_theme.index:
            past = price_theme.loc[past_date]
        else:
            if len(price_theme) > lb:
                past = price_theme.iloc[-(lb + 1)]
            else:
                past = price_theme.iloc[0]
        results[f'{lb}d'] = (current - past) / past
    returns_df = pd.DataFrame(results)
    returns_df.index.name = 'Coin'
    st.dataframe(returns_df)

st.success("Done.")

