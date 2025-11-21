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
start_dt = st.sidebar.date_input("Start date", value=pd.to_datetime("2024-12-31"))
end_dt = st.sidebar.date_input("End date", value=pd.to_datetime("2025-11-10"))

default_excel_relpath = Path("Input-Files") / "Themes_mapping.xlsx"
st.sidebar.markdown("**Excel file (read-only, relative path)**")
st.sidebar.write(f"Using: `{default_excel_relpath}`")

timeframe = st.sidebar.selectbox("Timeframe", options=["1d", "4h", "1h"], index=0)
limit = st.sidebar.number_input("OHLCV limit (most recent bars)", value=90, min_value=2, max_value=1000)
sleep_seconds = st.sidebar.number_input("Sleep between calls (s)", value=0.2, min_value=0.0, step=0.05)

show_price_theme = st.sidebar.checkbox("Show price_theme (wide prices)", value=False)
show_returns_df = st.sidebar.checkbox("Show returns_df (raw returns)", value=False)

@st.cache_data(show_spinner=False)
def read_theme_excel(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    return df

@st.cache_resource
def get_exchange():
    exchange = ccxt.kucoin()
    try:
        _ = exchange.load_markets()
    except Exception:
        pass
    return exchange

def fetch_price_theme(exchange, theme_list, timeframe="1d", limit=90, sleep_seconds=0.2):
    frames = []
    pairs = [f"{sym}/USDT" for sym in theme_list]

    try:
        markets = exchange.load_markets()
        available_pairs = set(markets.keys())
    except Exception:
        available_pairs = set()

    valid_pairs = [pair for pair in pairs if (not available_pairs) or (pair in available_pairs)]
    if not valid_pairs:
        valid_pairs = pairs

    progress = st.progress(0)
    total = len(valid_pairs) or 1
    i = 0
    for pair in valid_pairs:
        i += 1
        try:
            base = pair.split('/')[0]
            ohlcv = exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)
            df_ohlc = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_ohlc['timestamp'] = pd.to_datetime(df_ohlc['timestamp'], unit='ms')
            df_ohlc = df_ohlc[['timestamp', 'close']].rename(columns={'close': base})
            frames.append(df_ohlc.set_index('timestamp'))
        except Exception as e:
            st.sidebar.write(f"⚠️ Failed fetch for {pair}: {e}")
        finally:
            progress.progress(min(i / total, 1.0))
            time.sleep(sleep_seconds)

    if not frames:
        return pd.DataFrame()

    price_theme = pd.concat(frames, axis=1).sort_index()
    price_theme = price_theme.dropna(axis=1, how='all')
    return price_theme

def compute_final_df(price_theme, theme_list, themes, timeframe):
    if price_theme.empty:
        return pd.DataFrame()

    # ---------- New logic for timeframe-aware lookbacks & labels ----------
    # When timeframe is daily -> keep day-based lookbacks as before
    # When timeframe is hourly (1h or 4h) -> use a set of lookback *bars* and label columns in hours

    # Base bars to use for hourly timeframes (these are counts of bars)
    bars_list = [1, 2, 3, 4, 6, 12, 24]  # will produce 1H,2H,3H,... for 1h timeframe

    if timeframe.lower().endswith('h'):
        # e.g., '1h' -> 1, '4h' -> 4
        timeframe_hours = int(timeframe.lower().replace('h', ''))
        # column display labels will be number_of_hours with H suffix
        period_keys = [f"{(b * timeframe_hours)}H" for b in bars_list]  # e.g. 4H,8H,...
        # Use internal keys as bars (we will compute returns based on bars)
        internal_bars = bars_list
        is_hourly = True
    else:
        # daily case: keep original day lookbacks
        lookbacks_days = [1, 3, 5, 10, 15, 30, 60]
        period_keys = [f"{lb}d" for lb in lookbacks_days]
        internal_bars = None
        is_hourly = False

    # ---------- compute returns ----------
    results = {}
    target_idx = len(price_theme) - 1
    # current price vector (last row)
    current = price_theme.iloc[target_idx]

    if is_hourly:
        # For each bar-count, pick the row that is N bars before the last row: index -1 - N
        for b, key in zip(internal_bars, period_keys):
            idx = target_idx - b
            if idx >= 0:
                past = price_theme.iloc[idx]
            else:
                past = price_theme.iloc[0]  # fallback if not enough history
            returns = (current - past) / past
            results[key] = returns
    else:
        # daily lookbacks: use time delta in days (as original)
        for key in period_keys:
            days = int(key.replace('d', ''))
            past_date = price_theme.index[-1] - pd.Timedelta(days=days)
            if past_date in price_theme.index:
                past = price_theme.loc[past_date]
            else:
                # fallback to N rows before end: use iloc if enough rows exist
                if len(price_theme) > days:
                    past = price_theme.iloc[-1 - days]
                else:
                    past = price_theme.iloc[0]
            returns = (current - past) / past
            results[key] = returns

    returns_df = pd.DataFrame(results)
    returns_df.index.name = 'Coin'

    # map tickers to themes
    ticker_to_theme = dict(zip(theme_list, themes))
    remaining_tickers = returns_df.T.columns.to_list()

    theme_values = []
    for t in remaining_tickers:
        theme_values.append(ticker_to_theme.get(t, ticker_to_theme.get(t.upper(), "Unknown")))

    returns_df['Theme'] = theme_values
    returns_df['Coin'] = returns_df.index

    returns_df = returns_df.drop_duplicates(subset=['Coin', 'Theme'])

    # Melt & compute theme medians and excess returns (works with either H or d keys)
    lookback_cols = list(results.keys())

    df_long = returns_df.melt(id_vars=['Coin', 'Theme'], value_vars=lookback_cols,
                              var_name='Period', value_name='Return')

    df_long['Theme_Avg'] = df_long.groupby(['Theme', 'Period'])['Return'].transform('median')
    df_long['Excess_Return'] = df_long['Return'] - df_long['Theme_Avg']

    returns_wide = df_long.pivot(index=['Coin', 'Theme'], columns='Period', values='Return')
    excess_wide = df_long.pivot(index=['Coin', 'Theme'], columns='Period', values='Excess_Return')

    # rename excess cols to include suffix similarly to original (e.g. '1H_Excess')
    excess_wide.columns = [f"{col}_Excess" for col in excess_wide.columns]

    final_df = pd.concat([returns_wide, excess_wide], axis=1)

    final_df = final_df * 100
    final_df = final_df.round(2)

    # Interleave base and excess columns in display order
    sorted_display_cols = []
    for base in lookback_cols:
        sorted_display_cols.append(base)
        excess_label = f"{base}_Excess"
        if excess_label in final_df.columns:
            sorted_display_cols.append(excess_label)
    sorted_display_cols = [c for c in sorted_display_cols if c in final_df.columns]
    final_df = final_df[sorted_display_cols]

    final_df = final_df.reset_index()
    final_df = final_df.sort_values(by=['Theme', 'Coin'])

    # Theme averages (compute from returns_df using the same lookback keys)
    numeric_cols_internal = lookback_cols
    # For theme averages, compute on the raw returns_df (which has columns like '1H' or '1d')
    theme_avg_returns = returns_df.groupby('Theme')[numeric_cols_internal].mean() * 100
    theme_avg_returns = theme_avg_returns.round(2)
    global_avg = returns_df[numeric_cols_internal].mean() * 100
    global_avg = global_avg.round(2)

    theme_excess = theme_avg_returns.subtract(global_avg, axis=1)
    theme_excess.columns = [f"{col}_Excess" for col in theme_excess.columns]

    theme_combined = pd.concat([theme_avg_returns, theme_excess], axis=1)
    theme_combined['Coin'] = theme_combined.index + '_average'
    theme_combined = theme_combined.reset_index()  # includes 'Theme'

    # Reorder columns to match final_df if possible
    cols_to_use = ['Coin', 'Theme'] + sorted_display_cols
    theme_combined = theme_combined[[c for c in cols_to_use if c in theme_combined.columns]]

    final_df = pd.concat([final_df, theme_combined], ignore_index=True)

    # Sort so _average rows appear first within each Theme, then others alphabetical
    final_df['is_avg'] = final_df['Coin'].astype(str).str.endswith('_average')
    final_df['avg_rank'] = (~final_df['is_avg']).astype(int)
    final_df = final_df.sort_values(by=['Theme', 'avg_rank', 'Coin']).reset_index(drop=True)
    final_df = final_df.drop(columns=['is_avg', 'avg_rank'], errors='ignore')

    return final_df

# --- Main flow ---
excel_path = default_excel_relpath
if not excel_path.exists():
    st.error(f"Excel file not found at: {excel_path}. Please upload Themes_mapping.xlsx into Input-Files/")
    st.stop()

with st.spinner("Reading theme Excel..."):
    df_themes = read_theme_excel(str(excel_path))

expected_required_cols = {"Symbol", "Theme"}
cols = set(df_themes.columns)
if not expected_required_cols.issubset(cols):
    if 'symbol' in (c.lower() for c in cols) and 'theme' in (c.lower() for c in cols):
        df_themes.columns = [c.strip().title() for c in df_themes.columns]
    else:
        st.error(f"Expected columns {expected_required_cols} not found in Excel. Found columns: {list(cols)}")
        st.stop()

theme_list = df_themes['Symbol'].tolist()
themes = df_themes['Theme'].tolist()

exchange = get_exchange()

st.subheader("Price fetch")
with st.spinner("Fetching OHLCV for coins..."):
    price_theme = fetch_price_theme(exchange, theme_list, timeframe=timeframe, limit=limit, sleep_seconds=sleep_seconds)

if price_theme.empty:
    st.error("No price data fetched for the provided symbols/pairs. Check the exchange and symbol names.")
    st.stop()

if show_price_theme:
    st.subheader("price_theme (wide)")
    st.dataframe(price_theme)

with st.spinner("Computing returns and excess returns..."):
    final_df = compute_final_df(price_theme, theme_list, themes, timeframe)

if final_df.empty:
    st.error("final_df is empty — likely due to insufficient price data.")
    st.stop()

# Theme filter in the sidebar
all_themes = sorted(final_df['Theme'].dropna().unique().tolist())
selected_themes = st.sidebar.multiselect("Filter Themes (show only selected)", options=all_themes, default=all_themes)

display_df = final_df[final_df['Theme'].isin(selected_themes)].copy()

st.subheader("final_df (returns + excess returns)")
st.dataframe(display_df)

if show_returns_df:
    st.subheader("Raw returns_df (latest target date)")
    # small reuse of compute logic for debug display
    st.dataframe(display_df[display_df.columns.intersection(['Coin'] + list(display_df.columns[2:]))])

st.success("Done.")
