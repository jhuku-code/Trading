# app.py
import time
from pathlib import Path
from datetime import datetime

import ccxt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Theme Returns Dashboard", layout="wide")
st.title("Theme Returns Dashboard")

# ---------------- Sidebar controls ----------------
st.sidebar.header("Settings")

start_dt = st.sidebar.date_input("Start date (info only)", value=pd.to_datetime("2024-12-31"))
end_dt = st.sidebar.date_input("End date (info only)", value=pd.to_datetime("2025-11-10"))

DEFAULT_EXCEL = Path("Input-Files") / "Themes_mapping.xlsx"
st.sidebar.markdown("**Excel file (read-only, relative path)**")
st.sidebar.write(f"Using: `{DEFAULT_EXCEL}`")

timeframe = st.sidebar.selectbox("Timeframe", options=["1d", "4h", "1h"], index=0)
limit = st.sidebar.number_input("OHLCV limit (most recent bars)", value=90, min_value=2, max_value=2000)
sleep_seconds = st.sidebar.number_input("Sleep between calls (s)", value=0.2, min_value=0.0, step=0.05)

show_price_theme = st.sidebar.checkbox("Show price_theme (wide prices)", value=False)
show_returns_df = st.sidebar.checkbox("Show returns_df (raw returns)", value=False)

# ---------------- Session state initialization ----------------
if "price_theme" not in st.session_state:
    st.session_state.price_theme = None
if "final_df" not in st.session_state:
    st.session_state.final_df = None
if "last_fetch" not in st.session_state:
    st.session_state.last_fetch = None
if "fetch_status" not in st.session_state:
    st.session_state.fetch_status = "Idle"
if "failures" not in st.session_state:
    st.session_state.failures = []

# ---------------- Helpers ----------------
@st.cache_data(show_spinner=False)
def read_theme_excel(path: str) -> pd.DataFrame:
    """Load Themes_mapping.xlsx (cached to avoid repeated disk IO)."""
    return pd.read_excel(path, engine="openpyxl")

@st.cache_resource
def get_exchange():
    """Initialize exchange object once per session."""
    exchange = ccxt.kucoin()  # keep kucoin as in your original; switch to ccxt.binance() if desired
    try:
        _ = exchange.load_markets()
    except Exception:
        # ignore load errors - we handle per-pair fetch exceptions later
        pass
    return exchange

def fetch_price_theme(exchange, theme_list, timeframe="1d", limit=90, sleep_seconds=0.2):
    """
    Fetch OHLCV for every symbol in theme_list from the exchange.
    Returns (price_theme_df, failures_list).
    """
    frames = []
    pairs = [f"{sym}/USDT" for sym in theme_list]

    try:
        markets = exchange.load_markets()
        available_pairs = set(markets.keys())
    except Exception:
        available_pairs = set()

    # candidate pairs: prefer those in available_pairs if we can list markets
    candidate_pairs = [pair for pair in pairs if (not available_pairs) or (pair in available_pairs)]
    if not candidate_pairs:
        candidate_pairs = pairs

    total = len(candidate_pairs) or 1
    progress = st.progress(0)
    i = 0
    failures = []

    for pair in candidate_pairs:
        i += 1
        try:
            base = pair.split('/')[0]
            ohlcv = exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)
            df_ohlc = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_ohlc['timestamp'] = pd.to_datetime(df_ohlc['timestamp'], unit='ms')
            df_ohlc = df_ohlc[['timestamp', 'close']].rename(columns={'close': base})
            frames.append(df_ohlc.set_index('timestamp'))
        except Exception as e:
            failures.append((pair, str(e)))
        finally:
            progress.progress(min(i / total, 1.0))
            time.sleep(sleep_seconds)

    if not frames:
        return pd.DataFrame(), failures

    price_theme = pd.concat(frames, axis=1).sort_index()
    price_theme = price_theme.dropna(axis=1, how='all')
    return price_theme, failures

def compute_final_df(price_theme, theme_list, themes, timeframe):
    """
    Compute final_df with returns and excess returns.
    timeframe-aware: for '1d' uses day lookbacks; for '1h'/'4h' uses bar-based lookbacks and H labels.
    """
    if price_theme is None or price_theme.empty:
        return pd.DataFrame()

    # Decide lookbacks + labels
    if timeframe.lower().endswith('h'):
        # Bars we want to look back (counts of bars)
        bars_list = [1, 2, 3, 4, 6, 12, 24]  # you can modify this if you want different buckets
        timeframe_hours = int(timeframe.lower().replace('h', ''))
        period_keys = [f"{(b * timeframe_hours)}H" for b in bars_list]  # e.g., 4H,8H for 4h timeframe
        is_hourly = True
        internal_bars = bars_list
    else:
        lookbacks_days = [1, 3, 5, 10, 15, 30, 60]
        period_keys = [f"{lb}d" for lb in lookbacks_days]
        is_hourly = False
        internal_bars = None

    # compute returns relative to last available bar
    results = {}
    last_idx = len(price_theme) - 1
    current = price_theme.iloc[last_idx]

    if is_hourly:
        for b, key in zip(internal_bars, period_keys):
            idx = last_idx - b
            if idx >= 0:
                past = price_theme.iloc[idx]
            else:
                past = price_theme.iloc[0]
            results[key] = (current - past) / past
    else:
        for key in period_keys:
            days = int(key.replace('d', ''))
            past_date = price_theme.index[-1] - pd.Timedelta(days=days)
            if past_date in price_theme.index:
                past = price_theme.loc[past_date]
            else:
                if len(price_theme) > days:
                    past = price_theme.iloc[-1 - days]
                else:
                    past = price_theme.iloc[0]
            results[key] = (current - past) / past

    # Build returns_df
    returns_df = pd.DataFrame(results)
    returns_df.index.name = 'Coin'

    # Map tickers to themes (be forgiving about case and whitespace)
    ticker_to_theme = dict(zip([str(t).strip() for t in theme_list], [str(t).strip() for t in themes]))
    remaining_tickers = returns_df.T.columns.to_list()

    theme_values = []
    for t in remaining_tickers:
        t_stripped = str(t).strip()
        theme_values.append(ticker_to_theme.get(t_stripped, ticker_to_theme.get(t_stripped.upper(), "Unknown")))

    returns_df['Theme'] = theme_values
    returns_df['Coin'] = returns_df.index
    returns_df = returns_df.drop_duplicates(subset=['Coin', 'Theme'])

    lookback_cols = list(results.keys())

    # Melt, compute theme medians and excess returns
    df_long = returns_df.melt(id_vars=['Coin', 'Theme'], value_vars=lookback_cols,
                              var_name='Period', value_name='Return')
    df_long['Theme_Avg'] = df_long.groupby(['Theme', 'Period'])['Return'].transform('median')
    df_long['Excess_Return'] = df_long['Return'] - df_long['Theme_Avg']

    returns_wide = df_long.pivot(index=['Coin', 'Theme'], columns='Period', values='Return')
    excess_wide = df_long.pivot(index=['Coin', 'Theme'], columns='Period', values='Excess_Return')
    # make excess column names consistent like '4H_Excess' or '1d_Excess'
    excess_wide.columns = [f"{col}_Excess" for col in excess_wide.columns]

    final_df = pd.concat([returns_wide, excess_wide], axis=1)
    final_df = final_df * 100
    final_df = final_df.round(2)

    # Interleave base and excess columns for display
    sorted_display_cols = []
    for base in lookback_cols:
        sorted_display_cols.append(base)
        excess_label = f"{base}_Excess"
        if excess_label in final_df.columns:
            sorted_display_cols.append(excess_label)
    sorted_display_cols = [c for c in sorted_display_cols if c in final_df.columns]
    final_df = final_df[sorted_display_cols]

    final_df = final_df.reset_index()

    # Theme averages and excess relative to global average
    theme_avg_returns = returns_df.groupby('Theme')[lookback_cols].mean() * 100
    theme_avg_returns = theme_avg_returns.round(2)
    global_avg = returns_df[lookback_cols].mean() * 100
    global_avg = global_avg.round(2)
    theme_excess = theme_avg_returns.subtract(global_avg, axis=1)
    theme_excess.columns = [f"{col}_Excess" for col in theme_excess.columns]

    theme_combined = pd.concat([theme_avg_returns, theme_excess], axis=1)
    theme_combined['Coin'] = theme_combined.index + '_average'
    theme_combined = theme_combined.reset_index()  # 'Theme' becomes a column

    # Align theme_combined columns to final_df where possible
    cols_to_use = ['Coin', 'Theme'] + sorted_display_cols
    theme_combined = theme_combined[[c for c in cols_to_use if c in theme_combined.columns]]

    # Append theme averages to final_df
    final_df = pd.concat([final_df, theme_combined], ignore_index=True)

    # Normalize Theme column
    final_df['Theme'] = final_df['Theme'].fillna('Unknown').astype(str).str.strip()

    # Sort so that within each Theme, averages appear first, then coins alphabetically
    final_df['is_avg'] = final_df['Coin'].astype(str).str.endswith('_average')
    final_df['avg_rank'] = (~final_df['is_avg']).astype(int)  # avg -> 0, others -> 1
    final_df = final_df.sort_values(by=['Theme', 'avg_rank', 'Coin']).reset_index(drop=True)
    final_df = final_df.drop(columns=['is_avg', 'avg_rank'], errors='ignore')

    return final_df

# ---------------- UI: Fetch button & status ----------------
st.sidebar.markdown("---")
st.sidebar.markdown("### Data control (manual)")

if st.sidebar.button("🔄 Refresh / Fetch Data"):
    excel_path = DEFAULT_EXCEL
    if not excel_path.exists():
        st.error(f"Excel file not found at: {excel_path}. Upload Themes_mapping.xlsx into Input-Files/")
    else:
        st.session_state.fetch_status = "Fetching"
        try:
            df_themes = read_theme_excel(str(excel_path))
            cols_lower = [c.lower() for c in df_themes.columns]
            if not ("symbol" in cols_lower and "theme" in cols_lower):
                st.error("Excel must contain 'Symbol' and 'Theme' columns (case-insensitive).")
                st.session_state.fetch_status = "Idle"
            else:
                # Normalize column names and extract symbol/theme columns
                symbol_col = next(c for c in df_themes.columns if c.lower() == 'symbol')
                theme_col = next(c for c in df_themes.columns if c.lower() == 'theme')
                theme_list = df_themes[symbol_col].astype(str).str.strip().tolist()
                themes = df_themes[theme_col].astype(str).str.strip().tolist()

                exchange = get_exchange()
                with st.spinner("Fetching OHLCV from exchange (this only runs on Refresh)..."):
                    price_theme, failures = fetch_price_theme(exchange, theme_list, timeframe=timeframe, limit=limit, sleep_seconds=sleep_seconds)

                st.session_state.price_theme = price_theme
                st.session_state.failures = failures
                st.session_state.last_fetch = datetime.utcnow()
                st.session_state.fetch_status = "Computing"

                with st.spinner("Computing final_df (returns & excess returns)..."):
                    st.session_state.final_df = compute_final_df(price_theme, theme_list, themes, timeframe)

                st.session_state.fetch_status = "Done"

                # show failed pairs in sidebar
                if failures:
                    st.sidebar.warning(f"Failed to fetch {len(failures)} pair(s). See list below.")
                    for pair, err in failures:
                        st.sidebar.write(f"⚠️ {pair}: {err}")

        except Exception as e:
            st.exception(e)
            st.session_state.fetch_status = "Idle"

# show fetch status / last fetch
st.sidebar.markdown("")
st.sidebar.write(f"**Fetch status:** {st.session_state.fetch_status}")
if st.session_state.last_fetch:
    st.sidebar.write(f"**Last fetched (UTC):** {st.session_state.last_fetch.strftime('%Y-%m-%d %H:%M:%S')}")

st.sidebar.markdown("---")
st.sidebar.markdown("Note: Changing filters/timeframe does NOT trigger network fetch. Click Refresh to re-run download & compute.")

# ---------------- Main display ----------------
st.subheader("final_df (returns + excess returns)")

if st.session_state.final_df is None or st.session_state.final_df.empty:
    st.info("No data available. Click **Refresh / Fetch Data** in the sidebar to download price data and compute final_df.")
    # preview Excel if present
    if DEFAULT_EXCEL.exists():
        try:
            st.subheader("Themes_mapping.xlsx preview (top 50 rows)")
            st.dataframe(read_theme_excel(str(DEFAULT_EXCEL)).head(50))
        except Exception:
            pass
else:
    final_df = st.session_state.final_df.copy()

    # Clean Theme column for filter UI
    final_df['Theme'] = final_df['Theme'].fillna('Unknown').astype(str).str.strip()

    # Build Theme selector from final_df (robust)
    all_themes = sorted(final_df['Theme'].unique().tolist())
    st.sidebar.markdown("### Theme filter")
    select_all = st.sidebar.checkbox("Select all themes (for display)", value=True, key="select_all_themes_main")
    if select_all:
        selected_themes = st.sidebar.multiselect("Choose themes to display", options=all_themes, default=all_themes, key="theme_multiselect_main")
    else:
        selected_themes = st.sidebar.multiselect("Choose themes to display", options=all_themes, default=[], key="theme_multiselect_main")

    if not selected_themes:
        st.warning("No themes selected. Use the sidebar to select one or more themes to display.")
        display_df = final_df.iloc[0:0]
    else:
        display_df = final_df[final_df['Theme'].isin(selected_themes)].copy()

        # ensure sorting: theme, averages first, then alphabetical coin
        display_df['is_avg'] = display_df['Coin'].astype(str).str.endswith('_average')
        display_df['avg_rank'] = (~display_df['is_avg']).astype(int)
        display_df = display_df.sort_values(by=['Theme', 'avg_rank', 'Coin']).reset_index(drop=True)
        display_df = display_df.drop(columns=['is_avg', 'avg_rank'], errors='ignore')

    # Split averages into its own DataFrame and coins into another DataFrame
    averages_df = display_df[display_df['Coin'].astype(str).str.endswith('_average')].copy()
    coins_df = display_df[~display_df['Coin'].astype(str).str.endswith('_average')].copy()

    # Show counts and the dataframes
    st.write(f"Showing Theme(s): {', '.join(selected_themes) if selected_themes else 'None'}")
    st.write(f"Averages rows: {len(averages_df)} — Coin rows: {len(coins_df)} — Total rows: {len(display_df)}")

    if not averages_df.empty:
        st.subheader("Theme averages (rows with Coin ending '_average')")
        st.dataframe(averages_df)
    else:
        st.info("No theme-average rows to display for the selected themes.")

    if not coins_df.empty:
        st.subheader("Coin rows (non-average coins)")
        st.dataframe(coins_df)
    else:
        st.info("No coin rows to display for the selected themes.")

    # Optional debug views
    if show_price_theme and st.session_state.price_theme is not None:
        st.subheader("price_theme (wide)")
        st.dataframe(st.session_state.price_theme)

    if show_returns_df:
        st.subheader("Raw returns_df (latest target date) — sample")
        # Show a compact sample view of returns columns
        lookback_cols = [c for c in display_df.columns if (c.endswith('H') or c.endswith('d')) and not c.endswith('_Excess')]
        cols_to_show = ['Coin'] + lookback_cols
        st.dataframe(display_df[display_df.columns.intersection(cols_to_show)].head(200))

st.success("Ready.")
