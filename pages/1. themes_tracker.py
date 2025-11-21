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

# Date inputs (not used directly in computation, kept for reference)
start_dt = st.sidebar.date_input("Start date", value=pd.to_datetime("2024-12-31"))
end_dt = st.sidebar.date_input("End date", value=pd.to_datetime("2025-11-10"))

default_excel_relpath = Path("Input-Files") / "Themes_mapping.xlsx"
st.sidebar.markdown("**Excel file (read-only, relative path)**")
st.sidebar.write(f"Using: `{default_excel_relpath}`")

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
if "price_theme_version" not in st.session_state:
    st.session_state.price_theme_version = None
if "price_timeframe" not in st.session_state:
    st.session_state.price_timeframe = None
if "theme_values" not in st.session_state:
    st.session_state.theme_values = None
if "theme_list" not in st.session_state:
    st.session_state.theme_list = None

# ---------------- Helpers ----------------
@st.cache_data(show_spinner=False)
def read_theme_excel(path: str) -> pd.DataFrame:
    return pd.read_excel(path, engine="openpyxl")

@st.cache_resource
def get_exchange():
    # keep kucoin by default as you used earlier; swap to binance if needed
    exchange = ccxt.kucoin()
    try:
        _ = exchange.load_markets()
    except Exception:
        # ignore load errors; we'll handle per-symbol fetch errors downstream
        pass
    return exchange

def fetch_price_theme(exchange, theme_list, timeframe="1d", limit=90, sleep_seconds=0.2):
    """
    theme_list: list of tickers (normalized, e.g., 'BTC', 'ETH')
    returns: (price_theme_df, failures_list)
    """
    frames = []
    pairs = [f"{sym}/USDT" for sym in theme_list]

    # try load markets to get available pairs, but continue if fails
    try:
        markets = exchange.load_markets()
        available_pairs = set(markets.keys())
    except Exception:
        available_pairs = set()

    # choose candidate pairs: if load_markets succeeded filter, else attempt all
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
            # ensure base ticker is uppercase to match Excel normalization
            base = base.upper()
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

    # Normalize column names to uppercase (defensive)
    price_theme.columns = [str(c).upper() for c in price_theme.columns]

    return price_theme, failures

def compute_final_df(price_theme, theme_list, themes, timeframe):
    if price_theme is None or price_theme.empty:
        return pd.DataFrame()

    # Determine lookback keys depending on timeframe
    if timeframe.lower().endswith('h'):
        # bars_list defines how many bars to look back (1,2,3,4,6,12,24)
        bars_list = [1, 2, 3, 4, 6, 12, 24]
        timeframe_hours = int(timeframe.lower().replace('h', ''))
        period_keys = [f"{(b * timeframe_hours)}H" for b in bars_list]  # e.g., 4H,8H
        is_hourly = True
        internal_bars = bars_list
    else:
        lookbacks_days = [1, 3, 5, 10, 15, 30, 60]
        period_keys = [f"{lb}d" for lb in lookbacks_days]
        is_hourly = False
        internal_bars = None

    # compute returns relative to last available row
    results = {}
    last_idx = len(price_theme) - 1
    current = price_theme.iloc[last_idx]

    if is_hourly:
        for b, key in zip(internal_bars, period_keys):
            idx = last_idx - b
            if idx >= 0:
                past = price_theme.iloc[idx]
            else:
                # fallback to earliest available
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

    returns_df = pd.DataFrame(results)
    returns_df.index.name = 'Coin'

    # Map coin -> theme (be forgiving on case)
    # theme_list should already be normalized to match price_theme columns (uppercase)
    ticker_to_theme = dict(zip(theme_list, themes))
    remaining_tickers = returns_df.T.columns.to_list()
    theme_values = []
    for t in remaining_tickers:
        theme_values.append(ticker_to_theme.get(t, ticker_to_theme.get(t.upper(), "Unknown")))

    returns_df['Theme'] = theme_values
    returns_df['Coin'] = returns_df.index
    returns_df = returns_df.drop_duplicates(subset=['Coin', 'Theme'])

    lookback_cols = list(results.keys())

    df_long = returns_df.melt(id_vars=['Coin', 'Theme'], value_vars=lookback_cols,
                              var_name='Period', value_name='Return')
    df_long['Theme_Avg'] = df_long.groupby(['Theme', 'Period'])['Return'].transform('median')
    df_long['Excess_Return'] = df_long['Return'] - df_long['Theme_Avg']

    returns_wide = df_long.pivot(index=['Coin', 'Theme'], columns='Period', values='Return')
    excess_wide = df_long.pivot(index=['Coin', 'Theme'], columns='Period', values='Excess_Return')
    excess_wide.columns = [f"{col}_Excess" for col in excess_wide.columns]

    final_df = pd.concat([returns_wide, excess_wide], axis=1)
    final_df = final_df * 100
    final_df = final_df.round(2)

    # Interleave base & excess columns
    sorted_display_cols = []
    for base in lookback_cols:
        sorted_display_cols.append(base)
        excess_label = f"{base}_Excess"
        if excess_label in final_df.columns:
            sorted_display_cols.append(excess_label)
    sorted_display_cols = [c for c in sorted_display_cols if c in final_df.columns]
    final_df = final_df[sorted_display_cols]

    final_df = final_df.reset_index()

    # Theme averages + excess vs global
    theme_avg_returns = returns_df.groupby('Theme')[lookback_cols].mean() * 100
    theme_avg_returns = theme_avg_returns.round(2)
    global_avg = returns_df[lookback_cols].mean() * 100
    global_avg = global_avg.round(2)

    theme_excess = theme_avg_returns.subtract(global_avg, axis=1)
    theme_excess.columns = [f"{col}_Excess" for col in theme_excess.columns]

    theme_combined = pd.concat([theme_avg_returns, theme_excess], axis=1)
    theme_combined['Coin'] = theme_combined.index + '_average'
    theme_combined = theme_combined.reset_index()  # brings 'Theme' as column

    # Reorder columns to match final_df if possible
    cols_to_use = ['Coin', 'Theme'] + sorted_display_cols
    theme_combined = theme_combined[[c for c in cols_to_use if c in theme_combined.columns]]

    final_df = pd.concat([final_df, theme_combined], ignore_index=True)

    # Sort: for each Theme, put _average first, then coins alphabetically
    final_df['Theme'] = final_df['Theme'].fillna('Unknown').astype(str).str.strip()
    final_df['is_avg'] = final_df['Coin'].astype(str).str.endswith('_average')
    final_df['avg_rank'] = (~final_df['is_avg']).astype(int)  # avg -> 0, others -> 1
    final_df = final_df.sort_values(by=['Theme', 'avg_rank', 'Coin']).reset_index(drop=True)
    final_df = final_df.drop(columns=['is_avg', 'avg_rank'], errors='ignore')

    return final_df

# ---------------- UI: Fetch button & status ----------------
st.sidebar.markdown("---")
st.sidebar.markdown("### Data control")

# explicit refresh button - only action that triggers fetch+compute
if st.sidebar.button("🔄 Refresh / Fetch Data"):
    # validate excel exists first
    excel_path = default_excel_relpath
    if not excel_path.exists():
        st.error(f"Excel file not found at: {excel_path}. Upload Themes_mapping.xlsx into Input-Files/")
    else:
        st.session_state.fetch_status = "Fetching"
        try:
            df_themes = read_theme_excel(str(excel_path))
            # ensure required columns exist
            cols_lower = [c.lower() for c in df_themes.columns]
            if not ("symbol" in cols_lower and "theme" in cols_lower):
                st.error("Excel must contain 'Symbol' and 'Theme' columns (case-insensitive).")
                st.session_state.fetch_status = "Idle"
            else:
                # normalize column names to 'Symbol' and 'Theme' if necessary
                df_themes.columns = [c.strip() for c in df_themes.columns]
                # Map correct columns (case-insensitive)
                symbol_col = next(c for c in df_themes.columns if c.lower() == 'symbol')
                theme_col = next(c for c in df_themes.columns if c.lower() == 'theme')

                # Normalize symbols to uppercase to match fetch naming; preserve theme names as provided (trimmed)
                theme_list_raw = df_themes[symbol_col].astype(str).str.strip().tolist()
                themes_raw = df_themes[theme_col].astype(str).str.strip().tolist()

                theme_list = [s.upper() for s in theme_list_raw]
                themes = [t for t in themes_raw]

                # start fetch
                exchange = get_exchange()
                with st.spinner("Fetching OHLCV from exchange (this only runs on Refresh)..."):
                    price_theme, failures = fetch_price_theme(exchange, theme_list, timeframe=timeframe, limit=limit, sleep_seconds=sleep_seconds)

                # store and compute
                st.session_state.price_theme = price_theme
                st.session_state.last_fetch = datetime.utcnow()
                st.session_state.fetch_status = "Computing"
                with st.spinner("Computing final_df (returns & excess returns)..."):
                    st.session_state.final_df = compute_final_df(price_theme, theme_list, themes, timeframe)
                st.session_state.fetch_status = "Done"

                # --- NEW: expose the price_theme + theme mapping + friendly timeframe to other pages ---
                # Ensure price_theme columns are uppercase strings
                if price_theme is not None and not price_theme.empty:
                    cols = [str(c).upper() for c in price_theme.columns]
                    # rename to canonical uppercase (defensive)
                    price_theme.columns = cols
                    st.session_state.price_theme = price_theme
                else:
                    cols = theme_list  # fallback if fetch failed

                # Create mapping ticker -> Theme (case-insensitive)
                ticker_to_theme = {t: th for t, th in zip(theme_list, themes)}
                ticker_to_theme_upper = {t.upper(): th for t, th in zip(theme_list, themes)}

                # Create theme_values list aligned to price_theme.columns
                theme_values_aligned = []
                for c in cols:
                    theme_values_aligned.append(
                        ticker_to_theme.get(c, ticker_to_theme_upper.get(c.upper(), "Unknown"))
                    )

                # Save to session_state: other page will read these keys
                st.session_state['price_theme'] = st.session_state.get('price_theme')  # DataFrame of wide prices
                st.session_state['theme_values'] = theme_values_aligned  # list aligned to columns order
                st.session_state['theme_list'] = cols  # list of tickers representing columns order

                # Save a friendly timeframe label for the charting page
                tf_map = {"1d": "daily", "4h": "4hourly", "1h": "1hourly"}
                st.session_state['price_timeframe'] = tf_map.get(timeframe, timeframe)

                # Optional: a small version token so other pages can detect updates and clear cache
                st.session_state['price_theme_version'] = datetime.utcnow().timestamp()

                # show any failed pairs in sidebar
                if failures:
                    st.sidebar.warning(f"Failed to fetch {len(failures)} pair(s). See sidebar for details.")
                    for pair, err in failures:
                        st.sidebar.write(f"⚠️ {pair}: {err}")
        except Exception as e:
            st.exception(e)
            st.session_state.fetch_status = "Idle"

# Show current fetch status + last fetch time
st.sidebar.markdown("")
st.sidebar.write(f"**Fetch status:** {st.session_state.fetch_status}")
if st.session_state.last_fetch:
    st.sidebar.write(f"**Last fetched (UTC):** {st.session_state.last_fetch.strftime('%Y-%m-%d %H:%M:%S')}")

st.sidebar.markdown("---")
st.sidebar.markdown("Note: Changing filters/timeframe does NOT trigger network fetch. Click Refresh to re-run download & compute.")

# ---------------- Main display ----------------
st.subheader("final_df (returns + excess returns)")

# if we have computed final_df, show filter widgets and data; else show message
if st.session_state.final_df is None or st.session_state.final_df.empty:
    st.info("No data available. Click **Refresh / Fetch Data** in the sidebar to download price data and compute final_df.")
    # allow user to still upload or inspect excel if desired
    excel_path = default_excel_relpath
    if excel_path.exists():
        try:
            df_preview = read_theme_excel(str(excel_path))
            st.subheader("Themes_mapping.xlsx preview")
            st.dataframe(df_preview.head(50))
        except Exception:
            pass
else:
    final_df = st.session_state.final_df.copy()

    # Normalize Theme column for robust filtering
    final_df['Theme'] = final_df['Theme'].fillna('Unknown').astype(str).str.strip()

    # Theme multiselect built from final_df (cleaned)
    all_themes = sorted(final_df['Theme'].unique().tolist())
    # convenience select-all checkbox
    select_all = st.sidebar.checkbox("Select all themes (for display)", value=True, key="select_all_themes")
    if select_all:
        selected_themes = st.sidebar.multiselect("Choose themes to display", options=all_themes, default=all_themes, key="theme_multiselect")
    else:
        selected_themes = st.sidebar.multiselect("Choose themes to display", options=all_themes, default=[], key="theme_multiselect")

    if not selected_themes:
        st.warning("No themes selected. Use the sidebar to select one or more themes to display.")
        display_df = final_df.iloc[0:0]
    else:
        display_df = final_df[final_df['Theme'].isin(selected_themes)].copy()
        # maintain the sorting: theme, averages first, then alphabetical coin
        display_df['is_avg'] = display_df['Coin'].astype(str).str.endswith('_average')
        display_df['avg_rank'] = (~display_df['is_avg']).astype(int)
        display_df = display_df.sort_values(by=['Theme', 'avg_rank', 'Coin']).reset_index(drop=True)
        display_df = display_df.drop(columns=['is_avg', 'avg_rank'], errors='ignore')

    st.write(f"Showing {len(display_df)} rows for {len(selected_themes)} theme(s).")
    # show the dataframe
    st.dataframe(display_df)

    # optional debug views
    if show_price_theme:
        st.subheader("price_theme (wide)")
        st.dataframe(st.session_state.price_theme)

    if show_returns_df:
        st.subheader("Raw returns_df (latest target date)")
        # Recompute a simple returns_df for debug display (or show subset of final_df)
        lookback_cols = [c for c in display_df.columns if (str(c).endswith('H') or str(c).endswith('d')) and not str(c).endswith('_Excess')]
        cols_to_show = ['Coin'] + lookback_cols
        st.dataframe(display_df[display_df.columns.intersection(cols_to_show)].head(200))

st.success("Ready.")
