"""
=============================================================================
 SIGNAL CONSOLIDATOR (Long/Short Market-Neutral)
=============================================================================

PURPOSE
-------
Standalone Streamlit page that consolidates the buy/sell logic from pages
1, 8, 9, 10, 11, 14, 15 into a single hybrid framework, deduplicates the
correlated signals, and adds the missing extension / jump guardrails.

This page DOES NOT depend on any other page — it fetches its own OHLCV data
from KuCoin (via ccxt) and reads the themes mapping from the same Excel file
used by page 1. Run it directly.

STRATEGY OVERVIEW
-----------------
The buy thesis has 3 legs that must all point the right way (bracket A/B/C),
plus 2 hard rejections (Stage 0: liquidity + jump; Stage 2: extension).
Sell signals are the symmetric mirror.

  STAGE 0 — UNIVERSE FILTER (hard, applied first)
    • Min history (≥ longest_lookback + buffer)
    • Liquidity: median 30-bar USD volume ≥ user threshold
    • Recent jump: max |close-to-close return| in last 5 bars < 15%
    • Wick jump:  max (high-low)/prev_close in last 5 bars < 20%

  STAGE 1 — BUCKET SIGNALS (continuous, cross-sectional z-scores)
    Bucket A — Theme-relative momentum
        A1: 30d excess log-return vs theme median
        A2: 60d excess log-return vs theme median
        A3: 14d excess log-return vs theme median
    Bucket B — BTC-relative momentum
        B1: 30d Coin/BTC log-return
        B2: 90d Coin/BTC log-return
        B3: 14d Coin/BTC log-return
    Bucket C — Absolute (universe-z-scored)
        C1: 30d ATR-scaled return, CS z-score
        C2: 60d ATR-scaled return, CS z-score
        C3: Multi-MA trend agreement (count of price > SMA-20/50/100), CS z-score

    Each bucket = element-wise mean of its three constituent CS z-scores.

  STAGE 2 — EXTENSION FILTER (binary, applied as gate)
    BUY-side reject if EITHER:
        • price_z_60 > 2.0       (price >2σ above its own 60-bar mean)
        • 90d range_pct > 0.92   (within top 8% of 90-bar range)
    SELL-side reject if EITHER:
        • price_z_60 < -2.0
        • 90d range_pct < 0.08

  STAGE 3 — HYBRID GATING (the core of this page)
    HARD BINARY (3-of-3 consistency):
        BUY  qualified ⟺ z_A > 0 AND z_B > 0 AND z_C > 0  AND Stage-2 pass
        SELL qualified ⟺ z_A < 0 AND z_B < 0 AND z_C < 0  AND Stage-2 pass
        score = (z_A + z_B + z_C) / 3   (sign-flipped for sell ranking)

    SOFT PENALTY (Lagrangian):
        BUY score  = mean(z) − λ · Σ max(0, −z_bucket)²    | Stage-2 pass
        SELL score = −mean(z) − λ · Σ max(0,  z_bucket)²   | Stage-2 pass
        λ controls how harshly inconsistent buckets are penalised.
        The hard binary is the limit of soft as λ → ∞ with a step penalty.

  STAGE 4 — RANKING & SELECTION
    Top-N by score on each side, equal-weighted in dollar terms.
    The book is dollar-neutral by construction (long $X, short $X).

VALIDATION
----------
  • IC of each bucket vs forward 5/10/15/30d returns (cross-sectional Spearman)
  • Decile forward returns for HARD score (qualified universe only)
  • Decile forward returns for SOFT score (extension+jump-filtered universe)
  • Quintile-spread (top-bottom) over time

ALL ROLLING/EXPANDING WINDOWS USE PAST DATA ONLY — no lookahead bias.
=============================================================================
"""

import time
import warnings
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# =============================================================================
# 1. PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Signal Consolidator (L/S MN)",
    layout="wide",
    page_icon="🎯",
)
st.title("🎯 Signal Consolidator — Long/Short Market-Neutral")
st.caption(
    "Hybrid (gate + score) framework consolidating bucket A (theme-rel), "
    "B (BTC-rel), and C (absolute) momentum signals. Standalone — fetches own data."
)

# =============================================================================
# 2. SIDEBAR CONFIG
# =============================================================================
with st.sidebar:
    st.header("⚙️ Data Settings")
    default_excel_relpath = Path("Input-Files") / "Themes_mapping.xlsx"
    st.caption(f"Themes mapping: `{default_excel_relpath}`")
    timeframe = st.selectbox("Timeframe", ["1d", "4h", "1h"], index=0)
    limit = st.number_input(
        "OHLCV bars to fetch", value=500, min_value=200, max_value=2000, step=50,
        help="Need ≥ (longest lookback + backtest min history) bars. "
             "Default 500 ≈ 1.4 years on daily."
    )
    sleep_seconds = st.number_input("Sleep between fetches (s)", value=0.2, step=0.05)

    st.markdown("---")
    st.header("🚦 Stage 0 — Universe Filter")
    min_usd_vol = st.number_input(
        "Min median 30-bar USD volume ($)",
        value=1_000_000, min_value=0, max_value=100_000_000, step=100_000,
        help="Coins below this median dollar volume are dropped."
    )
    jump_thresh_pct = st.slider(
        "Max single-bar |return| in last 5 bars (%)",
        min_value=5.0, max_value=50.0, value=15.0, step=1.0,
        help="Coins with any close-to-close move ≥ this in last 5 bars are dropped — "
             "they have high mean-reversion probability."
    )
    wick_thresh_pct = st.slider(
        "Max (H-L)/prev_close in last 5 bars (%)",
        min_value=5.0, max_value=80.0, value=20.0, step=1.0,
        help="Catches intraday explosive moves even if close-to-close is muted."
    )

    st.markdown("---")
    st.header("🧱 Stage 2 — Extension Filter")
    price_z60_buy_max = st.slider(
        "BUY reject if price_z_60 > ",
        min_value=1.0, max_value=4.0, value=2.0, step=0.25,
    )
    range_pct_buy_max = st.slider(
        "BUY reject if 90d range pct > ",
        min_value=0.70, max_value=0.99, value=0.92, step=0.01,
    )
    # Sell mirror values are the negative/inverse — kept symmetric
    st.caption("Sell mirrors: price_z_60 < −X, range_pct < (1−Y)")

    st.markdown("---")
    st.header("🧪 Stage 3 — Hybrid Gating")
    lambda_penalty = st.slider(
        "Soft-gate λ (penalty weight)",
        min_value=0.0, max_value=5.0, value=1.5, step=0.25,
        help="0 = pure average (no consistency penalty). "
             "1.5 = moderate. ≥3 ≈ approaches hard binary behaviour."
    )

    st.markdown("---")
    st.header("📊 Selection")
    top_n = st.number_input("Top-N longs / Bottom-N shorts", min_value=3, max_value=30, value=8)

    st.markdown("---")
    st.header("🧪 Backtest / Validation")
    bt_min_history_bars = st.number_input(
        "Min history before backtest starts (bars)",
        min_value=60, max_value=500, value=120, step=10,
        help="Skip the first N bars of history when computing IC and decile metrics."
    )
    bt_eval_step = st.number_input(
        "Score evaluation step (bars)",
        min_value=1, max_value=20, value=1, step=1,
        help="1 = evaluate scores every bar (slow). 5 = every 5th bar (faster)."
    )

    st.markdown("---")
    fetch_btn = st.button("🔄 Fetch / Refresh Data", use_container_width=True)
    recompute_btn = st.button("🔁 Recompute Signals (no fetch)", use_container_width=True)


# =============================================================================
# 3. DATA FETCH (ccxt → KuCoin)
#    Adapted from page 3's OHLC dataloader, extended to keep volume.
# =============================================================================
@st.cache_data
def read_theme_excel(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, engine="openpyxl")


@st.cache_resource
def get_exchange():
    import ccxt
    ex = ccxt.kucoin()
    ex.load_markets()
    return ex


def fetch_ohlcv_multiindex(
    exchange, symbols: List[str], timeframe: str, limit: int, sleep_s: float
) -> pd.DataFrame:
    """
    Fetch OHLCV per symbol; return a single MultiIndex-column DataFrame.
    Columns are (symbol, field) where field ∈ {o, h, l, c, v}.
    """
    frames = []
    progress = st.progress(0.0, text="Fetching…")
    n = len(symbols)
    for i, sym in enumerate(symbols):
        try:
            ohlcv = exchange.fetch_ohlcv(f"{sym}/USDT", timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=["ts", "o", "h", "l", "c", "v"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms")
            df = df.set_index("ts")[["o", "h", "l", "c", "v"]]
            df.columns = pd.MultiIndex.from_product([[sym], df.columns])
            frames.append(df)
            time.sleep(sleep_s)
        except Exception as e:
            st.warning(f"Failed to fetch {sym}: {e}")
            continue
        progress.progress((i + 1) / n, text=f"Fetched {i+1}/{n}: {sym}")
    progress.empty()
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()


# Initialise session state slots
for key in ["sc_ohlcv", "sc_ticker_to_theme", "sc_results", "sc_last_fetch", "sc_timeframe"]:
    if key not in st.session_state:
        st.session_state[key] = None


# =============================================================================
# 4. RESPOND TO FETCH / RECOMPUTE
# =============================================================================
if fetch_btn:
    df_map = read_theme_excel(default_excel_relpath)
    df_map["Symbol"] = df_map["Symbol"].astype(str).str.upper()
    symbols = df_map["Symbol"].tolist()
    if "BTC" not in symbols:
        symbols = ["BTC"] + symbols
    ticker_to_theme = dict(zip(df_map["Symbol"], df_map["Theme"]))

    ex = get_exchange()
    with st.spinner("Fetching OHLCV from KuCoin…"):
        ohlcv = fetch_ohlcv_multiindex(ex, symbols, timeframe, int(limit), float(sleep_seconds))
    if ohlcv.empty:
        st.error("No data fetched. Check connectivity / symbols.")
        st.stop()

    st.session_state["sc_ohlcv"] = ohlcv
    st.session_state["sc_ticker_to_theme"] = ticker_to_theme
    st.session_state["sc_last_fetch"] = datetime.utcnow()
    st.session_state["sc_timeframe"] = timeframe
    st.session_state["sc_results"] = None  # invalidate cached signals
    st.success(
        f"Fetched {ohlcv.shape[1] // 5} symbols, "
        f"{ohlcv.shape[0]} bars ({timeframe})."
    )


# =============================================================================
# 5. EARLY EXIT IF NO DATA
# =============================================================================
ohlcv = st.session_state["sc_ohlcv"]
ticker_to_theme = st.session_state["sc_ticker_to_theme"]

if ohlcv is None or ohlcv.empty:
    st.info("👈 Click **Fetch / Refresh Data** in the sidebar to begin.")
    st.stop()


# =============================================================================
# 6. CORE COMPUTATION FUNCTIONS
# =============================================================================
def cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score: (x - row_mean) / row_std. Skips NaN."""
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0)


def avg_signals(*dfs: pd.DataFrame) -> pd.DataFrame:
    """Element-wise mean across DataFrames, ignoring NaN."""
    arr = np.stack([df.values for df in dfs], axis=0)
    avg = np.nanmean(arr, axis=0)
    return pd.DataFrame(avg, index=dfs[0].index, columns=dfs[0].columns)


def compute_atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Wilder True Range / ATR.
    TR = max(H - L, |H - prev_close|, |L - prev_close|)
    ATR = rolling mean of TR.
    """
    prev_close = close.shift(1)
    tr1 = (high - low).values
    tr2 = (high - prev_close).abs().values
    tr3 = (low - prev_close).abs().values
    tr_arr = np.maximum.reduce([tr1, tr2, tr3])
    tr = pd.DataFrame(tr_arr, index=high.index, columns=high.columns)
    return tr.rolling(period, min_periods=max(2, period // 2)).mean()


def rolling_ols_alpha(coin_ret: pd.DataFrame, btc_ret: pd.Series, window: int) -> pd.DataFrame:
    """
    Vectorised rolling OLS: regress each coin return on BTC return over `window` bars.
    Returns daily intercept (alpha). No look-ahead.
    """
    btc_mean = btc_ret.rolling(window).mean()
    btc_var = btc_ret.rolling(window).var()
    coin_mean = coin_ret.rolling(window).mean()
    cov = coin_ret.rolling(window).cov(btc_ret)
    beta = cov.div(btc_var, axis=0)
    alpha = coin_mean.sub(beta.mul(btc_mean, axis=0), axis=0)
    return alpha


# =============================================================================
# 7. SIGNAL PIPELINE — TIME SERIES FOR EVERY SIGNAL
# =============================================================================
@st.cache_data(show_spinner="Computing signal pipeline…", ttl=600)
def compute_pipeline(
    ohlcv_df: pd.DataFrame,
    ticker_to_theme: Dict[str, str],
    min_usd_vol: float,
    jump_thresh: float,
    wick_thresh: float,
    price_z60_buy_max: float,
    range_pct_buy_max: float,
    lambda_penalty: float,
) -> Dict:
    """
    Big single-pass computation. Returns a dict of TS DataFrames so downstream
    UI can pull whatever it needs without recomputing.

    Implements Stages 0, 1, 2, 3 as time series so we can validate historically.
    """
    # ---- Slice OHLCV ----
    close = ohlcv_df.xs("c", level=1, axis=1).copy()
    high = ohlcv_df.xs("h", level=1, axis=1).copy()
    low = ohlcv_df.xs("l", level=1, axis=1).copy()
    volume = ohlcv_df.xs("v", level=1, axis=1).copy()

    # Coerce to float
    close = close.apply(pd.to_numeric, errors="coerce")
    high = high.apply(pd.to_numeric, errors="coerce")
    low = low.apply(pd.to_numeric, errors="coerce")
    volume = volume.apply(pd.to_numeric, errors="coerce")

    # Universal log-return panel
    log_ret = np.log(close / close.shift(1))

    # =============================================================
    # STAGE 0 — UNIVERSE FILTERS (time-varying boolean masks)
    # =============================================================
    # Liquidity: median 30-bar USD volume
    usd_vol = close * volume
    median_usd_vol_30 = usd_vol.rolling(30, min_periods=15).median()
    liquidity_mask = median_usd_vol_30 >= min_usd_vol

    # Recent close-to-close jump (5-bar window, max |log return|)
    log_jump_thresh = np.log(1.0 + jump_thresh / 100.0)
    abs_logret_5d_max = log_ret.abs().rolling(5, min_periods=3).max()
    no_close_jump_mask = abs_logret_5d_max <= log_jump_thresh

    # Recent wick jump: (high - low) / prev_close
    wick_logthresh = wick_thresh / 100.0  # not log; just ratio
    wick_pct = (high - low) / close.shift(1)
    wick_5d_max = wick_pct.rolling(5, min_periods=3).max()
    no_wick_jump_mask = wick_5d_max <= wick_logthresh

    stage0_mask = liquidity_mask & no_close_jump_mask & no_wick_jump_mask

    # =============================================================
    # STAGE 1 — BUCKET A: THEME-RELATIVE
    # Excess log-return vs theme median.
    # =============================================================
    # Build a same-shape DataFrame where each coin gets its theme median per bar
    coin_to_theme = {c: ticker_to_theme.get(c, "UNKNOWN") for c in log_ret.columns}
    theme_map_series = pd.Series(coin_to_theme)
    themes = sorted(set(coin_to_theme.values()))

    theme_median_panel = pd.DataFrame(
        index=log_ret.index, columns=log_ret.columns, dtype=float
    )
    for theme in themes:
        coins_in = [c for c in log_ret.columns if coin_to_theme.get(c) == theme]
        if not coins_in:
            continue
        theme_med = log_ret[coins_in].median(axis=1)
        for coin in coins_in:
            theme_median_panel[coin] = theme_med

    excess_ret = log_ret.sub(theme_median_panel, fill_value=np.nan)

    A1 = excess_ret.rolling(30, min_periods=20).sum()
    A2 = excess_ret.rolling(60, min_periods=40).sum()
    A3 = excess_ret.rolling(14, min_periods=10).sum()

    z_A1 = cs_zscore(A1)
    z_A2 = cs_zscore(A2)
    z_A3 = cs_zscore(A3)
    bucket_A = avg_signals(z_A1, z_A2, z_A3)

    # =============================================================
    # STAGE 1 — BUCKET B: BTC-RELATIVE
    # Coin/BTC log-return at multiple horizons.
    # =============================================================
    if "BTC" not in log_ret.columns:
        # Fallback — make B all NaN
        bucket_B = pd.DataFrame(np.nan, index=log_ret.index, columns=log_ret.columns)
        z_B1 = z_B2 = z_B3 = bucket_B.copy()
    else:
        btc_log_ret = log_ret["BTC"]
        coin_vs_btc = log_ret.sub(btc_log_ret, axis=0)

        B1 = coin_vs_btc.rolling(30, min_periods=20).sum()
        B2 = coin_vs_btc.rolling(90, min_periods=60).sum()
        B3 = coin_vs_btc.rolling(14, min_periods=10).sum()

        z_B1 = cs_zscore(B1)
        z_B2 = cs_zscore(B2)
        z_B3 = cs_zscore(B3)
        bucket_B = avg_signals(z_B1, z_B2, z_B3)

    # =============================================================
    # STAGE 1 — BUCKET C: ABSOLUTE (universe-z-scored)
    # ATR-scaled returns + multi-MA trend agreement.
    # =============================================================
    atr_14 = compute_atr(high, low, close, period=14)
    atr_pct = (atr_14 / close).replace(0, np.nan)  # ATR as fraction of price
    # Vol-scale daily return by ATR fraction (lagged by 1 to avoid same-bar leak)
    atr_pct_lag = atr_pct.shift(1).replace(0, np.nan)
    vol_scaled_ret = log_ret.div(atr_pct_lag).clip(lower=-5.0, upper=5.0)

    C1 = vol_scaled_ret.rolling(30, min_periods=20).sum()
    C2 = vol_scaled_ret.rolling(60, min_periods=40).sum()

    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    sma_100 = close.rolling(100, min_periods=50).mean()
    trend_count = (
        (close > sma_20).astype(float)
        + (close > sma_50).astype(float)
        + (close > sma_100).astype(float)
    )  # 0..3
    # Mask where SMAs not available
    trend_count = trend_count.where(sma_100.notna())

    z_C1 = cs_zscore(C1)
    z_C2 = cs_zscore(C2)
    z_C3 = cs_zscore(trend_count)
    bucket_C = avg_signals(z_C1, z_C2, z_C3)

    # =============================================================
    # STAGE 2 — EXTENSION FILTER (binary, time-varying)
    # =============================================================
    sma_60 = close.rolling(60).mean()
    std_60 = close.rolling(60).std().replace(0, np.nan)
    price_z_60 = (close - sma_60) / std_60

    high_90 = close.rolling(90, min_periods=60).max()
    low_90 = close.rolling(90, min_periods=60).min()
    range_pct_90 = (close - low_90) / (high_90 - low_90).replace(0, np.nan)

    buy_extension_ok = (price_z_60 < price_z60_buy_max) & (range_pct_90 < range_pct_buy_max)
    sell_extension_ok = (price_z_60 > -price_z60_buy_max) & (range_pct_90 > (1.0 - range_pct_buy_max))

    # =============================================================
    # STAGE 3 — HYBRID GATING
    # =============================================================
    avg_z = avg_signals(bucket_A, bucket_B, bucket_C)

    # Buy penalty (penalises NEGATIVE bucket scores)
    buy_pen = (
        np.maximum(-bucket_A, 0.0) ** 2
        + np.maximum(-bucket_B, 0.0) ** 2
        + np.maximum(-bucket_C, 0.0) ** 2
    )
    # Sell penalty (penalises POSITIVE bucket scores — opposite of buy)
    sell_pen = (
        np.maximum(bucket_A, 0.0) ** 2
        + np.maximum(bucket_B, 0.0) ** 2
        + np.maximum(bucket_C, 0.0) ** 2
    )

    # Stage-0 mask layered with extension mask for each side
    buy_pre_gate = stage0_mask & buy_extension_ok
    sell_pre_gate = stage0_mask & sell_extension_ok

    # ---- HARD BINARY ----
    hard_buy_qual = (bucket_A > 0) & (bucket_B > 0) & (bucket_C > 0) & buy_pre_gate
    hard_buy_score = avg_z.where(hard_buy_qual)

    hard_sell_qual = (bucket_A < 0) & (bucket_B < 0) & (bucket_C < 0) & sell_pre_gate
    # Sell score: higher = stronger sell. Use negated avg_z so scores are positive for shorts.
    hard_sell_score = (-avg_z).where(hard_sell_qual)

    # ---- SOFT PENALTY ----
    soft_buy_score = (avg_z - lambda_penalty * buy_pen).where(buy_pre_gate)
    soft_sell_score = (-avg_z - lambda_penalty * sell_pen).where(sell_pre_gate)

    # =============================================================
    # PACKAGE EVERYTHING
    # =============================================================
    return {
        "close": close,
        "log_ret": log_ret,
        "atr_pct": atr_pct,
        # masks
        "stage0_mask": stage0_mask,
        "liquidity_mask": liquidity_mask,
        "no_close_jump_mask": no_close_jump_mask,
        "no_wick_jump_mask": no_wick_jump_mask,
        "buy_extension_ok": buy_extension_ok,
        "sell_extension_ok": sell_extension_ok,
        "price_z_60": price_z_60,
        "range_pct_90": range_pct_90,
        "median_usd_vol_30": median_usd_vol_30,
        # bucket TS
        "z_A1": z_A1, "z_A2": z_A2, "z_A3": z_A3, "bucket_A": bucket_A,
        "z_B1": z_B1, "z_B2": z_B2, "z_B3": z_B3, "bucket_B": bucket_B,
        "z_C1": z_C1, "z_C2": z_C2, "z_C3": z_C3, "bucket_C": bucket_C,
        "avg_z": avg_z,
        # gates
        "hard_buy_qual": hard_buy_qual, "hard_buy_score": hard_buy_score,
        "hard_sell_qual": hard_sell_qual, "hard_sell_score": hard_sell_score,
        "soft_buy_score": soft_buy_score, "soft_sell_score": soft_sell_score,
        # context
        "coin_to_theme": coin_to_theme,
    }


# =============================================================================
# 8. RUN PIPELINE
# =============================================================================
need_compute = (
    st.session_state["sc_results"] is None
    or fetch_btn
    or recompute_btn
)
if need_compute:
    st.session_state["sc_results"] = compute_pipeline(
        ohlcv,
        ticker_to_theme,
        min_usd_vol=float(min_usd_vol),
        jump_thresh=float(jump_thresh_pct),
        wick_thresh=float(wick_thresh_pct),
        price_z60_buy_max=float(price_z60_buy_max),
        range_pct_buy_max=float(range_pct_buy_max),
        lambda_penalty=float(lambda_penalty),
    )

R = st.session_state["sc_results"]


# =============================================================================
# 9. LATEST-BAR SIGNAL TABLES
# =============================================================================
last_ts = R["close"].index[-1]
ts_str = str(last_ts)[:16]

st.markdown("---")
st.subheader(f"📅 Latest signal — {ts_str}")

# Counts banner
n_total = R["close"].shape[1]
n_stage0 = int(R["stage0_mask"].iloc[-1].sum())
n_liquid = int(R["liquidity_mask"].iloc[-1].sum())
n_no_jump = int((R["no_close_jump_mask"] & R["no_wick_jump_mask"]).iloc[-1].sum())
n_hard_buy = int(R["hard_buy_qual"].iloc[-1].sum())
n_hard_sell = int(R["hard_sell_qual"].iloc[-1].sum())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total coins", n_total)
c2.metric("Liquid", n_liquid, delta=f"−{n_total - n_liquid} dropped", delta_color="off")
c3.metric("Stage-0 pass", n_stage0)
c4.metric("Hard BUY qualified", n_hard_buy)
c5.metric("Hard SELL qualified", n_hard_sell)


def build_signal_table(scores_row: pd.Series, R: dict, last_ts) -> pd.DataFrame:
    """Build a per-coin signal table at last bar from a scores Series."""
    s = scores_row.dropna().sort_values(ascending=False)
    if s.empty:
        return pd.DataFrame()
    rows = []
    for coin in s.index:
        rows.append({
            "Coin": coin,
            "Theme": R["coin_to_theme"].get(coin, "UNKNOWN"),
            "Score": round(float(s[coin]), 3),
            "z_A": round(float(R["bucket_A"].iloc[-1].get(coin, np.nan)), 3),
            "z_B": round(float(R["bucket_B"].iloc[-1].get(coin, np.nan)), 3),
            "z_C": round(float(R["bucket_C"].iloc[-1].get(coin, np.nan)), 3),
            "price_z_60": round(float(R["price_z_60"].iloc[-1].get(coin, np.nan)), 2),
            "range_pct_90": round(float(R["range_pct_90"].iloc[-1].get(coin, np.nan)), 2),
            "ATR_pct": round(float(R["atr_pct"].iloc[-1].get(coin, np.nan)) * 100, 2),
        })
    return pd.DataFrame(rows)


tab_hard, tab_soft, tab_diag = st.tabs([
    "🟢🔴 Hard Binary Gate",
    "🟢🔴 Soft Penalty Gate",
    "🔍 Bucket Inspector",
])

# ---------- HARD ----------
with tab_hard:
    st.markdown(
        f"**Hard binary**: coin must have **all 3 buckets agreeing in direction** "
        f"AND pass extension/jump/liquidity gates. "
        f"λ → ∞ limit. Top-{int(top_n)} per side."
    )
    hard_buy_tbl = build_signal_table(R["hard_buy_score"].iloc[-1], R, last_ts).head(int(top_n))
    hard_sell_tbl = build_signal_table(R["hard_sell_score"].iloc[-1], R, last_ts).head(int(top_n))

    cL, cR = st.columns(2)
    with cL:
        st.markdown("**🟢 Top BUY signals**")
        if hard_buy_tbl.empty:
            st.info("No qualified buy signals.")
        else:
            st.dataframe(
                hard_buy_tbl.style
                .background_gradient(subset=["Score"], cmap="Greens")
                .background_gradient(subset=["z_A", "z_B", "z_C"], cmap="RdYlGn", vmin=-2, vmax=2)
                .background_gradient(subset=["price_z_60"], cmap="RdYlGn_r", vmin=-3, vmax=3),
                hide_index=True,
                use_container_width=True,
                height=min(420, 60 + 35 * len(hard_buy_tbl)),
            )
    with cR:
        st.markdown("**🔴 Top SELL signals**")
        if hard_sell_tbl.empty:
            st.info("No qualified sell signals.")
        else:
            st.dataframe(
                hard_sell_tbl.style
                .background_gradient(subset=["Score"], cmap="Reds")
                .background_gradient(subset=["z_A", "z_B", "z_C"], cmap="RdYlGn", vmin=-2, vmax=2)
                .background_gradient(subset=["price_z_60"], cmap="RdYlGn_r", vmin=-3, vmax=3),
                hide_index=True,
                use_container_width=True,
                height=min(420, 60 + 35 * len(hard_sell_tbl)),
            )

# ---------- SOFT ----------
with tab_soft:
    st.markdown(
        f"**Soft penalty**: score = mean(z_A, z_B, z_C) − λ·Σmax(0, −z)² for buy "
        f"(reverse for sell). λ = **{lambda_penalty}**. "
        f"Negative-bucket coins are penalised but not rejected. "
        f"Top-{int(top_n)} per side after extension/jump/liquidity gates."
    )
    soft_buy_tbl = build_signal_table(R["soft_buy_score"].iloc[-1], R, last_ts).head(int(top_n))
    soft_sell_tbl = build_signal_table(R["soft_sell_score"].iloc[-1], R, last_ts).head(int(top_n))

    cL, cR = st.columns(2)
    with cL:
        st.markdown("**🟢 Top BUY signals**")
        if soft_buy_tbl.empty:
            st.info("No buy signals.")
        else:
            st.dataframe(
                soft_buy_tbl.style
                .background_gradient(subset=["Score"], cmap="Greens")
                .background_gradient(subset=["z_A", "z_B", "z_C"], cmap="RdYlGn", vmin=-2, vmax=2)
                .background_gradient(subset=["price_z_60"], cmap="RdYlGn_r", vmin=-3, vmax=3),
                hide_index=True,
                use_container_width=True,
                height=min(420, 60 + 35 * len(soft_buy_tbl)),
            )
    with cR:
        st.markdown("**🔴 Top SELL signals**")
        if soft_sell_tbl.empty:
            st.info("No sell signals.")
        else:
            st.dataframe(
                soft_sell_tbl.style
                .background_gradient(subset=["Score"], cmap="Reds")
                .background_gradient(subset=["z_A", "z_B", "z_C"], cmap="RdYlGn", vmin=-2, vmax=2)
                .background_gradient(subset=["price_z_60"], cmap="RdYlGn_r", vmin=-3, vmax=3),
                hide_index=True,
                use_container_width=True,
                height=min(420, 60 + 35 * len(soft_sell_tbl)),
            )

    # Soft vs hard agreement
    soft_buy_top = set(soft_buy_tbl["Coin"]) if not soft_buy_tbl.empty else set()
    hard_buy_top = set(hard_buy_tbl["Coin"]) if not hard_buy_tbl.empty else set()
    soft_sell_top = set(soft_sell_tbl["Coin"]) if not soft_sell_tbl.empty else set()
    hard_sell_top = set(hard_sell_tbl["Coin"]) if not hard_sell_tbl.empty else set()

    st.markdown("**Hard vs Soft top-N agreement**")
    a1, a2 = st.columns(2)
    with a1:
        st.markdown(
            f"BUY overlap: **{len(soft_buy_top & hard_buy_top)} / "
            f"{min(len(soft_buy_top), len(hard_buy_top))}** "
            f"&nbsp;|&nbsp; Soft-only: {sorted(soft_buy_top - hard_buy_top)} "
            f"&nbsp;|&nbsp; Hard-only: {sorted(hard_buy_top - soft_buy_top)}"
        )
    with a2:
        st.markdown(
            f"SELL overlap: **{len(soft_sell_top & hard_sell_top)} / "
            f"{min(len(soft_sell_top), len(hard_sell_top))}** "
            f"&nbsp;|&nbsp; Soft-only: {sorted(soft_sell_top - hard_sell_top)} "
            f"&nbsp;|&nbsp; Hard-only: {sorted(hard_sell_top - soft_sell_top)}"
        )


# ---------- DIAG ----------
with tab_diag:
    st.markdown(
        "**Bucket inspector** — view all buckets and constituent z-scores for any coin. "
        "Useful for sanity-checking why a coin is or isn't in the buy/sell list."
    )
    sel_coin = st.selectbox(
        "Select coin",
        options=sorted(R["close"].columns.tolist()),
    )
    if sel_coin:
        # Latest values
        last_row = {
            "z_A1": R["z_A1"].iloc[-1].get(sel_coin, np.nan),
            "z_A2": R["z_A2"].iloc[-1].get(sel_coin, np.nan),
            "z_A3": R["z_A3"].iloc[-1].get(sel_coin, np.nan),
            "bucket_A": R["bucket_A"].iloc[-1].get(sel_coin, np.nan),
            "z_B1": R["z_B1"].iloc[-1].get(sel_coin, np.nan),
            "z_B2": R["z_B2"].iloc[-1].get(sel_coin, np.nan),
            "z_B3": R["z_B3"].iloc[-1].get(sel_coin, np.nan),
            "bucket_B": R["bucket_B"].iloc[-1].get(sel_coin, np.nan),
            "z_C1": R["z_C1"].iloc[-1].get(sel_coin, np.nan),
            "z_C2": R["z_C2"].iloc[-1].get(sel_coin, np.nan),
            "z_C3": R["z_C3"].iloc[-1].get(sel_coin, np.nan),
            "bucket_C": R["bucket_C"].iloc[-1].get(sel_coin, np.nan),
            "avg_z": R["avg_z"].iloc[-1].get(sel_coin, np.nan),
        }
        df_view = pd.DataFrame.from_dict(last_row, orient="index", columns=["Latest z"])
        df_view["Latest z"] = df_view["Latest z"].astype(float).round(3)
        st.dataframe(df_view, use_container_width=False)

        # Time series chart of buckets
        sub_df = pd.DataFrame({
            "Bucket A": R["bucket_A"][sel_coin],
            "Bucket B": R["bucket_B"][sel_coin],
            "Bucket C": R["bucket_C"][sel_coin],
            "Avg z": R["avg_z"][sel_coin],
        })
        fig_buc = go.Figure()
        for col, color in zip(
            ["Bucket A", "Bucket B", "Bucket C", "Avg z"],
            ["#60a5fa", "#a78bfa", "#34d399", "#fbbf24"],
        ):
            fig_buc.add_trace(go.Scatter(
                x=sub_df.index, y=sub_df[col], name=col, mode="lines",
                line=dict(color=color, width=1.5 if col != "Avg z" else 2.5),
            ))
        fig_buc.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
        fig_buc.update_layout(
            title=f"{sel_coin} — bucket scores over time",
            template="plotly_dark",
            height=360,
            hovermode="x unified",
            yaxis_title="CS z-score",
        )
        st.plotly_chart(fig_buc, use_container_width=True)

        # Gate status timeline (last 90 bars)
        last_n = 90
        gate_df = pd.DataFrame({
            "Liquidity": R["liquidity_mask"][sel_coin].astype(int),
            "No close jump": R["no_close_jump_mask"][sel_coin].astype(int),
            "No wick jump": R["no_wick_jump_mask"][sel_coin].astype(int),
            "Buy extension OK": R["buy_extension_ok"][sel_coin].astype(int),
            "Sell extension OK": R["sell_extension_ok"][sel_coin].astype(int),
        }).iloc[-last_n:]
        st.markdown(f"**Gate status (last {last_n} bars)**")
        st.dataframe(gate_df.tail(15), use_container_width=True, height=300)


# =============================================================================
# 10. VALIDATION — IC + DECILE FORWARD RETURNS
# =============================================================================
st.markdown("---")
st.header("🧪 Validation")

st.caption(
    "All metrics use causal signals (no lookahead). "
    "Forward returns are absolute close-to-close log returns, and BTC-excess returns "
    "(more relevant for L/S MN). Long-side validation tests if HIGHER scores predict "
    "HIGHER forward returns; short-side validation tests if HIGHER (sell) scores predict "
    "LOWER forward returns."
)


def compute_forward_returns(close: pd.DataFrame, horizons: List[int]) -> Dict[int, pd.DataFrame]:
    """Forward log-returns at each horizon. Shifted backward into 'now' index."""
    fwd = {}
    for h in horizons:
        fwd[h] = np.log(close.shift(-h) / close)
    return fwd


def compute_btc_excess_forward_returns(
    close: pd.DataFrame, horizons: List[int]
) -> Dict[int, pd.DataFrame]:
    """Forward log-returns minus BTC's forward log-return at same horizon."""
    out = {}
    if "BTC" not in close.columns:
        for h in horizons:
            out[h] = np.log(close.shift(-h) / close)
        return out
    for h in horizons:
        fwd = np.log(close.shift(-h) / close)
        btc_fwd = fwd["BTC"]
        out[h] = fwd.sub(btc_fwd, axis=0)
    return out


@st.cache_data(show_spinner="Running validation backtest…", ttl=600)
def run_validation(
    R_close: pd.DataFrame,
    bucket_A: pd.DataFrame,
    bucket_B: pd.DataFrame,
    bucket_C: pd.DataFrame,
    avg_z: pd.DataFrame,
    hard_buy_score: pd.DataFrame,
    hard_sell_score: pd.DataFrame,
    soft_buy_score: pd.DataFrame,
    soft_sell_score: pd.DataFrame,
    min_history_bars: int,
    eval_step: int,
) -> Dict:
    """
    Compute IC and decile-wise forward returns.
    Returns a dict of result DataFrames.
    """
    horizons = [5, 10, 15, 30]

    fwd_abs = compute_forward_returns(R_close, horizons)
    fwd_btc = compute_btc_excess_forward_returns(R_close, horizons)

    # Restrict to backtest window
    eval_idx = R_close.index[min_history_bars::eval_step]
    eval_idx = eval_idx[: -max(horizons)]  # drop tail where forward returns aren't observable

    # ---------- IC ANALYSIS PER BUCKET ----------
    def per_bucket_IC(signal_df: pd.DataFrame, fwd_dict: Dict[int, pd.DataFrame]) -> pd.DataFrame:
        rows = []
        for h in horizons:
            ic_list = []
            for t in eval_idx:
                if t not in signal_df.index:
                    continue
                s = signal_df.loc[t]
                r = fwd_dict[h].loc[t]
                joined = pd.concat([s, r], axis=1, keys=["s", "r"]).dropna()
                if len(joined) < 5:
                    continue
                # Spearman = Pearson on ranks
                rs = joined["s"].rank()
                rr = joined["r"].rank()
                if rs.std() == 0 or rr.std() == 0:
                    continue
                ic = rs.corr(rr)
                ic_list.append(ic)
            if ic_list:
                arr = np.array(ic_list)
                rows.append({
                    "Horizon (bars)": h,
                    "IC mean": np.nanmean(arr),
                    "IC std": np.nanstd(arr),
                    "IR (mean/std)": np.nanmean(arr) / (np.nanstd(arr) + 1e-9),
                    "Pos IC %": (arr > 0).mean() * 100,
                    "n samples": len(arr),
                })
            else:
                rows.append({
                    "Horizon (bars)": h, "IC mean": np.nan, "IC std": np.nan,
                    "IR (mean/std)": np.nan, "Pos IC %": np.nan, "n samples": 0,
                })
        return pd.DataFrame(rows)

    ic_A_abs = per_bucket_IC(bucket_A, fwd_abs)
    ic_B_abs = per_bucket_IC(bucket_B, fwd_abs)
    ic_C_abs = per_bucket_IC(bucket_C, fwd_abs)
    ic_avg_abs = per_bucket_IC(avg_z, fwd_abs)

    ic_A_btc = per_bucket_IC(bucket_A, fwd_btc)
    ic_B_btc = per_bucket_IC(bucket_B, fwd_btc)
    ic_C_btc = per_bucket_IC(bucket_C, fwd_btc)
    ic_avg_btc = per_bucket_IC(avg_z, fwd_btc)

    # ---------- DECILE FORWARD RETURNS ----------
    def decile_fwd_returns(
        score_df: pd.DataFrame,
        fwd_dict: Dict[int, pd.DataFrame],
        n_bins: int = 10,
    ) -> pd.DataFrame:
        """
        For each evaluation timestamp:
          - Take coins with valid score.
          - Bucket into deciles by score.
          - Record forward returns per decile.
        Average forward returns by decile, by horizon.
        Returns wide DataFrame: index = decile (1..10), columns = horizon.
        """
        bucket_records = {h: {b: [] for b in range(1, n_bins + 1)} for h in horizons}
        for t in eval_idx:
            if t not in score_df.index:
                continue
            s = score_df.loc[t].dropna()
            if len(s) < n_bins:  # need at least one obs per bin
                continue
            try:
                bins = pd.qcut(s, n_bins, labels=False, duplicates="drop") + 1
            except ValueError:
                continue
            for h in horizons:
                if t not in fwd_dict[h].index:
                    continue
                fwd_row = fwd_dict[h].loc[t]
                for coin, b in bins.items():
                    if pd.isna(fwd_row.get(coin)):
                        continue
                    bucket_records[h][int(b)].append(float(fwd_row[coin]))
        # Aggregate
        out_rows = []
        for b in range(1, n_bins + 1):
            row = {"Decile": b}
            for h in horizons:
                vals = bucket_records[h][b]
                row[f"{h}d mean"] = float(np.mean(vals)) if vals else np.nan
                row[f"{h}d n"] = len(vals)
            out_rows.append(row)
        return pd.DataFrame(out_rows).set_index("Decile")

    # Hard mode — narrow universe, use 5 quintiles to avoid empty bins
    hard_buy_decile_abs = decile_fwd_returns(hard_buy_score, fwd_abs, n_bins=5)
    hard_buy_decile_btc = decile_fwd_returns(hard_buy_score, fwd_btc, n_bins=5)
    hard_sell_decile_abs = decile_fwd_returns(hard_sell_score, fwd_abs, n_bins=5)
    hard_sell_decile_btc = decile_fwd_returns(hard_sell_score, fwd_btc, n_bins=5)

    # Soft mode — broader universe, use 10 deciles
    soft_buy_decile_abs = decile_fwd_returns(soft_buy_score, fwd_abs, n_bins=10)
    soft_buy_decile_btc = decile_fwd_returns(soft_buy_score, fwd_btc, n_bins=10)
    soft_sell_decile_abs = decile_fwd_returns(soft_sell_score, fwd_abs, n_bins=10)
    soft_sell_decile_btc = decile_fwd_returns(soft_sell_score, fwd_btc, n_bins=10)

    return {
        "horizons": horizons,
        "ic_A_abs": ic_A_abs, "ic_B_abs": ic_B_abs, "ic_C_abs": ic_C_abs, "ic_avg_abs": ic_avg_abs,
        "ic_A_btc": ic_A_btc, "ic_B_btc": ic_B_btc, "ic_C_btc": ic_C_btc, "ic_avg_btc": ic_avg_btc,
        "hard_buy_decile_abs": hard_buy_decile_abs,
        "hard_buy_decile_btc": hard_buy_decile_btc,
        "hard_sell_decile_abs": hard_sell_decile_abs,
        "hard_sell_decile_btc": hard_sell_decile_btc,
        "soft_buy_decile_abs": soft_buy_decile_abs,
        "soft_buy_decile_btc": soft_buy_decile_btc,
        "soft_sell_decile_abs": soft_sell_decile_abs,
        "soft_sell_decile_btc": soft_sell_decile_btc,
        "n_eval_bars": len(eval_idx),
    }


run_bt = st.button("🧪 Run / Refresh Validation", type="primary")
if run_bt or "sc_validation" not in st.session_state:
    st.session_state["sc_validation"] = run_validation(
        R["close"], R["bucket_A"], R["bucket_B"], R["bucket_C"], R["avg_z"],
        R["hard_buy_score"], R["hard_sell_score"],
        R["soft_buy_score"], R["soft_sell_score"],
        min_history_bars=int(bt_min_history_bars),
        eval_step=int(bt_eval_step),
    )

V = st.session_state.get("sc_validation")

if V is None:
    st.info("Click **Run / Refresh Validation** above.")
    st.stop()

st.caption(f"Backtest used **{V['n_eval_bars']}** evaluation timestamps.")


# ---------- IC TABLE ----------
st.subheader("Information Coefficient (Spearman) by bucket and horizon")
ic_tab_abs, ic_tab_btc = st.tabs(["📈 vs Absolute Forward Returns", "🟠 vs BTC-Excess Forward Returns"])


def render_IC_block(label_to_df: Dict[str, pd.DataFrame]):
    cols = st.columns(len(label_to_df))
    for col, (label, df) in zip(cols, label_to_df.items()):
        with col:
            st.markdown(f"**{label}**")
            st.dataframe(
                df.style.format({
                    "IC mean": "{:+.3f}",
                    "IC std": "{:.3f}",
                    "IR (mean/std)": "{:+.2f}",
                    "Pos IC %": "{:.0f}%",
                }).background_gradient(
                    subset=["IC mean"], cmap="RdYlGn", vmin=-0.1, vmax=0.1
                ).background_gradient(
                    subset=["IR (mean/std)"], cmap="RdYlGn", vmin=-0.5, vmax=0.5
                ),
                use_container_width=True,
                hide_index=True,
            )


with ic_tab_abs:
    render_IC_block({
        "Bucket A": V["ic_A_abs"],
        "Bucket B": V["ic_B_abs"],
        "Bucket C": V["ic_C_abs"],
        "Avg z (composite)": V["ic_avg_abs"],
    })

with ic_tab_btc:
    render_IC_block({
        "Bucket A": V["ic_A_btc"],
        "Bucket B": V["ic_B_btc"],
        "Bucket C": V["ic_C_btc"],
        "Avg z (composite)": V["ic_avg_btc"],
    })


# ---------- DECILE FORWARD RETURNS ----------
st.markdown("---")
st.subheader("Quantile-binned forward returns (5 / 10 / 15 / 30 bars)")
st.caption(
    "Coins are sorted into bins by their **weighted score** at each historical timestamp, "
    "then average forward return per bin is reported. "
    "**Hard gate uses 5 quintiles** (qualified universe is narrow). "
    "**Soft gate uses 10 deciles** (broader universe). "
    "For BUY ranking, top bin = highest score (expected to outperform). "
    "For SELL ranking, top bin = strongest sell signal (expected to underperform on absolute, "
    "or have the most negative BTC-excess return). "
    "**Spread (correct dir)** flips sign for sell so positive = working as intended."
)


def plot_decile_bars(df: pd.DataFrame, title: str, color_seq: List[str]) -> go.Figure:
    """Grouped bar chart: quantile bins on X (5 or 10), one bar group per horizon."""
    fig = go.Figure()
    horizons_in = [c for c in df.columns if c.endswith(" mean")]
    n_bins = len(df.index)
    for h_col, color in zip(horizons_in, color_seq):
        h_label = h_col.replace(" mean", "")
        fig.add_trace(go.Bar(
            x=df.index.astype(str), y=df[h_col],
            name=h_label,
            marker_color=color,
            text=[f"{v*100:+.2f}%" if pd.notna(v) else "—" for v in df[h_col]],
            textposition="outside",
            textfont=dict(size=9),
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
    bin_label = "Decile" if n_bins == 10 else f"Quantile (1=lowest, {n_bins}=highest)"
    fig.update_layout(
        title=title,
        xaxis_title=bin_label,
        yaxis_title="Mean forward log-return",
        barmode="group",
        height=360,
        template="plotly_dark",
        legend=dict(orientation="h", y=-0.18),
        margin=dict(t=60, b=80),
    )
    return fig


def spread_table(df: pd.DataFrame, side: str) -> pd.DataFrame:
    """
    Top-bin minus bottom-bin spread per horizon.
    Works for any n_bins because it uses df.index.max() / .min().
    """
    rows = []
    horizons_in = [c for c in df.columns if c.endswith(" mean")]
    if df.empty or len(df.index) < 2:
        return pd.DataFrame()
    top_bin = df.index.max()
    bot_bin = df.index.min()
    for h_col in horizons_in:
        h_label = h_col.replace(" mean", "")
        d_top = df.loc[top_bin, h_col]
        d_bot = df.loc[bot_bin, h_col]
        if side == "buy":
            spread = d_top - d_bot   # higher bin expected to outperform
        else:
            spread = d_bot - d_top   # for sell, lowest bin (1) is the WEAKEST sell signal
                                     # so d_bot - d_top measures expected underperformance of strong sells
        rows.append({
            "Horizon": h_label,
            f"Q{top_bin} mean": d_top,
            f"Q{bot_bin} mean": d_bot,
            "Spread (correct dir)": spread,
        })
    return pd.DataFrame(rows)


def render_spread_table(sp: pd.DataFrame):
    """Render a spread table using whatever numeric columns it has."""
    if sp.empty:
        st.info("No spread data (insufficient samples).")
        return
    numeric_cols = [c for c in sp.columns if c != "Horizon"]
    fmt = {c: "{:+.4f}" for c in numeric_cols}
    st.dataframe(
        sp.style.format(fmt).background_gradient(
            subset=["Spread (correct dir)"], cmap="RdYlGn"
        ),
        hide_index=True, use_container_width=True,
    )


horizon_colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444"]

mode_tab_hard, mode_tab_soft = st.tabs([
    "🟦 Hard Binary Gate",
    "🟪 Soft Penalty Gate",
])

# --------- HARD ---------
with mode_tab_hard:
    st.markdown(
        "Decile bins are computed **only over coins that pass the hard gate** at each timestamp. "
        "Sample sizes per decile are smaller, but the test is more conservative."
    )

    sub_abs, sub_btc = st.tabs(["vs Absolute Returns", "vs BTC-Excess Returns"])

    with sub_abs:
        cL, cR = st.columns(2)
        with cL:
            st.markdown("**🟢 Hard BUY decile returns (absolute)**")
            st.plotly_chart(
                plot_decile_bars(V["hard_buy_decile_abs"], "Hard BUY", horizon_colors),
                use_container_width=True,
            )
            sp = spread_table(V["hard_buy_decile_abs"], "buy")
            render_spread_table(sp)
        with cR:
            st.markdown("**🔴 Hard SELL decile returns (absolute)**")
            st.plotly_chart(
                plot_decile_bars(V["hard_sell_decile_abs"], "Hard SELL", horizon_colors),
                use_container_width=True,
            )
            sp = spread_table(V["hard_sell_decile_abs"], "sell")
            render_spread_table(sp)

    with sub_btc:
        cL, cR = st.columns(2)
        with cL:
            st.markdown("**🟢 Hard BUY decile returns (BTC-excess)**")
            st.plotly_chart(
                plot_decile_bars(V["hard_buy_decile_btc"], "Hard BUY (BTC-excess)", horizon_colors),
                use_container_width=True,
            )
            sp = spread_table(V["hard_buy_decile_btc"], "buy")
            render_spread_table(sp)
        with cR:
            st.markdown("**🔴 Hard SELL decile returns (BTC-excess)**")
            st.plotly_chart(
                plot_decile_bars(V["hard_sell_decile_btc"], "Hard SELL (BTC-excess)", horizon_colors),
                use_container_width=True,
            )
            sp = spread_table(V["hard_sell_decile_btc"], "sell")
            render_spread_table(sp)

# --------- SOFT ---------
with mode_tab_soft:
    st.markdown(
        "Decile bins are computed over **all coins passing extension/jump/liquidity** at each "
        "timestamp (broader universe). Soft score includes the λ-penalty for inconsistent buckets."
    )

    sub_abs, sub_btc = st.tabs(["vs Absolute Returns", "vs BTC-Excess Returns"])

    with sub_abs:
        cL, cR = st.columns(2)
        with cL:
            st.markdown("**🟢 Soft BUY decile returns (absolute)**")
            st.plotly_chart(
                plot_decile_bars(V["soft_buy_decile_abs"], "Soft BUY", horizon_colors),
                use_container_width=True,
            )
            sp = spread_table(V["soft_buy_decile_abs"], "buy")
            render_spread_table(sp)
        with cR:
            st.markdown("**🔴 Soft SELL decile returns (absolute)**")
            st.plotly_chart(
                plot_decile_bars(V["soft_sell_decile_abs"], "Soft SELL", horizon_colors),
                use_container_width=True,
            )
            sp = spread_table(V["soft_sell_decile_abs"], "sell")
            render_spread_table(sp)

    with sub_btc:
        cL, cR = st.columns(2)
        with cL:
            st.markdown("**🟢 Soft BUY decile returns (BTC-excess)**")
            st.plotly_chart(
                plot_decile_bars(V["soft_buy_decile_btc"], "Soft BUY (BTC-excess)", horizon_colors),
                use_container_width=True,
            )
            sp = spread_table(V["soft_buy_decile_btc"], "buy")
            render_spread_table(sp)
        with cR:
            st.markdown("**🔴 Soft SELL decile returns (BTC-excess)**")
            st.plotly_chart(
                plot_decile_bars(V["soft_sell_decile_btc"], "Soft SELL (BTC-excess)", horizon_colors),
                use_container_width=True,
            )
            sp = spread_table(V["soft_sell_decile_btc"], "sell")
            render_spread_table(sp)


# =============================================================================
# 11. FOOTER
# =============================================================================
st.markdown("---")
st.caption(
    f"Last fetch: **{st.session_state.get('sc_last_fetch')}** | "
    f"Timeframe: **{st.session_state.get('sc_timeframe')}** | "
    f"λ: **{lambda_penalty}** | "
    f"top-N: **{int(top_n)}** | "
    f"Universe: BTC + {len(R['close'].columns)-1} alts"
)
