"""
=============================================================================
 PAGE 16 — SIGNAL CONSOLIDATOR (Long/Short Market-Neutral)
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

# Column reference — explains the abbreviated column names that appear in tables below.
with st.expander("📖 Column reference — what these names mean", expanded=False):
    st.markdown(
        """
**Score columns**

| Column | Meaning |
|---|---|
| **Score** | Final hybrid score used to rank coins. <br/> *Hard mode*: avg(Theme-rel z, BTC-rel z, Absolute z). <br/> *Soft mode*: same average minus λ × Σ max(0, −z_bucket)² (λ-penalty for buckets pulling against the trade). For sell-side both are sign-flipped so a higher score = stronger sell. |
| **Theme-rel z**  (Bucket A) | Cross-sectional z-score of momentum **vs the coin's theme peers**. Mean of three lookbacks (14d, 30d, 60d) of the coin's excess log-return over the median coin in its theme. |
| **BTC-rel z**  (Bucket B) | CS z-score of momentum **vs BTC**. Mean of three lookbacks (14d, 30d, 90d) of the Coin/BTC log-return. |
| **Absolute z**  (Bucket C) | CS z-score of **absolute momentum**. Mean of (a) 30d & 60d ATR-vol-scaled returns and (b) multi-MA trend agreement count (price > SMA-20 + price > SMA-50 + price > SMA-100). |

**Extension / risk columns**

| Column | Meaning |
|---|---|
| **price_z_60** | How many σ the close sits above (+) or below (−) its own 60-bar mean. > +2 = extended overbought; < −2 = extended oversold. |
| **range_pct_90** | Position within the 90-bar high-low range. 0 = at 90d low, 1 = at 90d high. > 0.92 trips the buy-extension gate; < 0.08 trips the sell-extension gate. |
| **ATR_pct** | 14-bar Average True Range as a % of price — a proxy for realized volatility. Useful when sizing positions. |

**Gating (Stage 3)**

| Mode | Buy qualified when… | Sell qualified when… |
|---|---|---|
| **Hard binary** | all 3 buckets `> 0` AND extension/jump/liquidity OK | all 3 buckets `< 0` AND extension/jump/liquidity OK (mirrored) |
| **Soft penalty** | any coin passing extension/jump/liquidity; the score is just penalized for negative buckets via `λ · Σ max(0, −z)²` | mirror — coin passes mirrored extension; score penalized for **positive** buckets |
        """
    )

with st.expander("📖 **Column key** — what the scores and metrics mean", expanded=False):
    st.markdown(
        """
| Column | Meaning |
|---|---|
| **Theme-rel z** *(bucket A)* | Cross-sectional z-score of how the coin has moved **vs other coins in its theme**. Mean of 3 horizons (14/30/60 bars). Captures intra-theme leadership. |
| **BTC-rel z** *(bucket B)* | Cross-sectional z-score of the **Coin/BTC ratio's return** at 14/30/90 bars. Captures alpha vs the market beta. |
| **Absolute z** *(bucket C)* | Cross-sectional z-score of **ATR-scaled returns** (30/60d) plus multi-MA trend agreement (price vs SMA-20/50/100). Captures absolute-price strength. |
| **Score** | The final ranking score. **Hard gate**: simple mean of the three bucket z-scores, only computed when all three are positive (buy) or all three negative (sell). **Soft gate**: mean − λ·Σmax(0,−z)² penalty for any bucket pulling the wrong way. |
| **price_z_60** | Standardised distance of price from its 60-bar mean. **>2** = stretched; buy candidates above this are rejected. |
| **range_pct_90** | Where price sits in its 90-bar high–low range. **>0.92** = near-top; buy candidates above this are rejected. |
| **ATR_pct** | Average True Range as % of price. Volatility scale. |
| **Cumulative log-return (%)** *(validation charts)* | Compounded log-return of holding the top-bin or bottom-bin basket through non-overlapping rebalances. **Q-top** (or D-top) = highest-score bin, **Q-bot** = lowest. **abs** = absolute return; **BTC-ex** = BTC-excess return (relevant for L/S MN). |
        """
    )

# -- GLOSSARY --
with st.expander("📖 Column glossary — what does each value mean?"):
    st.markdown(
        """
**Score** — the final ranking score for the gate type. Hard binary uses `(z_Theme + z_BTC + z_Abs)/3`
on coins that pass all gates. Soft penalty uses the same average minus
`λ × Σ max(0, −z_bucket)²`. Higher = stronger conviction.

**Bucket scores (averaged cross-sectional z-scores within each bucket):**
- **`z_Theme`** — theme-relative momentum bucket (was `z_A`). Average of CS z-scores of
  30d / 60d / 14d excess log-return vs the coin's theme median.
  Positive ⇒ outperforming theme peers.
- **`z_BTC`** — BTC-relative momentum bucket (was `z_B`). Average of CS z-scores of
  30d / 90d / 14d Coin/BTC log-returns. Positive ⇒ outperforming BTC.
- **`z_Abs`** — absolute / universe-wide momentum bucket (was `z_C`). Average of CS z-scores of
  30d / 60d ATR-scaled returns + a multi-MA trend agreement count
  (price > SMA-20, > SMA-50, > SMA-100). Positive ⇒ strong absolute uptrend in vol-adjusted terms.

**Filter / context columns:**
- **`price_z (60d)`** — number of σ the current close is above its own 60-bar mean. Buy reject if > 2.0.
- **`range_pct (90d)`** — where the current close sits in the 90-bar high-low range (0=low, 1=high). Buy reject if > 0.92.
- **`ATR_pct`** — 14-bar ATR as a percentage of price. Coin's typical bar move size.

**Conventions:** all returns are log-returns. All z-scores are cross-sectional
(within a single bar, across the universe). All rolling windows use past data only — no lookahead.
        """
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
        help="Skip the first N bars of history when computing forward-return curves."
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
            "Score": round(float(s[coin]), 2),
            "Theme-rel z": round(float(R["bucket_A"].iloc[-1].get(coin, np.nan)), 2),
            "BTC-rel z": round(float(R["bucket_B"].iloc[-1].get(coin, np.nan)), 2),
            "Absolute z": round(float(R["bucket_C"].iloc[-1].get(coin, np.nan)), 2),
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
                .format(precision=2)
                .background_gradient(subset=["Score"], cmap="Greens")
                .background_gradient(subset=["Theme-rel z", "BTC-rel z", "Absolute z"], cmap="RdYlGn", vmin=-2, vmax=2)
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
                .format(precision=2)
                .background_gradient(subset=["Score"], cmap="Reds")
                .background_gradient(subset=["Theme-rel z", "BTC-rel z", "Absolute z"], cmap="RdYlGn", vmin=-2, vmax=2)
                .background_gradient(subset=["price_z_60"], cmap="RdYlGn_r", vmin=-3, vmax=3),
                hide_index=True,
                use_container_width=True,
                height=min(420, 60 + 35 * len(hard_sell_tbl)),
            )

# ---------- SOFT ----------
with tab_soft:
    st.markdown(
        f"**Soft penalty**: score = mean(z_Theme, z_BTC, z_Abs) − λ·Σmax(0, −z)² for buy "
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
                .format(precision=2)
                .background_gradient(subset=["Score"], cmap="Greens")
                .background_gradient(subset=["Theme-rel z", "BTC-rel z", "Absolute z"], cmap="RdYlGn", vmin=-2, vmax=2)
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
                .format(precision=2)
                .background_gradient(subset=["Score"], cmap="Reds")
                .background_gradient(subset=["Theme-rel z", "BTC-rel z", "Absolute z"], cmap="RdYlGn", vmin=-2, vmax=2)
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
        # Latest values — use descriptive names with horizons
        last_row = {
            "Theme-rel z (30d)":      R["z_A1"].iloc[-1].get(sel_coin, np.nan),
            "Theme-rel z (60d)":      R["z_A2"].iloc[-1].get(sel_coin, np.nan),
            "Theme-rel z (14d)":      R["z_A3"].iloc[-1].get(sel_coin, np.nan),
            "Theme-rel z  (avg, A)":  R["bucket_A"].iloc[-1].get(sel_coin, np.nan),
            "BTC-rel z (30d)":        R["z_B1"].iloc[-1].get(sel_coin, np.nan),
            "BTC-rel z (90d)":        R["z_B2"].iloc[-1].get(sel_coin, np.nan),
            "BTC-rel z (14d)":        R["z_B3"].iloc[-1].get(sel_coin, np.nan),
            "BTC-rel z  (avg, B)":    R["bucket_B"].iloc[-1].get(sel_coin, np.nan),
            "Absolute z (30d ATR-scaled)": R["z_C1"].iloc[-1].get(sel_coin, np.nan),
            "Absolute z (60d ATR-scaled)": R["z_C2"].iloc[-1].get(sel_coin, np.nan),
            "Absolute z (Multi-MA trend)": R["z_C3"].iloc[-1].get(sel_coin, np.nan),
            "Absolute z  (avg, C)":   R["bucket_C"].iloc[-1].get(sel_coin, np.nan),
            "Composite avg z":        R["avg_z"].iloc[-1].get(sel_coin, np.nan),
        }
        df_view = pd.DataFrame.from_dict(last_row, orient="index", columns=["Latest z"])
        df_view["Latest z"] = df_view["Latest z"].astype(float).round(2)
        st.dataframe(df_view, use_container_width=False)

        # Time series chart of buckets
        sub_df = pd.DataFrame({
            "Theme-rel z (A)":  R["bucket_A"][sel_coin],
            "BTC-rel z (B)":    R["bucket_B"][sel_coin],
            "Absolute z (C)":   R["bucket_C"][sel_coin],
            "Composite avg z": R["avg_z"][sel_coin],
        })
        fig_buc = go.Figure()
        for col, color in zip(
            ["Theme-rel z (A)", "BTC-rel z (B)", "Absolute z (C)", "Composite avg z"],
            ["#60a5fa", "#a78bfa", "#34d399", "#fbbf24"],
        ):
            fig_buc.add_trace(go.Scatter(
                x=sub_df.index, y=sub_df[col], name=col, mode="lines",
                line=dict(color=color, width=1.5 if col != "Composite avg z" else 2.5),
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
    hard_buy_score: pd.DataFrame,
    hard_sell_score: pd.DataFrame,
    soft_buy_score: pd.DataFrame,
    soft_sell_score: pd.DataFrame,
    min_history_bars: int,
) -> Dict:
    """
    Cumulative-return validation.

    For each gate side (hard/soft × buy/sell) and each holding period h ∈ {5,10,15,30},
    sample at NON-OVERLAPPING intervals (every h bars) starting from `min_history_bars`.
    At each sample t:
      - rank coins with valid scores into n_bins,
      - compute mean forward h-bar log-return for the bottom bin (Q1) and top bin (Q-top),
      - in BOTH absolute and BTC-excess return space.
    Cumsum per series → cumulative log-return curve. The final-period
    summary spread (Q-top − Q-bot, sign-flipped for sell) is also returned.

    Returns nested dict: results[side]['series'][h] = DataFrame indexed by date,
                          results[side]['summary']    = DataFrame summary.
    """
    horizons = [5, 10, 15, 30]

    fwd_abs = compute_forward_returns(R_close, horizons)
    fwd_btc = compute_btc_excess_forward_returns(R_close, horizons)

    def compute_qcurve(score_df: pd.DataFrame, n_bins: int, side: str):
        """
        Returns (series_dict, summary_df).
        series_dict[h] = DataFrame[date, [Qbot_abs, Qtop_abs, Qbot_btc, Qtop_btc]] cumsum'd.
        """
        series_out = {}
        summary_rows = []
        base_idx = R_close.index
        if min_history_bars >= len(base_idx):
            return series_out, pd.DataFrame()

        for h in horizons:
            # Non-overlapping samples: every h bars, starting at min_history_bars,
            # ending h bars before the last (so forward return is observable).
            last_eligible_pos = len(base_idx) - h - 1
            if last_eligible_pos <= min_history_bars:
                series_out[h] = pd.DataFrame()
                summary_rows.append({
                    "Horizon": f"{h}d", "Q-top abs (final %)": np.nan,
                    "Q-bot abs (final %)": np.nan, "Spread abs (correct dir, %)": np.nan,
                    "Q-top BTC-ex (final %)": np.nan, "Q-bot BTC-ex (final %)": np.nan,
                    "Spread BTC-ex (correct dir, %)": np.nan, "n samples": 0,
                })
                continue

            sample_positions = list(range(min_history_bars, last_eligible_pos + 1, h))
            sample_idx = base_idx[sample_positions]

            records = []
            for t in sample_idx:
                if t not in score_df.index:
                    continue
                s = score_df.loc[t].dropna()
                if len(s) < n_bins:
                    continue
                try:
                    bins = pd.qcut(s, n_bins, labels=False, duplicates="drop") + 1
                except ValueError:
                    continue
                top_bin_id = int(bins.max())
                bot_bin_id = int(bins.min())
                if top_bin_id == bot_bin_id:
                    continue  # all coins in one bin → no spread
                qtop_coins = bins[bins == top_bin_id].index
                qbot_coins = bins[bins == bot_bin_id].index

                r_abs = fwd_abs[h].loc[t] if t in fwd_abs[h].index else None
                r_btc = fwd_btc[h].loc[t] if t in fwd_btc[h].index else None
                if r_abs is None or r_btc is None:
                    continue

                qtop_abs = float(r_abs.reindex(qtop_coins).dropna().mean())
                qbot_abs = float(r_abs.reindex(qbot_coins).dropna().mean())
                qtop_btc = float(r_btc.reindex(qtop_coins).dropna().mean())
                qbot_btc = float(r_btc.reindex(qbot_coins).dropna().mean())

                records.append({
                    "date": t,
                    "Qbot_abs": qbot_abs, "Qtop_abs": qtop_abs,
                    "Qbot_btc": qbot_btc, "Qtop_btc": qtop_btc,
                })

            if records:
                df = pd.DataFrame(records).set_index("date")
                df = df.cumsum()  # cumulative log returns
                series_out[h] = df

                # Convert log-cumulative to % for the final-row summary
                final_log = df.iloc[-1]
                qtop_abs_pct = (np.exp(final_log["Qtop_abs"]) - 1.0) * 100
                qbot_abs_pct = (np.exp(final_log["Qbot_abs"]) - 1.0) * 100
                qtop_btc_pct = (np.exp(final_log["Qtop_btc"]) - 1.0) * 100
                qbot_btc_pct = (np.exp(final_log["Qbot_btc"]) - 1.0) * 100

                if side == "buy":
                    spread_abs = qtop_abs_pct - qbot_abs_pct
                    spread_btc = qtop_btc_pct - qbot_btc_pct
                else:  # sell — Q-top is the strongest sell, expected to UNDERPERFORM
                    spread_abs = qbot_abs_pct - qtop_abs_pct
                    spread_btc = qbot_btc_pct - qtop_btc_pct

                summary_rows.append({
                    "Horizon": f"{h}d",
                    "Q-top abs (final %)": qtop_abs_pct,
                    "Q-bot abs (final %)": qbot_abs_pct,
                    "Spread abs (correct dir, %)": spread_abs,
                    "Q-top BTC-ex (final %)": qtop_btc_pct,
                    "Q-bot BTC-ex (final %)": qbot_btc_pct,
                    "Spread BTC-ex (correct dir, %)": spread_btc,
                    "n samples": len(df),
                })
            else:
                series_out[h] = pd.DataFrame()
                summary_rows.append({
                    "Horizon": f"{h}d", "Q-top abs (final %)": np.nan,
                    "Q-bot abs (final %)": np.nan, "Spread abs (correct dir, %)": np.nan,
                    "Q-top BTC-ex (final %)": np.nan, "Q-bot BTC-ex (final %)": np.nan,
                    "Spread BTC-ex (correct dir, %)": np.nan, "n samples": 0,
                })

        summary_df = pd.DataFrame(summary_rows)
        return series_out, summary_df

    # Hard mode → 5 quintiles (narrow universe)
    hard_buy_series, hard_buy_summary = compute_qcurve(hard_buy_score, n_bins=5, side="buy")
    hard_sell_series, hard_sell_summary = compute_qcurve(hard_sell_score, n_bins=5, side="sell")

    # Soft mode → 10 deciles (broader universe)
    soft_buy_series, soft_buy_summary = compute_qcurve(soft_buy_score, n_bins=10, side="buy")
    soft_sell_series, soft_sell_summary = compute_qcurve(soft_sell_score, n_bins=10, side="sell")

    return {
        "horizons": horizons,
        "hard_buy_series": hard_buy_series, "hard_buy_summary": hard_buy_summary,
        "hard_sell_series": hard_sell_series, "hard_sell_summary": hard_sell_summary,
        "soft_buy_series": soft_buy_series, "soft_buy_summary": soft_buy_summary,
        "soft_sell_series": soft_sell_series, "soft_sell_summary": soft_sell_summary,
    }


run_bt = st.button("🧪 Run / Refresh Validation", type="primary")
if run_bt or "sc_validation" not in st.session_state:
    st.session_state["sc_validation"] = run_validation(
        R["close"],
        R["hard_buy_score"], R["hard_sell_score"],
        R["soft_buy_score"], R["soft_sell_score"],
        min_history_bars=int(bt_min_history_bars),
    )

V = st.session_state.get("sc_validation")

if V is None:
    st.info("Click **Run / Refresh Validation** above.")
    st.stop()

# Quick coverage banner — sample counts from the summary tables
def _sample_count(summary_df: pd.DataFrame) -> int:
    """Read 'n samples' from a summary DataFrame across horizons. Returns max as a banner figure."""
    if summary_df is None or summary_df.empty or "n samples" not in summary_df.columns:
        return 0
    try:
        return int(summary_df["n samples"].max())
    except Exception:
        return 0


_n_max = max(
    _sample_count(V.get("hard_buy_summary")),
    _sample_count(V.get("soft_buy_summary")),
)
st.caption(
    f"Backtest uses non-overlapping holding periods. "
    f"Max rebalance count across horizons: **{_n_max}**."
)


# =============================================================================
# RENDERING HELPERS — cumulative-return line charts and 2dp summary tables
# =============================================================================
def plot_cumret_chart(
    series_df: pd.DataFrame,
    h: int,
    n_bins: int,
    side: str,
    gate: str,
) -> go.Figure:
    """
    Plot cumulative-log-return curves for the top and bottom bins,
    in BOTH absolute and BTC-excess return space, on a single chart.

    series_df columns expected: ['Qbot_abs', 'Qtop_abs', 'Qbot_btc', 'Qtop_btc']
                                already cumsum'd (cumulative log-return).
    Y-axis is rendered as % (log-return × 100). For ranges < ~30 % this is
    visually indistinguishable from compounded simple return.
    """
    fig = go.Figure()
    if series_df is None or series_df.empty:
        fig.update_layout(
            title=f"{gate.title()} {side.upper()} — {h}-bar holding (no data)",
            template="plotly_dark", height=320,
        )
        return fig

    # Display labels: "Q5" / "Q1" for hard (n_bins=5), "D10" / "D1" for soft (n_bins=10)
    top_lab = f"Q{n_bins}" if n_bins != 10 else "D10"
    bot_lab = "Q1" if n_bins != 10 else "D1"

    x = series_df.index
    fig.add_trace(go.Scatter(
        x=x, y=series_df["Qtop_abs"] * 100,
        name=f"{top_lab} (top) — abs", mode="lines",
        line=dict(color="#22c55e", width=2.2),
        hovertemplate="%{y:+.2f}%<extra>" + f"{top_lab} abs" + "</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=series_df["Qbot_abs"] * 100,
        name=f"{bot_lab} (bot) — abs", mode="lines",
        line=dict(color="#ef4444", width=2.2),
        hovertemplate="%{y:+.2f}%<extra>" + f"{bot_lab} abs" + "</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=series_df["Qtop_btc"] * 100,
        name=f"{top_lab} (top) — BTC-ex", mode="lines",
        line=dict(color="#10b981", width=2.0, dash="dash"),
        hovertemplate="%{y:+.2f}%<extra>" + f"{top_lab} BTC-ex" + "</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=series_df["Qbot_btc"] * 100,
        name=f"{bot_lab} (bot) — BTC-ex", mode="lines",
        line=dict(color="#f97316", width=2.0, dash="dash"),
        hovertemplate="%{y:+.2f}%<extra>" + f"{bot_lab} BTC-ex" + "</extra>",
    ))

    fig.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.35)
    title_side = "BUY" if side == "buy" else "SELL"
    fig.update_layout(
        title=f"{gate.title()} {title_side} — {h}-bar holding period",
        xaxis_title="Rebalance date",
        yaxis_title="Cumulative log-return (%)",
        template="plotly_dark",
        height=340,
        legend=dict(orientation="h", y=-0.22, font=dict(size=10)),
        hovermode="x unified",
        margin=dict(t=50, b=70, l=10, r=10),
    )
    return fig


def render_summary_table(summary_df: pd.DataFrame, side: str):
    """Render the 2dp summary table for a side. Includes spread (correct dir)."""
    if summary_df is None or summary_df.empty:
        st.info("No summary data.")
        return
    df = summary_df.copy()
    # Ensure n samples is integer (it's already int but safe)
    if "n samples" in df.columns:
        df["n samples"] = df["n samples"].astype(int)
    pct_cols = [c for c in df.columns if c.endswith("%)") and c != "n samples"]
    fmt = {c: "{:+.2f}" for c in pct_cols}
    spread_cols = [c for c in pct_cols if c.startswith("Spread ")]
    styled = df.style.format(fmt)
    if spread_cols:
        styled = styled.background_gradient(
            subset=spread_cols, cmap="RdYlGn", vmin=-15, vmax=15
        )
    st.dataframe(styled, hide_index=True, use_container_width=True)


# =============================================================================
# CUMULATIVE-RETURN CHARTS
# =============================================================================
st.markdown("---")
st.subheader("Cumulative-return validation — top vs bottom bins")
st.caption(
    "At each non-overlapping rebalance date (every h bars from the start of the backtest window), "
    "coins are sorted into bins by their **weighted score**. "
    "The chart shows the cumulative log-return (%) of the **top bin** and **bottom bin** — "
    "in **absolute** terms (solid lines) and in **BTC-excess** terms (dashed). "
    "**Hard gate** uses 5 quintiles (qualified universe is narrow). "
    "**Soft gate** uses 10 deciles (broader universe). "
    "For BUY, top should outperform bottom. For SELL, the top bin = strongest sell candidate, "
    "so it should underperform the bottom bin (the **Spread (correct dir)** in the summary "
    "flips the sign for sell so a positive spread always means the model is working)."
)


HORIZONS = [5, 10, 15, 30]


def render_holding_period_grid(series_dict: Dict[int, pd.DataFrame], n_bins: int, side: str, gate: str):
    """
    Render a 2x2 grid: one chart per holding period.
    Each chart has 4 lines (Q-top abs, Q-bot abs, Q-top BTC-ex, Q-bot BTC-ex).
    """
    rows = [HORIZONS[:2], HORIZONS[2:]]  # [[5, 10], [15, 30]]
    for row in rows:
        cols = st.columns(2)
        for col, h in zip(cols, row):
            with col:
                st.plotly_chart(
                    plot_cumret_chart(series_dict.get(h, pd.DataFrame()), h, n_bins, side, gate),
                    use_container_width=True,
                )


mode_tab_hard, mode_tab_soft = st.tabs([
    "🟦 Hard Binary Gate",
    "🟪 Soft Penalty Gate",
])

# --------- HARD ---------
with mode_tab_hard:
    st.markdown(
        "Bins are computed **only over coins that pass the hard gate** (all 3 buckets agree "
        "in direction + extension/jump/liquidity OK) at each rebalance date. "
        "Quintiles (5 bins) are used since the qualified universe is narrow."
    )
    sub_buy, sub_sell = st.tabs(["🟢 BUY side", "🔴 SELL side"])

    with sub_buy:
        render_holding_period_grid(V["hard_buy_series"], n_bins=5, side="buy", gate="Hard")
        st.markdown("**Final cumulative-return summary (each row is a separate holding period)**")
        render_summary_table(V["hard_buy_summary"], side="buy")

    with sub_sell:
        render_holding_period_grid(V["hard_sell_series"], n_bins=5, side="sell", gate="Hard")
        st.markdown("**Final cumulative-return summary (each row is a separate holding period)**")
        render_summary_table(V["hard_sell_summary"], side="sell")

# --------- SOFT ---------
with mode_tab_soft:
    st.markdown(
        "Bins are computed over **all coins passing extension/jump/liquidity** at each "
        "rebalance date (broader universe). The soft score includes a λ-penalty for "
        "buckets pulling against the trade direction. Deciles (10 bins) are used."
    )
    sub_buy, sub_sell = st.tabs(["🟢 BUY side", "🔴 SELL side"])

    with sub_buy:
        render_holding_period_grid(V["soft_buy_series"], n_bins=10, side="buy", gate="Soft")
        st.markdown("**Final cumulative-return summary (each row is a separate holding period)**")
        render_summary_table(V["soft_buy_summary"], side="buy")

    with sub_sell:
        render_holding_period_grid(V["soft_sell_series"], n_bins=10, side="sell", gate="Soft")
        st.markdown("**Final cumulative-return summary (each row is a separate holding period)**")
        render_summary_table(V["soft_sell_summary"], side="sell")


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
