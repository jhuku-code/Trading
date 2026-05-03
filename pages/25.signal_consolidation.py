"""
=============================================================================
 PAGE 16 — SIGNAL CONSOLIDATOR  (Long/Short Market-Neutral)
=============================================================================

PURPOSE
-------
Standalone Streamlit page consolidating buy/sell signal generation for a
long/short market-neutral crypto book. This page does NOT depend on any
other page — it fetches its own OHLCV from KuCoin (ccxt) and reads themes
from `Input-Files/Themes_mapping.xlsx`.

This is the v2 architecture, which fixes two known failure modes of the
naive "trend-following both sides" v1 design:

  FAILURE 1 — POST-EVENT CONTAMINATION (e.g. ENJ April-2026 spike)
    A single 5-day rip puts a coin near the top of every momentum bucket
    even though the move is now stale. The 5-bar jump filter from v1 was
    too short — by the time the signal fires, the spike has rolled out.

  FAILURE 2 — STALE TRENDS ON THE SELL SIDE (e.g. DOT in early May 2026)
    A coin that has been in a 6-month grinding downtrend is identified
    as a "sell" by short-window momentum signals — but rolling-window
    `price_z_60` and `range_pct_90` drift down with the price, so the
    coin never registers as "extended low". Most of the move is gone,
    funding is inverted, and the squeeze risk is asymmetric.

The four architectural fixes vs v1:

  (1) STRONGER JUMP FILTER — 30-bar window, plus a concentration-ratio
      check (max single-bar |return| / sum of |returns|).
  (2) SMOOTHNESS REQUIREMENT — R² of log(price) vs time over 30 bars.
      Real trends grind (high R²); spikes don't (low R²).
  (3) LONGER-WINDOW RANGE FILTER — 180-bar range_pct on top of the 90d
      check. Catches "already cooked" sell candidates and "already
      pumped" buy candidates.
  (4) TWO SELL SUB-STRATEGIES instead of symmetric trend-following:
        • BEST-OF-WORST (BoW): theme-leaders losing strength vs BTC.
          Theme-rel > 0, BTC-rel < 0, Absolute < 0. Early sell.
        • REVERSAL / EXHAUSTION (RE): coins that were STRONG over 180d
          and are now weakening over 14d and 30d. Catches tops.

STRATEGY OVERVIEW
-----------------

  STAGE 0 — UNIVERSE FILTER (hard, applied first)
    • Liquidity: median 30-bar USD volume ≥ threshold
    • Jump (close-to-close): max |log-return| in last 30 bars ≤ threshold
    • Concentration ratio: largest |log-return| / sum |log-return| ≤ threshold
    • Wick jump: max (H-L)/prev_close in last 5 bars ≤ threshold
    • Smoothness: R² of log(price) vs time over 30 bars ≥ threshold
        (smoothness applied to BUY and BoW sell only, NOT to RE sell)

  STAGE 1 — BUCKET SIGNALS (cross-sectional z-scores; long/short symmetric)
    Bucket A (Theme-rel)  = mean of CS-z(14d), CS-z(30d), CS-z(60d) excess
                            log-return vs theme median.
    Bucket B (BTC-rel)    = mean of CS-z(14d), CS-z(30d), CS-z(90d)
                            Coin/BTC log-return.
    Bucket C (Absolute)   = mean of CS-z(30d ATR-scaled return),
                                    CS-z(60d ATR-scaled return),
                                    CS-z(multi-MA trend count).

  STAGE 2 — EXTENSION / RANGE FILTERS (time-varying booleans)
    BUY rejects:    price_z_60 > 2.0,  range_pct_90 > 0.92,  range_pct_180 > 0.85
    BoW rejects:    price_z_60 < -2.0, range_pct_180 < 0.30
    RE rejects:     price_z_60 < -2.0, range_pct_180 < 0.50  (RE WANTS high range)

  STAGE 3 — HYBRID GATING (hard binary AND soft penalty for each strategy)

    BUY (trend-following, all-3-agree):
      Hard qualified ⟺ z_A > 0 ∧ z_B > 0 ∧ z_C > 0  ∧ buy_pre_gate
      Hard score      = mean(z)
      Soft score      = mean(z) − λ · Σ max(0, −z_bucket)²   (over buy_pre_gate)

    SELL — BEST-OF-WORST:
      Hard qualified ⟺ z_A > 0 ∧ z_B < 0 ∧ z_C < 0  ∧ bow_pre_gate
      Hard score      = z_A − z_B − z_C    (rewards the gap)
      Soft score      = z_A − z_B − z_C − λ · [max(0,−z_A)² + max(0,z_B)² + max(0,z_C)²]
                        (over bow_pre_gate)

    SELL — REVERSAL / EXHAUSTION:
      Hard qualified ⟺ ret_180d_pct ≥ Q3 ∧ ret_14d < 0 ∧ ret_30d < 0  ∧ re_pre_gate
      Hard score      = z_180d_ret − z_14d_ret − z_30d_ret
      Soft score      = z_180d_ret − z_14d_ret − z_30d_ret
                        − λ · [max(0,−z_180d)² + max(0,z_14d)² + max(0,z_30d)²]
                        (over re_pre_gate, still gated by 180d strength quartile)

  STAGE 4 — RANKING & SELECTION
    Top-N by score per side, equal-weighted. Book is dollar-neutral.
    Sell allocation can be split between BoW and RE (e.g. 50/50).

VALIDATION
----------
  Cumulative-return curves of top vs bottom bins, non-overlapping rebalances,
  separately for absolute and BTC-excess returns. One chart per holding
  period (5/10/15/30 bars), one tab per (gate × side × strategy) combination.

ALL ROLLING/EXPANDING WINDOWS USE PAST DATA ONLY — no lookahead.
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
    page_title="Signal Consolidator (L/S MN) v2",
    layout="wide",
    page_icon="🎯",
)
st.title("🎯 Signal Consolidator v2 — Long/Short Market-Neutral")
st.caption(
    "Trend-following BUY + asymmetric SELL (Best-of-Worst + Reversal/Exhaustion). "
    "Stronger jump filter, smoothness gate, and longer-window range checks. "
    "Standalone — fetches own data."
)

# Column reference — explains the abbreviated column names that appear in tables below.
with st.expander("📖 Column reference — what these names mean", expanded=False):
    st.markdown(
        """
**Bucket scores (cross-sectional z-scores; equal-weighted across horizons)**

| Column | Meaning |
|---|---|
| **Theme-rel z** (Bucket A) | Momentum vs the coin's theme peers. Mean of CS-z of 14d, 30d, 60d excess log-return over the median coin in its theme. |
| **BTC-rel z** (Bucket B) | Momentum vs BTC. Mean of CS-z of 14d, 30d, 90d Coin/BTC log-return. |
| **Absolute z** (Bucket C) | Absolute momentum. Mean of CS-z of 30d & 60d ATR-vol-scaled returns and the multi-MA trend agreement count (price > SMA-20 + price > SMA-50 + price > SMA-100). |

**Reversal/Exhaustion sell helpers**

| Column | Meaning |
|---|---|
| **Ret 180d** | Total log-return over the last 180 bars. RE requires this to be in the top quartile cross-sectionally — coins that were strong. |
| **Ret 14d / Ret 30d** | Recent log-returns. RE requires both to be negative — coins that are now weakening. |

**Risk / extension columns**

| Column | Meaning |
|---|---|
| **price_z_60** | (close − SMA-60) / σ-60. > +2 = extended overbought; < −2 = extended oversold. |
| **range_pct_90** | Position within the 90-bar high-low range. 0 = at 90d low, 1 = at 90d high. |
| **range_pct_180** | Same as above but over 180 bars. Catches stale trends — a coin grinding down for 6 months has range_pct_180 stuck near 0 even though range_pct_90 might be moderate. |
| **R² (30d)** | Coefficient of determination of log(price) vs time over 30 bars. **High = grinding trend** (good for BUY/BoW). **Low = noisy/jumpy** (rejected for BUY/BoW). RE does not use this filter — RE wants smoothness historically but breakdown now. |
| **Concen ratio** | max(\\|return\\|) / sum(\\|return\\|) over 30 bars. Detects single-bar moves masquerading as trends. > ~0.30 typically means one event dominates. |
| **ATR_pct** | 14-bar ATR as % of price. Realized volatility proxy. |

**Gating modes (Stage 3)**

| Mode | What it does |
|---|---|
| **Hard binary** | All required bucket signs must agree AND all extension/jump/smoothness gates must pass. Highest conviction; smaller universe. |
| **Soft penalty** | Coin must pass extension/jump/smoothness gates; the score is then *penalized* via `λ · Σ max(0, wrong-sign-z)²` rather than rejected. Broader universe; signal magnitude is preserved. |
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
        "OHLCV bars to fetch", value=600, min_value=300, max_value=2000, step=50,
        help="v2 uses a 180-bar range filter and 180-bar return for RE sell, "
             "so you need ≥ 240 bars (180 + warmup buffer). Default 600 ≈ 1.6 yrs daily."
    )
    sleep_seconds = st.number_input("Sleep between fetches (s)", value=0.2, step=0.05)

    st.markdown("---")
    st.header("🚦 Stage 0 — Universe Filter")
    min_usd_vol = st.number_input(
        "Min median 30-bar USD volume ($)",
        value=1_000_000, min_value=0, max_value=100_000_000, step=100_000,
        help="Coins below this median dollar volume are dropped at every bar."
    )
    jump_thresh_pct = st.slider(
        "JUMP filter — max single-bar |return| in last **30** bars (%)",
        min_value=10.0, max_value=80.0, value=25.0, step=1.0,
        help="v2 fix: was 5 bars in v1; ENJ-style spike still in window when "
             "30/60d momentum signals fire. 30-bar window catches it cleanly."
    )
    concen_thresh = st.slider(
        "CONCENTRATION RATIO — max( |ret| ) / sum( |ret| ) over 30 bars",
        min_value=0.10, max_value=0.80, value=0.30, step=0.02,
        help="Detects single-bar moves masquerading as trends. "
             "1/30 ≈ 0.033 = perfectly uniform; 1.0 = one bar drives everything. "
             "Threshold of 0.30 means no single bar can be > 30% of total |move|."
    )
    wick_thresh_pct = st.slider(
        "WICK filter — max (H−L)/prev_close in last 5 bars (%)",
        min_value=5.0, max_value=80.0, value=20.0, step=1.0,
        help="Catches intraday explosive moves even when close-to-close is muted."
    )
    smooth_r2_min = st.slider(
        "SMOOTHNESS — min R² of log(price) vs time over 30 bars",
        min_value=0.0, max_value=0.95, value=0.35, step=0.05,
        help="Applied to BUY and BoW SELL only (NOT to Reversal/Exhaustion sell). "
             "High R² = grinding trend; low R² = noisy/spiky. "
             "0.35 is moderate; 0.5+ is strict."
    )

    st.markdown("---")
    st.header("🧱 Stage 2 — Range / Extension")
    price_z60_max = st.slider(
        "BUY/SELL reject if |price_z_60| > ",
        min_value=1.0, max_value=4.0, value=2.0, step=0.25,
        help="Extension vs the 60-bar mean. Symmetric for buy and sell."
    )
    range_pct_90_buy_max = st.slider(
        "BUY reject if range_pct_90 > ",
        min_value=0.70, max_value=0.99, value=0.92, step=0.01,
    )
    range_pct_180_buy_max = st.slider(
        "BUY reject if range_pct_180 > ",
        min_value=0.60, max_value=0.99, value=0.85, step=0.01,
        help="v2 fix: catches coins already at multi-month highs."
    )
    range_pct_180_bow_min = st.slider(
        "BoW SELL reject if range_pct_180 < ",
        min_value=0.10, max_value=0.60, value=0.30, step=0.02,
        help="v2 fix for DOT-style 'already cooked' sells. Don't short coins "
             "that have been falling for so long they're near multi-month lows."
    )
    range_pct_180_re_min = st.slider(
        "RE SELL reject if range_pct_180 < ",
        min_value=0.30, max_value=0.95, value=0.50, step=0.02,
        help="Reversal/Exhaustion needs the coin to have been strong; range "
             "should be in the upper half of the 180-bar window."
    )

    st.markdown("---")
    st.header("🧪 Stage 3 — Hybrid Gating")
    lambda_penalty = st.slider(
        "Soft-gate λ (penalty weight)",
        min_value=0.0, max_value=5.0, value=1.5, step=0.25,
        help="0 = pure score (no consistency penalty). 1.5 = moderate. "
             "≥3 ≈ approaches hard binary behaviour."
    )
    re_top_quantile = st.slider(
        "RE — minimum 180d return percentile (cross-sectional)",
        min_value=0.50, max_value=0.95, value=0.75, step=0.05,
        help="A coin is eligible for RE sell only if its 180d return is in the "
             "top X percentile cross-sectionally. 0.75 = top quartile."
    )

    st.markdown("---")
    st.header("📊 Selection")
    top_n = st.number_input(
        "Top-N per list (Buy / BoW Sell / RE Sell)",
        min_value=3, max_value=30, value=8
    )

    st.markdown("---")
    st.header("🧪 Backtest / Validation")
    bt_min_history_bars = st.number_input(
        "Min history before backtest starts (bars)",
        min_value=180, max_value=500, value=200, step=10,
        help="v2 needs ≥180 bars warmup for the 180-bar return percentile."
    )

    st.markdown("---")
    fetch_btn = st.button("🔄 Fetch / Refresh Data", use_container_width=True)
    recompute_btn = st.button("🔁 Recompute Signals (no fetch)", use_container_width=True)


# =============================================================================
# 3. DATA FETCH (ccxt → KuCoin)
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
for key in [
    "sc_ohlcv", "sc_ticker_to_theme", "sc_results",
    "sc_last_fetch", "sc_timeframe", "sc_validation",
]:
    if key not in st.session_state:
        st.session_state[key] = None


# =============================================================================
# 4. RESPOND TO FETCH
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
    st.session_state["sc_results"] = None
    st.session_state["sc_validation"] = None
    st.success(
        f"Fetched {ohlcv.shape[1] // 5} symbols, "
        f"{ohlcv.shape[0]} bars ({timeframe})."
    )


# =============================================================================
# 5. EARLY EXIT
# =============================================================================
ohlcv = st.session_state["sc_ohlcv"]
ticker_to_theme = st.session_state["sc_ticker_to_theme"]
if ohlcv is None or ohlcv.empty:
    st.info("👈 Click **Fetch / Refresh Data** in the sidebar to begin.")
    st.stop()


# =============================================================================
# 6. CORE HELPERS
# =============================================================================
def cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score: (x − row_mean) / row_std. Skips NaN."""
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0)


def avg_signals(*dfs: pd.DataFrame) -> pd.DataFrame:
    """Element-wise nanmean across DataFrames."""
    arr = np.stack([df.values for df in dfs], axis=0)
    avg = np.nanmean(arr, axis=0)
    return pd.DataFrame(avg, index=dfs[0].index, columns=dfs[0].columns)


def compute_atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame,
                period: int = 14) -> pd.DataFrame:
    """Wilder True Range / ATR. TR = max(H−L, |H−prev_close|, |L−prev_close|)."""
    prev_close = close.shift(1)
    tr1 = (high - low).values
    tr2 = (high - prev_close).abs().values
    tr3 = (low - prev_close).abs().values
    tr_arr = np.maximum.reduce([tr1, tr2, tr3])
    tr = pd.DataFrame(tr_arr, index=high.index, columns=high.columns)
    return tr.rolling(period, min_periods=max(2, period // 2)).mean()


def rolling_r2_logprice(close: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """
    Vectorised R² of log(price) regressed on time index over a rolling window.

    For each window of size N, x = [0, 1, ..., N-1] (constant across windows).
    R² = corr(x, log_p)² since regressing on a single x yields R² = ρ².

    We compute it as:
      ρ = (Σxy − N·x̄·ȳ) / sqrt((Σx² − N·x̄²) · (Σy² − N·ȳ²))

    With x_i = i, Σx, Σx² are constants. Only y = log_price needs rolling sums.

    Trick for Σ(k·y_k) over a sliding window of length N at row t:
      Let pos = absolute row index, S1 = cumsum(pos · y).
      Σ_{j=t−N+1..t} j·y_j = S1[t] − S1[t−N]
      Σ_{k=0..N-1} k·y_{t−N+1+k} = (S1[t] − S1[t−N]) − (t−N+1)·sum_y[t]
    """
    log_p = np.log(close.replace(0, np.nan))

    N = window
    x = np.arange(N, dtype=float)
    sum_x = x.sum()
    sum_x2 = (x * x).sum()
    var_x = sum_x2 - (sum_x ** 2) / N  # = N(N²-1)/12

    sum_y = log_p.rolling(N, min_periods=N).sum()
    sum_y2 = (log_p ** 2).rolling(N, min_periods=N).sum()
    var_y = sum_y2 - (sum_y ** 2) / N

    pos = np.arange(len(log_p), dtype=float)
    pos_y = log_p.mul(pos, axis=0)
    S1 = pos_y.cumsum()
    S1_shift = S1.shift(N)
    j_sum_y = S1 - S1_shift
    start_idx = pos - N + 1
    sum_xy = j_sum_y.sub(sum_y.mul(start_idx, axis=0))

    cov_xy = sum_xy - sum_x * sum_y / N
    denom = np.sqrt(var_x * var_y.clip(lower=1e-12))
    rho = cov_xy.div(denom)
    return (rho ** 2).clip(lower=0.0, upper=1.0)


def rolling_concentration_ratio(log_ret: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """
    max(|log_ret|) / sum(|log_ret|) over a rolling window.

    Bounded in [1/N, 1]. 1/N = uniform; 1 = single-bar dominates entirely.
    A "smooth" trend has concentration ≈ 1/N + ε. A spike has ratio close to 1.
    """
    abs_ret = log_ret.abs()
    rolling_max = abs_ret.rolling(window, min_periods=max(5, window // 2)).max()
    rolling_sum = abs_ret.rolling(window, min_periods=max(5, window // 2)).sum()
    return rolling_max.div(rolling_sum.replace(0, np.nan))


# =============================================================================
# 7. SIGNAL PIPELINE — TIME SERIES FOR EVERY SIGNAL
# =============================================================================
@st.cache_data(show_spinner="Computing signal pipeline…", ttl=600)
def compute_pipeline(
    ohlcv_df: pd.DataFrame,
    ticker_to_theme: Dict[str, str],
    # Stage 0
    min_usd_vol: float,
    jump_thresh: float,        # %
    concen_thresh: float,      # ratio
    wick_thresh: float,        # %
    smooth_r2_min: float,      # in [0, 1]
    # Stage 2
    price_z60_max: float,
    range_pct_90_buy_max: float,
    range_pct_180_buy_max: float,
    range_pct_180_bow_min: float,
    range_pct_180_re_min: float,
    # Stage 3
    lambda_penalty: float,
    re_top_quantile: float,
) -> Dict:
    """Single-pass computation of all stages. Returns a dict of TS DataFrames."""
    # ---- Slice OHLCV ----
    close = ohlcv_df.xs("c", level=1, axis=1).apply(pd.to_numeric, errors="coerce")
    high = ohlcv_df.xs("h", level=1, axis=1).apply(pd.to_numeric, errors="coerce")
    low = ohlcv_df.xs("l", level=1, axis=1).apply(pd.to_numeric, errors="coerce")
    volume = ohlcv_df.xs("v", level=1, axis=1).apply(pd.to_numeric, errors="coerce")

    # Universal log-return panel
    log_ret = np.log(close / close.shift(1))

    # =============================================================
    # STAGE 0 — UNIVERSE FILTERS (time-varying boolean masks)
    # =============================================================
    # Liquidity: median 30-bar USD volume
    usd_vol = close * volume
    median_usd_vol_30 = usd_vol.rolling(30, min_periods=15).median()
    liquidity_mask = median_usd_vol_30 >= min_usd_vol

    # JUMP filter — 30-bar window (v2 fix)
    log_jump_thresh = np.log(1.0 + jump_thresh / 100.0)
    abs_logret_30d_max = log_ret.abs().rolling(30, min_periods=15).max()
    no_close_jump_mask = abs_logret_30d_max <= log_jump_thresh

    # CONCENTRATION RATIO — over 30 bars
    concen = rolling_concentration_ratio(log_ret, window=30)
    no_concen_mask = concen <= concen_thresh

    # WICK jump — kept at 5 bars (catches intraday only — short window OK)
    wick_pct = (high - low) / close.shift(1)
    wick_5d_max = wick_pct.rolling(5, min_periods=3).max()
    no_wick_jump_mask = wick_5d_max <= (wick_thresh / 100.0)

    # SMOOTHNESS — R² of log(price) vs time over 30 bars (v2 fix)
    r2_30d = rolling_r2_logprice(close, window=30)
    smooth_mask = r2_30d >= smooth_r2_min

    # Stage-0 mask WITHOUT smoothness (used by RE which doesn't require smoothness)
    stage0_no_smooth = liquidity_mask & no_close_jump_mask & no_concen_mask & no_wick_jump_mask
    # Stage-0 mask WITH smoothness (used by buy and BoW sell)
    stage0_smooth = stage0_no_smooth & smooth_mask

    # =============================================================
    # STAGE 1 — BUCKET A: THEME-RELATIVE
    # =============================================================
    coin_to_theme = {c: ticker_to_theme.get(c, "UNKNOWN") for c in log_ret.columns}
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
    # =============================================================
    if "BTC" not in log_ret.columns:
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
    # STAGE 1 — BUCKET C: ABSOLUTE
    # =============================================================
    atr_14 = compute_atr(high, low, close, period=14)
    atr_pct = (atr_14 / close).replace(0, np.nan)
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
    )
    trend_count = trend_count.where(sma_100.notna())

    z_C1 = cs_zscore(C1)
    z_C2 = cs_zscore(C2)
    z_C3 = cs_zscore(trend_count)
    bucket_C = avg_signals(z_C1, z_C2, z_C3)

    # =============================================================
    # RE-SPECIFIC SIGNALS — 180d strength + recent weakness
    # =============================================================
    ret_180d = log_ret.rolling(180, min_periods=120).sum()
    ret_30d = log_ret.rolling(30, min_periods=20).sum()
    ret_14d = log_ret.rolling(14, min_periods=10).sum()

    z_ret_180d = cs_zscore(ret_180d)
    z_ret_30d = cs_zscore(ret_30d)
    z_ret_14d = cs_zscore(ret_14d)

    # 180d return percentile (cross-sectional, per-row)
    ret_180d_pct = ret_180d.rank(axis=1, pct=True)

    # =============================================================
    # STAGE 2 — RANGE / EXTENSION FILTERS
    # =============================================================
    sma_60 = close.rolling(60).mean()
    std_60 = close.rolling(60).std().replace(0, np.nan)
    price_z_60 = (close - sma_60) / std_60

    high_90 = close.rolling(90, min_periods=60).max()
    low_90 = close.rolling(90, min_periods=60).min()
    range_pct_90 = (close - low_90) / (high_90 - low_90).replace(0, np.nan)

    high_180 = close.rolling(180, min_periods=120).max()
    low_180 = close.rolling(180, min_periods=120).min()
    range_pct_180 = (close - low_180) / (high_180 - low_180).replace(0, np.nan)

    # ---- BUY extension (long-side, trend-following) ----
    buy_extension_ok = (
        (price_z_60 < price_z60_max)
        & (range_pct_90 < range_pct_90_buy_max)
        & (range_pct_180 < range_pct_180_buy_max)
    )

    # ---- BoW extension (already-cooked check) ----
    bow_extension_ok = (
        (price_z_60 > -price_z60_max)
        & (range_pct_180 > range_pct_180_bow_min)
    )

    # ---- RE extension (must have been strong, not over-extended right now) ----
    re_extension_ok = (
        (price_z_60 > -price_z60_max)
        & (range_pct_180 > range_pct_180_re_min)
    )

    # =============================================================
    # PRE-GATES (Stage 0 + Stage 2, per side)
    # =============================================================
    # BUY: needs smoothness, plus range/extension + jump/concen/liquidity
    buy_pre_gate = stage0_smooth & buy_extension_ok

    # BoW: needs smoothness (we want clean trend within theme), plus BoW extension
    bow_pre_gate = stage0_smooth & bow_extension_ok

    # RE: does NOT require smoothness (we WANT recent breakdown), but needs jump/concen/liquidity
    re_pre_gate = stage0_no_smooth & re_extension_ok

    # =============================================================
    # STAGE 3 — HYBRID GATING
    # =============================================================
    avg_z_buy = avg_signals(bucket_A, bucket_B, bucket_C)

    # ---- BUY (trend-following) ----
    buy_pen = (
        np.maximum(-bucket_A, 0.0) ** 2
        + np.maximum(-bucket_B, 0.0) ** 2
        + np.maximum(-bucket_C, 0.0) ** 2
    )
    hard_buy_qual = (bucket_A > 0) & (bucket_B > 0) & (bucket_C > 0) & buy_pre_gate
    hard_buy_score = avg_z_buy.where(hard_buy_qual)
    soft_buy_score = (avg_z_buy - lambda_penalty * buy_pen).where(buy_pre_gate)

    # ---- BoW (Best-of-Worst) ----
    # Score: rewards (theme-leadership) − (BTC-rel weakness penalty) − (absolute weakness penalty)
    # i.e. bucket_A − bucket_B − bucket_C means high A, low (negative) B and C → high score
    bow_score_raw = bucket_A - bucket_B - bucket_C
    # Penalise wrong-direction movements:
    #   bucket_A should be POSITIVE → penalise if negative
    #   bucket_B should be NEGATIVE → penalise if positive
    #   bucket_C should be NEGATIVE → penalise if positive
    bow_pen = (
        np.maximum(-bucket_A, 0.0) ** 2
        + np.maximum(bucket_B, 0.0) ** 2
        + np.maximum(bucket_C, 0.0) ** 2
    )
    hard_bow_qual = (bucket_A > 0) & (bucket_B < 0) & (bucket_C < 0) & bow_pre_gate
    hard_bow_score = bow_score_raw.where(hard_bow_qual)
    soft_bow_score = (bow_score_raw - lambda_penalty * bow_pen).where(bow_pre_gate)

    # ---- RE (Reversal/Exhaustion) ----
    # Score: high 180d strength × high recent weakness
    # z_ret_180d POSITIVE (was strong) − z_ret_30d (negative now) − z_ret_14d (negative now)
    re_score_raw = z_ret_180d - z_ret_30d - z_ret_14d
    # Top-quartile gate (only consider coins with high enough 180d strength)
    re_strength_gate = ret_180d_pct >= re_top_quantile
    # Penalty form
    re_pen = (
        np.maximum(-z_ret_180d, 0.0) ** 2
        + np.maximum(z_ret_30d, 0.0) ** 2
        + np.maximum(z_ret_14d, 0.0) ** 2
    )
    hard_re_qual = (
        re_strength_gate
        & (ret_14d < 0)
        & (ret_30d < 0)
        & re_pre_gate
    )
    hard_re_score = re_score_raw.where(hard_re_qual)
    # Soft RE: still gate on 180d strength quartile (it's the defining feature) but
    # don't require recent returns to be strictly negative; let the penalty handle it
    soft_re_score = (re_score_raw - lambda_penalty * re_pen).where(re_pre_gate & re_strength_gate)

    # =============================================================
    # PACKAGE
    # =============================================================
    return {
        "close": close, "high": high, "low": low,
        "log_ret": log_ret, "atr_pct": atr_pct,
        # masks (Stage 0)
        "liquidity_mask": liquidity_mask,
        "no_close_jump_mask": no_close_jump_mask,
        "no_concen_mask": no_concen_mask,
        "no_wick_jump_mask": no_wick_jump_mask,
        "smooth_mask": smooth_mask,
        "stage0_smooth": stage0_smooth,
        "stage0_no_smooth": stage0_no_smooth,
        # diagnostic series
        "concen": concen, "r2_30d": r2_30d,
        "price_z_60": price_z_60,
        "range_pct_90": range_pct_90, "range_pct_180": range_pct_180,
        # range gates
        "buy_extension_ok": buy_extension_ok,
        "bow_extension_ok": bow_extension_ok,
        "re_extension_ok": re_extension_ok,
        # bucket TS
        "z_A1": z_A1, "z_A2": z_A2, "z_A3": z_A3, "bucket_A": bucket_A,
        "z_B1": z_B1, "z_B2": z_B2, "z_B3": z_B3, "bucket_B": bucket_B,
        "z_C1": z_C1, "z_C2": z_C2, "z_C3": z_C3, "bucket_C": bucket_C,
        "avg_z_buy": avg_z_buy,
        # RE-specific
        "ret_180d": ret_180d, "ret_30d": ret_30d, "ret_14d": ret_14d,
        "z_ret_180d": z_ret_180d, "z_ret_30d": z_ret_30d, "z_ret_14d": z_ret_14d,
        "ret_180d_pct": ret_180d_pct,
        # gates
        "hard_buy_qual": hard_buy_qual, "hard_buy_score": hard_buy_score,
        "soft_buy_score": soft_buy_score,
        "hard_bow_qual": hard_bow_qual, "hard_bow_score": hard_bow_score,
        "soft_bow_score": soft_bow_score,
        "hard_re_qual": hard_re_qual, "hard_re_score": hard_re_score,
        "soft_re_score": soft_re_score,
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
        concen_thresh=float(concen_thresh),
        wick_thresh=float(wick_thresh_pct),
        smooth_r2_min=float(smooth_r2_min),
        price_z60_max=float(price_z60_max),
        range_pct_90_buy_max=float(range_pct_90_buy_max),
        range_pct_180_buy_max=float(range_pct_180_buy_max),
        range_pct_180_bow_min=float(range_pct_180_bow_min),
        range_pct_180_re_min=float(range_pct_180_re_min),
        lambda_penalty=float(lambda_penalty),
        re_top_quantile=float(re_top_quantile),
    )

R = st.session_state["sc_results"]


# =============================================================================
# 9. LATEST-BAR SIGNAL TABLES
# =============================================================================
last_ts = R["close"].index[-1]
ts_str = str(last_ts)[:16]

st.markdown("---")
st.subheader(f"📅 Latest signal — {ts_str}")

n_total = R["close"].shape[1]
n_liquid = int(R["liquidity_mask"].iloc[-1].sum())
n_smooth = int(R["smooth_mask"].iloc[-1].sum())
n_hard_buy = int(R["hard_buy_qual"].iloc[-1].sum())
n_hard_bow = int(R["hard_bow_qual"].iloc[-1].sum())
n_hard_re = int(R["hard_re_qual"].iloc[-1].sum())

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Coins (total)", n_total)
c2.metric("Liquid", n_liquid, delta=f"−{n_total - n_liquid}", delta_color="off")
c3.metric("Smooth (R²≥thr)", n_smooth)
c4.metric("Hard BUY", n_hard_buy)
c5.metric("Hard BoW Sell", n_hard_bow)
c6.metric("Hard RE Sell", n_hard_re)


def build_buy_table(scores_row: pd.Series, R: dict) -> pd.DataFrame:
    """Per-coin BUY table at last bar."""
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
            "R² (30d)": round(float(R["r2_30d"].iloc[-1].get(coin, np.nan)), 2),
            "Concen ratio": round(float(R["concen"].iloc[-1].get(coin, np.nan)), 2),
            "price_z_60": round(float(R["price_z_60"].iloc[-1].get(coin, np.nan)), 2),
            "range_pct_180": round(float(R["range_pct_180"].iloc[-1].get(coin, np.nan)), 2),
            "ATR_pct": round(float(R["atr_pct"].iloc[-1].get(coin, np.nan)) * 100, 2),
        })
    return pd.DataFrame(rows)


def build_bow_table(scores_row: pd.Series, R: dict) -> pd.DataFrame:
    """Per-coin Best-of-Worst SELL table at last bar."""
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
            "R² (30d)": round(float(R["r2_30d"].iloc[-1].get(coin, np.nan)), 2),
            "price_z_60": round(float(R["price_z_60"].iloc[-1].get(coin, np.nan)), 2),
            "range_pct_180": round(float(R["range_pct_180"].iloc[-1].get(coin, np.nan)), 2),
            "ATR_pct": round(float(R["atr_pct"].iloc[-1].get(coin, np.nan)) * 100, 2),
        })
    return pd.DataFrame(rows)


def build_re_table(scores_row: pd.Series, R: dict) -> pd.DataFrame:
    """Per-coin Reversal/Exhaustion SELL table at last bar."""
    s = scores_row.dropna().sort_values(ascending=False)
    if s.empty:
        return pd.DataFrame()
    rows = []
    for coin in s.index:
        rows.append({
            "Coin": coin,
            "Theme": R["coin_to_theme"].get(coin, "UNKNOWN"),
            "Score": round(float(s[coin]), 2),
            "Ret 180d": round(float(R["ret_180d"].iloc[-1].get(coin, np.nan)), 2),
            "Ret 30d": round(float(R["ret_30d"].iloc[-1].get(coin, np.nan)), 2),
            "Ret 14d": round(float(R["ret_14d"].iloc[-1].get(coin, np.nan)), 2),
            "180d pctile": round(float(R["ret_180d_pct"].iloc[-1].get(coin, np.nan)), 2),
            "price_z_60": round(float(R["price_z_60"].iloc[-1].get(coin, np.nan)), 2),
            "range_pct_180": round(float(R["range_pct_180"].iloc[-1].get(coin, np.nan)), 2),
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
        f"**Hard binary**: all required bucket signs must agree AND all "
        f"extension/jump/smoothness gates must pass. Top-{int(top_n)} per list."
    )
    hard_buy_tbl = build_buy_table(R["hard_buy_score"].iloc[-1], R).head(int(top_n))
    hard_bow_tbl = build_bow_table(R["hard_bow_score"].iloc[-1], R).head(int(top_n))
    hard_re_tbl = build_re_table(R["hard_re_score"].iloc[-1], R).head(int(top_n))

    st.markdown("**🟢 Top BUY signals (trend-following)**")
    if hard_buy_tbl.empty:
        st.info("No qualified buy signals.")
    else:
        st.dataframe(
            hard_buy_tbl.style
            .background_gradient(subset=["Score"], cmap="Greens")
            .background_gradient(subset=["Theme-rel z", "BTC-rel z", "Absolute z"],
                                 cmap="RdYlGn", vmin=-2, vmax=2)
            .background_gradient(subset=["R² (30d)"], cmap="Greens", vmin=0, vmax=1)
            .background_gradient(subset=["Concen ratio"], cmap="RdYlGn_r", vmin=0, vmax=0.5)
            .background_gradient(subset=["price_z_60"], cmap="RdYlGn_r", vmin=-3, vmax=3),
            hide_index=True, use_container_width=True,
            height=min(420, 60 + 35 * len(hard_buy_tbl)),
        )

    cL, cR = st.columns(2)
    with cL:
        st.markdown("**🔴 Top SELL — Best-of-Worst (theme-leaders losing strength)**")
        if hard_bow_tbl.empty:
            st.info("No qualified BoW sell signals.")
        else:
            st.dataframe(
                hard_bow_tbl.style
                .background_gradient(subset=["Score"], cmap="Reds")
                .background_gradient(subset=["Theme-rel z"], cmap="RdYlGn", vmin=-2, vmax=2)
                .background_gradient(subset=["BTC-rel z", "Absolute z"], cmap="RdYlGn", vmin=-2, vmax=2),
                hide_index=True, use_container_width=True,
                height=min(420, 60 + 35 * len(hard_bow_tbl)),
            )
    with cR:
        st.markdown("**🔴 Top SELL — Reversal/Exhaustion (was strong, now breaking)**")
        if hard_re_tbl.empty:
            st.info("No qualified RE sell signals.")
        else:
            st.dataframe(
                hard_re_tbl.style
                .background_gradient(subset=["Score"], cmap="Reds")
                .background_gradient(subset=["Ret 180d"], cmap="Greens", vmin=0, vmax=2)
                .background_gradient(subset=["Ret 14d", "Ret 30d"], cmap="RdYlGn", vmin=-0.5, vmax=0.5),
                hide_index=True, use_container_width=True,
                height=min(420, 60 + 35 * len(hard_re_tbl)),
            )

# ---------- SOFT ----------
with tab_soft:
    st.markdown(
        f"**Soft penalty**: extension/jump/smoothness gates required, but bucket "
        f"agreement is not strictly enforced — wrong-sign buckets are penalised "
        f"via `λ · Σ max(0, wrong-sign-z)²`. λ = **{lambda_penalty}**. "
        f"Top-{int(top_n)} per list."
    )
    soft_buy_tbl = build_buy_table(R["soft_buy_score"].iloc[-1], R).head(int(top_n))
    soft_bow_tbl = build_bow_table(R["soft_bow_score"].iloc[-1], R).head(int(top_n))
    soft_re_tbl = build_re_table(R["soft_re_score"].iloc[-1], R).head(int(top_n))

    st.markdown("**🟢 Top BUY signals (trend-following)**")
    if soft_buy_tbl.empty:
        st.info("No buy signals.")
    else:
        st.dataframe(
            soft_buy_tbl.style
            .background_gradient(subset=["Score"], cmap="Greens")
            .background_gradient(subset=["Theme-rel z", "BTC-rel z", "Absolute z"],
                                 cmap="RdYlGn", vmin=-2, vmax=2)
            .background_gradient(subset=["R² (30d)"], cmap="Greens", vmin=0, vmax=1)
            .background_gradient(subset=["Concen ratio"], cmap="RdYlGn_r", vmin=0, vmax=0.5)
            .background_gradient(subset=["price_z_60"], cmap="RdYlGn_r", vmin=-3, vmax=3),
            hide_index=True, use_container_width=True,
            height=min(420, 60 + 35 * len(soft_buy_tbl)),
        )

    cL, cR = st.columns(2)
    with cL:
        st.markdown("**🔴 Top SELL — Best-of-Worst**")
        if soft_bow_tbl.empty:
            st.info("No BoW sell signals.")
        else:
            st.dataframe(
                soft_bow_tbl.style
                .background_gradient(subset=["Score"], cmap="Reds")
                .background_gradient(subset=["Theme-rel z"], cmap="RdYlGn", vmin=-2, vmax=2)
                .background_gradient(subset=["BTC-rel z", "Absolute z"], cmap="RdYlGn", vmin=-2, vmax=2),
                hide_index=True, use_container_width=True,
                height=min(420, 60 + 35 * len(soft_bow_tbl)),
            )
    with cR:
        st.markdown("**🔴 Top SELL — Reversal/Exhaustion**")
        if soft_re_tbl.empty:
            st.info("No RE sell signals.")
        else:
            st.dataframe(
                soft_re_tbl.style
                .background_gradient(subset=["Score"], cmap="Reds")
                .background_gradient(subset=["Ret 180d"], cmap="Greens", vmin=0, vmax=2)
                .background_gradient(subset=["Ret 14d", "Ret 30d"], cmap="RdYlGn", vmin=-0.5, vmax=0.5),
                hide_index=True, use_container_width=True,
                height=min(420, 60 + 35 * len(soft_re_tbl)),
            )


# ---------- DIAG ----------
with tab_diag:
    st.markdown(
        "**Bucket inspector** — view all bucket scores, RE-specific signals, "
        "and gate diagnostics for any coin. Useful for debugging why a coin is "
        "or isn't in any of the lists."
    )
    sel_coin = st.selectbox(
        "Select coin",
        options=sorted(R["close"].columns.tolist()),
    )
    if sel_coin:
        last_row = {
            "Theme-rel z (14d)":           R["z_A3"].iloc[-1].get(sel_coin, np.nan),
            "Theme-rel z (30d)":           R["z_A1"].iloc[-1].get(sel_coin, np.nan),
            "Theme-rel z (60d)":           R["z_A2"].iloc[-1].get(sel_coin, np.nan),
            "Theme-rel z  (avg, A)":       R["bucket_A"].iloc[-1].get(sel_coin, np.nan),
            "BTC-rel z (14d)":             R["z_B3"].iloc[-1].get(sel_coin, np.nan),
            "BTC-rel z (30d)":             R["z_B1"].iloc[-1].get(sel_coin, np.nan),
            "BTC-rel z (90d)":             R["z_B2"].iloc[-1].get(sel_coin, np.nan),
            "BTC-rel z  (avg, B)":         R["bucket_B"].iloc[-1].get(sel_coin, np.nan),
            "Absolute z (30d ATR-scaled)": R["z_C1"].iloc[-1].get(sel_coin, np.nan),
            "Absolute z (60d ATR-scaled)": R["z_C2"].iloc[-1].get(sel_coin, np.nan),
            "Absolute z (Multi-MA trend)": R["z_C3"].iloc[-1].get(sel_coin, np.nan),
            "Absolute z  (avg, C)":        R["bucket_C"].iloc[-1].get(sel_coin, np.nan),
            "Composite avg z (BUY)":       R["avg_z_buy"].iloc[-1].get(sel_coin, np.nan),
            "—":                           np.nan,
            "Ret 180d":                    R["ret_180d"].iloc[-1].get(sel_coin, np.nan),
            "Ret 30d":                     R["ret_30d"].iloc[-1].get(sel_coin, np.nan),
            "Ret 14d":                     R["ret_14d"].iloc[-1].get(sel_coin, np.nan),
            "180d return pctile":          R["ret_180d_pct"].iloc[-1].get(sel_coin, np.nan),
            "— ":                          np.nan,
            "R² (30d)":                    R["r2_30d"].iloc[-1].get(sel_coin, np.nan),
            "Concen ratio":                R["concen"].iloc[-1].get(sel_coin, np.nan),
            "price_z_60":                  R["price_z_60"].iloc[-1].get(sel_coin, np.nan),
            "range_pct_90":                R["range_pct_90"].iloc[-1].get(sel_coin, np.nan),
            "range_pct_180":               R["range_pct_180"].iloc[-1].get(sel_coin, np.nan),
        }
        df_view = pd.DataFrame.from_dict(last_row, orient="index", columns=["Latest value"])
        df_view["Latest value"] = pd.to_numeric(df_view["Latest value"], errors="coerce").round(2)
        st.dataframe(df_view, use_container_width=False, height=720)

        # Time series chart of buckets
        sub_df = pd.DataFrame({
            "Theme-rel z (A)":  R["bucket_A"][sel_coin],
            "BTC-rel z (B)":    R["bucket_B"][sel_coin],
            "Absolute z (C)":   R["bucket_C"][sel_coin],
            "Composite (BUY)":  R["avg_z_buy"][sel_coin],
        })
        fig_buc = go.Figure()
        for col, color in zip(
            ["Theme-rel z (A)", "BTC-rel z (B)", "Absolute z (C)", "Composite (BUY)"],
            ["#60a5fa", "#a78bfa", "#34d399", "#fbbf24"],
        ):
            fig_buc.add_trace(go.Scatter(
                x=sub_df.index, y=sub_df[col], name=col, mode="lines",
                line=dict(color=color, width=1.5 if col != "Composite (BUY)" else 2.5),
            ))
        fig_buc.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
        fig_buc.update_layout(
            title=f"{sel_coin} — bucket scores over time",
            template="plotly_dark", height=320,
            hovermode="x unified", yaxis_title="CS z-score",
        )
        st.plotly_chart(fig_buc, use_container_width=True)

        # Smoothness + concentration timeline
        sub_diag = pd.DataFrame({
            "R² (30d)": R["r2_30d"][sel_coin],
            "Concen ratio (30d)": R["concen"][sel_coin],
        })
        fig_diag = go.Figure()
        fig_diag.add_trace(go.Scatter(
            x=sub_diag.index, y=sub_diag["R² (30d)"], name="R² (30d)",
            mode="lines", line=dict(color="#22c55e", width=2),
            yaxis="y1",
        ))
        fig_diag.add_trace(go.Scatter(
            x=sub_diag.index, y=sub_diag["Concen ratio (30d)"], name="Concen ratio",
            mode="lines", line=dict(color="#f97316", width=2),
            yaxis="y2",
        ))
        fig_diag.add_hline(y=smooth_r2_min, line_dash="dot", line_color="#22c55e",
                           opacity=0.5, annotation_text="R² threshold")
        fig_diag.update_layout(
            title=f"{sel_coin} — smoothness (R²) vs concentration ratio over time",
            template="plotly_dark", height=320,
            hovermode="x unified",
            yaxis=dict(title="R² (left)", side="left", range=[0, 1]),
            yaxis2=dict(title="Concen ratio (right)", side="right", overlaying="y",
                        range=[0, 1]),
        )
        st.plotly_chart(fig_diag, use_container_width=True)

        # Gate status timeline (last 90 bars)
        last_n = 90
        gate_df = pd.DataFrame({
            "Liquidity":         R["liquidity_mask"][sel_coin].astype(int),
            "No close jump":     R["no_close_jump_mask"][sel_coin].astype(int),
            "No concen":         R["no_concen_mask"][sel_coin].astype(int),
            "No wick":           R["no_wick_jump_mask"][sel_coin].astype(int),
            "Smooth (R²)":       R["smooth_mask"][sel_coin].astype(int),
            "Buy ext OK":        R["buy_extension_ok"][sel_coin].astype(int),
            "BoW ext OK":        R["bow_extension_ok"][sel_coin].astype(int),
            "RE ext OK":         R["re_extension_ok"][sel_coin].astype(int),
        }).iloc[-last_n:]
        st.markdown(f"**Gate status (last {last_n} bars)**")
        st.dataframe(gate_df.tail(15), use_container_width=True, height=280)


# =============================================================================
# 10. VALIDATION — CUMULATIVE RETURN CURVES
# =============================================================================
st.markdown("---")
st.header("🧪 Validation")

st.caption(
    "All metrics use causal signals (no lookahead). At each non-overlapping "
    "rebalance date (every h bars from the start of the backtest window), "
    "coins are sorted into bins by their **weighted score**; the chart shows "
    "the cumulative log-return (%) of the **top bin** and **bottom bin** in "
    "**absolute** terms (solid) and **BTC-excess** terms (dashed). "
    "For BUY/BoW/RE, the top bin should outperform the bottom bin in absolute "
    "*and* BTC-excess space; for SELL strategies the top bin = strongest sell "
    "candidate, so it should *underperform* the bottom bin. The summary's "
    "**Spread (correct dir)** flips sign for sell so positive = working."
)


def compute_forward_returns(close: pd.DataFrame, horizons: List[int]) -> Dict[int, pd.DataFrame]:
    """Forward log-returns at each horizon."""
    return {h: np.log(close.shift(-h) / close) for h in horizons}


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
    close: pd.DataFrame,
    score_dict: Dict[str, pd.DataFrame],
    n_bins_dict: Dict[str, int],
    side_dict: Dict[str, str],
    min_history_bars: int,
) -> Dict:
    """
    Generic validation runner. Takes a dict of {label: score_df}, plus matching
    dicts of bin counts and 'buy'/'sell' side semantics.

    Returns {label: {'series': {h: DataFrame}, 'summary': DataFrame}}.
    """
    horizons = [5, 10, 15, 30]
    fwd_abs = compute_forward_returns(close, horizons)
    fwd_btc = compute_btc_excess_forward_returns(close, horizons)

    base_idx = close.index

    def compute_qcurve(score_df: pd.DataFrame, n_bins: int, side: str):
        series_out = {}
        summary_rows = []

        for h in horizons:
            last_eligible_pos = len(base_idx) - h - 1
            if last_eligible_pos <= min_history_bars:
                series_out[h] = pd.DataFrame()
                summary_rows.append({
                    "Horizon": f"{h}d",
                    "Q-top abs (final %)": np.nan, "Q-bot abs (final %)": np.nan,
                    "Spread abs (correct dir, %)": np.nan,
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
                    continue
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
                df = pd.DataFrame(records).set_index("date").cumsum()
                series_out[h] = df

                final_log = df.iloc[-1]
                qtop_abs_pct = (np.exp(final_log["Qtop_abs"]) - 1.0) * 100
                qbot_abs_pct = (np.exp(final_log["Qbot_abs"]) - 1.0) * 100
                qtop_btc_pct = (np.exp(final_log["Qtop_btc"]) - 1.0) * 100
                qbot_btc_pct = (np.exp(final_log["Qbot_btc"]) - 1.0) * 100

                if side == "buy":
                    spread_abs = qtop_abs_pct - qbot_abs_pct
                    spread_btc = qtop_btc_pct - qbot_btc_pct
                else:  # 'sell' — top bin is strongest sell, expected to underperform
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
                    "Horizon": f"{h}d",
                    "Q-top abs (final %)": np.nan, "Q-bot abs (final %)": np.nan,
                    "Spread abs (correct dir, %)": np.nan,
                    "Q-top BTC-ex (final %)": np.nan, "Q-bot BTC-ex (final %)": np.nan,
                    "Spread BTC-ex (correct dir, %)": np.nan, "n samples": 0,
                })

        return series_out, pd.DataFrame(summary_rows)

    out = {}
    for label, score_df in score_dict.items():
        n_bins = n_bins_dict[label]
        side = side_dict[label]
        series_out, summary_df = compute_qcurve(score_df, n_bins=n_bins, side=side)
        out[label] = {"series": series_out, "summary": summary_df}
    return out


run_bt = st.button("🧪 Run / Refresh Validation", type="primary")
need_validation = run_bt or st.session_state.get("sc_validation") is None
if need_validation:
    score_dict = {
        "hard_buy": R["hard_buy_score"], "soft_buy": R["soft_buy_score"],
        "hard_bow": R["hard_bow_score"], "soft_bow": R["soft_bow_score"],
        "hard_re":  R["hard_re_score"],  "soft_re":  R["soft_re_score"],
    }
    n_bins_dict = {
        "hard_buy": 5, "soft_buy": 10,
        "hard_bow": 5, "soft_bow": 10,
        "hard_re":  5, "soft_re":  10,
    }
    side_dict = {
        "hard_buy": "buy",  "soft_buy": "buy",
        "hard_bow": "sell", "soft_bow": "sell",
        "hard_re":  "sell", "soft_re":  "sell",
    }
    st.session_state["sc_validation"] = run_validation(
        R["close"], score_dict, n_bins_dict, side_dict,
        min_history_bars=int(bt_min_history_bars),
    )

V = st.session_state.get("sc_validation")
if V is None:
    st.info("Click **Run / Refresh Validation** above.")
    st.stop()


# =============================================================================
# 11. RENDERING HELPERS
# =============================================================================
def plot_cumret_chart(
    series_df: pd.DataFrame,
    h: int, n_bins: int, side: str, gate: str, strategy: str,
) -> go.Figure:
    """One holding-period chart with 4 cumulative-return lines."""
    fig = go.Figure()
    if series_df is None or series_df.empty:
        fig.update_layout(
            title=f"{gate} {strategy} {side.upper()} — {h}-bar (no data)",
            template="plotly_dark", height=320,
        )
        return fig

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
        title=f"{gate} {strategy} {title_side} — {h}-bar holding",
        xaxis_title="Rebalance date",
        yaxis_title="Cumulative log-return (%)",
        template="plotly_dark", height=340,
        legend=dict(orientation="h", y=-0.22, font=dict(size=10)),
        hovermode="x unified", margin=dict(t=50, b=70, l=10, r=10),
    )
    return fig


def render_summary_table(summary_df: pd.DataFrame):
    """Render the 2dp summary table."""
    if summary_df is None or summary_df.empty:
        st.info("No summary data.")
        return
    df = summary_df.copy()
    if "n samples" in df.columns:
        df["n samples"] = df["n samples"].astype(int)
    pct_cols = [c for c in df.columns if c.endswith("%)")]
    fmt = {c: "{:+.2f}" for c in pct_cols}
    spread_cols = [c for c in pct_cols if c.startswith("Spread ")]
    styled = df.style.format(fmt)
    if spread_cols:
        styled = styled.background_gradient(
            subset=spread_cols, cmap="RdYlGn", vmin=-15, vmax=15
        )
    st.dataframe(styled, hide_index=True, use_container_width=True)


HORIZONS = [5, 10, 15, 30]


def render_holding_period_grid(
    series_dict: Dict[int, pd.DataFrame],
    n_bins: int, side: str, gate: str, strategy: str,
):
    """2x2 grid of holding-period charts."""
    rows = [HORIZONS[:2], HORIZONS[2:]]
    for row in rows:
        cols = st.columns(2)
        for col, h in zip(cols, row):
            with col:
                st.plotly_chart(
                    plot_cumret_chart(
                        series_dict.get(h, pd.DataFrame()),
                        h, n_bins, side, gate, strategy,
                    ),
                    use_container_width=True,
                )


# =============================================================================
# 12. VALIDATION — TABS BY GATE × STRATEGY
# =============================================================================
mode_tab_hard, mode_tab_soft = st.tabs([
    "🟦 Hard Binary Gate",
    "🟪 Soft Penalty Gate",
])


def _validation_tabs(gate_label: str, gate_results: Dict[str, dict],
                     gate_short: str, n_bins_buy: int, n_bins_sell: int):
    """Render BUY / BoW SELL / RE SELL tabs for a given gating mode."""
    st.markdown(
        f"Bins are computed over the universe defined by the {gate_label} pre-gate "
        f"(extension/jump/smoothness/liquidity). "
        + ("**Quintiles (5 bins).**" if n_bins_buy == 5 else "**Deciles (10 bins).**")
        + " Cumulative log-return curves of top vs bottom bin — solid = absolute, "
        f"dashed = BTC-excess."
    )
    sub_buy, sub_bow, sub_re = st.tabs([
        "🟢 BUY (trend-following)",
        "🔴 SELL — Best-of-Worst",
        "🔴 SELL — Reversal/Exhaustion",
    ])

    with sub_buy:
        render_holding_period_grid(
            gate_results["buy"]["series"], n_bins=n_bins_buy,
            side="buy", gate=gate_short, strategy="BUY",
        )
        st.markdown("**Final cumulative-return summary**")
        render_summary_table(gate_results["buy"]["summary"])

    with sub_bow:
        render_holding_period_grid(
            gate_results["bow"]["series"], n_bins=n_bins_sell,
            side="sell", gate=gate_short, strategy="BoW",
        )
        st.markdown("**Final cumulative-return summary**")
        render_summary_table(gate_results["bow"]["summary"])

    with sub_re:
        render_holding_period_grid(
            gate_results["re"]["series"], n_bins=n_bins_sell,
            side="sell", gate=gate_short, strategy="RE",
        )
        st.markdown("**Final cumulative-return summary**")
        render_summary_table(gate_results["re"]["summary"])


with mode_tab_hard:
    _validation_tabs(
        gate_label="HARD",
        gate_short="Hard",
        gate_results={"buy": V["hard_buy"], "bow": V["hard_bow"], "re": V["hard_re"]},
        n_bins_buy=5, n_bins_sell=5,
    )

with mode_tab_soft:
    _validation_tabs(
        gate_label="SOFT",
        gate_short="Soft",
        gate_results={"buy": V["soft_buy"], "bow": V["soft_bow"], "re": V["soft_re"]},
        n_bins_buy=10, n_bins_sell=10,
    )


# =============================================================================
# 13. FOOTER
# =============================================================================
st.markdown("---")
st.caption(
    f"Last fetch: **{st.session_state.get('sc_last_fetch')}** | "
    f"Timeframe: **{st.session_state.get('sc_timeframe')}** | "
    f"λ: **{lambda_penalty}** | "
    f"R² min: **{smooth_r2_min}** | "
    f"top-N: **{int(top_n)}** | "
    f"Universe: BTC + {len(R['close'].columns)-1} alts"
)
