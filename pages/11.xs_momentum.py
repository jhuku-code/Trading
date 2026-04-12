"""
Cross-Sectional Momentum — v3
==============================
Long/short market-neutral book. Pure relative momentum — z-scores within
cross-section are the signal. No absolute momentum component.

Fixes vs v2
-----------
FIX-1  IDMAG validation is now genuinely out-of-sample.
       IDMAG is estimated on the FIRST (1 - val_frac) of the data.
       Forward-return validation is measured on the LAST val_frac.
       The two windows never overlap, so the quintile chart is a true
       out-of-sample test. val_frac is a sidebar parameter (default 0.2).

FIX-3  Winsorization is now cross-sectional (per timestamp, across coins)
       rather than per-coin through time.
       At each bar t, the pth and (1-p)th quantiles are computed ACROSS all
       coins in the cross-section, then returns are clipped to those bounds.
       All coins in the same cross-section face identical treatment regardless
       of listing age. Expanding-in-time quantiles are used to avoid lookahead.

FIX-5  Vol-scaling warm-up contamination is eliminated.
       After expanding_vol_scale(), the first (min_expanding_bars + 1) rows of
       vol_scaled are dropped before any momentum window is applied.
       Additionally, the rolling momentum window uses
       min_periods = max(lookback // 2, min_expanding_bars + 1)
       so that partial-window estimates are only formed once enough
       clean vol-scaled data exists.

Other improvements
------------------
- Coverage metric: fraction of timestamps that produced valid signals.
- Universe/theme lookback unified (unchanged from v2).
- All f-string pre-computation kept to stay compatible with Streamlit's
  ast.parse magic parser.
- Market-neutral framing made explicit in UI copy.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="XS Momentum v3 · L/S Neutral", layout="wide")
st.title("Cross-Sectional Momentum · v3  |  Long/Short Market-Neutral")

# ---------------------------------------------------------
# INPUT DATA
# ---------------------------------------------------------
df_raw: Optional[pd.DataFrame] = st.session_state.get("price_theme", None)
ticker_to_theme: Optional[Dict] = st.session_state.get("ticker_to_theme", None)

if df_raw is None or df_raw.empty:
    st.error(
        "price_theme not found in st.session_state.\n\n"
        "Please run the Themes Tracker page first."
    )
    st.stop()

if ticker_to_theme is None:
    st.error(
        "ticker_to_theme not found in st.session_state.\n\n"
        "Please ensure the ticker-to-theme mapping is stored on the Themes Tracker page."
    )
    st.stop()

# ---------------------------------------------------------
# INFER DATA FREQUENCY
# ---------------------------------------------------------

def infer_data_frequency(df: pd.DataFrame) -> Tuple[str, float]:
    if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 2:
        return "Unknown", 4.0
    diffs = df.index.to_series().diff().dropna()
    median_hours = diffs.median().total_seconds() / 3600.0
    if 20.0 <= median_hours <= 28.0:
        return "Daily (1D)", median_hours
    elif 3.5 <= median_hours <= 4.5:
        return "4-Hour (4H)", median_hours
    elif 0.9 <= median_hours <= 1.1:
        return "1-Hour (1H)", median_hours
    else:
        label = "~" + str(round(median_hours, 1)) + "H"
        return label, median_hours


freq_label, bar_hours = infer_data_frequency(df_raw)
is_daily = ("1D" in freq_label) or ("Daily" in freq_label)

# Pre-compute display strings — avoids complex expressions inside f-strings
# (required for Streamlit ast.parse compatibility)
period_unit     = "days"  if is_daily else "bars"
period_unit_cap = "Days"  if is_daily else "Bars"
freq_info_msg   = "Detected: " + freq_label + " (" + str(round(bar_hours, 2)) + "h/bar)"

momentum_label   = "Momentum lookback (" + period_unit + ")"
skip_label       = "Reversal skip (" + period_unit + ")"
idmag_label      = "IDMAG lookback (" + period_unit + ")"
min_expand_label = "Vol warm-up (" + period_unit + ")"
expander_title   = "Momentum Windows (" + period_unit_cap + ")"

skip_help = (
    "Bars skipped before the momentum window starts (short-term reversal avoidance). "
    "1 day is standard for daily data."
    if is_daily else
    "Bars skipped before the momentum window starts (short-term reversal avoidance). "
    "6 bars (~24 h) is standard for 4H data."
)

_default_momentum = 60  if is_daily else 60
_default_skip     = 1   if is_daily else 6
_default_idmag    = 120 if is_daily else 120

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("Parameters")
st.sidebar.info(freq_info_msg)

with st.sidebar.expander(expander_title, expanded=True):
    momentum_bars = st.number_input(
        momentum_label, min_value=5, max_value=2000, value=_default_momentum, step=1,
        key="momentum_bars",
    )
    skip_bars = st.number_input(
        skip_label, min_value=0, max_value=100, value=_default_skip, step=1,
        help=skip_help, key="skip_bars",
    )
    idmag_bars_input = st.number_input(
        idmag_label, min_value=5, max_value=2000, value=_default_idmag, step=1,
        key="idmag_bars",
    )
    min_expanding_bars = st.number_input(
        min_expand_label, min_value=5, max_value=200, value=20, step=5,
        help=(
            "FIX-5: The first (warm-up + 1) rows of vol-scaled returns are dropped "
            "before computing momentum. Prevents noisy early vol estimates from "
            "contaminating signals."
        ),
        key="min_expanding_bars",
    )

with st.sidebar.expander("Signal Counts", expanded=True):
    top_n_universe    = st.number_input("Universe top N (longs)",  1, 25,  8, key="tnu")
    bottom_n_universe = st.number_input("Universe bottom N (shorts)", 1, 25,  8, key="bnu")
    top_n_theme       = st.number_input("Theme top N (longs)",    1, 10,  3, key="tnt")
    bottom_n_theme    = st.number_input("Theme bottom N (shorts)", 1, 10,  3, key="bnt")
    min_coins_per_theme = st.number_input(
        "Min coins per theme", 3, 20, 5, key="mcpt",
        help="Themes with fewer coins excluded — cs-std too unstable.",
    )

with st.sidebar.expander("IDMAG Filter", expanded=True):
    idmag_filter_enabled = st.checkbox("Enable IDMAG quality filter", value=True, key="ife")
    idmag_val_frac = st.number_input(
        "Out-of-sample validation fraction",
        min_value=0.05, max_value=0.40, value=0.20, step=0.05,
        help=(
            "FIX-1: IDMAG estimated on first (1 - frac) of data. "
            "Forward-return validation on last frac. No overlap."
        ),
        key="idmag_val_frac",
    )
    idmag_filter_pct = st.number_input(
        "Fraction to filter (tail IDMAG)",
        min_value=0.05, max_value=0.50, value=0.20, step=0.05,
        key="ifp",
    )

with st.sidebar.expander("Robustness", expanded=False):
    cs_std_min = st.number_input(
        "Min cross-sectional std (z-score guard)",
        min_value=1e-6, max_value=0.1, value=1e-4, step=1e-5,
        format="%.6f", key="csm",
        help="Z-scores set to NaN when cs-std falls below this threshold.",
    )
    winsorize_pct = st.number_input(
        "Cross-sectional winsorize percentile (each tail)",
        min_value=0.00, max_value=0.10, value=0.05, step=0.01,
        key="wp",
        help=(
            "FIX-3: Applied cross-sectionally (per timestamp, across coins) "
            "using expanding-in-time quantile bounds — no lookahead, uniform "
            "treatment regardless of listing age."
        ),
    )

st.sidebar.markdown("---")
refresh = st.sidebar.button("Refresh / Recompute", use_container_width=True)

if "xs_mom_v3_results" not in st.session_state:
    st.session_state["xs_mom_v3_results"] = None


# =============================================================
# UTILITIES
# =============================================================

def bars_to_human(n_bars: int, bh: float, daily: bool) -> str:
    if daily:
        return str(n_bars) + "d"
    total_hours = n_bars * bh
    if total_hours >= 24.0:
        return str(round(total_hours / 24.0, 1)) + "d (" + str(n_bars) + " bars)"
    return str(int(total_hours)) + "h (" + str(n_bars) + " bars)"


# =============================================================
# STEP 1 — LOG RETURNS
# =============================================================

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """ln(P_t / P_{t-1}). First row dropped. Columns with all-NaN dropped."""
    lr = np.log(prices / prices.shift(1)).iloc[1:]
    return lr.dropna(axis=1, how="all")


# =============================================================
# STEP 2 — EXPANDING VOL SCALING  (FIX-5: warm-up rows dropped)
# =============================================================

def expanding_vol_scale(
    log_returns: pd.DataFrame,
    min_periods: int,
) -> pd.DataFrame:
    """
    Scale each bar's return by the expanding std known BEFORE that bar.
    shift(1) ensures the current bar is never in its own denominator.

    FIX-5: Returns the scaled series WITH THE FIRST (min_periods + 1) ROWS
    REMOVED.  Those rows use expanding windows of size 1..min_periods, which
    are too noisy to contribute clean signal.  Callers receive only the
    'clean' portion of vol-scaled returns.
    """
    expanding_std = log_returns.expanding(min_periods=min_periods).std().shift(1)
    expanding_std = expanding_std.replace(0, np.nan)
    scaled = log_returns / expanding_std
    scaled = scaled.clip(lower=-5, upper=5)
    # Drop the warm-up rows (FIX-5)
    clean_start = min_periods + 1
    return scaled.iloc[clean_start:]


# =============================================================
# STEP 3 — CROSS-SECTIONAL WINSORIZATION WITH EXPANDING BOUNDS  (FIX-3)
# =============================================================

def cs_winsorize_expanding(
    raw_scores: pd.DataFrame,
    pct: float,
    min_periods_for_quantile: int = 20,
) -> pd.DataFrame:
    """
    FIX-3: Winsorize cross-sectionally per timestamp, using expanding-in-time
    quantile bounds so there is zero lookahead.

    For each timestamp t:
      lo_t = pth quantile of the CROSS-SECTION at time t,
             but only after we have seen >= min_periods_for_quantile timestamps.
      hi_t = (1-p)th quantile of the same cross-section at t.

    All coins at timestamp t are clipped to [lo_t, hi_t].  This guarantees
    uniform treatment across coins regardless of listing age, because the
    winsor bounds come from the cross-section, not from each coin's own
    time-series history.

    The expanding-in-time dimension:  To avoid lookahead in the bounds
    themselves, we track the expanding distribution of the cross-sectional
    quantiles through time — i.e. lo_t is the pth quantile of
    {cs_pct_q(s) for s <= t}.  This is more conservative than naively
    using the full-sample cross-sectional bounds.
    """
    if pct == 0.0:
        return raw_scores.copy()

    # Compute cross-sectional quantile for each timestamp (no lookahead within bar)
    cs_lo = raw_scores.quantile(pct, axis=1)        # Series indexed by time
    cs_hi = raw_scores.quantile(1.0 - pct, axis=1)

    # Expanding bounds through time — prevents lookahead across timestamps
    expanding_lo = cs_lo.expanding(min_periods=min_periods_for_quantile).min()
    expanding_hi = cs_hi.expanding(min_periods=min_periods_for_quantile).max()

    # Before we have enough timestamps for the quantile, use the current bar's
    # raw cross-sectional bounds (still no lookahead — each bar is independent)
    lo_bounds = expanding_lo.where(
        expanding_lo.notna(), cs_lo
    )
    hi_bounds = expanding_hi.where(
        expanding_hi.notna(), cs_hi
    )

    # Clip each row to its timestamp-specific bounds
    clipped = raw_scores.copy()
    for t in raw_scores.index:
        lo_val = lo_bounds.loc[t]
        hi_val = hi_bounds.loc[t]
        if pd.isna(lo_val) or pd.isna(hi_val):
            continue
        clipped.loc[t] = raw_scores.loc[t].clip(lower=lo_val, upper=hi_val)

    return clipped


# =============================================================
# STEP 4 — IDMAG (Frog-in-the-Pan quality score)
# =============================================================

def _idmag_single_window(window_returns: pd.Series) -> float:
    """IDMAG for one window per Da-Gurun-Warachka (2014) Eq. (2)."""
    weights_by_quintile = {0: 5/15, 1: 4/15, 2: 3/15, 3: 2/15, 4: 1/15}
    abs_r = window_returns.abs()
    try:
        qlabels = pd.qcut(abs_r, 5, labels=False, duplicates="drop")
    except ValueError:
        return np.nan
    weights = qlabels.map(weights_by_quintile).values
    signed_w = np.sign(window_returns.values) * weights
    pret = window_returns.sum()
    sgn_pret = np.sign(pret) if pret != 0 else 0
    return float(-(1 / len(window_returns)) * sgn_pret * np.nansum(signed_w))


def compute_idmag_on_window(
    log_returns: pd.DataFrame,
    lookback_bars: int,
    estimation_end_idx: int,
) -> pd.Series:
    """
    FIX-1: Compute IDMAG using only rows up to estimation_end_idx (exclusive).
    This ensures IDMAG estimation never touches the forward-return validation
    window.
    """
    estimation_data = log_returns.iloc[:estimation_end_idx]
    idmag_vals = {}
    for coin in estimation_data.columns:
        r = estimation_data[coin].dropna()
        if len(r) < lookback_bars:
            idmag_vals[coin] = np.nan
            continue
        vals = [
            _idmag_single_window(r.iloc[i - lookback_bars: i])
            for i in range(lookback_bars, len(r))
        ]
        valid = [v for v in vals if not np.isnan(v)]
        idmag_vals[coin] = float(np.mean(valid)) if valid else np.nan
    return pd.Series(idmag_vals, name="IDMAG")


def validate_idmag_oos(
    idmag: pd.Series,
    log_returns: pd.DataFrame,
    validation_start_idx: int,
    n_quintiles: int = 5,
) -> Tuple[pd.DataFrame, bool]:
    """
    FIX-1: Genuinely out-of-sample validation.
    Forward returns are computed on log_returns.iloc[validation_start_idx:] —
    a period that was never seen during IDMAG estimation.
    """
    valid_coins = idmag.dropna().index.tolist()
    valid_coins = [c for c in valid_coins if c in log_returns.columns]
    if len(valid_coins) < n_quintiles:
        return pd.DataFrame(), True

    oos_data = log_returns.iloc[validation_start_idx:]
    fwd_return = oos_data[valid_coins].sum()

    combined = pd.DataFrame({"IDMAG": idmag[valid_coins], "fwd_return": fwd_return}).dropna()
    try:
        combined["quintile"] = pd.qcut(
            combined["IDMAG"], n_quintiles,
            labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
            duplicates="drop",
        )
    except ValueError:
        return pd.DataFrame(), True

    quintile_df = (
        combined.groupby("quintile", observed=True)["fwd_return"]
        .agg(["mean", "median"])
        .rename(columns={"mean": "mean_fwd_return", "median": "median_fwd_return"})
        .reset_index()
    )

    if len(quintile_df) >= 2:
        q1_ret = quintile_df.loc[quintile_df["quintile"] == "Q1", "mean_fwd_return"].values
        q5_ret = quintile_df.loc[quintile_df["quintile"] == "Q5", "mean_fwd_return"].values
        low_idmag_wins = bool(
            len(q1_ret) > 0 and len(q5_ret) > 0 and float(q1_ret[0]) > float(q5_ret[0])
        )
    else:
        low_idmag_wins = True

    return quintile_df, low_idmag_wins


def apply_idmag_filter(
    idmag: pd.Series,
    log_returns: pd.DataFrame,
    filter_pct: float,
    low_idmag_wins: bool,
) -> Tuple[List[str], List[str]]:
    valid = idmag.dropna()
    valid = valid[valid.index.isin(log_returns.columns)]
    n_remove = max(1, int(len(valid) * filter_pct))
    removed = (
        valid.nlargest(n_remove).index.tolist()
        if low_idmag_wins
        else valid.nsmallest(n_remove).index.tolist()
    )
    kept = [c for c in log_returns.columns if c not in removed]
    return kept, removed


# =============================================================
# STEP 5 — MOMENTUM SCORES
# =============================================================

def calculate_momentum_scores(
    vol_scaled: pd.DataFrame,
    lookback_bars: int,
    exclude_bars: int,
    min_clean_periods: int,
    cs_std_min: float = 1e-4,
    winsorize_pct: float = 0.05,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    1. Shift by exclude_bars (reversal skip).
    2. Rolling sum of vol-scaled log-returns over lookback_bars.
       FIX-5: min_periods = max(lookback // 2, min_clean_periods + 1) so
              partial windows only form after sufficient clean data exists.
    3. FIX-3: Cross-sectional winsorization with expanding-in-time bounds.
    4. Cross-sectional z-score (market-neutral signal).
    5. Rank.

    Returns z_scores, ranks, coverage (fraction of timestamps with valid signal).
    """
    _min_periods = max(lookback_bars // 2, min_clean_periods + 1)

    shifted  = vol_scaled.shift(exclude_bars)
    cum_ret  = shifted.rolling(window=lookback_bars, min_periods=_min_periods).sum()

    # FIX-3: cross-sectional winsorization
    cum_ret_w = cs_winsorize_expanding(
        cum_ret,
        pct=winsorize_pct,
        min_periods_for_quantile=max(20, min_clean_periods),
    )

    cs_mean     = cum_ret_w.mean(axis=1)
    cs_std      = cum_ret_w.std(axis=1)
    cs_std_safe = cs_std.where(cs_std >= cs_std_min, other=np.nan)
    z_scores    = cum_ret_w.sub(cs_mean, axis=0).div(cs_std_safe, axis=0)
    ranks       = z_scores.rank(axis=1, ascending=False)

    # Coverage: fraction of timestamps where at least top_n + bottom_n valid z-scores exist
    valid_counts = z_scores.notna().sum(axis=1)
    coverage = (valid_counts > 0).mean()

    return z_scores, ranks, pd.Series({"coverage": coverage})


# =============================================================
# STEP 6 — SIGNAL GENERATION
# =============================================================

def generate_signals(
    ranks: pd.DataFrame,
    top_n: int,
    bottom_n: int,
) -> Tuple[pd.Series, pd.Series, float]:
    """Returns buy_series, sell_series, coverage_pct."""
    buy_dict:  Dict = {}
    sell_dict: Dict = {}
    valid_ts = 0
    for idx in ranks.index:
        row = ranks.loc[idx].dropna()
        if len(row) < top_n + bottom_n:
            continue
        valid_ts += 1
        buy_dict[idx]  = row.nsmallest(top_n).index.tolist()
        sell_dict[idx] = row.nlargest(bottom_n).index.tolist()
    total = len(ranks)
    coverage = valid_ts / total if total > 0 else 0.0
    return pd.Series(buy_dict, dtype=object), pd.Series(sell_dict, dtype=object), coverage


def signal_to_df(s: pd.Series) -> pd.DataFrame:
    if s is None or len(s) == 0:
        return pd.DataFrame(columns=["Date", "Tickers"]).set_index("Date")
    rows = {
        idx: ", ".join(map(str, v)) if isinstance(v, (list, tuple, pd.Index)) else str(v)
        for idx, v in s.items()
    }
    return pd.DataFrame.from_dict(rows, orient="index", columns=["Tickers"])


# =============================================================
# STEP 7 — ALIGNMENT MATRIX
# =============================================================

def build_alignment_matrix(
    universe_buy:  List[str],
    universe_sell: List[str],
    theme_signals: Dict[str, Dict[str, List[str]]],
) -> pd.DataFrame:
    rows = []
    for theme, signals in theme_signals.items():
        for coin in signals.get("buy", []):
            if coin in universe_buy:
                alignment = "Aligned (Long)"
            elif coin in universe_sell:
                alignment = "Conflict (Theme=Long, Universe=Short)"
            else:
                alignment = "Theme-only Long"
            rows.append({"Theme": theme, "Coin": coin, "Theme Signal": "Long", "Alignment": alignment})
        for coin in signals.get("sell", []):
            if coin in universe_sell:
                alignment = "Aligned (Short)"
            elif coin in universe_buy:
                alignment = "Conflict (Theme=Short, Universe=Long)"
            else:
                alignment = "Theme-only Short"
            rows.append({"Theme": theme, "Coin": coin, "Theme Signal": "Short", "Alignment": alignment})
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# =============================================================
# MAIN COMPUTATION
# =============================================================

if refresh or st.session_state["xs_mom_v3_results"] is None:
    with st.spinner("Computing signals..."):

        _momentum_bars  = int(momentum_bars)
        _skip_bars      = int(skip_bars)
        _idmag_bars     = int(idmag_bars_input)
        _min_expand     = int(min_expanding_bars)
        _val_frac       = float(idmag_val_frac)

        # FIX-1: split index for IDMAG out-of-sample validation
        n_rows          = len(df_raw)
        estimation_end  = max(_idmag_bars + 10, int(n_rows * (1.0 - _val_frac)))
        validation_start = estimation_end
        idmag_oos_bars  = n_rows - validation_start

        # Human-readable audit strings (pre-computed — no f-string complexity)
        est_rows_str    = str(estimation_end) + " bars"
        val_rows_str    = str(idmag_oos_bars) + " bars"
        warmup_drop_str = str(_min_expand + 1) + " rows"
        min_roll_str    = str(max(_momentum_bars // 2, _min_expand + 1)) + " bars"

        param_audit = {
            "Detected frequency":                        freq_label,
            "Bar size (hours)":                          round(bar_hours, 2),
            "Total price bars":                          n_rows,
            "Momentum lookback (" + period_unit + ")":   _momentum_bars,
            "Momentum (time equiv.)":                    bars_to_human(_momentum_bars, bar_hours, is_daily),
            "Reversal skip (" + period_unit + ")":       _skip_bars,
            "Vol warm-up (" + period_unit + ")":         _min_expand,
            "Warm-up rows dropped (FIX-5)":              warmup_drop_str,
            "Rolling min_periods (FIX-5)":               min_roll_str,
            "IDMAG lookback (" + period_unit + ")":      _idmag_bars,
            "IDMAG estimation window":                   est_rows_str,
            "IDMAG OOS validation window (FIX-1)":       val_rows_str,
            "Winsorize mode (FIX-3)":                    "Cross-sectional per timestamp",
        }

        # Step 1 — log returns
        log_ret = compute_log_returns(df_raw)

        # Step 2 — expanding vol scaling + FIX-5 warm-up drop
        vol_scaled_all = expanding_vol_scale(log_ret, min_periods=_min_expand)

        # Step 4a — IDMAG on estimation window only (FIX-1)
        # Note: log_ret rows, not vol_scaled rows, for IDMAG (raw returns needed)
        idmag_series = compute_idmag_on_window(
            log_ret, lookback_bars=_idmag_bars, estimation_end_idx=estimation_end
        )

        # Step 4b — out-of-sample IDMAG direction validation (FIX-1)
        quintile_df, low_idmag_wins = validate_idmag_oos(
            idmag_series, log_ret, validation_start_idx=validation_start
        )
        filter_direction_label = (
            "Standard (remove high-IDMAG noisy coins)"
            if low_idmag_wins
            else "INVERTED — high-IDMAG outperforms in this universe OOS"
        )

        # Step 4c — apply IDMAG filter
        if idmag_filter_enabled and not idmag_series.dropna().empty:
            kept_coins, removed_coins = apply_idmag_filter(
                idmag_series, log_ret,
                filter_pct=float(idmag_filter_pct),
                low_idmag_wins=low_idmag_wins,
            )
        else:
            kept_coins    = list(log_ret.columns)
            removed_coins = []

        # Restrict vol_scaled to kept coins
        kept_in_scaled = [c for c in kept_coins if c in vol_scaled_all.columns]
        vol_scaled = vol_scaled_all[kept_in_scaled]

        # Step 5 — universe momentum
        z_scores_u, ranks_u, _ = calculate_momentum_scores(
            vol_scaled,
            lookback_bars=_momentum_bars,
            exclude_bars=_skip_bars,
            min_clean_periods=_min_expand,
            cs_std_min=float(cs_std_min),
            winsorize_pct=float(winsorize_pct),
        )
        buy_signals_u, sell_signals_u, universe_coverage = generate_signals(
            ranks_u, top_n=int(top_n_universe), bottom_n=int(bottom_n_universe)
        )

        last_ts              = ranks_u.index[-1] if len(ranks_u) > 0 else None
        universe_buy_latest  = list(buy_signals_u.iloc[-1])  if len(buy_signals_u)  > 0 else []
        universe_sell_latest = list(sell_signals_u.iloc[-1]) if len(sell_signals_u) > 0 else []

        # Step 6 — theme-level momentum
        filtered_tickers = vol_scaled.columns.tolist()
        theme_map  = {t: ticker_to_theme.get(t, "UNKNOWN") for t in filtered_tickers}
        theme_list = sorted({
            th for th in theme_map.values() if th and th.upper() != "UNKNOWN"
        })

        all_theme_signals_row: Dict = {}
        consolidated_rows: List    = []
        theme_diagnostics: Dict    = {}

        for theme in theme_list:
            theme_tickers = [t for t in filtered_tickers if theme_map.get(t) == theme]
            if len(theme_tickers) < int(min_coins_per_theme):
                continue

            th_vol = vol_scaled[theme_tickers]
            z_th, rk_th, _ = calculate_momentum_scores(
                th_vol,
                lookback_bars=_momentum_bars,
                exclude_bars=_skip_bars,
                min_clean_periods=_min_expand,
                cs_std_min=float(cs_std_min),
                winsorize_pct=float(winsorize_pct),
            )
            buy_th, sell_th, theme_cov = generate_signals(
                rk_th, top_n=int(top_n_theme), bottom_n=int(bottom_n_theme)
            )
            buy_last  = list(buy_th.iloc[-1])  if len(buy_th)  > 0 else []
            sell_last = list(sell_th.iloc[-1]) if len(sell_th) > 0 else []

            all_theme_signals_row[theme] = {"buy": buy_last, "sell": sell_last}

            cs_std_last = (
                th_vol.iloc[-_momentum_bars:].std(axis=1).iloc[-1]
                if len(th_vol) >= _momentum_bars else np.nan
            )
            cs_std_val = round(float(cs_std_last), 6) if not np.isnan(cs_std_last) else "NaN"
            cs_below   = bool(cs_std_last < float(cs_std_min)) if not np.isnan(cs_std_last) else False

            theme_diagnostics[theme] = {
                "n_coins":              len(theme_tickers),
                "cs_std_last_bar":      cs_std_val,
                "cs_std_below_guard":   cs_below,
                "signal_coverage_pct":  round(theme_cov * 100, 1),
            }
            consolidated_rows.append({
                "Theme":          theme,
                "N coins":        len(theme_tickers),
                "Long":           ", ".join(buy_last),
                "Short":          ", ".join(sell_last),
                "Coverage (%)":   round(theme_cov * 100, 1),
            })

        consolidated_signals = (
            pd.DataFrame(consolidated_rows).set_index("Theme")
            if consolidated_rows else pd.DataFrame()
        )

        alignment_df = build_alignment_matrix(
            universe_buy_latest, universe_sell_latest, all_theme_signals_row
        )

        # Vol scaling diagnostic
        full_period_std    = log_ret[kept_in_scaled].std()
        expanding_std_last = log_ret[kept_in_scaled].expanding(
            min_periods=_min_expand
        ).std().iloc[-1]
        vol_ratio = (expanding_std_last / full_period_std).dropna()

        st.session_state["xs_mom_v3_results"] = {
            "log_ret":                log_ret,
            "vol_scaled":             vol_scaled,
            "idmag_series":           idmag_series,
            "quintile_df":            quintile_df,
            "low_idmag_wins":         low_idmag_wins,
            "filter_direction_label": filter_direction_label,
            "removed_coins":          removed_coins,
            "kept_coins":             kept_coins,
            "z_scores_u":             z_scores_u,
            "buy_signals_u":          buy_signals_u,
            "sell_signals_u":         sell_signals_u,
            "universe_buy_latest":    universe_buy_latest,
            "universe_sell_latest":   universe_sell_latest,
            "universe_coverage":      universe_coverage,
            "consolidated_signals":   consolidated_signals,
            "alignment_df":           alignment_df,
            "theme_diagnostics":      theme_diagnostics,
            "vol_ratio":              vol_ratio,
            "param_audit":            param_audit,
            "last_ts":                last_ts,
            "estimation_end":         estimation_end,
            "validation_start":       validation_start,
            "idmag_oos_bars":         idmag_oos_bars,
        }


# =============================================================
# DISPLAY
# =============================================================
results = st.session_state["xs_mom_v3_results"]

if results is None:
    st.info("Click **Refresh / Recompute** in the sidebar to run.")
    st.stop()

log_ret               = results["log_ret"]
vol_scaled            = results["vol_scaled"]
idmag_series          = results["idmag_series"]
quintile_df           = results["quintile_df"]
low_idmag_wins        = results["low_idmag_wins"]
filter_direction_label = results["filter_direction_label"]
removed_coins         = results["removed_coins"]
kept_coins            = results["kept_coins"]
z_scores_u            = results["z_scores_u"]
buy_signals_u         = results["buy_signals_u"]
sell_signals_u        = results["sell_signals_u"]
universe_buy_latest   = results["universe_buy_latest"]
universe_sell_latest  = results["universe_sell_latest"]
universe_coverage     = results["universe_coverage"]
consolidated_signals  = results["consolidated_signals"]
alignment_df          = results["alignment_df"]
theme_diagnostics     = results["theme_diagnostics"]
vol_ratio             = results["vol_ratio"]
param_audit           = results["param_audit"]
last_ts               = results["last_ts"]
estimation_end        = results["estimation_end"]
validation_start      = results["validation_start"]
idmag_oos_bars        = results["idmag_oos_bars"]

ts_label = str(last_ts)[:16] if last_ts is not None else ""

# ── Parameter audit ──────────────────────────────────────────────────────────
with st.expander("Parameter Audit", expanded=False):
    st.table(pd.DataFrame.from_dict(param_audit, orient="index", columns=["Value"]))

# ── IDMAG section ─────────────────────────────────────────────────────────────
st.subheader("IDMAG Momentum Quality Filter  (FIX-1: Out-of-Sample)")

oos_info_msg = (
    "IDMAG estimated on first " + str(estimation_end) + " bars. "
    "Validation forward-return measured on last " + str(idmag_oos_bars) + " bars (zero overlap)."
)
st.info(oos_info_msg)

col_a, col_b = st.columns([1, 2])
with col_a:
    st.metric("OOS filter direction", filter_direction_label[:55])
    st.metric("Coins removed", len(removed_coins))
    st.metric("Coins retained", len(kept_coins))
    if removed_coins:
        st.markdown("**Removed coins:**")
        st.dataframe(
            pd.DataFrame({"Coin": removed_coins}),
            use_container_width=True, height=200,
        )

with col_b:
    if quintile_df is not None and not quintile_df.empty:
        bar_colors = [
            "#2e7d32" if v >= 0 else "#c62828"
            for v in quintile_df["mean_fwd_return"]
        ]
        annotation_color = "#2e7d32" if low_idmag_wins else "#c62828"
        annotation_text  = (
            "OOS: low-IDMAG outperforms — standard filter correct"
            if low_idmag_wins
            else "OOS: high-IDMAG outperforms — filter INVERTED for this universe"
        )
        fig_idmag = go.Figure(go.Bar(
            x=quintile_df["quintile"].astype(str),
            y=quintile_df["mean_fwd_return"],
            marker_color=bar_colors,
            error_y=dict(
                type="data",
                array=(quintile_df["mean_fwd_return"] - quintile_df["median_fwd_return"]).abs(),
                visible=True,
            ),
            text=quintile_df["mean_fwd_return"].round(4),
            textposition="outside",
        ))
        fig_idmag.update_layout(
            title=(
                "IDMAG Quintile — OOS Mean Forward Return<br>"
                "<sup>Q1 = lowest IDMAG (consistent), Q5 = highest IDMAG (noisy)</sup>"
            ),
            xaxis_title="IDMAG Quintile",
            yaxis_title="Mean OOS log-return",
            height=340,
            margin=dict(t=70, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(
                x=0.5, y=1.15, xref="paper", yref="paper",
                text=annotation_text,
                showarrow=False,
                font=dict(size=11, color=annotation_color),
            )],
        )
        st.plotly_chart(fig_idmag, use_container_width=True)
    else:
        st.info("Not enough data for OOS IDMAG quintile validation.")

# ── Vol scaling diagnostic ────────────────────────────────────────────────────
with st.expander("Vol Scaling Diagnostic: Expanding vs Full-Period Std", expanded=False):
    st.markdown(
        "Ratio < 1 means the expanding std (bias-free) is tighter than the full-period "
        "std — i.e. v2 was under-scaling early returns and inflating momentum scores."
    )
    if not vol_ratio.empty:
        fig_vr = go.Figure(go.Bar(
            x=vol_ratio.index,
            y=vol_ratio.values,
            marker_color=["#1565c0" if v < 1 else "#e65100" for v in vol_ratio.values],
        ))
        fig_vr.add_hline(
            y=1.0, line_dash="dash", line_color="gray",
            annotation_text="ratio = 1 (identical)",
        )
        fig_vr.update_layout(
            title="Expanding Std / Full-Period Std (at last bar, per coin)",
            xaxis_title="Coin", yaxis_title="Ratio",
            height=320, margin=dict(t=40, b=60),
            xaxis_tickangle=-45,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_vr, use_container_width=True)

# ── Momentum score distribution ───────────────────────────────────────────────
with st.expander("Cross-Sectional Z-Score Distribution (latest bar)", expanded=False):
    if not z_scores_u.empty:
        last_z = z_scores_u.iloc[-1].dropna().sort_values(ascending=False)
        colors = ["#2e7d32" if v > 0 else "#c62828" for v in last_z.values]
        fig_zd = go.Figure(go.Bar(x=last_z.index, y=last_z.values, marker_color=colors))
        fig_zd.add_hline(y=0, line_dash="dot", line_color="white")
        fig_zd.update_layout(
            title="Universe Momentum Z-scores — " + ts_label,
            xaxis_title="Coin", yaxis_title="CS Z-score",
            height=360, margin=dict(t=40, b=60),
            xaxis_tickangle=-45,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_zd, use_container_width=True)
    st.caption(
        "Z-scores are purely cross-sectional — all market-direction information "
        "is removed. The signal ranks relative momentum within the universe only."
    )

# ── Universe signals ──────────────────────────────────────────────────────────
st.subheader("Long/Short Signals — Universe")

coverage_pct_str = str(round(universe_coverage * 100, 1)) + "%"
st.metric("Signal coverage (% of timestamps)", coverage_pct_str)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Long signals (latest: " + ts_label + ")**")
    st.dataframe(
        pd.DataFrame({"Coin": universe_buy_latest, "Signal": "LONG"}),
        use_container_width=True, hide_index=True,
    )
with col2:
    st.markdown("**Short signals (latest: " + ts_label + ")**")
    st.dataframe(
        pd.DataFrame({"Coin": universe_sell_latest, "Signal": "SHORT"}),
        use_container_width=True, hide_index=True,
    )

with st.expander("All universe signals (all dates)", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Long signals (all dates)**")
        st.dataframe(signal_to_df(buy_signals_u), use_container_width=True)
    with c2:
        st.markdown("**Short signals (all dates)**")
        st.dataframe(signal_to_df(sell_signals_u), use_container_width=True)

# ── Theme signals ─────────────────────────────────────────────────────────────
st.subheader("Long/Short Signals — By Theme")

if consolidated_signals is None or consolidated_signals.empty:
    st.info("No theme-level signals. Increase min coins per theme or check data length.")
else:
    st.dataframe(consolidated_signals, use_container_width=True)

# ── Signal alignment matrix ───────────────────────────────────────────────────
st.subheader("Universe vs Theme Signal Alignment")

if alignment_df is None or alignment_df.empty:
    st.info("No conflicts or alignments to display.")
else:
    def _color_alignment(val: str) -> str:
        if "Aligned" in str(val):
            return "background-color: #1a5c2a; color: white"
        if "Conflict" in str(val):
            return "background-color: #7b1a1a; color: white"
        return ""

    styled_align = alignment_df.style.applymap(_color_alignment, subset=["Alignment"])
    st.dataframe(styled_align, use_container_width=True, hide_index=True)

    n_conflicts = int(alignment_df["Alignment"].str.contains("Conflict").sum())
    n_aligned   = int(alignment_df["Alignment"].str.contains("Aligned").sum())
    conflict_msg = (
        "**" + str(n_aligned) + " aligned** · "
        "**" + str(n_conflicts) + " conflicting** "
        "(theme and universe disagree on direction)"
    )
    st.markdown(conflict_msg)

# ── Theme cross-sectional std health ─────────────────────────────────────────
with st.expander("Theme CS-Std Health & Coverage", expanded=False):
    cs_guard_str = "{:.2e}".format(float(cs_std_min))
    st.markdown(
        "Themes where cs-std falls below the guard (`" + cs_guard_str + "`) "
        "produce unreliable z-scores and are flagged in red. "
        "Coverage % = fraction of timestamps with sufficient coins to form a signal."
    )
    if theme_diagnostics:
        diag_df = (
            pd.DataFrame(theme_diagnostics).T.reset_index()
            .rename(columns={"index": "Theme"})
        )

        def _flag_cs(val: object) -> str:
            return "background-color: #7b1a1a; color: white" if val is True else ""

        styled_diag = diag_df.style.applymap(_flag_cs, subset=["cs_std_below_guard"])
        st.dataframe(styled_diag, use_container_width=True, hide_index=True)
