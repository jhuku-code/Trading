"""
Cross-Sectional Momentum — v2
==============================
Fixes applied vs v1
---------------------
P1  Vol scaling uses expanding std (no lookahead bias).
P2  Log returns used throughout; momentum = rolling sum of log returns
    = log of compounded return (geometrically correct).
P3  IDMAG filter direction validated empirically: coins are binned into
    quintiles and forward returns are compared. If low-IDMAG does NOT
    outperform high-IDMAG in this universe, the filter is inverted or
    disabled, and a diagnostic chart is shown.
P5  exclude_last is expressed in bars, derived from data frequency label.
P6  Cross-sectional std guard: z-scores are set to NaN when cs-std is below
    a minimum threshold to avoid division-by-near-zero in small theme baskets.
P7  Lookback windows unified: one sidebar parameter drives both universe and
    theme calculations. Universe and theme signals are displayed side-by-side
    with explicit conflict detection.

Diagnostics added
------------------
- IDMAG quintile forward-return chart (validates filter direction)
- Vol scaling comparison: expanding vs full-period std
- Momentum score distribution per timestamp
- Universe vs theme signal alignment matrix
- Parameter audit table showing effective bar counts for each window
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="XS Momentum v2", layout="wide")
st.title("Cross-Sectional Momentum · v2")

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
    """
    Infer data frequency from the DataFrame index.
    Returns (label, bar_hours) where label is a human-readable string
    like '1D' or '4H', and bar_hours is the median bar size in hours.
    """
    if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 2:
        return "Unknown", 4.0
    diffs = df.index.to_series().diff().dropna()
    median_hours = diffs.median().total_seconds() / 3600.0

    if 20 <= median_hours <= 28:
        return "Daily (1D)", median_hours
    elif 3.5 <= median_hours <= 4.5:
        return "4-Hour (4H)", median_hours
    elif 0.9 <= median_hours <= 1.1:
        return "1-Hour (1H)", median_hours
    else:
        return f"~{median_hours:.1f}H", median_hours

freq_label, bar_hours = infer_data_frequency(df_raw)
is_daily = "1D" in freq_label or "Daily" in freq_label

# Human-readable period label for sidebar
period_unit = "days" if is_daily else "bars"
period_unit_cap = period_unit.capitalize()

# ---------------------------------------------------------
# SIDEBAR PARAMETERS
# ---------------------------------------------------------
st.sidebar.header("Parameters")

st.sidebar.info(f"📅 Detected frequency: **{freq_label}** ({bar_hours:.2f}h/bar)")

# Default lookback values — in bars, scaled to be economically meaningful
# For daily: 60 days ≈ 3 months. For 4H: 60 bars ≈ 10 days.
# We keep the same bar counts but display them with the right unit.
_default_momentum = 60 if is_daily else 60
_default_skip     = 1  if is_daily else 6
_default_idmag    = 120 if is_daily else 120

with st.sidebar.expander(f"Momentum Windows ({period_unit_cap})", expanded=True):
    momentum_bars = st.number_input(
        f"Momentum lookback ({period_unit})",
        min_value=5, max_value=2000, value=_default_momentum, step=1,
        help=f"Same lookback applied at universe AND theme level. "
             f"1 {period_unit[:-1]} = {bar_hours:.2f}h of data.",
        key="momentum_bars",
    )
    skip_bars = st.number_input(
        f"Short-term reversal skip ({period_unit})",
        min_value=0, max_value=100, value=_default_skip, step=1,
        help=(
            f"Bars shifted before computing cumulative return. "
            f"Avoids the short-term reversal effect. "
            f"{'1 day' if is_daily else '6 bars (24h)'} is a reasonable default."
        ),
        key="skip_bars",
    )
    idmag_bars = st.number_input(
        f"IDMAG lookback ({period_unit})",
        min_value=5, max_value=2000, value=_default_idmag, step=1,
        help=(
            f"Should be close to (or equal to) the momentum lookback "
            f"so IDMAG quality and momentum signal measure the same window."
        ),
        key="idmag_bars",
    )
    min_expanding_bars = st.number_input(
        f"Min {period_unit} for expanding vol (warm-up)",
        min_value=5, max_value=100, value=20, step=5,
        key="min_expanding_bars",
    )

with st.sidebar.expander("Signal Counts", expanded=True):
    top_n_universe = st.number_input("Universe top N (buys)", 1, 25, 8, key="tnu")
    bottom_n_universe = st.number_input("Universe bottom N (sells)", 1, 25, 8, key="bnu")
    top_n_theme = st.number_input("Theme top N (buys)", 1, 10, 3, key="tnt")
    bottom_n_theme = st.number_input("Theme bottom N (sells)", 1, 10, 3, key="bnt")
    min_coins_per_theme = st.number_input(
        "Min coins per theme", 3, 20, 5, key="mcpt",
        help="Themes with fewer coins are excluded (cs-std too unstable).",
    )

with st.sidebar.expander("IDMAG Filter", expanded=True):
    idmag_filter_enabled = st.checkbox("Enable IDMAG quality filter", value=True, key="ife")
    idmag_filter_pct = st.number_input(
        "Pct to filter out (top IDMAG)",
        min_value=0.05, max_value=0.5, value=0.20, step=0.05,
        help=(
            "If empirical validation shows low-IDMAG outperforms high-IDMAG, "
            "the top X% by IDMAG is removed. If inverted, the bottom X% is removed."
        ),
        key="ifp",
    )

with st.sidebar.expander("Robustness", expanded=False):
    cs_std_min = st.number_input(
        "Min cross-sectional std (z-score guard)",
        min_value=1e-6, max_value=0.1, value=1e-4, step=1e-5,
        format="%.6f",
        key="csm",
        help="Z-scores set to NaN when cs-std falls below this.",
    )
    winsorize_pct = st.number_input(
        "Winsorize percentile (each tail)",
        min_value=0.0, max_value=0.1, value=0.05, step=0.01,
        key="wp",
        help="Applied using expanding quantiles to avoid lookahead.",
    )

st.sidebar.markdown("---")
refresh = st.sidebar.button("🔄 Refresh / Recompute", use_container_width=True)

if "xs_mom_v2_results" not in st.session_state:
    st.session_state["xs_mom_v2_results"] = None


# =============================================================
# UTILITIES
# =============================================================

def bars_to_human(n_bars: int, bar_hours: float, is_daily: bool) -> str:
    """Convert a bar count to a human-readable duration string."""
    if is_daily:
        return f"{n_bars}d"
    total_hours = n_bars * bar_hours
    if total_hours >= 24:
        return f"{total_hours/24:.1f}d ({n_bars} bars)"
    return f"{total_hours:.0f}h ({n_bars} bars)"


# =============================================================
# STEP 1 — LOG RETURNS  (P2)
# =============================================================

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Log returns: r_t = ln(P_t / P_{t-1})
    Rolling sum of log returns = ln of compounded return (geometrically exact).
    """
    lr = np.log(prices / prices.shift(1))
    lr = lr.iloc[1:]
    lr = lr.dropna(axis=1, how="all")
    return lr


# =============================================================
# STEP 2 — EXPANDING VOL SCALING  (P1)
# =============================================================

def expanding_vol_scale(
    log_returns: pd.DataFrame,
    min_periods: int = 20,
) -> pd.DataFrame:
    """
    Scale each bar's return by the expanding std known BEFORE that bar.
    expanding_std is shifted by 1 so we never use the current bar's return
    in its own denominator. No lookahead bias.
    """
    expanding_std = log_returns.expanding(min_periods=min_periods).std().shift(1)
    expanding_std = expanding_std.replace(0, np.nan)
    scaled = log_returns / expanding_std
    scaled = scaled.clip(lower=-5, upper=5)
    return scaled


# =============================================================
# STEP 3 — IDMAG (Frog-in-the-Pan quality score)
# =============================================================

def _idmag_single_window(window_returns: pd.Series) -> float:
    """
    IDMAG for one window, Equation (2) of Da-Gurun-Warachka (2014).
    Weights: Q1 (smallest |r|) = 5/15, …, Q5 (largest |r|) = 1/15.
    Returns NaN if the window cannot be quintile-split.
    """
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


def compute_idmag(
    log_returns: pd.DataFrame,
    lookback_bars: int,
) -> pd.Series:
    """
    Compute a single scalar IDMAG per coin averaged over all rolling windows.
    Returns a Series indexed by coin name.
    """
    idmag_vals = {}
    for coin in log_returns.columns:
        r = log_returns[coin].dropna()
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


def validate_idmag_direction(
    idmag: pd.Series,
    log_returns: pd.DataFrame,
    forward_bars: int,
    n_quintiles: int = 5,
) -> Tuple[pd.DataFrame, bool]:
    """
    P3: Empirically test whether low-IDMAG coins outperform high-IDMAG coins.
    """
    valid_coins = idmag.dropna().index.tolist()
    valid_coins = [c for c in valid_coins if c in log_returns.columns]
    if len(valid_coins) < n_quintiles:
        return pd.DataFrame(), True

    fwd_return = log_returns[valid_coins].iloc[-forward_bars:].sum()
    combined = pd.DataFrame({"IDMAG": idmag[valid_coins], "fwd_return": fwd_return})
    combined = combined.dropna()

    try:
        combined["quintile"] = pd.qcut(
            combined["IDMAG"], n_quintiles,
            labels=[f"Q{i+1}" for i in range(n_quintiles)],
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
    """
    Returns (kept_coins, removed_coins).
    """
    valid = idmag.dropna()
    valid = valid[valid.index.isin(log_returns.columns)]
    n_remove = max(1, int(len(valid) * filter_pct))

    if low_idmag_wins:
        removed = valid.nlargest(n_remove).index.tolist()
    else:
        removed = valid.nsmallest(n_remove).index.tolist()

    kept = [c for c in log_returns.columns if c not in removed]
    return kept, removed


# =============================================================
# STEP 4 — MOMENTUM SCORES  (P2, P5, P6, P7)
# =============================================================

def calculate_momentum_scores(
    vol_scaled: pd.DataFrame,
    lookback_bars: int,
    exclude_bars: int,
    cs_std_min: float = 1e-4,
    winsorize_pct: float = 0.05,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Momentum = rolling sum of vol-scaled log returns over lookback_bars,
    after skipping the most recent exclude_bars (reversal avoidance).
    """
    shifted = vol_scaled.shift(exclude_bars)
    cum_ret = shifted.rolling(window=lookback_bars, min_periods=lookback_bars // 2).sum()

    # Winsorize using expanding quantiles — no lookahead
    cum_ret_winsorized = cum_ret.copy()
    for col in cum_ret.columns:
        series = cum_ret[col]
        lo = series.expanding(min_periods=10).quantile(winsorize_pct)
        hi = series.expanding(min_periods=10).quantile(1 - winsorize_pct)
        cum_ret_winsorized[col] = series.clip(lower=lo, upper=hi)

    cs_mean = cum_ret_winsorized.mean(axis=1)
    cs_std = cum_ret_winsorized.std(axis=1)
    cs_std_safe = cs_std.where(cs_std >= cs_std_min, other=np.nan)
    z_scores = cum_ret_winsorized.sub(cs_mean, axis=0).div(cs_std_safe, axis=0)
    ranks = z_scores.rank(axis=1, ascending=False)

    return z_scores, ranks


# =============================================================
# STEP 5 — SIGNAL GENERATION
# =============================================================

def generate_signals(
    ranks: pd.DataFrame,
    top_n: int,
    bottom_n: int,
) -> Tuple[pd.Series, pd.Series]:
    buy_dict, sell_dict = {}, {}
    for idx in ranks.index:
        row = ranks.loc[idx].dropna()
        n_valid = len(row)
        if n_valid < top_n + bottom_n:
            continue
        buy_dict[idx] = row.nsmallest(top_n).index.tolist()
        sell_dict[idx] = row.nlargest(bottom_n).index.tolist()
    return pd.Series(buy_dict, dtype=object), pd.Series(sell_dict, dtype=object)


def signal_to_df(s: pd.Series) -> pd.DataFrame:
    if s is None or len(s) == 0:
        return pd.DataFrame(columns=["Date", "Tickers"]).set_index("Date")
    rows = {
        idx: ", ".join(map(str, v)) if isinstance(v, (list, tuple, pd.Index)) else str(v)
        for idx, v in s.items()
    }
    return pd.DataFrame.from_dict(rows, orient="index", columns=["Tickers"])


# =============================================================
# STEP 6 — SIGNAL ALIGNMENT (P7)
# =============================================================

def build_alignment_matrix(
    universe_buy: List[str],
    universe_sell: List[str],
    theme_signals: Dict[str, Dict[str, List[str]]],
) -> pd.DataFrame:
    rows = []
    for theme, signals in theme_signals.items():
        for coin in signals.get("buy", []):
            if coin in universe_buy:
                alignment = "✅ Aligned (Buy)"
            elif coin in universe_sell:
                alignment = "⚠️ Conflict (Theme=Buy, Universe=Sell)"
            else:
                alignment = "— Theme-only Buy"
            rows.append({"Theme": theme, "Coin": coin, "Theme Signal": "Buy", "Alignment": alignment})
        for coin in signals.get("sell", []):
            if coin in universe_sell:
                alignment = "✅ Aligned (Sell)"
            elif coin in universe_buy:
                alignment = "⚠️ Conflict (Theme=Sell, Universe=Buy)"
            else:
                alignment = "— Theme-only Sell"
            rows.append({"Theme": theme, "Coin": coin, "Theme Signal": "Sell", "Alignment": alignment})
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# =============================================================
# MAIN COMPUTATION
# =============================================================

if refresh or st.session_state["xs_mom_v2_results"] is None:
    with st.spinner("Computing signals…"):

        # All windows are already in bars — no conversion needed.
        # bar_hours is used only for human-readable audit labels.
        _momentum_bars  = int(momentum_bars)
        _skip_bars      = int(skip_bars)
        _idmag_bars     = int(idmag_bars)
        _forward_bars   = int(momentum_bars)   # IDMAG forward validation = same as momentum

        param_audit = {
            "Detected frequency": freq_label,
            "Bar size (hours)": round(bar_hours, 2),
            f"Momentum lookback ({period_unit})": _momentum_bars,
            "Momentum lookback (equivalent time)": bars_to_human(_momentum_bars, bar_hours, is_daily),
            f"Reversal skip ({period_unit})": _skip_bars,
            "Reversal skip (equivalent time)": bars_to_human(_skip_bars, bar_hours, is_daily),
            f"IDMAG lookback ({period_unit})": _idmag_bars,
            "IDMAG lookback (equivalent time)": bars_to_human(_idmag_bars, bar_hours, is_daily),
            f"IDMAG forward validation ({period_unit})": _forward_bars,
            f"Min expanding vol ({period_unit})": int(min_expanding_bars),
        }

        # ── P2: Log returns ──────────────────────────────────────────────────
        log_ret = compute_log_returns(df_raw)

        # ── P1: Expanding vol scaling ────────────────────────────────────────
        vol_scaled_all = expanding_vol_scale(log_ret, min_periods=int(min_expanding_bars))

        # ── IDMAG ────────────────────────────────────────────────────────────
        idmag_series = compute_idmag(log_ret, lookback_bars=_idmag_bars)

        # ── P3: Validate IDMAG direction ─────────────────────────────────────
        quintile_df, low_idmag_wins = validate_idmag_direction(
            idmag_series, log_ret, forward_bars=_forward_bars
        )
        filter_direction_label = (
            "Standard (remove high-IDMAG)" if low_idmag_wins
            else "⚠️ INVERTED — removing low-IDMAG (high-IDMAG outperforms in this universe)"
        )

        # ── Apply IDMAG filter ───────────────────────────────────────────────
        if idmag_filter_enabled and not idmag_series.dropna().empty:
            kept_coins, removed_coins = apply_idmag_filter(
                idmag_series, log_ret,
                filter_pct=float(idmag_filter_pct),
                low_idmag_wins=low_idmag_wins,
            )
        else:
            kept_coins = list(log_ret.columns)
            removed_coins = []

        vol_scaled = vol_scaled_all[kept_coins]

        # ── Universe momentum  (P7: same lookback as theme) ─────────────────
        z_scores_u, ranks_u = calculate_momentum_scores(
            vol_scaled,
            lookback_bars=_momentum_bars,
            exclude_bars=_skip_bars,
            cs_std_min=float(cs_std_min),
            winsorize_pct=float(winsorize_pct),
        )
        buy_signals_u, sell_signals_u = generate_signals(
            ranks_u, top_n=int(top_n_universe), bottom_n=int(bottom_n_universe)
        )

        # ── Latest universe signals ──────────────────────────────────────────
        last_ts = ranks_u.index[-1] if len(ranks_u) > 0 else None
        universe_buy_latest  = buy_signals_u.iloc[-1]  if len(buy_signals_u)  > 0 else []
        universe_sell_latest = sell_signals_u.iloc[-1] if len(sell_signals_u) > 0 else []
        if not isinstance(universe_buy_latest, list):
            universe_buy_latest  = list(universe_buy_latest)  if hasattr(universe_buy_latest, '__iter__') else []
        if not isinstance(universe_sell_latest, list):
            universe_sell_latest = list(universe_sell_latest) if hasattr(universe_sell_latest, '__iter__') else []

        # ── Theme-level momentum  (P7: same lookback_bars) ──────────────────
        filtered_tickers = vol_scaled.columns.tolist()
        theme_map = {t: ticker_to_theme.get(t, "UNKNOWN") for t in filtered_tickers}

        theme_list = sorted({
            th for th in theme_map.values()
            if th and th.upper() != "UNKNOWN"
        })

        all_theme_signals_row = {}
        consolidated_rows = []
        theme_diagnostics = {}

        for theme in theme_list:
            theme_tickers = [t for t in filtered_tickers if theme_map.get(t) == theme]
            if len(theme_tickers) < int(min_coins_per_theme):
                continue

            th_vol = vol_scaled[theme_tickers]
            z_th, rk_th = calculate_momentum_scores(
                th_vol,
                lookback_bars=_momentum_bars,   # P7: unified
                exclude_bars=_skip_bars,
                cs_std_min=float(cs_std_min),
                winsorize_pct=float(winsorize_pct),
            )
            buy_th, sell_th = generate_signals(
                rk_th, top_n=int(top_n_theme), bottom_n=int(bottom_n_theme)
            )

            buy_last  = list(buy_th.iloc[-1])  if len(buy_th)  > 0 else []
            sell_last = list(sell_th.iloc[-1]) if len(sell_th) > 0 else []

            all_theme_signals_row[theme] = {"buy": buy_last, "sell": sell_last}

            cs_std_last = th_vol.iloc[-_momentum_bars:].std(axis=1).iloc[-1] if len(th_vol) >= _momentum_bars else np.nan

            theme_diagnostics[theme] = {
                "n_coins": len(theme_tickers),
                "cs_std_last_bar": round(float(cs_std_last), 6) if not np.isnan(cs_std_last) else "NaN",
                "cs_std_below_guard": bool(cs_std_last < float(cs_std_min)) if not np.isnan(cs_std_last) else False,
            }

            consolidated_rows.append({
                "Theme": theme,
                "N coins": len(theme_tickers),
                "Buy": ", ".join(buy_last),
                "Sell": ", ".join(sell_last),
            })

        consolidated_signals = (
            pd.DataFrame(consolidated_rows).set_index("Theme")
            if consolidated_rows else pd.DataFrame()
        )

        # ── Signal alignment matrix (P7) ─────────────────────────────────────
        alignment_df = build_alignment_matrix(
            universe_buy_latest, universe_sell_latest, all_theme_signals_row
        )

        # ── Vol scaling comparison diagnostic ────────────────────────────────
        full_period_std = log_ret[kept_coins].std()
        expanding_std_last = log_ret[kept_coins].expanding(
            min_periods=int(min_expanding_bars)
        ).std().iloc[-1]
        vol_ratio = (expanding_std_last / full_period_std).dropna()

        # Store everything
        st.session_state["xs_mom_v2_results"] = {
            "log_ret": log_ret,
            "vol_scaled": vol_scaled,
            "idmag_series": idmag_series,
            "quintile_df": quintile_df,
            "low_idmag_wins": low_idmag_wins,
            "filter_direction_label": filter_direction_label,
            "removed_coins": removed_coins,
            "kept_coins": kept_coins,
            "z_scores_u": z_scores_u,
            "buy_signals_u": buy_signals_u,
            "sell_signals_u": sell_signals_u,
            "universe_buy_latest": universe_buy_latest,
            "universe_sell_latest": universe_sell_latest,
            "consolidated_signals": consolidated_signals,
            "alignment_df": alignment_df,
            "theme_diagnostics": theme_diagnostics,
            "vol_ratio": vol_ratio,
            "param_audit": param_audit,
            "last_ts": last_ts,
        }


# =============================================================
# DISPLAY
# =============================================================
results = st.session_state["xs_mom_v2_results"]

if results is None:
    st.info("Click **Refresh / Recompute** in the sidebar to run the strategy.")
    st.stop()

(
    log_ret, vol_scaled, idmag_series, quintile_df, low_idmag_wins,
    filter_direction_label, removed_coins, kept_coins,
    z_scores_u, buy_signals_u, sell_signals_u,
    universe_buy_latest, universe_sell_latest,
    consolidated_signals, alignment_df, theme_diagnostics,
    vol_ratio, param_audit, last_ts,
) = (
    results["log_ret"], results["vol_scaled"], results["idmag_series"],
    results["quintile_df"], results["low_idmag_wins"],
    results["filter_direction_label"], results["removed_coins"], results["kept_coins"],
    results["z_scores_u"], results["buy_signals_u"], results["sell_signals_u"],
    results["universe_buy_latest"], results["universe_sell_latest"],
    results["consolidated_signals"], results["alignment_df"], results["theme_diagnostics"],
    results["vol_ratio"], results["param_audit"], results["last_ts"],
)

# ── Parameter audit ──────────────────────────────────────────────────────────
with st.expander("📐 Parameter Audit (bar-count translation)", expanded=False):
    st.table(pd.DataFrame.from_dict(param_audit, orient="index", columns=["Value"]))

# ── IDMAG section ─────────────────────────────────────────────────────────────
st.subheader("IDMAG Momentum Quality Filter")

col_a, col_b = st.columns([1, 2])
with col_a:
    st.metric("Filter direction", filter_direction_label[:40])
    st.metric("Coins removed", len(removed_coins))
    st.metric("Coins retained", len(kept_coins))
    if removed_coins:
        st.markdown("**Removed coins:**")
        st.dataframe(pd.DataFrame({"Coin": removed_coins}), use_container_width=True, height=200)

with col_b:
    if quintile_df is not None and not quintile_df.empty:
        bar_colors = ["#2e7d32" if v >= 0 else "#c62828"
                      for v in quintile_df["mean_fwd_return"]]
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
                "IDMAG Quintile → Mean Forward Return<br>"
                "<sup>Q1 = lowest IDMAG (consistent momentum), Q5 = highest IDMAG (noisy momentum)</sup>"
            ),
            xaxis_title="IDMAG Quintile",
            yaxis_title="Mean forward log-return",
            height=320,
            margin=dict(t=60, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(
                x=0.5, y=1.12, xref="paper", yref="paper",
                text=(
                    "✅ Low-IDMAG outperforms → standard filter correct"
                    if low_idmag_wins
                    else "⚠️ High-IDMAG outperforms → filter INVERTED for this universe"
                ),
                showarrow=False,
                font=dict(size=11, color="#2e7d32" if low_idmag_wins else "#c62828"),
            )],
        )
        st.plotly_chart(fig_idmag, use_container_width=True)
    else:
        st.info("Not enough data for IDMAG quintile validation.")

# ── Vol scaling diagnostic ────────────────────────────────────────────────────
with st.expander("🔬 Vol Scaling Diagnostic: Expanding vs Full-Period Std", expanded=False):
    st.markdown(
        "Ratio of expanding-std (used in v2) to full-period-std (v1 lookahead). "
        "Values < 1 mean the expanding estimate is tighter at this point in time — "
        "showing where v1 was under-scaling (inflating momentum scores)."
    )
    if not vol_ratio.empty:
        fig_vr = go.Figure(go.Bar(
            x=vol_ratio.index,
            y=vol_ratio.values,
            marker_color=["#1565c0" if v < 1 else "#e65100" for v in vol_ratio.values],
        ))
        fig_vr.add_hline(y=1.0, line_dash="dash", line_color="gray",
                         annotation_text="ratio = 1 (no difference)")
        fig_vr.update_layout(
            title="Expanding Std / Full-Period Std (last bar, per coin)",
            xaxis_title="Coin", yaxis_title="Ratio",
            height=320, margin=dict(t=40, b=60),
            xaxis_tickangle=-45,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_vr, use_container_width=True)

# ── Momentum score distribution ───────────────────────────────────────────────
with st.expander("📊 Momentum Score Distribution (latest bar)", expanded=False):
    if not z_scores_u.empty:
        last_z = z_scores_u.iloc[-1].dropna().sort_values(ascending=False)
        colors = ["#2e7d32" if v > 0 else "#c62828" for v in last_z.values]
        fig_zd = go.Figure(go.Bar(
            x=last_z.index, y=last_z.values,
            marker_color=colors,
        ))
        fig_zd.add_hline(y=0, line_dash="dot", line_color="white")
        fig_zd.update_layout(
            title=f"Universe Momentum Z-scores — {str(last_ts)[:16]}",
            xaxis_title="Coi
