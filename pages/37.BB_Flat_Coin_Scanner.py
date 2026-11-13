"""Bollinger Band squeeze and flat-performer scanner.

Reads the coin universe and OHLC data from the Theme Returns Dashboard page.
Both pages must run in the same Streamlit multipage app/session.
"""
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="BB & Flat Coin Scanner", layout="wide")
st.title("BB Squeeze & Flat Coin Scanner")
st.caption(
    "Uses the latest prices fetched on Theme Returns Dashboard. Fetch data there first, "
    "then navigate here using the Streamlit sidebar."
)

# -------------------------------------------------------------------
# Shared Theme Returns session data
# -------------------------------------------------------------------
prices = st.session_state.get("price_theme")
highs = st.session_state.get("high_theme")
lows = st.session_state.get("low_theme")
ticker_to_theme = st.session_state.get("ticker_to_theme", {}) or {}
timeframe = st.session_state.get("price_timeframe")
last_fetch = st.session_state.get("price_theme_version") or st.session_state.get("last_fetch")

if not isinstance(prices, pd.DataFrame) or prices.empty:
    st.warning(
        "No Theme Returns data found in this session. Open Theme Returns Dashboard, "
        "click Fetch Data, then return here using the built-in page navigation."
    )
    st.stop()

prices = prices.copy().sort_index()
prices.columns = [str(c).upper().strip() for c in prices.columns]
if isinstance(highs, pd.DataFrame):
    highs = highs.copy().sort_index()
    highs.columns = [str(c).upper().strip() for c in highs.columns]
else:
    highs = None
if isinstance(lows, pd.DataFrame):
    lows = lows.copy().sort_index()
    lows.columns = [str(c).upper().strip() for c in lows.columns]
else:
    lows = None

ticker_to_theme = {str(k).upper().strip(): v for k, v in ticker_to_theme.items()}
timeframe = timeframe or "unknown"
st.caption(f"Source timeframe: **{timeframe}** · Last fetch: **{last_fetch or 'not recorded'}**")

# -------------------------------------------------------------------
# Sidebar controls (defaults preserve the notebook)
# -------------------------------------------------------------------
with st.sidebar:
    st.header("Bollinger Band Settings")
    bb_length = st.number_input("BB Length", min_value=2, max_value=200, value=20)
    bb_mult = st.number_input("BB Std Dev Multiplier", min_value=0.1, max_value=5.0, value=2.0, step=0.1)
    bb_ma_type = st.selectbox("BB Basis MA", ["SMA", "EMA", "WMA"])
    bb_bw_avg_window = st.number_input("Bandwidth Average Window", min_value=5, max_value=500, value=48)
    bb_contraction_bars = st.number_input("Minimum Contraction Bars", min_value=1, max_value=50, value=3)
    bb_lookback = st.number_input("BB Inflection Lookback", min_value=1, max_value=100, value=8)
    bb_slope_window = st.number_input("Basis Slope Window", min_value=1, max_value=30, value=3)
    bb_tight_pct = st.number_input("Tight Squeeze Threshold", min_value=0.0, max_value=0.9, value=0.20, step=0.05)
    watchlist_size = st.number_input("Squeeze Watchlist Size", min_value=1, max_value=100, value=10)

    st.divider()
    st.header("Flat Performer Settings")
    flat_hours = st.number_input("Flat Lookback (hours)", min_value=1.0, max_value=720.0, value=6.0, step=1.0)
    flat_min = st.number_input("Minimum Return %", min_value=-100.0, max_value=100.0, value=-0.5, step=0.1)
    flat_max = st.number_input("Maximum Return %", min_value=-100.0, max_value=100.0, value=0.5, step=0.1)
    flat_top_n = st.number_input("Flat List Rows", min_value=1, max_value=200, value=20)
    flat_sort_abs = st.checkbox("Sort by absolute return (flattest first)", value=False)

if flat_min > flat_max:
    st.error("Minimum Return % must be less than or equal to Maximum Return %.")
    st.stop()

# -------------------------------------------------------------------
# Bollinger Band functions — port of notebook cells 38, 40, 42, 48
# -------------------------------------------------------------------
def compute_wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, int(period) + 1, dtype=float)
    return series.rolling(int(period), min_periods=int(period)).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )

def compute_ma(series: pd.Series, length: int, ma_type: str) -> pd.Series:
    if ma_type == "EMA":
        return series.ewm(span=length, adjust=False).mean()
    if ma_type == "WMA":
        return compute_wma(series, length)
    return series.rolling(length, min_periods=length).mean()

def find_inflection(bandwidth: pd.Series, lookback: int, min_contraction: int):
    n = len(bandwidth)
    if n < lookback + min_contraction + 2:
        return None, 0
    for i in range(1, lookback + 1):
        idx = -i
        prev_idx = idx - 1
        curr_bw, prev_bw = bandwidth.iloc[idx], bandwidth.iloc[prev_idx]
        if pd.isna(curr_bw) or pd.isna(prev_bw) or curr_bw <= prev_bw:
            continue
        count, check_idx = 0, prev_idx
        while True:
            prior_idx = check_idx - 1
            if prior_idx < -n:
                break
            now, before = bandwidth.iloc[check_idx], bandwidth.iloc[prior_idx]
            if pd.isna(now) or pd.isna(before) or now >= before:
                break
            count += 1
            check_idx = prior_idx
        if count >= min_contraction:
            return idx, count
    return None, 0

def measure_squeeze(bandwidth, idx, contraction_len, avg_bw, tight_threshold):
    n = len(bandwidth)
    abs_idx = n + idx
    start = max(0, abs_idx - contraction_len)
    end = abs_idx
    if start >= end:
        return "MILD", 0.0
    pre_bw = bandwidth.iloc[start:end + 1].dropna()
    if pre_bw.empty or avg_bw == 0:
        return "MILD", 0.0
    ratio = (avg_bw - pre_bw.min()) / avg_bw
    return ("TIGHT" if ratio >= tight_threshold else "MILD"), round(ratio, 3)

def detect_bb_signal(close, basis, upper, lower, bandwidth,
                     avg_window, lookback, min_contraction, slope_window, tight_threshold):
    n = len(close)
    if n < avg_window + lookback + min_contraction + 2:
        return None
    avg_bw = bandwidth.iloc[-avg_window:].mean()
    if pd.isna(avg_bw) or avg_bw == 0:
        return None
    idx, contraction_len = find_inflection(bandwidth, lookback, min_contraction)
    if idx is None:
        return None
    if not (upper.iloc[idx] > upper.iloc[idx - 1] and lower.iloc[idx] < lower.iloc[idx - 1]):
        return None
    slope_start = max(idx - slope_window, -n)
    slope = basis.iloc[idx] - basis.iloc[slope_start]
    if pd.isna(slope) or slope == 0:
        return None
    direction = "LONG" if slope > 0 else "SHORT"
    squeeze_type, squeeze_ratio = measure_squeeze(
        bandwidth, idx, contraction_len, avg_bw, tight_threshold
    )
    cc, pc = close.iloc[idx], close.iloc[idx - 1]
    cb, pb = basis.iloc[idx], basis.iloc[idx - 1]
    if any(pd.isna(x) for x in (cc, pc, cb, pb)):
        return None
    if direction == "LONG":
        crossed, positioned = pc <= pb and cc > cb, cc > cb and pc > pb
    else:
        crossed, positioned = pc >= pb and cc < cb, cc < cb and pc < pb
    if crossed:
        entry = "CROSSOVER"
    elif positioned:
        entry = "POSITIONED"
    else:
        return None
    last_basis = basis.iloc[-1]
    if pd.isna(last_basis) or last_basis == 0:
        return None
    spread_pct = (close.iloc[-1] - last_basis) / last_basis * 100.0
    return {
        "direction": direction,
        "squeeze": squeeze_type,
        "entry": entry,
        "spread": round(spread_pct, 2),
        "depth": round(squeeze_ratio * 100, 1),
        "contraction": contraction_len,
        "bars_ago": abs(idx),
    }

def scan_bb(prices):
    long_rows, short_rows, skipped, cache = [], [], [], {}
    signals = set()
    min_bars = int(bb_length + bb_bw_avg_window + bb_lookback + bb_contraction_bars + 2)
    for sym in prices.columns:
        close = pd.to_numeric(prices[sym], errors="coerce").dropna().astype(float)
        if len(close) < min_bars:
            skipped.append(sym)
            continue
        basis = compute_ma(close, int(bb_length), bb_ma_type)
        dev = float(bb_mult) * close.rolling(int(bb_length), min_periods=int(bb_length)).std()
        upper, lower = basis + dev, basis - dev
        bandwidth = upper - lower
        cache[sym] = bandwidth
        result = detect_bb_signal(
            close, basis, upper, lower, bandwidth,
            int(bb_bw_avg_window), int(bb_lookback), int(bb_contraction_bars),
            int(bb_slope_window), float(bb_tight_pct)
        )
        if result is None:
            continue
        row = {
            "Coin": sym,
            "Theme": ticker_to_theme.get(sym, "UNKNOWN"),
            "Close": round(float(close.iloc[-1]), 8),
            "Basis": round(float(basis.iloc[-1]), 8),
            "Spread %": result["spread"],
            "Squeeze": result["squeeze"],
            "Entry": result["entry"],
            "Squeeze Depth %": result["depth"],
            "Contraction Bars": result["contraction"],
            "Signal (bars ago)": result["bars_ago"],
        }
        (long_rows if result["direction"] == "LONG" else short_rows).append(row)
        signals.add(sym)

    def ordered(rows):
        df = pd.DataFrame(rows)
        if not df.empty:
            df["_sq"] = df["Squeeze"].map({"TIGHT": 0, "MILD": 1})
            df["_entry"] = df["Entry"].map({"CROSSOVER": 0, "POSITIONED": 1})
            df = df.sort_values(
                ["_sq", "_entry", "Squeeze Depth %"],
                ascending=[True, True, False]
            ).drop(columns=["_sq", "_entry"]).reset_index(drop=True)
            df.index += 1
        return df

    # Current active squeeze watchlist: still below average BW and contracting.
    watch_rows = []
    for sym, bandwidth in cache.items():
        if sym in signals or len(bandwidth.dropna()) < int(bb_bw_avg_window) + 2:
            continue
        avg_bw = bandwidth.iloc[-int(bb_bw_avg_window):].mean()
        curr_bw = bandwidth.iloc[-1]
        if pd.isna(avg_bw) or avg_bw == 0 or pd.isna(curr_bw):
            continue
        depth = (avg_bw - curr_bw) / avg_bw
        if depth <= 0:
            continue
        contraction = 0
        for j in range(1, len(bandwidth)):
            if bandwidth.iloc[-j] < bandwidth.iloc[-(j + 1)]:
                contraction += 1
            else:
                break
        if contraction:
            watch_rows.append({
                "Coin": sym,
                "Theme": ticker_to_theme.get(sym, "UNKNOWN"),
                "BW": float(curr_bw),
                "Avg BW": float(avg_bw),
                "Squeeze Depth %": round(float(depth * 100), 1),
                "Contraction Bars": contraction,
                "_depth": float(depth),
                "_contraction": contraction,
            })

    watch = pd.DataFrame(watch_rows)
    if not watch.empty:
        max_contraction = watch["_contraction"].max()
        watch["_norm"] = watch["_contraction"] / max_contraction if max_contraction > 0 else 0
        watch["Score"] = (watch["_depth"] * 0.6 + watch["_norm"] * 0.4) * 100
        watch = watch.drop(columns=["_depth", "_contraction", "_norm"])
        watch = watch.sort_values("Score", ascending=False).head(int(watchlist_size)).reset_index(drop=True)
        watch.index += 1
    else:
        watch = pd.DataFrame(columns=[
            "Coin", "Theme", "BW", "Avg BW", "Squeeze Depth %",
            "Contraction Bars", "Score"
        ])
    return ordered(long_rows), ordered(short_rows), watch, skipped

with st.spinner("Calculating Bollinger Band signals..."):
    bb_long, bb_short, bb_watch, bb_skipped = scan_bb(prices)

# -------------------------------------------------------------------
# Flat Performers — port of notebook cell 54
# -------------------------------------------------------------------
def timeframe_minutes(tf):
    if not tf or len(tf) < 2:
        return None
    units = {"m": 1, "h": 60, "d": 1440, "w": 10080}
    try:
        n, unit = float(tf[:-1]), tf[-1]
    except (ValueError, TypeError):
        return None
    return n * units[unit] if unit in units else None

def scan_flat():
    tf_min = timeframe_minutes(timeframe)
    if not tf_min or tf_min <= 0:
        return pd.DataFrame(), [], None, "Cannot interpret source timeframe."
    bars = int(round(float(flat_hours) * 60 / tf_min))
    if bars < 1:
        return pd.DataFrame(), [], bars, (
            f"{timeframe} candles are too coarse for a {flat_hours:g}-hour lookback. "
            "Choose 1h on Theme Returns Dashboard for the notebook's default 6-hour window."
        )
    rows, skipped = [], []
    has_ranges = isinstance(highs, pd.DataFrame) and isinstance(lows, pd.DataFrame)
    for sym in prices.columns:
        close = pd.to_numeric(prices[sym], errors="coerce").dropna().astype(float)
        if len(close) < bars + 1:
            skipped.append(sym)
            continue
        start_close, end_close = close.iloc[-(bars + 1)], close.iloc[-1]
        if pd.isna(start_close) or start_close == 0:
            skipped.append(sym)
            continue
        ret_pct = (end_close / start_close - 1.0) * 100.0
        if not (float(flat_min) <= ret_pct <= float(flat_max)):
            continue
        range_pct = np.nan
        if has_ranges and sym in highs.columns and sym in lows.columns:
            h = pd.to_numeric(highs[sym], errors="coerce").dropna().astype(float)
            l = pd.to_numeric(lows[sym], errors="coerce").dropna().astype(float)
            if len(h) >= bars and len(l) >= bars:
                range_pct = (h.iloc[-bars:].max() - l.iloc[-bars:].min()) / start_close * 100.0
        rows.append({
            "Coin": sym,
            "Theme": ticker_to_theme.get(sym, "UNKNOWN"),
            "Start Close": float(start_close),
            "Last Close": float(end_close),
            "Return %": round(float(ret_pct), 3),
            "Range %": round(float(range_pct), 2) if pd.notna(range_pct) else np.nan,
        })
    flat = pd.DataFrame(rows)
    if not flat.empty:
        if flat_sort_abs:
            flat = flat.iloc[flat["Return %"].abs().argsort()]
        else:
            flat = flat.sort_values("Return %", ascending=True)
        flat = flat.head(int(flat_top_n)).reset_index(drop=True)
        flat.index += 1
    return flat, skipped, bars, None

flat_df, flat_skipped, flat_bars, flat_error = scan_flat()

# -------------------------------------------------------------------
# Display
# -------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Coins scanned", len(prices.columns))
c2.metric("BB Long", len(bb_long))
c3.metric("BB Short", len(bb_short))
c4.metric("Flat performers", len(flat_df))

st.subheader("Bollinger Band Squeeze-Inflection Signals")
st.caption(
    "A signal requires bandwidth to turn up after the configured number of declining bars, "
    "the upper/lower bands to diverge, and a directional basis slope. Entry is CROSSOVER or POSITIONED."
)
left, right = st.columns(2)
with left:
    st.markdown("#### 🟢 BB Long Signals")
    if bb_long.empty:
        st.info("No bullish BB inflection signals in the configured lookback.")
    else:
        st.dataframe(bb_long, use_container_width=True)
with right:
    st.markdown("#### 🔴 BB Short Signals")
    if bb_short.empty:
        st.info("No bearish BB inflection signals in the configured lookback.")
    else:
        st.dataframe(bb_short, use_container_width=True)

st.markdown("#### ⏳ BB Squeeze Watchlist — Potential Next Breakouts")
st.caption("Excludes coins with a current BB signal. Score = 60% squeeze depth + 40% normalized contraction length.")
if bb_watch.empty:
    st.info("No coins are currently below average bandwidth and actively contracting.")
else:
    st.dataframe(bb_watch, use_container_width=True)

if bb_skipped:
    with st.expander(f"BB skipped coins ({len(bb_skipped)}) — insufficient candle history"):
        st.write(", ".join(sorted(bb_skipped)))

st.divider()
st.subheader("Flat Performers — Range-Bound Coins")
if flat_error:
    st.warning(flat_error)
else:
    tf_min = timeframe_minutes(timeframe)
    actual_hours = flat_bars * tf_min / 60 if tf_min else None
    st.caption(
        f"Return band: {flat_min:+.3f}% to {flat_max:+.3f}% over {flat_hours:g} requested hours "
        f"({flat_bars} × {timeframe} bars; effective window ≈ {actual_hours:g}h). "
        f"Sort: {'absolute return' if flat_sort_abs else 'signed return'}; top {int(flat_top_n)}."
    )
    if flat_df.empty:
        st.info("No coins are within the selected return band for this window.")
    else:
        st.dataframe(
            flat_df,
            use_container_width=True,
            column_config={
                "Start Close": st.column_config.NumberColumn(format="%.8g"),
                "Last Close": st.column_config.NumberColumn(format="%.8g"),
                "Return %": st.column_config.NumberColumn(format="%+.3f%%"),
                "Range %": st.column_config.NumberColumn(format="%.2f%%"),
            }
        )
    if highs is None or lows is None:
        st.caption("Range % is unavailable because the Theme Returns page has not shared high/low data.")
    if flat_skipped:
        with st.expander(f"Flat scan skipped coins ({len(flat_skipped)}) — insufficient data"):
            st.write(", ".join(sorted(flat_skipped)))

st.caption("Refresh data on Theme Returns Dashboard; this page performs calculations only.")
