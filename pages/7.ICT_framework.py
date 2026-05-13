# ict_framework.py
# ICT Decision Framework — Python/Streamlit adaptation
# v6.2-aligned: array-based liquidity, multi-bar weekly bias, daily range P/D
#
# SIGNAL LOGIC (per coin, hourly+ resolution):
#   LONG  : weekly_bull AND daily_draw_bull AND h_discount AND ssl_raid_recent
#   SHORT : weekly_bear AND daily_draw_bear AND h_premium  AND bsl_raid_recent
#
# v6.2 FIXES APPLIED:
#   1. Weekly bias: multi-bar slope + price-vs-EMA + neutral state allowed
#   2. Daily liquidity: array-based with sweep removal, nearest above/below
#   3. Hourly stop hunt: array-based with sweep removal, nearest-level raids
#   4. P/D zone: daily range mode (Ep2) + hourly N-bar range (toggle)
#   5. Daily draw alignment: simplified for array-based (level existence = unswept)
#   6. Ranking uses array-derived nearest unswept BSL/SSL levels

import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="ICT Framework Scanner", layout="wide")

st.markdown("""
<style>
  .main { background: #0a0a0f; }
  .block-container { padding-top: 1.5rem; }
  .sig-header { font-family: 'Courier New', monospace; letter-spacing: 2px; }
  div[data-testid="metric-container"] {
    background: #111120;
    border: 1px solid #2a2a3e;
    border-radius: 6px;
    padding: 12px;
  }
</style>
""", unsafe_allow_html=True)

st.title("📐 ICT Decision Framework — Multi-Coin Scanner")
st.caption("Weekly bias → Daily draw → Hourly P/D → Hourly stop hunt  |  v6.2-aligned")

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar parameters
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ ICT Parameters")

    st.subheader("① Weekly Bias")
    bias_ema_len    = st.slider("Weekly EMA Length", 5, 50, 20)
    bias_slope_bars = st.slider("Slope Confirmation Bars", 1, 10, 3,
                                help="EMA must slope consistently over this many weekly bars")

    st.subheader("② Daily Liquidity Draw")
    daily_swing      = st.slider("Daily Swing Lookback (bars)", 2, 20, 5)
    daily_max_levels = st.slider("Max Stored Daily Swings", 3, 30, 10,
                                 help="How many recent unswept daily pivots to keep per side")

    st.subheader("③ Premium / Discount + Stop Hunt")
    pd_method   = st.selectbox("P/D Calculation",
                               ["Daily Range (Ep2)", "Hourly N-Bar Range"],
                               help="Ep2: 'low of the day and high of the day … midpoint … premium/discount'")
    pd_lookback = st.slider("Hourly Range Lookback (bars)", 20, 200, 50,
                            help="Only used when P/D = 'Hourly N-Bar Range'")
    htf_swing      = st.slider("Hourly Swing Lookback (bars)", 2, 20, 5)
    htf_max_levels = st.slider("Max Stored Hourly Swings", 3, 30, 10)
    raid_window    = st.slider("Raid Valid Window (bars)", 2, 30, 8,
                               help="How many 1H bars a stop-hunt stays 'recent'")

    st.subheader("④ Filters")
    require_pd    = st.checkbox("Require P/D Confirmation", True)
    require_daily = st.checkbox("Require Daily Draw Alignment", True)

    st.subheader("⑤ Display")
    top_n = st.slider("Top N coins per side", 5, 30, 15)

# ─────────────────────────────────────────────────────────────────────────────
# Pivot helpers (unchanged — these are correct)
# ─────────────────────────────────────────────────────────────────────────────

def pivot_high(high: np.ndarray, left: int, right: int) -> np.ndarray:
    """Return array of pivot-high values (NaN where not a pivot)."""
    n = len(high)
    out = np.full(n, np.nan)
    for i in range(left, n - right):
        window = high[i - left : i + right + 1]
        if high[i] == window.max():
            out[i] = high[i]
    return out


def pivot_low(low: np.ndarray, left: int, right: int) -> np.ndarray:
    """Return array of pivot-low values (NaN where not a pivot)."""
    n = len(low)
    out = np.full(n, np.nan)
    for i in range(left, n - right):
        window = low[i - left : i + right + 1]
        if low[i] == window.min():
            out[i] = low[i]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Array-based liquidity tracking (ported from Pine v6.2)
# ─────────────────────────────────────────────────────────────────────────────

class LiquidityTracker:
    """
    Maintains arrays of unswept swing highs (BSL) and swing lows (SSL).
    Sweeps are removed as price takes them out.
    Nearest above/below current price is the actionable level.

    This replaces the 'last_valid(pivot)' approach that caused the
    D-BSL/D-SSL TF-instability bug and stale-level tracking.
    """

    def __init__(self, max_levels: int = 10):
        self.max_levels = max_levels
        self.highs: list[float] = []   # BSL candidates (unswept swing highs)
        self.lows: list[float] = []    # SSL candidates (unswept swing lows)

    def add_high(self, level: float):
        """Register a confirmed pivot high."""
        if np.isnan(level):
            return
        # Avoid exact duplicates from repeated pivot confirmation
        if level not in self.highs:
            self.highs.append(level)
            # Cap size: drop oldest
            while len(self.highs) > self.max_levels:
                self.highs.pop(0)

    def add_low(self, level: float):
        """Register a confirmed pivot low."""
        if np.isnan(level):
            return
        if level not in self.lows:
            self.lows.append(level)
            while len(self.lows) > self.max_levels:
                self.lows.pop(0)

    def sweep_highs(self, bar_high: float):
        """Remove any BSL levels that price has traded through."""
        if np.isnan(bar_high):
            return
        self.highs = [h for h in self.highs if h > bar_high]

    def sweep_lows(self, bar_low: float):
        """Remove any SSL levels that price has traded through."""
        if np.isnan(bar_low):
            return
        self.lows = [l for l in self.lows if l < bar_low]

    def nearest_above(self, price: float) -> float:
        """Nearest unswept level ABOVE price (BSL target)."""
        above = [h for h in self.highs if h > price]
        return min(above) if above else np.nan

    def nearest_below(self, price: float) -> float:
        """Nearest unswept level BELOW price (SSL target)."""
        below = [l for l in self.lows if l < price]
        return max(below) if below else np.nan


def build_liquidity_arrays(
    highs: np.ndarray,
    lows: np.ndarray,
    pivot_h: np.ndarray,
    pivot_l: np.ndarray,
    max_levels: int,
) -> LiquidityTracker:
    """
    Walk through bar history, adding pivots and sweeping levels,
    returning the final state of the tracker.
    """
    tracker = LiquidityTracker(max_levels=max_levels)
    n = len(highs)
    for i in range(n):
        # Add newly confirmed pivots
        if not np.isnan(pivot_h[i]):
            tracker.add_high(pivot_h[i])
        if not np.isnan(pivot_l[i]):
            tracker.add_low(pivot_l[i])
        # Sweep removal: any level taken out by this bar's high/low
        tracker.sweep_highs(highs[i])
        tracker.sweep_lows(lows[i])
    return tracker


# ─────────────────────────────────────────────────────────────────────────────
# Stop-hunt detection using array-based tracking
# ─────────────────────────────────────────────────────────────────────────────

def detect_raids_array(
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    pivot_h: np.ndarray,
    pivot_l: np.ndarray,
    max_levels: int,
    raid_window: int,
) -> tuple[bool, bool, float, float]:
    """
    Walk through hourly bars, maintaining an array of unswept pivots.
    At each bar, check if the nearest BSL/SSL has been raided (wick
    through + close back inside).

    Returns:
        bsl_raid_recent: bool — buy-stop raid within window of last bar
        ssl_raid_recent: bool — sell-stop raid within window of last bar
        final_bsl: float — nearest unswept BSL at end of series
        final_ssl: float — nearest unswept SSL at end of series
    """
    tracker = LiquidityTracker(max_levels=max_levels)
    n = len(h)
    bsl_raid_bar = -99999
    ssl_raid_bar = -99999

    for i in range(n):
        # Register pivots
        if not np.isnan(pivot_h[i]):
            tracker.add_high(pivot_h[i])
        if not np.isnan(pivot_l[i]):
            tracker.add_low(pivot_l[i])

        # Find nearest levels BEFORE sweep removal (to detect raids this bar)
        bsl_target = tracker.nearest_above(c[i])
        ssl_target = tracker.nearest_below(c[i])

        # Raid detection: wick through + close back inside
        if not np.isnan(bsl_target) and h[i] >= bsl_target and c[i] < bsl_target:
            bsl_raid_bar = i
        if not np.isnan(ssl_target) and l[i] <= ssl_target and c[i] > ssl_target:
            ssl_raid_bar = i

        # Sweep removal AFTER raid check
        tracker.sweep_highs(h[i])
        tracker.sweep_lows(l[i])

    last_idx = n - 1
    bsl_raid_recent = (last_idx - bsl_raid_bar) <= raid_window
    ssl_raid_recent = (last_idx - ssl_raid_bar) <= raid_window

    # Final nearest levels for ranking
    final_price = c[-1] if n > 0 else np.nan
    final_bsl = tracker.nearest_above(final_price)
    final_ssl = tracker.nearest_below(final_price)

    return bsl_raid_recent, ssl_raid_recent, final_bsl, final_ssl


# ─────────────────────────────────────────────────────────────────────────────
# Core ICT computation for a single coin
# ─────────────────────────────────────────────────────────────────────────────

def compute_ict(sym: str, ohlc_multi: pd.DataFrame) -> dict | None:
    """
    Run the ICT framework for one symbol.
    Returns a dict of signal components + ranking metrics, or None on error.
    """
    try:
        # ── Extract per-symbol OHLC ──────────────────────────────────────
        o = ohlc_multi[sym]["o"].values.astype(float)
        h = ohlc_multi[sym]["h"].values.astype(float)
        l = ohlc_multi[sym]["l"].values.astype(float)
        c = ohlc_multi[sym]["c"].values.astype(float)
        idx = ohlc_multi.index

        if len(c) < max(bias_ema_len * 7, pd_lookback + 20, 100):
            return None

        df_raw = pd.DataFrame({"o": o, "h": h, "l": l, "c": c}, index=idx)

        # ─────────────────────────────────────────────────────────────────
        # SECTION 1 — WEEKLY BIAS
        #
        # FIX 1: Multi-bar slope + price-vs-EMA + neutral state
        #
        # Ep2: "each week before the new trading week begins … you want
        # to try to get a read on what you think that next weekly candle
        # is going to do — is it going to go higher or lower"
        #
        # Auto: EMA must slope in one direction over bias_slope_bars
        #       AND weekly close must be on the correct side of the EMA.
        #       If neither condition is met → neutral (no signals fire).
        # ─────────────────────────────────────────────────────────────────
        df_weekly = df_raw["c"].resample("W").last().dropna()
        if len(df_weekly) < bias_ema_len + bias_slope_bars + 2:
            return None

        w_ema = df_weekly.ewm(span=bias_ema_len, adjust=False).mean()

        w_ema_now  = w_ema.iloc[-1]
        w_ema_then = w_ema.iloc[-(1 + bias_slope_bars)]
        w_close    = df_weekly.iloc[-1]

        weekly_bull = bool(w_ema_now > w_ema_then and w_close > w_ema_now)
        weekly_bear = bool(w_ema_now < w_ema_then and w_close < w_ema_now)
        # Both False = neutral week → no signal fires

        # ─────────────────────────────────────────────────────────────────
        # SECTION 2 — DAILY LIQUIDITY DRAW
        #
        # FIX 2: Array-based tracking with sweep removal.
        #
        # Ep2: "on the daily chart you're looking for swing highs and
        # swing lows … below old lows sell stops … above old highs
        # buy stops"
        #
        # Old code: last_valid(d_ph) → most recent pivot regardless of
        # sweep status. WRONG: a swept level is dead liquidity.
        #
        # New: walk daily bars, add pivots, remove swept, pick nearest
        # above/below the daily close.
        # ─────────────────────────────────────────────────────────────────
        df_daily_h = df_raw["h"].resample("D").max().dropna()
        df_daily_l = df_raw["l"].resample("D").min().dropna()
        df_daily_c = df_raw["c"].resample("D").last().dropna()

        if len(df_daily_h) < daily_swing * 2 + 5:
            return None

        d_h_arr = df_daily_h.values
        d_l_arr = df_daily_l.values

        d_ph = pivot_high(d_h_arr, daily_swing, daily_swing)
        d_pl = pivot_low(d_l_arr,  daily_swing, daily_swing)

        d_tracker = build_liquidity_arrays(
            d_h_arr, d_l_arr, d_ph, d_pl, daily_max_levels
        )

        d_close_last = df_daily_c.iloc[-1]
        d_bsl_level = d_tracker.nearest_above(d_close_last)
        d_ssl_level = d_tracker.nearest_below(d_close_last)

        # FIX 5: With array-based tracking, nearest_above already
        # guarantees the level is above price and unswept.
        # Alignment = a target level exists in the bias direction.
        daily_draw_bull = weekly_bull and not np.isnan(d_bsl_level)
        daily_draw_bear = weekly_bear and not np.isnan(d_ssl_level)
        daily_aligned_bull = daily_draw_bull if require_daily else True
        daily_aligned_bear = daily_draw_bear if require_daily else True

        # ─────────────────────────────────────────────────────────────────
        # SECTION 3 — PREMIUM / DISCOUNT + STOP HUNT
        #
        # FIX 3: Array-based hourly liquidity tracking with sweep removal.
        # FIX 4: Daily range P/D mode.
        #
        # Ep2 (P/D): "low of the day and high of the day thus far …
        # split it … midpoint … above that 50% level = premium"
        #
        # Ep2 (raid): "above old highs … buy stops … the market went
        # up here where those buy stops are going to be resting …
        # once this occurs … you want to drop down to lower TFs"
        # ─────────────────────────────────────────────────────────────────
        n = len(c)

        # P/D equilibrium
        if pd_method == "Daily Range (Ep2)":
            # Use the most recent daily bar's high/low
            pd_range_high = d_h_arr[-1]
            pd_range_low  = d_l_arr[-1]
        else:
            # Hourly N-bar rolling window
            lb = min(pd_lookback, n - 1)
            pd_range_high = float(np.max(h[-lb:]))
            pd_range_low  = float(np.min(l[-lb:]))

        h_equilibrium = (pd_range_high + pd_range_low) / 2.0
        h_close_last = c[-1]

        h_premium  = h_close_last > h_equilibrium
        h_discount = h_close_last < h_equilibrium

        # Hourly stop-hunt detection via array-based tracking
        h_ph = pivot_high(h, htf_swing, htf_swing)
        h_pl = pivot_low(l,  htf_swing, htf_swing)

        bsl_raid_recent, ssl_raid_recent, h_bsl_level, h_ssl_level = \
            detect_raids_array(
                h, l, c, h_ph, h_pl,
                max_levels=htf_max_levels,
                raid_window=raid_window,
            )

        # P/D gating
        pd_sell_ok = h_premium  if require_pd else True
        pd_buy_ok  = h_discount if require_pd else True

        # ─────────────────────────────────────────────────────────────────
        # SECTION 4 — COMPOSITE SIGNALS (hourly+ only, no MSS/FVG/OB)
        # ─────────────────────────────────────────────────────────────────
        buy_signal  = (weekly_bull and daily_aligned_bull
                       and pd_buy_ok and ssl_raid_recent)

        sell_signal = (weekly_bear and daily_aligned_bear
                       and pd_sell_ok and bsl_raid_recent)

        # ─────────────────────────────────────────────────────────────────
        # FIX 6: RANKING METRIC — uses array-derived nearest levels
        #
        # h_bsl_level = nearest unswept hourly swing high above price
        # h_ssl_level = nearest unswept hourly swing low below price
        #
        # These are the actual actionable stop levels, not stale pivots.
        # ─────────────────────────────────────────────────────────────────
        price = h_close_last

        if not np.isnan(h_ssl_level) and price > 0:
            pct_to_ssl = (price - h_ssl_level) / price * 100.0
        else:
            pct_to_ssl = np.nan

        if not np.isnan(h_bsl_level) and price > 0:
            pct_to_bsl = (h_bsl_level - price) / price * 100.0
        else:
            pct_to_bsl = np.nan

        # Condition breakdown for dashboard
        conditions_long = {
            "Weekly Bullish":     weekly_bull,
            "Daily Draw (→ BSL)": daily_draw_bull,
            "H Discount Zone":    bool(h_discount),
            "SSL Raid Recent":    bool(ssl_raid_recent),
        }
        conditions_short = {
            "Weekly Bearish":     weekly_bear,
            "Daily Draw (→ SSL)": daily_draw_bear,
            "H Premium Zone":     bool(h_premium),
            "BSL Raid Recent":    bool(bsl_raid_recent),
        }

        long_score  = sum(conditions_long.values())
        short_score = sum(conditions_short.values())

        return {
            "symbol":          sym,
            "price":           price,
            # Signals
            "buy_signal":      buy_signal,
            "sell_signal":     sell_signal,
            # Weekly state
            "weekly_bull":     weekly_bull,
            "weekly_bear":     weekly_bear,
            "weekly_neutral":  not weekly_bull and not weekly_bear,
            # Daily draw levels (array-based, nearest unswept)
            "d_bsl_level":     d_bsl_level,
            "d_ssl_level":     d_ssl_level,
            "d_bsl_count":     len(d_tracker.highs),
            "d_ssl_count":     len(d_tracker.lows),
            # Hourly levels (array-based, nearest unswept)
            "h_bsl_level":     h_bsl_level,
            "h_ssl_level":     h_ssl_level,
            "h_equilibrium":   h_equilibrium,
            # P/D
            "pd_range_high":   pd_range_high,
            "pd_range_low":    pd_range_low,
            # Proximity (for ranking)
            "pct_to_ssl":      pct_to_ssl,
            "pct_to_bsl":      pct_to_bsl,
            # Condition breakdown
            "conditions_long": conditions_long,
            "conditions_short":conditions_short,
            "long_score":      long_score,
            "short_score":     short_score,
        }

    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main: load data and run scanner
# ─────────────────────────────────────────────────────────────────────────────

ohlc_multi: pd.DataFrame | None = st.session_state.get("ohlc_multi")

if ohlc_multi is None:
    st.warning("⚠️ No OHLC data found. Please run the **OHLC Data Loader** page first.")
    st.stop()

symbols = list(ohlc_multi.columns.get_level_values(0).unique())
timeframe_label = st.session_state.get("timeframe", "unknown")

st.info(f"Running ICT scanner on **{len(symbols)} coins** | TF: `{timeframe_label}` | "
        f"{len(ohlc_multi)} bars per coin | P/D: `{pd_method}`")

# ── Run scanner ──────────────────────────────────────────────────────────────
if st.button("▶ Run ICT Scanner", type="primary"):

    results = []
    prog = st.progress(0, text="Scanning...")

    for i, sym in enumerate(symbols):
        r = compute_ict(sym, ohlc_multi)
        if r:
            results.append(r)
        prog.progress((i + 1) / len(symbols), text=f"Scanning {sym}…")

    prog.empty()

    if not results:
        st.error("No results computed. Check OHLC data quality.")
        st.stop()

    st.session_state["ict_results"] = results
    st.success(f"Scan complete — {len(results)} coins processed.")

# ─────────────────────────────────────────────────────────────────────────────
# Display results
# ─────────────────────────────────────────────────────────────────────────────

results = st.session_state.get("ict_results")
if not results:
    st.info("Click **▶ Run ICT Scanner** to scan all coins.")
    st.stop()

# ── Summary metrics ──────────────────────────────────────────────────────────
n_buy     = sum(1 for r in results if r["buy_signal"])
n_sell    = sum(1 for r in results if r["sell_signal"])
n_neutral = sum(1 for r in results if r["weekly_neutral"])

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Scanned",    len(results))
m2.metric("🟢 Long Signals",   n_buy)
m3.metric("🔴 Short Signals",  n_sell)
m4.metric("⚪ Neutral Weeks",  n_neutral)

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# LONG SIGNAL TABLE
# Ranked by % distance to SSL (stop-loss for longs)
# ═════════════════════════════════════════════════════════════════════════════

st.subheader("🟢 Long Signal Candidates — Ranked by Proximity to SSL (Stop Level)")
st.caption(
    "**SSL** = nearest unswept hourly swing low = stop-loss level for longs.  "
    "Lower % = price is closer to its stop — tightest risk / most actionable setup."
)

long_rows = [r for r in results if r["buy_signal"] and not np.isnan(r["pct_to_ssl"])]
long_rows.sort(key=lambda x: x["pct_to_ssl"])

if not long_rows:
    near_long = [r for r in results
                 if r["long_score"] >= 3 and not r["buy_signal"]
                 and not r["weekly_neutral"]]
    near_long.sort(key=lambda x: (x["long_score"], -(x["pct_to_ssl"] or 999)),
                   reverse=True)

    st.info("No full long signals. Showing near-miss coins (3/4 conditions met).")
    near_long = near_long[:top_n]

    if near_long:
        def _long_row(r):
            conds = r["conditions_long"]
            missing = [k for k, v in conds.items() if not v]
            return {
                "Coin":           r["symbol"],
                "Price":          f"{r['price']:.4f}",
                "SSL Level":      f"{r['h_ssl_level']:.4f}" if not np.isnan(r['h_ssl_level']) else "—",
                "% to SSL":       f"{r['pct_to_ssl']:.2f}%" if not np.isnan(r['pct_to_ssl']) else "—",
                "Daily BSL":      f"{r['d_bsl_level']:.4f}" if not np.isnan(r['d_bsl_level']) else "—",
                "H-EQ":           f"{r['h_equilibrium']:.4f}",
                "D-Levels":       f"{r['d_bsl_count']}/{r['d_ssl_count']}",
                "Conditions Met": f"{r['long_score']}/4",
                "Missing":        ", ".join(missing),
            }
        st.dataframe(
            pd.DataFrame([_long_row(r) for r in near_long]),
            use_container_width=True, hide_index=True
        )
else:
    long_rows = long_rows[:top_n]

    def _long_signal_row(r):
        conds = r["conditions_long"]
        checks = " ".join("✅" if v else "❌" for v in conds.values())
        return {
            "Coin":          r["symbol"],
            "Price":         f"{r['price']:.4f}",
            "SSL (Stop)":    f"{r['h_ssl_level']:.4f}" if not np.isnan(r['h_ssl_level']) else "—",
            "% to SSL ▲":    round(r["pct_to_ssl"], 3),
            "Daily BSL Tgt": f"{r['d_bsl_level']:.4f}" if not np.isnan(r['d_bsl_level']) else "—",
            "R:R to BSL":    (
                f"{(r['d_bsl_level'] - r['price']) / (r['price'] - r['h_ssl_level']):.1f}x"
                if (not np.isnan(r['d_bsl_level']) and not np.isnan(r['h_ssl_level'])
                    and r['price'] > r['h_ssl_level'])
                else "—"
            ),
            "H-EQ":          f"{r['h_equilibrium']:.4f}",
            "D-Levels":      f"{r['d_bsl_count']}/{r['d_ssl_count']}",
            "Checks":        checks,
        }

    df_long = pd.DataFrame([_long_signal_row(r) for r in long_rows])

    def _color_pct(val):
        try:
            v = float(val)
            if v < 2:   return "background-color: #0d3b1f; color: #4ade80"
            if v < 5:   return "background-color: #1a3b1a; color: #86efac"
            if v < 10:  return "background-color: #2a2a0d; color: #fde68a"
            return "background-color: #3b0d0d; color: #fca5a5"
        except Exception:
            return ""

    styled_long = df_long.style.map(_color_pct, subset=["% to SSL ▲"])
    st.dataframe(styled_long, use_container_width=True, hide_index=True)

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# SHORT SIGNAL TABLE
# Ranked by % distance to BSL (stop-loss for shorts)
# ═════════════════════════════════════════════════════════════════════════════

st.subheader("🔴 Short Signal Candidates — Ranked by Proximity to BSL (Stop Level)")
st.caption(
    "**BSL** = nearest unswept hourly swing high = stop-loss level for shorts.  "
    "Lower % = price is closer to its stop — tightest risk / most actionable setup."
)

short_rows = [r for r in results if r["sell_signal"] and not np.isnan(r["pct_to_bsl"])]
short_rows.sort(key=lambda x: x["pct_to_bsl"])

if not short_rows:
    near_short = [r for r in results
                  if r["short_score"] >= 3 and not r["sell_signal"]
                  and not r["weekly_neutral"]]
    near_short.sort(key=lambda x: x["short_score"], reverse=True)

    st.info("No full short signals. Showing near-miss coins (3/4 conditions met).")
    near_short = near_short[:top_n]

    if near_short:
        def _short_near_row(r):
            conds = r["conditions_short"]
            missing = [k for k, v in conds.items() if not v]
            return {
                "Coin":           r["symbol"],
                "Price":          f"{r['price']:.4f}",
                "BSL Level":      f"{r['h_bsl_level']:.4f}" if not np.isnan(r['h_bsl_level']) else "—",
                "% to BSL":       f"{r['pct_to_bsl']:.2f}%" if not np.isnan(r['pct_to_bsl']) else "—",
                "Daily SSL":      f"{r['d_ssl_level']:.4f}" if not np.isnan(r['d_ssl_level']) else "—",
                "H-EQ":           f"{r['h_equilibrium']:.4f}",
                "D-Levels":       f"{r['d_bsl_count']}/{r['d_ssl_count']}",
                "Conditions Met": f"{r['short_score']}/4",
                "Missing":        ", ".join(missing),
            }
        st.dataframe(
            pd.DataFrame([_short_near_row(r) for r in near_short]),
            use_container_width=True, hide_index=True
        )
else:
    short_rows = short_rows[:top_n]

    def _short_signal_row(r):
        conds = r["conditions_short"]
        checks = " ".join("✅" if v else "❌" for v in conds.values())
        return {
            "Coin":          r["symbol"],
            "Price":         f"{r['price']:.4f}",
            "BSL (Stop)":    f"{r['h_bsl_level']:.4f}" if not np.isnan(r['h_bsl_level']) else "—",
            "% to BSL ▲":    round(r["pct_to_bsl"], 3),
            "Daily SSL Tgt": f"{r['d_ssl_level']:.4f}" if not np.isnan(r['d_ssl_level']) else "—",
            "R:R to SSL":    (
                f"{(r['price'] - r['d_ssl_level']) / (r['h_bsl_level'] - r['price']):.1f}x"
                if (not np.isnan(r['d_ssl_level']) and not np.isnan(r['h_bsl_level'])
                    and r['h_bsl_level'] > r['price'])
                else "—"
            ),
            "H-EQ":          f"{r['h_equilibrium']:.4f}",
            "D-Levels":      f"{r['d_bsl_count']}/{r['d_ssl_count']}",
            "Checks":        checks,
        }

    df_short = pd.DataFrame([_short_signal_row(r) for r in short_rows])

    def _color_pct_s(val):
        try:
            v = float(val)
            if v < 2:   return "background-color: #3b0d0d; color: #f87171"
            if v < 5:   return "background-color: #3b1a0d; color: #fca5a5"
            if v < 10:  return "background-color: #2a2a0d; color: #fde68a"
            return "background-color: #1a1a2e; color: #94a3b8"
        except Exception:
            return ""

    styled_short = df_short.style.map(_color_pct_s, subset=["% to BSL ▲"])
    st.dataframe(styled_short, use_container_width=True, hide_index=True)

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# FULL CONDITION BREAKDOWN — expander
# ═════════════════════════════════════════════════════════════════════════════

with st.expander("📋 Full Condition Breakdown (All Coins)", expanded=False):
    st.caption(
        "All 4 ICT conditions per coin. Neutral-week coins have greyed-out bias columns. "
        "Sort by score to find near-miss setups."
    )

    def _breakdown_row(r):
        cl = r["conditions_long"]
        cs = r["conditions_short"]
        signal_str = ""
        if r["buy_signal"]:
            signal_str += "🟢 LONG "
        if r["sell_signal"]:
            signal_str += "🔴 SHORT"
        if r["weekly_neutral"]:
            signal_str = "⚪ NEUTRAL"
        if not signal_str:
            signal_str = "—"

        return {
            "Coin":        r["symbol"],
            "Price":       f"{r['price']:.4f}",
            "Signal":      signal_str,
            # Long checks
            "W Bull":      "✅" if cl["Weekly Bullish"]     else ("⚪" if r["weekly_neutral"] else "❌"),
            "D Draw→BSL":  "✅" if cl["Daily Draw (→ BSL)"] else "❌",
            "H Discount":  "✅" if cl["H Discount Zone"]    else "❌",
            "SSL Raid":    "✅" if cl["SSL Raid Recent"]    else "❌",
            "L Score":     r["long_score"],
            # Short checks
            "W Bear":      "✅" if cs["Weekly Bearish"]     else ("⚪" if r["weekly_neutral"] else "❌"),
            "D Draw→SSL":  "✅" if cs["Daily Draw (→ SSL)"] else "❌",
            "H Premium":   "✅" if cs["H Premium Zone"]     else "❌",
            "BSL Raid":    "✅" if cs["BSL Raid Recent"]    else "❌",
            "S Score":     r["short_score"],
            # Levels
            "H-SSL":       f"{r['h_ssl_level']:.4f}" if not np.isnan(r['h_ssl_level']) else "—",
            "H-BSL":       f"{r['h_bsl_level']:.4f}" if not np.isnan(r['h_bsl_level']) else "—",
            "H-EQ":        f"{r['h_equilibrium']:.4f}",
            "D-Liq":       f"{r['d_bsl_count']}/{r['d_ssl_count']}",
        }

    df_all = pd.DataFrame([_breakdown_row(r) for r in results])
    df_all = df_all.sort_values(
        ["L Score", "S Score"], ascending=False
    ).reset_index(drop=True)

    st.dataframe(df_all, use_container_width=True, hide_index=True)
