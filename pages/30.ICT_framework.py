# ict_framework.py
# ICT Decision Framework — Python/Streamlit adaptation
# Adapted from PineScript ICT MTF v6 by stripping sub-hourly constructs (MSS, FVG, OB)
# Lowest supported timeframe: 1H
#
# SIGNAL LOGIC (per coin):
#   LONG  : weekly_bull AND daily_draw_bull AND h_discount AND ssl_raid_recent
#   SHORT : weekly_bear AND daily_draw_bear AND h_premium  AND bsl_raid_recent
#
# OUTPUT: Coins ranked by proximity to their stop-loss level
#   Long  candidates → ranked by % distance to SSL (stop below recent swing low)
#   Short candidates → ranked by % distance to BSL (stop above recent swing high)

import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="ICT Framework Scanner", layout="wide")

st.markdown("""
<style>
  /* Dark trading terminal feel */
  .main { background: #0a0a0f; }
  .block-container { padding-top: 1.5rem; }

  /* Signal cards */
  .sig-header { font-family: 'Courier New', monospace; letter-spacing: 2px; }

  /* Metric cells */
  div[data-testid="metric-container"] {
    background: #111120;
    border: 1px solid #2a2a3e;
    border-radius: 6px;
    padding: 12px;
  }
</style>
""", unsafe_allow_html=True)

st.title("📐 ICT Decision Framework — Multi-Coin Scanner")
st.caption("Weekly bias → Daily draw → Hourly P/D → Hourly stop hunt. Lowest TF: 1H.")

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar parameters
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ ICT Parameters")

    st.subheader("① Weekly Bias")
    bias_ema_len = st.slider("Weekly EMA Length", 5, 50, 20)

    st.subheader("② Daily Liquidity Draw")
    daily_swing = st.slider("Daily Swing Lookback (bars)", 2, 20, 5)

    st.subheader("③ Hourly P/D + Stop Hunt")
    pd_lookback = st.slider("Hourly Range Lookback (bars)", 20, 200, 50)
    htf_swing   = st.slider("Hourly Swing Lookback (bars)", 2, 20, 5)
    raid_window = st.slider("Raid Valid Window (bars)", 2, 30, 8,
                            help="How many 1H bars a stop-hunt stays 'recent'")

    st.subheader("④ Filters")
    require_pd    = st.checkbox("Require Hourly P/D Confirmation", True)
    require_daily = st.checkbox("Require Daily Draw Alignment",     True)

    st.subheader("⑤ Display")
    top_n = st.slider("Top N coins per side", 5, 30, 15)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def pivot_high(high: np.ndarray, left: int, right: int) -> np.ndarray:
    """Return array of pivot-high values (NaN where not a pivot)."""
    n = len(high)
    out = np.full(n, np.nan)
    for i in range(left, n - right):
        window = high[i - left: i + right + 1]
        if high[i] == window.max():
            out[i] = high[i]
    return out


def pivot_low(low: np.ndarray, left: int, right: int) -> np.ndarray:
    """Return array of pivot-low values (NaN where not a pivot)."""
    n = len(low)
    out = np.full(n, np.nan)
    for i in range(left, n - right):
        window = low[i - left: i + right + 1]
        if low[i] == window.min():
            out[i] = low[i]
    return out


def last_valid(arr: np.ndarray):
    """Return last non-NaN value, or NaN if none."""
    idx = np.where(~np.isnan(arr))[0]
    return arr[idx[-1]] if len(idx) else np.nan


def last_valid_idx(arr: np.ndarray):
    """Return index of last non-NaN value, or -1."""
    idx = np.where(~np.isnan(arr))[0]
    return int(idx[-1]) if len(idx) else -1


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
        idx = ohlc_multi.index  # DatetimeIndex at base TF (1H/4H/1D)

        if len(c) < max(bias_ema_len * 5, pd_lookback + 20):
            return None

        # ─────────────────────────────────────────────────────────────────
        # SECTION 1 — WEEKLY BIAS
        # Resample OHLC to weekly; compute EMA of weekly close; check slope
        # ─────────────────────────────────────────────────────────────────
        df_raw = pd.DataFrame({"o": o, "h": h, "l": l, "c": c}, index=idx)

        df_weekly = df_raw["c"].resample("W").last().dropna()
        if len(df_weekly) < bias_ema_len + 2:
            return None

        w_ema = df_weekly.ewm(span=bias_ema_len, adjust=False).mean()
        weekly_bull = bool(w_ema.iloc[-1] > w_ema.iloc[-2])
        weekly_bear = not weekly_bull

        # ─────────────────────────────────────────────────────────────────
        # SECTION 2 — DAILY LIQUIDITY DRAW
        # Resample to daily; find most recent daily swing high (BSL) and low (SSL)
        # ─────────────────────────────────────────────────────────────────
        df_daily_h = df_raw["h"].resample("D").max().dropna()
        df_daily_l = df_raw["l"].resample("D").min().dropna()
        df_daily_c = df_raw["c"].resample("D").last().dropna()

        if len(df_daily_h) < daily_swing * 2 + 5:
            return None

        d_ph = pivot_high(df_daily_h.values, daily_swing, daily_swing)
        d_pl = pivot_low(df_daily_l.values,  daily_swing, daily_swing)

        d_bsl_level = last_valid(d_ph)   # most recent daily swing high
        d_ssl_level = last_valid(d_pl)   # most recent daily swing low
        d_close_last = df_daily_c.iloc[-1]

        # Draw-on-liquidity alignment
        #   Bull: price hasn't yet reached daily BSL (swing high)
        #   Bear: price hasn't yet reached daily SSL (swing low)
        daily_draw_bull = (not np.isnan(d_bsl_level)) and (d_close_last < d_bsl_level)
        daily_draw_bear = (not np.isnan(d_ssl_level)) and (d_close_last > d_ssl_level)
        daily_aligned_bull = daily_draw_bull if require_daily else True
        daily_aligned_bear = daily_draw_bear if require_daily else True

        # ─────────────────────────────────────────────────────────────────
        # SECTION 3 — HOURLY PREMIUM / DISCOUNT + STOP HUNT
        # Use base-TF (1H or whatever was fetched) arrays directly.
        # If TF is daily/4H the "hourly" logic still applies at that TF.
        # ─────────────────────────────────────────────────────────────────
        n = len(c)

        # Range high/low over lookback → equilibrium
        lb = min(pd_lookback, n - 1)
        h_range_high = np.max(h[-lb:])
        h_range_low  = np.min(l[-lb:])
        h_equilibrium = (h_range_high + h_range_low) / 2.0

        h_close_last = c[-1]
        h_high_last  = h[-1]
        h_low_last   = l[-1]

        h_premium  = h_close_last > h_equilibrium
        h_discount = h_close_last < h_equilibrium

        # Hourly swing pivots
        h_ph = pivot_high(h, htf_swing, htf_swing)
        h_pl = pivot_low(l,  htf_swing, htf_swing)

        h_swing_high = last_valid(h_ph)
        h_swing_low  = last_valid(h_pl)
        h_swing_high_idx = last_valid_idx(h_ph)
        h_swing_low_idx  = last_valid_idx(h_pl)

        # Stop hunt on each historical bar → track last raid bar index
        bsl_raid_bar = -9999
        ssl_raid_bar = -9999

        # Vectorised: wick above swing high AND close back below = BSL raided
        # We scan the last (raid_window * 4 + htf_swing * 2) bars for efficiency
        scan_start = max(0, n - raid_window * 4 - htf_swing * 2 - 10)

        running_sh = np.nan
        running_sl = np.nan

        for i in range(scan_start, n):
            if not np.isnan(h_ph[i]):
                running_sh = h_ph[i]
            if not np.isnan(h_pl[i]):
                running_sl = h_pl[i]

            if not np.isnan(running_sh):
                if h[i] > running_sh and c[i] < running_sh:
                    bsl_raid_bar = i

            if not np.isnan(running_sl):
                if l[i] < running_sl and c[i] > running_sl:
                    ssl_raid_bar = i

        bsl_raid_recent = (n - 1 - bsl_raid_bar) <= raid_window
        ssl_raid_recent = (n - 1 - ssl_raid_bar) <= raid_window

        # P/D gating
        pd_sell_ok = h_premium  if require_pd else True
        pd_buy_ok  = h_discount if require_pd else True

        # ─────────────────────────────────────────────────────────────────
        # SECTION 4 — COMPOSITE SIGNALS (no MSS/FVG/OB — hourly+ only)
        # ─────────────────────────────────────────────────────────────────
        buy_signal  = (weekly_bull and daily_aligned_bull
                       and pd_buy_ok and ssl_raid_recent)

        sell_signal = (weekly_bear and daily_aligned_bear
                       and pd_sell_ok and bsl_raid_recent)

        # ─────────────────────────────────────────────────────────────────
        # RANKING METRIC — proximity to stop-loss level
        #   Long  stop → below recent swing low (h_swing_low = SSL)
        #   Short stop → above recent swing high (h_swing_high = BSL)
        #
        # pct_to_sl:
        #   Long  → (close - ssl) / close  — smaller = closer to danger / better entry setup
        #   Short → (bsl - close) / close  — smaller = closer to danger / better setup
        # ─────────────────────────────────────────────────────────────────
        price = h_close_last

        if not np.isnan(h_swing_low) and price > 0:
            pct_to_ssl = (price - h_swing_low) / price * 100.0
        else:
            pct_to_ssl = np.nan

        if not np.isnan(h_swing_high) and price > 0:
            pct_to_bsl = (h_swing_high - price) / price * 100.0
        else:
            pct_to_bsl = np.nan

        # Condition check breakdown for dashboard table
        conditions_long = {
            "Weekly Bullish":      weekly_bull,
            "Daily Draw (→ BSL)":  daily_draw_bull,
            "H Discount Zone":     h_discount,
            "SSL Raid Recent":     ssl_raid_recent,
        }
        conditions_short = {
            "Weekly Bearish":      weekly_bear,
            "Daily Draw (→ SSL)":  daily_draw_bear,
            "H Premium Zone":      h_premium,
            "BSL Raid Recent":     bsl_raid_recent,
        }

        long_score  = sum(conditions_long.values())
        short_score = sum(conditions_short.values())

        return {
            "symbol":          sym,
            "price":           price,
            # Signals
            "buy_signal":      buy_signal,
            "sell_signal":     sell_signal,
            # Levels
            "d_bsl_level":     d_bsl_level,
            "d_ssl_level":     d_ssl_level,
            "h_swing_high":    h_swing_high,
            "h_swing_low":     h_swing_low,
            "h_equilibrium":   h_equilibrium,
            # Proximity
            "pct_to_ssl":      pct_to_ssl,   # for long ranking (lower = closer)
            "pct_to_bsl":      pct_to_bsl,   # for short ranking (lower = closer)
            # Weekly / conditions
            "weekly_bull":     weekly_bull,
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

st.info(f"Running ICT scanner on **{len(symbols)} coins** | Timeframe: `{timeframe_label}` | "
        f"{len(ohlc_multi)} bars per coin")

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
n_buy  = sum(1 for r in results if r["buy_signal"])
n_sell = sum(1 for r in results if r["sell_signal"])
n_both = sum(1 for r in results if r["buy_signal"] and r["sell_signal"])

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Scanned",   len(results))
m2.metric("🟢 Long Signals",  n_buy)
m3.metric("🔴 Short Signals", n_sell)
m4.metric("⚡ Both Signals",  n_both)

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# LONG SIGNAL TABLE
# Ranked by % distance to SSL (stop-loss for longs = swing low)
# Closest SSL first = highest-priority / tightest risk setup
# ═════════════════════════════════════════════════════════════════════════════

st.subheader("🟢 Long Signal Candidates — Ranked by Proximity to SSL (Stop Level)")
st.caption(
    "**SSL** = most recent hourly swing low = stop-loss level for long trades.  "
    "Lower % = price is closer to its stop — tightest risk / most actionable setup."
)

long_rows = [r for r in results if r["buy_signal"] and not np.isnan(r["pct_to_ssl"])]
long_rows.sort(key=lambda x: x["pct_to_ssl"])

if not long_rows:
    # Show near-miss: coins with 3/4 long conditions met
    near_long = [r for r in results if r["long_score"] >= 3 and not r["buy_signal"]]
    near_long.sort(key=lambda x: (x["long_score"], -(x["pct_to_ssl"] or 999)), reverse=True)

    st.info("No full long signals. Showing near-miss coins (3/4 conditions met).")
    near_long = near_long[:top_n]

    if near_long:
        def _long_row(r):
            conds = r["conditions_long"]
            missing = [k for k, v in conds.items() if not v]
            return {
                "Coin":            r["symbol"],
                "Price":           f"{r['price']:.4f}",
                "SSL Level":       f"{r['h_swing_low']:.4f}" if not np.isnan(r['h_swing_low']) else "—",
                "% to SSL":        f"{r['pct_to_ssl']:.2f}%" if not np.isnan(r['pct_to_ssl']) else "—",
                "Daily BSL":       f"{r['d_bsl_level']:.4f}" if not np.isnan(r['d_bsl_level']) else "—",
                "H-EQ":            f"{r['h_equilibrium']:.4f}",
                "Conditions Met":  f"{r['long_score']}/4",
                "Missing":         ", ".join(missing),
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
            "SSL (Stop)":    f"{r['h_swing_low']:.4f}" if not np.isnan(r['h_swing_low']) else "—",
            "% to SSL ▲":    round(r["pct_to_ssl"], 3),
            "Daily BSL Tgt": f"{r['d_bsl_level']:.4f}" if not np.isnan(r['d_bsl_level']) else "—",
            "R:R to BSL":    (
                f"{(r['d_bsl_level'] - r['price']) / (r['price'] - r['h_swing_low']):.1f}x"
                if (not np.isnan(r['d_bsl_level']) and not np.isnan(r['h_swing_low'])
                    and r['price'] > r['h_swing_low'])
                else "—"
            ),
            "H-EQ":          f"{r['h_equilibrium']:.4f}",
            "Checks":        checks,
        }

    df_long = pd.DataFrame([_long_signal_row(r) for r in long_rows])

    # Colour-code % to SSL: green = tight = good
    def _color_pct(val):
        try:
            v = float(val)
            if v < 2:   return "background-color: #0d3b1f; color: #4ade80"
            if v < 5:   return "background-color: #1a3b1a; color: #86efac"
            if v < 10:  return "background-color: #2a2a0d; color: #fde68a"
            return "background-color: #3b0d0d; color: #fca5a5"
        except Exception:
            return ""

    styled_long = df_long.style.applymap(_color_pct, subset=["% to SSL ▲"])
    st.dataframe(styled_long, use_container_width=True, hide_index=True)

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# SHORT SIGNAL TABLE
# Ranked by % distance to BSL (stop-loss for shorts = swing high)
# ═════════════════════════════════════════════════════════════════════════════

st.subheader("🔴 Short Signal Candidates — Ranked by Proximity to BSL (Stop Level)")
st.caption(
    "**BSL** = most recent hourly swing high = stop-loss level for short trades.  "
    "Lower % = price is closer to its stop — tightest risk / most actionable setup."
)

short_rows = [r for r in results if r["sell_signal"] and not np.isnan(r["pct_to_bsl"])]
short_rows.sort(key=lambda x: x["pct_to_bsl"])

if not short_rows:
    near_short = [r for r in results if r["short_score"] >= 3 and not r["sell_signal"]]
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
                "BSL Level":      f"{r['h_swing_high']:.4f}" if not np.isnan(r['h_swing_high']) else "—",
                "% to BSL":       f"{r['pct_to_bsl']:.2f}%" if not np.isnan(r['pct_to_bsl']) else "—",
                "Daily SSL":      f"{r['d_ssl_level']:.4f}" if not np.isnan(r['d_ssl_level']) else "—",
                "H-EQ":           f"{r['h_equilibrium']:.4f}",
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
            "BSL (Stop)":    f"{r['h_swing_high']:.4f}" if not np.isnan(r['h_swing_high']) else "—",
            "% to BSL ▲":    round(r["pct_to_bsl"], 3),
            "Daily SSL Tgt": f"{r['d_ssl_level']:.4f}" if not np.isnan(r['d_ssl_level']) else "—",
            "R:R to SSL":    (
                f"{(r['price'] - r['d_ssl_level']) / (r['h_swing_high'] - r['price']):.1f}x"
                if (not np.isnan(r['d_ssl_level']) and not np.isnan(r['h_swing_high'])
                    and r['h_swing_high'] > r['price'])
                else "—"
            ),
            "H-EQ":          f"{r['h_equilibrium']:.4f}",
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

    styled_short = df_short.style.applymap(_color_pct_s, subset=["% to BSL ▲"])
    st.dataframe(styled_short, use_container_width=True, hide_index=True)

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# FULL CONDITION BREAKDOWN — expander
# ═════════════════════════════════════════════════════════════════════════════

with st.expander("📋 Full Condition Breakdown (All Coins)", expanded=False):
    st.caption("Shows all 4 ICT conditions for every coin. Sort by score to find near-miss setups.")

    def _breakdown_row(r):
        cl = r["conditions_long"]
        cs = r["conditions_short"]
        return {
            "Coin":             r["symbol"],
            "Price":            f"{r['price']:.4f}",
            "Signal":           ("🟢 LONG" if r["buy_signal"] else "") +
                                ("🔴 SHORT" if r["sell_signal"] else "") or "—",
            # Long checks
            "W Bull":           "✅" if cl["Weekly Bullish"]     else "❌",
            "D Draw→BSL":       "✅" if cl["Daily Draw (→ BSL)"] else "❌",
            "H Discount":       "✅" if cl["H Discount Zone"]    else "❌",
            "SSL Raid":         "✅" if cl["SSL Raid Recent"]    else "❌",
            "Long Score":       r["long_score"],
            # Short checks
            "W Bear":           "✅" if cs["Weekly Bearish"]     else "❌",
            "D Draw→SSL":       "✅" if cs["Daily Draw (→ SSL)"] else "❌",
            "H Premium":        "✅" if cs["H Premium Zone"]     else "❌",
            "BSL Raid":         "✅" if cs["BSL Raid Recent"]    else "❌",
            "Short Score":      r["short_score"],
            # Key levels
            "H-SSL":            f"{r['h_swing_low']:.4f}"  if not np.isnan(r['h_swing_low'])  else "—",
            "H-BSL":            f"{r['h_swing_high']:.4f}" if not np.isnan(r['h_swing_high']) else "—",
            "H-EQ":             f"{r['h_equilibrium']:.4f}",
        }

    df_all = pd.DataFrame([_breakdown_row(r) for r in results])
    df_all = df_all.sort_values(
        ["Long Score", "Short Score"], ascending=False
    ).reset_index(drop=True)

    st.dataframe(df_all, use_container_width=True, hide_index=True)
