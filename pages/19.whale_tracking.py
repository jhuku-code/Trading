"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           CRYPTO WHALE TRACKER  —  Hyperliquid Perpetuals                  ║
║           Single-file Streamlit page  (drop into pages/ folder)            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Data source : Hyperliquid public API — free, no auth required             ║
║                https://api.hyperliquid.xyz/info                            ║
║                https://stats-data.hyperliquid.xyz/Mainnet/leaderboard      ║
║  Strategy    : Elite-only signals (top N wallets by monthly PnL)           ║
║                Two independent signal methods:                             ║
║                  Binary   — % of elite wallets net long                    ║
║                  Notional — % of elite USD notional that is long           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  SETUP                                                                     ║
║  pip install streamlit pandas numpy plotly requests openpyxl               ║
║                                                                            ║
║  Standalone : streamlit run whale_tracker.py                               ║
║  Multi-page : copy to   pages/whale_tracker.py                             ║
║               Comment out st.set_page_config() if your main app.py        ║
║               already calls it.                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import io
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Crypto Whale Tracker",
    page_icon="🐋",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — THEME
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');
:root {
    --bg:#0a0e1a; --card:#111827; --border:#1e3a5f; --text:#e2e8f0;
    --muted:#64748b; --green:#00ff88; --red:#ff4444; --yellow:#fbbf24; --blue:#38bdf8;
}
html,body,[class*="css"]         { font-family:'Inter',sans-serif; }
.stApp                           { background-color:var(--bg); }
section[data-testid="stSidebar"]{ background:#0d1117; border-right:1px solid var(--border); }
.wt-header { background:linear-gradient(90deg,#0a0e1a 0%,#0d1f3c 50%,#0a0e1a 100%);
             border-bottom:1px solid var(--border); padding:18px 24px 14px;
             margin-bottom:24px; border-radius:0 0 12px 12px; }
.wt-header h1 { font-family:'Space Mono',monospace; font-size:1.9rem;
                letter-spacing:2px; color:var(--green); margin:0; }
.wt-header p  { color:var(--muted); font-size:0.82rem; margin:4px 0 0; letter-spacing:1px; }
.wt-section   { font-family:'Space Mono',monospace; font-size:0.82rem; letter-spacing:2px;
                color:var(--muted); text-transform:uppercase; border-bottom:1px solid var(--border);
                padding-bottom:6px; margin:24px 0 14px; }
.wt-kpi       { background:var(--card); border:1px solid var(--border);
                border-radius:10px; padding:16px 20px; text-align:center; }
.wt-kpi .val  { font-family:'Space Mono',monospace; font-size:1.65rem; font-weight:700; line-height:1.1; }
.wt-kpi .lbl  { font-size:0.70rem; color:var(--muted); letter-spacing:1.4px; text-transform:uppercase; margin-top:5px; }
.wt-coin-card { background:var(--card); border:1px solid #064e3b; border-radius:10px; padding:20px; }
.wt-coin-card table { width:100%; border-collapse:collapse; font-size:0.85rem; }
.wt-coin-card td    { padding:5px 0; }
.wt-mock-banner { background:#1c1a00; border:1px solid #854d0e; border-radius:8px;
                  padding:10px 16px; margin-bottom:18px; color:var(--yellow); font-size:0.82rem; }
.wt-disclaimer  { font-size:0.70rem; color:#475569; line-height:1.7; }
.stDataFrame    { border-radius:8px; overflow:hidden; }
div[data-testid="stDownloadButton"]>button {
    background:#0d1f3c!important; border:1px solid #1e4d8c!important;
    color:#38bdf8!important; font-family:'Space Mono',monospace!important;
    font-size:0.78rem!important; border-radius:6px!important; padding:6px 14px!important; }
div[data-testid="stDownloadButton"]>button:hover {
    background:#1a3a6e!important; border-color:#38bdf8!important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════

_DEFAULT_N_ELITE     = 20
_DEFAULT_MIN_WALLETS = 3
_DEFAULT_DIV_THRESH  = 20   # LONG >70%  SHORT <30%


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — HYPERLIQUID API
# ══════════════════════════════════════════════════════════════════════════════

_INFO_URL           = "https://api.hyperliquid.xyz/info"
_STATS_URL          = "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard"
_STATS_WINDOWS      = ["month", "week", "allTime", "day"]
_HEADERS            = {"Content-Type": "application/json"}
_BATCH              = 10
_DELAY              = 0.25
_MIN_LB_ROWS        = 50
MIN_NOTIONAL_USD    = 500
MAX_LEVERAGE_SIGNAL = 25
MIN_PARTICIPATION   = 20

_last_api_error: str = ""


def _post(payload: dict, timeout: int = 20):
    global _last_api_error
    try:
        r = requests.post(_INFO_URL, json=payload, headers=_HEADERS, timeout=timeout)
        r.raise_for_status()
        _last_api_error = ""
        return r.json()
    except requests.exceptions.HTTPError as e:
        _last_api_error = f"HTTP {e.response.status_code}: {e.response.text[:120]}"
    except requests.exceptions.ConnectionError as e:
        _last_api_error = f"Connection error: {str(e)[:120]}"
    except requests.exceptions.Timeout:
        _last_api_error = "Request timed out"
    except Exception as e:
        _last_api_error = f"{type(e).__name__}: {str(e)[:120]}"
    return None


def _get(url: str, timeout: int = 20):
    global _last_api_error
    try:
        hdrs = {**_HEADERS, "Accept": "application/json", "User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=hdrs, timeout=timeout)
        r.raise_for_status()
        _last_api_error = ""
        return r.json()
    except Exception as e:
        _last_api_error = f"{type(e).__name__}: {str(e)[:120]}"
    return None


def fetch_leaderboard() -> tuple:
    """Returns (DataFrame[address, pnl, account_value], is_live: bool)."""
    global _last_api_error

    def _parse(data) -> pd.DataFrame:
        if not data:
            return pd.DataFrame()
        rows_raw = (data.get("leaderboardRows") if isinstance(data, dict)
                    else data if isinstance(data, list) else None)
        if not rows_raw:
            return pd.DataFrame()
        rows = [{"address":       r.get("ethAddress", r.get("address", "")),
                 "pnl":           float(r.get("pnl", r.get("windowPnl", 0))),
                 "account_value": float(r.get("accountValue", 0))}
                for r in rows_raw if r.get("ethAddress") or r.get("address")]
        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        return df[df["address"] != ""].copy() if not df.empty else df

    for url in [_STATS_URL] + [f"{_STATS_URL}?window={w}" for w in _STATS_WINDOWS]:
        df = _parse(_get(url))
        if not df.empty and len(df) >= _MIN_LB_ROWS:
            return df.sort_values("pnl", ascending=False).reset_index(drop=True), True

    if not df.empty:
        _last_api_error = f"Leaderboard only {len(df)} rows. Falling back to demo."
    return _mock_leaderboard(), False


def fetch_positions(address: str, for_signal: bool = True) -> list:
    data = _post({"type": "clearinghouseState", "user": address})
    if not data or "assetPositions" not in data:
        return []
    out = []
    for ap in data.get("assetPositions", []):
        p        = ap.get("position", {})
        size     = float(p.get("szi", 0))
        if size == 0:
            continue
        lev_raw  = p.get("leverage", {})
        leverage = (float(lev_raw.get("value", lev_raw.get("rawUsd", 1)))
                    if isinstance(lev_raw, dict) else float(lev_raw or 1))
        notional = abs(float(p.get("positionValue", 0)))
        if for_signal and (notional < MIN_NOTIONAL_USD or leverage > MAX_LEVERAGE_SIGNAL):
            continue
        out.append({
            "coin":           p.get("coin", "UNKNOWN"),
            "direction":      "long" if size > 0 else "short",
            "size":           size,
            "entry_px":       float(p.get("entryPx", 0)),
            "unrealized_pnl": float(p.get("unrealizedPnl", 0)),
            "leverage":       round(leverage, 1),
            "notional":       notional,
        })
    return out


def fetch_funding_rates() -> dict:
    data = _post({"type": "metaAndAssetCtxs"})
    if not data or not isinstance(data, list) or len(data) < 2:
        return {}
    universe = data[0].get("universe", [])
    return {universe[i].get("name", ""): float(ctx.get("funding", 0)) * 100
            for i, ctx in enumerate(data[1]) if i < len(universe)}


def fetch_prices() -> dict:
    data = _post({"type": "allMids"})
    return {k: float(v) for k, v in data.items()} if isinstance(data, dict) else {}


def _build_position_matrix(addresses: list) -> tuple:
    """Returns (dir_df, not_df, signal_raw, display_raw)."""
    signal_raw, display_raw, all_coins = {}, {}, set()
    for i, addr in enumerate(addresses):
        sig_pos           = fetch_positions(addr, for_signal=True)
        disp_pos          = fetch_positions(addr, for_signal=False)
        signal_raw[addr]  = sig_pos
        display_raw[addr] = disp_pos
        all_coins.update(p["coin"] for p in sig_pos)
        if (i + 1) % _BATCH == 0:
            time.sleep(_DELAY)

    if not all_coins:
        empty = pd.DataFrame(index=addresses)
        return empty, empty, signal_raw, display_raw

    coins = sorted(all_coins)
    dir_rows, not_rows = {}, {}
    for addr, positions in signal_raw.items():
        pos_map = {p["coin"]: p for p in positions}
        dir_rows[addr] = {c: (1 if pos_map[c]["direction"] == "long" else -1)
                           if c in pos_map else 0 for c in coins}
        not_rows[addr] = {c: pos_map[c]["notional"] if c in pos_map else 0.0
                           for c in coins}
    return (pd.DataFrame(dir_rows, index=coins).T,
            pd.DataFrame(not_rows, index=coins).T,
            signal_raw, display_raw)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — SIGNAL ENGINE
# ══════════════════════════════════════════════════════════════════════════════

_SIG_COLS = [
    "coin", "signal_binary", "signal_notional",
    "elite_wallets", "elite_long_pct",
    "long_notional_usd", "short_notional_usd", "net_notional_pct",
    "total_notional_usd", "funding_rate_8h_pct", "price_usd",
]


def _empty_signals() -> pd.DataFrame:
    return pd.DataFrame(columns=_SIG_COLS)


def compute_signals(
    leaderboard: pd.DataFrame,
    n_elite: int,
    min_wallets: int,
    div_threshold: int,
    funding_rates: dict = None,
    prices: dict = None,
    use_mock: bool = False,
) -> tuple:
    """
    Elite-only, two-signal computation.

    Binary signal   : % of elite wallets net long
      LONG  if elite_long_pct > 50 + div_threshold  (default >70%)
      SHORT if elite_long_pct < 50 - div_threshold  (default <30%)

    Notional signal : % of elite USD notional that is long
      LONG  if net_notional_pct > 50 + div_threshold
      SHORT if net_notional_pct < 50 - div_threshold
    """
    if use_mock:
        return _mock_signals(div_threshold), _mock_meta(n_elite)

    t0          = time.time()
    e_addrs     = leaderboard.head(n_elite)["address"].tolist()
    e_dir, e_not, e_sig_raw, e_disp_raw = _build_position_matrix(e_addrs)

    e_active      = sum(1 for v in e_sig_raw.values() if v)
    participation = e_active / max(len(e_addrs), 1) * 100
    funding       = funding_rates or {}
    px            = prices or {}
    lo            = 50 + div_threshold
    hi            = 50 - div_threshold
    all_coins     = list(e_dir.columns) if not e_dir.empty else []
    rows          = []

    for coin in all_coins:
        e_col = e_dir[coin] if coin in e_dir.columns else pd.Series(0, index=e_addrs)
        e_pos = e_col[e_col != 0]
        if len(e_pos) < min_wallets:
            continue

        # Binary
        e_long_pct    = (e_pos > 0).sum() / max(len(e_pos), 1) * 100
        signal_binary = ("LONG" if e_long_pct > lo else "SHORT" if e_long_pct < hi else "NEUTRAL")

        # Notional
        if coin in e_not.columns:
            nc            = e_not[coin]
            long_not      = float(nc[e_col > 0].sum())
            short_not     = float(nc[e_col < 0].sum())
        else:
            long_not = short_not = 0.0
        total_not       = long_not + short_not
        net_not_pct     = long_not / total_not * 100 if total_not > 0 else 50.0
        signal_notional = ("LONG" if net_not_pct > lo else "SHORT" if net_not_pct < hi else "NEUTRAL")

        rows.append({
            "coin":                coin,
            "signal_binary":       signal_binary,
            "signal_notional":     signal_notional,
            "elite_wallets":       len(e_pos),
            "elite_long_pct":      round(e_long_pct, 1),
            "long_notional_usd":   round(long_not),
            "short_notional_usd":  round(short_not),
            "net_notional_pct":    round(net_not_pct, 1),
            "total_notional_usd":  round(total_not),
            "funding_rate_8h_pct": round(funding.get(coin, 0), 4),
            "price_usd":           px.get(coin),
        })

    signals_df = (pd.DataFrame(rows).sort_values("elite_long_pct", ascending=False)
                    .reset_index(drop=True)) if rows else _empty_signals()

    meta = {
        "elite_addresses":   e_addrs,
        "elite_raw":         e_sig_raw,
        "elite_display_raw": e_disp_raw,
        "participation_pct": round(participation, 1),
        "elite_active":      e_active,
        "low_participation": participation < MIN_PARTICIPATION,
        "fetch_seconds":     round(time.time() - t0, 1),
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "is_mock":           False,
        "leaderboard":       leaderboard,
        "params":            dict(n_elite=n_elite, min_wallets=min_wallets,
                                  div_threshold=div_threshold),
    }
    return signals_df, meta


def _build_wallet_detail(raw: dict, leaderboard: pd.DataFrame, group: str) -> pd.DataFrame:
    rows = []
    for addr, positions in raw.items():
        lb_row     = leaderboard[leaderboard["address"] == addr]
        wallet_pnl = float(lb_row["pnl"].values[0]) if not lb_row.empty else 0.0
        for p in positions:
            rows.append({
                "wallet":         addr[:8] + "…" + addr[-6:],
                "wallet_address": addr,
                "group":          group,
                "wallet_pnl_usd": round(wallet_pnl),
                "coin":           p["coin"],
                "direction":      p["direction"].upper(),
                "notional":       round(p["notional"]),
                "leverage":       p["leverage"],
                "entry_px":       p["entry_px"],
                "upnl":           round(p["unrealized_pnl"]),
                "in_signal":      ("⚠️ dust"     if p["notional"] < MIN_NOTIONAL_USD else
                                   "⚠️ high-lev" if p["leverage"] > MAX_LEVERAGE_SIGNAL else "✅"),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _build_top_bottom_table(top_raw: dict, bot_raw: dict,
                             leaderboard: pd.DataFrame) -> pd.DataFrame:
    """Flat table of positions for rank #1–5 and bottom #1–5 wallets."""
    rows = []
    for group, raw_dict in [("top5", top_raw), ("bottom5", bot_raw)]:
        for addr, positions in raw_dict.items():
            lb_row = leaderboard[leaderboard["address"] == addr]
            rank   = int(lb_row.index[0]) + 1 if not lb_row.empty else 0
            pnl    = float(lb_row["pnl"].values[0]) if not lb_row.empty else 0.0
            if not positions:
                rows.append({"group": group, "rank": rank,
                              "wallet": addr[:8]+"…"+addr[-6:],
                              "wallet_pnl_usd": round(pnl),
                              "coin": "—", "direction": "—",
                              "notional_usd": 0, "leverage": 0,
                              "entry_price": 0, "unrealized_pnl": 0})
                continue
            for p in positions:
                rows.append({"group": group, "rank": rank,
                              "wallet": addr[:8]+"…"+addr[-6:],
                              "wallet_pnl_usd": round(pnl),
                              "coin": p["coin"],
                              "direction": p["direction"].upper(),
                              "notional_usd": round(p["notional"]),
                              "leverage": p["leverage"],
                              "entry_price": p["entry_px"],
                              "unrealized_pnl": round(p["unrealized_pnl"])})
    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows).sort_values(["group", "rank"])
              .reset_index(drop=True))


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — MOCK DATA
# ══════════════════════════════════════════════════════════════════════════════

_COINS = [
    "BTC","ETH","SOL","XRP","DOGE","AVAX","LINK","UNI","AAVE","ARB",
    "OP","SUI","APT","INJ","TIA","NEAR","FTM","ATOM","DOT","WLD",
    "PENDLE","RNDR","IMX","MANTA","BLUR","PYTH","JTO","BONK","WIF","PEPE",
]
_MOCK_PRICES = {
    "BTC":97400,"ETH":3200,"SOL":175,"XRP":0.58,"DOGE":0.15,"AVAX":38,
    "LINK":18,"UNI":9.5,"AAVE":340,"ARB":1.2,"OP":2.8,"SUI":1.9,"APT":9,
    "INJ":28,"TIA":7,"NEAR":7.5,"FTM":0.55,"ATOM":10,"DOT":8.5,"WLD":5,
    "PENDLE":4,"RNDR":9,"IMX":2.2,"MANTA":1.8,"BLUR":0.35,"PYTH":0.45,
    "JTO":3.5,"BONK":0.00003,"WIF":3.2,"PEPE":0.0000085,
}


def _mock_leaderboard() -> pd.DataFrame:
    rng  = np.random.default_rng(42)
    n    = 200
    base = rng.lognormal(10.5, 2, n)
    sign = rng.choice([-1, 1], n, p=[0.38, 0.62])
    pnls = np.sort(base * sign)[::-1]
    return pd.DataFrame({
        "address":       [f"0x{i:040x}" for i in range(1, n+1)],
        "pnl":           pnls,
        "account_value": np.abs(pnls) * rng.uniform(2, 6, n),
    })


def _mock_signals(div_threshold: int) -> pd.DataFrame:
    seed = int(datetime.now().strftime("%Y%m%d"))
    rng  = np.random.default_rng(seed)
    lo   = 50 + div_threshold
    hi   = 50 - div_threshold
    rows = []
    for coin in _COINS:
        el   = rng.uniform(10, 90)
        ln   = float(rng.uniform(500_000, 20_000_000))
        sn   = float(rng.uniform(500_000, 20_000_000))
        tn   = ln + sn
        np_  = ln / tn * 100
        rows.append({
            "coin":                coin,
            "signal_binary":       "LONG" if el > lo else "SHORT" if el < hi else "NEUTRAL",
            "signal_notional":     "LONG" if np_ > lo else "SHORT" if np_ < hi else "NEUTRAL",
            "elite_wallets":       int(rng.integers(3, 14)),
            "elite_long_pct":      round(el, 1),
            "long_notional_usd":   round(ln),
            "short_notional_usd":  round(sn),
            "net_notional_pct":    round(np_, 1),
            "total_notional_usd":  round(tn),
            "funding_rate_8h_pct": round(float(rng.uniform(-0.05, 0.15)), 4),
            "price_usd":           _MOCK_PRICES.get(coin),
        })
    return (pd.DataFrame(rows).sort_values("elite_long_pct", ascending=False)
              .reset_index(drop=True))


def _mock_meta(n_elite: int) -> dict:
    rng     = np.random.default_rng(42)
    e_addrs = [f"0x{i:040x}" for i in range(1, n_elite+1)]

    def _fake_pos(addr_list, bias):
        raw = {}
        for addr in addr_list:
            coins = rng.choice(_COINS, int(rng.integers(1, 7)), replace=False).tolist()
            raw[addr] = [{"coin": c,
                          "direction": "long" if rng.random() < (0.72 if bias else 0.28) else "short",
                          "size":      round(float(rng.uniform(0.01, 5.0)), 4),
                          "entry_px":  round(_MOCK_PRICES.get(c, 1.0)*rng.uniform(0.85, 1.05), 4),
                          "unrealized_pnl": round(float(rng.uniform(-8000, 30000)), 2),
                          "leverage":  int(rng.integers(2, 20)),
                          "notional":  round(float(rng.uniform(1000, 5_000_000)), 2)}
                         for c in coins]
        return raw

    lb          = _mock_leaderboard()
    e_raw       = _fake_pos(e_addrs, True)
    top5_addrs  = [f"0x{i:040x}" for i in range(1, 6)]
    bot5_addrs  = [f"0x{i:040x}" for i in range(196, 201)]

    return {
        "elite_addresses":   e_addrs,
        "elite_raw":         e_raw,
        "elite_display_raw": e_raw,
        "top5_raw":          _fake_pos(top5_addrs, True),
        "bottom5_raw":       _fake_pos(bot5_addrs, False),
        "participation_pct": 72.0,
        "elite_active":      int(n_elite * 0.72),
        "low_participation": False,
        "fetch_seconds":     2.4,
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "is_mock":           True,
        "leaderboard":       lb,
        "params":            dict(n_elite=n_elite, min_wallets=_DEFAULT_MIN_WALLETS,
                                  div_threshold=_DEFAULT_DIV_THRESH),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

_BG    = "#0a0e1a"
_CARD  = "#111827"
_GRID  = "#1e3a5f"
_GREEN = "#00ff88"
_RED   = "#ff4444"
_MUTED = "#64748b"
_TEXT  = "#e2e8f0"
_YLW   = "#fbbf24"
_BLUE  = "#38bdf8"
_BASE  = dict(paper_bgcolor=_BG, plot_bgcolor=_BG,
              font=dict(color=_TEXT, family="Inter,sans-serif", size=12),
              margin=dict(l=12, r=12, t=36, b=12))


def _ax(**kw):
    return dict(gridcolor=_GRID, zerolinecolor=_GRID, tickcolor=_MUTED, linecolor=_GRID, **kw)


def _chart_elite_score(df: pd.DataFrame, dt: int) -> go.Figure:
    """Horizontal bar: elite_long_pct − 50, thresholds at ±dt."""
    d = df.assign(score=df["elite_long_pct"] - 50).sort_values("score")
    colors = [_GREEN if v > dt else _RED if v < -dt else _MUTED for v in d["score"]]
    fig = go.Figure(go.Bar(
        x=d["score"], y=d["coin"], orientation="h", marker_color=colors,
        text=[f"{v:+.1f}" for v in d["score"]], textposition="outside",
        textfont=dict(size=10, family="Space Mono,monospace"),
        customdata=np.stack([d["elite_long_pct"], d["elite_wallets"],
                             d["signal_binary"], d["signal_notional"]], axis=-1),
        hovertemplate=("<b>%{y}</b><br>Score: %{x:+.1f}<br>"
                       "Elite long%%: %{customdata[0]:.1f}%%<br>"
                       "Wallets: %{customdata[1]}<br>"
                       "Binary: %{customdata[2]} · Notional: %{customdata[3]}"
                       "<extra></extra>"),
    ))
    fig.add_vline(x= dt, line=dict(color=_GREEN, dash="dot", width=1))
    fig.add_vline(x=-dt, line=dict(color=_RED,   dash="dot", width=1))
    fig.add_vline(x=0,   line=dict(color=_MUTED,  width=0.5))
    fig.add_annotation(x= dt, y=0, yref="paper", text=f" {50+dt}%",
                       showarrow=False, font=dict(color=_GREEN, size=9), xanchor="left")
    fig.add_annotation(x=-dt, y=0, yref="paper", text=f"{50-dt}% ",
                       showarrow=False, font=dict(color=_RED, size=9), xanchor="right")
    fig.update_layout(**_BASE,
        title=dict(text="Elite Positioning — Deviation from 50% Neutral",
                   font=dict(size=13, color=_MUTED), x=0),
        xaxis=dict(title="← Bearish  |  Deviation from 50%  |  Bullish →",
                   range=[-55, 55], **_ax()),
        yaxis=dict(tickfont=dict(family="Space Mono,monospace", size=10), **_ax()),
        height=max(380, len(d)*22), showlegend=False)
    return fig


def _chart_binary_vs_notional(df: pd.DataFrame, dt: int) -> go.Figure:
    """Scatter: wallet % long (x) vs notional % long (y). Color = signal agreement."""
    cmap = {("LONG","LONG"): _GREEN, ("SHORT","SHORT"): _RED,
            ("NEUTRAL","NEUTRAL"): _MUTED}
    colors = [cmap.get((r["signal_binary"], r["signal_notional"]), _YLW)
              for _, r in df.iterrows()]
    sizes  = np.clip(np.log1p(df["total_notional_usd"] / 1e5) * 5, 8, 36)
    lo     = 50 + dt
    hi     = 50 - dt

    fig = go.Figure()
    for x0, y0, x1, y1, fill in [
        (lo, lo, 100, 100, "rgba(0,255,136,0.05)"),
        (0,  0,  hi,  hi,  "rgba(255,68,68,0.05)"),
    ]:
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, fillcolor=fill, line_width=0)

    fig.add_trace(go.Scatter(
        x=df["elite_long_pct"], y=df["net_notional_pct"],
        mode="markers+text", marker=dict(color=colors, size=sizes, opacity=0.85,
                                          line=dict(color=_BG, width=1)),
        text=df["coin"], textposition="top center",
        textfont=dict(size=9, family="Space Mono,monospace"),
        customdata=np.stack([df["signal_binary"], df["signal_notional"],
                             df["total_notional_usd"]/1e6], axis=-1),
        hovertemplate=("<b>%{text}</b><br>Binary long%%: %{x:.1f}%%<br>"
                       "Notional long%%: %{y:.1f}%%<br>"
                       "Binary: %{customdata[0]} · Notional: %{customdata[1]}<br>"
                       "Total: $%{customdata[2]:.1f}M<extra></extra>"),
    ))
    fig.add_shape(type="line", x0=0, y0=0, x1=100, y1=100,
                  line=dict(color=_MUTED, dash="dot", width=1))
    for val, col in [(lo, _GREEN), (hi, _RED)]:
        fig.add_vline(x=val, line=dict(color=col, dash="dot", width=1))
        fig.add_hline(y=val, line=dict(color=col, dash="dot", width=1))
    for txt, xp, yp, col in [("BOTH LONG", lo+1, lo+2, _GREEN),
                               ("BOTH SHORT", 1, hi-7, _RED),
                               ("MIXED", lo+1, hi-7, _YLW)]:
        fig.add_annotation(x=xp, y=yp, text=txt, showarrow=False,
                           font=dict(color=col, size=8, family="Space Mono,monospace"), opacity=0.6)
    fig.update_layout(**_BASE,
        title=dict(text="Binary vs Notional — Signal Agreement Map",
                   font=dict(size=13, color=_MUTED), x=0),
        xaxis=dict(title="Elite Wallets — % Net Long (Binary)", range=[0, 100], **_ax()),
        yaxis=dict(title="Elite Notional — % Net Long (Notional)", range=[0, 100], **_ax()),
        height=480)
    return fig


def _chart_coin_breakdown(df: pd.DataFrame, coin: str, dt: int) -> go.Figure:
    row = df[df["coin"] == coin]
    if row.empty:
        return go.Figure()
    row    = row.iloc[0]
    cats   = ["Long", "Short"]
    bv     = [row["elite_long_pct"],   100 - row["elite_long_pct"]]
    nv     = [row["net_notional_pct"], 100 - row["net_notional_pct"]]
    b_sig  = row["signal_binary"]
    n_sig  = row["signal_notional"]
    bc     = _GREEN if b_sig == "LONG" else _RED if b_sig == "SHORT" else _MUTED
    nc     = _GREEN if n_sig == "LONG" else _RED if n_sig == "SHORT" else _MUTED

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Binary (wallet %)", x=cats, y=bv,
                         marker_color=[_GREEN, _RED], opacity=0.85,
                         text=[f"{v:.1f}%" for v in bv], textposition="auto"))
    fig.add_trace(go.Bar(name="Notional (USD %)", x=cats, y=nv,
                         marker_color=[_GREEN, _RED], opacity=0.40,
                         text=[f"{v:.1f}%" for v in nv], textposition="auto"))
    fig.add_hline(y=50+dt, line=dict(color=_GREEN, dash="dot", width=1))
    fig.add_hline(y=50-dt, line=dict(color=_RED,   dash="dot", width=1))
    fig.update_layout(**_BASE,
        title=dict(
            text=(f"{coin}  |  Binary: <span style='color:{bc}'>{b_sig}</span>"
                  f"  |  Notional: <span style='color:{nc}'>{n_sig}</span>"),
            font=dict(size=13, family="Space Mono,monospace", color=_TEXT), x=0),
        barmode="group",
        yaxis=dict(title="% Long / Short", range=[0, 115], **_ax()),
        xaxis=dict(**_ax()),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        height=340)
    return fig


def _chart_notional_exposure(df: pd.DataFrame) -> go.Figure:
    sig = df[df["signal_binary"].isin(["LONG","SHORT"]) |
             df["signal_notional"].isin(["LONG","SHORT"])].copy()
    if sig.empty:
        return go.Figure()
    sig = sig.sort_values("net_notional_pct", ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Long",  x=sig["long_notional_usd"]/1e6,  y=sig["coin"],
                         orientation="h", marker_color=_GREEN, opacity=0.8,
                         text=[f"${v/1e6:.1f}M" for v in sig["long_notional_usd"]],
                         textposition="auto"))
    fig.add_trace(go.Bar(name="Short", x=-(sig["short_notional_usd"]/1e6), y=sig["coin"],
                         orientation="h", marker_color=_RED, opacity=0.8,
                         text=[f"${v/1e6:.1f}M" for v in sig["short_notional_usd"]],
                         textposition="auto"))
    fig.update_layout(**_BASE,
        title=dict(text="Elite Notional Exposure — Signal Coins ($M)",
                   font=dict(size=13, color=_MUTED), x=0),
        xaxis=dict(title="← Short  |  USD Millions  |  Long →", **_ax()),
        yaxis=dict(tickfont=dict(family="Space Mono,monospace", size=10), **_ax()),
        barmode="relative", height=max(320, len(sig)*28),
        legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig


def _chart_funding(df: pd.DataFrame) -> go.Figure:
    d = df[df["signal_binary"].isin(["LONG","SHORT"]) |
           df["signal_notional"].isin(["LONG","SHORT"])].copy()
    d = d.sort_values("funding_rate_8h_pct", ascending=False)
    if d.empty:
        return go.Figure()
    colors = [_GREEN if (r["signal_binary"]=="LONG"  and r["funding_rate_8h_pct"]<0) else
              _RED   if (r["signal_binary"]=="SHORT" and r["funding_rate_8h_pct"]>0) else _YLW
              for _, r in d.iterrows()]
    fig = go.Figure(go.Bar(x=d["coin"], y=d["funding_rate_8h_pct"], marker_color=colors,
                            text=[f"{v:+.4f}%" for v in d["funding_rate_8h_pct"]],
                            textposition="outside",
                            textfont=dict(size=9, family="Space Mono,monospace"),
                            hovertemplate="<b>%{x}</b><br>Funding (8h): %{y:.4f}%<extra></extra>"))
    for y in [0.01, -0.01]:
        fig.add_hline(y=y, line=dict(color=_YLW, dash="dot", width=1))
    fig.update_layout(**_BASE,
        title=dict(text="Funding Rates — Signal Coins (8h %)",
                   font=dict(size=13, color=_MUTED), x=0),
        yaxis=dict(title="Funding Rate %", **_ax()),
        xaxis=dict(**_ax()), height=320)
    return fig


def _chart_leaderboard(lb: pd.DataFrame, n_elite: int) -> go.Figure:
    fig = go.Figure()
    for x, name, color, op, nb in [
        (lb["pnl"].values,               "All Wallets",            _MUTED, 0.40, 40),
        (lb.head(n_elite)["pnl"].values,  f"Elite (Top {n_elite})", _GREEN, 0.85, 20),
    ]:
        fig.add_trace(go.Histogram(x=x, name=name, marker_color=color, opacity=op, nbinsx=nb))
    fig.update_layout(**_BASE,
        title=dict(text="Leaderboard PnL Distribution",
                   font=dict(size=13, color=_MUTED), x=0),
        xaxis=dict(title="Total PnL (USD)", **_ax()),
        yaxis=dict(title="Wallet Count", **_ax()),
        barmode="overlay", legend=dict(bgcolor="rgba(0,0,0,0)"), height=300)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7b — EXCEL EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def _positions_to_df(raw: dict, group: str) -> pd.DataFrame:
    cols = ["group","wallet_address","coin","direction","size",
            "entry_price","notional_usd","leverage","unrealized_pnl"]
    rows = []
    for addr, positions in raw.items():
        for p in positions:
            rows.append({"group": group, "wallet_address": addr,
                         "coin": p.get("coin",""), "direction": p.get("direction","").upper(),
                         "size": p.get("size", 0), "entry_price": p.get("entry_px", 0),
                         "notional_usd": round(p.get("notional", 0), 2),
                         "leverage": p.get("leverage", 0),
                         "unrealized_pnl": round(p.get("unrealized_pnl", 0), 2)})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)


def _build_excel(sheets: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            if df is None or (hasattr(df, "empty") and df.empty):
                pd.DataFrame(["No data"]).to_excel(writer, sheet_name=name[:31],
                                                   index=False, header=False)
            else:
                df.to_excel(writer, sheet_name=name[:31], index=False)
    return buf.getvalue()


def _dl_btn(label: str, data: bytes, filename: str):
    st.download_button(label=label, data=data, file_name=filename,
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — SESSION STATE CACHING
# ══════════════════════════════════════════════════════════════════════════════

_SS_KEY = "whale_tracker_v3"


def _needs_fetch(force_mock: bool) -> bool:
    if _SS_KEY not in st.session_state:
        return True
    return st.session_state[_SS_KEY].get("is_mock") != force_mock


def _run_fetch(n_elite: int, min_wallets: int, div_threshold: int,
               force_mock: bool) -> dict:
    if force_mock:
        signals, meta = compute_signals(pd.DataFrame(), n_elite, min_wallets,
                                        div_threshold, use_mock=True)
        meta["is_mock"]     = True
        meta["data_source"] = "Demo (simulated)"
        cache = {**meta, "signals": signals}
        st.session_state[_SS_KEY] = cache
        return cache

    lb, is_live = fetch_leaderboard()
    if not is_live:
        signals, meta = compute_signals(lb, n_elite, min_wallets,
                                        div_threshold, use_mock=True)
        meta["is_mock"]     = True
        meta["data_source"] = "Demo (API unreachable)"
    else:
        funding = fetch_funding_rates()
        prices  = fetch_prices()
        signals, meta = compute_signals(lb, n_elite, min_wallets,
                                        div_threshold, funding, prices)
        meta["is_mock"]     = False
        meta["data_source"] = f"Live — {len(lb)} wallets, elite top {n_elite}"

        # Fetch top5 / bottom5 for Table 4
        top5 = lb.head(5)["address"].tolist()
        bot5 = lb.tail(5)["address"].tolist()
        meta["top5_raw"]    = {a: fetch_positions(a, for_signal=False) for a in top5}
        time.sleep(_DELAY)
        meta["bottom5_raw"] = {a: fetch_positions(a, for_signal=False) for a in bot5}

    cache = {**meta, "signals": signals}
    st.session_state[_SS_KEY] = cache
    return cache


def _recompute_signals(cache: dict, n_elite: int,
                        min_wallets: int, div_threshold: int) -> pd.DataFrame:
    if cache.get("is_mock"):
        return _mock_signals(div_threshold)

    lb        = cache.get("leaderboard", pd.DataFrame())
    elite_raw = cache.get("elite_raw", {})
    fetched   = list(elite_raw.keys())
    n_eff     = min(n_elite, len(fetched))
    e_addrs   = (lb.head(n_eff)["address"].tolist()
                 if not lb.empty and len(lb) >= n_eff else fetched[:n_eff])

    all_coins = sorted({p["coin"] for pos in elite_raw.values() for p in pos})
    if not all_coins:
        return _empty_signals()

    lo = 50 + div_threshold
    hi = 50 - div_threshold

    dir_rows, not_rows = {}, {}
    for addr in e_addrs:
        pm = {p["coin"]: p for p in elite_raw.get(addr, [])}
        dir_rows[addr] = {c: (1 if pm[c]["direction"]=="long" else -1) if c in pm else 0
                           for c in all_coins}
        not_rows[addr] = {c: pm[c]["notional"] if c in pm else 0.0 for c in all_coins}

    e_dir = pd.DataFrame(dir_rows, index=all_coins).T
    e_not = pd.DataFrame(not_rows, index=all_coins).T
    rows  = []

    for coin in all_coins:
        ec  = e_dir[coin] if coin in e_dir.columns else pd.Series(0, index=e_addrs)
        ep  = ec[ec != 0]
        if len(ep) < min_wallets:
            continue
        elp = (ep > 0).sum() / max(len(ep), 1) * 100
        sb  = "LONG" if elp > lo else "SHORT" if elp < hi else "NEUTRAL"
        if coin in e_not.columns:
            nc  = e_not[coin]
            ln  = float(nc[ec > 0].sum())
            sn  = float(nc[ec < 0].sum())
        else:
            ln = sn = 0.0
        tn  = ln + sn
        np_ = ln / tn * 100 if tn > 0 else 50.0
        sn_ = "LONG" if np_ > lo else "SHORT" if np_ < hi else "NEUTRAL"
        rows.append({"coin": coin, "signal_binary": sb, "signal_notional": sn_,
                     "elite_wallets": len(ep), "elite_long_pct": round(elp, 1),
                     "long_notional_usd": round(ln), "short_notional_usd": round(sn),
                     "net_notional_pct": round(np_, 1), "total_notional_usd": round(tn),
                     "funding_rate_8h_pct": 0.0, "price_usd": None})

    if not rows:
        return _empty_signals()
    return (pd.DataFrame(rows).sort_values("elite_long_pct", ascending=False)
              .reset_index(drop=True))


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — UI COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

def _render_sidebar() -> tuple:
    with st.sidebar:
        st.markdown("<div style='font-family:Space Mono,monospace;color:#00ff88;"
                    "font-size:1.1rem;letter-spacing:2px;margin-bottom:8px;'>"
                    "🐋 WHALE TRACKER</div>", unsafe_allow_html=True)
        st.divider()
        st.markdown("**Strategy Parameters**")
        st.caption("Sliders apply instantly. 'Fetch Fresh Data' hits the API.")

        n_elite = st.slider("Elite wallets (top N by monthly PnL)",
                             5, 50, _DEFAULT_N_ELITE, 5,
                             help="Top N wallets by PnL. Only these drive signals.")
        min_wallets = st.slider("Min elite wallets per coin",
                                 1, 5, _DEFAULT_MIN_WALLETS, 1,
                                 help="Minimum elite wallets holding a coin to qualify.")
        div_threshold = st.slider("Threshold deviation from 50% (±)",
                                   5, 40, _DEFAULT_DIV_THRESH, 5,
                                   help="At ±20: LONG >70%, SHORT <30%.")

        lo = 50 + div_threshold
        hi = 50 - div_threshold

        st.divider()
        st.markdown(
            f"<div style='font-size:0.78rem;color:#94a3b8;line-height:2.0;'>"
            f"<b>Signal Logic</b><br>"
            f"<code style='color:#00ff88;'>Binary</code> = % elite wallets long<br>"
            f"<code style='color:#38bdf8;'>Notional</code> = % elite USD long<br>"
            f"✅ LONG  → metric &gt; {lo}%<br>"
            f"🚫 SHORT → metric &lt; {hi}%<br>"
            f"⚪ NEUTRAL → {hi}%–{lo}%<br><br>"
            f"<b>Position filters</b><br>"
            f"🗑️ Dust &lt;${MIN_NOTIONAL_USD:,} excluded<br>"
            f"⚡ Leverage &gt;{MAX_LEVERAGE_SIGNAL}x excluded</div>",
            unsafe_allow_html=True)
        st.divider()

        force_mock = st.toggle("Use demo data", value=False)
        do_refresh = st.button("🔄 Fetch Fresh Data", use_container_width=True)

        if _SS_KEY in st.session_state:
            c   = st.session_state[_SS_KEY]
            ts  = c.get("timestamp", "")
            src = c.get("data_source", "")
            if ts:
                dt_ = datetime.fromisoformat(ts)
                st.caption(f"📦 {dt_.strftime('%H:%M UTC')} "
                           f"({(datetime.now(timezone.utc)-dt_).seconds//60}m ago)")
                st.caption(f"📡 {src}")
        else:
            st.caption("📦 No data yet")

        if _last_api_error:
            st.markdown(f"<div style='background:#1a0a0a;border:1px solid #7f1d1d;"
                        f"border-radius:6px;padding:8px 12px;margin-top:8px;"
                        f"font-size:0.70rem;color:#fca5a5;word-break:break-all;'>"
                        f"⚠️ {_last_api_error}</div>", unsafe_allow_html=True)

        st.divider()
        st.markdown("<div class='wt-disclaimer'>⚠️ Not financial advice.<br>"
                    "Research purposes only.</div>", unsafe_allow_html=True)

    return n_elite, min_wallets, div_threshold, force_mock, do_refresh


def _render_header(meta: dict):
    ts   = meta.get("timestamp", "")
    dt_s = datetime.fromisoformat(ts).strftime("%d %b %Y, %H:%M UTC") if ts else "—"
    mode, col = ("DEMO DATA", "#fbbf24") if meta.get("is_mock") else ("LIVE DATA", "#00ff88")
    p    = meta.get("params", {})
    src  = meta.get("data_source", "")
    st.markdown(
        f"<div class='wt-header'><h1>🐋 CRYPTO WHALE TRACKER</h1>"
        f"<p>HYPERLIQUID PERPETUALS · {dt_s} · <span style='color:{col};'>{mode}</span>"
        f"{f' · {src}' if src else ''} · "
        f"Elite {p.get('n_elite','?')} · Threshold ±{p.get('div_threshold','?')} · "
        f"Scan {meta.get('fetch_seconds','—')}s</p></div>",
        unsafe_allow_html=True)

    if meta.get("is_mock"):
        forced = meta.get("data_source") == "Demo (simulated)"
        err    = _last_api_error
        msg = ("⚠️ <b>Demo Mode.</b> Toggle off 'Use demo data' and click 'Fetch Fresh Data'." if forced
               else f"⚠️ <b>API Unreachable.</b> Showing simulated data."
                    f"{f'<br><code style=\"font-size:0.75rem;\">{err}</code>' if err else ''}"
                    f" Try running locally.")
        st.markdown(f"<div class='wt-mock-banner'>{msg}</div>", unsafe_allow_html=True)
    else:
        lb  = meta.get("leaderboard", pd.DataFrame())
        ne  = p.get("n_elite", _DEFAULT_N_ELITE)
        if not lb.empty and len(lb) < ne + 20:
            st.markdown(
                f"<div style='background:#1c1500;border:1px solid #854d0e;"
                f"border-radius:8px;padding:10px 16px;margin-bottom:12px;"
                f"color:#fbbf24;font-size:0.82rem;'>⚠️ Leaderboard only {len(lb)} rows. "
                f"Reduce Elite wallet count.</div>", unsafe_allow_html=True)


def _render_kpis(signals: pd.DataFrame, dt: int):
    bl = signals[signals["signal_binary"]   == "LONG"]
    bs = signals[signals["signal_binary"]   == "SHORT"]
    nl = signals[signals["signal_notional"] == "LONG"]
    ns = signals[signals["signal_notional"] == "SHORT"]
    ae = signals["elite_long_pct"].mean() if not signals.empty else 50
    sentiment = "BULLISH" if ae > 50+dt else "BEARISH" if ae < 50-dt else "NEUTRAL"
    sc = _GREEN if sentiment == "BULLISH" else _RED if sentiment == "BEARISH" else _YLW

    for col, (val, lbl, color) in zip(st.columns(5), [
        (str(len(bl)), "BINARY LONGS",    _GREEN),
        (str(len(bs)), "BINARY SHORTS",   _RED),
        (str(len(nl)), "NOTIONAL LONGS",  "#00ccff"),
        (str(len(ns)), "NOTIONAL SHORTS", "#ff8888"),
        (f"<span style='color:{sc};'>{sentiment}</span>", "MARKET BIAS", _MUTED),
    ]):
        col.markdown(f"<div class='wt-kpi'><div class='val' style='color:{color};'>{val}</div>"
                     f"<div class='lbl'>{lbl}</div></div>", unsafe_allow_html=True)


def _render_signal_table(df: pd.DataFrame, signal_type: str, signal_col: str):
    filtered = df[df[signal_col] == signal_type].copy()
    if filtered.empty:
        st.info(f"No {signal_type} signals.")
        return
    val_col   = "elite_long_pct"  if signal_col == "signal_binary"   else "net_notional_pct"
    label_col = "Binary Long %" if signal_col == "signal_binary" else "Notional Long %"
    disp = filtered[["coin", val_col, "elite_wallets",
                      "total_notional_usd", "funding_rate_8h_pct", "price_usd"]].copy()
    disp.columns = ["Coin", label_col, "Elite #", "Total Notional", "Funding 8h%", "Price"]
    disp[label_col]       = disp[label_col].apply(lambda x: f"{x:.1f}%")
    disp["Total Notional"] = disp["Total Notional"].apply(
        lambda x: f"${x/1e6:.2f}M" if x >= 1e6 else f"${x/1e3:.0f}K")
    disp["Price"] = disp["Price"].apply(
        lambda x: f"${x:,.4f}" if pd.notna(x) and x < 1
                  else f"${x:,.2f}" if pd.notna(x) else "—")
    st.dataframe(disp, use_container_width=True,
                 height=min(400, 55+len(disp)*37), hide_index=True)


def _render_wallet_table(meta: dict, leaderboard: pd.DataFrame, group: str):
    raw = meta.get(f"{group}_display_raw", meta.get(f"{group}_raw", {}))
    if not raw:
        st.info("No wallet data available.")
        return
    df = _build_wallet_detail(raw, leaderboard, group)
    if df.empty:
        st.info("No open positions found.")
        return
    df["notional"] = df["notional"].apply(lambda x: f"${x:,.0f}")
    df["upnl"]     = df["upnl"].apply(lambda x: f"+${x:,.0f}" if x >= 0 else f"-${abs(x):,.0f}")
    st.dataframe(
        df[["wallet","wallet_pnl_usd","coin","direction","notional","leverage","entry_px","upnl","in_signal"]],
        column_config={
            "wallet_pnl_usd": st.column_config.NumberColumn("Wallet PnL ($)", format="$%.0f"),
            "leverage":       st.column_config.NumberColumn("Lev", format="%.1fx"),
            "in_signal":      st.column_config.TextColumn("In Signal?",
                help=f"✅ included · ⚠️ dust = <${MIN_NOTIONAL_USD:,} · ⚠️ high-lev = >{MAX_LEVERAGE_SIGNAL}x"),
        },
        use_container_width=True, height=400, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    n_elite, min_wallets, div_threshold, force_mock, do_refresh = _render_sidebar()

    if do_refresh or _needs_fetch(force_mock):
        with st.spinner("Scanning Hyperliquid wallets…"):
            cache = _run_fetch(n_elite, min_wallets, div_threshold, force_mock)
    else:
        cache = st.session_state[_SS_KEY]

    signals     = _recompute_signals(cache, n_elite, min_wallets, div_threshold)
    leaderboard = cache.get("leaderboard", _mock_leaderboard())
    meta        = {**cache, "params": dict(n_elite=n_elite, min_wallets=min_wallets,
                                            div_threshold=div_threshold)}

    bin_long  = signals[signals["signal_binary"]   == "LONG"].reset_index(drop=True)
    bin_short = signals[signals["signal_binary"]   == "SHORT"].reset_index(drop=True)
    not_long  = signals[signals["signal_notional"] == "LONG"].reset_index(drop=True)
    not_short = signals[signals["signal_notional"] == "SHORT"].reset_index(drop=True)
    lo        = 50 + div_threshold
    hi        = 50 - div_threshold
    ts_str    = datetime.now().strftime("%Y%m%d_%H%M")

    _render_header(meta)
    _render_kpis(signals, div_threshold)

    tab_ov, tab_lg, tab_sh, tab_dg, tab_raw = st.tabs([
        "📊  Overview", "🟢  Long Picks", "🔴  Short Picks", "🔬  Diagnostics", "📋  Raw Data"])

    # ── Overview ──────────────────────────────────────────────────────────────
    with tab_ov:
        st.markdown("<div class='wt-section'>Elite Positioning Score — All Coins</div>",
                    unsafe_allow_html=True)
        st.caption(f"Score = elite_long_pct − 50. LONG >+{div_threshold} (>{lo}%),  "
                   f"SHORT <−{div_threshold} (<{hi}%)")
        st.plotly_chart(_chart_elite_score(signals, div_threshold),
                        use_container_width=True, config={"displayModeBar": False})

        st.markdown("<div class='wt-section'>Binary vs Notional — Signal Agreement Map</div>",
                    unsafe_allow_html=True)
        st.caption("🟢 Both LONG · 🔴 Both SHORT · 🟡 Mixed (disagree) · ⚫ Neutral")
        if not signals.empty:
            st.plotly_chart(_chart_binary_vs_notional(signals, div_threshold),
                            use_container_width=True, config={"displayModeBar": False})

    # ── Long Picks ────────────────────────────────────────────────────────────
    with tab_lg:
        st.markdown(f"<div class='wt-section'>Long Picks — elite_long_pct &gt; {lo}%</div>",
                    unsafe_allow_html=True)
        cb, cn = st.columns(2)
        with cb:
            st.markdown(f"<div style='font-family:Space Mono,monospace;color:#00ff88;"
                        f"font-size:0.80rem;letter-spacing:1px;margin-bottom:8px;'>"
                        f"🔵 BINARY — wallet count &gt; {lo}%</div>", unsafe_allow_html=True)
            _render_signal_table(signals, "LONG", "signal_binary")
        with cn:
            st.markdown(f"<div style='font-family:Space Mono,monospace;color:#00ccff;"
                        f"font-size:0.80rem;letter-spacing:1px;margin-bottom:8px;'>"
                        f"💧 NOTIONAL — USD notional &gt; {lo}%</div>", unsafe_allow_html=True)
            _render_signal_table(signals, "LONG", "signal_notional")

        all_long = pd.concat([bin_long, not_long]).drop_duplicates("coin")
        if not all_long.empty:
            st.markdown("<div class='wt-section'>Drill-Down</div>", unsafe_allow_html=True)
            sel = st.selectbox("Select coin", all_long["coin"].tolist(),
                               key="lg_drill", label_visibility="collapsed")
            row = signals[signals["coin"] == sel].iloc[0]
            ca, cb2 = st.columns(2)
            with ca:
                st.plotly_chart(_chart_coin_breakdown(signals, sel, div_threshold),
                                use_container_width=True, config={"displayModeBar": False})
            with cb2:
                bs  = row["signal_binary"]
                ns  = row["signal_notional"]
                bc  = _GREEN if bs=="LONG" else _RED if bs=="SHORT" else _MUTED
                nc  = _GREEN if ns=="LONG" else _RED if ns=="SHORT" else _MUTED
                st.markdown(
                    f"<div class='wt-coin-card'>"
                    f"<div style='font-family:Space Mono,monospace;color:#00ff88;"
                    f"font-size:1.4rem;font-weight:700;'>{sel}</div>"
                    f"<div style='color:#64748b;font-size:0.75rem;margin-bottom:16px;'>"
                    f"SIGNAL DETAILS</div><table>"
                    f"<tr><td style='color:#64748b;'>Binary signal</td>"
                    f"<td style='color:{bc};text-align:right;font-family:Space Mono,monospace;'>{bs}</td></tr>"
                    f"<tr><td style='color:#64748b;'>Notional signal</td>"
                    f"<td style='color:{nc};text-align:right;font-family:Space Mono,monospace;'>{ns}</td></tr>"
                    f"<tr><td style='color:#64748b;'>Elite wallets</td>"
                    f"<td style='color:#e2e8f0;text-align:right;'>{int(row['elite_wallets'])}</td></tr>"
                    f"<tr><td style='color:#64748b;'>Elite long %</td>"
                    f"<td style='color:#e2e8f0;text-align:right;'>{row['elite_long_pct']:.1f}%</td></tr>"
                    f"<tr><td style='color:#64748b;'>Long notional</td>"
                    f"<td style='color:#e2e8f0;text-align:right;'>${row['long_notional_usd']/1e6:.2f}M</td></tr>"
                    f"<tr><td style='color:#64748b;'>Short notional</td>"
                    f"<td style='color:#e2e8f0;text-align:right;'>${row['short_notional_usd']/1e6:.2f}M</td></tr>"
                    f"<tr><td style='color:#64748b;'>Net notional %</td>"
                    f"<td style='color:#e2e8f0;text-align:right;'>{row['net_notional_pct']:.1f}%</td></tr>"
                    f"<tr><td style='color:#64748b;'>Funding (8h)</td>"
                    f"<td style='color:#fbbf24;text-align:right;'>{row['funding_rate_8h_pct']:+.4f}%</td></tr>"
                    f"</table></div>", unsafe_allow_html=True)

    # ── Short Picks ───────────────────────────────────────────────────────────
    with tab_sh:
        st.markdown(f"<div class='wt-section'>Short Picks — elite_long_pct &lt; {hi}%</div>",
                    unsafe_allow_html=True)
        sb_col, sn_col = st.columns(2)
        with sb_col:
            st.markdown(f"<div style='font-family:Space Mono,monospace;color:#ff4444;"
                        f"font-size:0.80rem;letter-spacing:1px;margin-bottom:8px;'>"
                        f"🔵 BINARY — wallet count &lt; {hi}%</div>", unsafe_allow_html=True)
            _render_signal_table(signals, "SHORT", "signal_binary")
        with sn_col:
            st.markdown(f"<div style='font-family:Space Mono,monospace;color:#ff8888;"
                        f"font-size:0.80rem;letter-spacing:1px;margin-bottom:8px;'>"
                        f"💧 NOTIONAL — USD notional &lt; {hi}%</div>", unsafe_allow_html=True)
            _render_signal_table(signals, "SHORT", "signal_notional")

        if not bin_short.empty or not not_short.empty:
            st.markdown("<div class='wt-section'>Funding Cost Check</div>",
                        unsafe_allow_html=True)
            st.plotly_chart(_chart_funding(signals), use_container_width=True,
                            config={"displayModeBar": False})
            st.caption("🟡 Yellow = funding works against short. Prefer negative funding.")

        all_short = pd.concat([bin_short, not_short]).drop_duplicates("coin")
        if not all_short.empty:
            st.markdown("<div class='wt-section'>Drill-Down</div>", unsafe_allow_html=True)
            sel_s = st.selectbox("Select coin", all_short["coin"].tolist(),
                                 key="sh_drill", label_visibility="collapsed")
            st.plotly_chart(_chart_coin_breakdown(signals, sel_s, div_threshold),
                            use_container_width=True, config={"displayModeBar": False})

    # ── Diagnostics ───────────────────────────────────────────────────────────
    with tab_dg:
        st.markdown("<div class='wt-section'>Notional Exposure — Signal Coins</div>",
                    unsafe_allow_html=True)
        st.plotly_chart(_chart_notional_exposure(signals), use_container_width=True,
                        config={"displayModeBar": False})

        st.markdown("<div class='wt-section'>Wallet Universe — PnL Distribution</div>",
                    unsafe_allow_html=True)
        st.plotly_chart(_chart_leaderboard(leaderboard, n_elite), use_container_width=True,
                        config={"displayModeBar": False})

        st.markdown("<div class='wt-section'>Signal Summary & Scan Quality</div>",
                    unsafe_allow_html=True)
        cx, cy, cz = st.columns(3)
        with cx:
            st.markdown("**Binary Signals**")
            st.dataframe(
                signals[["coin","signal_binary","elite_long_pct","elite_wallets"]]
                .rename(columns={"signal_binary":"Signal","elite_long_pct":"Long %",
                                  "elite_wallets":"Elite #"}),
                hide_index=True, use_container_width=True, height=380)
        with cy:
            st.markdown("**Notional Signals**")
            st.dataframe(
                signals[["coin","signal_notional","net_notional_pct",
                          "long_notional_usd","short_notional_usd"]]
                .assign(long_notional_usd=lambda d: d["long_notional_usd"].apply(
                            lambda x: f"${x/1e6:.1f}M"),
                        short_notional_usd=lambda d: d["short_notional_usd"].apply(
                            lambda x: f"${x/1e6:.1f}M"))
                .rename(columns={"signal_notional":"Signal","net_notional_pct":"Net Long %",
                                  "long_notional_usd":"Long $","short_notional_usd":"Short $"}),
                hide_index=True, use_container_width=True, height=380)
        with cz:
            st.markdown("**Coverage**")
            part_pct = meta.get("participation_pct", 0)
            e_active = meta.get("elite_active", 0)
            low_part = meta.get("low_participation", False)
            part_col = _RED if low_part else (_YLW if part_pct < 40 else _GREEN)
            st.metric("Coins scanned",    len(signals))
            st.metric("Binary signals",   len(bin_long)+len(bin_short))
            st.metric("Notional signals", len(not_long)+len(not_short))
            st.metric("Elite active",     f"{e_active}/{n_elite}")
            st.markdown(
                f"<div style='margin-top:12px;background:var(--card);"
                f"border:1px solid {'#7f1d1d' if low_part else '#1e3a5f'};"
                f"border-radius:8px;padding:12px;font-size:0.80rem;'>"
                f"<b style='color:{part_col};'>Participation: {part_pct:.0f}%</b>"
                f"{'<br><span style=\"color:#f87171;\">⚠️ Low — signals may be thin</span>' if low_part else ''}"
                f"</div>", unsafe_allow_html=True)

    # ── Raw Data ──────────────────────────────────────────────────────────────
    with tab_raw:
        lb_full        = leaderboard.copy()
        if not lb_full.empty:
            lb_full.insert(0, "rank", range(1, len(lb_full)+1))

        elite_disp_raw = meta.get("elite_display_raw", meta.get("elite_raw", {}))
        elite_pos_df   = _positions_to_df(elite_disp_raw, "elite")
        top5_raw       = meta.get("top5_raw", {})
        bot5_raw       = meta.get("bottom5_raw", {})
        top5bot5_df    = _build_top_bottom_table(top5_raw, bot5_raw, leaderboard)

        # Master download
        st.markdown("<div class='wt-section'>Download All Data</div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='background:#0d1f3c;border:1px solid #1e4d8c;border-radius:10px;"
            "padding:16px 20px;margin-bottom:8px;'>"
            "<div style='font-family:Space Mono,monospace;color:#38bdf8;"
            "font-size:0.82rem;letter-spacing:1px;margin-bottom:10px;'>"
            "📥 MASTER WORKBOOK — 4 sheets</div>"
            "<div style='font-size:0.78rem;color:#94a3b8;line-height:1.8;'>"
            "Sheet 1 · <b>Signals</b> — binary + notional signals for all coins<br>"
            "Sheet 2 · <b>Leaderboard</b> — full wallet ranking by PnL<br>"
            "Sheet 3 · <b>Elite Positions</b> — all open positions, top N wallets<br>"
            "Sheet 4 · <b>Top5 Bottom5</b> — holdings of rank #1–5 and bottom #1–5"
            "</div></div>", unsafe_allow_html=True)
        _dl_btn(f"⬇️  Download Master Workbook  ({ts_str}).xlsx",
                _build_excel({"Signals": signals, "Leaderboard": lb_full,
                               "Elite Positions": elite_pos_df, "Top5 Bottom5": top5bot5_df}),
                f"hyperliquid_whale_{ts_str}.xlsx")

        st.divider()

        # Table 1 — Signals
        st.markdown("<div class='wt-section'>Table 1 — Computed Signals</div>",
                    unsafe_allow_html=True)
        c1, d1 = st.columns([4, 1])
        with c1:
            st.caption(f"{len(signals)} coins · threshold ±{div_threshold} (>{lo}% LONG, <{hi}% SHORT)")
        with d1:
            _dl_btn("⬇️ Download", _build_excel({"Signals": signals}), f"hl_signals_{ts_str}.xlsx")
        st.dataframe(signals, use_container_width=True, hide_index=True, height=420)

        st.divider()

        # Table 2 — Full Leaderboard
        st.markdown("<div class='wt-section'>Table 2 — Full Leaderboard</div>",
                    unsafe_allow_html=True)
        c2, d2 = st.columns([4, 1])
        with c2:
            st.caption(f"{len(lb_full):,} wallets · 🟢 top {n_elite} = elite · monthly PnL")
        with d2:
            _dl_btn("⬇️ Download", _build_excel({"Leaderboard": lb_full}),
                    f"hl_leaderboard_{ts_str}.xlsx")
        if not lb_full.empty:
            def _clb(row):
                r = row.get("rank", 0)
                return (["background-color:#052e1a;color:#00ff88"] * len(row)
                        if r <= n_elite else [""] * len(row))
            st.dataframe(lb_full.style.apply(_clb, axis=1).format(
                {c: "${:,.0f}" for c in lb_full.columns if "pnl" in c.lower() or "value" in c.lower()}),
                use_container_width=True, height=500, hide_index=True)
        else:
            st.info("Leaderboard not available.")

        st.divider()

        # Table 3 — Elite Positions
        st.markdown("<div class='wt-section'>Table 3 — Elite Wallet Positions</div>",
                    unsafe_allow_html=True)
        n_ep = len(elite_pos_df)
        n_ew = elite_pos_df["wallet_address"].nunique() if n_ep else 0
        c3, d3 = st.columns([4, 1])
        with c3:
            st.caption(f"{n_ep:,} positions · {n_ew} wallets · unfiltered (⚠️ = excluded from signal)")
        with d3:
            _dl_btn("⬇️ Download", _build_excel({"Elite Positions": elite_pos_df}),
                    f"hl_elite_pos_{ts_str}.xlsx")
        if not elite_pos_df.empty:
            d = elite_pos_df.copy()
            d["in_signal"] = d.apply(
                lambda r: ("⚠️ dust"     if r["notional_usd"] < MIN_NOTIONAL_USD else
                           "⚠️ high-lev" if r["leverage"] > MAX_LEVERAGE_SIGNAL else "✅"), axis=1)
            st.dataframe(d, column_config={
                "wallet_address": st.column_config.TextColumn("Wallet", width="medium"),
                "notional_usd":   st.column_config.NumberColumn("Notional ($)", format="$%.2f"),
                "unrealized_pnl": st.column_config.NumberColumn("uPnL ($)", format="$%.2f"),
                "leverage":       st.column_config.NumberColumn("Lev", format="%.1fx"),
                "in_signal":      st.column_config.TextColumn("In Signal?"),
            }, use_container_width=True, height=420, hide_index=True)
        else:
            st.info("No elite positions available.")

        st.divider()

        # Table 4 — Top 5 / Bottom 5
        st.markdown("<div class='wt-section'>Table 4 — Top 5 & Bottom 5 Wallet Holdings</div>",
                    unsafe_allow_html=True)
        st.markdown(
            "<div style='background:#0d1f3c;border:1px solid #1e3a5f;border-radius:8px;"
            "padding:10px 16px;margin-bottom:12px;font-size:0.80rem;color:#94a3b8;line-height:1.8;'>"
            "🟢 <b>top5</b> — wallets ranked #1–#5 by PnL (highest earners)<br>"
            "🔴 <b>bottom5</b> — wallets ranked bottom #1–#5 by PnL (biggest losers)<br>"
            "Unfiltered raw positions — no dust/leverage filters applied.</div>",
            unsafe_allow_html=True)
        n_t5 = len(top5bot5_df[top5bot5_df["group"]=="top5"])    if not top5bot5_df.empty else 0
        n_b5 = len(top5bot5_df[top5bot5_df["group"]=="bottom5"]) if not top5bot5_df.empty else 0
        c4, d4 = st.columns([4, 1])
        with c4:
            st.caption(f"top5: {n_t5} positions · bottom5: {n_b5} positions")
        with d4:
            _dl_btn("⬇️ Download", _build_excel({"Top5 Bottom5": top5bot5_df}),
                    f"hl_top5bot5_{ts_str}.xlsx")
        if not top5bot5_df.empty:
            def _ct5(row):
                return (["background-color:#052e1a;color:#00ff88"] * len(row)
                        if row.get("group") == "top5"
                        else ["background-color:#2a0a0a;color:#ff8888"] * len(row))
            st.dataframe(
                top5bot5_df.style.apply(_ct5, axis=1).format(
                    {"wallet_pnl_usd": "${:,.0f}", "notional_usd": "${:,.0f}",
                     "unrealized_pnl": "${:,.0f}", "entry_price": "${:,.4f}"}),
                use_container_width=True, height=500, hide_index=True)
        elif top5_raw or bot5_raw:
            st.info("Top5/bottom5 fetched but no open positions found.")
        else:
            st.info("Top5/bottom5 is fetched during live API scan. "
                    "Click 'Fetch Fresh Data' with demo mode off.")


main()
