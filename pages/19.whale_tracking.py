"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           CRYPTO WHALE TRACKER  —  Hyperliquid Perpetuals                  ║
║           Single-file Streamlit page  (drop into pages/ folder)            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Data source : Hyperliquid public API — free, no auth required             ║
║                https://api.hyperliquid.xyz/info                            ║
║  Strategy    : Divergence between elite wallets (top N by PnL)             ║
║                and contra wallets (bottom N by PnL)                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  SETUP                                                                     ║
║  pip install streamlit pandas numpy plotly requests                        ║
║                                                                            ║
║  Standalone : streamlit run whale_tracker.py                               ║
║  Multi-page : copy to   pages/whale_tracker.py                             ║
║               Comment out st.set_page_config() below if your main         ║
║               app.py already calls it.                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — PAGE CONFIG
#  Comment out if your main app.py already calls st.set_page_config()
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Crypto Whale Tracker",
    page_icon="🐋",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — THEME  (mirrors .streamlit/config.toml, fully inline)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

:root {
    --bg:      #0a0e1a;
    --card:    #111827;
    --border:  #1e3a5f;
    --text:    #e2e8f0;
    --muted:   #64748b;
    --green:   #00ff88;
    --red:     #ff4444;
    --yellow:  #fbbf24;
    --blue:    #38bdf8;
}
html, body, [class*="css"]         { font-family: 'Inter', sans-serif; }
.stApp                             { background-color: var(--bg); }
section[data-testid="stSidebar"]  { background: #0d1117;
                                     border-right: 1px solid var(--border); }

.wt-header {
    background: linear-gradient(90deg,#0a0e1a 0%,#0d1f3c 50%,#0a0e1a 100%);
    border-bottom: 1px solid var(--border);
    padding: 18px 24px 14px; margin-bottom: 24px;
    border-radius: 0 0 12px 12px;
}
.wt-header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 1.9rem; letter-spacing: 2px; color: var(--green); margin: 0;
}
.wt-header p { color: var(--muted); font-size:0.82rem; margin:4px 0 0; letter-spacing:1px; }

.wt-section {
    font-family: 'Space Mono', monospace; font-size: 0.82rem;
    letter-spacing: 2px; color: var(--muted); text-transform: uppercase;
    border-bottom: 1px solid var(--border); padding-bottom: 6px; margin: 24px 0 14px;
}
.wt-kpi {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px 20px; text-align: center;
}
.wt-kpi .val { font-family:'Space Mono',monospace; font-size:1.65rem; font-weight:700; line-height:1.1; }
.wt-kpi .lbl { font-size:0.70rem; color:var(--muted); letter-spacing:1.4px;
                text-transform:uppercase; margin-top:5px; }
.wt-coin-card {
    background: var(--card); border: 1px solid #064e3b;
    border-radius: 10px; padding: 20px;
}
.wt-coin-card table { width:100%; border-collapse:collapse; font-size:0.85rem; }
.wt-coin-card td    { padding: 5px 0; }
.wt-mock-banner {
    background:#1c1a00; border:1px solid #854d0e; border-radius:8px;
    padding:10px 16px; margin-bottom:18px; color:var(--yellow); font-size:0.82rem;
}
.wt-disclaimer { font-size:0.70rem; color:#475569; line-height:1.7; }
.stDataFrame { border-radius:8px; overflow:hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — STRATEGY PARAMETER DEFAULTS
#  These are the fallback defaults; actual values come from the sidebar inputs
#  at runtime and are stored in st.session_state.
# ══════════════════════════════════════════════════════════════════════════════

_DEFAULT_N_ELITE      = 20
_DEFAULT_N_CONTRA     = 20
_DEFAULT_MIN_WALLETS  = 3
_DEFAULT_DIV_THRESH   = 40


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — HYPERLIQUID API  (free, public, no key needed)
# ══════════════════════════════════════════════════════════════════════════════

_API_URL = "https://api.hyperliquid.xyz/info"
_HEADERS = {"Content-Type": "application/json"}
_BATCH   = 10      # wallet addresses per batch
_DELAY   = 0.25    # seconds between batches


def _post(payload: dict, timeout: int = 20):
    try:
        r = requests.post(_API_URL, json=payload, headers=_HEADERS, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def fetch_leaderboard() -> pd.DataFrame:
    """Leaderboard sorted descending by PnL. Falls back to mock on API failure."""
    data = _post({"type": "leaderboard"})
    if data and "leaderboardRows" in data:
        rows = [
            {
                "address":       r.get("ethAddress", ""),
                "pnl":           float(r.get("pnl", 0)),
                "account_value": float(r.get("accountValue", 0)),
            }
            for r in data["leaderboardRows"]
            if r.get("ethAddress")
        ]
        df = pd.DataFrame(rows)
        df = df[df["address"] != ""].copy()
        if not df.empty:
            return df.sort_values("pnl", ascending=False).reset_index(drop=True)
    return _mock_leaderboard()


def fetch_positions(address: str) -> list:
    """Open positions for one wallet address."""
    data = _post({"type": "clearinghouseState", "user": address})
    if not data or "assetPositions" not in data:
        return []
    out = []
    for ap in data.get("assetPositions", []):
        p    = ap.get("position", {})
        size = float(p.get("szi", 0))
        if size == 0:
            continue
        lev_raw  = p.get("leverage", {})
        leverage = (float(lev_raw.get("value", lev_raw.get("rawUsd", 1)))
                    if isinstance(lev_raw, dict) else float(lev_raw or 1))
        out.append({
            "coin":           p.get("coin", "UNKNOWN"),
            "direction":      "long" if size > 0 else "short",
            "size":           size,
            "entry_px":       float(p.get("entryPx", 0)),
            "unrealized_pnl": float(p.get("unrealizedPnl", 0)),
            "leverage":       round(leverage, 1),
            "notional":       abs(float(p.get("positionValue", 0))),
        })
    return out


def fetch_funding_rates() -> dict:
    """Returns {coin: funding_rate_8h_%} for all assets."""
    data = _post({"type": "metaAndAssetCtxs"})
    if not data or not isinstance(data, list) or len(data) < 2:
        return {}
    universe = data[0].get("universe", [])
    return {
        universe[i].get("name", ""): float(ctx.get("funding", 0)) * 100
        for i, ctx in enumerate(data[1]) if i < len(universe)
    }


def fetch_prices() -> dict:
    """Returns {coin: mid_price_usd}."""
    data = _post({"type": "allMids"})
    return {k: float(v) for k, v in data.items()} if isinstance(data, dict) else {}


def _build_position_matrix(addresses: list) -> tuple:
    """
    Batch-fetches positions. Returns (dir_df, notional_df, raw_dict).
    dir_df: wallets × coins, values +1=long / -1=short / 0=flat
    """
    raw, all_coins = {}, set()
    for i, addr in enumerate(addresses):
        positions = fetch_positions(addr)
        raw[addr] = positions
        all_coins.update(p["coin"] for p in positions)
        if (i + 1) % _BATCH == 0:
            time.sleep(_DELAY)

    if not all_coins:
        empty = pd.DataFrame(index=addresses)
        return empty, empty, raw

    coins = sorted(all_coins)
    dir_rows, not_rows = {}, {}
    for addr, positions in raw.items():
        pos_map = {p["coin"]: p for p in positions}
        dir_rows[addr] = {c: (1 if pos_map[c]["direction"] == "long" else -1)
                           if c in pos_map else 0 for c in coins}
        not_rows[addr] = {c: pos_map[c]["notional"] if c in pos_map else 0.0
                           for c in coins}

    return (pd.DataFrame(dir_rows, index=coins).T,
            pd.DataFrame(not_rows, index=coins).T,
            raw)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — SIGNAL ENGINE
# ══════════════════════════════════════════════════════════════════════════════

_SIG_COLS = [
    "coin", "signal", "divergence_score", "elite_wallets", "elite_long_pct",
    "contra_wallets", "contra_long_pct", "elite_notional_usd",
    "funding_rate_8h_pct", "price_usd",
]


def _empty_signals() -> pd.DataFrame:
    """Return a correctly-shaped empty DataFrame so downstream code never crashes."""
    return pd.DataFrame(columns=_SIG_COLS)


def compute_signals(
    leaderboard: pd.DataFrame,
    n_elite: int,
    n_contra: int,
    min_wallets: int,
    div_threshold: int,
    funding_rates: dict = None,
    prices: dict = None,
    use_mock: bool = False,
) -> tuple:
    """
    Core signal computation.

    divergence_score  =  elite_long_pct  −  contra_long_pct
    LONG   :  score ≥ +div_threshold
    SHORT  :  score ≤ −div_threshold
    NEUTRAL:  |score| < div_threshold

    All 4 strategy parameters are passed explicitly — no hidden globals.
    Returns (signals_df, meta_dict).
    """
    if use_mock:
        return _mock_signals(div_threshold), _mock_meta(n_elite, n_contra)

    t0           = time.time()
    elite_addrs  = leaderboard.head(n_elite)["address"].tolist()
    contra_addrs = leaderboard.tail(n_contra)["address"].tolist()

    elite_dir,  elite_not,  elite_raw  = _build_position_matrix(elite_addrs)
    contra_dir, contra_not, contra_raw = _build_position_matrix(contra_addrs)

    all_coins = list(set(elite_dir.columns) | set(contra_dir.columns))
    funding   = funding_rates or {}
    px        = prices or {}
    rows      = []

    for coin in all_coins:
        e_col = (elite_dir[coin]  if coin in elite_dir.columns
                 else pd.Series(0, index=elite_addrs))
        c_col = (contra_dir[coin] if coin in contra_dir.columns
                 else pd.Series(0, index=contra_addrs))

        e_pos = e_col[e_col != 0]
        c_pos = c_col[c_col != 0]
        if len(e_pos) + len(c_pos) < min_wallets:
            continue

        e_long_pct = (e_pos > 0).sum() / max(len(e_pos), 1) * 100
        c_long_pct = (c_pos > 0).sum() / max(len(c_pos), 1) * 100
        div_score  = e_long_pct - c_long_pct

        e_not_col = (elite_not[coin] if coin in elite_not.columns
                     else pd.Series(0, index=elite_addrs))

        rows.append({
            "coin":                coin,
            "signal":              ("LONG"  if div_score >= div_threshold  else
                                    "SHORT" if div_score <= -div_threshold else "NEUTRAL"),
            "divergence_score":    round(div_score, 1),
            "elite_wallets":       len(e_pos),
            "elite_long_pct":      round(e_long_pct, 1),
            "contra_wallets":      len(c_pos),
            "contra_long_pct":     round(c_long_pct, 1),
            "elite_notional_usd":  round(float(e_not_col.sum())),
            "funding_rate_8h_pct": round(funding.get(coin, 0), 4),
            "price_usd":           px.get(coin),
        })

    # ── BUGFIX: guard against empty rows before sort ──────────────────────────
    if not rows:
        signals_df = _empty_signals()
    else:
        signals_df = (pd.DataFrame(rows)
                        .sort_values("divergence_score", ascending=False)
                        .reset_index(drop=True))

    meta = {
        "elite_addresses":  elite_addrs,
        "contra_addresses": contra_addrs,
        "elite_raw":        elite_raw,
        "contra_raw":       contra_raw,
        "fetch_seconds":    round(time.time() - t0, 1),
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "is_mock":          False,
        "leaderboard":      leaderboard,
        "params":           dict(n_elite=n_elite, n_contra=n_contra,
                                 min_wallets=min_wallets, div_threshold=div_threshold),
    }
    return signals_df, meta


def _build_wallet_detail(raw: dict, leaderboard: pd.DataFrame, group: str) -> pd.DataFrame:
    rows = []
    for addr, positions in raw.items():
        pnl_row    = leaderboard[leaderboard["address"] == addr]
        wallet_pnl = float(pnl_row["pnl"].values[0]) if not pnl_row.empty else 0.0
        for p in positions:
            rows.append({
                "wallet":         addr[:8] + "…" + addr[-6:],
                "group":          group,
                "wallet_pnl_usd": round(wallet_pnl),
                "coin":           p["coin"],
                "direction":      p["direction"].upper(),
                "notional":       round(p["notional"]),
                "leverage":       p["leverage"],
                "entry_px":       p["entry_px"],
                "upnl":           round(p["unrealized_pnl"]),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


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
        "address":       [f"0x{i:040x}" for i in range(1, n + 1)],
        "pnl":           pnls,
        "account_value": np.abs(pnls) * rng.uniform(2, 6, n),
    })


def _mock_signals(div_threshold: int) -> pd.DataFrame:
    seed = int(datetime.now().strftime("%Y%m%d"))   # stable within a day
    rng  = np.random.default_rng(seed)
    rows = []
    for coin in _COINS:
        e_long = rng.uniform(10, 90)
        c_long = rng.uniform(10, 90)
        div    = e_long - c_long
        rows.append({
            "coin":                coin,
            "signal":              ("LONG"  if div >= div_threshold  else
                                    "SHORT" if div <= -div_threshold else "NEUTRAL"),
            "divergence_score":    round(div, 1),
            "elite_wallets":       int(rng.integers(3, 14)),
            "elite_long_pct":      round(e_long, 1),
            "contra_wallets":      int(rng.integers(3, 11)),
            "contra_long_pct":     round(c_long, 1),
            "elite_notional_usd":  int(rng.integers(500_000, 40_000_000)),
            "funding_rate_8h_pct": round(float(rng.uniform(-0.05, 0.15)), 4),
            "price_usd":           _MOCK_PRICES.get(coin),
        })
    return (pd.DataFrame(rows)
              .sort_values("divergence_score", ascending=False)
              .reset_index(drop=True))


def _mock_meta(n_elite: int, n_contra: int) -> dict:
    rng     = np.random.default_rng(42)
    e_addrs = [f"0x{i:040x}" for i in range(1, n_elite + 1)]
    c_addrs = [f"0x{i:040x}" for i in range(181, 181 + n_contra)]

    def _fake_pos(addr_list, bias_long):
        raw = {}
        for addr in addr_list:
            coins = rng.choice(_COINS, int(rng.integers(1, 6)), replace=False).tolist()
            raw[addr] = [{
                "coin":           c,
                "direction":      "long" if (rng.random() < (0.70 if bias_long else 0.30)) else "short",
                "size":           round(rng.uniform(0.01, 5.0), 4),
                "entry_px":       round(_MOCK_PRICES.get(c, 1.0) * rng.uniform(0.85, 1.05), 4),
                "unrealized_pnl": round(float(rng.uniform(-5000, 20000)), 2),
                "leverage":       int(rng.integers(2, 20)),
                "notional":       round(rng.uniform(0.01, 5.0) * _MOCK_PRICES.get(c, 1.0), 2),
            } for c in coins]
        return raw

    lb = _mock_leaderboard()
    return {
        "elite_addresses":  e_addrs,
        "contra_addresses": c_addrs,
        "elite_raw":        _fake_pos(e_addrs, bias_long=True),
        "contra_raw":       _fake_pos(c_addrs, bias_long=False),
        "fetch_seconds":    2.4,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "is_mock":          True,
        "leaderboard":      lb,
        "params":           dict(n_elite=n_elite, n_contra=n_contra,
                                 min_wallets=_DEFAULT_MIN_WALLETS,
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

_BASE  = dict(
    paper_bgcolor=_BG, plot_bgcolor=_BG,
    font=dict(color=_TEXT, family="Inter, sans-serif", size=12),
    margin=dict(l=12, r=12, t=36, b=12),
)


def _ax(**kw):
    return dict(gridcolor=_GRID, zerolinecolor=_GRID,
                tickcolor=_MUTED, linecolor=_GRID, **kw)


def _chart_divergence(df: pd.DataFrame, div_threshold: int) -> go.Figure:
    d      = df.sort_values("divergence_score")
    colors = [_GREEN if v >= div_threshold else
              _RED   if v <= -div_threshold else _MUTED
              for v in d["divergence_score"]]
    fig = go.Figure(go.Bar(
        x=d["divergence_score"], y=d["coin"], orientation="h",
        marker_color=colors,
        text=[f"{v:+.0f}" for v in d["divergence_score"]],
        textposition="outside",
        textfont=dict(size=10, family="Space Mono, monospace"),
        hovertemplate="<b>%{y}</b><br>Divergence: %{x:.1f}<extra></extra>",
    ))
    fig.add_vline(x= div_threshold, line=dict(color=_GREEN, dash="dot", width=1))
    fig.add_vline(x=-div_threshold, line=dict(color=_RED,   dash="dot", width=1))
    fig.add_vline(x=0,              line=dict(color=_MUTED,  width=0.5))
    fig.update_layout(
        **_BASE,
        title=dict(text="Divergence Score — All Coins",
                   font=dict(size=13, color=_MUTED), x=0),
        xaxis=dict(title="← Bearish  |  Score  |  Bullish →",
                   range=[-110, 110], **_ax()),
        yaxis=dict(tickfont=dict(family="Space Mono, monospace", size=10), **_ax()),
        height=max(360, len(d) * 22), showlegend=False,
    )
    return fig


def _chart_positioning_map(df: pd.DataFrame) -> go.Figure:
    sig_color = {"LONG": _GREEN, "SHORT": _RED, "NEUTRAL": _MUTED}
    colors    = [sig_color[s] for s in df["signal"]]
    sizes     = np.clip(np.log1p(df["elite_notional_usd"] / 1e5) * 6, 8, 40)
    fig = go.Figure()
    for rect, fill in [((50,0,100,50), "rgba(0,255,136,0.04)"),
                        ((0,50,50,100), "rgba(255,68,68,0.04)")]:
        fig.add_shape(type="rect", x0=rect[0], y0=rect[1],
                      x1=rect[2], y1=rect[3], fillcolor=fill, line_width=0)
    fig.add_trace(go.Scatter(
        x=df["elite_long_pct"], y=df["contra_long_pct"],
        mode="markers+text",
        marker=dict(color=colors, size=sizes, opacity=0.85,
                    line=dict(color=_BG, width=1)),
        text=df["coin"],
        textposition="top center",
        textfont=dict(size=9, family="Space Mono, monospace"),
        customdata=np.stack([df["divergence_score"],
                             df["elite_notional_usd"] / 1e6, df["signal"]], axis=-1),
        hovertemplate=(
            "<b>%{text}</b><br>Elite long: %{x:.1f}%<br>Contra long: %{y:.1f}%<br>"
            "Divergence: %{customdata[0]:+.1f}<br>Smart $: $%{customdata[1]:.1f}M<br>"
            "Signal: %{customdata[2]}<extra></extra>"
        ),
    ))
    fig.add_shape(type="line", x0=0, y0=0, x1=100, y1=100,
                  line=dict(color=_MUTED, dash="dot", width=1))
    for txt, xp, yp, col in [("LONG ZONE", 85, 5, _GREEN),
                               ("SHORT ZONE", 5, 92, _RED)]:
        fig.add_annotation(x=xp, y=yp, text=txt, showarrow=False,
                           font=dict(color=col, size=8,
                                     family="Space Mono, monospace"), opacity=0.5)
    fig.update_layout(
        **_BASE,
        title=dict(text="Smart Money vs Contra — Positioning Map",
                   font=dict(size=13, color=_MUTED), x=0),
        xaxis=dict(title="Elite Wallets — % Net Long", range=[0, 100], **_ax()),
        yaxis=dict(title="Contra Wallets — % Net Long", range=[0, 100], **_ax()),
        height=480,
    )
    return fig


def _chart_coin_breakdown(df: pd.DataFrame, coin: str) -> go.Figure:
    row = df[df["coin"] == coin]
    if row.empty:
        return go.Figure()
    row     = row.iloc[0]
    cats    = ["Long", "Short"]
    e_vals  = [row["elite_long_pct"],  100 - row["elite_long_pct"]]
    c_vals  = [row["contra_long_pct"], 100 - row["contra_long_pct"]]
    sig_col = _GREEN if row["signal"] == "LONG" else (_RED if row["signal"] == "SHORT" else _MUTED)
    fig = go.Figure()
    for name, vals, op in [("Elite Wallets", e_vals, 0.85),
                            ("Contra Wallets", c_vals, 0.35)]:
        fig.add_trace(go.Bar(
            name=name, x=cats, y=vals, marker_color=[_GREEN, _RED], opacity=op,
            text=[f"{v:.1f}%" for v in vals], textposition="auto",
        ))
    fig.update_layout(
        **_BASE,
        title=dict(
            text=f"{coin}  |  Div: {row['divergence_score']:+.1f}  |  {row['signal']}",
            font=dict(size=14, color=sig_col, family="Space Mono, monospace"), x=0,
        ),
        barmode="group",
        yaxis=dict(title="% of Wallets", range=[0, 110], **_ax()),
        xaxis=dict(**_ax()),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        height=320,
    )
    return fig


def _chart_notional(long_df: pd.DataFrame, short_df: pd.DataFrame) -> go.Figure:
    long_df  = long_df.nlargest(8, "elite_notional_usd").copy()
    short_df = short_df.nlargest(8, "elite_notional_usd").copy()
    fig = go.Figure()
    for name, df_part, mult, color in [
        ("Long Signals",  long_df,   1,  _GREEN),
        ("Short Signals", short_df, -1,  _RED),
    ]:
        fig.add_trace(go.Bar(
            name=name,
            x=df_part["elite_notional_usd"] / 1e6 * mult,
            y=df_part["coin"], orientation="h",
            marker_color=color, opacity=0.8,
            text=[f"${v/1e6:.1f}M" for v in df_part["elite_notional_usd"]],
            textposition="auto",
        ))
    fig.update_layout(
        **_BASE,
        title=dict(text="Smart Money Notional — Signal Coins ($M)",
                   font=dict(size=13, color=_MUTED), x=0),
        xaxis=dict(title="← Short  |  USD Millions  |  Long →", **_ax()),
        yaxis=dict(tickfont=dict(family="Space Mono, monospace", size=10), **_ax()),
        barmode="relative", height=380,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def _chart_funding(df: pd.DataFrame) -> go.Figure:
    d = df[df["signal"].isin(["LONG", "SHORT"])].sort_values(
        "funding_rate_8h_pct", ascending=False)
    if d.empty:
        return go.Figure()
    colors = [
        _GREEN if (r["signal"] == "LONG"  and r["funding_rate_8h_pct"] < 0) else
        _RED   if (r["signal"] == "SHORT" and r["funding_rate_8h_pct"] > 0) else _YLW
        for _, r in d.iterrows()
    ]
    fig = go.Figure(go.Bar(
        x=d["coin"], y=d["funding_rate_8h_pct"], marker_color=colors,
        text=[f"{v:+.4f}%" for v in d["funding_rate_8h_pct"]],
        textposition="outside", textfont=dict(size=9, family="Space Mono, monospace"),
        hovertemplate="<b>%{x}</b><br>Funding (8h): %{y:.4f}%<extra></extra>",
    ))
    for y_val in [0.01, -0.01]:
        fig.add_hline(y=y_val, line=dict(color=_YLW, dash="dot", width=1))
    fig.update_layout(
        **_BASE,
        title=dict(text="Funding Rates — Signal Coins (8h %)",
                   font=dict(size=13, color=_MUTED), x=0),
        yaxis=dict(title="Funding Rate %", **_ax()),
        xaxis=dict(**_ax()), height=320,
    )
    return fig


def _chart_leaderboard(lb: pd.DataFrame, n_elite: int, n_contra: int) -> go.Figure:
    fig = go.Figure()
    for x, name, color, opacity, nbins in [
        (lb["pnl"].values,               "All Wallets",             _MUTED, 0.40, 40),
        (lb.head(n_elite)["pnl"].values,  f"Elite (Top {n_elite})",  _GREEN, 0.85, 20),
        (lb.tail(n_contra)["pnl"].values, f"Contra (Bot {n_contra})", _RED,  0.85, 20),
    ]:
        fig.add_trace(go.Histogram(x=x, name=name, marker_color=color,
                                   opacity=opacity, nbinsx=nbins))
    fig.update_layout(
        **_BASE,
        title=dict(text="Leaderboard PnL Distribution",
                   font=dict(size=13, color=_MUTED), x=0),
        xaxis=dict(title="Total PnL (USD)", **_ax()),
        yaxis=dict(title="Wallet Count", **_ax()),
        barmode="overlay", legend=dict(bgcolor="rgba(0,0,0,0)"), height=300,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — DATA LOADING  (session_state cache — survives page switches)
#
#  We use st.session_state rather than st.cache_data so that:
#   - Data persists when the user navigates away and returns
#   - The refresh button is the ONLY trigger for a new API call
#   - Changing sidebar parameters updates the display instantly without
#     re-hitting the API (signals are recomputed from cached raw data)
# ══════════════════════════════════════════════════════════════════════════════

_SS_KEY = "whale_tracker_cache"   # session_state key


def _needs_fetch(force_mock: bool) -> bool:
    """True if we have no cached data or the data mode has changed."""
    if _SS_KEY not in st.session_state:
        return True
    cached = st.session_state[_SS_KEY]
    # Refetch if mode changed (live ↔ mock)
    if cached.get("is_mock") != force_mock:
        return True
    return False


def _run_fetch(n_elite: int, n_contra: int, min_wallets: int,
               div_threshold: int, force_mock: bool) -> dict:
    """
    Hits the Hyperliquid API (or mock), stores raw result in session_state,
    returns the cache dict.
    """
    if force_mock:
        signals, meta = compute_signals(
            pd.DataFrame(), n_elite, n_contra, min_wallets, div_threshold,
            use_mock=True,
        )
        meta["is_mock"] = True
    else:
        lb = fetch_leaderboard()
        # Detect real vs mock fallback (real addresses are 42-char 0x… strings)
        sample  = lb["address"].iloc[0] if len(lb) > 0 else ""
        is_live = len(sample) == 42 and sample.startswith("0x")

        if not is_live:
            signals, meta = compute_signals(
                lb, n_elite, n_contra, min_wallets, div_threshold, use_mock=True,
            )
            meta["is_mock"] = True
        else:
            funding = fetch_funding_rates()
            prices  = fetch_prices()
            signals, meta = compute_signals(
                lb, n_elite, n_contra, min_wallets, div_threshold, funding, prices,
            )
            meta["is_mock"] = False

    cache = {**meta, "signals": signals, "is_mock": meta["is_mock"]}
    st.session_state[_SS_KEY] = cache
    return cache


def _recompute_signals(cache: dict, n_elite: int, n_contra: int,
                        min_wallets: int, div_threshold: int) -> pd.DataFrame:
    """
    Re-runs the divergence calculation on already-fetched raw position data,
    so parameter changes are instant without another API round-trip.
    """
    if cache.get("is_mock"):
        return _mock_signals(div_threshold)

    lb           = cache.get("leaderboard", pd.DataFrame())
    elite_addrs  = lb.head(n_elite)["address"].tolist()
    contra_addrs = lb.tail(n_contra)["address"].tolist()
    elite_raw    = cache.get("elite_raw", {})
    contra_raw   = cache.get("contra_raw", {})
    funding      = {}   # funding/prices not re-fetched on param change
    px           = {}

    # Rebuild direction matrices from cached raw positions
    def _raw_to_matrix(raw_dict, addr_list):
        all_coins = sorted({p["coin"] for positions in raw_dict.values()
                            for p in positions})
        if not all_coins:
            return pd.DataFrame(index=addr_list), pd.DataFrame(index=addr_list)
        dir_rows, not_rows = {}, {}
        for addr in addr_list:
            positions = raw_dict.get(addr, [])
            pos_map   = {p["coin"]: p for p in positions}
            dir_rows[addr] = {c: (1 if pos_map[c]["direction"] == "long" else -1)
                               if c in pos_map else 0 for c in all_coins}
            not_rows[addr] = {c: pos_map[c]["notional"] if c in pos_map else 0.0
                               for c in all_coins}
        return (pd.DataFrame(dir_rows, index=all_coins).T,
                pd.DataFrame(not_rows, index=all_coins).T)

    elite_dir,  elite_not  = _raw_to_matrix(elite_raw,  elite_addrs)
    contra_dir, contra_not = _raw_to_matrix(contra_raw, contra_addrs)

    all_coins = list(set(elite_dir.columns) | set(contra_dir.columns))
    rows = []
    for coin in all_coins:
        e_col = (elite_dir[coin]  if coin in elite_dir.columns
                 else pd.Series(0, index=elite_addrs))
        c_col = (contra_dir[coin] if coin in contra_dir.columns
                 else pd.Series(0, index=contra_addrs))
        e_pos = e_col[e_col != 0]
        c_pos = c_col[c_col != 0]
        if len(e_pos) + len(c_pos) < min_wallets:
            continue
        e_long = (e_pos > 0).sum() / max(len(e_pos), 1) * 100
        c_long = (c_pos > 0).sum() / max(len(c_pos), 1) * 100
        div    = e_long - c_long
        e_not_col = (elite_not[coin] if coin in elite_not.columns
                     else pd.Series(0, index=elite_addrs))
        rows.append({
            "coin":                coin,
            "signal":              ("LONG"  if div >= div_threshold  else
                                    "SHORT" if div <= -div_threshold else "NEUTRAL"),
            "divergence_score":    round(div, 1),
            "elite_wallets":       len(e_pos),
            "elite_long_pct":      round(e_long, 1),
            "contra_wallets":      len(c_pos),
            "contra_long_pct":     round(c_long, 1),
            "elite_notional_usd":  round(float(e_not_col.sum())),
            "funding_rate_8h_pct": round(funding.get(coin, 0), 4),
            "price_usd":           px.get(coin),
        })

    if not rows:
        return _empty_signals()
    return (pd.DataFrame(rows)
              .sort_values("divergence_score", ascending=False)
              .reset_index(drop=True))


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — UI COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

def _render_sidebar() -> tuple:
    """
    Draws sidebar with parameter inputs and refresh button.
    Returns (n_elite, n_contra, min_wallets, div_threshold, force_mock, do_refresh)
    """
    with st.sidebar:
        st.markdown(
            "<div style='font-family:Space Mono,monospace;color:#00ff88;"
            "font-size:1.1rem;letter-spacing:2px;margin-bottom:8px;'>"
            "🐋 WHALE TRACKER</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        # ── Strategy Parameters ───────────────────────────────────────────────
        st.markdown("**Strategy Parameters**")
        st.caption("Adjust and click Refresh to apply to a new API fetch. "
                   "Parameter changes on existing data apply instantly.")

        n_elite = st.slider(
            "Elite wallets (top N by PnL)",
            min_value=5, max_value=50, value=_DEFAULT_N_ELITE, step=5,
            help="Wallets ranked in the top N by total PnL — your signal generators",
        )
        n_contra = st.slider(
            "Contra wallets (bottom N by PnL)",
            min_value=5, max_value=50, value=_DEFAULT_N_CONTRA, step=5,
            help="Wallets ranked in the bottom N — used as a fade signal",
        )
        min_wallets = st.slider(
            "Min wallets per coin",
            min_value=2, max_value=10, value=_DEFAULT_MIN_WALLETS, step=1,
            help="Coin must appear in at least this many wallets to generate a signal",
        )
        div_threshold = st.slider(
            "Divergence threshold (±)",
            min_value=10, max_value=70, value=_DEFAULT_DIV_THRESH, step=5,
            help="Score must exceed ±this value to declare LONG or SHORT",
        )

        st.divider()

        # ── Signal formula reminder ───────────────────────────────────────────
        st.markdown(
            f"<div style='font-size:0.78rem;color:#94a3b8;line-height:1.9;'>"
            f"<code style='color:#00ff88'>score = elite_long% − contra_long%</code><br>"
            f"✅ LONG  → score ≥ +{div_threshold}<br>"
            f"🚫 SHORT → score ≤ −{div_threshold}<br>"
            f"⚪ NEUTRAL → |score| &lt; {div_threshold}</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        # ── Controls ─────────────────────────────────────────────────────────
        force_mock = st.toggle(
            "Use demo data", value=False,
            help="Simulate without hitting the Hyperliquid API",
        )
        do_refresh = st.button("🔄 Fetch Fresh Data", use_container_width=True,
                               help="Re-run the full API scan. "
                                    "Parameter changes alone do not require a refresh.")

        # ── Cache status ──────────────────────────────────────────────────────
        if _SS_KEY in st.session_state:
            cached = st.session_state[_SS_KEY]
            ts     = cached.get("timestamp", "")
            if ts:
                dt = datetime.fromisoformat(ts)
                age_min = (datetime.now(timezone.utc) - dt).seconds // 60
                st.caption(f"📦 Cache: {dt.strftime('%H:%M UTC')} "
                           f"({age_min}m ago)")
        else:
            st.caption("📦 No cached data yet")

        st.divider()
        st.markdown(
            "<div class='wt-disclaimer'>⚠️ Not financial advice.<br>"
            "Research purposes only.<br>Manage your own risk.</div>",
            unsafe_allow_html=True,
        )

    return n_elite, n_contra, min_wallets, div_threshold, force_mock, do_refresh


def _render_header(meta: dict):
    ts   = meta.get("timestamp", "")
    dt_s = (datetime.fromisoformat(ts).strftime("%d %b %Y, %H:%M UTC")
            if ts else "—")
    mode, col = ("DEMO DATA", "#fbbf24") if meta.get("is_mock") else ("LIVE DATA", "#00ff88")
    p    = meta.get("params", {})
    st.markdown(
        f"<div class='wt-header'>"
        f"<h1>🐋 CRYPTO WHALE TRACKER</h1>"
        f"<p>HYPERLIQUID PERPETUALS · {dt_s} · "
        f"<span style='color:{col};'>{mode}</span> · "
        f"Elite {p.get('n_elite','?')} · Contra {p.get('n_contra','?')} · "
        f"Threshold ±{p.get('div_threshold','?')} · "
        f"Scan {meta.get('fetch_seconds','—')}s</p>"
        f"</div>",
        unsafe_allow_html=True,
    )
    if meta.get("is_mock"):
        st.markdown(
            "<div class='wt-mock-banner'>⚠️ <b>Demo Mode:</b> Displaying simulated data. "
            "Toggle off 'Use demo data' in the sidebar, then click "
            "'Fetch Fresh Data' to connect to the live Hyperliquid API.</div>",
            unsafe_allow_html=True,
        )


def _render_kpis(signals: pd.DataFrame):
    long_df   = signals[signals["signal"] == "LONG"]
    short_df  = signals[signals["signal"] == "SHORT"]
    total_not = long_df["elite_notional_usd"].sum() if not long_df.empty else 0
    avg_elite = signals["elite_long_pct"].mean() if not signals.empty else 50
    sentiment = ("BULLISH" if avg_elite > 60 else
                 "BEARISH" if avg_elite < 40 else "NEUTRAL")
    s_col     = _GREEN if sentiment == "BULLISH" else (_RED if sentiment == "BEARISH" else _YLW)

    kpis = [
        (str(len(long_df)),  "LONG SIGNALS",  _GREEN),
        (str(len(short_df)), "SHORT SIGNALS", _RED),
        (str(len(long_df) + len(short_df)), "TOTAL SIGNALS", _BLUE),
        (f"${total_not/1e6:.1f}M", "SMART $ IN LONGS", _GREEN),
        (f"<span style='color:{s_col};'>{sentiment}</span>", "MARKET BIAS", _MUTED),
    ]
    for col, (val, lbl, color) in zip(st.columns(5), kpis):
        col.markdown(
            f"<div class='wt-kpi'>"
            f"<div class='val' style='color:{color};'>{val}</div>"
            f"<div class='lbl'>{lbl}</div></div>",
            unsafe_allow_html=True,
        )


def _render_signal_table(df: pd.DataFrame, signal_type: str):
    filtered = df[df["signal"] == signal_type].copy()
    if filtered.empty:
        st.info(f"No {signal_type} signals at the current threshold.")
        return
    display = filtered[[
        "coin","divergence_score","elite_wallets","elite_long_pct",
        "contra_wallets","contra_long_pct","elite_notional_usd",
        "funding_rate_8h_pct","price_usd",
    ]].copy()
    display.columns = [
        "Coin","Divergence","Elite #","Elite Long %",
        "Contra #","Contra Long %","Smart $ (USD)","Funding 8h %","Price (USD)",
    ]
    display["Divergence"]    = display["Divergence"].apply(lambda x: f"{x:+.1f}")
    display["Smart $ (USD)"] = display["Smart $ (USD)"].apply(
        lambda x: f"${x/1e6:.2f}M" if x >= 1e6 else f"${x/1e3:.0f}K")
    display["Elite Long %"]  = display["Elite Long %"].apply(lambda x: f"{x:.1f}%")
    display["Contra Long %"] = display["Contra Long %"].apply(lambda x: f"{x:.1f}%")
    display["Price (USD)"]   = display["Price (USD)"].apply(
        lambda x: (f"${x:,.4f}" if pd.notna(x) and x < 1
                   else f"${x:,.2f}" if pd.notna(x) else "—"))
    st.dataframe(display, use_container_width=True,
                 height=min(400, 55 + len(display) * 37), hide_index=True)


def _render_wallet_table(meta: dict, leaderboard: pd.DataFrame, group: str):
    raw = meta.get(f"{group}_raw", {})
    if not raw:
        st.info("No wallet data available.")
        return
    df = _build_wallet_detail(raw, leaderboard, group)
    if df.empty:
        st.info("No open positions found.")
        return
    df["notional"] = df["notional"].apply(lambda x: f"${x:,.0f}")
    df["upnl"]     = df["upnl"].apply(
        lambda x: f"+${x:,.0f}" if x >= 0 else f"-${abs(x):,.0f}")
    st.dataframe(
        df[["wallet","wallet_pnl_usd","coin","direction",
            "notional","leverage","entry_px","upnl"]],
        column_config={
            "wallet_pnl_usd": st.column_config.NumberColumn(
                "Wallet PnL ($)", format="$%.0f"),
            "leverage": st.column_config.NumberColumn("Lev", format="%dx"),
        },
        use_container_width=True, height=400, hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── 1. Sidebar (always first — reads parameters before any data call) ─────
    (n_elite, n_contra, min_wallets,
     div_threshold, force_mock, do_refresh) = _render_sidebar()

    # ── 2. Fetch or use cache ─────────────────────────────────────────────────
    #  Fetch conditions:
    #    a) No cache exists yet
    #    b) Data mode changed (live ↔ mock)
    #    c) User clicked "Fetch Fresh Data"
    #  Parameter slider changes do NOT trigger a re-fetch; signals are
    #  recomputed instantly from the cached raw position data.
    if do_refresh or _needs_fetch(force_mock):
        with st.spinner("Scanning Hyperliquid wallets…"):
            cache = _run_fetch(n_elite, n_contra, min_wallets,
                               div_threshold, force_mock)
    else:
        cache = st.session_state[_SS_KEY]

    # ── 3. Recompute signals on cached raw data with current slider values ────
    signals     = _recompute_signals(cache, n_elite, n_contra,
                                     min_wallets, div_threshold)
    leaderboard = cache.get("leaderboard", _mock_leaderboard())
    long_df     = signals[signals["signal"] == "LONG"].reset_index(drop=True)
    short_df    = signals[signals["signal"] == "SHORT"].reset_index(drop=True)

    # Patch params into meta for the header display
    meta        = {**cache, "params": dict(n_elite=n_elite, n_contra=n_contra,
                                            min_wallets=min_wallets,
                                            div_threshold=div_threshold)}

    # ── 4. Render ─────────────────────────────────────────────────────────────
    _render_header(meta)
    _render_kpis(signals)

    tab_ov, tab_lg, tab_sh, tab_dg, tab_raw = st.tabs([
        "📊  Overview",
        "🟢  Long Picks",
        "🔴  Short Picks",
        "🔬  Diagnostics",
        "📋  Raw Data",
    ])

    # ── Overview ──────────────────────────────────────────────────────────────
    with tab_ov:
        st.markdown("<div class='wt-section'>Divergence Scores — All Coins</div>",
                    unsafe_allow_html=True)
        st.plotly_chart(_chart_divergence(signals, div_threshold),
                        use_container_width=True, config={"displayModeBar": False})
        st.markdown("<div class='wt-section'>Smart Money vs Contra Positioning Map</div>",
                    unsafe_allow_html=True)
        if not signals.empty:
            st.plotly_chart(_chart_positioning_map(signals),
                            use_container_width=True, config={"displayModeBar": False})

    # ── Long Picks ────────────────────────────────────────────────────────────
    with tab_lg:
        st.markdown(
            f"<div class='wt-section'>Long Signals — "
            f"{len(long_df)} coins above +{div_threshold} divergence</div>",
            unsafe_allow_html=True,
        )
        _render_signal_table(signals, "LONG")

        if not long_df.empty:
            st.markdown("<div class='wt-section'>Drill-Down</div>",
                        unsafe_allow_html=True)
            selected = st.selectbox("Select coin", long_df["coin"].tolist(),
                                    key="lg_drill", label_visibility="collapsed")
            row = signals[signals["coin"] == selected].iloc[0]
            col_a, col_b = st.columns(2)
            with col_a:
                st.plotly_chart(_chart_coin_breakdown(signals, selected),
                                use_container_width=True,
                                config={"displayModeBar": False})
            with col_b:
                fr_col = _YLW if row["funding_rate_8h_pct"] > 0.05 else _GREEN
                st.markdown(
                    f"<div class='wt-coin-card'>"
                    f"<div style='font-family:Space Mono,monospace;color:#00ff88;"
                    f"font-size:1.4rem;font-weight:700;'>{selected}</div>"
                    f"<div style='color:#64748b;font-size:0.75rem;"
                    f"margin-bottom:16px;letter-spacing:1px;'>LONG SIGNAL</div>"
                    f"<table>"
                    f"<tr><td style='color:#64748b;'>Divergence Score</td>"
                    f"<td style='color:#00ff88;text-align:right;"
                    f"font-family:Space Mono,monospace;'>"
                    f"{row['divergence_score']:+.1f}</td></tr>"
                    f"<tr><td style='color:#64748b;'>Elite Wallets Long</td>"
                    f"<td style='color:#e2e8f0;text-align:right;'>"
                    f"{row['elite_long_pct']:.1f}%</td></tr>"
                    f"<tr><td style='color:#64748b;'>Contra Wallets Long</td>"
                    f"<td style='color:#e2e8f0;text-align:right;'>"
                    f"{row['contra_long_pct']:.1f}%</td></tr>"
                    f"<tr><td style='color:#64748b;'>Elite # in Position</td>"
                    f"<td style='color:#e2e8f0;text-align:right;'>"
                    f"{int(row['elite_wallets'])}</td></tr>"
                    f"<tr><td style='color:#64748b;'>Smart $ Notional</td>"
                    f"<td style='color:#e2e8f0;text-align:right;'>"
                    f"${row['elite_notional_usd']/1e6:.2f}M</td></tr>"
                    f"<tr><td style='color:#64748b;'>Funding Rate (8h)</td>"
                    f"<td style='color:{fr_col};text-align:right;'>"
                    f"{row['funding_rate_8h_pct']:+.4f}%</td></tr>"
                    f"</table></div>",
                    unsafe_allow_html=True,
                )

    # ── Short Picks ───────────────────────────────────────────────────────────
    with tab_sh:
        st.markdown(
            f"<div class='wt-section'>Short Signals — "
            f"{len(short_df)} coins below −{div_threshold} divergence</div>",
            unsafe_allow_html=True,
        )
        _render_signal_table(signals, "SHORT")

        if not short_df.empty:
            st.markdown("<div class='wt-section'>Funding Cost Check</div>",
                        unsafe_allow_html=True)
            st.plotly_chart(_chart_funding(signals),
                            use_container_width=True, config={"displayModeBar": False})
            st.markdown(
                "<div style='font-size:0.75rem;color:#64748b;padding:4px 0 16px;'>"
                "🟡 Yellow = funding works against the short. "
                "Prefer coins with negative funding (you earn while short).</div>",
                unsafe_allow_html=True,
            )
            st.markdown("<div class='wt-section'>Drill-Down</div>",
                        unsafe_allow_html=True)
            sel_s = st.selectbox("Select coin", short_df["coin"].tolist(),
                                 key="sh_drill", label_visibility="collapsed")
            st.plotly_chart(_chart_coin_breakdown(signals, sel_s),
                            use_container_width=True, config={"displayModeBar": False})

    # ── Diagnostics ───────────────────────────────────────────────────────────
    with tab_dg:
        st.markdown("<div class='wt-section'>Smart Money Notional — Signal Coins</div>",
                    unsafe_allow_html=True)
        if not long_df.empty or not short_df.empty:
            st.plotly_chart(_chart_notional(long_df, short_df),
                            use_container_width=True, config={"displayModeBar": False})

        st.markdown("<div class='wt-section'>Wallet Universe — PnL Distribution</div>",
                    unsafe_allow_html=True)
        st.plotly_chart(_chart_leaderboard(leaderboard, n_elite, n_contra),
                        use_container_width=True, config={"displayModeBar": False})

        st.markdown("<div class='wt-section'>Signal Summary</div>",
                    unsafe_allow_html=True)
        cx, cy, cz = st.columns(3)
        with cx:
            st.markdown("**Long Signals**")
            if not long_df.empty:
                st.dataframe(
                    long_df[["coin","divergence_score","elite_wallets","elite_long_pct"]]
                    .rename(columns={"divergence_score":"Score","elite_wallets":"Elite #",
                                     "elite_long_pct":"Elite Long%"}),
                    hide_index=True, use_container_width=True,
                )
        with cy:
            st.markdown("**Short Signals**")
            if not short_df.empty:
                st.dataframe(
                    short_df[["coin","divergence_score","elite_wallets","elite_long_pct"]]
                    .rename(columns={"divergence_score":"Score","elite_wallets":"Elite #",
                                     "elite_long_pct":"Elite Long%"}),
                    hide_index=True, use_container_width=True,
                )
        with cz:
            st.markdown("**Coverage**")
            total = len(signals)
            st.metric("Coins scanned", total)
            st.metric("With signals",  len(long_df) + len(short_df))
            st.metric("Signal rate",
                      f"{(len(long_df)+len(short_df))/max(total,1)*100:.1f}%")
            st.metric("Avg div (longs)",
                      f"{long_df['divergence_score'].mean():.1f}"
                      if not long_df.empty else "—")

    # ── Raw Data ──────────────────────────────────────────────────────────────
    with tab_raw:
        col_e, col_c = st.columns(2)
        with col_e:
            st.markdown("<div class='wt-section'>Elite Wallets — Open Positions</div>",
                        unsafe_allow_html=True)
            _render_wallet_table(meta, leaderboard, "elite")
        with col_c:
            st.markdown("<div class='wt-section'>Contra Wallets — Open Positions</div>",
                        unsafe_allow_html=True)
            _render_wallet_table(meta, leaderboard, "contra")

        st.markdown("<div class='wt-section'>Full Signal Table</div>",
                    unsafe_allow_html=True)
        st.dataframe(signals, use_container_width=True, hide_index=True)

        st.markdown("<div class='wt-section'>Leaderboard Snapshot (Top 50)</div>",
                    unsafe_allow_html=True)
        if not leaderboard.empty:
            st.dataframe(
                leaderboard.head(50).style.format(
                    {"pnl": "${:,.0f}", "account_value": "${:,.0f}"}),
                use_container_width=True, height=400,
            )


main()
