# Full Revised Crypto Whale Tracker — Complete Updated Code

```python
import io
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Crypto Whale Tracker",
    page_icon="🐋",
    layout="wide",
)


# ============================================================
# DEFAULTS
# ============================================================

_DEFAULT_N_ELITE = 20
_DEFAULT_MIN_WALLETS = 2
_DEFAULT_DIV_THRESH = 20

MIN_NOTIONAL_USD = 500
MAX_LEVERAGE_SIGNAL = 25

_INFO_URL = "https://api.hyperliquid.xyz/info"
_STATS_URL = "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard"
_HEADERS = {"Content-Type": "application/json"}


# ============================================================
# API HELPERS
# ============================================================


def _post(payload: dict):
    try:
        r = requests.post(_INFO_URL, json=payload, headers=_HEADERS, timeout=20)
        r.raise_for_status()
        return r.json()
    except:
        return None



def _get(url: str):
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        return r.json()
    except:
        return None


# ============================================================
# LEADERBOARD
# ============================================================


def fetch_leaderboard():

    data = _get(_STATS_URL)

    if not data:
        return pd.DataFrame()

    rows = data.get("leaderboardRows", data)

    out = []

    for r in rows:
        out.append(
            {
                "address": r.get("ethAddress", ""),
                "pnl": float(r.get("pnl", 0)),
                "account_value": float(r.get("accountValue", 0)),
            }
        )

    df = pd.DataFrame(out)

    return df.sort_values("pnl", ascending=False).reset_index(drop=True)


# ============================================================
# POSITIONS
# ============================================================


def fetch_positions(address: str, for_signal=True):

    data = _post({"type": "clearinghouseState", "user": address})

    if not data:
        return []

    out = []

    for ap in data.get("assetPositions", []):

        p = ap.get("position", {})

        size = float(p.get("szi", 0))

        if size == 0:
            continue

        notional = abs(float(p.get("positionValue", 0)))

        lev_raw = p.get("leverage", {})

        leverage = (
            float(lev_raw.get("value", 1))
            if isinstance(lev_raw, dict)
            else float(lev_raw)
        )

        if for_signal:

            if notional < MIN_NOTIONAL_USD:
                continue

            if leverage > MAX_LEVERAGE_SIGNAL:
                continue

        out.append(
            {
                "coin": p.get("coin"),
                "direction": "long" if size > 0 else "short",
                "size": size,
                "notional": notional,
                "entry_px": float(p.get("entryPx", 0)),
                "unrealized_pnl": float(p.get("unrealizedPnl", 0)),
                "leverage": leverage,
            }
        )

    return out


# ============================================================
# FUNDING + PRICES
# ============================================================


def fetch_funding_rates():

    data = _post({"type": "metaAndAssetCtxs"})

    if not data:
        return {}

    universe = data[0].get("universe", [])

    out = {}

    for i, ctx in enumerate(data[1]):
        if i < len(universe):
            out[universe[i]["name"]] = float(ctx.get("funding", 0)) * 100

    return out



def fetch_prices():

    data = _post({"type": "allMids"})

    if not data:
        return {}

    return {k: float(v) for k, v in data.items()}


# ============================================================
# BUILD POSITION MATRIX
# ============================================================


def _build_position_matrix(addresses):

    raw = {}
    all_coins = set()

    for addr in addresses:

        positions = fetch_positions(addr, for_signal=True)

        raw[addr] = positions

        for p in positions:
            all_coins.add(p["coin"])

    if not all_coins:
        return pd.DataFrame(), pd.DataFrame(), raw

    coins = sorted(all_coins)

    dir_rows = {}
    not_rows = {}

    for addr, positions in raw.items():

        pos_map = {p["coin"]: p for p in positions}

        dir_rows[addr] = {}
        not_rows[addr] = {}

        for c in coins:

            if c in pos_map:

                dir_rows[addr][c] = (
                    1 if pos_map[c]["direction"] == "long" else -1
                )

                not_rows[addr][c] = pos_map[c]["notional"]

            else:

                dir_rows[addr][c] = 0
                not_rows[addr][c] = 0

    return (
        pd.DataFrame(dir_rows, index=coins).T,
        pd.DataFrame(not_rows, index=coins).T,
        raw,
    )


# ============================================================
# SIGNAL ENGINE
# ============================================================


def compute_signals(
    leaderboard,
    n_elite,
    min_wallets,
    div_threshold,
    funding_rates,
    prices,
):

    elite_addrs = leaderboard.head(n_elite)["address"].tolist()

    elite_dir, elite_not, elite_raw = _build_position_matrix(elite_addrs)

    all_coins = elite_dir.columns.tolist()

    rows = []

    for coin in all_coins:

        e_col = elite_dir[coin]
        e_not_col = elite_not[coin]

        e_pos = e_col[e_col != 0]

        if len(e_pos) < min_wallets:
            continue

        long_wallets = (e_pos > 0).sum()
        short_wallets = (e_pos < 0).sum()

        elite_long_pct = long_wallets / len(e_pos) * 100

        long_notional = e_not_col[e_col > 0].sum()
        short_notional = e_not_col[e_col < 0].sum()

        total_notional = long_notional + short_notional

        if total_notional == 0:
            continue

        net_exposure_pct = (
            (long_notional - short_notional)
            / total_notional
        ) * 100

        binary_signal = (
            "LONG"
            if elite_long_pct >= (50 + div_threshold)
            else "SHORT"
            if elite_long_pct <= (50 - div_threshold)
            else "NEUTRAL"
        )

        notional_signal = (
            "LONG"
            if net_exposure_pct >= div_threshold
            else "SHORT"
            if net_exposure_pct <= -div_threshold
            else "NEUTRAL"
        )

        rows.append(
            {
                "coin": coin,
                "binary_signal": binary_signal,
                "notional_signal": notional_signal,
                "elite_wallets": len(e_pos),
                "elite_long_pct": round(elite_long_pct, 1),
                "net_exposure_pct": round(net_exposure_pct, 1),
                "long_notional_usd": round(float(long_notional)),
                "short_notional_usd": round(float(short_notional)),
                "total_notional_usd": round(float(total_notional)),
                "funding_rate_8h_pct": round(funding_rates.get(coin, 0), 4),
                "price_usd": prices.get(coin),
            }
        )

    signals = pd.DataFrame(rows)

    if not signals.empty:
        signals = signals.sort_values(
            "net_exposure_pct",
            ascending=False,
        ).reset_index(drop=True)

    return signals, elite_raw


# ============================================================
# TOP/BOTTOM WALLET TABLE
# ============================================================


def build_top_bottom_wallet_table(leaderboard):

    top_wallets = leaderboard.head(5)["address"].tolist()
    bottom_wallets = leaderboard.tail(5)["address"].tolist()

    rows = []

    for addr in top_wallets:

        positions = fetch_positions(addr, for_signal=False)

        for p in positions:

            rows.append(
                {
                    "group": "TOP 5",
                    "wallet": addr[:8] + "…",
                    "coin": p["coin"],
                    "direction": p["direction"].upper(),
                    "notional": round(p["notional"]),
                    "leverage": p["leverage"],
                    "upnl": round(p["unrealized_pnl"]),
                }
            )

    for addr in bottom_wallets:

        positions = fetch_positions(addr, for_signal=False)

        for p in positions:

            rows.append(
                {
                    "group": "BOTTOM 5",
                    "wallet": addr[:8] + "…",
                    "coin": p["coin"],
                    "direction": p["direction"].upper(),
                    "notional": round(p["notional"]),
                    "leverage": p["leverage"],
                    "upnl": round(p["unrealized_pnl"]),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# UI
# ============================================================

st.title("🐋 Crypto Whale Tracker")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Parameters")

    n_elite = st.slider(
        "Elite Wallets",
        5,
        50,
        _DEFAULT_N_ELITE,
        5,
    )

    min_wallets = st.slider(
        "Minimum Wallets",
        1,
        10,
        _DEFAULT_MIN_WALLETS,
    )

    div_threshold = st.slider(
        "Threshold",
        5,
        50,
        _DEFAULT_DIV_THRESH,
        5,
    )

    refresh = st.button("Refresh")


# ============================================================
# LOAD DATA
# ============================================================

with st.spinner("Loading Hyperliquid data..."):

    leaderboard = fetch_leaderboard()

    funding_rates = fetch_funding_rates()
    prices = fetch_prices()

    signals, elite_raw = compute_signals(
        leaderboard,
        n_elite,
        min_wallets,
        div_threshold,
        funding_rates,
        prices,
    )


# ============================================================
# KPI
# ============================================================

long_binary = len(signals[signals["binary_signal"] == "LONG"])
short_binary = len(signals[signals["binary_signal"] == "SHORT"])

long_notional = len(signals[signals["notional_signal"] == "LONG"])
short_notional = len(signals[signals["notional_signal"] == "SHORT"])

c1, c2, c3, c4 = st.columns(4)

c1.metric("Binary Long", long_binary)
c2.metric("Binary Short", short_binary)
c3.metric("Notional Long", long_notional)
c4.metric("Notional Short", short_notional)


# ============================================================
# TABS
# ============================================================

(
    tab_overview,
    tab_binary,
    tab_notional,
    tab_whales,
) = st.tabs(
    [
        "Overview",
        "Binary Signals",
        "Notional Signals",
        "Whale Holdings",
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

with tab_overview:

    st.subheader("All Signals")

    st.dataframe(
        signals,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# BINARY SIGNALS
# ============================================================

with tab_binary:

    st.subheader("Binary Wallet Vote Signals")

    binary_df = signals[
        signals["binary_signal"] != "NEUTRAL"
    ]

    st.dataframe(
        binary_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# NOTIONAL SIGNALS
# ============================================================

with tab_notional:

    st.subheader("Capital Weighted Signals")

    notional_df = signals[
        signals["notional_signal"] != "NEUTRAL"
    ]

    st.dataframe(
        notional_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# WHALE HOLDINGS
# ============================================================

with tab_whales:

    st.subheader("Top 5 vs Bottom 5 Wallet Holdings")

    whale_df = build_top_bottom_wallet_table(leaderboard)

    st.dataframe(
        whale_df,
        use_container_width=True,
        hide_index=True,
        height=700,
    )


# ============================================================
# CHART
# ============================================================

st.subheader("Net Exposure")

if not signals.empty:

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=signals["coin"],
            y=signals["net_exposure_pct"],
        )
    )

    fig.update_layout(height=500)

    st.plotly_chart(fig, use_container_width=True)

```
