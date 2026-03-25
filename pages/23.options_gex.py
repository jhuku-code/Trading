"""
BTC Gamma Exposure (GEX) Dashboard — Deribit
=============================================
Computes Total GEX and per-Strike GEX profile from live Deribit options data.

GEX formula (per contract):
  GEX_i = Gamma_i × OI_i × Spot² × ContractSize × sign_i
  where sign_i = +1 for calls (dealer long gamma), -1 for puts (dealer short gamma)
  assuming dealers are net short to the market (standard assumption).

Total GEX = Σ GEX_i  (in USD terms, billions for readability)

Positive GEX → Dealers long gamma → Market-stabilising (sell rallies, buy dips)
Negative GEX → Dealers short gamma → Market-destabilising (chase moves)
"""

import math
from datetime import datetime, timezone

import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="BTC GEX Dashboard",
    layout="wide",
    page_icon="⚡",
)

st.title("⚡ BTC Gamma Exposure (GEX) — Deribit")
st.caption(
    "Dealer Gamma Exposure aggregated across all strikes and expiries. "
    "Positive = stabilising / Negative = destabilising. Data: Deribit live options chain."
)

BASE_URL = "https://www.deribit.com/api/v2"
CONTRACT_SIZE = 1.0  # 1 BTC per Deribit option contract


# ─────────────────────────────────────────────
# Deribit helpers (reused from your skew app)
# ─────────────────────────────────────────────
def safe_get(url, params=None):
    r = requests.get(url, params=params, timeout=15)
    data = r.json()
    if "error" in data:
        raise ValueError(f"Deribit API error: {data['error']}")
    if "result" not in data:
        raise ValueError(f"Unexpected response: {data}")
    return data["result"]


def get_instruments(currency="BTC"):
    return safe_get(
        f"{BASE_URL}/public/get_instruments",
        {"currency": currency, "kind": "option", "expired": "false"},
    )


def get_index_price(currency="BTC"):
    res = safe_get(
        f"{BASE_URL}/public/get_index_price",
        {"index_name": f"{currency.lower()}_usd"},
    )
    return res["index_price"]


def get_order_book(instrument_name):
    return safe_get(
        f"{BASE_URL}/public/get_order_book",
        {"instrument_name": instrument_name},
    )


def bs_gamma(S, K, T, r, sigma):
    """Black-Scholes gamma (fallback if greeks missing from API)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return math.exp(-0.5 * d1**2) / (S * sigma * math.sqrt(T) * math.sqrt(2 * math.pi))


# ─────────────────────────────────────────────
# Core GEX computation
# ─────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def compute_gex(currency="BTC", max_expiry_days=90):
    """
    Fetch all options, compute GEX per strike+expiry, return:
      - total_gex (float, USD)
      - df_by_strike (DataFrame): GEX aggregated by strike
      - df_by_expiry (DataFrame): GEX aggregated by expiry
      - df_raw (DataFrame): full per-contract detail
      - spot (float)
    """
    instruments = get_instruments(currency)
    spot = get_index_price(currency)
    now_ms = datetime.now(timezone.utc).timestamp() * 1000

    rows = []
    errors = []

    # Filter to expiries within max_expiry_days to keep API calls manageable
    expiry_set = sorted({
        inst["expiration_timestamp"] for inst in instruments
        if (inst["expiration_timestamp"] - now_ms) / 86_400_000 <= max_expiry_days
    })

    if not expiry_set:
        raise ValueError("No expiries found within the selected window.")

    progress = st.progress(0, text="Fetching options chain…")
    total_instruments = [
        inst for inst in instruments
        if inst["expiration_timestamp"] in set(expiry_set)
    ]

    for idx, inst in enumerate(total_instruments):
        progress.progress(
            (idx + 1) / len(total_instruments),
            text=f"Fetching {idx+1}/{len(total_instruments)}: {inst['instrument_name']}",
        )
        try:
            ob = get_order_book(inst["instrument_name"])
        except Exception as e:
            errors.append(str(e))
            continue

        greeks = ob.get("greeks") or {}
        gamma = greeks.get("gamma")
        open_interest = ob.get("open_interest", 0) or 0
        mark_iv = ob.get("mark_iv")

        tte_days = (inst["expiration_timestamp"] - now_ms) / 86_400_000
        T = tte_days / 365.0

        # Fallback: compute gamma via BS if not returned
        if gamma is None and mark_iv and T > 0:
            gamma = bs_gamma(spot, inst["strike"], T, 0.0, mark_iv / 100.0)
        if gamma is None:
            gamma = 0.0

        # GEX sign: dealers assumed net short options to market
        # Call → dealer short call → long gamma (+1)
        # Put  → dealer short put  → long gamma (+1) ← wait, this is debated
        # Standard convention (SpotGamma / InsiderFinance):
        #   Calls add positive GEX, puts add negative GEX
        sign = 1 if inst["option_type"] == "call" else -1

        gex = gamma * open_interest * CONTRACT_SIZE * spot**2 * sign

        rows.append(
            {
                "instrument": inst["instrument_name"],
                "strike": inst["strike"],
                "expiry_ts": inst["expiration_timestamp"],
                "tte_days": tte_days,
                "option_type": inst["option_type"],
                "gamma": gamma,
                "open_interest": open_interest,
                "mark_iv": mark_iv,
                "gex": gex,
            }
        )

    progress.empty()

    if not rows:
        raise ValueError("No option data could be retrieved.")

    df_raw = pd.DataFrame(rows)

    # Expiry label
    df_raw["expiry_label"] = df_raw["expiry_ts"].apply(
        lambda ts: datetime.utcfromtimestamp(ts / 1000).strftime("%d %b %y")
    )

    # ── Aggregate by strike ──────────────────────────
    df_by_strike = (
        df_raw.groupby("strike")["gex"]
        .sum()
        .reset_index()
        .sort_values("strike")
    )
    df_by_strike["gex_bn"] = df_by_strike["gex"] / 1e9  # convert to $bn
    df_by_strike["color"] = df_by_strike["gex_bn"].apply(
        lambda x: "#00d4a4" if x >= 0 else "#ff4b6e"
    )

    # ── Aggregate by expiry ──────────────────────────
    df_by_expiry = (
        df_raw.groupby(["expiry_label", "tte_days"])["gex"]
        .sum()
        .reset_index()
        .sort_values("tte_days")
    )
    df_by_expiry["gex_bn"] = df_by_expiry["gex"] / 1e9

    # ── Total GEX ────────────────────────────────────
    total_gex = df_raw["gex"].sum()

    return total_gex, df_by_strike, df_by_expiry, df_raw, spot, errors


# ─────────────────────────────────────────────
# Sidebar controls
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    max_expiry_days = st.slider(
        "Max expiry window (days)",
        min_value=7, max_value=180, value=60, step=7,
        help="Only include options expiring within this many days.",
    )
    strike_range_pct = st.slider(
        "Strike range around spot (%)",
        min_value=10, max_value=100, value=40, step=5,
        help="Clip the GEX profile chart to this % band around current spot.",
    )
    show_raw = st.checkbox("Show raw options table", value=False)

    st.markdown("---")
    refresh = st.button("🔄 Fetch / Refresh data", use_container_width=True)
    st.caption("Data cached for 60 s. Click to force refresh.")

# ─────────────────────────────────────────────
# Data fetch
# ─────────────────────────────────────────────
if "gex_data" not in st.session_state:
    st.session_state["gex_data"] = None

if refresh:
    st.cache_data.clear()
    with st.spinner("Fetching live BTC options data from Deribit…"):
        try:
            result = compute_gex(max_expiry_days=max_expiry_days)
            st.session_state["gex_data"] = result
            st.success("✅ Data refreshed successfully.")
        except Exception as e:
            st.error(f"❌ Error: {e}")

# Auto-load on first visit
if st.session_state["gex_data"] is None and not refresh:
    with st.spinner("Loading BTC options data from Deribit…"):
        try:
            result = compute_gex(max_expiry_days=max_expiry_days)
            st.session_state["gex_data"] = result
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.stop()

if st.session_state["gex_data"] is None:
    st.stop()

total_gex, df_by_strike, df_by_expiry, df_raw, spot, errors = st.session_state["gex_data"]

if errors:
    with st.expander(f"⚠️ {len(errors)} instruments had fetch errors"):
        st.write(errors[:20])

# ─────────────────────────────────────────────
# Timestamp
# ─────────────────────────────────────────────
st.caption(f"🕐 Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  |  BTC Spot: **${spot:,.0f}**")

# ─────────────────────────────────────────────
# Top metrics
# ─────────────────────────────────────────────
total_gex_bn = total_gex / 1e9
gex_sign = "🟢 Positive (Stabilising)" if total_gex_bn >= 0 else "🔴 Negative (Destabilising)"
call_gex = df_raw[df_raw["option_type"] == "call"]["gex"].sum() / 1e9
put_gex = df_raw[df_raw["option_type"] == "put"]["gex"].sum() / 1e9
total_oi = df_raw["open_interest"].sum()
num_strikes = df_raw["strike"].nunique()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total GEX", f"${total_gex_bn:.2f}B", gex_sign)
col2.metric("Call GEX", f"${call_gex:.2f}B")
col3.metric("Put GEX", f"${put_gex:.2f}B")
col4.metric("Total Open Interest", f"{total_oi:,.0f} BTC")
col5.metric("Strikes Covered", str(num_strikes))

st.markdown("---")

# ─────────────────────────────────────────────
# Chart 1: GEX Profile by Strike
# ─────────────────────────────────────────────
st.subheader("📊 GEX Profile by Strike")
st.caption(
    "Each bar shows the net dealer gamma exposure at that strike. "
    "Large positive bars = strong dealer support (gamma wall). "
    "Large negative bars = potential volatility accelerant."
)

# Clip to ± strike_range_pct% of spot
lo = spot * (1 - strike_range_pct / 100)
hi = spot * (1 + strike_range_pct / 100)
df_plot = df_by_strike[(df_by_strike["strike"] >= lo) & (df_by_strike["strike"] <= hi)].copy()

fig_profile = go.Figure()

fig_profile.add_trace(
    go.Bar(
        x=df_plot["strike"],
        y=df_plot["gex_bn"],
        marker_color=df_plot["color"].tolist(),
        name="GEX",
        hovertemplate="Strike: $%{x:,.0f}<br>GEX: $%{y:.3f}B<extra></extra>",
    )
)

# Spot line
fig_profile.add_vline(
    x=spot,
    line_dash="dash",
    line_color="white",
    annotation_text=f"  Spot ${spot:,.0f}",
    annotation_font_color="white",
    annotation_position="top right",
)

fig_profile.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0.2)",
    xaxis_title="Strike (USD)",
    yaxis_title="GEX (USD Billions)",
    height=420,
    margin=dict(l=60, r=20, t=20, b=60),
    bargap=0.1,
    legend=dict(orientation="h", y=1.02, x=0),
)
fig_profile.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
fig_profile.update_yaxes(gridcolor="rgba(255,255,255,0.05)", zeroline=True, zerolinecolor="rgba(255,255,255,0.3)")

st.plotly_chart(fig_profile, use_container_width=True)

# ─────────────────────────────────────────────
# Chart 2: GEX by Expiry
# ─────────────────────────────────────────────
st.subheader("📅 GEX by Expiry Date")
st.caption("How gamma exposure is distributed across upcoming expiries.")

fig_expiry = go.Figure()
fig_expiry.add_trace(
    go.Bar(
        x=df_by_expiry["expiry_label"],
        y=df_by_expiry["gex_bn"],
        marker_color=df_by_expiry["gex_bn"].apply(
            lambda x: "#00d4a4" if x >= 0 else "#ff4b6e"
        ).tolist(),
        hovertemplate="Expiry: %{x}<br>GEX: $%{y:.3f}B<extra></extra>",
    )
)

fig_expiry.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0.2)",
    xaxis_title="Expiry",
    yaxis_title="GEX (USD Billions)",
    height=380,
    margin=dict(l=60, r=20, t=20, b=60),
)
fig_expiry.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
fig_expiry.update_yaxes(gridcolor="rgba(255,255,255,0.05)", zeroline=True, zerolinecolor="rgba(255,255,255,0.3)")

st.plotly_chart(fig_expiry, use_container_width=True)

# ─────────────────────────────────────────────
# Chart 3: Call vs Put GEX heatmap by strike × expiry
# ─────────────────────────────────────────────
st.subheader("🌡️ GEX Heatmap: Strike × Expiry")
st.caption("Net GEX at each strike/expiry node. Useful for identifying gamma walls at specific dates.")

df_heat = df_raw[
    (df_raw["strike"] >= lo) & (df_raw["strike"] <= hi)
].groupby(["strike", "expiry_label", "tte_days"])["gex"].sum().reset_index()

# Pivot for heatmap
expiry_order = df_heat.sort_values("tte_days")["expiry_label"].unique().tolist()
pivot = df_heat.pivot_table(index="strike", columns="expiry_label", values="gex", fill_value=0)
pivot = pivot.reindex(columns=[c for c in expiry_order if c in pivot.columns])
pivot = pivot.sort_index()

# Normalise per column for readability (each expiry's max)
z_data = (pivot.values / 1e9).tolist()

fig_heat = go.Figure(
    go.Heatmap(
        z=z_data,
        x=pivot.columns.tolist(),
        y=[f"${k:,.0f}" for k in pivot.index.tolist()],
        colorscale=[
            [0, "#ff4b6e"],
            [0.5, "#1a1a2e"],
            [1, "#00d4a4"],
        ],
        zmid=0,
        hovertemplate="Strike: %{y}<br>Expiry: %{x}<br>GEX: $%{z:.3f}B<extra></extra>",
        colorbar=dict(title="GEX ($B)", tickfont=dict(color="white")),
    )
)

fig_heat.add_hline(
    y=f"${min(pivot.index, key=lambda k: abs(k - spot)):,.0f}",
    line_dash="dash",
    line_color="white",
    annotation_text="  ATM",
    annotation_font_color="white",
)

fig_heat.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis_title="Expiry",
    yaxis_title="Strike",
    height=500,
    margin=dict(l=80, r=20, t=20, b=60),
)

st.plotly_chart(fig_heat, use_container_width=True)

# ─────────────────────────────────────────────
# Chart 4: GEX Flip Level
# ─────────────────────────────────────────────
st.subheader("🔀 GEX Flip Level")
st.caption(
    "The strike where cumulative GEX (scanning from low → high strike) crosses zero. "
    "Price above flip = positive gamma regime. Price below flip = negative gamma regime."
)

df_flip = df_by_strike.sort_values("strike").copy()
df_flip["cumulative_gex_bn"] = df_flip["gex_bn"].cumsum()

# Find flip strike
flip_crossings = df_flip[
    (df_flip["cumulative_gex_bn"].shift(1, fill_value=0) * df_flip["cumulative_gex_bn"]) < 0
]
flip_strike = flip_crossings["strike"].values[0] if len(flip_crossings) else None

fig_flip = go.Figure()
fig_flip.add_trace(
    go.Scatter(
        x=df_flip["strike"],
        y=df_flip["cumulative_gex_bn"],
        mode="lines",
        line=dict(color="#a78bfa", width=2),
        fill="tozeroy",
        fillcolor="rgba(167,139,250,0.1)",
        name="Cumulative GEX",
        hovertemplate="Strike: $%{x:,.0f}<br>Cum. GEX: $%{y:.3f}B<extra></extra>",
    )
)

fig_flip.add_vline(
    x=spot,
    line_dash="dash",
    line_color="white",
    annotation_text=f"  Spot ${spot:,.0f}",
    annotation_font_color="white",
)

if flip_strike:
    fig_flip.add_vline(
        x=flip_strike,
        line_dash="dot",
        line_color="#facc15",
        annotation_text=f"  Flip ${flip_strike:,.0f}",
        annotation_font_color="#facc15",
        annotation_position="top left",
    )
    st.info(
        f"⚡ **GEX Flip Level: ${flip_strike:,.0f}** — "
        f"{'Spot is ABOVE flip (positive gamma regime 🟢)' if spot > flip_strike else 'Spot is BELOW flip (negative gamma regime 🔴)'}"
    )
else:
    st.warning("GEX flip level not detected in current strike range (all-positive or all-negative).")

fig_flip.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0.2)",
    xaxis_title="Strike (USD)",
    yaxis_title="Cumulative GEX (USD Billions)",
    height=380,
    margin=dict(l=60, r=20, t=20, b=60),
)
fig_flip.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
fig_flip.update_yaxes(gridcolor="rgba(255,255,255,0.05)", zeroline=True, zerolinecolor="rgba(255,255,255,0.3)")

st.plotly_chart(fig_flip, use_container_width=True)

# ─────────────────────────────────────────────
# Raw data table
# ─────────────────────────────────────────────
if show_raw:
    st.subheader("🗃️ Raw Options Data")
    display_cols = ["instrument", "strike", "expiry_label", "tte_days", "option_type",
                    "gamma", "open_interest", "mark_iv", "gex"]
    st.dataframe(
        df_raw[display_cols]
        .sort_values(["tte_days", "strike"])
        .assign(gex=lambda d: d["gex"] / 1e6)  # show in $M
        .rename(columns={"gex": "gex_$M"})
        .style.format({
            "strike": "${:,.0f}",
            "tte_days": "{:.1f}d",
            "gamma": "{:.6f}",
            "open_interest": "{:,.1f}",
            "mark_iv": "{:.1f}%",
            "gex_$M": "${:,.2f}M",
        }),
        use_container_width=True,
        height=400,
    )

# ─────────────────────────────────────────────
# Methodology notes
# ─────────────────────────────────────────────
with st.expander("📖 Methodology & Interpretation"):
    st.markdown(
        """
### GEX Formula
For each option contract *i*:

```
GEX_i = Gamma_i × OI_i × Spot² × ContractSize × sign_i
```

- **Gamma**: from Deribit's `greeks.gamma` (fallback: Black-Scholes if unavailable)
- **OI**: open interest in BTC contracts
- **ContractSize**: 1 BTC (Deribit standard)
- **sign**: **+1 for calls, −1 for puts** (standard SpotGamma/dealer convention — dealers assumed net short options to the market)

**Total GEX** = Σ GEX_i across all strikes and expiries (reported in USD billions)

---

### Interpretation

| GEX State | Dealer Position | Market Behaviour |
|---|---|---|
| **Positive** | Net long gamma | Sell rallies, buy dips → **mean-reverting / low vol** |
| **Negative** | Net short gamma | Chase moves → **trending / high vol** |

**Key levels:**
- **Gamma Wall**: strike with largest positive GEX bar → acts as magnet/resistance
- **GEX Flip Level**: strike where cumulative GEX crosses zero → regime change boundary
- **Put Wall**: large negative GEX → dealers buy aggressively if breached (can cause sharp moves)

---

### Caveats
- This assumes dealers are uniformly short options — actual positioning varies by desk.
- Deribit OI data is EOD-like; intraday changes not fully reflected.
- Gamma is most meaningful near expiry (high |Gamma|); far-dated options contribute less.
- For a full GEX *time series*, you would need to poll and store this snapshot periodically.
        """
    )
