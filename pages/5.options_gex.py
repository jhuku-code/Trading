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
import streamlit as st
import plotly.graph_objects as go

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
# Deribit helpers
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
    """Black-Scholes gamma — fallback if greeks missing from API."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return math.exp(-0.5 * d1**2) / (S * sigma * math.sqrt(T) * math.sqrt(2 * math.pi))


# ─────────────────────────────────────────────
# Core GEX computation
# ─────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def compute_gex(currency="BTC", max_expiry_days=60):
    """
    Fetch all options within max_expiry_days, compute GEX per contract, return:
      - total_gex      (float, USD)
      - df_by_strike   (DataFrame): GEX summed by strike
      - df_by_expiry   (DataFrame): GEX summed by expiry
      - df_raw         (DataFrame): full per-contract detail
      - spot           (float)
      - errors         (list[str])
    """
    instruments = get_instruments(currency)
    spot = get_index_price(currency)
    now_ms = datetime.now(timezone.utc).timestamp() * 1000

    # Filter to expiries within the window
    expiry_set = {
        inst["expiration_timestamp"]
        for inst in instruments
        if 0 < (inst["expiration_timestamp"] - now_ms) / 86_400_000 <= max_expiry_days
    }

    if not expiry_set:
        raise ValueError("No expiries found within the selected window.")

    filtered = [inst for inst in instruments if inst["expiration_timestamp"] in expiry_set]

    rows = []
    errors = []
    progress = st.progress(0, text="Fetching options chain…")

    for idx, inst in enumerate(filtered):
        progress.progress(
            (idx + 1) / len(filtered),
            text=f"Fetching {idx + 1}/{len(filtered)}: {inst['instrument_name']}",
        )
        try:
            ob = get_order_book(inst["instrument_name"])
        except Exception as e:
            errors.append(str(e))
            continue

        greeks = ob.get("greeks") or {}
        gamma = greeks.get("gamma")
        open_interest = ob.get("open_interest") or 0
        mark_iv = ob.get("mark_iv")

        tte_days = (inst["expiration_timestamp"] - now_ms) / 86_400_000
        T = tte_days / 365.0

        # Fallback: compute gamma via BS if not returned by API
        if gamma is None and mark_iv and T > 0:
            gamma = bs_gamma(spot, inst["strike"], T, 0.0, mark_iv / 100.0)
        if gamma is None:
            gamma = 0.0

        # Standard SpotGamma / InsiderFinance dealer convention:
        #   Calls → dealer net short call → long gamma  → +1
        #   Puts  → dealer net short put  → short gamma → -1
        sign = 1 if inst["option_type"] == "call" else -1
        gex = gamma * open_interest * CONTRACT_SIZE * spot**2 * sign

        rows.append({
            "instrument":    inst["instrument_name"],
            "strike":        inst["strike"],
            "expiry_ts":     inst["expiration_timestamp"],
            "tte_days":      tte_days,
            "option_type":   inst["option_type"],
            "gamma":         gamma,
            "open_interest": open_interest,
            "mark_iv":       mark_iv,
            "gex":           gex,
        })

    progress.empty()

    if not rows:
        raise ValueError("No option data could be retrieved.")

    df_raw = pd.DataFrame(rows)

    df_raw["expiry_label"] = df_raw["expiry_ts"].apply(
        lambda ts: datetime.utcfromtimestamp(ts / 1000).strftime("%d %b %y")
    )

    # Aggregate by strike
    df_by_strike = (
        df_raw.groupby("strike")["gex"]
        .sum()
        .reset_index()
        .sort_values("strike")
    )
    df_by_strike["gex_bn"] = df_by_strike["gex"] / 1e9
    df_by_strike["color"] = df_by_strike["gex_bn"].apply(
        lambda x: "#00d4a4" if x >= 0 else "#ff4b6e"
    )

    # Aggregate by expiry
    df_by_expiry = (
        df_raw.groupby(["expiry_label", "tte_days"])["gex"]
        .sum()
        .reset_index()
        .sort_values("tte_days")
    )
    df_by_expiry["gex_bn"] = df_by_expiry["gex"] / 1e9

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
        help="Clip the GEX profile / heatmap charts to this % band around current spot.",
    )
    show_raw = st.checkbox("Show raw options table", value=False)
    st.markdown("---")
    refresh = st.button("🔄 Fetch / Refresh data", use_container_width=True)
    st.caption("Data is cached for 60 s. Click to force a refresh.")


# ─────────────────────────────────────────────
# Data fetch
# ─────────────────────────────────────────────
if "gex_data" not in st.session_state:
    st.session_state["gex_data"] = None

if refresh:
    st.cache_data.clear()

if refresh or st.session_state["gex_data"] is None:
    with st.spinner("Fetching live BTC options data from Deribit…"):
        try:
            st.session_state["gex_data"] = compute_gex(max_expiry_days=max_expiry_days)
            if refresh:
                st.success("✅ Data refreshed successfully.")
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.stop()

if st.session_state["gex_data"] is None:
    st.stop()

total_gex, df_by_strike, df_by_expiry, df_raw, spot, errors = st.session_state["gex_data"]

if errors:
    with st.expander(f"⚠️ {len(errors)} instruments had fetch errors (click to expand)"):
        st.write(errors[:30])

# Timestamp + spot
st.caption(
    f"🕐 Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  "
    f"|  BTC Spot: **${spot:,.0f}**"
)

# ─────────────────────────────────────────────
# Top-level metric cards
# ─────────────────────────────────────────────
total_gex_bn = total_gex / 1e9
call_gex_bn  = df_raw[df_raw["option_type"] == "call"]["gex"].sum() / 1e9
put_gex_bn   = df_raw[df_raw["option_type"] == "put"]["gex"].sum() / 1e9
total_oi     = df_raw["open_interest"].sum()
num_strikes  = df_raw["strike"].nunique()
gex_regime   = "🟢 Positive (Stabilising)" if total_gex_bn >= 0 else "🔴 Negative (Destabilising)"

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total GEX",           f"${total_gex_bn:.2f}B", gex_regime)
col2.metric("Call GEX",            f"${call_gex_bn:.2f}B")
col3.metric("Put GEX",             f"${put_gex_bn:.2f}B")
col4.metric("Total Open Interest", f"{total_oi:,.0f} BTC")
col5.metric("Strikes Covered",     str(num_strikes))

st.markdown("---")

# Strike window used by per-strike charts
lo = spot * (1 - strike_range_pct / 100)
hi = spot * (1 + strike_range_pct / 100)


# ─────────────────────────────────────────────
# Chart 1: GEX Profile by Strike
# ─────────────────────────────────────────────
st.subheader("📊 GEX Profile by Strike")
st.caption(
    "Each bar = net dealer gamma exposure at that strike. "
    "Large green bars = gamma wall (support / resistance). "
    "Large red bars = potential volatility accelerant."
)

df_plot = df_by_strike[
    (df_by_strike["strike"] >= lo) & (df_by_strike["strike"] <= hi)
].copy()

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
)
fig_profile.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
fig_profile.update_yaxes(
    gridcolor="rgba(255,255,255,0.05)",
    zeroline=True,
    zerolinecolor="rgba(255,255,255,0.3)",
)
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
fig_expiry.update_yaxes(
    gridcolor="rgba(255,255,255,0.05)",
    zeroline=True,
    zerolinecolor="rgba(255,255,255,0.3)",
)
st.plotly_chart(fig_expiry, use_container_width=True)


# ─────────────────────────────────────────────
# Chart 3: GEX Heatmap — Strike × Expiry
# ─────────────────────────────────────────────
st.subheader("🌡️ GEX Heatmap: Strike × Expiry")
st.caption(
    "Net GEX at each strike / expiry node. "
    "Useful for spotting gamma walls concentrated at specific dates."
)

df_heat = (
    df_raw[
        (df_raw["strike"] >= lo) & (df_raw["strike"] <= hi)
    ]
    .groupby(["strike", "expiry_label", "tte_days"])["gex"]
    .sum()
    .reset_index()
)

expiry_order = df_heat.sort_values("tte_days")["expiry_label"].unique().tolist()
pivot = df_heat.pivot_table(
    index="strike", columns="expiry_label", values="gex", fill_value=0
)
pivot = pivot.reindex(columns=[c for c in expiry_order if c in pivot.columns])
pivot = pivot.sort_index()  # ascending strikes → category index 0 = lowest strike

y_labels = [f"${k:,.0f}" for k in pivot.index.tolist()]
z_data   = (pivot.values / 1e9).tolist()

fig_heat = go.Figure(
    go.Heatmap(
        z=z_data,
        x=pivot.columns.tolist(),
        y=y_labels,
        colorscale=[
            [0.0, "#ff4b6e"],
            [0.5, "#1a1a2e"],
            [1.0, "#00d4a4"],
        ],
        zmid=0,
        hovertemplate="Strike: %{y}<br>Expiry: %{x}<br>GEX: $%{z:.3f}B<extra></extra>",
        colorbar=dict(title="GEX ($B)", tickfont=dict(color="white")),
    )
)

# ── ATM line ─────────────────────────────────────────────────────────────────
# NOTE: add_hline() raises a TypeError on categorical (string) y-axes because
# Plotly internally tries to compute float(sum(y_values)) / len(y_values) to
# position the annotation, which fails for strings like "$85,000".
#
# Fix: use add_shape() + add_annotation() with xref/yref="paper" so we work
# entirely in normalised [0, 1] coordinates, computing the ATM row's fraction
# manually from its index position in the sorted y_labels list.
#
# In Plotly heatmaps, category index 0 sits at the BOTTOM of the plot area
# (y_paper ≈ 0) and index n-1 sits at the TOP (y_paper ≈ 1). The centre of
# category i is therefore at y_frac = (i + 0.5) / n.
# ─────────────────────────────────────────────────────────────────────────────
atm_strike = min(pivot.index, key=lambda k: abs(k - spot))
atm_label  = f"${atm_strike:,.0f}"

if atm_label in y_labels:
    n       = len(y_labels)
    atm_idx = y_labels.index(atm_label)
    y_frac  = (atm_idx + 0.5) / n

    fig_heat.add_shape(
        type="line",
        xref="paper", yref="paper",
        x0=0, x1=1,
        y0=y_frac, y1=y_frac,
        line=dict(color="white", width=1.5, dash="dash"),
    )
    fig_heat.add_annotation(
        xref="paper", yref="paper",
        x=1.01, y=y_frac,
        text="ATM",
        showarrow=False,
        font=dict(color="white", size=11),
        xanchor="left",
    )

fig_heat.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis_title="Expiry",
    yaxis_title="Strike",
    height=500,
    margin=dict(l=80, r=60, t=20, b=60),  # extra right margin for ATM label
)
st.plotly_chart(fig_heat, use_container_width=True)


# ─────────────────────────────────────────────
# Chart 4: Cumulative GEX — Flip Level
# ─────────────────────────────────────────────
st.subheader("🔀 GEX Flip Level")
st.caption(
    "Cumulative GEX scanned from lowest → highest strike. "
    "The zero-crossing = GEX flip level. "
    "Spot above flip → positive gamma regime. Spot below flip → negative gamma regime."
)

df_flip = df_by_strike.sort_values("strike").copy()
df_flip["cumulative_gex_bn"] = df_flip["gex_bn"].cumsum()

# Detect first zero-crossing
prev = df_flip["cumulative_gex_bn"].shift(1, fill_value=0)
flip_crossings = df_flip[(prev * df_flip["cumulative_gex_bn"]) < 0]
flip_strike = float(flip_crossings["strike"].values[0]) if len(flip_crossings) else None

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
    annotation_position="top right",
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
    regime = (
        "Spot is ABOVE flip → positive gamma regime 🟢"
        if spot > flip_strike
        else "Spot is BELOW flip → negative gamma regime 🔴"
    )
    st.info(f"⚡ **GEX Flip Level: ${flip_strike:,.0f}** — {regime}")
else:
    st.warning(
        "GEX flip level not detected in current strike range "
        "(uniform sign across all strikes)."
    )

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
fig_flip.update_yaxes(
    gridcolor="rgba(255,255,255,0.05)",
    zeroline=True,
    zerolinecolor="rgba(255,255,255,0.3)",
)
st.plotly_chart(fig_flip, use_container_width=True)


# ─────────────────────────────────────────────
# Raw options table (optional)
# ─────────────────────────────────────────────
if show_raw:
    st.subheader("🗃️ Raw Options Data")
    display_cols = [
        "instrument", "strike", "expiry_label", "tte_days",
        "option_type", "gamma", "open_interest", "mark_iv", "gex",
    ]
    st.dataframe(
        df_raw[display_cols]
        .sort_values(["tte_days", "strike"])
        .assign(gex=lambda d: d["gex"] / 1e6)
        .rename(columns={"gex": "gex_$M"})
        .style.format({
            "strike":        "${:,.0f}",
            "tte_days":      "{:.1f}d",
            "gamma":         "{:.6f}",
            "open_interest": "{:,.1f}",
            "mark_iv":       "{:.1f}%",
            "gex_$M":        "${:,.2f}M",
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
GEX_i = Gamma_i × OI_i × ContractSize × Spot² × sign_i
```

| Term | Source | Notes |
|---|---|---|
| **Gamma** | Deribit `greeks.gamma` | Falls back to Black-Scholes if API returns `null` |
| **OI** | Deribit `open_interest` | In BTC contracts |
| **ContractSize** | 1 BTC | Deribit standard |
| **sign** | +1 calls / −1 puts | Dealers assumed net short options to the market |

**Total GEX** = Σ GEX_i across all included strikes and expiries, reported in USD billions.

---

### Interpretation

| GEX | Dealer position | Expected market behaviour |
|---|---|---|
| **Positive** | Net long gamma | Sell rallies, buy dips → **mean-reverting, low vol** |
| **Negative** | Net short gamma | Chase moves → **trending, high vol** |

**Key levels to watch:**
- **Gamma Wall** — strike with the largest positive GEX bar; acts as a price magnet / resistance.
- **GEX Flip Level** — cumulative GEX crosses zero; regime-change boundary.
- **Put Wall** — large negative GEX cluster; dealers must buy aggressively if spot breaches it.

---

### Caveats
- Assumes dealers are uniformly net short — actual desk-level positioning varies.
- Deribit OI data updates intraday but not tick-by-tick.
- Gamma contribution is highest near expiry; far-dated options matter less.
- For a GEX *time series*, poll and persist `total_gex` snapshots (e.g. hourly cron + DB).
        """
    )
