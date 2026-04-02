"""
BTC Gamma Exposure (GEX) Dashboard — Deribit
=============================================
Computes Total GEX and per-Strike GEX profile from live Deribit options data.

GEX formula (per contract):
  GEX_i = Gamma_i × OI_i × Spot² × ContractSize × sign_i

  sign convention (standard SpotGamma / industry):
    +1 for calls  →  call OI dominates  →  positive / stabilising net GEX
    -1 for puts   →  put OI dominates   →  negative / destabilising net GEX

  Interpretation: net GEX = call_gamma_exposure − put_gamma_exposure aggregated
  by strike.  A positive total means call-side gamma dominates, so dealers
  collectively act as if net long gamma (sell rallies, buy dips).  A negative
  total means put-side gamma dominates and dealers chase moves.

  This is a *modelling convention*, not a statement that any single dealer is
  literally long gamma on calls; actual desk positioning varies.

Total GEX = Σ GEX_i  (USD, reported in billions for readability)

Positive GEX → Market-stabilising (mean-reverting, low vol)
Negative GEX → Market-destabilising (trending, high vol)

Fixes vs. v1
────────────
1. OI units   – Deribit `get_order_book` returns OI in USD notional; dividing
                by spot converts to contract count before applying the formula.
2. Sign conv  – code unchanged (+1 calls / −1 puts); misleading docstring fixed.
3. OTM filter – instruments with |delta| < 0.05 or > 0.95 are excluded (deep
                OTM/ITM: near-zero gamma, unreliable IV, stale OI artefacts).
4. DTE filter – instruments with tte_days < 2 are excluded (expiry-day pinning
                is a distinct phenomenon; gamma explodes to noise levels).
5. Stale UI   – sidebar `max_expiry_days` change now triggers an automatic
                re-fetch without requiring the user to click Refresh.
6. Flip level – zero-crossing nearest to spot is used instead of first crossing.
7. Deprecation– `datetime.utcfromtimestamp` replaced with timezone-aware call.
8. Rate limit – 50 ms sleep between order-book requests to respect Deribit limits.
"""

import math
import time
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
    "Positive = stabilising / Negative = destabilising. "
    "Data: Deribit live options chain."
)

BASE_URL      = "https://www.deribit.com/api/v2"
CONTRACT_SIZE = 1.0   # 1 BTC per Deribit option contract
SLEEP_S       = 0.05  # inter-request delay to respect Deribit rate limits


# ─────────────────────────────────────────────
# Deribit helpers
# ─────────────────────────────────────────────
def safe_get(url: str, params: dict | None = None) -> dict:
    r = requests.get(url, params=params, timeout=15)
    data = r.json()
    if "error" in data:
        raise ValueError(f"Deribit API error: {data['error']}")
    if "result" not in data:
        raise ValueError(f"Unexpected response: {data}")
    return data["result"]


def get_instruments(currency: str = "BTC") -> list[dict]:
    return safe_get(
        f"{BASE_URL}/public/get_instruments",
        {"currency": currency, "kind": "option", "expired": "false"},
    )


def get_index_price(currency: str = "BTC") -> float:
    res = safe_get(
        f"{BASE_URL}/public/get_index_price",
        {"index_name": f"{currency.lower()}_usd"},
    )
    return res["index_price"]


def get_order_book(instrument_name: str) -> dict:
    time.sleep(SLEEP_S)
    return safe_get(
        f"{BASE_URL}/public/get_order_book",
        {"instrument_name": instrument_name},
    )


def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes gamma — fallback if greeks missing from API."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        return math.exp(-0.5 * d1 ** 2) / (S * sigma * math.sqrt(T) * math.sqrt(2 * math.pi))
    except (ValueError, ZeroDivisionError):
        return 0.0


def fmt_expiry(ts_ms: float) -> str:
    """Convert millisecond timestamp to 'DD Mon YY' label (timezone-aware)."""
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%d %b %y")


# ─────────────────────────────────────────────
# Core GEX computation
# ─────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def compute_gex(currency: str = "BTC", max_expiry_days: int = 60) -> tuple:
    """
    Fetch all options within max_expiry_days, apply quality filters, compute
    GEX per contract and return aggregated results.

    Returns
    -------
    total_gex    : float   – net GEX in USD
    df_by_strike : DataFrame
    df_by_expiry : DataFrame
    df_raw       : DataFrame – full per-contract detail
    spot         : float
    errors       : list[str]
    skipped      : dict[str, int] – filter counters
    """
    instruments = get_instruments(currency)
    spot        = get_index_price(currency)
    now_ms      = datetime.now(timezone.utc).timestamp() * 1000

    # ── Filter to expiries within the window ─────────────────────────────────
    expiry_set = {
        inst["expiration_timestamp"]
        for inst in instruments
        if 0 < (inst["expiration_timestamp"] - now_ms) / 86_400_000 <= max_expiry_days
    }

    if not expiry_set:
        raise ValueError("No expiries found within the selected window.")

    filtered = [
        inst for inst in instruments
        if inst["expiration_timestamp"] in expiry_set
    ]

    rows    = []
    errors  = []
    skipped = {"dte": 0, "delta": 0, "no_data": 0}
    progress = st.progress(0, text="Fetching options chain…")

    for idx, inst in enumerate(filtered):
        progress.progress(
            (idx + 1) / len(filtered),
            text=f"Fetching {idx + 1}/{len(filtered)}: {inst['instrument_name']}",
        )

        # ── Fix 4: DTE < 2 filter ─────────────────────────────────────────
        tte_days = (inst["expiration_timestamp"] - now_ms) / 86_400_000
        if tte_days < 2:
            skipped["dte"] += 1
            continue

        try:
            ob = get_order_book(inst["instrument_name"])
        except Exception as e:
            errors.append(f"{inst['instrument_name']}: {e}")
            skipped["no_data"] += 1
            continue

        greeks  = ob.get("greeks") or {}
        gamma   = greeks.get("gamma")
        mark_iv = ob.get("mark_iv")
        T       = tte_days / 365.0

        # ── Fix 3: delta-based OTM filter ────────────────────────────────
        raw_delta = greeks.get("delta")
        if raw_delta is not None:
            abs_delta = abs(raw_delta)
            if abs_delta < 0.05 or abs_delta > 0.95:
                skipped["delta"] += 1
                continue
        # If delta unavailable, fall through and let gamma speak for itself.

        # ── Gamma: use API value or BS fallback ──────────────────────────
        if gamma is None:
            if mark_iv and T > 0:
                # r=0.05 proxy for BTC funding/carry; minor for short tenors
                gamma = bs_gamma(spot, inst["strike"], T, 0.05, mark_iv / 100.0)
            else:
                gamma = 0.0

        # ── Fix 1: OI units ──────────────────────────────────────────────
        # Deribit `get_order_book` → `open_interest` is in USD notional.
        # Divide by spot to convert to number of contracts before use.
        oi_usd       = ob.get("open_interest") or 0.0
        oi_contracts = oi_usd / spot if spot > 0 else 0.0

        # Skip instruments with zero OI after conversion
        if oi_contracts == 0.0:
            skipped["no_data"] += 1
            continue

        # ── Fix 2: sign convention ────────────────────────────────────────
        # Standard industry convention:
        #   Net GEX = call_gamma_exposure − put_gamma_exposure
        #   +1 calls, −1 puts
        # Positive net GEX → call-side dominates → stabilising market behaviour.
        # Negative net GEX → put-side dominates → destabilising market behaviour.
        sign = 1 if inst["option_type"] == "call" else -1
        gex  = gamma * oi_contracts * CONTRACT_SIZE * spot ** 2 * sign

        rows.append({
            "instrument":    inst["instrument_name"],
            "strike":        inst["strike"],
            "expiry_ts":     inst["expiration_timestamp"],
            "tte_days":      tte_days,
            "option_type":   inst["option_type"],
            "gamma":         gamma,
            "delta":         raw_delta,
            "open_interest": oi_contracts,   # in contracts
            "oi_usd":        oi_usd,         # raw USD notional for reference
            "mark_iv":       mark_iv,
            "gex":           gex,
        })

    progress.empty()

    if not rows:
        raise ValueError(
            "No option data passed filters. "
            f"Skipped — DTE<2: {skipped['dte']}, deep OTM/ITM: {skipped['delta']}, "
            f"no data / zero OI: {skipped['no_data']}."
        )

    df_raw = pd.DataFrame(rows)
    df_raw["expiry_label"] = df_raw["expiry_ts"].apply(fmt_expiry)

    # ── Aggregate by strike ───────────────────────────────────────────────────
    df_by_strike = (
        df_raw.groupby("strike")["gex"]
        .sum()
        .reset_index()
        .sort_values("strike")
    )
    df_by_strike["gex_bn"] = df_by_strike["gex"] / 1e9
    df_by_strike["color"]  = df_by_strike["gex_bn"].apply(
        lambda x: "#00d4a4" if x >= 0 else "#ff4b6e"
    )

    # ── Aggregate by expiry ───────────────────────────────────────────────────
    df_by_expiry = (
        df_raw.groupby(["expiry_label", "tte_days"])["gex"]
        .sum()
        .reset_index()
        .sort_values("tte_days")
    )
    df_by_expiry["gex_bn"] = df_by_expiry["gex"] / 1e9

    total_gex = df_raw["gex"].sum()

    return total_gex, df_by_strike, df_by_expiry, df_raw, spot, errors, skipped


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
# Data fetch — Fix 5: auto re-fetch on expiry slider change
# ─────────────────────────────────────────────
if "gex_data" not in st.session_state:
    st.session_state["gex_data"]         = None
    st.session_state["last_expiry_days"] = None

# Detect expiry window change even without clicking Refresh
expiry_changed = st.session_state["last_expiry_days"] != max_expiry_days

if refresh:
    st.cache_data.clear()

if refresh or expiry_changed or st.session_state["gex_data"] is None:
    st.session_state["last_expiry_days"] = max_expiry_days
    with st.spinner("Fetching live BTC options data from Deribit…"):
        try:
            st.session_state["gex_data"] = compute_gex(
                max_expiry_days=max_expiry_days
            )
            if refresh:
                st.success("✅ Data refreshed successfully.")
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.stop()

if st.session_state["gex_data"] is None:
    st.stop()

total_gex, df_by_strike, df_by_expiry, df_raw, spot, errors, skipped = (
    st.session_state["gex_data"]
)

# ── Fetch errors ─────────────────────────────────────────────────────────────
if errors:
    with st.expander(f"⚠️ {len(errors)} instruments had fetch errors (click to expand)"):
        st.write(errors[:30])

# ── Filter summary ───────────────────────────────────────────────────────────
with st.expander("🔍 Filter summary", expanded=False):
    st.markdown(
        f"- **DTE < 2 excluded:** {skipped['dte']} instruments  \n"
        f"- **Deep OTM/ITM excluded (|Δ| < 0.05 or > 0.95):** {skipped['delta']} instruments  \n"
        f"- **No data / zero OI excluded:** {skipped['no_data']} instruments  \n"
        f"- **Included in GEX:** {len(df_raw)} instruments"
    )

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
total_oi     = df_raw["open_interest"].sum()   # in contracts
num_strikes  = df_raw["strike"].nunique()
gex_regime   = "🟢 Positive (Stabilising)" if total_gex_bn >= 0 else "🔴 Negative (Destabilising)"

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total GEX",           f"${total_gex_bn:.2f}B",     gex_regime)
col2.metric("Call GEX",            f"${call_gex_bn:.2f}B")
col3.metric("Put GEX",             f"${put_gex_bn:.2f}B")
col4.metric("Total Open Interest", f"{total_oi:,.0f} contracts")
col5.metric("Strikes Covered",     str(num_strikes))

st.markdown("---")

# Strike window used by per-strike charts (strike_range_pct is display-only,
# no re-fetch needed)
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
pivot = pivot.sort_index()

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

# ── ATM line on categorical y-axis ───────────────────────────────────────────
# add_hline() fails on string (categorical) y-axes; use add_shape() in paper
# coordinates instead.  Category index 0 = bottom (y_paper ≈ 0), n-1 = top.
# Centre of category i → y_frac = (i + 0.5) / n
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
    margin=dict(l=80, r=60, t=20, b=60),
)
st.plotly_chart(fig_heat, use_container_width=True)


# ─────────────────────────────────────────────
# Chart 4: Cumulative GEX — Flip Level
# ─────────────────────────────────────────────
st.subheader("🔀 GEX Flip Level")
st.caption(
    "Cumulative GEX scanned from lowest → highest strike. "
    "The zero-crossing nearest to spot = GEX flip level. "
    "Spot above flip → positive gamma regime. Spot below flip → negative gamma regime."
)

df_flip = df_by_strike.sort_values("strike").copy()
df_flip["cumulative_gex_bn"] = df_flip["gex_bn"].cumsum()

# ── Fix 6: nearest-to-spot zero crossing ─────────────────────────────────────
prev           = df_flip["cumulative_gex_bn"].shift(1, fill_value=0)
flip_crossings = df_flip[(prev * df_flip["cumulative_gex_bn"]) < 0].copy()

if len(flip_crossings):
    flip_crossings["dist_to_spot"] = (flip_crossings["strike"] - spot).abs()
    flip_strike = float(
        flip_crossings.loc[flip_crossings["dist_to_spot"].idxmin(), "strike"]
    )
else:
    flip_strike = None

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
        "option_type", "delta", "gamma", "open_interest", "mark_iv", "gex",
    ]
    st.dataframe(
        df_raw[display_cols]
        .sort_values(["tte_days", "strike"])
        .assign(gex=lambda d: d["gex"] / 1e6)
        .rename(columns={"gex": "gex_$M", "open_interest": "oi_contracts"})
        .style.format({
            "strike":       "${:,.0f}",
            "tte_days":     "{:.1f}d",
            "delta":        "{:.3f}",
            "gamma":        "{:.6f}",
            "oi_contracts": "{:,.2f}",
            "mark_iv":      "{:.1f}%",
            "gex_$M":       "${:,.2f}M",
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
GEX_i = Gamma_i × OI_i(contracts) × ContractSize × Spot² × sign_i
```

| Term | Source | Notes |
|---|---|---|
| **Gamma** | Deribit `greeks.gamma` | Falls back to Black-Scholes (r=5%) if API returns `null` |
| **OI (contracts)** | `open_interest` ÷ Spot | Deribit returns OI in USD notional; dividing by spot converts to contracts |
| **ContractSize** | 1 BTC | Deribit standard |
| **sign** | +1 calls / −1 puts | Industry convention: net GEX = call exposure − put exposure |

**Total GEX** = Σ GEX_i across all included strikes and expiries, reported in USD billions.

---

### Filters Applied

| Filter | Threshold | Rationale |
|---|---|---|
| **DTE** | ≥ 2 days | Sub-2-day gamma explodes to noise; expiry-day pinning is a distinct phenomenon |
| **Delta** | 0.05 ≤ \|Δ\| ≤ 0.95 | Excludes deep OTM/ITM: near-zero gamma, unreliable IV, stale OI artefacts |
| **Zero OI** | OI > 0 contracts | No meaningful exposure |

---

### Sign Convention

The +1/−1 sign is **not** a statement about actual dealer positioning.  It is the
standard industry convention for aggregating call and put gamma into a single net figure:

- **Positive GEX** → call-side gamma dominates → dealers collectively hedge as if
  net long gamma → sell rallies, buy dips → **mean-reverting, low vol**
- **Negative GEX** → put-side gamma dominates → dealers hedge as if net short gamma
  → chase moves → **trending, high vol**

---

### Key Levels

| Level | Definition |
|---|---|
| **Gamma Wall** | Strike with the largest positive GEX bar; acts as price magnet / resistance |
| **Put Wall** | Large negative GEX cluster; dealers must buy aggressively if spot breaches it |
| **GEX Flip Level** | Cumulative GEX zero-crossing nearest to spot; regime-change boundary |

---

### Caveats
- Actual dealer positioning varies; this model assumes a uniform short-options stance.
- Deribit OI updates intraday but not tick-by-tick.
- For a GEX *time series*, poll and persist `total_gex` snapshots (e.g. hourly cron + DB).
        """
    )
