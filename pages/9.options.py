# app.py

import math
from datetime import datetime, timezone

import requests
import pandas as pd
import streamlit as st

BASE_URL = "https://www.deribit.com/api/v2"

st.set_page_config(page_title="BTC Options Vol Dashboard", layout="wide")

st.title("BTC Options Volatility & 25Δ Skew (Deribit)")
st.caption("Constant-maturity 30D ATM IV (VX30) and 25Δ skew, built from live Deribit options data.")


# --------------------
# Deribit Helpers
# --------------------
def safe_get(url, params=None):
    r = requests.get(url, params=params)
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
    res = safe_get(
        f"{BASE_URL}/public/get_order_book",
        {"instrument_name": instrument_name},
    )
    return res


def find_atm_iv(instruments, expiry_ts, spot):
    """Find ATM option (closest strike) and get its IV (decimal)."""
    candidates = [
        inst for inst in instruments if inst["expiration_timestamp"] == expiry_ts
    ]
    if not candidates:
        raise ValueError(f"No options found for expiry {expiry_ts}")

    closest = min(candidates, key=lambda x: abs(x["strike"] - spot))
    ob = get_order_book(closest["instrument_name"])
    iv = ob.get("mark_iv")

    if iv is None:
        raise ValueError(f"No mark_iv for {closest['instrument_name']}")

    return iv / 100.0  # convert % → decimal


def find_25delta_options(instruments, expiry_ts):
    """Find 25-delta put and call options and get their IVs (decimal)."""
    candidates = [
        inst for inst in instruments if inst["expiration_timestamp"] == expiry_ts
    ]
    if not candidates:
        raise ValueError(f"No options found for expiry {expiry_ts}")

    puts = [inst for inst in candidates if inst["option_type"] == "put"]
    calls = [inst for inst in candidates if inst["option_type"] == "call"]

    put_25d = None
    call_25d = None
    min_put_diff = float("inf")
    min_call_diff = float("inf")

    # Find put closest to |delta| = 0.25
    for put in puts:
        ob = get_order_book(put["instrument_name"])
        greeks = ob.get("greeks")
        if greeks and greeks.get("delta") is not None:
            delta = abs(greeks["delta"])
            diff = abs(delta - 0.25)
            if diff < min_put_diff:
                min_put_diff = diff
                put_25d = (put["instrument_name"], ob.get("mark_iv"))

    # Find call closest to |delta| = 0.25
    for call in calls:
        ob = get_order_book(call["instrument_name"])
        greeks = ob.get("greeks")
        if greeks and greeks.get("delta") is not None:
            delta = abs(greeks["delta"])
            diff = abs(delta - 0.25)
            if diff < min_call_diff:
                min_call_diff = diff
                call_25d = (call["instrument_name"], ob.get("mark_iv"))

    if (
        put_25d is None
        or call_25d is None
        or put_25d[1] is None
        or call_25d[1] is None
    ):
        raise ValueError(
            f"Could not find 25-delta options with valid IVs for expiry {expiry_ts}"
        )

    return put_25d[1] / 100.0, call_25d[1] / 100.0  # convert % → decimal


def get_vx30(currency="BTC", target_days=30):
    """
    Calculate constant-maturity 30-day ATM IV (vx30) and 25Δ skew,
    plus nearby expiry metrics.
    """
    instruments = get_instruments(currency)
    spot = get_index_price(currency)

    now = datetime.now(timezone.utc).timestamp() * 1000
    expiries = sorted({inst["expiration_timestamp"] for inst in instruments})
    expiries_days = [(ts, (ts - now) / 1000 / 86400) for ts in expiries]

    before = [e for e in expiries_days if e[1] < target_days]
    after = [e for e in expiries_days if e[1] > target_days]

    if not before or not after:
        raise ValueError("No expiries available around target maturity")

    T1, d1 = before[-1]
    T2, d2 = after[0]

    # ATM IVs
    iv1 = find_atm_iv(instruments, T1, spot)
    iv2 = find_atm_iv(instruments, T2, spot)

    # 25Δ put/call IVs
    put_iv1, call_iv1 = find_25delta_options(instruments, T1)
    put_iv2, call_iv2 = find_25delta_options(instruments, T2)

    # 25Δ skew (Put IV - Call IV)
    skew1 = put_iv1 - call_iv1
    skew2 = put_iv2 - call_iv2

    # Linear interpolation weight for 30 days
    w = (target_days - d1) / (d2 - d1)

    # Interpolated 30D skew
    skew30 = skew1 * (1 - w) + skew2 * w

    # Variance interpolation for ATM IV
    var1 = (iv1**2) * (d1 / 365.0)
    var2 = (iv2**2) * (d2 / 365.0)
    var30 = var1 * (1 - w) + var2 * w
    iv30 = math.sqrt(var30 * (365.0 / target_days))

    return iv30, iv1, d1, iv2, d2, skew30, skew1, skew2


# --------------------
# Streamlit UI
# --------------------

# Use session state to persist the latest fetched values
if "vx_data" not in st.session_state:
    st.session_state["vx_data"] = None

st.markdown("### Controls")

col_btn, col_info = st.columns([1, 3])

with col_btn:
    refresh = st.button("🔄 Refresh data")

with col_info:
    st.write(
        "Click **Refresh data** to pull the latest BTC option vol and 25Δ skew from Deribit."
    )

if refresh:
    with st.spinner("Fetching latest BTC options vol data from Deribit..."):
        try:
            vx30, iv1, d1, iv2, d2, skew30, skew1, skew2 = get_vx30()
            st.session_state["vx_data"] = {
                "vx30": vx30,
                "iv1": iv1,
                "d1": d1,
                "iv2": iv2,
                "d2": d2,
                "skew30": skew30,
                "skew1": skew1,
                "skew2": skew2,
            }
            st.success("Data refreshed successfully.")
        except Exception as e:
            st.error(f"Error while fetching data: {e}")

data = st.session_state["vx_data"]

st.markdown("---")

if data is None:
    st.info("No data yet. Click **Refresh data** to calculate VX30 and skew.")
else:
    vx30 = data["vx30"]
    iv1 = data["iv1"]
    d1 = data["d1"]
    iv2 = data["iv2"]
    d2 = data["d2"]
    skew30 = data["skew30"]
    skew1 = data["skew1"]
    skew2 = data["skew2"]

    # --------------------
    # Top-level metrics
    # --------------------
    st.subheader("Constant Maturity Metrics (30 days)")

    c1, c2 = st.columns(2)
    c1.metric("VX30 (ATM IV)", f"{vx30 * 100:.2f} %")
    c2.metric("25Δ Skew (30D)", f"{skew30 * 100:.2f} %")

    st.caption("25Δ Skew is defined as 25Δ Put IV minus 25Δ Call IV (in implied vol points).")

    # --------------------
    # Nearby expiries used
    # --------------------
    st.subheader("Nearby Expiries Used for Interpolation")

    col_short, col_long = st.columns(2)

    with col_short:
        st.markdown("##### Shorter Expiry")
        st.write(f"**Days to expiry (d1):** {d1:.1f} days")
        st.write(f"**ATM IV (iv1):** {iv1 * 100:.2f} %")
        st.write(f"**25Δ Skew (skew1):** {skew1 * 100:.2f} %")

    with col_long:
        st.markdown("##### Longer Expiry")
        st.write(f"**Days to expiry (d2):** {d2:.1f} days")
        st.write(f"**ATM IV (iv2):** {iv2 * 100:.2f} %")
        st.write(f"**25Δ Skew (skew2):** {skew2 * 100:.2f} %")

    st.markdown("---")

    # --------------------
    # Tabular display of all requested variables
    # --------------------
    st.subheader("All Calculated Metrics")

    metrics_df = pd.DataFrame(
        {
            "Variable": [
                "vx30",
                "iv1",
                "d1",
                "iv2",
                "d2",
                "skew30",
                "skew1",
                "skew2",
            ],
            "Label / Description": [
                "30D constant-maturity ATM IV",
                "ATM IV of shorter expiry",
                "Days to shorter expiry",
                "ATM IV of longer expiry",
                "Days to longer expiry",
                "30D constant-maturity 25Δ skew",
                "25Δ skew of shorter expiry",
                "25Δ skew of longer expiry",
            ],
            "Value": [
                vx30 * 100,  # expressed in %
                iv1 * 100,   # expressed in %
                d1,
                iv2 * 100,   # expressed in %
                d2,
                skew30 * 100,  # expressed in %
                skew1 * 100,   # expressed in %
                skew2 * 100,   # expressed in %
            ],
            "Unit": [
                "%",
                "%",
                "days",
                "%",
                "days",
                "%",
                "%",
                "%",
            ],
        }
    )

    st.dataframe(metrics_df, use_container_width=True)

    st.markdown(
        """
**Notes**

- **vx30** is computed via variance interpolation between the two nearest expiries around 30 days.  
- **25Δ Skew** = 25Δ Put IV − 25Δ Call IV (positive values indicate downside protection premium).
"""
    )
