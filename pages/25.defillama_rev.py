# defillama_rev.py

import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="DeFiLlama Revenue Dashboard", layout="wide")

# ==========================
#   DATA LOADER (CACHED)
# ==========================

@st.cache_data(ttl=60 * 30)  # cache for 30 minutes
def load_defillama_revenue():
    url = (
        "https://api.llama.fi/overview/fees"
        "?excludeTotalDataChart=true"
        "&excludeTotalDataChartBreakdown=true"
        "&dataType=dailyRevenue"
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    # Extract protocol data
    protocols = data.get("protocols", [])
    df_rev = pd.DataFrame(protocols)

    # === Step 1: Split into chain_rev and protocol_rev based on protocolType ===
    chain_rev = df_rev[df_rev["protocolType"] == "chain"].copy()
    protocol_rev = df_rev[df_rev["protocolType"] == "protocol"].copy()

    # === Step 2: Clean 'parentProtocol' column in protocol_rev ===
    protocol_rev["parentProtocol"] = protocol_rev["parentProtocol"].str.replace(
        "parent#", "", regex=False
    )

    # === Step 3: Replace blanks in parentProtocol with slug values ===
    protocol_rev["parentProtocol"] = protocol_rev["parentProtocol"].fillna(
        protocol_rev["slug"]
    )

    # === Step 4: Keep only required columns ===
    keep_cols = [
        "total24h",
        "total7d",
        "total30d",
        "name",
        "category",
        "slug",
        "parentProtocol",
    ]
    chain_rev = chain_rev[keep_cols].copy()
    protocol_rev = protocol_rev[keep_cols].copy()

    # Sort by parentProtocol and total30d descending (so highest total30d per group is first)
    protocol_rev_sorted = protocol_rev.sort_values(
        by=["parentProtocol", "total30d"], ascending=[True, False]
    )

    # Group by parentProtocol and aggregate
    protocol_rev_consolidated = (
        protocol_rev_sorted.groupby("parentProtocol", as_index=False).agg(
            {
                "total24h": "sum",
                "total7d": "sum",
                "total30d": "sum",
                "name": "first",  # from row with highest total30d
                "category": "first",
                "slug": "first",
            }
        )
    )
    protocol_rev_consolidated.reset_index(drop=True, inplace=True)

    # === Step 5: Chain rev market share calculations ===
    for col in ["total24h", "total7d", "total30d"]:
        total_sum = chain_rev[col].sum()
        chain_rev[f"{col}_share"] = chain_rev[col] / total_sum

    # Market share change: 7d - 30d
    chain_rev["share_change"] = (
        chain_rev["total7d_share"] - chain_rev["total30d_share"]
    )

    # Top 20 chains by share_change
    top20_chains = (
        chain_rev.sort_values("share_change", ascending=False)
        .head(20)["name"]
        .tolist()
    )

    # === Step 6: Protocol rev market share calculations ===
    for col in ["total24h", "total7d", "total30d"]:
        total_sum = protocol_rev_consolidated[col].sum()
        protocol_rev_consolidated[f"{col}_share"] = (
            protocol_rev_consolidated[col] / total_sum
        )

    # Market share change: 7d - 30d
    protocol_rev_consolidated["share_change"] = (
        protocol_rev_consolidated["total7d_share"]
        - protocol_rev_consolidated["total30d_share"]
    )

    # Top 20 protocols by share_change (using parentProtocol)
    top20_protocols = (
        protocol_rev_consolidated.sort_values("share_change", ascending=False)
        .head(20)["parentProtocol"]
        .tolist()
    )

    return chain_rev, protocol_rev_consolidated, top20_chains, top20_protocols


# ==========================
#   REFRESH BUTTON
# ==========================

st.sidebar.title("Controls")

if st.sidebar.button("🔄 Refresh data"):
    load_defillama_revenue.clear()  # clear cache
    st.experimental_rerun()

# ==========================
#   MAIN APP
# ==========================

st.title("DeFiLlama Revenue Dashboard")

with st.spinner("Loading DeFiLlama revenue data..."):
    (
        chain_rev,
        protocol_rev_consolidated,
        top20_chains,
        top20_protocols,
    ) = load_defillama_revenue()

# --------------------------
# Top 20 Chains & Protocols
# --------------------------
st.header("Top Movers by Revenue Market Share Change (7d vs 30d)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Top 20 Chains")
    top20_chain_df = (
        chain_rev[chain_rev["name"].isin(top20_chains)]
        .sort_values("share_change", ascending=False)
        .reset_index(drop=True)
    )
    st.dataframe(
        top20_chain_df[
            [
                "name",
                "total24h",
                "total7d",
                "total30d",
                "total24h_share",
                "total7d_share",
                "total30d_share",
                "share_change",
            ]
        ]
    )

    # Chart: share_change for top 20 chains
    chart_data_chains = top20_chain_df.set_index("name")["share_change"]
    st.bar_chart(chart_data_chains)

with col2:
    st.subheader("Top 20 Protocols")
    top20_protocol_df = (
        protocol_rev_consolidated[
            protocol_rev_consolidated["parentProtocol"].isin(top20_protocols)
        ]
        .sort_values("share_change", ascending=False)
        .reset_index(drop=True)
    )
    st.dataframe(
        top20_protocol_df[
            [
                "parentProtocol",
                "name",
                "category",
                "total24h",
                "total7d",
                "total30d",
                "total24h_share",
                "total7d_share",
                "total30d_share",
                "share_change",
            ]
        ]
    )

    # Chart: share_change for top 20 protocols
    chart_data_protocols = top20_protocol_df.set_index("parentProtocol")[
        "share_change"
    ]
    st.bar_chart(chart_data_protocols)

st.markdown("---")

# --------------------------
# Interactive Chain Details
# --------------------------
st.header("Chain Revenue Details")

all_chains = sorted(chain_rev["name"].dropna().unique().tolist())
default_chain = "Ethereum" if "Ethereum" in all_chains else all_chains[0]

selected_chain = st.selectbox(
    "Select a Chain", all_chains, index=all_chains.index(default_chain)
)

chain_rev_filter = chain_rev[chain_rev["name"] == selected_chain]

if not chain_rev_filter.empty:
    st.subheader(f"Selected Chain: {selected_chain}")

    row = chain_rev_filter.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("24h Revenue", f"{row['total24h']:.2f}")
    c2.metric("7d Revenue", f"{row['total7d']:.2f}")
    c3.metric("30d Revenue", f"{row['total30d']:.2f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("24h Mkt Share", f"{row['total24h_share']*100:.2f}%")
    c5.metric("7d Mkt Share", f"{row['total7d_share']*100:.2f}%")
    c6.metric("30d Mkt Share", f"{row['total30d_share']*100:.2f}%")

    # Chart for this chain (one chart just for chain)
    chart_chain = pd.DataFrame(
        {
            "Period": ["24h", "7d", "30d"],
            "Revenue": [
                row["total24h"],
                row["total7d"],
                row["total30d"],
            ],
        }
    ).set_index("Period")
    st.subheader("Chain Revenue (24h / 7d / 30d)")
    st.bar_chart(chart_chain)
else:
    st.warning("No data found for selected chain.")

st.markdown("---")

# --------------------------
# Interactive Protocol Details
# --------------------------
st.header("Protocol Revenue Details")

all_protocols = sorted(
    protocol_rev_consolidated["parentProtocol"].dropna().unique().tolist()
)

# Default to uniswap if present (case-insensitive)
default_idx = 0
for i, p in enumerate(all_protocols):
    if p.lower() == "uniswap":
        default_idx = i
        break

selected_protocol = st.selectbox(
    "Select a Protocol (parentProtocol)", all_protocols, index=default_idx
)

protocol_rev_filter = protocol_rev_consolidated[
    protocol_rev_consolidated["parentProtocol"] == selected_protocol
]

if not protocol_rev_filter.empty:
    st.subheader(f"Selected Protocol: {selected_protocol}")

    row_p = protocol_rev_filter.iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("24h Revenue", f"{row_p['total24h']:.2f}")
    c2.metric("7d Revenue", f"{row_p['total7d']:.2f}")
    c3.metric("30d Revenue", f"{row_p['total30d']:.2f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("24h Mkt Share", f"{row_p['total24h_share']*100:.2f}%")
    c5.metric("7d Mkt Share", f"{row_p['total7d_share']*100:.2f}%")
    c6.metric("30d Mkt Share", f"{row_p['total30d_share']*100:.2f}%")

    # Chart for this protocol (one chart just for protocol)
    chart_protocol = pd.DataFrame(
        {
            "Period": ["24h", "7d", "30d"],
            "Revenue": [
                row_p["total24h"],
                row_p["total7d"],
                row_p["total30d"],
            ],
        }
    ).set_index("Period")
    st.subheader("Protocol Revenue (24h / 7d / 30d)")
    st.bar_chart(chart_protocol)
else:
    st.warning("No data found for selected protocol.")
