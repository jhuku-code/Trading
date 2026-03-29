# defillama_fees.py

import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="DefiLlama Fees Dashboard", layout="wide")

st.title("DefiLlama Fees Dashboard")

# -----------------------------
# Cached data loader
# -----------------------------
@st.cache_data(ttl=3600)
def load_fees_data():
    url = 'https://api.llama.fi/overview/fees?excludeTotalDataChart=true&excludeTotalDataChartBreakdown=true'
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    # Extract protocol data
    protocols = data.get("protocols", [])

    # Convert to DataFrame
    df_fees = pd.DataFrame(protocols)

    # Make sure columns exist
    for col in ["protocolType", "parentProtocol", "slug", "name", "category"]:
        if col not in df_fees.columns:
            df_fees[col] = None

    # === Split into chain_fees and protocol_fees based on protocolType ===
    chain_fees = df_fees[df_fees['protocolType'] == 'chain'].copy()
    protocol_fees = df_fees[df_fees['protocolType'] == 'protocol'].copy()

    # === Clean 'parentProtocol' column in protocol_fees ===
    protocol_fees['parentProtocol'] = protocol_fees['parentProtocol'].astype(str)
    protocol_fees['parentProtocol'] = protocol_fees['parentProtocol'].str.replace(
        'parent#', '', regex=False
    )
    # Replace 'None' string back to NaN so fillna works properly
    protocol_fees['parentProtocol'] = protocol_fees['parentProtocol'].replace("None", pd.NA)

    # === Replace blanks in parentProtocol with slug values ===
    protocol_fees['parentProtocol'] = protocol_fees['parentProtocol'].fillna(protocol_fees['slug'])

    # === Keep only required columns ===
    keep_cols = ['total24h', 'total7d', 'total30d', 'name', 'category', 'slug', 'parentProtocol']
    chain_fees = chain_fees[keep_cols]
    protocol_fees = protocol_fees[keep_cols]

    # Fill NaNs with 0 for numeric columns
    for col in ['total24h', 'total7d', 'total30d']:
        chain_fees[col] = pd.to_numeric(chain_fees[col], errors='coerce').fillna(0)
        protocol_fees[col] = pd.to_numeric(protocol_fees[col], errors='coerce').fillna(0)

    # === Consolidate protocols by parentProtocol ===
    protocol_fees_sorted = protocol_fees.sort_values(
        by=["parentProtocol", "total30d"], ascending=[True, False]
    )

    protocol_fees_consolidated = (
        protocol_fees_sorted.groupby("parentProtocol", as_index=False).agg({
            "total24h": "sum",
            "total7d": "sum",
            "total30d": "sum",
            "name": "first",       # from row with highest total30d (sorted)
            "category": "first",
            "slug": "first"
        })
    )
    protocol_fees_consolidated.reset_index(drop=True, inplace=True)

    # === Chain fees market share calculations ===
    for col in ['total24h', 'total7d', 'total30d']:
        total_sum = chain_fees[col].sum()
        if total_sum == 0:
            chain_fees[f"{col}_share"] = 0
        else:
            chain_fees[f"{col}_share"] = chain_fees[col] / total_sum

    # Market share change: 7d - 30d
    chain_fees['share_change'] = chain_fees['total7d_share'] - chain_fees['total30d_share']

    # Top 20 chains by share_change
    top20_chains_df = chain_fees.sort_values('share_change', ascending=False).head(20)
    top20_chains = top20_chains_df['name'].tolist()

    # === Protocol fees market share calculations ===
    for col in ['total24h', 'total7d', 'total30d']:
        total_sum = protocol_fees_consolidated[col].sum()
        if total_sum == 0:
            protocol_fees_consolidated[f"{col}_share"] = 0
        else:
            protocol_fees_consolidated[f"{col}_share"] = protocol_fees_consolidated[col] / total_sum

    # Market share change: 7d - 30d
    protocol_fees_consolidated['share_change'] = (
        protocol_fees_consolidated['total7d_share'] -
        protocol_fees_consolidated['total30d_share']
    )

    # Top 20 protocols by share_change (using parentProtocol)
    top20_protocols_df = protocol_fees_consolidated.sort_values(
        'share_change', ascending=False
    ).head(20)
    top20_protocols = top20_protocols_df['parentProtocol'].tolist()

    return (
        df_fees,
        chain_fees,
        protocol_fees_consolidated,
        top20_chains_df,
        top20_protocols_df,
        top20_chains,
        top20_protocols
    )


# -----------------------------
# Refresh button
# -----------------------------
col_refresh, _ = st.columns([1, 3])
with col_refresh:
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()   # clear cached data
        st.experimental_rerun() # reload the app with fresh data

# -----------------------------
# Load data
# -----------------------------
(
    df_fees,
    chain_fees,
    protocol_fees_consolidated,
    top20_chains_df,
    top20_protocols_df,
    top20_chains,
    top20_protocols
) = load_fees_data()

# -----------------------------
# Display Top 20 sections
# -----------------------------
st.markdown("## Top 20 by Market Share Change (7d vs 30d)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Top 20 Chains (7d vs 30d share change)")
    st.dataframe(
        top20_chains_df[['name', 'total24h', 'total7d', 'total30d', 'share_change']],
        use_container_width=True
    )

    chains_chart_data = top20_chains_df[['name', 'share_change']].set_index('name')
    st.bar_chart(chains_chart_data)

with col2:
    st.subheader("Top 20 Protocols (7d vs 30d share change)")
    st.dataframe(
        top20_protocols_df[['parentProtocol', 'total24h', 'total7d', 'total30d', 'share_change']],
        use_container_width=True
    )

    protocols_chart_data = top20_protocols_df[['parentProtocol', 'share_change']].set_index('parentProtocol')
    st.bar_chart(protocols_chart_data)

# -----------------------------
# Interactive selection: Chain
# -----------------------------
st.markdown("---")
st.markdown("## Chain Details")

chain_options = chain_fees['name'].dropna().unique().tolist()
chain_options = sorted(chain_options)

selected_chain = st.selectbox(
    "Select a chain",
    options=chain_options,
    index=chain_options.index("Ethereum") if "Ethereum" in chain_options else 0
)

chain_fees_filter = chain_fees[chain_fees["name"] == selected_chain]

st.write(f"### Selected Chain: {selected_chain}")
st.dataframe(chain_fees_filter, use_container_width=True)

# Chart for selected chain: total24h, total7d, total30d
if not chain_fees_filter.empty:
    chain_row = chain_fees_filter.iloc[0]
    chain_chart_df = pd.DataFrame({
        "window": ["total24h", "total7d", "total30d"],
        "fees": [chain_row["total24h"], chain_row["total7d"], chain_row["total30d"]]
    }).set_index("window")

    st.bar_chart(chain_chart_df)

# -----------------------------
# Interactive selection: Protocol
# -----------------------------
st.markdown("---")
st.markdown("## Protocol Details")

protocol_options = protocol_fees_consolidated['parentProtocol'].dropna().unique().tolist()
protocol_options = sorted(protocol_options)

selected_protocol = st.selectbox(
    "Select a protocol (parentProtocol)",
    options=protocol_options,
    index=protocol_options.index("uniswap") if "uniswap" in protocol_options else 0
)

protocol_fees_filter = protocol_fees_consolidated[
    protocol_fees_consolidated["parentProtocol"] == selected_protocol
]

st.write(f"### Selected Protocol: {selected_protocol}")
st.dataframe(protocol_fees_filter, use_container_width=True)

# Chart for selected protocol: total24h, total7d, total30d
if not protocol_fees_filter.empty:
    proto_row = protocol_fees_filter.iloc[0]
    proto_chart_df = pd.DataFrame({
        "window": ["total24h", "total7d", "total30d"],
        "fees": [proto_row["total24h"], proto_row["total7d"], proto_row["total30d"]]
    }).set_index("window")

    st.bar_chart(proto_chart_df)
