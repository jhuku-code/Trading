# app.py

import time
from datetime import datetime, timedelta

import pandas as pd
import requests
import streamlit as st
from dateutil.relativedelta import relativedelta
from pathlib import Path

st.set_page_config(page_title="DeFi TVL Growth & Valuations", layout="wide")

st.title("DeFi TVL Growth & Valuations Dashboard")

DATA_PATH = Path("Input-Files") / "Protocol_links.xlsx"


@st.cache_data(show_spinner="Loading TVL & valuation data...")
def load_all_data():
    # -----------------------------
    # TVL ANALYSIS
    # -----------------------------
    # Read Excel from repo path
    df_links_tvl = pd.read_excel(DATA_PATH, sheet_name="TVL", engine="openpyxl")

    cutoff_date = pd.Timestamp.now() - relativedelta(months=6)
    combined_df = pd.DataFrame()

    # Loop through each protocol link and build combined_df
    for _, row in df_links_tvl.iterrows():
        symbol = row["Symbol"]
        url = row["Url"]

        try:
            resp = requests.get(url)
            resp.raise_for_status()
            data = resp.json()

            tvl_data = data.get("tvl", [])
            df = pd.DataFrame(tvl_data)

            # Handle possible alternate column names
            if "totalLiquidityUSD" not in df.columns and "totalLiquidity" in df.columns:
                df.rename(columns={"totalLiquidity": "totalLiquidityUSD"}, inplace=True)

            if df.empty or "date" not in df.columns or "totalLiquidityUSD" not in df.columns:
                continue

            df["date"] = pd.to_datetime(df["date"], unit="s")
            df = df[df["date"] >= cutoff_date]

            # Keep only date + liquidity, rename liquidity to symbol
            df = df[["date", "totalLiquidityUSD"]].rename(columns={"totalLiquidityUSD": symbol})
            df.set_index("date", inplace=True)

            combined_df = combined_df.join(df, how="outer")

        except Exception as e:
            # In Streamlit, just print errors to the log
            print(f"Error processing {symbol} ({url}): {e}")

        # Sleep to avoid overloading APIs (adjust if needed)
        time.sleep(2)

    # Final formatting
    combined_df = combined_df.sort_index().reset_index()
    combined_df.set_index("date", inplace=True)

    # Filter to yesterday midnight
    yesterday_midnight = (datetime.now() - timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    combined_df = combined_df[combined_df.index <= yesterday_midnight]

    # Map symbol -> category
    combined_list = combined_df.columns.to_list()
    symbol_to_category = dict(zip(df_links_tvl["Symbol"], df_links_tvl["Category"]))

    # We only take categories for symbols that actually have data
    df_categories = [symbol_to_category[symbol] for symbol in combined_list if symbol in symbol_to_category]

    # Make a transposed df with a category column
    df_T = combined_df.T
    df_T["category"] = df_categories

    # Unique category list
    category_list_short = list(set(df_categories))

    # Build category-wise datasets
    category_datasets = {}
    for category in category_list_short:
        filtered_df = df_T[df_T["category"] == category]
        transposed_df = filtered_df.T
        transposed_df = transposed_df.drop("category", axis=0)
        # Store directly by category name
        category_datasets[category] = transposed_df

    # Build percent_tvl_dfs: % share of TVL per protocol in each category
    percent_tvl_dfs = {}
    for category, df_cat in category_datasets.items():
        df_cat = df_cat.copy()
        df_cat["total_tvl"] = df_cat.sum(axis=1)
        percent_df = df_cat.drop(columns="total_tvl").div(df_cat["total_tvl"], axis=0) * 100
        percent_tvl_dfs[category] = percent_df

    # -----------------------------
    # VALUATIONS (Mcap / TVL)
    # -----------------------------
    df_links_mcap = pd.read_excel(DATA_PATH, sheet_name="Mcap", engine="openpyxl")

    cutoff_date_m = pd.Timestamp.now() - relativedelta(months=6)
    combined_df_m = pd.DataFrame()  # kept for consistency, not used directly below

    defillama_symbols = df_links_mcap["Symbol"].to_list()

    # CoinGecko symbol -> id mapping
    coin_list = requests.get("https://api.coingecko.com/api/v3/coins/list").json()
    symbol_to_id = {c["symbol"]: c["id"] for c in coin_list}

    ids = [symbol_to_id[s.lower()] for s in defillama_symbols if s.lower() in symbol_to_id]

    if ids:
        params = {
            "vs_currency": "usd",
            "ids": ",".join(ids),
            "order": "market_cap_desc",
            "sparkline": False,
        }
        resp = requests.get("https://api.coingecko.com/api/v3/coins/markets", params=params)
        resp.raise_for_status()
        data = resp.json()
    else:
        raise ValueError("No valid CoinGecko IDs found for provided symbols.")

    df_market_test = pd.DataFrame(data)
    df_market_caps = df_market_test[["symbol", "market_cap"]]

    # Uppercase symbol for merge
    df_market_caps["symbol"] = df_market_caps["symbol"].str.upper()
    df_market_caps.set_index("symbol", inplace=True)

    # Latest TVL row from combined_df
    TVL_latest = combined_df.tail(1)
    df_renamed_axis = TVL_latest.T.rename_axis("symbol")

    # Merge TVL and mcap
    merged_df = pd.merge(df_renamed_axis, df_market_caps, left_index=True, right_index=True)

    merged_df.rename(columns={merged_df.columns[0]: "TVL"}, inplace=True)
    merged_df = merged_df.dropna()

    merged_df["mcap/TVL"] = merged_df["market_cap"] / merged_df["TVL"]
    merged_df = merged_df.sort_values(by="mcap/TVL", ascending=True)

    return combined_df, percent_tvl_dfs, merged_df, category_list_short


# -----------------------------
# SIDEBAR: REFRESH & CONTROLS
# -----------------------------
st.sidebar.header("Controls")

if st.sidebar.button("🔄 Refresh data"):
    # Clear cache so that next call re-downloads everything
    load_all_data.clear()

combined_df, percent_tvl_dfs, merged_df, category_list_short = load_all_data()

# -----------------------------
# MAIN LAYOUT
# -----------------------------
tab1, tab2, tab3 = st.tabs(["% TVL by Protocol", "Valuations (Mcap/TVL)", "Raw TVL Data"])

# -----------------------------
# TAB 1: % TVL by Protocol (percent_df display & filter)
# -----------------------------
with tab1:
    st.subheader("Category-wise % TVL (percent_df)")

    if not percent_tvl_dfs:
        st.warning("No percent TVL data available.")
    else:
        category = st.selectbox("Select Category", sorted(percent_tvl_dfs.keys()))
        percent_df = percent_tvl_dfs[category].copy()

        st.markdown(f"**Selected category:** `{category}`")

        # Filter by protocol/coin name (columns)
        all_names = percent_df.columns.tolist()
        selected_names = st.multiselect(
            "Filter by protocol (name/symbol)",
            options=all_names,
            default=all_names,
        )

        if selected_names:
            percent_df_filtered = percent_df[selected_names]
        else:
            percent_df_filtered = percent_df

        st.write("### percent_df (filtered)")
        st.dataframe(percent_df_filtered)

        # Optional line chart
        st.write("### % TVL Over Time")
        st.line_chart(percent_df_filtered)

# -----------------------------
# TAB 2: Valuations (merged_df display & filter)
# -----------------------------
with tab2:
    st.subheader("Protocol Valuations (Market Cap / TVL)")
    st.caption("Using latest available TVL from combined_df and market caps from CoinGecko.")

    # Text filter by symbol/name
    name_filter = st.text_input("Filter by symbol/name (contains):", "")

    filtered_merged_df = merged_df.copy()

    if name_filter:
        mask = filtered_merged_df.index.str.contains(name_filter, case=False, na=False)
        filtered_merged_df = filtered_merged_df[mask]

    # Multiselect for specific symbols
    selected_symbols = st.multiselect(
        "Or select specific symbols:",
        options=merged_df.index.tolist(),
        default=[],
    )

    if selected_symbols:
        filtered_merged_df = filtered_merged_df.loc[
            filtered_merged_df.index.intersection(selected_symbols)
        ]

    st.write("### merged_df (filtered)")
    st.dataframe(filtered_merged_df)

# -----------------------------
# TAB 3: Raw TVL data
# -----------------------------
with tab3:
    st.subheader("Raw TVL Time Series (combined_df)")
    st.caption("This is the combined TVL dataframe across all protocols.")

    # Filter columns (protocols) by name
    all_cols = combined_df.columns.tolist()
    selected_cols = st.multiselect(
        "Filter protocols (columns):",
        options=all_cols,
        default=all_cols,
    )

    combined_filtered = combined_df[selected_cols] if selected_cols else combined_df

    st.dataframe(combined_filtered)

    st.write("### TVL Over Time")
    st.line_chart(combined_filtered)
