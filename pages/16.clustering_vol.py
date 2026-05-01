import streamlit as st
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

st.set_page_config(page_title="BTC Adjusted Clustering", layout="wide")
st.title("BTC-Adjusted Correlation Clustering")

# ----------------------------------------------------------
# CHECK PRICE DATA
# ----------------------------------------------------------

if "price_theme" not in st.session_state or st.session_state.price_theme is None:
    st.warning("Price data not available. Run Theme Returns Dashboard first.")
    st.stop()

price_df = st.session_state.price_theme.copy()

if price_df.shape[1] < 2:
    st.warning("Not enough coins to perform clustering.")
    st.stop()

# ----------------------------------------------------------
# SIDEBAR PARAMETERS
# ----------------------------------------------------------

st.sidebar.header("Clustering Settings")

lookback = st.sidebar.number_input(
    "Return lookback bars",
    min_value=30,
    max_value=2000,
    value=120
)

corr_threshold = st.sidebar.slider(
    "Correlation threshold",
    min_value=0.3,
    max_value=0.95,
    value=0.70,
    step=0.05
)

vol_window = st.sidebar.number_input(
    "Volatility window",
    min_value=10,
    max_value=365,
    value=30
)

# convert correlation threshold to clustering distance
distance_threshold = 1 - corr_threshold

# ----------------------------------------------------------
# COMPUTE RETURNS
# ----------------------------------------------------------

returns = price_df.pct_change().dropna()

if len(returns) > lookback:
    returns = returns.iloc[-lookback:]

# ----------------------------------------------------------
# CHECK BTC EXISTS
# ----------------------------------------------------------

if "BTC" not in returns.columns:
    st.error("BTC column not found in price data. BTC is required.")
    st.stop()

btc_returns = returns["BTC"]

# ----------------------------------------------------------
# COMPUTE BTC BETAS + RESIDUAL RETURNS
# ----------------------------------------------------------

residual_returns = pd.DataFrame(index=returns.index)
betas = {}

for coin in returns.columns:

    if coin == "BTC":
        residual_returns[coin] = returns[coin]
        betas[coin] = 1.0
        continue

    coin_r = returns[coin]

    beta = np.cov(coin_r, btc_returns)[0,1] / np.var(btc_returns)

    betas[coin] = beta

    residual = coin_r - beta * btc_returns

    residual_returns[coin] = residual

beta_df = pd.DataFrame.from_dict(
    betas,
    orient="index",
    columns=["BTC_Beta"]
)

beta_df.index.name = "Coin"
beta_df = beta_df.reset_index()

# ----------------------------------------------------------
# BTC ADJUSTED CORRELATION
# ----------------------------------------------------------

corr_matrix = residual_returns.corr()

# ----------------------------------------------------------
# HIERARCHICAL CLUSTERING (DYNAMIC)
# ----------------------------------------------------------

distance_matrix = 1 - corr_matrix

condensed_dist = squareform(distance_matrix.values)

linkage_matrix = linkage(condensed_dist, method="average")

cluster_labels = fcluster(
    linkage_matrix,
    t=distance_threshold,
    criterion="distance"
)

cluster_df = pd.DataFrame({
    "Coin": corr_matrix.columns,
    "Cluster": cluster_labels
})

cluster_df = cluster_df.sort_values(["Cluster", "Coin"]).reset_index(drop=True)

# ----------------------------------------------------------
# CLUSTER MAP (CLUSTER → COINS)
# ----------------------------------------------------------

cluster_map = (
    cluster_df
    .groupby("Cluster")["Coin"]
    .apply(list)
    .to_dict()
)

# ----------------------------------------------------------
# VOLATILITY CALCULATION
# ----------------------------------------------------------

volatility = returns.rolling(vol_window).std().iloc[-1]

vol_df = pd.DataFrame({
    "Coin": volatility.index,
    "Volatility": volatility.values
})

vol_df = vol_df.sort_values("Volatility", ascending=False)

# ----------------------------------------------------------
# SAVE RESULTS FOR OTHER PAGES
# ----------------------------------------------------------

st.session_state.coin_volatility = vol_df
st.session_state.cluster_assignments = cluster_df
st.session_state.cluster_corr_matrix = corr_matrix
st.session_state.coin_btc_betas = beta_df
st.session_state.cluster_map = cluster_map

# ----------------------------------------------------------
# DISPLAY SUMMARY
# ----------------------------------------------------------

st.subheader("Clustering Summary")

st.write(f"Correlation threshold: **{corr_threshold}**")
st.write(f"Clusters detected: **{len(cluster_map)}**")

# ----------------------------------------------------------
# SHOW CLUSTERS
# ----------------------------------------------------------

st.subheader("Clusters and Coins")

for cluster_id, coins in cluster_map.items():

    st.markdown(f"### Cluster {cluster_id}")

    coin_str = ", ".join(coins)

    st.write(coin_str)

# ----------------------------------------------------------
# SHOW CLUSTER ASSIGNMENTS TABLE
# ----------------------------------------------------------

st.subheader("Cluster Assignments")

st.dataframe(cluster_df)

# ----------------------------------------------------------
# SHOW VOLATILITY
# ----------------------------------------------------------

st.subheader("Coin Volatility")

st.dataframe(vol_df)

# ----------------------------------------------------------
# OPTIONAL DEBUG SECTIONS
# ----------------------------------------------------------

with st.expander("BTC Betas"):

    st.dataframe(beta_df)

with st.expander("BTC-Adjusted Correlation Matrix"):

    st.dataframe(corr_matrix)

st.success("Dynamic clustering completed.")
