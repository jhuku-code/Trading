import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Alpha Percentile vs BTC", layout="wide")
st.title("Alpha Percentile vs BTC")

# ---------------------------------------------------------
# REFRESH BUTTON
# ---------------------------------------------------------
if st.button("🔄 Refresh data"):
    st.experimental_rerun()

# ---------------------------------------------------------
# INPUT DATA FROM SESSION STATE
# ---------------------------------------------------------
df_price_alpha = st.session_state.get("price_theme", None)

if df_price_alpha is None:
    st.error("`price_theme` not found in session_state. Please load prices into `st.session_state['price_theme']` first.")
    st.stop()

# Ensure we are working with a copy and sorted index
df_price_alpha = df_price_alpha.sort_index().copy()

# ---------------------------------------------------------
# CHECK BTC COLUMN
# ---------------------------------------------------------
if "BTC" not in df_price_alpha.columns:
    st.error("BTC column not found in `price_theme`. Please ensure there is a 'BTC' column.")
    st.stop()

# ---------------------------------------------------------
# STEP 1: Compute returns
# ---------------------------------------------------------
df_ret = df_price_alpha.pct_change()

# Clean df_ret: drop all-NaN columns
df_ret = df_ret.dropna(axis=1, how="all").copy()

# Re-check BTC presence after cleaning
if "BTC" not in df_ret.columns:
    st.error("BTC column missing in returns after cleaning. Check input data.")
    st.stop()

btc_ret = df_ret["BTC"]

# ---------------------------------------------------------
# STEP 2: Function to compute alpha
# ---------------------------------------------------------
def calc_alpha(coin_window, btc_window):
    mask = (~np.isnan(coin_window)) & (~np.isnan(btc_window))
    if mask.sum() < 30:   # require at least 30 valid obs in 90-day window
        return np.nan
    X = btc_window[mask].reshape(-1, 1)
    y = coin_window[mask].reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    return model.intercept_[0]

# ---------------------------------------------------------
# STEP 3: Compute rolling 90-day alpha for all coins
# ---------------------------------------------------------
window = 90
alpha_dict = {}

for coin in df_ret.columns:
    if coin == "BTC":
        continue
    values = []
    for i in range(len(df_ret)):
        if i < window - 1:
            values.append(np.nan)
        else:
            coin_window = df_ret[coin].iloc[i-window+1:i+1].values
            btc_window  = btc_ret.iloc[i-window+1:i+1].values
            values.append(calc_alpha(coin_window, btc_window))
    alpha_dict[coin] = values

df_alpha = pd.DataFrame(alpha_dict, index=df_ret.index)
df_alpha = df_alpha * 100
df_alpha = df_alpha.round(2)

# ---------------------------------------------------------
# STEP 4: Percentiles
# ---------------------------------------------------------
# 4a: Historical percentiles (within-coin over history)
df_alpha_hist_pct = df_alpha.apply(lambda x: x.rank(pct=True))

# 4b: Cross-sectional percentiles (within-day across coins)
df_alpha_xsec_pct = df_alpha.rank(axis=1, pct=True)

# ---------------------------------------------------------
# STEP 5: Last-day rankings and top/bottom 20%
# ---------------------------------------------------------
last_hist_pct = df_alpha_hist_pct.iloc[-1].dropna()
last_xsec_pct = df_alpha_xsec_pct.iloc[-1].dropna()

# Top/bottom 20% (historical)
top_20_hist = last_hist_pct.nlargest(max(1, int(len(last_hist_pct) * 0.20)))
bottom_20_hist = last_hist_pct.nsmallest(max(1, int(len(last_hist_pct) * 0.20)))

# Top/bottom 20% (cross-sectional)
top_20_xsec = last_xsec_pct.nlargest(max(1, int(len(last_xsec_pct) * 0.20)))
bottom_20_xsec = last_xsec_pct.nsmallest(max(1, int(len(last_xsec_pct) * 0.20)))

# Get the "2nd decile"-like slices: [17:]
top_20_hist_2nd = top_20_hist.iloc[17:]
bottom_20_hist_2nd = bottom_20_hist.iloc[17:]
top_20_xsec_2nd = top_20_xsec.iloc[17:]
bottom_20_xsec_2nd = bottom_20_xsec.iloc[17:]

# Convert to DataFrames for table display
def series_to_table(s, value_name="Percentile"):
    if s is None or s.empty:
        return pd.DataFrame(columns=["Coin", value_name])
    df = s.reset_index()
    df.columns = ["Coin", value_name]
    return df

top_20_hist_table = series_to_table(top_20_hist_2nd, "Hist Percentile")
bottom_20_hist_table = series_to_table(bottom_20_hist_2nd, "Hist Percentile")
top_20_xsec_table = series_to_table(top_20_xsec_2nd, "X-sec Percentile")
bottom_20_xsec_table = series_to_table(bottom_20_xsec_2nd, "X-sec Percentile")

# ---------------------------------------------------------
# DISPLAY TABLES
# ---------------------------------------------------------
st.subheader("Alpha Percentile Tables (2nd decile slices)")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Top 20% Historical (2nd slice: [17:])**")
    st.dataframe(top_20_hist_table, use_container_width=True)

with col2:
    st.markdown("**Bottom 20% Historical (2nd slice: [17:])**")
    st.dataframe(bottom_20_hist_table, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    st.markdown("**Top 20% Cross-sectional (2nd slice: [17:])**")
    st.dataframe(top_20_xsec_table, use_container_width=True)

with col4:
    st.markdown("**Bottom 20% Cross-sectional (2nd slice: [17:])**")
    st.dataframe(bottom_20_xsec_table, use_container_width=True)

# ---------------------------------------------------------
# PLOTTING SECTION
# ---------------------------------------------------------
st.subheader("Rolling 90-Day Alpha vs BTC")

available_coins = list(df_alpha.columns)
default_coin = "BNB" if "BNB" in available_coins else (available_coins[0] if available_coins else None)

if not available_coins:
    st.warning("No coins available in df_alpha for plotting.")
else:
    coins_to_plot = st.multiselect(
        "Select coin(s) to plot alpha",
        options=available_coins,
        default=[default_coin] if default_coin else [],
    )

    # Plot rolling alpha
    if coins_to_plot:
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        for coin in coins_to_plot:
            if coin in df_alpha.columns:
                series = df_alpha[coin].dropna()
                ax1.plot(series.index, series.values, label=coin)
        ax1.set_title("Rolling 90-Day Alpha vs BTC")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Alpha")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend()
        st.pyplot(fig1)

        # Plot cross-sectional alpha percentile
        st.subheader("Rolling 90-Day Cross-sectional Alpha Percentile vs BTC")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        for coin in coins_to_plot:
            if coin in df_alpha_xsec_pct.columns:
                series = df_alpha_xsec_pct[coin].dropna()
                ax2.plot(series.index, series.values, label=coin)
        ax2.set_title("Rolling 90-Day Cross-sectional Alpha Percentile vs BTC")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Percentile")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend()
        st.pyplot(fig2)
    else:
        st.info("Select at least one coin to see the alpha plots.")
