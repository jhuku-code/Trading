# altcoin_diffusion_matplotlib.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

st.set_page_config(page_title="Altcoin Diffusion Index (matplotlib)", layout="wide")
st.title("Altcoin Diffusion Index (with Moving Average — matplotlib)")

# ------- Helper functions -------
@st.cache_data
def calc_altcoin_diffusion_index(prices: pd.DataFrame, window: int = 180, btc_col: str = "BTC") -> pd.Series:
    """
    Calculates the altcoin diffusion index over a given lookback window (in periods).
    """
    # Ensure datetime index
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices = prices.copy()
        prices.index = pd.to_datetime(prices.index)

    returns = prices.pct_change(window)

    if btc_col not in returns.columns:
        raise KeyError(f"BTC column '{btc_col}' not found in price DataFrame columns: {list(returns.columns)}")

    btc_returns = returns[btc_col]
    altcoins = [col for col in returns.columns if col != btc_col]
    if len(altcoins) == 0:
        return pd.Series(0.0, index=returns.index, name="DiffusionIndex")

    outperform = (returns[altcoins].gt(btc_returns, axis=0)).sum(axis=1)
    diffusion_index = 100.0 * outperform / len(altcoins)
    diffusion_index = diffusion_index.rename("DiffusionIndex")
    return diffusion_index

def find_btc_column(cols):
    if "BTC" in cols:
        return "BTC"
    for c in cols:
        up = c.upper()
        if up == "BTC" or up.startswith("BTC") or up.endswith("BTC"):
            return c
    return cols[0]

# ------- Load price data -------
st.markdown("**Data source:** tries `st.session_state['price_theme']` (from your `app.py`). Otherwise upload CSV.")

price_df = None
if "price_theme" in st.session_state and isinstance(st.session_state["price_theme"], pd.DataFrame):
    price_df = st.session_state["price_theme"]
    st.success("Loaded `price_theme` DataFrame from `st.session_state`.")
else:
    uploaded = st.file_uploader("Upload CSV (index=date/time, columns=tickers)", type=["csv"])
    if uploaded is not None:
        price_df = pd.read_csv(uploaded, index_col=0)
        st.success("CSV uploaded.")

if price_df is None:
    st.info("No price data found. Click sample to generate synthetic demo data or set `st.session_state['price_theme']`.")
    if st.button("Show synthetic sample data"):
        dates = pd.date_range(end=pd.Timestamp.now(), periods=400, freq="D")
        np.random.seed(0)
        price_df = pd.DataFrame({
            "BTC": 1000 * (1 + np.random.normal(0, 0.01, size=len(dates))).cumprod(),
            "ALT1": 200 * (1 + np.random.normal(0, 0.015, size=len(dates))).cumprod(),
            "ALT2": 50 * (1 + np.random.normal(0, 0.02, size=len(dates))).cumprod(),
            "ALT3": 20 * (1 + np.random.normal(0, 0.03, size=len(dates))).cumprod(),
        }, index=dates)
    else:
        st.stop()

# Ensure index is datetime
try:
    if not isinstance(price_df.index, pd.DatetimeIndex):
        price_df = price_df.copy()
        price_df.index = pd.to_datetime(price_df.index)
except Exception:
    st.error("Could not parse index to datetime.")
    st.stop()

st.write("### Available columns (first 20):")
st.write(list(price_df.columns[:20]))

suggested_btc = find_btc_column(list(price_df.columns))
btc_col = st.selectbox("BTC column (benchmark)", options=list(price_df.columns), index=list(price_df.columns).index(suggested_btc))

col1, col2 = st.columns([1,1])
with col1:
    window = st.number_input("Diffusion window (periods for pct_change)", min_value=1, value=90, step=1)
with col2:
    ma_period = st.number_input("Moving average period (periods)", min_value=1, value=30, step=1)

# Compute values
try:
    diffusion_index = calc_altcoin_diffusion_index(price_df, window=window, btc_col=btc_col)
except Exception as e:
    st.error(f"Error computing diffusion index: {e}")
    st.stop()

diff_ma = diffusion_index.rolling(window=ma_period, min_periods=1).mean().rename(f"MA_{ma_period}")

# Prepare plotting DataFrame
plot_df = pd.concat([diffusion_index, diff_ma], axis=1).dropna(subset=["DiffusionIndex"])

# ---------- Matplotlib plot ----------
st.write("### Altcoin Diffusion Index (matplotlib)")

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(plot_df.index, plot_df["DiffusionIndex"], label="Diffusion Index", linewidth=1.5)
ax.plot(plot_df.index, plot_df[f"MA_{ma_period}"], linestyle=(0, (4,2)), label=f"MA {ma_period}", linewidth=1.5)

ax.set_title(f"Altcoin Diffusion Index — window={window}, MA={ma_period}")
ax.set_ylabel("Diffusion Index (%)")
ax.set_xlabel("Date")
ax.grid(True, alpha=0.25)
ax.legend(loc="upper left")

# Improve date formatting on x-axis
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
fig.autofmt_xdate()

st.pyplot(fig)

# Latest values + download
st.write("### Latest values")
latest_row = plot_df.iloc[-1]
st.metric("Diffusion Index", f"{latest_row['DiffusionIndex']:.2f}%")
st.metric(f"MA {ma_period}", f"{latest_row[f'MA_{ma_period}']:.2f}%")

csv = plot_df.reset_index().rename(columns={"index":"date"}).to_csv(index=False)
st.download_button("Download diffusion index CSV", data=csv, file_name="diffusion_index.csv", mime="text/csv")

st.write("### Recent rows")
st.dataframe(plot_df.tail(10).rename(columns={f"MA_{ma_period}": f"MA_{ma_period}"}))
