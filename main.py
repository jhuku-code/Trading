# main.py
import streamlit as st
import pandas as pd
from typing import Dict

st.set_page_config(page_title="Sentiment Tracker Hub", layout="wide")

# --- Your link dictionary (easy to update) ---
SENTIMENT_LINKS: Dict[str, str] = {
    "Coinbase BTC Premium Index (Coinglass)": "https://www.coinglass.com/pro/i/coinbase-bitcoin-premium-index",
    "BTC Liquidation Map (Coinglass)": "https://www.coinglass.com/pro/futures/LiquidationMap",
    "Perp futures vs Spot Volume (Coinglass)": "https://www.coinglass.com/pro/perpteual-spot-volume",
    "BTC 25Δ Skew (The Block)": "https://www.theblock.co/data/crypto-markets/options/btc-option-skew-delta-25",
    "Crypto Fear & Greed Index (Alternative.me)": "https://alternative.me/crypto/fear-and-greed-index/",
    "Volmex BTC Volatility (Volmex charts)": "https://charts.volmex.finance/symbol/BVIV",
    "Funding Rate Heatmap (Coinglass)": "https://www.coinglass.com/FundingRateHeatMap",
    "Messari Sentiment Signals (Messari)": "https://messari.io/signals?filter=curated&change=absolute&hide=wrapped-coins,stablecoins",
}

# --- Sidebar controls ---
st.sidebar.title("Sentiment Sources")
selected_name = st.sidebar.selectbox("Choose a source to view", list(SENTIMENT_LINKS.keys()))

embed_height = st.sidebar.slider("Embed height (px)", min_value=300, max_value=1500, value=800, step=50)
open_in_new_tab = st.sidebar.checkbox("Also show 'Open in new tab' button", value=True)

# Extra quick table view if wanted
if st.sidebar.checkbox("Show list as table", value=False):
    df = pd.DataFrame(list(SENTIMENT_LINKS.items()), columns=["Name", "URL"])
    st.sidebar.dataframe(df)

# --- Main content ---
st.title("Sentiment Tracker Hub")
st.markdown(
    """
    Select a source on the left. If the site allows embedding, it will appear inline.
    If embedding is blocked by the site, a direct link is provided instead.
    """
)

url = SENTIMENT_LINKS[selected_name]

# Attempt embedding
st.subheader(selected_name)
st.caption(url)

# Show iframe embed (Streamlit iframe). Note: some sites disallow embedding via X-Frame-Options.
try:
    # streamlit.components.v1.iframe works well for many pages
    import streamlit.components.v1 as components
    components.iframe(url, height=embed_height, scrolling=True)
    st.info("If the site refuses to embed (empty/blocked frame), use the link below.")
except Exception as e:
    st.error(f"Embedding failed: {e}")

# Fallback link + button that opens in new tab
st.markdown("---")
if open_in_new_tab:
    # Safe HTML anchor that opens new tab; Streamlit supports unsafe HTML here.
    link_html = f'<a href="{url}" target="_blank" rel="noopener noreferrer">Open {selected_name} in a new tab</a>'
    st.markdown(link_html, unsafe_allow_html=True)
else:
    st.markdown(f"[Open {selected_name}]({url})")

# Optional: show raw url for copy/paste
with st.expander("Show raw URL"):
    st.write(url)

# Helpful note about embedding issues
st.info(
    "Note: some sites set HTTP headers (e.g. X-Frame-Options or CSP) to prevent embedding. "
    "If you see a blank area or the site doesn't load inside the app, use the 'Open in new tab' link above."
)
