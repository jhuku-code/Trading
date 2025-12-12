import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Fear & Greed Index", layout="wide")

st.title("Crypto Fear & Greed Index — Embedded from CoinMarketCap")

components.html(
    f"""
    <iframe 
        src="https://coinmarketcap.com/charts/fear-and-greed-index/" 
        style="width: 100%; height: 800px; border:none;"
    ></iframe>
    """,
    height=820
)
