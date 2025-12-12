import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Fear & Greed (embed)", layout="wide")
st.title("Crypto Fear & Greed — Embedded (Alternative.me image)")

# Alternative.me image widget (always shows latest)
img_url = "https://alternative.me/crypto/fear-and-greed-index.png"

st.markdown("**Source:** alternative.me — updates daily")
st.image(img_url, use_column_width=True)

# optional: show a link back to the source
st.markdown("[Open full index on alternative.me](https://alternative.me/crypto/fear-and-greed-index/)")
