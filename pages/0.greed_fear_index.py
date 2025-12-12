import streamlit as st
import requests
import pandas as pd
import altair as alt

st.set_page_config(page_title="Fear & Greed Index Time Series", layout="wide")
st.title("Crypto Fear & Greed Index — Time Series (Alternative.me)")

API_URL = "https://api.alternative.me/fng/?limit=0&format=json"

@st.cache_data(ttl=24*60*60)
def fetch_fng_ts():
    r = requests.get(API_URL, timeout=10)
    r.raise_for_status()
    data = r.json().get("data", [])
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_numeric(df["timestamp"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="s")
    df["value"] = pd.to_numeric(df["value"])
    df.rename(columns={"value_classification": "classification"}, inplace=True)
    return df.sort_values("date").reset_index(drop=True)

df = fetch_fng_ts()

st.subheader("Crypto Fear & Greed Index Over Time")

chart = (
    alt.Chart(df)
    .mark_line(point=True)
    .encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("value:Q", title="Fear & Greed Index (0–100)"),
        tooltip=[
            alt.Tooltip("date:T"),
            alt.Tooltip("value:Q", title="Index"),
            alt.Tooltip("classification:N", title="Classification"),
        ],
    )
    .properties(height=500)
)

st.altair_chart(chart, use_container_width=True)

with st.expander("Raw data"):
    st.dataframe(df[["date", "value", "classification"]])
