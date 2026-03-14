import time
import requests
import pandas as pd
import numpy as np
import io
import streamlit as st
from datetime import datetime
from dateutil.relativedelta import relativedelta

st.set_page_config(page_title="OI Market Share Dashboard", layout="wide")
st.title("Open Interest Market Share Dashboard")

API_KEY = st.secrets.get("API_KEY")
BASE_URL = "https://api.coinalyze.net/v1"

GITHUB_URL = "https://raw.githubusercontent.com/jhuku-code/Trading/main/Input-Files/perps_list.xlsx"

months_back = st.sidebar.number_input("Lookback window (months)",1,12,3)
top_bottom_n = st.sidebar.number_input("Top / Bottom symbols",5,100,15)

# ---------------------------------------------------------
# LOAD SYMBOLS
# ---------------------------------------------------------

@st.cache_data(ttl=3600)
def load_symbols(url):
    resp = requests.get(url)
    df = pd.read_excel(io.BytesIO(resp.content))
    return df["Symbol"].dropna().astype(str).tolist()

symbols = load_symbols(GITHUB_URL)

# ---------------------------------------------------------
# FETCH OI DATA
# ---------------------------------------------------------

def chunked(iterable,n):
    for i in range(0,len(iterable),n):
        yield iterable[i:i+n]

@st.cache_data(ttl=1800)
def fetch_oi(symbols,months_back):

    url=f"{BASE_URL}/open-interest-history"

    from_ts=int((datetime.now()-relativedelta(months=months_back)).timestamp())
    to_ts=int(datetime.now().timestamp())

    rows=[]

    for batch in chunked(symbols,20):

        params={
            "symbols":",".join(batch),
            "interval":"4hour",
            "from":from_ts,
            "to":to_ts,
            "api_key":API_KEY
        }

        r=requests.get(url,params=params)

        if r.status_code!=200:
            continue

        data=r.json()

        for entry in data:

            symbol=entry["symbol"]
            hist=entry["history"]

            df=pd.DataFrame(hist)

            df["time"]=pd.to_datetime(df["t"],unit="s")
            df.rename(columns={"o":"oi"},inplace=True)

            df=df[["time","oi"]]
            df["symbol"]=symbol

            rows.append(df)

        time.sleep(1)

    df=pd.concat(rows)

    df=df.pivot(index="time",columns="symbol",values="oi").sort_index()

    return df

# ---------------------------------------------------------
# LOAD DATA FROM SESSION OR FETCH
# ---------------------------------------------------------

if "oi_data" in st.session_state:

    oi_data=st.session_state["oi_data"]

else:

    if st.button("Load OI Data"):

        with st.spinner("Downloading Open Interest..."):

            oi_data=fetch_oi(symbols,months_back)

            st.session_state["oi_data"]=oi_data
            st.session_state["oi_last_update"]=datetime.now()

    else:

        st.info("Click 'Load OI Data'")
        st.stop()

st.success("OI data loaded")

st.write("Data shape:",oi_data.shape)

# ---------------------------------------------------------
# TOTAL OI
# ---------------------------------------------------------

st.subheader("Total Open Interest")

total_oi=oi_data.sum(axis=1)

st.line_chart(total_oi)

# ---------------------------------------------------------
# MARKET SHARE
# ---------------------------------------------------------

share=oi_data.div(oi_data.sum(axis=1),axis=0)*100

latest=share.iloc[-1]

avg=share.iloc[:-6].mean()

diff=latest-avg

df=pd.DataFrame({
    "Latest Share":latest,
    "Average Share":avg,
    "Difference":diff
}).sort_values("Difference",ascending=False)

# ---------------------------------------------------------
# TABLES
# ---------------------------------------------------------

col1,col2=st.columns(2)

with col1:
    st.subheader("Top Increase")
    st.dataframe(df.head(top_bottom_n))

with col2:
    st.subheader("Top Decrease")
    st.dataframe(df.tail(top_bottom_n))

# ---------------------------------------------------------
# SYMBOL CHART
# ---------------------------------------------------------

st.subheader("Symbol Market Share")

sym=st.selectbox("Symbol",oi_data.columns)

series=share[sym]

st.line_chart(series)
