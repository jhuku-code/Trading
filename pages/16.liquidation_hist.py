import time
import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from dateutil.relativedelta import relativedelta

st.set_page_config(page_title="Liquidations Monitor",layout="wide")
st.title("Liquidations Monitor")

API_KEY=st.secrets.get("API_KEY")
BASE_URL="https://api.coinalyze.net/v1"

GITHUB_URL="https://raw.githubusercontent.com/jhuku-code/Trading/main/Input-Files/perps_list.xlsx"

top_n=st.number_input("Top N",5,100,15)
months_back=st.number_input("Lookback months",1,12,3)

# ---------------------------------------------------------
# LOAD SYMBOLS
# ---------------------------------------------------------

@st.cache_data(ttl=3600)
def load_symbols(url):

    df=pd.read_excel(url)

    return df["Symbol"].dropna().astype(str).tolist()

symbols=load_symbols(GITHUB_URL)

# ---------------------------------------------------------
# FETCH LIQUIDATIONS
# ---------------------------------------------------------

def chunked(iterable,n):
    for i in range(0,len(iterable),n):
        yield iterable[i:i+n]

@st.cache_data(ttl=1800)
def fetch_liq(symbols):

    url=f"{BASE_URL}/liquidation-history"

    from_ts=int((datetime.now()-relativedelta(months=months_back)).timestamp())
    to_ts=int(datetime.now().timestamp())

    rows=[]

    for batch in chunked(symbols,20):

        params={
            "symbols":",".join(batch),
            "interval":"4hour",
            "from":from_ts,
            "to":to_ts,
            "convert_to_usd":"true",
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

            df.rename(columns={
                "l":"long_liq",
                "s":"short_liq"
            },inplace=True)

            df=df[["time","long_liq","short_liq"]]

            df["symbol"]=symbol

            rows.append(df)

        time.sleep(1)

    df=pd.concat(rows)

    long=df.pivot(index="time",columns="symbol",values="long_liq")
    short=df.pivot(index="time",columns="symbol",values="short_liq")

    return long,short

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

if "liq_long_data" in st.session_state:

    liq_long=st.session_state["liq_long_data"]
    liq_short=st.session_state["liq_short_data"]

else:

    if st.button("Load Liquidation Data"):

        with st.spinner("Downloading liquidations..."):

            liq_long,liq_short=fetch_liq(symbols)

            st.session_state["liq_long_data"]=liq_long
            st.session_state["liq_short_data"]=liq_short

    else:

        st.info("Click 'Load Liquidation Data'")
        st.stop()

st.success("Liquidations loaded")

# ---------------------------------------------------------
# TOTAL LIQUIDATIONS
# ---------------------------------------------------------

total_long=liq_long.sum(axis=1)
total_short=liq_short.sum(axis=1)

df_total=pd.DataFrame({
    "Total Long":total_long,
    "Total Short":total_short
})

st.subheader("Total Liquidations")

st.line_chart(df_total)

# ---------------------------------------------------------
# LONG / SHORT RATIO
# ---------------------------------------------------------

ratio=liq_long.div(liq_short.replace(0,np.nan))

latest=ratio.iloc[-1]

top=latest.sort_values(ascending=False)

col1,col2=st.columns(2)

with col1:

    st.subheader("Highest L/S Ratio")

    st.dataframe(top.head(top_n).to_frame("Ratio"))

with col2:

    st.subheader("Lowest L/S Ratio")

    st.dataframe(top.tail(top_n).to_frame("Ratio"))

# ---------------------------------------------------------
# SYMBOL CHART
# ---------------------------------------------------------

sym=st.selectbox("Symbol",liq_long.columns)

df_plot=pd.DataFrame({
    "Long":liq_long[sym],
    "Short":liq_short[sym]
})

st.line_chart(df_plot)
