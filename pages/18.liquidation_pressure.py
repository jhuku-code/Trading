import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta

st.set_page_config(page_title="Liquidation Pressure Index", layout="wide")
st.title("Liquidation Pressure Index Dashboard")

API_KEY = st.secrets.get("API_KEY")
BASE_URL = "https://api.coinalyze.net/v1"

PERPS_URL = "https://raw.githubusercontent.com/jhuku-code/Trading/main/Input-Files/perps_list.xlsx"

# -----------------------------------------------------
# USER CONTROLS
# -----------------------------------------------------

col1,col2,col3,col4 = st.columns(4)

with col1:
    lookback_months = st.number_input("Lookback Months",1,12,3)

with col2:
    interval = st.selectbox(
        "Interval",
        ["1hour","2hour","4hour","6hour","12hour","daily"],
        index=2
    )

with col3:
    z_window = st.slider("Zscore Window",50,400,200)

with col4:
    smoothing = st.slider("Indicator Smoothing",1,48,12)

btc_weight = st.slider("BTC Weight in Market LPI",0.0,1.0,0.7)

# -----------------------------------------------------
# LOAD SYMBOLS
# -----------------------------------------------------

@st.cache_data(ttl=3600)
def load_symbols():
    df = pd.read_excel(PERPS_URL)
    return df["Symbol"].dropna().astype(str).tolist()

symbols = load_symbols()

st.write(f"{len(symbols)} perpetual symbols loaded")

# -----------------------------------------------------
# HELPERS
# -----------------------------------------------------

def chunked(iterable,n):
    for i in range(0,len(iterable),n):
        yield iterable[i:i+n]

# -----------------------------------------------------
# FETCH LIQUIDATIONS
# -----------------------------------------------------

@st.cache_data(ttl=1800)
def fetch_liquidations(symbols):

    url = f"{BASE_URL}/liquidation-history"

    from_ts = int((datetime.now()-relativedelta(months=lookback_months)).timestamp())
    to_ts = int(datetime.now().timestamp())

    rows = []

    for batch in chunked(symbols,20):

        params = {
            "symbols":",".join(batch),
            "interval":interval,
            "from":from_ts,
            "to":to_ts,
            "convert_to_usd":"true",
            "api_key":API_KEY
        }

        r = requests.get(url,params=params)

        if r.status_code != 200:
            continue

        data = r.json()

        if not isinstance(data,list):
            continue

        for entry in data:

            symbol = entry.get("symbol")
            history = entry.get("history",[])

            if not history:
                continue

            df = pd.DataFrame(history)

            if df.empty:
                continue

            df["time"] = pd.to_datetime(df["t"],unit="s")
            df["symbol"] = symbol
            df.rename(columns={"l":"long","s":"short"},inplace=True)

            rows.append(df[["time","symbol","long","short"]])

        time.sleep(1)

    if len(rows)==0:
        return pd.DataFrame(),pd.DataFrame()

    df = pd.concat(rows,ignore_index=True)

    long_df = df.pivot(index="time",columns="symbol",values="long")
    short_df = df.pivot(index="time",columns="symbol",values="short")

    return long_df,short_df

# -----------------------------------------------------
# FETCH OPEN INTEREST
# -----------------------------------------------------

@st.cache_data(ttl=1800)
def fetch_oi(symbols):

    url = f"{BASE_URL}/open-interest-history"

    from_ts = int((datetime.now()-relativedelta(months=lookback_months)).timestamp())
    to_ts = int(datetime.now().timestamp())

    rows = []

    for batch in chunked(symbols,20):

        params = {
            "symbols":",".join(batch),
            "interval":interval,
            "from":from_ts,
            "to":to_ts,
            "api_key":API_KEY
        }

        r = requests.get(url,params=params)

        if r.status_code != 200:
            continue

        data = r.json()

        if not isinstance(data,list):
            continue

        for entry in data:

            symbol = entry.get("symbol")
            history = entry.get("history",[])

            if not history:
                continue

            df = pd.DataFrame(history)

            if df.empty:
                continue

            df["time"] = pd.to_datetime(df["t"],unit="s")
            df["symbol"] = symbol
            df.rename(columns={"o":"oi"},inplace=True)

            rows.append(df[["time","symbol","oi"]])

        time.sleep(1)

    if len(rows)==0:
        return pd.DataFrame()

    df = pd.concat(rows,ignore_index=True)

    return df.pivot(index="time",columns="symbol",values="oi")

# -----------------------------------------------------
# FETCH FUNDING
# -----------------------------------------------------

@st.cache_data(ttl=1800)
def fetch_funding(symbols):

    url = f"{BASE_URL}/funding-rate-history"

    from_ts = int((datetime.now()-relativedelta(months=lookback_months)).timestamp())
    to_ts = int(datetime.now().timestamp())

    rows=[]

    for batch in chunked(symbols,20):

        params={
            "symbols":",".join(batch),
            "interval":interval,
            "from":from_ts,
            "to":to_ts,
            "api_key":API_KEY
        }

        r=requests.get(url,params=params)

        if r.status_code!=200:
            continue

        data=r.json()

        if not isinstance(data,list):
            continue

        for entry in data:

            symbol=entry.get("symbol")
            history=entry.get("history",[])

            if not history:
                continue

            df=pd.DataFrame(history)

            if df.empty:
                continue

            df["time"]=pd.to_datetime(df["t"],unit="s")
            df["symbol"]=symbol
            df.rename(columns={"o":"fund"},inplace=True)

            rows.append(df[["time","symbol","fund"]])

        time.sleep(1)

    if len(rows)==0:
        return pd.DataFrame()

    df=pd.concat(rows,ignore_index=True)

    return df.pivot(index="time",columns="symbol",values="fund")*100

# -----------------------------------------------------
# FETCH BTC PRICE
# -----------------------------------------------------

@st.cache_data(ttl=1800)
def fetch_btc():

    url="https://api.binance.com/api/v3/klines"

    params={"symbol":"BTCUSDT","interval":"4h","limit":1000}

    data=requests.get(url,params=params).json()

    df=pd.DataFrame(data)

    df["time"]=pd.to_datetime(df[0],unit="ms")
    df["price"]=df[4].astype(float)

    return df[["time","price"]].set_index("time")

# -----------------------------------------------------
# RUN
# -----------------------------------------------------

if st.button("Run LPI Analysis"):

    with st.spinner("Fetching datasets..."):

        liq_long,liq_short = fetch_liquidations(symbols)
        oi_df = fetch_oi(symbols)
        funding_df = fetch_funding(symbols)
        btc_price = fetch_btc()

    if liq_long.empty or oi_df.empty or funding_df.empty:
        st.error("Required datasets missing from API.")
        st.stop()

    # -------------------------------------------------
    # ALIGN DATASETS
    # -------------------------------------------------

    common_symbols = (
        set(liq_long.columns)
        & set(oi_df.columns)
        & set(funding_df.columns)
    )

    common_symbols = list(common_symbols)

    liq_long = liq_long[common_symbols]
    liq_short = liq_short[common_symbols]
    oi_df = oi_df[common_symbols]
    funding_df = funding_df[common_symbols]

    # -------------------------------------------------
    # LPI COMPONENTS
    # -------------------------------------------------

    liq_signal = np.log((liq_long+1)/(liq_short+1))

    oi_velocity = oi_df.pct_change(6)

    funding_accel = funding_df.diff()

    # -------------------------------------------------
    # ZSCORES
    # -------------------------------------------------

    liq_z = (liq_signal - liq_signal.rolling(z_window).mean()) / liq_signal.rolling(z_window).std()
    oi_z = (oi_velocity - oi_velocity.rolling(z_window).mean()) / oi_velocity.rolling(z_window).std()
    fund_z = (funding_accel - funding_accel.rolling(z_window).mean()) / funding_accel.rolling(z_window).std()

    lpi_df = 0.4*liq_z + 0.35*oi_z + 0.25*fund_z

    lpi_df = lpi_df.rolling(smoothing).mean()

    # -------------------------------------------------
    # MARKET LPI (BTC + ETH)
    # -------------------------------------------------

    btc_cols = [c for c in lpi_df.columns if "BTC" in c]
    eth_cols = [c for c in lpi_df.columns if "ETH" in c]

    btc_lpi = lpi_df[btc_cols].mean(axis=1)
    eth_lpi = lpi_df[eth_cols].mean(axis=1)

    market_lpi = btc_weight*btc_lpi + (1-btc_weight)*eth_lpi

    # -------------------------------------------------
    # BTC PRICE VS MARKET LPI
    # -------------------------------------------------

    st.subheader("BTC Price vs Market LPI")

    plot_df = pd.concat([btc_price,market_lpi.rename("LPI")],axis=1).dropna()

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=plot_df.index,y=plot_df["price"],name="BTC Price"))
    fig.add_trace(go.Scatter(x=plot_df.index,y=plot_df["LPI"],name="Market LPI",yaxis="y2"))

    fig.update_layout(
        height=600,
        yaxis=dict(title="BTC Price"),
        yaxis2=dict(title="LPI",overlaying="y",side="right"),
        hovermode="x unified"
    )

    st.plotly_chart(fig,use_container_width=True)

    # -------------------------------------------------
    # TOP/BOTTOM COINS
    # -------------------------------------------------

    st.subheader("Top / Bottom Coins by Latest LPI")

    latest = lpi_df.iloc[-1].dropna().sort_values(ascending=False)

    col1,col2 = st.columns(2)

    with col1:
        st.markdown("### Top LPI")
        st.dataframe(latest.head(20))

    with col2:
        st.markdown("### Bottom LPI")
        st.dataframe(latest.tail(20))

    # -------------------------------------------------
    # COIN LPI CHART
    # -------------------------------------------------

    st.subheader("Coin-specific LPI Time Series")

    coin = st.selectbox("Select Coin",lpi_df.columns)

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(x=lpi_df.index,y=lpi_df[coin],name="LPI"))

    fig2.update_layout(height=500,hovermode="x unified")

    st.plotly_chart(fig2,use_container_width=True)
