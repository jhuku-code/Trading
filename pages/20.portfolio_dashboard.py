import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Portfolio Allocation Dashboard", layout="wide")
st.title("Portfolio Allocation Dashboard")

# ------------------------------------------------------------
# REQUIRED DATA
# ------------------------------------------------------------

required = [
    "ticker_to_theme",
    "coin_volatility",
    "coin_btc_betas"
]

for r in required:
    if r not in st.session_state:
        st.warning(f"{r} not available. Run prerequisite pages.")
        st.stop()

ticker_to_theme = st.session_state.ticker_to_theme
vol_df = st.session_state.coin_volatility.set_index("Coin")
beta_df = st.session_state.coin_btc_betas.set_index("Coin")

# ------------------------------------------------------------
# PORTFOLIO STRUCTURE
# ------------------------------------------------------------

structure = {
    "Spot":[
        "Fundamental",
        "Thematic",
        "Momentum",
        "Volatility Breakout",
        "Trend Following",
        "Return to trend",
        "Value buys"
    ],
    "Futures":[
        "Momentum",
        "Volatility Breakout",
        "Trend Following",
        "Return to trend",
        "Value buys",
        "Long /short"
    ],
    "Options":[
        "Tail risk",
        "Premia Income",
        "Term Structure"
    ],
    "Cash":["Cash"]
}

# ------------------------------------------------------------
# BUILD CLEAN INPUT TABLE
# ------------------------------------------------------------

rows = []

for cat, subs in structure.items():

    rows.append({
        "Category":cat,
        "SubCategory":"",
        "Coin":"",
        "Weight %":"",
        "Value":"",
        "Leverage":""
    })

    for sub in subs:

        rows.append({
            "Category":"",
            "SubCategory":sub,
            "Coin":"",
            "Weight %":"",
            "Value":"",
            "Leverage":""
        })

        for i in range(5):

            rows.append({
                "Category":"",
                "SubCategory":"",
                "Coin":"",
                "Weight %":0,
                "Value":0,
                "Leverage":1
            })

input_df = pd.DataFrame(rows)

# ------------------------------------------------------------
# PORTFOLIO INPUT TABLE
# ------------------------------------------------------------

st.subheader("Portfolio Inputs")

edited = st.data_editor(
    input_df,
    use_container_width=True,
    disabled=["Category","SubCategory"]
)

# ------------------------------------------------------------
# CLEAN PORTFOLIO DATA
# ------------------------------------------------------------

portfolio_rows = []

current_category = ""
current_sub = ""

for _, row in edited.iterrows():

    if row["Category"] != "":
        current_category = row["Category"]
        continue

    if row["SubCategory"] != "":
        current_sub = row["SubCategory"]
        continue

    coin = str(row["Coin"]).upper()

    if coin == "":
        continue

    weight_val = pd.to_numeric(row["Weight %"], errors="coerce")
    weight = 0 if pd.isna(weight_val) else weight_val / 100

    value = pd.to_numeric(row["Value"], errors="coerce")
    leverage = pd.to_numeric(row["Leverage"], errors="coerce")

    portfolio_rows.append({
        "Category":current_category,
        "SubCategory":current_sub,
        "Coin":coin,
        "Weight":weight,
        "Value":0 if pd.isna(value) else value,
        "Leverage":1 if pd.isna(leverage) else leverage
    })

portfolio = pd.DataFrame(portfolio_rows)

if len(portfolio) == 0:
    st.info("Enter coins in the Portfolio Inputs table.")
    st.stop()

# ------------------------------------------------------------
# ADD THEME / RISK / BETA
# ------------------------------------------------------------

portfolio["Theme"] = portfolio["Coin"].map(ticker_to_theme)
portfolio["Risk"] = portfolio["Coin"].map(vol_df["Volatility"])
portfolio["Beta"] = portfolio["Coin"].map(beta_df["BTC_Beta"])

# ensure numeric
for col in ["Risk","Beta"]:
    portfolio[col] = pd.to_numeric(portfolio[col], errors="coerce").fillna(0)

# ------------------------------------------------------------
# CONTRIBUTIONS
# ------------------------------------------------------------

portfolio["CTR"] = portfolio["Weight"] * portfolio["Risk"] * portfolio["Leverage"]
portfolio["CTB"] = portfolio["Weight"] * portfolio["Beta"] * portfolio["Leverage"]

portfolio["CTR"] = portfolio["CTR"] / portfolio["CTR"].sum()
portfolio["CTB"] = portfolio["CTB"] / portfolio["CTB"].sum()

# ------------------------------------------------------------
# TOTALS
# ------------------------------------------------------------

cat_totals = portfolio.groupby("Category")["Weight"].sum()
sub_totals = portfolio.groupby(["Category","SubCategory"])["Weight"].sum()
theme_totals = portfolio.groupby("Theme")["Weight"].sum()

# ------------------------------------------------------------
# CONSTRAINT SETTINGS
# ------------------------------------------------------------

st.sidebar.header("Constraints")

spot_min,spot_max = st.sidebar.slider("Spot",0.0,1.0,(0.30,0.70))
fut_min,fut_max = st.sidebar.slider("Futures",0.0,1.0,(0.10,0.40))
opt_min,opt_max = st.sidebar.slider("Options",0.0,1.0,(0.0,0.40))
cash_min,cash_max = st.sidebar.slider("Cash",0.0,1.0,(0.0,0.40))

sub_limit = st.sidebar.slider("Subcategory max",0.0,1.0,0.25)
theme_limit = st.sidebar.slider("Theme max",0.0,1.0,0.25)

# ------------------------------------------------------------
# CONSTRAINT CHECK
# ------------------------------------------------------------

violations = []

for cat,val in cat_totals.items():

    if cat=="Spot" and not(spot_min<=val<=spot_max):
        violations.append(f"Category constraint violated: {cat}")

    if cat=="Futures" and not(fut_min<=val<=fut_max):
        violations.append(f"Category constraint violated: {cat}")

    if cat=="Options" and not(opt_min<=val<=opt_max):
        violations.append(f"Category constraint violated: {cat}")

    if cat=="Cash" and not(cash_min<=val<=cash_max):
        violations.append(f"Category constraint violated: {cat}")

for (cat,sub),val in sub_totals.items():
    if val > sub_limit:
        violations.append(f"Subcategory exceeded: {cat} / {sub}")

for theme,val in theme_totals.items():
    if theme!="L1" and val>theme_limit:
        violations.append(f"Theme exceeded: {theme}")

# ------------------------------------------------------------
# DASHBOARD
# ------------------------------------------------------------

col1,col2 = st.columns(2)

with col1:

    st.subheader("Portfolio Allocation")

    fig = px.pie(
        values=cat_totals.values,
        names=cat_totals.index
    )

    st.plotly_chart(fig,use_container_width=True)

with col2:

    st.subheader("Risk Contribution by Coin")

    fig = px.bar(
        portfolio,
        x="Coin",
        y="CTR",
        color="Category"
    )

    st.plotly_chart(fig,use_container_width=True)

# ------------------------------------------------------------
# SECOND ROW
# ------------------------------------------------------------

col3,col4 = st.columns(2)

with col3:

    st.subheader("Beta Contribution by Coin")

    fig = px.bar(
        portfolio,
        x="Coin",
        y="CTB",
        color="Category"
    )

    st.plotly_chart(fig,use_container_width=True)

with col4:

    st.subheader("Theme Exposure")

    fig = px.bar(
        portfolio,
        x="Theme",
        y="Weight",
        color="Coin"
    )

    st.plotly_chart(fig,use_container_width=True)

# ------------------------------------------------------------
# CONSTRAINT MONITOR
# ------------------------------------------------------------

st.subheader("Constraint Monitor")

if len(violations)==0:
    st.success("All constraints satisfied")
else:
    for v in violations:
        st.error(v)

# ------------------------------------------------------------
# PORTFOLIO METRICS
# ------------------------------------------------------------

col5,col6 = st.columns(2)

portfolio_vol = np.sqrt(
    np.sum((portfolio["Weight"]*portfolio["Risk"]*portfolio["Leverage"])**2)
)

portfolio_beta = np.sum(
    portfolio["Weight"]*portfolio["Beta"]*portfolio["Leverage"]
)

col5.metric("Portfolio Volatility",f"{portfolio_vol:.2%}")
col6.metric("Portfolio BTC Beta",round(portfolio_beta,2))

# ------------------------------------------------------------
# PORTFOLIO TABLE
# ------------------------------------------------------------

st.subheader("Portfolio Table")

for cat in structure:

    cat_df = portfolio[portfolio["Category"]==cat]

    if len(cat_df)==0:
        continue

    st.markdown(f"## {cat}")

    cat_ctr = cat_df["CTR"].sum()
    cat_ctb = cat_df["CTB"].sum()

    st.write(f"Category CTR: {cat_ctr:.2%} | CTB: {cat_ctb:.2%}")

    for sub in structure[cat]:

        sub_df = cat_df[cat_df["SubCategory"]==sub]

        if len(sub_df)==0:
            continue

        st.markdown(f"### {sub}")

        sub_ctr = sub_df["CTR"].sum()
        sub_ctb = sub_df["CTB"].sum()

        st.write(f"Subcategory CTR: {sub_ctr:.2%} | CTB: {sub_ctb:.2%}")

        st.dataframe(sub_df)

st.success("Dashboard Ready")
