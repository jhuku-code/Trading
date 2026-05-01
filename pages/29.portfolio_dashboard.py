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
        "Fundamental","Thematic","Momentum","Volatility Breakout",
        "Trend Following","Return to trend","Value buys"
    ],
    "Futures":[
        "Momentum","Volatility Breakout","Trend Following",
        "Return to trend","Value buys","Long /short"
    ],
    "Options":[
        "Tail risk","Premia Income","Term Structure"
    ],
    "Cash":["Cash"]
}

# ------------------------------------------------------------
# BUILD INPUT TABLE (FIXED TYPES)
# ------------------------------------------------------------

rows = []

for cat, subs in structure.items():

    rows.append({
        "Category":cat, "SubCategory":"",
        "Coin":"", "Weight %":np.nan, "Value":np.nan, "Leverage":np.nan,
        "RowType":"Category"
    })

    for sub in subs:

        rows.append({
            "Category":"", "SubCategory":sub,
            "Coin":"", "Weight %":np.nan, "Value":np.nan, "Leverage":np.nan,
            "RowType":"SubCategory"
        })

        for _ in range(5):

            rows.append({
                "Category":"", "SubCategory":"",
                "Coin":"", "Weight %":0.0, "Value":0.0, "Leverage":1.0,
                "RowType":"Coin"
            })

input_df = pd.DataFrame(rows)

# force numeric columns
for col in ["Weight %","Value","Leverage"]:
    input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

# ------------------------------------------------------------
# PORTFOLIO INPUT TABLE
# ------------------------------------------------------------

st.subheader("Portfolio Inputs")

edited = st.data_editor(
    input_df,
    use_container_width=True,
    disabled=["Category","SubCategory","RowType"],
    column_config={
        "Coin": st.column_config.TextColumn("Coin"),
        "Weight %": st.column_config.NumberColumn("Weight %", step=1),
        "Value": st.column_config.NumberColumn("Value ($)", step=1000),
        "Leverage": st.column_config.NumberColumn("Leverage", step=0.1)
    }
)

# ------------------------------------------------------------
# BUILD PORTFOLIO DATAFRAME
# ------------------------------------------------------------

portfolio_rows = []
current_category = ""
current_sub = ""

for _, row in edited.iterrows():

    if row["RowType"] == "Category":
        current_category = row["Category"]
        continue

    if row["RowType"] == "SubCategory":
        current_sub = row["SubCategory"]
        continue

    coin = str(row["Coin"]).upper()

    if coin == "":
        continue

    weight_raw = pd.to_numeric(row["Weight %"], errors="coerce")
    weight = 0 if pd.isna(weight_raw) else weight_raw / 100

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

if portfolio.empty:
    st.info("Enter coins to begin.")
    st.stop()

# ------------------------------------------------------------
# ADD DATA
# ------------------------------------------------------------

portfolio["Theme"] = portfolio["Coin"].map(ticker_to_theme)
portfolio["Risk"] = portfolio["Coin"].map(vol_df["Volatility"]).fillna(0)
portfolio["Beta"] = portfolio["Coin"].map(beta_df["BTC_Beta"]).fillna(0)

# ------------------------------------------------------------
# CONTRIBUTIONS
# ------------------------------------------------------------

portfolio["CTR"] = portfolio["Weight"] * portfolio["Risk"] * portfolio["Leverage"]
portfolio["CTB"] = portfolio["Weight"] * portfolio["Beta"] * portfolio["Leverage"]

if portfolio["CTR"].sum() != 0:
    portfolio["CTR"] /= portfolio["CTR"].sum()

if portfolio["CTB"].sum() != 0:
    portfolio["CTB"] /= portfolio["CTB"].sum()

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
total_weight = portfolio["Weight"].sum()

if total_weight > 0:

    for cat,val in cat_totals.items():
        if val == 0:
            continue

        if cat=="Spot" and not(spot_min<=val<=spot_max):
            violations.append(f"Category constraint violated: {cat}")

        if cat=="Futures" and not(fut_min<=val<=fut_max):
            violations.append(f"Category constraint violated: {cat}")

        if cat=="Options" and not(opt_min<=val<=opt_max):
            violations.append(f"Category constraint violated: {cat}")

        if cat=="Cash" and not(cash_min<=val<=cash_max):
            violations.append(f"Category constraint violated: {cat}")

    for (cat,sub),val in sub_totals.items():
        if val>sub_limit:
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
    st.plotly_chart(px.pie(values=cat_totals.values, names=cat_totals.index),
                    use_container_width=True)

with col2:
    st.subheader("Risk Contribution")
    st.plotly_chart(px.bar(portfolio, x="Coin", y="CTR", color="Category"),
                    use_container_width=True)

col3,col4 = st.columns(2)

with col3:
    st.subheader("Beta Contribution")
    st.plotly_chart(px.bar(portfolio, x="Coin", y="CTB", color="Category"),
                    use_container_width=True)

with col4:
    st.subheader("Theme Exposure")
    st.plotly_chart(px.bar(portfolio, x="Theme", y="Weight", color="Coin"),
                    use_container_width=True)

# ------------------------------------------------------------
# CONSTRAINT MONITOR
# ------------------------------------------------------------

st.subheader("Constraint Monitor")

if total_weight == 0:
    st.info("Enter weights to activate constraints.")
elif not violations:
    st.success("All constraints satisfied")
else:
    for v in violations:
        st.error(v)

# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------

col5,col6 = st.columns(2)

portfolio_vol = np.sqrt(np.sum((portfolio["Weight"]*portfolio["Risk"]*portfolio["Leverage"])**2))
portfolio_beta = np.sum(portfolio["Weight"]*portfolio["Beta"]*portfolio["Leverage"])

col5.metric("Portfolio Volatility", f"{portfolio_vol:.2%}")
col6.metric("Portfolio BTC Beta", round(portfolio_beta,2))

# ------------------------------------------------------------
# PORTFOLIO TABLE
# ------------------------------------------------------------

st.subheader("Portfolio Table")

for cat in structure:

    cat_df = portfolio[portfolio["Category"]==cat]

    if cat_df.empty:
        continue

    st.markdown(f"## {cat}")
    st.write(f"Category CTR: {cat_df['CTR'].sum():.2%} | CTB: {cat_df['CTB'].sum():.2%}")

    for sub in structure[cat]:

        sub_df = cat_df[cat_df["SubCategory"]==sub]

        if sub_df.empty:
            continue

        st.markdown(f"### {sub}")
        st.write(f"Subcategory CTR: {sub_df['CTR'].sum():.2%} | CTB: {sub_df['CTB'].sum():.2%}")
        st.dataframe(sub_df)

st.success("Dashboard Ready")
