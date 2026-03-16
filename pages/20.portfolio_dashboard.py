import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Portfolio Allocation Dashboard", layout="wide")
st.title("Portfolio Allocation Dashboard")

# ----------------------------------------------------------
# CHECK REQUIRED DATA
# ----------------------------------------------------------

required = [
    "price_theme",
    "ticker_to_theme",
    "coin_volatility",
    "coin_btc_betas"
]

for r in required:
    if r not in st.session_state:
        st.warning(f"{r} not found. Please run the required pages first.")
        st.stop()

ticker_to_theme = st.session_state.ticker_to_theme
vol_df = st.session_state.coin_volatility.set_index("Coin")
beta_df = st.session_state.coin_btc_betas.set_index("Coin")

# ----------------------------------------------------------
# PORTFOLIO STRUCTURE
# ----------------------------------------------------------

structure = {
    "Spot": [
        "Fundamental",
        "Thematic",
        "Momentum",
        "Volatility Breakout",
        "Trend Following",
        "Return to trend",
        "Value buys"
    ],
    "Futures": [
        "Momentum",
        "Volatility Breakout",
        "Trend Following",
        "Return to trend",
        "Value buys",
        "Long /short"
    ],
    "Options": [
        "Tail risk",
        "Premia Income",
        "Term Structure"
    ],
    "Cash": ["Cash"]
}

# ----------------------------------------------------------
# CREATE EDITABLE INPUT TABLE
# ----------------------------------------------------------

rows = []

for cat, subs in structure.items():

    for sub in subs:

        for i in range(5):

            rows.append({
                "Category": cat,
                "SubCategory": sub,
                "Name": "",
                "Theme": "",
                "Weight": 0.0,
                "Value": 0.0,
                "Leverage": 1.0,
                "Risk": np.nan,
                "Beta": np.nan,
                "CTR": np.nan,
                "CTB": np.nan
            })

portfolio_df = pd.DataFrame(rows)

# ----------------------------------------------------------
# INPUT TABLE
# ----------------------------------------------------------

st.subheader("Portfolio Inputs")

edited_df = st.data_editor(
    portfolio_df,
    use_container_width=True,
    num_rows="dynamic",
    column_config={
        "Weight": st.column_config.NumberColumn(step=0.01),
        "Value": st.column_config.NumberColumn(step=1000),
        "Leverage": st.column_config.NumberColumn(step=0.1)
    }
)

# ----------------------------------------------------------
# ENRICH DATA
# ----------------------------------------------------------

for i, row in edited_df.iterrows():

    coin = str(row["Name"]).upper()

    if coin == "":
        continue

    if coin in ticker_to_theme:
        edited_df.loc[i, "Theme"] = ticker_to_theme[coin]

    if coin in vol_df.index:
        edited_df.loc[i, "Risk"] = vol_df.loc[coin, "Volatility"]

    if coin in beta_df.index:
        edited_df.loc[i, "Beta"] = beta_df.loc[coin, "BTC_Beta"]

# ----------------------------------------------------------
# CALCULATE CTR / CTB
# ----------------------------------------------------------

edited_df["CTR"] = edited_df["Weight"] * edited_df["Risk"] * edited_df["Leverage"]
edited_df["CTB"] = edited_df["Weight"] * edited_df["Beta"] * edited_df["Leverage"]

total_ctr = edited_df["CTR"].sum()
total_ctb = edited_df["CTB"].sum()

if total_ctr != 0:
    edited_df["CTR"] = edited_df["CTR"] / total_ctr

if total_ctb != 0:
    edited_df["CTB"] = edited_df["CTB"] / total_ctb

# ----------------------------------------------------------
# CATEGORY TOTALS
# ----------------------------------------------------------

cat_totals = edited_df.groupby("Category")["Weight"].sum()
sub_totals = edited_df.groupby(["Category", "SubCategory"])["Weight"].sum()
theme_totals = edited_df.groupby("Theme")["Weight"].sum()

# ----------------------------------------------------------
# CONSTRAINT SETTINGS
# ----------------------------------------------------------

st.sidebar.header("Constraints")

spot_min, spot_max = st.sidebar.slider("Spot Range", 0.0, 1.0, (0.30, 0.70))
fut_min, fut_max = st.sidebar.slider("Futures Range", 0.0, 1.0, (0.10, 0.40))
opt_min, opt_max = st.sidebar.slider("Options Range", 0.0, 1.0, (0.00, 0.40))
cash_min, cash_max = st.sidebar.slider("Cash Range", 0.0, 1.0, (0.00, 0.40))

sub_limit = st.sidebar.slider("Sub Category Limit", 0.0, 1.0, 0.25)
theme_limit = st.sidebar.slider("Theme Limit", 0.0, 1.0, 0.25)

# ----------------------------------------------------------
# CONSTRAINT CHECK
# ----------------------------------------------------------

violations = []

if not (spot_min <= cat_totals.get("Spot", 0) <= spot_max):
    violations.append("Spot allocation outside range")

if not (fut_min <= cat_totals.get("Futures", 0) <= fut_max):
    violations.append("Futures allocation outside range")

if not (opt_min <= cat_totals.get("Options", 0) <= opt_max):
    violations.append("Options allocation outside range")

if not (cash_min <= cat_totals.get("Cash", 0) <= cash_max):
    violations.append("Cash allocation outside range")

if any(sub_totals > sub_limit):
    violations.append("Sub-category weight limit exceeded")

theme_excess = theme_totals[(theme_totals > theme_limit) & (theme_totals.index != "L1")]

if len(theme_excess) > 0:
    violations.append("Theme exposure exceeded")

# ----------------------------------------------------------
# DASHBOARD LAYOUT
# ----------------------------------------------------------

col1, col2 = st.columns(2)

# ----------------------------------------------------------
# PORTFOLIO ALLOCATION PIE
# ----------------------------------------------------------

with col1:

    st.subheader("Portfolio Allocation")

    fig = px.pie(
        values=cat_totals.values,
        names=cat_totals.index
    )

    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# RISK CONTRIBUTION
# ----------------------------------------------------------

with col2:

    st.subheader("Risk Contribution by Coin")

    risk_df = edited_df.groupby("Name")["CTR"].sum()

    risk_df = risk_df[risk_df.index != ""]

    fig = px.bar(
        risk_df,
        labels={"value": "CTR", "index": "Coin"}
    )

    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# SECOND ROW
# ----------------------------------------------------------

col3, col4 = st.columns(2)

# ----------------------------------------------------------
# THEME EXPOSURE
# ----------------------------------------------------------

with col3:

    st.subheader("Theme Exposure")

    theme_df = theme_totals[theme_totals.index != ""]

    fig = px.bar(
        theme_df,
        labels={"value": "Weight", "index": "Theme"}
    )

    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# CONSTRAINT MONITOR
# ----------------------------------------------------------

with col4:

    st.subheader("Constraint Monitor")

    if len(violations) == 0:
        st.success("All constraints satisfied")
    else:
        for v in violations:
            st.error(v)

# ----------------------------------------------------------
# PORTFOLIO METRICS
# ----------------------------------------------------------

col5, col6 = st.columns(2)

portfolio_vol = np.sqrt(
    np.sum(
        (edited_df["Weight"] *
         edited_df["Risk"] *
         edited_df["Leverage"]) ** 2
    )
)

portfolio_beta = (
    edited_df["Weight"] *
    edited_df["Beta"] *
    edited_df["Leverage"]
).sum()

col5.metric("Portfolio Volatility", f"{portfolio_vol:.2%}")
col6.metric("Portfolio BTC Beta", round(portfolio_beta, 2))

# ----------------------------------------------------------
# HIERARCHICAL TABLE DISPLAY
# ----------------------------------------------------------

st.subheader("Portfolio Table")

for cat in structure:

    st.markdown(f"## {cat}")

    subs = structure[cat]

    for sub in subs:

        sub_df = edited_df[
            (edited_df["Category"] == cat) &
            (edited_df["SubCategory"] == sub)
        ]

        sub_df = sub_df[sub_df["Name"] != ""]

        if len(sub_df) == 0:
            continue

        st.markdown(f"### {sub}")

        st.dataframe(sub_df)

st.success("Dashboard Ready")
