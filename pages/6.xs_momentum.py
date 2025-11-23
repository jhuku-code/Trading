import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="XS Momentum / Momentum Quality", layout="wide")
st.title("Momentum Quality (XS Momentum)")

# ---------------------------------------------------------
# INPUT DATA FROM SESSION STATE
# ---------------------------------------------------------
df_4h = st.session_state.get("price_theme", None)
ticker_to_theme = st.session_state.get("ticker_to_theme", None)

if df_4h is None:
    st.error(
        "price_theme not found in st.session_state. "
        "Please run the Themes Tracker page first to populate price_theme."
    )
    st.stop()

if ticker_to_theme is None:
    st.error(
        "ticker_to_theme not found in st.session_state. "
        "Please ensure the ticker-to-theme mapping is stored there on Themes Tracker page."
    )
    st.stop()

# ---------------------------------------------------------
# SIDEBAR: PARAMETERS
# ---------------------------------------------------------
st.sidebar.header("Signal Parameters")

top_n_universe = st.sidebar.slider("Universe: Top N buys", 1, 25, 8)
bottom_n_universe = st.sidebar.slider("Universe: Bottom N sells", 1, 25, 8)

top_n_theme = st.sidebar.slider("Theme: Top N buys", 1, 10, 3)
bottom_n_theme = st.sidebar.slider("Theme: Bottom N sells", 1, 10, 3)

st.sidebar.markdown(
    "_Change sliders and then click **Refresh data** on main page to recalc signals._"
)

# ---------------------------------------------------------
# REFRESH BUTTON
# ---------------------------------------------------------
refresh = st.button("Refresh data")

if "xs_momentum_results" not in st.session_state:
    st.session_state["xs_momentum_results"] = None

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def clean_returns(returns_h: pd.DataFrame, rolling_window: int = 60) -> pd.DataFrame:
    cleaned_data = returns_h.copy()
    if (cleaned_data.iloc[0] == 0).all():
        cleaned_data = cleaned_data.iloc[1:].copy()

    for col in cleaned_data.columns:
        zero_mask = cleaned_data[col] == 0
        if zero_mask.sum() > 0:
            rolling_mean = cleaned_data[col].rolling(window=rolling_window, min_periods=1).mean()
            cleaned_data.loc[zero_mask, col] = rolling_mean.loc[zero_mask]

    return cleaned_data


def compute_idmag_weighted(
    returns_cleaned_h: pd.DataFrame,
    lookback: int = 30
):
    """
    Compute IDMAG as in Equation (2) of 'Frog in the Pan'.
    """
    weights_by_quintile = {
        0: 5 / 15,  # Q1 (smallest abs)
        1: 4 / 15,
        2: 3 / 15,
        3: 2 / 15,
        4: 1 / 15,  # Q5 (largest abs)
    }

    idmag_all = {}

    for coin in returns_cleaned_h.columns:
        r = returns_cleaned_h[coin].dropna()

        if len(r) < lookback:
            idmag_all[coin] = np.nan
            continue

        idmag_values = []

        for i in range(lookback, len(r)):
            window_returns = r.iloc[i - lookback : i]
            abs_returns = window_returns.abs()

            # Rank returns into 5 quintiles (labels 0-4)
            quintile_labels = pd.qcut(
                abs_returns, 5, labels=False, duplicates="drop"
            )

            # Assign weights according to quintile
            weights = quintile_labels.map(weights_by_quintile)

            # Compute sgn(Return_i) * w_i
            signed_weights = np.sign(window_returns) * weights.values

            # Compute PRET
            pret = window_returns.sum()

            # IDMAG formula
            sgn_pret = np.sign(pret) if pret != 0 else 0
            idmag = -(1 / lookback) * sgn_pret * np.sum(signed_weights)
            idmag_values.append(idmag)

        # Average across all rolling windows
        idmag_all[coin] = np.mean(idmag_values) if len(idmag_values) > 0 else np.nan

    # Create dataframe
    idmag_df = pd.DataFrame(
        [{"Name": coin, "IDMAG": idmag_all[coin]} for coin in idmag_all if not pd.isna(idmag_all[coin])]
    )
    idmag_df = idmag_df.sort_values(by="IDMAG", ascending=True)

    # Filter top 80% (drop top 20%)
    cutoff = int(len(idmag_df) * 0.8)
    Mom_Qual = idmag_df.iloc[:cutoff]["Name"].tolist()

    # Filter dataframes
    returns_cleaned_h_filtered = returns_cleaned_h[Mom_Qual]

    return {
        "idmag_df": idmag_df,
        "Mom_Qual": Mom_Qual,
        "returns_cleaned_h_filtered": returns_cleaned_h_filtered,
    }


def get_high_idmag_list(idmag_df: pd.DataFrame, top_pct: float = 0.2):
    """
    Returns list of coin names in the top `top_pct` by IDMAG value.
    """
    if len(idmag_df) == 0:
        return []

    top_cutoff = int(len(idmag_df) * top_pct)
    if top_cutoff == 0:
        return []

    high_idmag_df = idmag_df.iloc[-top_cutoff:]  # Top 20% → end of ascending-sorted df
    High_IDMAG_List = high_idmag_df["Name"].tolist()
    return High_IDMAG_List


def clean_vol_scaled_returns(vol_scaled_returns: pd.DataFrame, rolling_window: int = 60):
    cleaned_data = vol_scaled_returns.copy()

    if (cleaned_data.iloc[0] == 0).all():
        cleaned_data = cleaned_data.iloc[1:].copy()

    for col in cleaned_data.columns:
        zero_mask = cleaned_data[col] == 0
        if zero_mask.sum() > 0:
            rolling_mean = cleaned_data[col].rolling(window=rolling_window, min_periods=1).mean()
            cleaned_data.loc[zero_mask, col] = rolling_mean.loc[zero_mask]

        # Remove extreme outliers
        mean_val = cleaned_data[col].mean()
        std_val = cleaned_data[col].std()
        if std_val > 0:
            outlier_mask = np.abs(cleaned_data[col] - mean_val) > 5 * std_val
            cleaned_data.loc[outlier_mask, col] = (
                np.sign(cleaned_data.loc[outlier_mask, col]) * 5 * std_val + mean_val
            )

    return cleaned_data


def calculate_momentum_scores(
    vol_scaled_returns: pd.DataFrame,
    lookback: int = 60,
    exclude_last: int = 6,
    winsorize_limit: float = 2.0,
):
    shifted_returns = vol_scaled_returns.shift(exclude_last)
    cum_returns = shifted_returns.rolling(window=lookback).sum()

    # Winsorize by column
    for col in cum_returns.columns:
        col_data = cum_returns[col].dropna()
        if len(col_data) > 0:
            q5 = col_data.quantile(0.05)
            q95 = col_data.quantile(0.95)
            cum_returns[col] = cum_returns[col].clip(lower=q5, upper=q95)

    # Cross-sectional z-scores
    mean_cs = cum_returns.mean(axis=1, skipna=True).values[:, None]
    std_cs = cum_returns.std(axis=1, skipna=True).values[:, None]
    z_scores = (cum_returns - mean_cs) / std_cs

    # Ranks: higher z = better momentum ⇒ higher rank
    ranks = z_scores.rank(axis=1, ascending=False)

    return z_scores, ranks


def generate_signals(ranks: pd.DataFrame, top_n: int = 8, bottom_n: int = 8):
    buy_signals_dict = {}
    sell_signals_dict = {}

    for idx in ranks.index:
        row = ranks.loc[idx].dropna()
        if len(row) >= top_n:
            buy_list = row.nsmallest(top_n).index.tolist()
            sell_list = row.nlargest(bottom_n).index.tolist()
            buy_signals_dict[idx] = buy_list
            sell_signals_dict[idx] = sell_list

    buy_signals = pd.Series(buy_signals_dict)
    sell_signals = pd.Series(sell_signals_dict)
    return buy_signals, sell_signals


def format_signal_series_for_display(signal_series: pd.Series) -> pd.DataFrame:
    """
    Turn a pd.Series with index=date, values=list/iterable of tickers
    into a DataFrame with Date / Ticketers column as comma-separated strings.
    """
    def _to_str(val):
        if isinstance(val, (list, tuple, pd.Index, np.ndarray)):
            return ", ".join(map(str, val))
        return str(val)

    df = pd.DataFrame(
        {
            "Date": signal_series.index,
            "Tickers": signal_series.apply(_to_str),
        }
    ).set_index("Date")
    return df


# ---------------------------------------------------------
# MAIN CALCULATION (RUN ON REFRESH OR FIRST TIME)
# ---------------------------------------------------------
if refresh or st.session_state["xs_momentum_results"] is None:
    # ===== RETURNS & CLEANING =====
    returns_h_raw = df_4h.pct_change()
    returns_cleaned_h = clean_returns(returns_h_raw).dropna()

    # ===== IDMAG & MOMENTUM QUALITY FILTER =====
    idmag_result = compute_idmag_weighted(
        returns_cleaned_h=returns_cleaned_h,
        lookback=360,  # configurable
    )

    idmag_df = idmag_result["idmag_df"]
    Mom_Qual = idmag_result["Mom_Qual"]
    returns_cleaned_h_filtered = idmag_result["returns_cleaned_h_filtered"]

    High_IDMAG_List = get_high_idmag_list(idmag_df, top_pct=0.2)

    # ===== VOLATILITY SCALING =====
    returns_h = returns_cleaned_h_filtered.copy()

    # Full period volatility (per coin)
    vol_full_period = returns_h.std()

    # Scale each coin's returns by its full-period volatility
    vol_scaled_returns = returns_h.divide(vol_full_period, axis="columns")

    # Clean up NaN/inf
    vol_scaled_returns = vol_scaled_returns.replace([np.inf, -np.inf], np.nan).fillna(0)
    vol_scaled_returns_cleaned = clean_vol_scaled_returns(vol_scaled_returns)

    # ===== EXCESS MOMENTUM ACROSS UNIVERSE =====
    lookback_universe = 60
    z_scores_universe, ranks_universe = calculate_momentum_scores(
        vol_scaled_returns_cleaned,
        lookback=lookback_universe,
        exclude_last=6,
    )
    buy_signals_universe, sell_signals_universe = generate_signals(
        ranks_universe,
        top_n=top_n_universe,
        bottom_n=bottom_n_universe,
    )

    # Universe signals formatted for display
    universe_buy_df = format_signal_series_for_display(buy_signals_universe)
    universe_sell_df = format_signal_series_for_display(sell_signals_universe)

    # ===== EXCESS MOMENTUM FOR EACH SECTOR / THEME =====
    # Map filtered tickers to themes
    filtered_tickers_h = vol_scaled_returns_cleaned.columns.to_list()
    filtered_theme_values_h = [ticker_to_theme.get(t, "UNKNOWN") for t in filtered_tickers_h]

    vol_scaled_returns_cleaned_T = vol_scaled_returns_cleaned.T
    vol_scaled_returns_cleaned_T["Theme"] = filtered_theme_values_h

    # Short list of themes (can edit as needed)
    theme_list_short = ["DEX", "AI", "DEPIN", "L2", "GAMEFI", "RWA", "ZKPROOF", "CROSS_CHAIN", "MEME", "LENDING"]

    # Build per-theme datasets (time index, tickers as columns, like vol_scaled_returns_cleaned)
    theme_datasets = {}
    for theme in theme_list_short:
        theme_tickers = [
            t for t in filtered_tickers_h
            if ticker_to_theme.get(t, None) == theme
        ]
        if len(theme_tickers) >= 1:
            theme_datasets[theme] = vol_scaled_returns_cleaned[theme_tickers]

    # Generate buy/sell signals per theme (using only last date)
    all_signals = []

    for theme_name, dataset in theme_datasets.items():
        try:
            z_scores_theme, ranks_theme = calculate_momentum_scores(
                dataset, lookback=90, exclude_last=6
            )
            buy_signals_theme, sell_signals_theme = generate_signals(
                ranks_theme,
                top_n=top_n_theme,
                bottom_n=bottom_n_theme,
            )

            if len(buy_signals_theme.index) == 0:
                buy_last_list = []
                sell_last_list = []
            else:
                last_date = buy_signals_theme.index[-1]
                buy_last = buy_signals_theme.loc[last_date]
                sell_last = sell_signals_theme.loc[last_date]

                def _to_iter(x):
                    if isinstance(x, (list, tuple, pd.Index, np.ndarray)):
                        return list(x)
                    return [x] if x is not None else []

                buy_last_list = _to_iter(buy_last)
                sell_last_list = _to_iter(sell_last)

            buy_signals_str = ", ".join(map(str, buy_last_list)) if buy_last_list else ""
            sell_signals_str = ", ".join(map(str, sell_last_list)) if sell_last_list else ""

            all_signals.append(
                {
                    "Theme": theme_name,
                    "Buy": buy_signals_str,
                    "Sell": sell_signals_str,
                }
            )

        except Exception:
            all_signals.append(
                {
                    "Theme": theme_name,
                    "Buy": "",
                    "Sell": "",
                }
            )

    consolidated_signals = pd.DataFrame(all_signals).set_index("Theme")

    # Store everything in session_state
    st.session_state["xs_momentum_results"] = {
        "idmag_df": idmag_df,
        "High_IDMAG_List": High_IDMAG_List,
        "universe_buy_df": universe_buy_df,
        "universe_sell_df": universe_sell_df,
        "consolidated_signals": consolidated_signals,
    }

# ---------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------
results = st.session_state["xs_momentum_results"]

if results is None:
    st.info("Click 'Refresh data' to run the XS Momentum calculations.")
    st.stop()

idmag_df = results["idmag_df"]
High_IDMAG_List = results["High_IDMAG_List"]
universe_buy_df = results["universe_buy_df"]
universe_sell_df = results["universe_sell_df"]
consolidated_signals = results["consolidated_signals"]

# ---- High IDMAG coins (filtered out) ----
st.subheader("High IDMAG Coins (Top 20% – Filtered Out)")
st.write(f"Number of coins filtered out: **{len(High_IDMAG_List)}**")
st.dataframe(pd.DataFrame({"Name": High_IDMAG_List}), use_container_width=True)

# ---- Universe momentum signals ----
st.subheader("Excess Momentum Signals – Universe")

if len(universe_buy_df) > 0:
    available_datetimes = universe_buy_df.index.unique()
    # Ensure sorted
    available_datetimes = available_datetimes.sort_values()

    selected_dt = st.selectbox(
        "Select datetime (4H bar) for universe signals",
        options=list(available_datetimes),
        index=len(available_datetimes) - 1,
        format_func=lambda x: x.strftime("%Y-%m-%d %H:%M"),
    )

    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        st.markdown("**Selected Datetime – Buy Signals**")
        try:
            selected_buy_row = universe_buy_df.loc[selected_dt]
            st.write(selected_buy_row["Tickers"] if "Tickers" in selected_buy_row else selected_buy_row)
        except KeyError:
            st.write("No buy signals for selected datetime.")

    with col_sel2:
        st.markdown("**Selected Datetime – Sell Signals**")
        try:
            selected_sell_row = universe_sell_df.loc[selected_dt]
            st.write(selected_sell_row["Tickers"] if "Tickers" in selected_sell_row else selected_sell_row)
        except KeyError:
            st.write("No sell signals for selected datetime.")

    with st.expander("Show all universe buy/sell signals (all dates)"):
        col_u1, col_u2 = st.columns(2)
        with col_u1:
            st.markdown("**Universe Buy Signals (all dates)**")
            st.dataframe(universe_buy_df, use_container_width=True)
        with col_u2:
            st.markdown("**Universe Sell Signals (all dates)**")
            st.dataframe(universe_sell_df, use_container_width=True)
else:
    st.info("No universe signals available. Try adjusting parameters and clicking Refresh.")

# ---- Sector / Theme momentum signals ----
st.subheader("Excess Momentum Signals – By Theme / Sector")
st.dataframe(consolidated_signals, use_container_width=True)
