import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, Optional


# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Volatility Breakout", layout="wide")
st.title("Volatility Breakout Signals")


# ---------------------------------------------------------
# INPUT DATA FROM SESSION STATE
# ---------------------------------------------------------
df_h = st.session_state.get("price_theme", None)

if df_h is None or df_h.empty:
    st.error(
        "price_theme not found in st.session_state or is empty.\n\n"
        "Please go to the Themes Tracker page, fetch data, and try again."
    )
    st.stop()


# ---------------------------------------------------------
# SIDEBAR PARAMETERS
# ---------------------------------------------------------
st.sidebar.header("Volatility Breakout Parameters")

historical_window = st.sidebar.number_input(
    "Historical window (bars)", min_value=10, max_value=365, value=90, step=5
)
volatility_window = st.sidebar.number_input(
    "Volatility window (bars)", min_value=2, max_value=48, value=6, step=1
)
historical_threshold = st.sidebar.number_input(
    "Historical z-score threshold", min_value=0.1, max_value=5.0, value=1.5, step=0.1
)
cross_sectional_threshold = st.sidebar.number_input(
    "Cross-sectional z-score threshold", min_value=0.1, max_value=5.0, value=1.0, step=0.1
)

# KEY PART: downside penalty (>1 penalises downside volatility)
downside_penalty = st.sidebar.number_input(
    "Downside volatility penalty", min_value=0.5, max_value=5.0, value=1.5, step=0.1,
    help=">1 means downside volatility reduces the breakout score; upside is preferred."
)

top_n_total = st.sidebar.number_input(
    "Top N signals (base ranking)", min_value=5, max_value=100, value=20, step=5
)

# ---------------------------------------------------------
# REFRESH BUTTON
# ---------------------------------------------------------
refresh = st.button("Refresh data")

if "vol_brkout_results" not in st.session_state:
    st.session_state["vol_brkout_results"] = None


# ---------------------------------------------------------
# CLASS DEFINITION (DIRECTIONAL VOL ADDED)
# ---------------------------------------------------------
class VolatilityBreakoutSignal:
    """
    Volatility breakout signal generator that considers both historical 
    and cross-sectional volatility patterns.
    
    Modified so that upside volatility is preferred and downside volatility
    is penalised.
    """
    
    def __init__(self, 
                 historical_window: int = 30,
                 volatility_window: int = 6,
                 cross_sectional_window: int = 20,
                 historical_threshold: float = 2.0,
                 cross_sectional_threshold: float = 1.5,
                 downside_penalty: float = 1.5):
        """
        Parameters:
        -----------
        historical_window : int
            Lookback period for calculating historical volatility baseline
        volatility_window : int  
            Rolling window for current volatility calculation
        cross_sectional_window : int
            Reserved for future use (cross-sectional smoothing)
        historical_threshold : float
            Z-score threshold for historical volatility breakout
        cross_sectional_threshold : float
            Z-score threshold for cross-sectional volatility breakout
        downside_penalty : float
            Penalty factor for downside volatility. 
            >1 penalises downside more, <1 softens the penalty.
        """
        self.historical_window = historical_window
        self.volatility_window = volatility_window
        self.cross_sectional_window = cross_sectional_window
        self.historical_threshold = historical_threshold
        self.cross_sectional_threshold = cross_sectional_threshold
        self.downside_penalty = downside_penalty
    
    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate log returns from price data."""
        return np.log(df / df.shift(1)).dropna()
    
    def calculate_rolling_volatility(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate rolling 'directional' volatility.
        
        Upside volatility is rewarded, downside volatility is penalised:
        
        combined_vol = std(positive returns) - downside_penalty * std(negative returns)
        """
        # Separate upside and downside
        pos_returns = returns.clip(lower=0)
        neg_returns = returns.clip(upper=0)  # <= 0
        
        upside_vol = pos_returns.rolling(window=self.volatility_window).std()
        downside_vol = neg_returns.rolling(window=self.volatility_window).std().abs()
        
        # Directional volatility: upside preferred, downside penalised
        combined_vol = upside_vol - self.downside_penalty * downside_vol
        
        return combined_vol
    
    def calculate_historical_volatility_zscore(self, volatility: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate z-score of current volatility vs its own historical volatility.
        """
        historical_mean = volatility.rolling(window=self.historical_window).mean()
        historical_std = volatility.rolling(window=self.historical_window).std()
        
        # Avoid division by zero
        historical_std = historical_std.replace(0, np.nan)
        
        z_scores = (volatility - historical_mean) / historical_std
        return z_scores
    
    def calculate_cross_sectional_zscore(self, volatility: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate z-score of each coin's volatility vs cross-sectional average
        at each timestamp.
        """
        cross_sectional_mean = volatility.mean(axis=1)
        cross_sectional_std = volatility.std(axis=1)
        
        cs_mean_matrix = np.broadcast_to(cross_sectional_mean.values.reshape(-1, 1), 
                                         volatility.shape)
        cs_std_matrix = np.broadcast_to(cross_sectional_std.values.reshape(-1, 1), 
                                        volatility.shape)
        
        cs_mean_df = pd.DataFrame(cs_mean_matrix, 
                                  index=volatility.index, 
                                  columns=volatility.columns)
        cs_std_df = pd.DataFrame(cs_std_matrix, 
                                 index=volatility.index, 
                                 columns=volatility.columns)
        
        cs_std_df = cs_std_df.replace(0, np.nan)
        
        z_scores = (volatility - cs_mean_df) / cs_std_df
        return z_scores
    
    def calculate_excess_volatility_score(self, 
                                          historical_zscore: pd.DataFrame,
                                          cross_sectional_zscore: pd.DataFrame,
                                          historical_weight: float = 0.6,
                                          cross_sectional_weight: float = 0.4) -> pd.DataFrame:
        """
        Combine historical and cross-sectional z-scores into an excess volatility score.
        """
        excess_score = (historical_weight * historical_zscore + 
                        cross_sectional_weight * cross_sectional_zscore)
        return excess_score
    
    def generate_signals(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Generate volatility breakout signals and rankings.
        """
        # Step 1: Calculate returns
        returns = self.calculate_returns(df)
        
        # Step 2: Calculate rolling directional volatility
        volatility = self.calculate_rolling_volatility(returns)
        
        # Step 3: Historical z-scores
        historical_zscore = self.calculate_historical_volatility_zscore(volatility)
        
        # Step 4: Cross-sectional z-scores
        cross_sectional_zscore = self.calculate_cross_sectional_zscore(volatility)
        
        # Step 5: Combined excess volatility score
        excess_volatility = self.calculate_excess_volatility_score(
            historical_zscore, cross_sectional_zscore
        )
        
        # Step 6: Binary signals from thresholds
        historical_signals = (historical_zscore > self.historical_threshold).astype(int)
        cross_sectional_signals = (cross_sectional_zscore > self.cross_sectional_threshold).astype(int)
        
        # Combined signal: either condition must be met
        combined_signals = (historical_signals | cross_sectional_signals).astype(int)
        
        # Step 7: Percentile rankings (0–100)
        rankings = excess_volatility.rank(axis=1, pct=True) * 100
        
        # Step 8: Strength-based rankings (0–100 scale, cross-sectional)
        strength_rankings = pd.DataFrame(index=excess_volatility.index, 
                                         columns=excess_volatility.columns)
        
        for timestamp in excess_volatility.index:
            row_data = excess_volatility.loc[timestamp].dropna()
            if len(row_data) > 0:
                ranked = row_data.rank(ascending=False)
                max_rank = len(row_data)
                strength_scores = ((max_rank - ranked + 1) / max_rank) * 100
                strength_rankings.loc[timestamp, strength_scores.index] = strength_scores
        
        return {
            'signals': combined_signals,
            'rankings': rankings,
            'strength_rankings': strength_rankings,
            'excess_volatility': excess_volatility,
            'historical_zscore': historical_zscore,
            'cross_sectional_zscore': cross_sectional_zscore,
            'volatility': volatility,
            'historical_signals': historical_signals,
            'cross_sectional_signals': cross_sectional_signals,
        }
    
    def get_top_signals(self, 
                        signals_dict: Dict[str, pd.DataFrame], 
                        timestamp: Optional[str] = None,
                        top_n: int = 10,
                        require_both_signals: bool = True) -> pd.DataFrame:
        """
        Get top N coins with strongest volatility breakout signals.
        """
        if signals_dict['signals'].empty:
            return pd.DataFrame()
        
        if timestamp is None:
            timestamp = signals_dict['signals'].index[-1]
        
        signals = signals_dict['signals'].loc[timestamp]
        excess_vol = signals_dict['excess_volatility'].loc[timestamp].dropna()
        strength_rank = signals_dict['strength_rankings'].loc[timestamp]
        hist_z = signals_dict['historical_zscore'].loc[timestamp]
        cs_z = signals_dict['cross_sectional_zscore'].loc[timestamp]
        
        if require_both_signals:
            # Filter for coins with combined signal == 1
            signal_coins = signals[signals == 1].index
            
            if len(signal_coins) == 0:
                # No signals → return empty
                return pd.DataFrame()
            
            valid_coins = signal_coins
        else:
            valid_coins = excess_vol.index
        
        # Summary DF
        summary = pd.DataFrame({
            'coin': valid_coins,
            'excess_volatility_score': excess_vol[valid_coins].values,
            'strength_ranking': strength_rank[valid_coins].values,
            'historical_zscore': hist_z[valid_coins].values,
            'cross_sectional_zscore': cs_z[valid_coins].values,
            'has_signal': signals[valid_coins].values if require_both_signals else 'N/A'
        })
        
        # Individual flags
        hist_signals = (hist_z > self.historical_threshold).astype(int)
        cs_signals = (cs_z > self.cross_sectional_threshold).astype(int)
        
        summary['historical_signal'] = hist_signals[valid_coins].values
        summary['cross_sectional_signal'] = cs_signals[valid_coins].values
        
        # Sort by excess volatility
        summary = summary.sort_values('excess_volatility_score', ascending=False)
        
        return summary.head(top_n).reset_index(drop=True)

    def diagnose_signals(self, 
                         signals_dict: Dict[str, pd.DataFrame], 
                         timestamp: Optional[str] = None) -> Dict:
        """
        Diagnose signal counts at a given timestamp.
        """
        if signals_dict['signals'].empty:
            return {}
        
        if timestamp is None:
            timestamp = signals_dict['signals'].index[-1]
        
        hist_z = signals_dict['historical_zscore'].loc[timestamp].dropna()
        cs_z = signals_dict['cross_sectional_zscore'].loc[timestamp].dropna()
        
        hist_signals = (hist_z > self.historical_threshold).sum()
        cs_signals = (cs_z > self.cross_sectional_threshold).sum()
        both_signals = ((hist_z > self.historical_threshold) & 
                        (cs_z > self.cross_sectional_threshold)).sum()
        
        diagnostics = {
            'timestamp': timestamp,
            'total_coins': len(hist_z),
            'historical_threshold': self.historical_threshold,
            'cross_sectional_threshold': self.cross_sectional_threshold,
            'coins_above_historical_threshold': int(hist_signals),
            'coins_above_cross_sectional_threshold': int(cs_signals),
            'coins_above_both_thresholds': int(both_signals),
            'max_historical_zscore': float(hist_z.max()),
            'max_cross_sectional_zscore': float(cs_z.max()),
            'mean_historical_zscore': float(hist_z.mean()),
            'mean_cross_sectional_zscore': float(cs_z.mean()),
            'std_historical_zscore': float(hist_z.std()),
            'std_cross_sectional_zscore': float(cs_z.std())
        }
        
        return diagnostics


# ---------------------------------------------------------
# MAIN CALCULATION (RUN ON REFRESH OR FIRST TIME)
# ---------------------------------------------------------
if refresh or st.session_state["vol_brkout_results"] is None:
    # Initialise signal object with current sidebar params
    vb_signal = VolatilityBreakoutSignal(
        historical_window=historical_window,
        volatility_window=volatility_window,
        historical_threshold=historical_threshold,
        cross_sectional_threshold=cross_sectional_threshold,
        downside_penalty=downside_penalty,
    )

    # Generate all signals
    signals_dict = vb_signal.generate_signals(df_h)

    # Base top N ranking (e.g. 20)
    base_top = vb_signal.get_top_signals(
        signals_dict,
        top_n=top_n_total,
        require_both_signals=True
    )

    # Replicate your original: take rows 11–20 (0-based 10:20)
    if base_top is not None and not base_top.empty:
        top_signals = base_top.iloc[10:20].reset_index(drop=True)
    else:
        top_signals = pd.DataFrame()

    # Diagnostics (optional)
    diagnostics = vb_signal.diagnose_signals(signals_dict)

    st.session_state["vol_brkout_results"] = {
        "signals_dict": signals_dict,
        "base_top": base_top,
        "top_signals": top_signals,
        "diagnostics": diagnostics,
    }


# ---------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------
results = st.session_state["vol_brkout_results"]

if results is None:
    st.info("Click **Refresh data** to compute volatility breakout signals.")
    st.stop()

signals_dict = results["signals_dict"]
base_top = results["base_top"]
top_signals = results["top_signals"]
diagnostics = results["diagnostics"]

st.subheader("Top Signals (Rows 11–20 from base ranking)")
if top_signals is None or top_signals.empty:
    st.warning("No breakout signals under current parameters. Try lowering thresholds.")
else:
    st.dataframe(top_signals, use_container_width=True)

with st.expander("Show Base Top Ranking (Top N)", expanded=False):
    if base_top is not None and not base_top.empty:
        st.dataframe(base_top, use_container_width=True)
    else:
        st.write("No base top ranking available.")

with st.expander("Diagnostics", expanded=False):
    if diagnostics:
        st.json(diagnostics)
    else:
        st.write("No diagnostics available (no signals).")
