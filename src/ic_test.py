"""
ic_test.py
==========
Institutional-grade IC (Information Coefficient) testing suite.
Implements:
  - Cross-sectional rank IC
  - ICIR (IC Information Ratio)
  - Newey-West robust t-statistic
  - IC decay analysis across multiple horizons
"""

import numpy as np
import pandas as pd
from scipy import stats


def compute_daily_rank_ic(panel: pd.DataFrame,
                           factor_col: str,
                           return_col: str,
                           min_stocks: int = 10) -> pd.DataFrame:
    """
    Compute daily cross-sectional rank IC.
    
    At each date t, rank-transform both the factor values and forward returns
    across all stocks, then compute Pearson correlation between the ranks.
    
    Parameters
    ----------
    panel : pd.DataFrame
        Panel data with columns: [date, code, factor_col, return_col]
    factor_col : str
        Name of the factor column.
    return_col : str
        Name of the forward return column.
    min_stocks : int
        Minimum number of stocks required to compute IC for a given date.
    
    Returns
    -------
    pd.DataFrame
        Daily IC series with columns: [date, ic, p_value, n_stocks]
    """
    results = []
    
    for date, group in panel.groupby('date'):
        valid = group[[factor_col, return_col]].dropna()
        n = len(valid)
        
        if n < min_stocks:
            continue
        
        # Rank transform (Spearman-style)
        factor_ranks = stats.rankdata(valid[factor_col].values)
        return_ranks = stats.rankdata(valid[return_col].values)
        
        # Pearson correlation on ranks = Spearman rank correlation
        ic, p_val = stats.pearsonr(factor_ranks, return_ranks)
        
        results.append({
            'date': date,
            'ic': ic,
            'p_value': p_val,
            'n_stocks': n
        })
    
    if not results:
        raise ValueError(f"No valid IC dates found. Check data availability.")
    
    return pd.DataFrame(results).sort_values('date').reset_index(drop=True)


def compute_ic_statistics(ic_series: np.ndarray) -> dict:
    """
    Compute comprehensive IC statistics.
    
    Parameters
    ----------
    ic_series : np.ndarray
        Array of daily IC values.
    
    Returns
    -------
    dict
        Dictionary containing:
        - ic_mean : float  — Mean IC
        - ic_std  : float  — Standard deviation of IC
        - icir    : float  — IC Information Ratio (mean / std)
        - t_stat  : float  — Simple t-statistic
        - p_value : float  — Two-sided p-value
        - ic_pos_ratio : float — Fraction of positive IC days
        - nw_tstat : float — Newey-West robust t-statistic
        - cumulative_ic : float — Sum of all IC values
    """
    n = len(ic_series)
    ic_mean = float(np.mean(ic_series))
    ic_std = float(np.std(ic_series, ddof=1))
    icir = ic_mean / ic_std if ic_std > 0 else 0.0
    
    # Simple t-test
    t_stat = ic_mean / (ic_std / np.sqrt(n)) if ic_std > 0 else 0.0
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))
    
    # IC positive ratio
    ic_pos_ratio = float(np.sum(ic_series > 0) / n)
    
    # Newey-West robust standard error
    # Adjusts for autocorrelation in the IC time series
    nw_tstat = _newey_west_tstat(ic_series)
    
    return {
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'icir': icir,
        't_stat': t_stat,
        'p_value': p_value,
        'ic_pos_ratio': ic_pos_ratio,
        'n_periods': n,
        'nw_tstat': nw_tstat,
        'cumulative_ic': float(np.sum(ic_series))
    }


def _newey_west_tstat(series: np.ndarray, lags: int = None) -> float:
    """
    Compute Newey-West heteroskedasticity and autocorrelation consistent
    (HAC) t-statistic.
    
    Uses Bartlett kernel weights for autocovariance estimation.
    Lag selection follows the rule-of-thumb: lags = floor(n^{1/4}).
    
    Parameters
    ----------
    series : np.ndarray
        Time series data.
    lags : int, optional
        Number of lags for autocovariance. Default: floor(n^{1/4}).
    
    Returns
    -------
    float
        Newey-West adjusted t-statistic.
    """
    n = len(series)
    if n < 3:
        return 0.0
    
    if lags is None:
        lags = int(n ** 0.25)
    lags = min(lags, n - 2)
    
    mean_ser = np.mean(series)
    
    # Autocovariance at lag 0
    gamma0 = np.sum((series - mean_ser) ** 2) / n
    
    # Newey-West variance with Bartlett weights
    nw_var = gamma0
    for j in range(1, lags + 1):
        cov = np.sum((series[:-j] - mean_ser) * (series[j:] - mean_ser)) / n
        weight = 1.0 - j / (lags + 1)  # Bartlett kernel
        nw_var += 2.0 * weight * cov
    
    nw_se = np.sqrt(nw_var / n) if nw_var > 0 else 1e-10
    return float(mean_ser / nw_se)


def full_ic_analysis(panel: pd.DataFrame,
                      factor_col: str = 'sdl_zscore',
                      horizons: list = [1, 5, 10, 20]) -> dict:
    """
    Run complete IC analysis across multiple horizons.
    
    For each horizon:
      1. Compute daily cross-sectional rank IC
      2. Compute IC statistics with Newey-West adjustment
      3. Store full IC time series
    
    Parameters
    ----------
    panel : pd.DataFrame
        Panel with factor and forward return columns.
    factor_col : str
        Factor column name.
    horizons : list
        Forward horizons to test.
    
    Returns
    -------
    dict
        Nested dict: {horizon: {stats, ic_series_df}}
    """
    results = {}
    
    for h in horizons:
        ret_col = f'fwd_{h}d'
        if ret_col not in panel.columns:
            print(f"  Warning: {ret_col} not in panel, skipping.")
            continue
        
        # Compute daily IC
        ic_df = compute_daily_rank_ic(panel, factor_col, ret_col)
        
        # Compute statistics
        stats_dict = compute_ic_statistics(ic_df['ic'].values)
        
        # Determine significance
        nw_t = stats_dict['nw_tstat']
        if abs(nw_t) > 1.96:
            verdict = "SIGNIFICANT (p<0.05)"
        elif abs(nw_t) > 1.28:
            verdict = "MARGINAL (p<0.10)"
        else:
            verdict = "INSIGNIFICANT"
        stats_dict['verdict'] = verdict
        
        results[str(h)] = {
            'stats': stats_dict,
            'ic_series': ic_df
        }
    
    return results
