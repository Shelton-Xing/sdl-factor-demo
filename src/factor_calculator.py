"""
factor_calculator.py
====================
SDL (Smart-Dumb Lag) factor computation.
Note: This is a DEMO implementation. The exact formula and parameters used
in production research are proprietary and not disclosed here.
"""

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# [REDACTED] Core proprietary parameters.
# The production SDL factor incorporates narrative heat decomposition,
# attention jerk dynamics, and institutional flow latency estimation.
# For academic demonstration, we use a simplified normalized flow proxy.
# ---------------------------------------------------------------------------
_SCALING_FACTOR = 1e4  # Scaling for cross-sectional comparability


def compute_sdl(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the SDL factor for each stock on each date.
    
    The SDL (Smart-Dumb Lag) factor measures the information asymmetry
    between institutional investors and retail traders. 
    
    In production, SDL = f( institutional_flow_rate, narrative_divergence, 
                              attention_jerk, order_flow_toxicity )
    
    In this academic demo, SDL is approximated by:
        SDL = 主力净流入额 / 收盘价  (scaled)
    
    This captures the directional intensity of institutional trading,
    which has been shown to contain predictive information about
    medium-term price movements (see README for instructions).
    
    Parameters
    ----------
    panel : pd.DataFrame
        Must contain columns: [date, code, flow, close]
    
    Returns
    -------
    pd.DataFrame
        Original panel with added columns: [sdl_norm, sdl_zscore]
    """
    df = panel.copy()
    
    # Ensure date is a regular column, not index
    if df.index.name == 'date' or 'date' not in df.columns:
        df = df.reset_index()
    
    # Compute normalized flow: institutional flow scaled by price
    # This adjusts for the fact that 1 billion RMB of flow on a 50 CNY stock
    # is very different from 1 billion on a 500 CNY stock
    df['sdl_raw'] = df['flow'] / df['close'] / _SCALING_FACTOR
    
    # Cross-sectional standardization for each date
    # Makes SDL values comparable across different market conditions
    sdl_list = []
    for d, grp in df.groupby('date'):
        g = grp.copy()
        vals = g['sdl_raw'].values
        if len(vals) > 1 and np.std(vals, ddof=1) > 0:
            g['sdl_zscore'] = (vals - np.mean(vals)) / np.std(vals, ddof=1)
        else:
            g['sdl_zscore'] = 0.0
        sdl_list.append(g)
    
    df = pd.concat(sdl_list, ignore_index=True)
    
    # Rank transformation for IC calculation (more robust than raw values)
    rank_list = []
    for d, grp in df.groupby('date'):
        g = grp.copy()
        g['sdl_rank'] = stats.rankdata(g['sdl_raw'].fillna(0).values)
        rank_list.append(g)
    
    df = pd.concat(rank_list, ignore_index=True)
    return df


def compute_forward_returns(panel: pd.DataFrame, horizons=[1, 5, 10, 20]) -> pd.DataFrame:
    """
    Compute forward price returns for specified horizons.
    
    Parameters
    ----------
    panel : pd.DataFrame
        Must contain columns: [date, code, close]
    horizons : list
        List of forward horizons in trading days.
    
    Returns
    -------
    pd.DataFrame
        Panel with added columns: fwd_{h}d for each horizon.
    """
    df = panel.copy()
    df.sort_values(['code', 'date'], inplace=True)
    
    for h in horizons:
        col = f'fwd_{h}d'
        # Use .shift(-h) / current to get forward return
        df[col] = df.groupby('code')['close'].transform(
            lambda x: x.shift(-h) / x - 1
        )
    
    return df
