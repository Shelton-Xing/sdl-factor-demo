"""
backtest.py
===========
Group portfolio backtest for SDL factor evaluation.
Implements quintile portfolio construction and monotonicity test.
"""

import numpy as np
import pandas as pd


def quintile_portfolio_test(panel: pd.DataFrame,
                             factor_col: str = 'sdl_zscore',
                             return_col: str = 'fwd_5d',
                             n_groups: int = 5) -> dict:
    """
    Construct quintile portfolios sorted by SDL factor value.
    
    Methodology:
      1. At each date, sort all stocks by factor value
      2. Divide into n_groups equal-sized portfolios (Q1 = lowest SDL, Q5 = highest)
      3. Compute equal-weighted forward return for each portfolio
      4. Assess monotonicity: do higher SDL stocks earn higher forward returns?
    
    Parameters
    ----------
    panel : pd.DataFrame
        Panel with factor and forward return columns.
    factor_col : str
        Factor column name.
    return_col : str
        Forward return column name.
    n_groups : int
        Number of portfolios (default: 5 for quintiles).
    
    Returns
    -------
    dict
        Results containing:
        - group_means : dict  {group_label: mean_return}
        - group_stds  : dict  {group_label: return_std}
        - group_counts : dict {group_label: count}
        - spread : float — Q5 mean return minus Q1 mean return
        - monotonicity : float — rank correlation between group and return
    """
    valid = panel[[factor_col, return_col]].dropna().copy()
    
    if len(valid) < n_groups * 10:
        raise ValueError(f"Not enough observations ({len(valid)}) for group test.")
    
    # Sort by factor value and assign groups
    valid['group'] = pd.qcut(
        valid[factor_col].rank(method='first'),
        n_groups,
        labels=list(range(n_groups)),
        duplicates='drop'
    )
    
    # Compute group statistics
    group_stats = valid.groupby('group')[return_col].agg(['mean', 'std', 'count'])
    
    groups = sorted(group_stats.index)
    
    # Build clean dict
    group_means = {}
    group_stds = {}
    group_counts = {}
    for g in groups:
        label = f'Q{g+1}'
        group_means[label] = float(group_stats.loc[g, 'mean'])
        group_stds[label] = float(group_stats.loc[g, 'std'])
        group_counts[label] = int(group_stats.loc[g, 'count'])
    
    # Top-bottom spread
    spread = group_means[f'Q{n_groups}'] - group_means['Q1']
    
    # Monotonicity test: rank correlation between group number and return
    from scipy import stats
    group_numbers = list(range(1, n_groups + 1))
    returns = [group_means[f'Q{i+1}'] for i in range(n_groups)]
    monotonicity, _ = stats.spearmanr(group_numbers, returns)
    
    return {
        'group_means': group_means,
        'group_stds': group_stds,
        'group_counts': group_counts,
        'spread': spread,
        'spread_annualized': (1 + spread) ** (252 / 5) - 1,  # Approx annualization
        'monotonicity': float(monotonicity)
    }


def summarize_results(ic_results: dict,
                       group_results: dict) -> dict:
    """
    Produce a consolidated summary of all factor test results.
    
    Parameters
    ----------
    ic_results : dict
        Output from ic_test.full_ic_analysis()
    group_results : dict
        Output from quintile_portfolio_test()
    
    Returns
    -------
    dict
        Formatted summary ready for display and reporting.
    """
    summary = {}
    
    # IC summary table
    ic_rows = []
    for h, data in sorted(ic_results.items()):
        s = data['stats']
        ic_rows.append({
            'horizon': f'{h}d',
            'ic_mean': s['ic_mean'],
            'ic_std': s['ic_std'],
            'icir': s['icir'],
            't_stat': s['t_stat'],
            'p_value': s['p_value'],
            'ic_pos': s['ic_pos_ratio'],
            'nw_t': s['nw_tstat'],
            'verdict': s['verdict'],
            'n_days': s['n_periods']
        })
    summary['ic_table'] = ic_rows
    
    # Group test summary
    summary['group'] = group_results
    
    return summary
