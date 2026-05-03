"""
visualization.py
=================
Publication-quality visualizations for SDL factor analysis.
Outputs PNG figures suitable for academic papers and applications.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# Global style settings for academic publication quality
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
})


COLOR_PALETTE = {
    'primary': '#1a5276',
    'secondary': '#e74c3c',
    'accent': '#27ae60',
    'neutral': '#7f8c8d',
    'quintile': ['#2c3e50', '#3498db', '#2ecc71', '#f39c12', '#e74c3c'],
}


def plot_ic_timeseries(ic_results: dict,
                        horizon: str = '5',
                        output_path: str = None):
    """
    Plot the daily IC time series with cumulative IC overlay.
    
    Parameters
    ----------
    ic_results : dict
        Output from ic_test.full_ic_analysis()
    horizon : str
        Horizon to plot (e.g., '1', '5', '10', '20').
    output_path : str, optional
        Path to save the figure.
    """
    if horizon not in ic_results:
        available = list(ic_results.keys())
        raise ValueError(f"Horizon {horizon} not found. Available: {available}")
    
    data = ic_results[horizon]
    ic_series = data['ic_series']
    stats = data['stats']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                     gridspec_kw={'height_ratios': [2, 1]})
    
    dates = ic_series['date'].values
    ics = ic_series['ic'].values
    
    # --- Top panel: Daily IC ---
    ax1.bar(range(len(ics)), ics, color=[
        COLOR_PALETTE['accent'] if v > 0 else COLOR_PALETTE['secondary']
        for v in ics
    ], width=0.8, alpha=0.7, label='Daily IC')
    
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.axhline(y=np.mean(ics), color=COLOR_PALETTE['primary'],
                linewidth=1.5, linestyle='--',
                label=f'Mean IC = {stats["ic_mean"]:.4f}')
    
    ax1.set_ylabel('Rank IC')
    ax1.set_title(f'Daily Cross-Sectional Rank IC — {horizon}d Forward Horizon')
    ax1.legend(loc='upper right')
    
    # Format x-axis with date labels
    n = len(dates)
    tick_positions = np.linspace(0, n - 1, min(10, n), dtype=int)
    tick_labels = [str(d)[:10] for d in dates[tick_positions]]
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, rotation=45)
    
    # --- Bottom panel: Cumulative IC ---
    cum_ic = np.cumsum(ics)
    ax2.fill_between(range(len(cum_ic)), 0, cum_ic,
                      color=COLOR_PALETTE['primary'], alpha=0.3)
    ax2.plot(range(len(cum_ic)), cum_ic, color=COLOR_PALETTE['primary'],
             linewidth=1.5, label=f'Cumul. IC = {cum_ic[-1]:.2f}')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    
    ax2.set_ylabel('Cumulative IC')
    ax2.set_xlabel('Trading Day')
    ax2.legend(loc='upper right')
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, rotation=45)
    
    # Stats annotation
    textstr = (f'ICIR = {stats["icir"]:.3f}  |  '
               f'NW t = {stats["nw_tstat"]:.3f}  |  '
               f'IC > 0 = {stats["ic_pos_ratio"]:.1%}')
    fig.suptitle(textstr, y=1.02, fontsize=11,
                 fontstyle='italic', color=COLOR_PALETTE['neutral'])
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        print(f"  Figure saved: {output_path}")
    
    plt.close()


def plot_group_returns(group_results: dict, output_path: str = None):
    """
    Plot quintile portfolio returns with monotonicity check.
    
    Parameters
    ----------
    group_results : dict
        Output from backtest.quintile_portfolio_test()
    output_path : str, optional
    """
    means = group_results['group_means']
    groups = [f'Q{i+1}' for i in range(5)]
    values = [means[g] for g in groups]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(groups, values,
                  color=COLOR_PALETTE['quintile'],
                  width=0.6, edgecolor='white', linewidth=1.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.0001 if val >= 0 else -0.0005),
                f'{val:.4%}',
                ha='center', va='bottom' if val >= 0 else 'top',
                fontsize=11, fontweight='bold')
    
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('5-Day Forward Return')
    ax.set_xlabel('SDL Factor Quintile (Q1 = Lowest → Q5 = Highest)')
    ax.set_title('Quintile Portfolio Returns — SDL Factor')
    
    # Monotonicity and spread annotation
    spread = group_results['spread']
    mono = group_results['monotonicity']
    annualized = group_results['spread_annualized']
    
    textstr = (f'Q5 − Q1 Spread: {spread:.4%}  '
               f'({annualized:.1%} annualized)\n'
               f'Monotonicity ρ = {mono:.3f}')
    ax.text(0.97, 0.95, textstr, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        print(f"  Figure saved: {output_path}")
    
    plt.close()


def plot_decay_analysis(ic_results: dict, output_path: str = None):
    """
    Plot IC decay across horizons — key diagnostic for factor persistence.
    
    Parameters
    ----------
    ic_results : dict
        Output from ic_test.full_ic_analysis()
    output_path : str, optional
    """
    horizons = sorted(ic_results.keys())
    icirs = [ic_results[h]['stats']['icir'] for h in horizons]
    nw_ts = [ic_results[h]['stats']['nw_tstat'] for h in horizons]
    
    x = [f'{h}d' for h in horizons]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ICIR decay
    colors_icir = [COLOR_PALETTE['accent'] if v > 0.1 else
                   COLOR_PALETTE['neutral'] if v > 0 else
                   COLOR_PALETTE['secondary'] for v in icirs]
    
    ax1.bar(x, icirs, color=colors_icir, width=0.5, edgecolor='white')
    ax1.axhline(y=0.5, color=COLOR_PALETTE['accent'], linestyle='--',
                linewidth=1, alpha=0.7, label='ICIR = 0.5 (strong)')
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.set_ylabel('IC Information Ratio (ICIR)')
    ax1.set_title('ICIR Decay Across Horizons')
    ax1.legend()
    
    # Newey-West t-stat decay
    colors_nw = [COLOR_PALETTE['accent'] if v > 1.96 else
                 '#f39c12' if v > 1.28 else
                 COLOR_PALETTE['secondary'] for v in nw_ts]
    
    ax2.bar(x, nw_ts, color=colors_nw, width=0.5, edgecolor='white')
    ax2.axhline(y=1.96, color=COLOR_PALETTE['primary'], linestyle='--',
                linewidth=1, alpha=0.7, label='95% significance')
    ax2.axhline(y=-1.96, color=COLOR_PALETTE['primary'], linestyle='--',
                linewidth=1, alpha=0.7)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_ylabel('Newey-West Robust t-statistic')
    ax2.set_title('Statistical Significance Across Horizons')
    ax2.legend()
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        print(f"  Figure saved: {output_path}")
    
    plt.close()


def plot_ic_distribution(ic_results: dict,
                          horizon: str = '5',
                          output_path: str = None):
    """
    Plot distribution of daily IC values with normal fit.
    
    Parameters
    ----------
    ic_results : dict
        Output from ic_test.full_ic_analysis()
    horizon : str
        Horizon to plot.
    output_path : str, optional
    """
    if horizon not in ic_results:
        return
    
    ics = ic_results[horizon]['ic_series']['ic'].values
    stats_dict = ic_results[horizon]['stats']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram
    n, bins, patches = ax.hist(ics, bins=30, density=True, alpha=0.6,
                                color=COLOR_PALETTE['primary'],
                                edgecolor='white')
    
    # Normal fit
    mu = np.mean(ics)
    sigma = np.std(ics, ddof=1)
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
    from scipy.stats import norm
    ax.plot(x, norm.pdf(x, mu, sigma), 'r-', linewidth=2,
            label=f'N({mu:.3f}, {sigma:.3f})')
    
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.axvline(x=mu, color='red', linestyle='-', linewidth=1.5,
               label=f'Mean = {mu:.4f}')
    
    ax.set_xlabel('Daily Rank IC')
    ax.set_ylabel('Density')
    ax.set_title(f'Distribution of Daily IC — {horizon}d Horizon')
    ax.legend()
    
    textstr = (f'ICIR = {stats_dict["icir"]:.3f}\n'
               f'NW t = {stats_dict["nw_tstat"]:.3f}\n'
               f'IC > 0: {stats_dict["ic_pos_ratio"]:.1%}')
    ax.text(0.97, 0.97, textstr, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        print(f"  Figure saved: {output_path}")
    
    plt.close()
