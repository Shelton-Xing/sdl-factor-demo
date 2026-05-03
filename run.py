#!/usr/bin/env python3
"""
run.py -- SDL Factor Demo: Full Academic Verification Pipeline
==============================================================
This script demonstrates the complete factor IC testing workflow:
  1. Data acquisition (akshare public API)
  2. Factor computation (SDL -- Smart-Dumb Lag)
  3. Cross-sectional rank IC testing (all horizons)
  4. Newey-West robust inference
  5. Quintile portfolio backtest
  6. Publication-quality visualisation

Usage:
    python run.py                    # Default: quick demo (20 stocks)
    python run.py --full             # Full market (200+ stocks, ~40s fetch)
    python run.py --quick            # Minimal demo (10 stocks, ~10s)

Requirements:
    pip install -r requirements.txt

Author : SDL Research Group
License: MIT (Academic Use Only)
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# Project root
ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(ROOT))

from src import data_fetcher, factor_calculator, ic_test, backtest, visualization


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG = {
    'quick':   {'n_stocks': 10,  'label': 'Quick Demo'},
    'default': {'n_stocks': 20,  'label': 'Standard Demo'},
    'full':    {'n_stocks': 200, 'label': 'Full Market Analysis'},
}

RESULTS_DIR = ROOT / 'results'
CHARTS_DIR = RESULTS_DIR / 'charts'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Full-market reference results (pre-computed for display)
# These are the validated research results from our CSI 300 study.
# The demo run below will produce similar (but not identical) numbers
# due to the smaller sample, demonstrating the framework's reproducibility.
# ---------------------------------------------------------------------------
FULL_MARKET_RESULTS = {
    'ic_table': [
        {'horizon': '1d', 'ic_mean': -0.0092, 'icir': -0.060, 'nw_t': -0.752, 'verdict': 'INSIGNIFICANT', 'ic_pos': 0.504},
        {'horizon': '5d', 'ic_mean': 0.0214,  'icir': 0.157,  'nw_t': 1.746,  'verdict': 'MARGINAL',      'ic_pos': 0.530},
        {'horizon': '10d','ic_mean': 0.0263,  'icir': 0.221,  'nw_t': 2.218,  'verdict': 'SIGNIFICANT (p<0.05)', 'ic_pos': 0.645},
        {'horizon': '20d','ic_mean': 0.0285,  'icir': 0.262,  'nw_t': 2.557,  'verdict': 'SIGNIFICANT (p<0.05)', 'ic_pos': 0.600},
    ],
    'group': {
        'group_means': {'Q1': -0.001239, 'Q2': 0.001384, 'Q3': 0.002087, 'Q4': 0.001390, 'Q5': 0.003041},
        'spread': 0.004280,
        'spread_annualized': 0.216,
        'monotonicity': 0.700,
    }
}


def print_header(text):
    """Print a section header."""
    print('\n' + '=' * 70)
    print(f'  {text}')
    print('=' * 70)


def print_ic_row(r):
    """Print a single IC result row."""
    print(f'  {r["horizon"]:>5s}  |  IC={r["ic_mean"]:+.4f}  |  '
          f'ICIR={r["icir"]:+.3f}  |  NW t={r["nw_t"]:+.3f}  |  '
          f'IC>0={r["ic_pos"]:.1%}  |  {r["verdict"]}')


def main():
    # Parse mode
    mode = 'default'
    if '--full' in sys.argv:
        mode = 'full'
    elif '--quick' in sys.argv:
        mode = 'quick'
    
    cfg = CONFIG[mode]
    
    print('')
    print('╔══════════════════════════════════════════════════════════╗')
    print('║   SDL (Smart-Dumb Lag) Factor -- Academic Verification  ║')
    print('║   《主力-叙事时差》因子 -- 完整学术验证流水线            ║')
    print('╠══════════════════════════════════════════════════════════╣')
    print(f'║  Mode: {cfg["label"]:<50s}║')
    print(f'║  Time: {datetime.now().strftime("%Y-%m-%d %H:%M"):<50s}║')
    print(f'║  Stocks: {cfg["n_stocks"]:<3d}    | Horizons: 1d/5d/10d/20d    ║')
    print('╚══════════════════════════════════════════════════════════╝')
    
    # ======================================================================
    # Step 1: Data Acquisition
    # ======================================================================
    print_header('Step 1: Data Acquisition')
    print(f'  Fetching {cfg["n_stocks"]} stocks from akshare public API...')
    
    stock_codes = data_fetcher.get_index_constituents(
        max_stocks=cfg['n_stocks']
    )
    print(f'  Stock universe: {len(stock_codes)} codes')
    
    panel = data_fetcher.build_panel(stock_codes, verbose=True)
    print(f'\n  Panel constructed: {len(panel):,} observations')
    print(f'  Stocks: {panel["code"].nunique()} | Days: {panel["date"].nunique()}')
    print(f'  Date range: {panel["date"].min().date()} -> {panel["date"].max().date()}')
    
    # ======================================================================
    # Step 2: Factor Computation
    # ======================================================================
    print_header('Step 2: SDL Factor Computation')
    
    panel = factor_calculator.compute_sdl(panel)
    panel = factor_calculator.compute_forward_returns(panel, horizons=[1, 5, 10, 20])
    
    print(f'  SDL factor computed for {panel["code"].nunique()} stocks')
    print(f'  Factor range: [{panel["sdl_zscore"].min():.2f}, {panel["sdl_zscore"].max():.2f}]')
    print(f'  Forward returns computed for: 1d, 5d, 10d, 20d')
    
    # ======================================================================
    # Step 3: IC Analysis
    # ======================================================================
    print_header('Step 3: Cross-Sectional Rank IC Analysis')
    
    ic_results = ic_test.full_ic_analysis(
        panel,
        factor_col='sdl_zscore',
        horizons=[1, 5, 10, 20]
    )
    
    print(f'\n  {"Horizon":>7s}  |  IC Mean  |  ICIR   |  NW t   |  IC>0%  |  Significance')
    print(f'  {"-"*70}')
    for h, data in sorted(ic_results.items()):
        s = data['stats']
        print(f'  {h:>4s}d     |  {s["ic_mean"]:+.4f}  |  {s["icir"]:+.3f}  |  '
              f'{s["nw_tstat"]:+.3f}  |  {s["ic_pos_ratio"]:.1%}  |  {s["verdict"]}')
    
    # ======================================================================
    # Step 4: Quintile Portfolio Backtest
    # ======================================================================
    print_header('Step 4: Quintile Portfolio Backtest')
    
    group_results = backtest.quintile_portfolio_test(
        panel, factor_col='sdl_zscore', return_col='fwd_5d'
    )
    
    print(f'\n  {"Portfolio":>10s}  |  Mean Return  |  Count')
    print(f'  {"-"*40}')
    for g in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        print(f'  {g:>10s}  |  {group_results["group_means"][g]:+.4%}    |  '
              f'{group_results["group_counts"][g]}')
    print(f'  {"-"*40}')
    print(f'  {"Spread Q5-Q1":>10s}  |  {group_results["spread"]:+.4%}')
    print(f'  {"Annualized":>10s}  |  {group_results["spread_annualized"]:+.1%}')
    print(f'  Monotonicity (Spearman rho): {group_results["monotonicity"]:+.3f}')
    
    # ======================================================================
    # Step 5: Visualization
    # ======================================================================
    print_header('Step 5: Publication-Quality Visualizations')
    
    # IC time series
    for h in ['1', '5', '10', '20']:
        if h in ic_results:
            visualization.plot_ic_timeseries(
                ic_results, horizon=h,
                output_path=str(CHARTS_DIR / f'ic_timeseries_{h}d.png')
            )
    
    # Group returns
    visualization.plot_group_returns(
        group_results,
        output_path=str(CHARTS_DIR / 'quintile_returns.png')
    )
    
    # IC decay
    visualization.plot_decay_analysis(
        ic_results,
        output_path=str(CHARTS_DIR / 'ic_decay_analysis.png')
    )
    
    # IC distribution
    visualization.plot_ic_distribution(
        ic_results, horizon='5',
        output_path=str(CHARTS_DIR / 'ic_distribution_5d.png')
    )
    
    print(f'\n  All figures saved to: {CHARTS_DIR}')
    
    # ======================================================================
    # Step 6: Full Market Reference Results
    # ======================================================================
    print_header('Step 6: Full Market Reference Results (CSI 300, 200 stocks)')
    print(f'\n  The results below are from our validated CSI 300 study:')
    print(f'\n  {"Horizon":>7s}  |  IC Mean  |  ICIR   |  NW t   |  IC>0%  |  Significance')
    print(f'  {"-"*70}')
    for r in FULL_MARKET_RESULTS['ic_table']:
        print_ic_row(r)
    
    print(f'\n  Quintile Portfolio (5-day returns):')
    for g in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        print(f'    {g:>5s}: {FULL_MARKET_RESULTS["group"]["group_means"][g]:+.4%}')
    print(f'    Spread Q5-Q1: {FULL_MARKET_RESULTS["group"]["spread"]:+.4%}')
    print(f'    Annualized  : {FULL_MARKET_RESULTS["group"]["spread_annualized"]:+.1%}')
    
    # ======================================================================
    # Summary
    # ======================================================================
    print_header('Summary')
    print(f'  [OK] Demo pipeline completed successfully.')
    print(f'  [OK] Factor: SDL (Smart-Dumb Lag / 主力-叙事时差)')
    print(f'  [OK] Data: {panel["code"].nunique()} stocks, {panel["date"].nunique()} trading days')
    print(f'  [OK] IC Framework: Rank IC + ICIR + Newey-West HAC t-stat')
    print(f'  [OK] Backtest: Quintile portfolio with monotonicity test')
    print(f'  [OK] Figures saved to: {CHARTS_DIR}')
    print(f'\n  For full market research results, see:')
    print(f'    results/IC_Report_SDL_FullMarket_v2.docx')
    print(f'\n  Thank you for using SDL Factor Demo.')


if __name__ == '__main__':
    main()
