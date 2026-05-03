"""
data_fetcher.py
===============
Fetch A-share stock data via akshare public API.
Used for academic demonstration only — NOT for live trading.
"""

import pandas as pd
import akshare as ak
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

DEFAULT_STOCKS = [
    "600519", "000858", "600036", "000333", "000651",  # 消费/银行龙头
    "002415", "300750", "600276", "002304", "000568",  # 科技/医药/白酒
    "000001", "000002", "601166", "600900", "601318",  # 金融/公用
    "002714", "000625", "601012", "300124", "002475",  # 制造/新能源
]

def get_index_constituents(index_code="000300", max_stocks=None):
    """
    Fetch CSI 300 constituent list.
    For demo purposes, returns a manageable subset.
    """
    try:
        df = ak.index_stock_cons(symbol=index_code)
        if 'constituent_code' in df.columns:
            codes = [str(c).strip().zfill(6) for c in df['constituent_code'].tolist()]
        elif '品种代码' in df.columns:
            codes = [str(c).strip().zfill(6) for c in df['品种代码'].tolist()]
        else:
            codes = [str(c).strip().zfill(6)[:6] for c in df.iloc[:, 0].tolist()]
        
        codes = [c for c in codes if c.isdigit() and len(c) == 6]
        codes = sorted(set(codes))
        
        if max_stocks and max_stocks < len(codes):
            codes = codes[:max_stocks]
        
        return codes
    except Exception as e:
        print(f"[Warning] Could not fetch index: {e}")
        print("[Info] Falling back to default stock list.")
        return DEFAULT_STOCKS[:max_stocks] if max_stocks else DEFAULT_STOCKS


def fetch_fund_flow(code):
    """
    Fetch individual stock fund flow data from akshare.
    Returns DataFrame with columns: [date, flow, close]
    """
    try:
        market = "sh" if code.startswith(('6', '9')) else "sz"
        flow = ak.stock_individual_fund_flow(stock=code, market=market)
        if flow.empty or len(flow) < 5:
            return None
        
        flow.columns = [str(c).strip() for c in flow.columns]
        flow['date'] = pd.to_datetime(flow['日期'])
        flow.sort_values('date', inplace=True)
        
        # Identify the institutional flow column
        flow_col = None
        for col in ['主力净流入-净额', '主力净流入', '主力净流入额']:
            if col in flow.columns:
                flow_col = col
                break
        if flow_col is None:
            return None
        
        # Get price data for normalization
        end = datetime.now()
        start = end - timedelta(days=250)
        price = ak.stock_zh_a_hist(
            symbol=code, period="daily",
            start_date=start.strftime('%Y%m%d'),
            end_date=end.strftime('%Y%m%d'),
            adjust="qfq"
        )
        if price.empty:
            return None
        
        price.columns = [str(c).strip() for c in price.columns]
        price['date'] = pd.to_datetime(price['日期'])
        price.sort_values('date', inplace=True)
        
        # Merge on date
        merged = pd.merge(
            flow[['date', flow_col]],
            price[['date', '收盘']],
            on='date', how='inner'
        )
        merged.columns = ['date', 'flow', 'close']
        merged['flow'] = pd.to_numeric(merged['flow'], errors='coerce')
        merged.dropna(subset=['flow', 'close'], inplace=True)
        merged.sort_values('date', inplace=True)
        merged.reset_index(drop=True, inplace=True)
        
        return merged
    except Exception:
        return None


def build_panel(stock_codes, max_workers=10, verbose=True):
    """
    Build a cross-sectional panel of fund flow data.
    
    Parameters
    ----------
    stock_codes : list
        List of A-share stock codes (6-digit strings).
    max_workers : int
        Number of parallel threads for data fetching.
    
    Returns
    -------
    pd.DataFrame
        Panel with columns: [date, code, flow, close]
    """
    results = []
    done = 0
    total = len(stock_codes)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_fund_flow, c): c for c in stock_codes}
        for f in as_completed(futures):
            done += 1
            r = f.result()
            if r is not None:
                code = futures[f]
                r['code'] = code
                results.append(r)
            if verbose and done % 20 == 0:
                print(f"  [{done}/{total}] stocks fetched...")
    
    if not results:
        raise RuntimeError("No data could be fetched. Check internet connectivity.")
    
    panel = pd.concat(results, ignore_index=True)
    panel.sort_values(['date', 'code'], inplace=True)
    panel.reset_index(drop=True, inplace=True)
    
    return panel
