
"""
backtest_selection.py

Runs an equal-weight daily rebalancing backtest on the stocks selected 
by your ensemble centrality mechanism (Psi = S_struct * Q).

Compares performance against SPY (S&P 500 ETF).

This code is for evaluating the performance of the 20 stocks that were already selected
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from tqdm import tqdm

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

DATA_DIR = "sp500_panel_data"
SELECTED_DIR = "selected_stocks"
OUTPUT_DIR = "backtest_results"
K = 20
SPLITS = ["train", "val", "test"]

plt.style.use('seaborn-v0_8-whitegrid')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# 2. METRIC FUNCTIONS (ROBUST TO Series OR DataFrame)
# ============================================================================

def annualized_return(cum_returns, n_days):
    if len(cum_returns) < 1 or n_days < 1:
        return np.nan
    if isinstance(cum_returns, pd.DataFrame):
        total_return = cum_returns.iloc[-1, 0] - 1
    else:
        total_return = cum_returns.iloc[-1] - 1
    total_return = float(total_return)
    if total_return <= -1:
        return -1.0
    return (1 + total_return) ** (252 / n_days) - 1

def max_drawdown(cum_returns):
    if len(cum_returns) < 2:
        return np.nan
    if isinstance(cum_returns, pd.DataFrame):
        cum_series = cum_returns.iloc[:, 0]
    else:
        cum_series = cum_returns
    peak = cum_series.expanding().max()
    drawdown = (cum_series - peak) / peak
    return float(drawdown.min())

def sharpe_ratio(returns, annualization_factor=252):
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    if len(returns) < 2 or returns.std() == 0:
        return np.nan
    return float(returns.mean() * np.sqrt(annualization_factor) / returns.std())

# ============================================================================
# 3. BACKTEST FUNCTION
# ============================================================================

def run_backtest_for_split(split, K=20):
    print(f"\n🚀 Backtesting {split.upper()} set (K={K})...")
    
    # --- Load selected stocks ---
    selected_path = os.path.join(SELECTED_DIR, f"selected_stocks_{split}_K{K}.csv")
    if not os.path.exists(selected_path):
        print(f"❌ Selected file not found: {selected_path}")
        return None, None, None, None

    selected_df = pd.read_csv(selected_path, parse_dates=['Date'])
    selected_df['Date'] = pd.to_datetime(selected_df['Date']).dt.date
    selected_df['Ticker_List'] = selected_df['Selected_Tickers'].str.split('|')
    print(f"   Loaded {len(selected_df)} rebalancing days.")
    
    # --- Load closing prices ---
    panel_path = os.path.join(DATA_DIR, f"{split}_panel.csv")
    if not os.path.exists(panel_path):
        print(f"❌ Panel data not found: {panel_path}")
        return None, None, None, None

    panel = pd.read_csv(panel_path, parse_dates=['Date'])
    panel['Date'] = pd.to_datetime(panel['Date']).dt.date
    price_matrix = panel.pivot(index='Date', columns='Ticker', values='Close')
    price_matrix = price_matrix.ffill()
    returns_matrix = price_matrix.pct_change()
    print(f"   Price matrix: {price_matrix.shape}")

    # --- Daily equal-weight backtest ---
    portfolio_returns = []
    rebalance_dates = []

    for _, row in tqdm(selected_df.iterrows(), total=len(selected_df), desc=f"Backtesting {split}"):
        date = row['Date']
        tickers = row['Ticker_List']
        try:
            date_idx = price_matrix.index.get_loc(date)
            if date_idx + 1 >= len(price_matrix.index):
                continue
            next_date = price_matrix.index[date_idx + 1]
        except KeyError:
            continue
        rets = returns_matrix.loc[next_date, tickers].dropna()
        if rets.empty:
            continue
        port_ret = rets.mean()
        if not np.isnan(port_ret):
            portfolio_returns.append(port_ret)
            rebalance_dates.append(next_date)

    if not portfolio_returns:
        print("⚠️ No valid returns computed.")
        return None, None, None, None

    port_ret_series = pd.Series(portfolio_returns, index=rebalance_dates, name='Portfolio')
    print(f"   Generated {len(port_ret_series)} daily returns.")

    # --- Benchmark (SPY) ---
    start_date = port_ret_series.index[0]
    end_date = port_ret_series.index[-1]
    try:
        spy = yf.download("SPY", start=start_date, end=end_date + pd.Timedelta(days=1), progress=False)
        spy_ret = spy['Close'].pct_change().dropna()
        spy_aligned = spy_ret.reindex(port_ret_series.index)
        if isinstance(spy_aligned, pd.DataFrame):
            spy_aligned = spy_aligned.squeeze()
        print(f"   SPY benchmark: {len(spy_aligned)} days aligned.")
    except Exception as e:
        print(f"⚠️ Could not download SPY: {e}")
        spy_aligned = pd.Series(index=port_ret_series.index, data=np.nan)

    # --- Cumulative returns (aligned) ---
    port_cum = (1 + port_ret_series).cumprod()
    spy_cum = (1 + spy_aligned).cumprod()

    # --- Metrics ---
    n_days = len(port_ret_series)
    port_ann_ret = annualized_return(port_cum, n_days)
    port_sharpe = sharpe_ratio(port_ret_series)
    port_dd = max_drawdown(port_cum)
    port_win_rate = (port_ret_series > 0).mean()

    spy_clean = spy_aligned.dropna()
    if len(spy_clean) > 0:
        spy_ann_ret = annualized_return(spy_cum, len(spy_clean))
        spy_sharpe = sharpe_ratio(spy_clean)
        spy_dd = max_drawdown(spy_cum)
        spy_win_rate = (spy_clean > 0).mean()
    else:
        spy_ann_ret = spy_sharpe = spy_dd = spy_win_rate = np.nan

    # --- Print summary ---
    print("\n" + "=" * 60)
    print(f"📊 EQUAL-WEIGHT BACKTEST RESULTS ({split.upper()}, K={K})")
    print("=" * 60)
    print(f"   Rebalancing days: {n_days}")
    print(f"   Portfolio Annualized Return:  {port_ann_ret:.2%}" if not np.isnan(port_ann_ret) else "   Portfolio Annualized Return:  N/A")
    print(f"   SPY Annualized Return:        {spy_ann_ret:.2%}" if not np.isnan(spy_ann_ret) else "   SPY Annualized Return:        N/A")
    print(f"   Portfolio Sharpe Ratio:       {port_sharpe:.3f}" if not np.isnan(port_sharpe) else "   Portfolio Sharpe Ratio:       N/A")
    print(f"   SPY Sharpe Ratio:             {spy_sharpe:.3f}" if not np.isnan(spy_sharpe) else "   SPY Sharpe Ratio:             N/A")
    print(f"   Portfolio Max Drawdown:       {port_dd:.2%}" if not np.isnan(port_dd) else "   Portfolio Max Drawdown:       N/A")
    print(f"   SPY Max Drawdown:             {spy_dd:.2%}" if not np.isnan(spy_dd) else "   SPY Max Drawdown:             N/A")
    print(f"   Portfolio Win Rate (daily):   {port_win_rate:.2%}" if not np.isnan(port_win_rate) else "   Portfolio Win Rate:           N/A")
    print(f"   SPY Win Rate (daily):         {spy_win_rate:.2%}" if not np.isnan(spy_win_rate) else "   SPY Win Rate:                 N/A")

    # --- Save results (now lengths are guaranteed to match) ---
    results = pd.DataFrame({
        'Date': port_cum.index,
        'Portfolio_CumReturn': port_cum.values,
        'SPY_CumReturn': spy_cum.values
    })
    results.to_csv(os.path.join(OUTPUT_DIR, f"backtest_{split}_K{K}.csv"), index=False)

    return port_ret_series, spy_aligned, port_cum, spy_cum

# ============================================================================
# 4. MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("📈 EQUAL-WEIGHT BACKTEST: SELECTION MECHANISM EVALUATION")
    print("=" * 70)

    all_port_cums = {}
    all_spy_cums = {}

    for split in SPLITS:
        port_ret, spy_ret, port_cum, spy_cum = run_backtest_for_split(split, K=K)
        if port_cum is not None:
            all_port_cums[split] = port_cum
            all_spy_cums[split] = spy_cum

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Equal-Weight Top-{K} Stocks vs. SPY (Daily Rebalancing)', fontsize=16)

    for idx, split in enumerate(SPLITS):
        ax = axes[idx]
        if split in all_port_cums:
            port_cum = all_port_cums[split]
            spy_cum = all_spy_cums[split]
            ax.plot(port_cum.index, port_cum, label=f'Top-{K} Portfolio', linewidth=2, color='blue')
            if not spy_cum.isna().all():
                ax.plot(spy_cum.index, spy_cum, label='SPY', linewidth=2, color='red', linestyle='--')
            ax.set_title(f'{split.upper()} ({len(port_cum)} days)')
            ax.set_xlabel('Date')
            ax.set_ylabel('Cumulative Return')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"backtest_K{K}_cumulative_returns.png"), dpi=150, bbox_inches='tight')
    plt.show()

    print("\n" + "=" * 70)
    print("🎉 BACKTEST COMPLETED!")
    print(f"📂 Results saved in: {OUTPUT_DIR}")
    print("=" * 70)
