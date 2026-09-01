

import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from empyrical import simple_returns, sharpe_ratio, sortino_ratio, max_drawdown, value_at_risk, conditional_value_at_risk

FLOAT = torch.float32

EPS = 1e-8

def plot_stocks_info(env, num_cols=5, plot_log=True, rolling_window=30):
    """
    Plot stock price performance and rolling Sharpe ratio for stocks in the environment.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    stock_names = env.stock_names
    num_stocks = len(stock_names)
    num_rows = int(np.ceil(num_stocks / num_cols))

    fig, ax = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 3 * num_rows), squeeze=False)
    fig.suptitle("Stocks Performance & Rolling Sharpe", fontsize=14)

    for i, stock in enumerate(stock_names):
        row = i // num_cols
        col = i % num_cols

        prices = env.market.get_stock_history(stock)
        values = np.array(prices)

        # ✅ Skip if no data
        if values is None or len(values) == 0:
            ax[row, col].set_title(f"{stock} (no data)")
            ax[row, col].axis("off")
            continue

        # Normalize values
        values = values / (values[0] + EPS)

        # Plot prices
        ax[row, col].plot(values, label="Price", alpha=0.8)

        # Optional: log returns and rolling sharpe
        if plot_log and len(values) > 1:
            log_ret = np.log(values[1:] / (values[:-1] + EPS))
            if len(log_ret) >= rolling_window:
                rolling_sharpe = (
                    pd.Series(log_ret)
                    .rolling(rolling_window)
                    .apply(lambda x: np.mean(x) / (np.std(x) + EPS))
                )
                ax[row, col].plot(
                    range(1, len(values)),
                    rolling_sharpe,
                    label=f"Rolling Sharpe ({rolling_window}d)",
                    alpha=0.6
                )

        ax[row, col].set_title(stock)
        ax[row, col].legend(fontsize=8)
        ax[row, col].grid(False, alpha=0.3)

    # Hide empty subplots
    for j in range(i + 1, num_rows * num_cols):
        row = j // num_cols
        col = j % num_cols
        ax[row, col].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def prices_to_logreturns(prices):
    # بررسی تعداد ابعاد
    if prices.ndim == 2:
        # فرض: شکل آرایه [window_length, price_features] است
        prices = prices[np.newaxis, :, :]  # تبدیل به [1, window_length, price_features]

    elif prices.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, but got shape {prices.shape}")

    new = prices[:, 1:, :]
    old = prices[:, :-1, :]

    log_rets = np.log(new + EPS) - np.log(old + EPS)

    return log_rets

def prices_to_norm(prices):
    # divide all entries by the closing price of each asset of current day

    closing = prices[:, -1, 3]
    closing = closing[:,None,None]

    norm = np.divide(prices, closing)

    return norm


def prices_to_range(prices):
    minimum = np.min(prices, axis=1)[:,None,:]
    maximum = np.max(prices, axis=1)[:,None,:]

    result = (prices - minimum) / (maximum - minimum + EPS)
    return result
    

def prices_to_simplereturns(prices):
    # shape: [num_stocks, window_length, price_features]
    # output shape: [num_stocks, window_length-1, price_features]

    new = prices[:, 1:, :]
    old = prices[:, :-1, :]

    simple_rets = np.divide(new, old) - 1

    return simple_rets


def remove_not_used(prices, cash=True, volume=True, open=True, high=True, low=True):
    if prices.ndim != 3:
    # Already 2D, return as is
         return prices

    # Feature indices in your prices array (adjust if your array has different order)
    feature_map = {'open': 0, 'high': 1, 'low': 2, 'volume': 3, 'cash': 4}
    dims_to_keep = []

    for feature, idx in feature_map.items():
        if idx < prices.shape[2]:  # make sure the feature exists
            keep = locals()[feature]  # get the bool argument
            if keep:
                dims_to_keep.append(idx)

    # Slice the prices array safely
    prices = prices[:, :, dims_to_keep]
    return prices


def rnn_transpose(prices):
    # use last, after all other transformations
    # used to get correct shape for lstm based networks

    # from [num_stocks, window_length, price_features]
    # to [window_length, num_stocks, price_features]

    prices = np.transpose(prices, (1, 0, 2))
    return prices


def cnn_transpose(prices):
    # use last, after all other transformations
    # used to get correct shape for cnn based networks

    # from [num_stocks, window_length, price_features]
    # to [price_features, num_stocks, window_length]

    prices = np.transpose(prices, (2, 0, 1))
    return prices


def cnn_rnn_transpose(prices):
    # use last, after all other transformations
    # used to get correct shape for cnn based networks

    # from [num_stocks, window_length, price_features]
    # to [price_features, window_length, num_stocks]

    prices = np.transpose(prices, (2, 1, 0))
    return prices
