from copy import deepcopy
import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
from torch.utils.data import Dataset


class Market:
    def __init__(self, start_date, end_date, window_length=30, stock_names=None):
        """
        Initialize Market environment.
        Downloads historical data, aligns by date, and prepares tensor data.
        """
        self.start_date = start_date
        self.end_date = end_date
        self.window_length = window_length
        self.stock_names = stock_names or []

        _date_fmt = "%Y-%m-%d"
        start_dt = dt.datetime.strptime(self.start_date, _date_fmt)
        end_dt = dt.datetime.strptime(self.end_date, _date_fmt)

        self.data = {}
        valid_stocks = []

        # --- Download data for each asset ---
        for stock in self.stock_names:
            df = yf.download(stock, start=start_dt, end=end_dt, progress=False)
            if df.empty or len(df) < self.window_length + 1:
                print(f"[Warning] Skipping '{stock}' — insufficient data.")
                continue

            df = df.drop(columns=["Adj Close"], errors="ignore")
            df = df.asfreq("D")  # ensure daily calendar
            df = df.fillna(method="ffill").fillna(method="bfill")
            self.data[stock] = df
            valid_stocks.append(stock)

        if not self.data:
            raise ValueError("No valid stock data found. Check symbols or date range.")

        # --- Align all assets on a common date index ---
        common_index = pd.date_range(start=start_dt, end=end_dt, freq="D")
        aligned_data = []
        for stock in valid_stocks:
            df = self.data[stock].reindex(common_index, method="ffill").fillna(method="bfill")
            aligned_data.append(df.to_numpy())

        self.data = np.stack(aligned_data, axis=0)  # shape: (assets, time, features)

        # --- Add cash asset (constant 1s) ---
        _cash_data = np.ones((1, self.data.shape[1], self.data.shape[2]))
        self.data = np.concatenate((_cash_data, self.data), axis=0)
        self.stock_names = ["CASH"] + valid_stocks

        # --- Load benchmark (SPY or fallback) ---
        try:
            snp_df = yf.download("SPY", start=start_dt, end=end_dt, progress=False)
            if snp_df.empty:
                raise Exception
            snp_df = snp_df.drop(columns=["Adj Close"], errors="ignore")
            snp_df = snp_df.reindex(common_index, method="ffill").fillna(method="bfill")
            print("[Info] Using SPY as benchmark.")
        except Exception:
            print("[Info] SPY not available. Using BTC-USD as proxy.")
            snp_df = self.data[1]  # fallback proxy

        self.snp = snp_df if isinstance(snp_df, np.ndarray) else snp_df.to_numpy()
        self.date_list = [str(d.date()) for d in common_index]

        self.tot_steps = max(0, len(self.date_list) - self.window_length)
        if self.tot_steps == 0:
            raise ValueError("Date range too short or window_length too large.")

        self.num_stocks = len(self.stock_names) - 1  # exclude CASH
        self.price_features = self.data.shape[-1]

        self.reset()

    # ====================================================
    #                  Core Market Functions
    # ====================================================
    def get_observation(self, step):
        """Return a sliding window of market data."""
        data_window = self.data[:, step:step + self.window_length, :]
        if data_window.shape[1] < self.window_length:
            padding = np.zeros((data_window.shape[0],
                                self.window_length - data_window.shape[1],
                                data_window.shape[2]))
            data_window = np.concatenate([data_window, padding], axis=1)
        return deepcopy(data_window)

    def step(self):
        """Move one step forward in time."""
        if self.current_step >= self.tot_steps - 1:
            done = True
            curr_obs = self.get_observation(self.current_step)
            next_obs = self.get_observation(self.current_step)
        else:
            done = False
            curr_obs = self.get_observation(self.current_step)
            self.current_step += 1
            next_obs = self.get_observation(self.current_step)
        self.next_step = self.current_step + 1
        return curr_obs, next_obs, done

    def reset(self):
        """Reset step counter."""
        self.current_step = 0
        self.next_step = 1
        return self.get_observation(self.current_step)

    def step_to_date(self, step=None):
        """Return the date corresponding to a step index."""
        if step is None:
            step = self.next_step - 1
        return self.date_list[step + self.window_length - 1]

    def get_stock_history(self, stock_name):
        """Return closing price history of a given stock."""
        if stock_name == "CASH":
            return np.ones(self.data.shape[1])
        if stock_name not in self.stock_names:
            raise ValueError(f"Stock '{stock_name}' not found in market data.")
        idx = self.stock_names.index(stock_name)
        return self.data[idx, :, 3]  # column 3 = 'Close' in OHLCV

    # ====================================================
    #            Compute Log Returns (for forecasting)
    # ====================================================
    def compute_log_returns(self):
        """
        Compute log returns of closing prices for all assets except CASH.
        Always returns array of shape (time, num_assets).
        """
        # Get closing prices (exclude CASH)
        close_prices = self.data[1:, :, 3]  # shape: (num_assets, time)

        if close_prices.shape[0] == 1:
            # Single asset case
            log_returns = np.diff(np.log(close_prices + 1e-8), axis=1)
            log_returns = np.concatenate([np.zeros((1, 1)), log_returns], axis=1)
        else:
            # Multi-asset case
            log_returns = np.diff(np.log(close_prices + 1e-8), axis=1)
            zero_pad = np.zeros((close_prices.shape[0], 1))
            log_returns = np.concatenate([zero_pad, log_returns], axis=1)

        return log_returns.T  # shape: (time, num_assets)

    # ====================================================
    #                    Torch Dataset
    # ====================================================
    def get_dataset(self):
        """Return a PyTorch Dataset wrapper."""
        return MarketDataset(self)


class MarketDataset(Dataset):
    def __init__(self, market):
        self.data = np.transpose(market.data, (1, 0, 2))  # (time, stocks, features)
        self.window_length = market.window_length
        self.num_steps = max(0, self.data.shape[0] - self.window_length)

    def __len__(self):
        return self.num_steps

    def __getitem__(self, idx):
        obs = self.data[idx:idx + self.window_length]
        truth = self.data[idx + 1:idx + self.window_length + 1]
        return obs.astype(np.float32), truth.astype(np.float32)
