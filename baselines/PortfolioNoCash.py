
"""
PortfolioNoCash.py
نسخه‌ی بدون نقدینگی (Cash حذف شده) - اصلاح‌شده
"""

import numpy as np
import pandas as pd

class PortfolioNoCash:
    def __init__(self, config, stock_names, sector_map, returns_df, features_df, vix_series, year):
        self.config = config
        self.window_length = getattr(config, 'window_length', 21)
        self.transaction_cost = getattr(config, 'transaction_cost', 0.001)
        self.risk_free_rate = getattr(config, 'risk_free_rate', 0.03)
        self.init_port_value = getattr(config, 'init_portfolio_value', 1e6)
        self.year = year
        
        self.stock_names = stock_names
        self.sector_map = sector_map
        self.num_stocks = len(stock_names)
        self.num_sectors = len(sector_map)
        self.sector_names = list(sector_map.keys())
        self.sector_indices = self._build_sector_indices()
        
        self.features_df = features_df
        self.vix_series = vix_series
        
        self.dates = sorted([d for d in returns_df.index if d.year == year])
        if not self.dates:
            raise ValueError(f"No trading days found for year {year}")
        self.returns_df = returns_df.loc[self.dates]
        
        self.current_idx = 0
        self.current_date = self.dates[0] if self.dates else None
        
        self.weights = np.array([1.0 / self.num_stocks] * self.num_stocks)
        self.port_value = self.init_port_value
        self.peak_value = self.init_port_value
        self.returns_history = []
        self.portfolio_values = [self.init_port_value]
        
        self.sector_returns_history = {s: [] for s in self.sector_names}
        self.sector_weights_history = {s: [] for s in self.sector_names}
        
        self.daily_returns = []
        self.daily_rewards = []
        self.daily_turnovers = []
        
        self.reset()
    
    def _build_sector_indices(self):
        sector_indices = {}
        for sector_name, tickers in self.sector_map.items():
            indices = []
            for ticker in tickers:
                if ticker in self.stock_names:
                    idx = self.stock_names.index(ticker)
                    indices.append(idx)
            sector_indices[sector_name] = indices
        return sector_indices
    
    def reset(self):
        self.current_idx = 0
        self.current_date = self.dates[0] if self.dates else None
        self.weights = np.array([1.0 / self.num_stocks] * self.num_stocks)
        self.port_value = self.init_port_value
        self.peak_value = self.init_port_value
        self.returns_history = []
        self.portfolio_values = [self.init_port_value]
        self.daily_returns = []
        self.daily_rewards = []
        self.daily_turnovers = []
        self.sector_returns_history = {s: [] for s in self.sector_names}
        self.sector_weights_history = {s: [] for s in self.sector_names}
        return self._get_obs()
    
    def _get_obs(self):
        prices = np.zeros((self.num_stocks, self.window_length, 5))
        return prices, self.weights.copy()
    
    def step(self, flat_action):
        # Parse action بدون نقدینگی
        idx = 0
        alphas = flat_action[idx:idx + self.num_sectors]
        idx += self.num_sectors
        if alphas.sum() > 0:
            alphas = alphas / alphas.sum()
        else:
            alphas = np.ones(self.num_sectors) / self.num_sectors
        
        sector_weights = {}
        for sector_name in self.sector_names:
            n = len(self.sector_map[sector_name])
            sw = flat_action[idx:idx + n]
            idx += n
            if sw.sum() > 0:
                sw = sw / sw.sum()
            else:
                sw = np.ones(n) / n
            sector_weights[sector_name] = sw
        
        # محاسبه وزن‌های نهایی (بدون نقدینگی)
        final_weights = np.zeros(self.num_stocks)
        for i, sector_name in enumerate(self.sector_names):
            tickers = self.sector_map[sector_name]
            for j, ticker in enumerate(tickers):
                if ticker in self.stock_names:
                    stock_idx = self.stock_names.index(ticker)
                    final_weights[stock_idx] = alphas[i] * sector_weights[sector_name][j]
        
        total = final_weights.sum()
        if total > 0:
            final_weights = final_weights / total
        else:
            final_weights = np.ones(self.num_stocks) / self.num_stocks
        
        # ✅ بازده امروز را به آرایه NumPy تبدیل کنید
        returns_today = np.array(self._get_returns_for_date(self.current_date))
        port_return = np.dot(final_weights, returns_today)
        
        turnover = np.sum(np.abs(final_weights - self.weights))
        cost = self.transaction_cost * turnover * self.port_value
        remaining_value = self.port_value - cost
        
        old_value = self.port_value
        self.port_value = remaining_value * (1 + port_return)
        
        if self.port_value > self.peak_value:
            self.peak_value = self.port_value
        
        self.returns_history.append(port_return)
        if len(self.returns_history) > 21:
            self.returns_history.pop(0)
        self.portfolio_values.append(self.port_value)
        self.daily_returns.append(port_return)
        self.daily_turnovers.append(turnover)
        
        # ✅ اصلاح بخش محاسبه بازده بخش‌ها
        for sector_name in self.sector_names:
            idxs = self.sector_indices[sector_name]
            if idxs:
                # idxs یک لیست است، اما returns_today یک آرایه NumPy است و از لیست به‌عنوان اندیس پشتیبانی می‌کند
                sector_ret = np.dot(final_weights[idxs], returns_today[idxs])
                self.sector_returns_history[sector_name].append(sector_ret)
                if len(self.sector_returns_history[sector_name]) > 21:
                    self.sector_returns_history[sector_name].pop(0)
                self.sector_weights_history[sector_name].append(final_weights[idxs].copy())
        
        self.weights = final_weights
        
        rewards = self._compute_rewards(port_return, turnover)
        self.daily_rewards.append(rewards['global_reward'])
        
        self.current_idx += 1
        done = (self.current_idx >= len(self.dates))
        if not done:
            self.current_date = self.dates[self.current_idx]
        
        next_obs = self._get_obs()
        info = {
            'port_value_old': old_value,
            'port_value_new': self.port_value,
            'weights_new': final_weights,
            'turnover': turnover,
            'cost': cost,
            'log_return': np.log(1 + port_return) if port_return > -1 else port_return,
            'simple_return': port_return,
            'sector_rewards': rewards['sector_rewards'],
            'date': self.current_date
        }
        return next_obs, rewards['global_reward'], done, info
    
    def _get_returns_for_date(self, date):
        if date in self.returns_df.index:
            row = self.returns_df.loc[date]
            return [row.get(ticker, 0.0) for ticker in self.stock_names]
        return [0.0] * self.num_stocks
    
    def _compute_rewards(self, port_return, turnover):
        if len(self.returns_history) >= 21:
            volatility = np.std(self.returns_history)
        else:
            volatility = 0.01
        sharpe = port_return / (volatility + 1e-6)
        drawdown = (self.peak_value - self.port_value) / (self.peak_value + 1e-6)
        lambda_turnover = getattr(self.config, 'lambda_turnover', 0.5)
        lambda_drawdown = getattr(self.config, 'lambda_drawdown', 0.3)
        global_reward = sharpe - lambda_turnover * turnover - lambda_drawdown * drawdown
        
        sector_rewards = {}
        for sector_name in self.sector_names:
            history = self.sector_returns_history.get(sector_name, [])
            if len(history) >= 5:
                sector_vol = np.std(history[-5:]) if len(history) >= 5 else 0.01
                sector_sharpe = np.mean(history[-5:]) / (sector_vol + 1e-6)
            else:
                sector_sharpe = 0.0
            w_history = self.sector_weights_history.get(sector_name, [])
            if len(w_history) >= 2:
                sector_turnover = np.sum(np.abs(w_history[-1] - w_history[-2]))
            else:
                sector_turnover = 0.0
            sector_rewards[sector_name] = sector_sharpe - 0.3 * sector_turnover
        
        return {
            'global_reward': global_reward,
            'sector_rewards': sector_rewards,
            'sharpe_ratio': sharpe,
            'drawdown': drawdown
        }
    
    def get_stats(self):
        if len(self.daily_returns) == 0:
            return {}
        daily_ret = np.array(self.daily_returns)
        daily_reward = np.array(self.daily_rewards)
        cum_return = self.port_value / self.init_port_value - 1
        sharpe = np.mean(daily_ret) / (np.std(daily_ret) + 1e-6) * np.sqrt(252)
        portfolio_values = np.array(self.portfolio_values)
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / (peak + 1e-6)
        max_drawdown = np.max(drawdown)
        total_turnover = np.sum(self.daily_turnovers)
        avg_turnover = np.mean(self.daily_turnovers)
        avg_reward = np.mean(daily_reward)
        std_reward = np.std(daily_reward)
        return {
            'year': self.year,
            'num_days': len(self.daily_returns),
            'initial_value': self.init_port_value,
            'final_value': self.port_value,
            'cumulative_return': cum_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'total_turnover': total_turnover,
            'avg_turnover': avg_turnover,
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'volatility': np.std(daily_ret),
            'mean_return': np.mean(daily_ret)
        }
