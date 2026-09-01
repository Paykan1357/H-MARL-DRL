"""
baseline_wrapper.py
Wrapper برای محیط Portfolio تا با باسلاین‌های کلاسیک سازگار شود.
"""

import numpy as np


class BaselineWrapper:
    def __init__(self, env):
        self.env = env
        self.stock_names = env.stock_names
        self.sector_map = env.sector_map
        self.num_stocks = len(self.stock_names)
        self.num_sectors = len(self.sector_map)
        self.sector_names = list(self.sector_map.keys())
        self.init_port_value = env.init_port_value
        self.transaction_cost = env.transaction_cost
        self.current_date = None

        self.action_space = type('ActionSpace', (), {
            'shape': (self.num_stocks,),
            'sample': lambda: np.ones(self.num_stocks) / self.num_stocks
        })()

        self.market = type('Market', (), {
            'step_to_date': self.step_to_date
        })()

    def reset(self):
        obs = self.env.reset()
        self.current_date = self.env.current_date
        return self._convert_obs(obs)

    def step(self, action_simple):
        hierarchical_action = self._simple_to_hierarchical(action_simple)
        obs, reward, done, info = self.env.step(hierarchical_action)
        self.current_date = self.env.current_date
        return self._convert_obs(obs), reward, done, info

    def _simple_to_hierarchical(self, simple_weights):
        cash_weight = 0.0
        alphas = []
        sector_weights_flat = []

        for sector_name in self.sector_names:
            tickers = self.sector_map[sector_name]
            alpha = 0.0

            for ticker in tickers:
                if ticker in self.stock_names:
                    idx = self.stock_names.index(ticker)
                    w = simple_weights[idx]
                    alpha += w
                    sector_weights_flat.append(w)

            alphas.append(alpha)

        total_alpha = sum(alphas)
        if total_alpha > 1e-8:
            alphas = [a / total_alpha for a in alphas]
        else:
            alphas = [1.0 / self.num_sectors] * self.num_sectors

        flat_action = np.concatenate([
            [cash_weight],
            alphas,
            sector_weights_flat
        ])

        return flat_action

    def _convert_obs(self, obs):
        if isinstance(obs, tuple) and len(obs) == 2:
            prices, weights = obs
            if prices.shape[0] == self.num_stocks + 1:
                prices = prices[1:]
            return prices
        return obs

    def step_to_date(self):
        if self.current_date is not None:
            if hasattr(self.current_date, 'strftime'):
                return self.current_date.strftime('%Y-%m-%d')
            return str(self.current_date)
        return "2023-01-01"

    def __getattr__(self, name):
        return getattr(self.env, name)
