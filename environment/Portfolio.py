
"""
Portfolio.py
محیط معاملاتی با پشتیبانی از اپیزودهای سالانه و تنظیمات پاداش از config
"""

import numpy as np
import pandas as pd

class Portfolio:
    def __init__(self, config, stock_names, sector_map, returns_df, features_df, vix_series, year):
        """
        پارامترها:
            config: تنظیمات (window_length, transaction_cost, lambda_drawdown, ...)
            stock_names: لیست سهام این سال (سبد انتخابی)
            sector_map: دیکشنری {sector_name: [ticker1, ticker2, ...]}
            returns_df: دیتافریم بازده روزانه
            features_df: دیتافریم ویژگی‌ها
            vix_series: سری زمانی VIX
            year: سال جاری (برای فیلتر کردن تاریخ‌ها)
        """
        # === تنظیمات (ذخیره config برای دسترسی به ضرایب پاداش) ===
        self.config = config
        self.window_length = getattr(config, 'window_length', 21)
        self.transaction_cost = getattr(config, 'transaction_cost', 0.001)
        self.risk_free_rate = getattr(config, 'risk_free_rate', 0.03)
        self.init_port_value = getattr(config, 'init_portfolio_value', 1e6)
        self.year = year
        
        # === سبد و بخش‌ها ===
        self.stock_names = stock_names
        self.sector_map = sector_map
        self.num_stocks = len(stock_names)
        self.num_sectors = len(sector_map)
        self.sector_names = list(sector_map.keys())
        self.sector_indices = self._build_sector_indices()
        
        # === داده‌های بازار (فیلتر شده بر اساس سال) ===
        self.features_df = features_df
        self.vix_series = vix_series
        
        # فیلتر کردن تاریخ‌ها بر اساس سال
        self.dates = sorted([d for d in returns_df.index if d.year == year])
        if not self.dates:
            raise ValueError(f"No trading days found for year {year}")
        
        # فیلتر کردن returns_df برای سال جاری
        self.returns_df = returns_df.loc[self.dates]
        
        self.current_idx = 0
        self.current_date = self.dates[0] if self.dates else None
        
        # === وزن‌ها و ارزش سبد ===
        self.weights = np.array([1.0] + [0.0] * self.num_stocks)
        self.port_value = self.init_port_value
        self.peak_value = self.init_port_value
        self.returns_history = []      # برای محاسبه‌ی نوسان
        self.portfolio_values = [self.init_port_value]
        
        # === تاریخچه‌ی بخش‌ها ===
        self.sector_returns_history = {s: [] for s in self.sector_names}
        self.sector_weights_history = {s: [] for s in self.sector_names}
        
        # === آمار روزانه ===
        self.daily_returns = []
        self.daily_rewards = []
        self.daily_turnovers = []
        
        self.reset()
    
    def _build_sector_indices(self):
        """ساخت نگاشت از نام بخش به اندیس‌های سهام در بردار وزن"""
        sector_indices = {}
        for sector_name, tickers in self.sector_map.items():
            indices = []
            for ticker in tickers:
                if ticker in self.stock_names:
                    idx = self.stock_names.index(ticker) + 1
                    indices.append(idx)
            sector_indices[sector_name] = indices
        return sector_indices
    
    def reset(self):
        """بازنشانی محیط به ابتدای سال"""
        self.current_idx = 0
        self.current_date = self.dates[0] if self.dates else None
        self.weights = np.array([1.0] + [0.0] * self.num_stocks)
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
        """ساخت وضعیت خام (برای مرحله‌ی بعد)"""
        prices = np.zeros((self.num_stocks + 1, self.window_length, 5))
        return prices, self.weights.copy()
    
    def step(self, flat_action):
        """اجرای یک روز معاملاتی"""
        # === 1. Parse Action ===
        action_parsed = self._parse_action(flat_action)
        cash_weight = action_parsed['cash']
        alphas = action_parsed['alphas']
        sector_weights = action_parsed['sector_weights']
        
        # === 2. محاسبه‌ی وزن‌های نهایی ===
        new_weights = self._compute_final_weights(cash_weight, alphas, sector_weights)
        
        # === 3. محاسبه‌ی بازده امروز ===
        returns_today = self._get_returns_for_date(self.current_date)
        cash_return = (1 + self.risk_free_rate) ** (1/252) - 1
        returns_all = np.array([cash_return] + list(returns_today))
        
        port_return = np.dot(new_weights, returns_all)
        
        # === 4. هزینه‌ی تراکنش ===
        turnover = np.sum(np.abs(new_weights - self.weights))
        cost = self.transaction_cost * turnover * self.port_value
        remaining_value = self.port_value - cost
        
        # === 5. به‌روزرسانی ارزش سبد ===
        old_value = self.port_value
        self.port_value = remaining_value * (1 + port_return)
        
        # === 6. به‌روزرسانی اوج ارزش ===
        if self.port_value > self.peak_value:
            self.peak_value = self.port_value
        
        # === 7. ذخیره‌سازی تاریخچه ===
        self.returns_history.append(port_return)
        if len(self.returns_history) > 21:
            self.returns_history.pop(0)
        
        self.portfolio_values.append(self.port_value)
        self.daily_returns.append(port_return)
        self.daily_turnovers.append(turnover)
        
        # === 8. ذخیره‌سازی تاریخچه‌ی بخش‌ها ===
        for sector_name in self.sector_names:
            idxs = self.sector_indices[sector_name]
            if idxs:
                sector_ret = np.dot(new_weights[idxs], returns_all[idxs])
                self.sector_returns_history[sector_name].append(sector_ret)
                if len(self.sector_returns_history[sector_name]) > 21:
                    self.sector_returns_history[sector_name].pop(0)
                self.sector_weights_history[sector_name].append(new_weights[idxs].copy())
        
        # === 9. به‌روزرسانی وزن‌ها ===
        self.weights = new_weights
        
        # === 10. محاسبه‌ی پاداش‌ها ===
        rewards = self._compute_rewards(port_return, turnover)
        self.daily_rewards.append(rewards['global_reward'])
        
        # === 11. رفتن به روز بعد ===
        self.current_idx += 1
        done = (self.current_idx >= len(self.dates))
        if not done:
            self.current_date = self.dates[self.current_idx]
        
        # === 12. وضعیت بعدی ===
        next_obs = self._get_obs()
        
        # === 13. اطلاعات اضافی ===
        info = {
            'port_value_old': old_value,
            'port_value_new': self.port_value,
            'weights_new': new_weights,
            'turnover': turnover,
            'cost': cost,
            'log_return': np.log(1 + port_return) if port_return > -1 else port_return,
            'simple_return': port_return,
            'sector_rewards': rewards['sector_rewards'],
            'date': self.current_date,
            'vix': self._get_vix(),
            'portfolio_return': port_return,
            'sharpe_ratio': rewards.get('sharpe_ratio', 0.0),
            'drawdown': rewards.get('drawdown', 0.0)
        }
        
        return next_obs, rewards['global_reward'], done, info
    
    def _parse_action(self, flat_action):
        """تفسیر بردار عمل تخت"""
        idx = 0
        cash_weight = flat_action[idx]
        idx += 1
        
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
        
        return {'cash': cash_weight, 'alphas': alphas, 'sector_weights': sector_weights}
    
    def _compute_final_weights(self, cash_weight, alphas, sector_weights):
        """محاسبه‌ی وزن‌های نهایی"""
        final_weights = np.zeros(self.num_stocks + 1)
        final_weights[0] = cash_weight
        
        for i, sector_name in enumerate(self.sector_names):
            tickers = self.sector_map[sector_name]
            for j, ticker in enumerate(tickers):
                if ticker in self.stock_names:
                    stock_idx = self.stock_names.index(ticker) + 1
                    final_weights[stock_idx] = alphas[i] * sector_weights[sector_name][j]
        
        total = final_weights.sum()
        if total > 0:
            final_weights = final_weights / total
        else:
            final_weights[0] = 1.0
        
        return final_weights
    
    def _get_returns_for_date(self, date):
        """دریافت بازده سهام در یک تاریخ مشخص"""
        if date in self.returns_df.index:
            row = self.returns_df.loc[date]
            return [row.get(ticker, 0.0) for ticker in self.stock_names]
        return [0.0] * self.num_stocks
    
    def _get_vix(self):
        """دریافت VIX در تاریخ جاری"""
        if self.current_date in self.vix_series.index:
            return float(self.vix_series.loc[self.current_date])
        return 20.0
    
    def _compute_rewards(self, port_return, turnover):
        """
        محاسبه‌ی پاداش‌های سراسری و بخشی
        ضرایب از config خوانده می‌شوند
        """
        # === پاداش سراسری (برای MAC) ===
        if len(self.returns_history) >= 21:
            volatility = np.std(self.returns_history)
        else:
            volatility = 0.01
        
        sharpe = port_return / (volatility + 1e-6)
        drawdown = (self.peak_value - self.port_value) / (self.peak_value + 1e-6)
        
        # 🔽 خواندن ضرایب از config (با fallback به مقادیر پیش‌فرض) 🔽
        lambda_turnover = getattr(self.config, 'lambda_turnover', 0.5)
        lambda_drawdown = getattr(self.config, 'lambda_drawdown', 0.3)
        
        global_reward = sharpe - lambda_turnover * turnover - lambda_drawdown * drawdown
        
        # === پاداش بخشی (برای PPOها) ===
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
            
            # جریمه گردش مالی بخشی (کمتر از جریمه سراسری)
            sector_rewards[sector_name] = sector_sharpe - 0.3 * sector_turnover
        
        return {
            'global_reward': global_reward,
            'sector_rewards': sector_rewards,
            'sharpe_ratio': sharpe,
            'drawdown': drawdown
        }
    
    # ========================================================================
    # متدهای گزارش‌گیری آماری
    # ========================================================================
    
    def get_stats(self):
        """گزارش کامل آماری پایان سال"""
        if len(self.daily_returns) == 0:
            return {}
        
        daily_ret = np.array(self.daily_returns)
        daily_reward = np.array(self.daily_rewards)
        
        # بازده کل
        cum_return = self.port_value / self.init_port_value - 1
        
        # نسبت شارپ سالانه
        sharpe = np.mean(daily_ret) / (np.std(daily_ret) + 1e-6) * np.sqrt(252)
        
        # حداکثر افت
        portfolio_values = np.array(self.portfolio_values)
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / (peak + 1e-6)
        max_drawdown = np.max(drawdown)
        
        # گردش مالی کل
        total_turnover = np.sum(self.daily_turnovers)
        avg_turnover = np.mean(self.daily_turnovers)
        
        # میانگین پاداش
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
