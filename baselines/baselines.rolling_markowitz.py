"""
rolling_markowitz.py
باسلاین مارکویتز با پنجره غلتان (Rolling Window)
برای مقایسه با مدل MARL در مقاله
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from base_agent import BaseAgent

EPS = 1e-8


class RollingMarkowitzAgent(BaseAgent):
    """
    پیاده‌سازی مارکویتز با پنجره غلتان.
    هر روز، بر اساس داده‌های پنجره‌ی گذشته، پرتفوی بهینه محاسبه می‌شود.
    """

    def __init__(self, name, env, seed, window_length=60, objective="sharpe"):
        """
        Parameters:
            name: نام عامل
            env: محیط بازار
            seed: دانه‌ی تصادفی
            window_length: تعداد روزهای گذشته برای محاسبه‌ی کوواریانس و بازده
            objective: تابع هدف ('sharpe' یا 'variance')
        """
        super().__init__(name, env, seed)
        self.window_length = window_length
        self.objective = objective
        self.results = []  # ذخیره‌سازی نتایج روزانه

    def set_eval(self):
        """برای سازگاری با کلاس BaseAgent."""
        pass

    def act(self, obs, exploration=False):
        """دریافت عمل بهینه."""
        return self.predict_action(obs, exploration=exploration)

    def predict_action(self, obs, exploration=False, verbose=False):
        """
        محاسبه‌ی وزن‌های بهینه بر اساس پنجره‌ی غلتان.
        """
        try:
            # --- استخراج داده‌ها ---
            if isinstance(obs, dict):
                prices = np.array(obs.get("prices"))
                weights = np.array(obs.get("weights")) if "weights" in obs else None
            elif isinstance(obs, tuple):
                prices, weights = obs
                prices = np.array(prices)
                weights = np.array(weights) if weights is not None else None
            else:
                prices = np.array(obs)
                weights = None

            # --- اطمینان از شکل داده ---
            if prices.ndim == 0:
                prices = np.zeros((self.env.action_space.shape[0], 1))
            elif prices.ndim == 1:
                prices = prices.reshape(-1, 1)
            elif prices.ndim == 3:
                prices = prices.reshape(prices.shape[0], -1)
            elif prices.ndim > 2:
                prices = prices.reshape(prices.shape[0], -1)

            if not np.isfinite(prices).all():
                raise ValueError("NaN or Inf in price data")

            # --- وزن‌های فعلی (برای هزینه‌ی تراکنش) ---
            if weights is not None:
                weights = np.ravel(weights)
                if weights.size != self.env.action_space.shape[0]:
                    weights = np.ones(self.env.action_space.shape[0]) / self.env.action_space.shape[0]
            else:
                weights = np.ones(self.env.action_space.shape[0]) / self.env.action_space.shape[0]

            # --- محاسبه‌ی بازده و کوواریانس از پنجره‌ی گذشته ---
            # فرض: prices دارای شکل (n_assets, window_length) است
            # اگر قیمت‌ها بیشتر از یک ویژگی دارند، از ستون Close (معمولاً index 3) استفاده می‌کنیم
            if prices.shape[0] > 1 and prices.shape[1] >= self.window_length:
                # استفاده از آخرین `window_length` روز
                close_prices = prices[:, -self.window_length:]
                returns = np.diff(close_prices, axis=1) / (close_prices[:, :-1] + EPS)
            else:
                # داده‌های کافی نیست، از وزن‌های مساوی استفاده کن
                action = np.ones(self.env.action_space.shape[0]) / self.env.action_space.shape[0]
                return action

            # --- محاسبه‌ی میانگین بازده و کوواریانس ---
            mu = np.mean(returns, axis=1)
            sigma = np.cov(returns)

            # --- بهینه‌سازی ---
            n_assets = len(mu)
            bounds = [(0, 1) for _ in range(n_assets)]  # محدودیت Long-only
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

            # تابع هدف
            if self.objective == "sharpe":
                # بیشینه‌سازی نسبت شارپ = کمینه‌سازی منفی شارپ
                def objective(w):
                    port_return = np.dot(w, mu)
                    port_vol = np.sqrt(np.dot(w, np.dot(sigma, w)) + EPS)
                    # هزینه‌ی تراکنش
                    turnover = np.sum(np.abs(w - weights))
                    transaction_cost = self.env.transaction_cost * turnover
                    return -(port_return / port_vol) + transaction_cost
            else:
                # کمینه‌سازی واریانس (Minimum Variance)
                def objective(w):
                    port_vol = np.sqrt(np.dot(w, np.dot(sigma, w)) + EPS)
                    turnover = np.sum(np.abs(w - weights))
                    transaction_cost = self.env.transaction_cost * turnover
                    return port_vol + transaction_cost

            # حل مسئله
            result = minimize(
                objective,
                weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'ftol': 1e-9, 'maxiter': 1000}
            )

            if result.success:
                action = result.x
            else:
                raise ValueError(f"Optimization failed: {result.message}")

            # --- نرمال‌سازی نهایی ---
            if np.sum(action) > 0:
                action = action / np.sum(action)
            else:
                action = np.zeros_like(action)
                action[0] = 1.0  # همه نقدینگی

        except Exception as e:
            print(f"[{self.name} ERROR] {e}")
            action = self._fallback_portfolio(weights)

        # --- ذخیره‌سازی نتایج (برای مقاله) ---
        if hasattr(self.env, 'market'):
            try:
                date = self.env.market.step_to_date()
            except:
                date = "Unknown"
        else:
            date = "Unknown"

        self.results.append({
            'date': date,
            'action': action.copy(),
            'objective': self.objective,
            'window_length': self.window_length
        })

        return action

    def _fallback_portfolio(self, weights):
        """پورتفوی امن در صورت شکست بهینه‌سازی."""
        if weights is not None and np.sum(weights) > 0:
            return np.array(weights, dtype=np.float32)
        else:
            safe = np.zeros(self.env.action_space.shape[0], dtype=np.float32)
            safe[0] = 1.0
            return safe

    def eval(self, env, exploration=False, render=False):
        """ارزیابی عامل روی محیط."""
        obs = env.reset()
        done = False
        rewards = []
        infos = []
        port_values = []

        while not done:
            action = self.act(obs)
            next_obs, reward, done, info = env.step(action)

            # تکمیل اطلاعات
            if "date" not in info and hasattr(env, "market"):
                step_idx = len(infos)
                if step_idx < len(env.market.date_list):
                    info["date"] = env.market.date_list[step_idx]
            if "weights_new" not in info:
                info["weights_new"] = action
            if "port_value_old" not in info:
                info["port_value_old"] = info.get("port_value_new", env.init_port_value)

            infos.append(info)
            rewards.append(reward)
            port_values.append(info.get("port_value_new", env.init_port_value))
            obs = next_obs

        final_value = port_values[-1] if port_values else env.init_port_value
        return sum(rewards), infos, final_value

    def save_results(self, output_dir="baseline_results"):
        """ذخیره‌سازی نتایج به‌صورت CSV برای مقاله."""
        os.makedirs(output_dir, exist_ok=True)
        if not self.results:
            print("⚠️ No results to save.")
            return

        # تبدیل به DataFrame
        df = pd.DataFrame(self.results)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # جدا کردن وزن‌ها به ستون‌های جداگانه
        weights_df = pd.DataFrame(df['action'].tolist(), index=df.index)
        weights_df.columns = [f"w_{i}" for i in range(weights_df.shape[1])]

        # ذخیره
        path = os.path.join(output_dir, f"{self.name}_weights.csv")
        weights_df.to_csv(path)
        print(f"✅ Results saved to: {path}")

        return weights_df
