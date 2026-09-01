

import os
import sys
import numpy as np
import pandas as pd
import torch
import time
from config_ablation import ConfigAblation
from Portfolio import Portfolio
from PortfolioSingleAgent import PortfolioSingleAgent
from sector_ppo_agent import SectorPPOAgent
from mac_sac_agent import MACSACAgent
from utils import load_data, build_sector_state, build_mac_state, get_sector_map_from_file, get_sector_map_from_basket

# ============================================================================
# 1. توابع محاسبه معیارهای کامل (کپی از test.py)
# ============================================================================

def calculate_risk_metrics(returns, confidence_level=0.95):
    """
    محاسبه معیارهای ریسک از سری بازده روزانه
    """
    if len(returns) < 2:
        return {
            'VaR_95': np.nan,
            'CVaR_95': np.nan,
            'Sortino': np.nan,
            'Calmar': np.nan,
            'Max_Drawdown': np.nan,
            'Sharpe': np.nan,
            'Mean_Log_Return': np.nan,
            'Std_Log_Return': np.nan,
            'Cumulative_Return': np.nan,
            'Volatility': np.nan
        }
    
    returns = pd.Series(returns)
    log_returns = np.log(1 + returns)
    
    mean_ret = returns.mean()
    std_ret = returns.std()
    risk_free_daily = 0.03 / 252
    excess_returns = returns - risk_free_daily
    
    # Sharpe
    sharpe = (excess_returns.mean() / (std_ret + 1e-8)) * np.sqrt(252)
    
    # Sortino
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) > 0:
        sortino = (excess_returns.mean() / (downside_returns.std() + 1e-8)) * np.sqrt(252)
    else:
        sortino = np.inf
    
    # حداکثر افت
    cum_values = (1 + returns).cumprod()
    peak = cum_values.expanding().max()
    drawdown = (cum_values - peak) / peak
    max_drawdown = drawdown.min()
    
    # کلمار
    cum_return = (1 + returns).prod() - 1
    calmar = cum_return / (abs(max_drawdown) + 1e-8)
    
    # VaR و CVaR
    var_95 = np.percentile(returns, (1 - confidence_level) * 100)
    cvar_95 = returns[returns <= var_95].mean() if np.any(returns <= var_95) else var_95
    
    return {
        'VaR_95': var_95,
        'CVaR_95': cvar_95,
        'Sortino': sortino,
        'Calmar': calmar,
        'Max_Drawdown': max_drawdown,
        'Sharpe': sharpe,
        'Mean_Log_Return': log_returns.mean(),
        'Std_Log_Return': log_returns.std(),
        'Cumulative_Return': cum_return,
        'Volatility': std_ret * np.sqrt(252)
    }


def calculate_concentration_metrics(weights_df):
    """
    محاسبه معیارهای تمرکز سبد از وزن سهام
    weights_df: دیتافریم با ستون‌های سهام و تاریخ (ستون 'date' موجود است)
    بازگرداندن: میانگین بیشینه وزن، HHI، ENB
    """
    if weights_df is None or weights_df.empty:
        return {'Avg_Max_Weight': np.nan, 'Avg_HHI': np.nan, 'Avg_ENB': np.nan}
    
    # حذف ستون تاریخ
    df = weights_df.drop(columns=['date'], errors='ignore')
    
    # بیشینه وزن در هر روز
    max_weights = df.max(axis=1)
    
    # HHI (مجموع مجذور وزن‌ها)
    hhi = (df ** 2).sum(axis=1)
    
    # ENB (تعداد مؤثر شرط‌ها)
    enb = 1 / hhi
    
    return {
        'Avg_Max_Weight': max_weights.mean(),
        'Avg_HHI': hhi.mean(),
        'Avg_ENB': enb.mean()
    }

# ============================================================================
# 2. تعریف کلاس SinglePPOAgent (کپی از train_ablation_single_agent.py)
# ============================================================================

class SinglePPOAgent:
    """یک عامل PPO واحد برای کل سبد (بدون MAC)"""
    def __init__(self, state_dim, action_dim, args=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = getattr(args, "ppo_gamma", 0.995)
        self.gae_lambda = getattr(args, "ppo_gae_lambda", 0.95)
        self.epsilon_clip = getattr(args, "ppo_epsilon_clip", 0.2)
        self.entropy_coef = getattr(args, "ppo_entropy_coef", 0.005)
        self.value_loss_coef = getattr(args, "ppo_value_loss_coef", 0.5)
        self.max_grad_norm = getattr(args, "ppo_max_grad_norm", 0.5)
        self.ppo_epochs = getattr(args, "ppo_epochs", 10)
        self.batch_size = getattr(args, "ppo_batch_size", 64)
        self.lr_actor = getattr(args, "ppo_lr_actor", 3e-4)
        self.lr_critic = getattr(args, "ppo_lr_critic", 1e-3)

        self.actor = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, action_dim),
            torch.nn.Softmax(dim=-1)
        ).to(self.device)
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.checkpoint_dir = None

    def get_action(self, state, deterministic=False):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            probs = self.actor(state)
            if deterministic:
                action = probs.argmax(dim=-1)
                log_prob = torch.log(probs.gather(1, action.unsqueeze(1)) + 1e-8).squeeze()
            else:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            value = self.critic(state)
        self.states.append(state.cpu().numpy().flatten())
        self.actions.append(action.item())
        self.log_probs.append(log_prob.item())
        self.values.append(value.item())
        return probs.squeeze().cpu().numpy(), action.item()

    def store_reward(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)

    def update(self):
        if len(self.states) < self.batch_size:
            return None
        states = torch.tensor(np.array(self.states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(self.actions), dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(np.array(self.log_probs), dtype=torch.float32, device=self.device)
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        with torch.no_grad():
            last_state = torch.tensor(self.states[-1], dtype=torch.float32, device=self.device).unsqueeze(0)
            last_value = self.critic(last_state).item()
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value * (1 - dones[t])
            else:
                next_value = values[t + 1] * (1 - dones[t])
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(len(states))
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                probs = self.actor(batch_states)
                dist = torch.distributions.Categorical(probs)
                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                values = self.critic(batch_states).squeeze(-1)
                critic_loss = torch.nn.functional.mse_loss(values, batch_returns)
                total_loss = actor_loss + self.value_loss_coef * critic_loss
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        return {"actor_loss": 0, "critic_loss": 0, "entropy": 0}

    def save(self, episode):
        if self.checkpoint_dir:
            torch.save(self.actor.state_dict(), f"{self.checkpoint_dir}/actor_ep{episode}.pth")
            torch.save(self.critic.state_dict(), f"{self.checkpoint_dir}/critic_ep{episode}.pth")

    def load(self, episode):
        if self.checkpoint_dir:
            actor_path = f"{self.checkpoint_dir}/actor_ep{episode}.pth"
            critic_path = f"{self.checkpoint_dir}/critic_ep{episode}.pth"
            if os.path.exists(actor_path):
                self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
                self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
                return True
        return False

# ============================================================================
# 3. تعریف تابع build_sector_state_no_node2vec (بدون Node2Vec)
# ============================================================================

def build_sector_state_no_node2vec(date, sector_tickers, features_df, vix_series, args):
    """ساخت State بدون Node2Vec (فقط داده‌های بازار + VIX)"""
    market_data = []
    for ticker in sector_tickers:
        row = features_df[(features_df['Date'] == date) & (features_df['Ticker'] == ticker)]
        if not row.empty:
            market_data.extend([
                float(row['Momentum_scaled'].iloc[0]),
                float(row['Volatility_scaled'].iloc[0]),
                float(row['NormVolume_scaled'].iloc[0])
            ])
        else:
            market_data.extend([0.0, 0.0, 0.0])

    vix_val = float(vix_series.loc[date]) if date in vix_series.index else 20.0
    risk_free = float(getattr(args, "risk_free_rate", 0.03))
    global_context = np.array([vix_val, 0.0, risk_free], dtype=np.float32)

    state = np.concatenate([np.array(market_data, dtype=np.float32), global_context])
    return state.astype(np.float32)

# ============================================================================
# 4. مسیرهای چک‌پوینت
# ============================================================================

CHECKPOINT_DIRS = {
    'no_node2vec': 'checkpoints_ablation_no_node2vec',
    'no_cash': 'checkpoints_ablation_no_cash',
    'single_agent': 'checkpoints_ablation_single_agent'
}

RESULTS_DIR = "results_ablation"

# ============================================================================
# 5. توابع گزارش و تست (با محاسبه کامل معیارها)
# ============================================================================

def print_test_report(stats, year, elapsed_time):
    if not stats:
        return
    print("\n" + "="*70)
    print(f"📊 TEST YEAR {year} REPORT")
    print("="*70)
    print(f"   📅 Trading Days:        {stats.get('num_days', 'N/A')}")
    print(f"   💰 Initial Value:       ${stats.get('initial_value', 0):,.2f}")
    print(f"   💰 Final Value:         ${stats.get('final_value', 0):,.2f}")
    print(f"   📈 Cumulative Return:   {stats.get('cumulative_return', np.nan):.2%}")
    print(f"   📊 Sharpe Ratio:        {stats.get('sharpe_ratio', np.nan):.4f}")
    print(f"   📉 Max Drawdown:        {stats.get('max_drawdown', np.nan):.2%}")
    print(f"   📊 Sortino Ratio:       {stats.get('Sortino', np.nan):.4f}")
    print(f"   📊 Calmar Ratio:        {stats.get('Calmar', np.nan):.4f}")
    print(f"   📉 VaR (95%):           {stats.get('VaR_95', np.nan):.4%}")
    print(f"   📉 CVaR (95%):          {stats.get('CVaR_95', np.nan):.4%}")
    print(f"   🔄 Total Turnover:      {stats.get('total_turnover', np.nan):.2f}")
    print(f"   🔄 Avg Daily Turnover:  {stats.get('avg_turnover', np.nan):.4f}")
    print(f"   📈 Mean Daily Return:   {stats.get('mean_return', np.nan):.4%}")
    print(f"   📉 Volatility:          {stats.get('volatility', np.nan):.4%}")
    print(f"   📊 HHI:                 {stats.get('Avg_HHI', np.nan):.4f}")
    print(f"   📊 ENB:                 {stats.get('Avg_ENB', np.nan):.2f}")
    print(f"   📊 Max Stock Weight:    {stats.get('Avg_Max_Weight', np.nan):.2%}")
    print(f"   ⏱️  Elapsed Time:        {elapsed_time:.1f} seconds")
    print("="*70 + "\n")


def test_single_seed(seed, checkpoint_base, mode, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🧪 Testing {mode} with seed={seed} on device={device}")

    returns_df, features_df, vix_series = load_data(args.data_dir, split="test")
    fixed_basket_path = "selected_stocks/fixed_test_basket_summary.csv"
    fixed_basket = pd.read_csv(fixed_basket_path)['Selected_Tickers'].iloc[0].split('|')
    full_sector_map = get_sector_map_from_file("sp500_panel_data/sp500_sectors_map.csv")

    seed_checkpoint_dir = os.path.join(checkpoint_base, f"seed_{seed}")
    all_results = []
    total_start_time = time.time()

    for year_idx, year in enumerate(args.test_years):
        print(f"\n{'='*60}")
        print(f"📅 TEST EPOCH {year_idx+1}/{len(args.test_years)}: Year {year} ({mode}, seed={seed})")
        print(f"📊 Fixed Basket: {', '.join(fixed_basket)}")
        print(f"{'='*60}")

        sector_map = get_sector_map_from_basket(fixed_basket, full_sector_map)
        missing = set(fixed_basket) - set([t for tickers in sector_map.values() for t in tickers])
        if missing:
            sector_map['Other'] = list(missing)
        print(f"📊 Sectors for {year}: {list(sector_map.keys())}")

        # ============================================================
        # حالت SINGLE_AGENT: استفاده از PortfolioSingleAgent
        # ============================================================
        if mode == 'single_agent':
            state_dim = 3*len(fixed_basket) + 64*len(fixed_basket) + 3
            action_dim = len(fixed_basket)
            agent = SinglePPOAgent(state_dim, action_dim, args)
            agent.checkpoint_dir = os.path.join(seed_checkpoint_dir, "single_agent")
            if not agent.load(2020):
                print("⚠️ Could not load Single Agent. Starting from scratch.")
            else:
                print(f"✅ Loaded Single Agent (state={state_dim}, action={action_dim})")

            env = PortfolioSingleAgent(
                config=args,
                stock_names=fixed_basket,
                sector_map=sector_map,
                returns_df=returns_df,
                features_df=features_df,
                vix_series=vix_series,
                year=year
            )
            obs = env.reset()
            done = False
            step_count = 0
            current_date = env.current_date
            history_list = []
            alpha_history = [] 
            weights_history = []
            daily_returns = []
            print(f"📅 Starting from date: {current_date}")

            while not done:
                state = build_sector_state(current_date, fixed_basket, features_df, vix_series, args)
                weights, _ = agent.get_action(state, deterministic=True)
                flat_action = weights
                next_obs, reward, done, info = env.step(flat_action)
                daily_return = info.get('simple_return', 0.0)
                daily_returns.append(daily_return)

                # ذخیره وزن‌های نهایی
                final_weights = env.weights  # برای PortfolioSingleAgent، weights آرایه‌ای از وزن سهام است
                weights_record = {'date': current_date}
                for i, ticker in enumerate(fixed_basket):
                    weights_record[ticker] = final_weights[i]
                weights_history.append(weights_record)

                history_list.append({
                    'date': current_date,
                    'port_value_new': env.port_value,
                    'simple_return': daily_return
                })
                obs = next_obs
                current_date = env.current_date
                step_count += 1
                if step_count % 50 == 0:
                    print(f"   📈 Day {step_count}/{len(env.dates)} | Value: ${env.port_value:,.2f}")

        # ============================================================
        # حالت‌های NO_NODE2VEC و NO_CASH: استفاده از Portfolio اصلی
        # ============================================================
        else:
            sector_agents = {}
            for sector_name, tickers in sector_map.items():
                if len(tickers) == 1:
                    continue
                if mode == 'no_node2vec':
                    state_dim = 3*len(tickers) + 3
                else:
                    state_dim = 3*len(tickers) + 64*len(tickers) + 3
                action_dim = len(tickers)
                agent = SectorPPOAgent(sector_name, state_dim, action_dim, args)
                agent.checkpoint_dir = os.path.join(seed_checkpoint_dir, f"ppo_{sector_name}")
                if agent.load(2020):
                    print(f"   ✅ Loaded {sector_name}")
                else:
                    print(f"   ⚠️ Could not load {sector_name}. Starting from scratch.")
                sector_agents[sector_name] = agent

            n_sectors = len(sector_map)
            sample_date = pd.to_datetime(f"{year}-01-02").date()
            sample_mac_state = build_mac_state(sample_date, split="test", mac_dir="mac_features", n_sectors=n_sectors)
            mac_state_dim = len(sample_mac_state)
            mac_agent = MACSACAgent(mac_state_dim, n_sectors, args)
            mac_agent.checkpoint_dir = os.path.join(seed_checkpoint_dir, "mac")
            if not mac_agent.load(2020):
                print("⚠️ Could not load MAC. Starting from scratch.")
            else:
                print(f"✅ Loaded MAC (state={mac_state_dim})")

            # بدون نقدینگی: action_dim را اصلاح کن
            if mode == 'no_cash':
                mac_agent.action_dim = n_sectors

            env = Portfolio(
                config=args,
                stock_names=fixed_basket,
                sector_map=sector_map,
                returns_df=returns_df,
                features_df=features_df,
                vix_series=vix_series,
                year=year
            )
            obs = env.reset()
            done = False
            step_count = 0
            current_date = env.current_date
            history_list = []
            weights_history = []
            alpha_history = []
            daily_returns = []
            print(f"📅 Starting from date: {current_date}")

            while not done:
                sector_actions = {}
                for sector_name, tickers in sector_map.items():
                    if len(tickers) == 1:
                        sector_actions[sector_name] = np.array([1.0], dtype=np.float32)
                        continue
                    agent = sector_agents[sector_name]
                    if mode == 'no_node2vec':
                        state = build_sector_state_no_node2vec(current_date, tickers, features_df, vix_series, args)
                    else:
                        state = build_sector_state(current_date, tickers, features_df, vix_series, args)
                    weights, _ = agent.get_action(state, deterministic=True)
                    sector_actions[sector_name] = weights

                mac_state = build_mac_state(current_date, split="test", mac_dir="mac_features", n_sectors=len(sector_map))
                mac_action = mac_agent.get_action(mac_state, deterministic=True)

                # ذخیره آلفاها
                alphas = mac_action[1:] if mode != 'no_cash' else mac_action
                alpha_record = {'date': current_date}
                for i, val in enumerate(alphas):
                    alpha_record[f'alpha_{i}'] = val
                alpha_history.append(alpha_record)

                if mode == 'no_cash':
                    flat_action = np.concatenate([mac_action] + [sector_actions[s] for s in sector_map.keys()])
                else:
                    flat_action = np.concatenate([mac_action] + [sector_actions[s] for s in sector_map.keys()])

                next_obs, reward, done, info = env.step(flat_action)
                daily_return = info.get('simple_return', 0.0)
                daily_returns.append(daily_return)

                # ذخیره وزن‌های نهایی
                final_weights = env.weights  # [cash, stock1, stock2, ...]
                weights_record = {'date': current_date}
                for i, ticker in enumerate(fixed_basket):
                    weights_record[ticker] = final_weights[i+1]
                weights_history.append(weights_record)

                history_list.append({
                    'date': current_date,
                    'port_value_new': env.port_value,
                    'simple_return': daily_return
                })
                obs = next_obs
                current_date = env.current_date
                step_count += 1
                if step_count % 50 == 0:
                    print(f"   📈 Day {step_count}/{len(env.dates)} | Value: ${env.port_value:,.2f}")

        # ---------- محاسبه معیارها از بازده روزانه ----------
        risk_metrics = calculate_risk_metrics(np.array(daily_returns)) if len(daily_returns) > 1 else {}
        
        # ---------- محاسبه معیارهای تمرکز سبد ----------
        if weights_history:
            weights_df = pd.DataFrame(weights_history)
            concentration_metrics = calculate_concentration_metrics(weights_df)
        else:
            concentration_metrics = {'Avg_Max_Weight': np.nan, 'Avg_HHI': np.nan, 'Avg_ENB': np.nan}

        elapsed_time = time.time() - total_start_time
        stats = env.get_stats()

        # اضافه کردن همه معیارها به stats
        stats.update(risk_metrics)
        stats.update(concentration_metrics)
        all_results.append(stats)

        print_test_report(stats, year, elapsed_time)

        # ذخیره تاریخچه‌ها
        os.makedirs(f"h_marl_results_ablation/{mode}", exist_ok=True)
        if history_list:
            history_df = pd.DataFrame(history_list)
            history_df['date'] = pd.to_datetime(history_df['date'])
            history_df.to_csv(f"h_marl_results_ablation/{mode}/h_marl_history_{year}_seed_{seed}.csv", index=False)
        if alpha_history and mode != 'single_agent':
            alpha_df = pd.DataFrame(alpha_history)
            alpha_df['date'] = pd.to_datetime(alpha_df['date'])
            alpha_df.to_csv(f"h_marl_results_ablation/{mode}/alpha_history_{year}_seed_{seed}.csv", index=False)
        if weights_history:
            weights_df = pd.DataFrame(weights_history)
            weights_df['date'] = pd.to_datetime(weights_df['date'])
            weights_df.to_csv(f"h_marl_results_ablation/{mode}/weights_history_{year}_seed_{seed}.csv", index=False)

    total_time = time.time() - total_start_time
    print(f"\n✅ Test completed for {mode} seed={seed} in {total_time/60:.1f} minutes.")
    return all_results, seed


def test_all():
    args = ConfigAblation()
    #all_modes = ['no_node2vec', 'no_cash', 'single_agent']
    all_modes = ['single_agent'] 

    for mode in all_modes:
        print(f"\n{'#'*70}")
        print(f"🧪 TESTING ABLATION: {mode.upper()}")
        print(f"{'#'*70}")

        checkpoint_base = CHECKPOINT_DIRS[mode]
        output_dir = os.path.join(RESULTS_DIR, mode)
        os.makedirs(output_dir, exist_ok=True)

        all_results = {}
        for seed in args.seeds:
            results, seed_val = test_single_seed(seed, checkpoint_base, mode, args)
            all_results[seed_val] = results
            results_df = pd.DataFrame(results)
            results_df.to_csv(f"{output_dir}/test_results_seed_{seed_val}.csv", index=False)
            print(f"💾 Test results for {mode} seed={seed_val} saved.")

        # خلاصه با تمام معیارها
        summary_data = []
        for seed, results in all_results.items():
            for stats in results:
                summary_data.append({
                    'seed': seed,
                    'year': stats.get('year', 0),
                    'cumulative_return': stats.get('cumulative_return', np.nan),
                    'sharpe_ratio': stats.get('sharpe_ratio', np.nan),
                    'max_drawdown': stats.get('max_drawdown', np.nan),
                    'final_value': stats.get('final_value', np.nan),
                    'volatility': stats.get('volatility', np.nan),
                    'mean_return': stats.get('mean_return', np.nan),
                    'num_days': stats.get('num_days', 0),
                    'VaR_95': stats.get('VaR_95', np.nan),
                    'CVaR_95': stats.get('CVaR_95', np.nan),
                    'Sortino': stats.get('Sortino', np.nan),
                    'Calmar': stats.get('Calmar', np.nan),
                    'Avg_Max_Weight': stats.get('Avg_Max_Weight', np.nan),
                    'Avg_HHI': stats.get('Avg_HHI', np.nan),
                    'Avg_ENB': stats.get('Avg_ENB', np.nan),
                    'total_turnover': stats.get('total_turnover', np.nan),
                    'avg_turnover': stats.get('avg_turnover', np.nan)
                })
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{output_dir}/test_summary.csv", index=False)
        print(f"✅ {mode} test summary saved to {output_dir}/test_summary.csv")


if __name__ == "__main__":
    test_all()
