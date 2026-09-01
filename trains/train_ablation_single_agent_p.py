

import os
import sys
import numpy as np
import pandas as pd
import torch
import random
import time
from config_ablation import ConfigAblation
#from Portfolio import Portfolio
from sector_ppo_agent import SectorPPOAgent

from utils import load_data, load_baskets, build_sector_state, get_sector_map_from_file, get_sector_map_from_basket
# در train_ablation_single_agent.py
from PortfolioSingleAgent import PortfolioSingleAgent as Portfolio  # ✅ جایگزین Portfolio

CHECKPOINT_BASE = "checkpoints_ablation_single_agent"
RESULTS_BASE = "results_ablation/single_agent"
os.makedirs(CHECKPOINT_BASE, exist_ok=True)
os.makedirs(RESULTS_BASE, exist_ok=True)


def load_data_combined(data_dir="sp500_panel_data"):
    from utils import load_data as load_data_single
    returns_train, features_train, vix_train = load_data_single(data_dir, split="train")
    returns_val, features_val, vix_val = load_data_single(data_dir, split="val")
    returns_combined = pd.concat([returns_train, returns_val])
    returns_combined = returns_combined[~returns_combined.index.duplicated(keep='first')]
    features_combined = pd.concat([features_train, features_val], ignore_index=True)
    features_combined = features_combined.drop_duplicates(subset=['Date', 'Ticker'], keep='first')
    vix_combined = pd.concat([vix_train, vix_val])
    vix_combined = vix_combined[~vix_combined.index.duplicated(keep='first')]
    return returns_combined, features_combined, vix_combined


def print_year_report(stats, year, elapsed_time):
    if not stats:
        return
    print("\n" + "="*70)
    print(f"📊 YEAR {year} REPORT")
    print("="*70)
    print(f"   📅 Trading Days:        {stats['num_days']}")
    print(f"   💰 Initial Value:       ${stats['initial_value']:,.2f}")
    print(f"   💰 Final Value:         ${stats['final_value']:,.2f}")
    print(f"   📈 Cumulative Return:   {stats['cumulative_return']:.2%}")
    print(f"   📊 Sharpe Ratio:        {stats['sharpe_ratio']:.4f}")
    print(f"   📉 Max Drawdown:        {stats['max_drawdown']:.2%}")
    print(f"   🔄 Total Turnover:      {stats['total_turnover']:.2f}")
    print(f"   🔄 Avg Daily Turnover:  {stats['avg_turnover']:.4f}")
    print(f"   📈 Mean Daily Return:   {stats['mean_return']:.4%}")
    print(f"   📉 Volatility:          {stats['volatility']:.4%}")
    print(f"   🎯 Avg Daily Reward:    {stats['avg_reward']:.4f}")
    print(f"   ⏱️  Elapsed Time:        {elapsed_time:.1f} seconds")
    print("="*70 + "\n")


class SinglePPOAgent:
    """یک عامل PPO واحد برای کل سبد"""
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


def train_single_seed(seed, args):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Training (Single Agent) with seed={seed} on device={device}")
    
    returns_df, features_df, vix_series = load_data_combined(args.data_dir)
    baskets = load_baskets(args.selection_dir, split="train")
    full_sector_map = get_sector_map_from_file("sp500_panel_data/sp500_sectors_map.csv")
    
    seed_checkpoint_dir = os.path.join(CHECKPOINT_BASE, f"seed_{seed}")
    os.makedirs(seed_checkpoint_dir, exist_ok=True)
    
    all_results = []
    start_time = time.time()
    
    for year_idx, year in enumerate(args.train_years):
        year_start_time = time.time()
        if year not in baskets:
            continue
        basket = baskets[year]
        print(f"\n{'='*60}")
        print(f"📅 EPOCH {year_idx+1}/{len(args.train_years)}: Trading Year {year} (Single Agent)")
        print(f"📊 Basket ({len(basket)} stocks): {', '.join(basket)}")
        print(f"{'='*60}")
        
        sector_map = get_sector_map_from_basket(basket, full_sector_map)
        missing = set(basket) - set([t for tickers in sector_map.values() for t in tickers])
        if missing:
            sector_map['Other'] = list(missing)
        print(f"📊 Sectors for {year}: {list(sector_map.keys())}")
        
        env = Portfolio(
            config=args,
            stock_names=basket,
            sector_map=sector_map,
            returns_df=returns_df,
            features_df=features_df,
            vix_series=vix_series,
            year=year
        )
        print(f"✅ Environment created. ({len(env.dates)} trading days)")
        
        obs = env.reset()
        done = False
        step_count = 0
        current_date = env.current_date
        
        reward_history = []
        
        # ✅ یک عامل واحد روی کل ۲۰ سهم
        state_dim = 3*len(basket) + 64*len(basket) + 3  # market + node2vec + context
        action_dim = len(basket)
        agent = SinglePPOAgent(state_dim, action_dim, args)
        agent.checkpoint_dir = os.path.join(seed_checkpoint_dir, "single_agent")
        os.makedirs(agent.checkpoint_dir, exist_ok=True)
        if year_idx > 0:
            prev_year = args.train_years[year_idx - 1]
            if agent.load(prev_year):
                print(f"   ✅ Loaded Single Agent from year {prev_year}")
            else:
                print(f"   ℹ️ No checkpoint for Single Agent from year {prev_year}. Starting fresh.")
        print(f"✅ Initialized Single PPO Agent (state={state_dim}, action={action_dim})")
        
        print(f"📅 Starting from date: {current_date}")
        while not done:
            try:
                state = build_sector_state(current_date, basket, features_df, vix_series, args)
                weights, _ = agent.get_action(state)
                
                # تبدیل به عمل تخت (بدون MAC)
                flat_action = weights  # فقط وزن سهام
                
                next_obs, reward, done, info = env.step(flat_action)
                
                reward_history.append(reward)
                agent.store_reward(reward, done)
                
                if step_count % 10 == 0 and step_count > 0:
                    agent.update()
                
                obs = next_obs
                current_date = env.current_date
                step_count += 1
                
                if step_count % 50 == 0:
                    progress = f"Day {step_count}/{len(env.dates)}"
                    print(f"   📈 {progress} | Reward: {reward:.4f} | Value: ${env.port_value:,.2f} | Return: {info['simple_return']:.4%}")
                
            except Exception as e:
                print(f"❌ Error at step {step_count} on date {current_date}: {e}")
                import traceback
                traceback.print_exc()
                break
        
        elapsed_time = time.time() - year_start_time
        stats = env.get_stats()
        all_results.append(stats)
        print_year_report(stats, year, elapsed_time)
        
        if reward_history:
            reward_df = pd.DataFrame({'step': list(range(len(reward_history))), 'reward': reward_history})
            reward_df.to_csv(os.path.join(seed_checkpoint_dir, f"reward_history_{year}.csv"), index=False)
        
        agent.save(year)
        print(f"💾 Checkpoints saved for year {year} (seed={seed}).")
    
    total_time = time.time() - start_time
    print(f"\n✅ Training completed for seed={seed} in {total_time/60:.1f} minutes.")
    return all_results, seed


def train_all_seeds():
    args = ConfigAblation()
    all_seeds_results = {}
    
    for seed in args.seeds:
        results, seed_val = train_single_seed(seed, args)
        all_seeds_results[seed_val] = results
        results_df = pd.DataFrame(results)
        results_df.to_csv(f"{RESULTS_BASE}/train_results_seed_{seed_val}.csv", index=False)
        print(f"💾 Train results for seed={seed_val} saved.")
    
    summary_data = []
    for seed, results in all_seeds_results.items():
        for stats in results:
            summary_data.append({
                'seed': seed,
                'year': stats.get('year', 0),
                'cumulative_return': stats.get('cumulative_return', 0.0),
                'sharpe_ratio': stats.get('sharpe_ratio', 0.0),
                'max_drawdown': stats.get('max_drawdown', 0.0),
                'num_days': stats.get('num_days', 0),
                'final_value': stats.get('final_value', 0.0)
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f"{RESULTS_BASE}/all_train_seeds_summary.csv", index=False)
    print(f"\n✅ All training seeds summary saved to {RESULTS_BASE}/all_train_seeds_summary.csv")


if __name__ == "__main__":
    train_all_seeds()
