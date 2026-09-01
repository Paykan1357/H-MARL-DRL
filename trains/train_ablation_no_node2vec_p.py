

import os
import sys
import numpy as np
import pandas as pd
import torch
import random
import time
from config_ablation import ConfigAblation
from Portfolio import Portfolio
from sector_ppo_agent import SectorPPOAgent
from mac_sac_agent import MACSACAgent
from utils import load_data, load_baskets, build_mac_state, get_sector_map_from_file, get_sector_map_from_basket

# تنظیم مسیر ذخیره‌سازی جداگانه
CHECKPOINT_BASE = "checkpoints_ablation_no_node2vec"
RESULTS_BASE = "results_ablation/no_node2vec"
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


def train_single_seed(seed, args):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Training (No Node2Vec) with seed={seed} on device={device}")
    
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
        print(f"📅 EPOCH {year_idx+1}/{len(args.train_years)}: Trading Year {year} (No Node2Vec)")
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
        
        sector_agents = {}
        for sector_name, tickers in sector_map.items():
            if len(tickers) == 1:
                print(f"ℹ️ Sector {sector_name} has only 1 stock. Skipping PPO agent.")
                continue
            state_dim = 3*len(tickers) + 3  # ✅ بدون Node2Vec (64*len(tickers) حذف شد)
            action_dim = len(tickers)
            agent = SectorPPOAgent(sector_name, state_dim, action_dim, args)
            agent.checkpoint_dir = os.path.join(seed_checkpoint_dir, f"ppo_{sector_name}")
            os.makedirs(agent.checkpoint_dir, exist_ok=True)
            if year_idx > 0:
                prev_year = args.train_years[year_idx - 1]
                if agent.load(prev_year):
                    print(f"   ✅ Loaded {sector_name} from year {prev_year}")
                else:
                    print(f"   ℹ️ No checkpoint for {sector_name} from year {prev_year}. Starting fresh.")
            sector_agents[sector_name] = agent
            print(f"✅ Initialized {sector_name} PPO Agent (state={state_dim}, action={action_dim})")
        
        sample_mac_state = build_mac_state(current_date, split="train", mac_dir="mac_features", n_sectors=len(sector_map))
        mac_state_dim = len(sample_mac_state)
        mac_agent = MACSACAgent(mac_state_dim, len(sector_map), args)
        mac_agent.checkpoint_dir = os.path.join(seed_checkpoint_dir, "mac")
        os.makedirs(mac_agent.checkpoint_dir, exist_ok=True)
        if year_idx > 0:
            prev_year = args.train_years[year_idx - 1]
            if mac_agent.load(prev_year):
                print(f"   ✅ Loaded MAC from year {prev_year}")
            else:
                print(f"   ℹ️ No MAC checkpoint from year {prev_year}. Starting fresh.")
        print(f"✅ Initialized MAC SAC Agent (state={mac_state_dim}, action={len(sector_map)+1})")
        
        print(f"📅 Starting from date: {current_date}")
        while not done:
            try:
                sector_actions = {}
                for sector_name, tickers in sector_map.items():
                    if len(tickers) == 1:
                        sector_actions[sector_name] = np.array([1.0], dtype=np.float32)
                        continue
                    if sector_name not in sector_agents:
                        sector_actions[sector_name] = np.ones(len(tickers), dtype=np.float32) / len(tickers)
                        continue
                    agent = sector_agents[sector_name]
                    state = build_sector_state_no_node2vec(current_date, tickers, features_df, vix_series, args)
                    weights, _ = agent.get_action(state)
                    sector_actions[sector_name] = weights
                
                mac_state = build_mac_state(current_date, split="train", mac_dir="mac_features", n_sectors=len(sector_map))
                mac_action = mac_agent.get_action(mac_state)
                
                flat_action = np.concatenate([mac_action] + [sector_actions[s] for s in sector_map.keys()])
                next_obs, reward, done, info = env.step(flat_action)
                
                reward_history.append(reward)
                
                for sector_name, agent in sector_agents.items():
                    if sector_name not in sector_map:
                        continue
                    sector_reward = info['sector_rewards'].get(sector_name, reward * 0.5)
                    agent.store_reward(sector_reward, done)
                mac_agent.store_transition(mac_state, mac_action, reward, mac_state, done)
                
                if step_count % 10 == 0 and step_count > 0:
                    for agent in sector_agents.values():
                        agent.update()
                    mac_agent.update()
                
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
        
        for sector_name, agent in sector_agents.items():
            agent.save(year)
        mac_agent.save(year)
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
