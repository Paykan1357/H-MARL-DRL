
import os
import numpy as np
import pandas as pd
import torch
import random
import time
from config import Config
from Portfolio import Portfolio
from sector_ppo_agent import SectorPPOAgent
from mac_sac_agent import MACSACAgent
from utils import load_data, build_sector_state, build_mac_state, get_sector_map_from_file, get_sector_map_from_basket


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
    
    # ✅ تبدیل به pandas Series برای استفاده از expanding()
    returns = pd.Series(returns)
    log_returns = np.log(1 + returns)
    
    mean_ret = returns.mean()
    std_ret = returns.std()
    risk_free_daily = 0.03 / 252
    excess_returns = returns - risk_free_daily
    
    # Sharpe
    sharpe = (excess_returns.mean() / (std_ret + 1e-8)) * np.sqrt(252)
    
    # Sortino (ریسک نزولی)
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) > 0:
        sortino = (excess_returns.mean() / (downside_returns.std() + 1e-8)) * np.sqrt(252)
    else:
        sortino = np.inf
    
    # ✅ حداکثر افت (با pandas.expanding())
    cum_values = (1 + returns).cumprod()
    peak = cum_values.expanding().max()
    drawdown = (cum_values - peak) / peak
    max_drawdown = drawdown.min()
    
    # کلمار
    cum_return = (1 + returns).prod() - 1
    calmar = cum_return / (abs(max_drawdown) + 1e-8)
    
    # VaR و CVaR (تاریخی)
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


def test_single_seed(seed, args):
    """تست برای یک سیِد مشخص با محاسبه کامل معیارها"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🧪 Testing with seed={seed} on device={device}")
    
    # ---------- بارگذاری داده‌ها ----------
    returns_df, features_df, vix_series = load_data(args.data_dir, split="test")
    
    fixed_basket_path = "selected_stocks/fixed_test_basket_summary.csv"
    if not os.path.exists(fixed_basket_path):
        print(f"❌ Fixed basket not found: {fixed_basket_path}")
        return None, seed
    
    fixed_basket = pd.read_csv(fixed_basket_path)['Selected_Tickers'].iloc[0].split('|')
    print(f"✅ Fixed basket loaded: {len(fixed_basket)} stocks")
    
    full_sector_map = get_sector_map_from_file("sp500_panel_data/sp500_sectors_map.csv")
    print(f"✅ Loaded full sector map with {len(full_sector_map)} sectors.")
    
    # مسیر چک‌پوینت این سیِد
    seed_checkpoint_dir = os.path.join(args.checkpoint_dir, f"seed_{seed}")
    
    all_results = []
    total_start_time = time.time()
    
    for year_idx, year in enumerate(args.test_years):
        print(f"\n{'='*60}")
        print(f"📅 TEST EPOCH {year_idx+1}/{len(args.test_years)}: Year {year} (seed={seed})")
        print(f"📊 Fixed Basket: {', '.join(fixed_basket)}")
        print(f"{'='*60}")
        
        # ---------- ساخت نگاشت بخش‌ها ----------
        sector_map = get_sector_map_from_basket(fixed_basket, full_sector_map)
        all_tickers_in_map = set([t for tickers in sector_map.values() for t in tickers])
        missing = set(fixed_basket) - all_tickers_in_map
        if missing:
            sector_map['Other'] = list(missing)
        print(f"📊 Sectors for {year}: {list(sector_map.keys())}")
        
        # ---------- ایجاد عامل‌ها از چک‌پوینت این سیِد ----------
        sector_agents = {}
        for sector_name, tickers in sector_map.items():
            if len(tickers) == 1:
                print(f"ℹ️ Sector {sector_name} has only 1 stock. Skipping agent.")
                continue
            
            state_dim = 3*len(tickers) + 64*len(tickers) + 3
            action_dim = len(tickers)
            agent = SectorPPOAgent(sector_name, state_dim, action_dim, args)
            agent.checkpoint_dir = os.path.join(seed_checkpoint_dir, f"ppo_{sector_name}")
            
            # ✅ بارگذاری از آخرین سال آموزش (2022) به‌جای 2020
            checkpoint_year = 2022
            if agent.load(checkpoint_year):
                print(f"   ✅ Loaded {sector_name} from year {checkpoint_year}")
            else:
                print(f"   ⚠️ Could not load {sector_name} from year {checkpoint_year}. Starting fresh.")
            
            sector_agents[sector_name] = agent
            print(f"✅ Initialized {sector_name} PPO Agent (state={state_dim}, action={action_dim})")
        
        # ---------- MAC Agent ----------
        n_sectors = len(sector_map)
        sample_date = pd.to_datetime(f"{year}-01-02").date()
        sample_mac_state = build_mac_state(sample_date, split="test", mac_dir="mac_features", n_sectors=n_sectors)
        mac_state_dim = len(sample_mac_state)
        
        mac_agent = MACSACAgent(mac_state_dim, n_sectors, args)
        mac_agent.checkpoint_dir = os.path.join(seed_checkpoint_dir, "mac")
        
        # ✅ بارگذاری از آخرین سال آموزش (2022)
        checkpoint_year = 2022
        if mac_agent.load(checkpoint_year):
            print(f"   ✅ Loaded MAC from year {checkpoint_year}")
        else:
            print(f"   ⚠️ Could not load MAC from year {checkpoint_year}. Starting fresh.")
        
        print(f"✅ Initialized MAC SAC Agent (state={mac_state_dim}, action={n_sectors+1})")
        
        # ---------- ایجاد محیط ----------
        try:
            print("⏳ Creating environment...")
            env = Portfolio(
                config=args,
                stock_names=fixed_basket,
                sector_map=sector_map,
                returns_df=returns_df,
                features_df=features_df,
                vix_series=vix_series,
                year=year
            )
            print(f"✅ Environment created successfully. ({len(env.dates)} trading days)")
        except Exception as e:
            print(f"❌ Error creating environment: {e}")
            continue
        
        obs = env.reset()
        done = False
        step_count = 0
        current_date = env.current_date
        
        # ---------- لیست‌ها برای ذخیره تاریخچه ----------
        history_list = []
        alpha_history = []
        weights_history = []
        daily_returns = []  # برای محاسبه معیارهای ریسک
        
        print(f"📅 Starting from date: {current_date}")
        
        while not done:
            try:
                # ---- گرفتن Actions از عامل‌های بخشی ----
                sector_actions = {}
                for sector_name, tickers in sector_map.items():
                    if len(tickers) == 1:
                        sector_actions[sector_name] = np.array([1.0], dtype=np.float32)
                        continue
                    
                    if sector_name not in sector_agents:
                        sector_actions[sector_name] = np.ones(len(tickers), dtype=np.float32) / len(tickers)
                        continue
                    
                    agent = sector_agents[sector_name]
                    state = build_sector_state(current_date, tickers, features_df, vix_series, args)
                    weights, _ = agent.get_action(state, deterministic=True)
                    sector_actions[sector_name] = weights
                
                # ---- گرفتن Action از MAC ----
                mac_state = build_mac_state(current_date, split="test", mac_dir="mac_features", n_sectors=len(sector_map))
                mac_action = mac_agent.get_action(mac_state, deterministic=True)
                
                # ---- ذخیره آلفاها برای هیت‌مپ ----
                alphas = mac_action[1:]  # حذف نقدینگی
                alpha_record = {'date': current_date}
                for i, val in enumerate(alphas):
                    alpha_record[f'alpha_{i}'] = val
                alpha_history.append(alpha_record)
                
                # ---- ساخت Action نهایی ----
                flat_action = np.concatenate([
                    mac_action,
                    *[sector_actions[s] for s in sector_map.keys()]
                ])
                
                # ---- اجرا در محیط ----
                next_obs, reward, done, info = env.step(flat_action)
                
                # ---- ذخیره بازده روزانه ----
                daily_return = info.get('simple_return', 0.0)
                daily_returns.append(daily_return)
                
                # ---- ذخیره وزن‌های نهایی سهام ----
                final_weights = env.weights  # [cash, stock1, stock2, ...]
                weights_record = {'date': current_date}
                for i, ticker in enumerate(fixed_basket):
                    weights_record[ticker] = final_weights[i+1]
                weights_history.append(weights_record)
                
                # ---- ذخیره تاریخچه روزانه ----
                history_list.append({
                    'date': current_date,
                    'port_value_new': env.port_value,
                    'simple_return': daily_return
                })
                
                obs = next_obs
                current_date = env.current_date
                step_count += 1
                
                if step_count % 50 == 0:
                    progress = f"Day {step_count}/{len(env.dates)}"
                    print(f"   📈 {progress} | Value: ${env.port_value:,.2f} | Return: {daily_return:.4%}")
                
            except Exception as e:
                print(f"❌ Error at step {step_count} on date {current_date}: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # ---------- محاسبه معیارها از بازده روزانه ----------
        risk_metrics = calculate_risk_metrics(np.array(daily_returns)) if len(daily_returns) > 1 else {}
        
        # ---------- محاسبه معیارهای تمرکز سبد ----------
        if weights_history:
            weights_df = pd.DataFrame(weights_history)
            concentration_metrics = calculate_concentration_metrics(weights_df)
        else:
            concentration_metrics = {'Avg_Max_Weight': np.nan, 'Avg_HHI': np.nan, 'Avg_ENB': np.nan}
        
        # ---------- پایان سال: آمار و ذخیره‌سازی ----------
        elapsed_time = time.time() - total_start_time
        stats = env.get_stats()
        
        # اضافه کردن همه معیارها به stats
        stats.update(risk_metrics)
        stats.update(concentration_metrics)
        all_results.append(stats)
        
        # گزارش تکمیلی
        print_test_report(stats, year, elapsed_time)
        
        # ذخیره تاریخچه‌ها در پوشه‌ی h_marl_results با پسوند seed
        os.makedirs("h_marl_results", exist_ok=True)
        
        if history_list:
            history_df = pd.DataFrame(history_list)
            history_df['date'] = pd.to_datetime(history_df['date'])
            history_df.to_csv(f"h_marl_results/h_marl_history_{year}_seed_{seed}.csv", index=False)
            print(f"✅ H-MARL history saved to h_marl_results/h_marl_history_{year}_seed_{seed}.csv")
        
        if alpha_history:
            alpha_df = pd.DataFrame(alpha_history)
            alpha_df['date'] = pd.to_datetime(alpha_df['date'])
            alpha_df.to_csv(f"h_marl_results/alpha_history_{year}_seed_{seed}.csv", index=False)
            print(f"✅ Alpha history saved to h_marl_results/alpha_history_{year}_seed_{seed}.csv")
        
        if weights_history:
            weights_df = pd.DataFrame(weights_history)
            weights_df['date'] = pd.to_datetime(weights_df['date'])
            weights_df.to_csv(f"h_marl_results/weights_history_{year}_seed_{seed}.csv", index=False)
            print(f"✅ Weights history saved to h_marl_results/weights_history_{year}_seed_{seed}.csv")
    
    # ---------- جمع‌بندی نتایج این سیِد ----------
    total_time = time.time() - total_start_time
    print(f"\n✅ Test completed for seed={seed} in {total_time/60:.1f} minutes.")
    return all_results, seed


def test_all_seeds():
    args = Config()
    all_seeds_results = {}
    
    for seed in args.seeds:
        results, seed_val = test_single_seed(seed, args)
        if results is not None:
            all_seeds_results[seed_val] = results
            
            # ذخیره نتایج خام این سیِد
            results_df = pd.DataFrame(results)
            results_df.to_csv(f"results/test_results_seed_{seed_val}.csv", index=False)
            print(f"💾 Test results for seed={seed_val} saved.")
    
    # ---------- خلاصه همه سیِدها ----------
    if all_seeds_results:
        summary_data = []
        for seed, results in all_seeds_results.items():
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
                    'Avg_ENB': stats.get('Avg_ENB', np.nan)
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv("results/all_test_seeds_summary.csv", index=False)
        print("\n✅ All test seeds summary saved to results/all_test_seeds_summary.csv")
    else:
        print("⚠️ No results to summarize.")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    test_all_seeds()
