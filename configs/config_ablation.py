
"""
config_ablation.py
تنظیمات مشترک برای Ablation Study
"""

class ConfigAblation:
    # ---------- مسیرهای داده ----------
    data_dir = "sp500_panel_data"
    graphs_dir = "sp500_graphs_fast"
    selection_dir = "selected_stocks_by_year"
    checkpoint_dir = "checkpoints"
    output_dir = "results"
    
    # ---------- Environment ----------
    window_length = 21
    transaction_cost = 0.001
    risk_free_rate = 0.03
    init_portfolio_value = 1e6
    
    # ---------- Training ----------
    train_years = list(range(2013, 2023))  # 2013-2022 (10 سال)
    test_years = [2023, 2024, 2025]
    
    # ---------- Seeds ----------
    seeds = [42, 123, 456, 789, 1010, 2020, 3030, 4040, 5050, 6060]
    
    # ---------- PPO ----------
    ppo_gamma = 0.995
    ppo_gae_lambda = 0.95
    ppo_epsilon_clip = 0.2
    ppo_entropy_coef = 0.005
    ppo_value_loss_coef = 0.5
    ppo_max_grad_norm = 0.5
    ppo_epochs = 10
    ppo_batch_size = 64
    ppo_lr_actor = 3e-4
    ppo_lr_critic = 1e-3
    
    # ---------- SAC ----------
    sac_gamma = 0.995
    sac_tau = 0.005
    sac_alpha = 0.2
    sac_batch_size = 256
    sac_lr = 3e-4
    sac_buffer_capacity = 100000
    
    # ---------- Reward Weights ----------
    omega_global = 0.4
    omega_local = 0.6
    lambda_turnover = 0.3
    lambda_drawdown = 0.15
