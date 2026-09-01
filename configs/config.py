
# config.py - نسخه بهبودیافته (قدم دوم)

class Config:
    """Hyperparameters for the Hierarchical MARL framework - IMPROVED VERSION"""
    
    # ---------- Environment ----------
    window_length = 21
    transaction_cost = 0.001
    risk_free_rate = 0.03
    init_portfolio_value = 1e6
    
    # ---------- Training ----------
    train_years = list(range(2013, 2023))  # 2013 to 2022 inclusive (9 سال)
    test_years = [2023, 2024, 2025] 
     
    # ---------- Seeds for reproducibility ----------
    seeds = [42, 123, 456, 789, 1010, 2020, 3030, 4040, 5050, 6060]
    # اگر می‌خواهید فقط یک سیِد خاص اجرا شود، از آرگومان خط فرمان استفاده کنید
    # seed = None  # در train.py و test.py از این استفاده می‌شود
    
    # ---------- PPO (Sector Agents) ----------
    ppo_gamma = 0.995              # 🔺 افزایش از 0.99 به 0.995 (تصمیمات بلندمدت‌تر)
    ppo_gae_lambda = 0.95
    ppo_epsilon_clip = 0.2
    ppo_entropy_coef = 0.005       # 🔻 کاهش از 0.01 به 0.005 (کمتر کاوش، بیشتر بهره‌برداری)
    ppo_value_loss_coef = 0.5
    ppo_max_grad_norm = 0.5
    ppo_epochs = 10
    ppo_batch_size = 64
    ppo_lr_actor = 3e-4
    ppo_lr_critic = 1e-3
    
    # ---------- SAC (MAC) ----------
    sac_gamma = 0.995              # 🔺 افزایش از 0.99 به 0.995
    sac_tau = 0.005
    sac_alpha = 0.2
    sac_batch_size = 256
    sac_lr = 3e-4
    sac_buffer_capacity = 100000
    
    # ---------- Reward Weights ----------
    omega_global = 0.4
    omega_local = 0.6
    
    # 🔽 🔽 🔽 تغییرات اصلی اینجا هستند 🔽 🔽 🔽
    lambda_turnover = 0.3          # 🔻 کاهش از 0.5 به 0.3 (اجازه بده بیشتر معامله کنه)
    lambda_drawdown = 0.15         # 🔻 کاهش از 0.3 به 0.15 (کمتر از افت بترسه)
    
    # ---------- Paths ----------
    data_dir = "sp500_panel_data"
    graphs_dir = "sp500_graphs_fast"
    selection_dir = "selected_stocks_by_year"
    checkpoint_dir = "checkpoints"
    output_dir = "results"
