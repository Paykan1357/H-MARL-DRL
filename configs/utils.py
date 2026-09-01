
"""
utils.py
توابع کمکی برای بارگذاری داده‌ها، Embedding‌ها، ویژگی‌های MAC و ساخت State
با پشتیبانی از ابعاد پویا برای MAC
"""

import os
import numpy as np
import pandas as pd
import datetime

# ============================================================================
# 1. بارگذاری داده‌های اصلی (بازده، ویژگی‌ها، VIX)
# ============================================================================

def load_data(data_dir="sp500_panel_data", split="train"):
    """
    بارگذاری بازده روزانه، ویژگی‌ها و سری VIX برای یک اسپلیت مشخص.
    """
    # بازده
    ret_path = os.path.join(data_dir, f"{split}_returns.csv")
    if not os.path.exists(ret_path):
        raise FileNotFoundError(f"Returns file not found: {ret_path}")
    returns_df = pd.read_csv(ret_path, parse_dates=['Date'])
    returns_df['Date'] = pd.to_datetime(returns_df['Date']).dt.date
    returns_df = returns_df.set_index('Date')

    # ویژگی‌ها
    feat_path = os.path.join(data_dir, f"{split}_features.csv")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Features file not found: {feat_path}")
    features_df = pd.read_csv(feat_path, parse_dates=['Date'])
    features_df['Date'] = pd.to_datetime(features_df['Date']).dt.date

    # VIX
    vix_path = os.path.join(data_dir, f"vix_{split}.csv")
    if not os.path.exists(vix_path):
        raise FileNotFoundError(f"VIX file not found: {vix_path}")
    vix_df = pd.read_csv(vix_path, parse_dates=['date'])
    vix_df['date'] = pd.to_datetime(vix_df['date']).dt.date
    vix_series = vix_df.set_index('date')['vix_close']

    return returns_df, features_df, vix_series


def load_baskets(selection_dir="selected_stocks_by_year", split="train"):
    """
    بارگذاری سبدهای انتخابی (سالانه برای آموزش، یا ثابت برای تست)
    """
    if split == "train":
        path = os.path.join(selection_dir, "selected_stocks_train_by_year.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Basket file not found: {path}")
        df = pd.read_csv(path)
        baskets = {}
        for _, row in df.iterrows():
            baskets[row['Year']] = row['Selected_Tickers'].split('|')
        return baskets
    else:
        # تست: یک سبد ثابت
        path = os.path.join("selected_stocks", "fixed_test_basket_summary.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fixed basket not found: {path}")
        df = pd.read_csv(path)
        return df['Selected_Tickers'].iloc[0].split('|')


# ============================================================================
# 2. بارگذاری Node2Vec Embeddings (از فایل‌های .npz)
# ============================================================================

def load_node2vec_embeddings(tickers, date, embed_dir="embeddings"):
    """
    بارگذاری بردارهای Node2Vec برای یک تاریخ مشخص.
    فایل‌های Embedding به‌صورت سالانه ذخیره شده‌اند.
    اگر فایل وجود نداشت، بردار صفر برمی‌گرداند (بدون مقدار تصادفی).
    """
    year = date.year
    embed_file = os.path.join(embed_dir, f"embeddings_train_{year}.npz")
    
    if not os.path.exists(embed_file):
        return {t: np.zeros(64, dtype=np.float32) for t in tickers}
    
    try:
        data = np.load(embed_file, allow_pickle=True)
        ticker_list = data['tickers'].tolist()
        embed_matrix = data['embeddings']
        
        embed_dict = {ticker: embed_matrix[i] for i, ticker in enumerate(ticker_list)}
        
        result = {}
        for t in tickers:
            vec = embed_dict.get(t)
            if vec is None:
                result[t] = np.zeros(64, dtype=np.float32)
            else:
                vec = np.asarray(vec, dtype=np.float32).flatten()
                if vec.shape[0] != 64:
                    vec = np.zeros(64, dtype=np.float32)
                result[t] = vec
        return result
    except Exception as e:
        print(f"⚠️ Error loading Node2Vec for year {year}: {e}. Using zeros.")
        return {t: np.zeros(64, dtype=np.float32) for t in tickers}


# ============================================================================
# 3. بارگذاری ویژگی‌های از پیش محاسبه‌شده‌ی MAC (از فایل‌های .parquet)
# ============================================================================

def load_mac_features(date, split="train", mac_dir="mac_features"):
    """
    بارگذاری ویژگی‌های MAC برای یک تاریخ مشخص از فایل‌های پارکت.
    فایل‌ها به‌صورت سالانه ذخیره شده‌اند.
    اگر فایل یا تاریخ وجود نداشت، State پیش‌فرض برمی‌گرداند.
    """
    year = date.year
    mac_file = os.path.join(mac_dir, f"mac_features_{split}_{year}.parquet")
    
    # اگر فایل وجود نداشت، State پیش‌فرض برگردان
    if not os.path.exists(mac_file):
        return _get_default_mac_state()
    
    try:
        df = pd.read_parquet(mac_file)
    except Exception as e:
        print(f"⚠️ Error reading MAC file {mac_file}: {e}")
        return _get_default_mac_state()
    
    # تشخیص ستون تاریخ
    date_col = None
    for col in df.columns:
        if col.lower() in ['date', 'dates', 'day']:
            date_col = col
            break
    
    if date_col is None:
        return _get_default_mac_state()
    
    # تبدیل به date
    df[date_col] = pd.to_datetime(df[date_col]).dt.date
    
    # فیلتر کردن برای تاریخ مورد نظر
    row = df[df[date_col] == date]
    if row.empty:
        return _get_default_mac_state()
    
    features = row.iloc[0]
    
    # تشخیص تعداد بخش‌ها
    sector_cols = [c for c in df.columns if c != date_col]
    n_sectors = 0
    for c in sector_cols:
        if c.startswith('sector_ret_'):
            n_sectors = max(n_sectors, int(c.split('_')[-1]) + 1)
    
    if n_sectors == 0:
        corr_cols = [c for c in sector_cols if c.startswith('corr_')]
        if corr_cols:
            n_sectors = int((1 + np.sqrt(1 + 8*len(corr_cols))) / 2)
        else:
            n_sectors = 4
    
    # استخراج داده‌ها
    sector_returns = np.array([features.get(f'sector_ret_{i}', 0.0) for i in range(n_sectors)], dtype=np.float32)
    sector_vols = np.array([features.get(f'sector_vol_{i}', 0.01) for i in range(n_sectors)], dtype=np.float32)
    
    corr_matrix = np.eye(n_sectors, dtype=np.float32)
    idx = 0
    for i in range(n_sectors):
        for j in range(i+1, n_sectors):
            val = features.get(f'corr_{i}_{j}', 0.0)
            corr_matrix[i, j] = val
            corr_matrix[j, i] = val
    
    confidences = np.array([features.get(f'confidence_{i}', 0.5) for i in range(n_sectors)], dtype=np.float32)
    
    return {
        'sector_returns': sector_returns,
        'sector_vols': sector_vols,
        'corr_matrix': corr_matrix,
        'confidences': confidences,
        'vix': float(features.get('vix', 20.0)),
        'market_return': float(features.get('market_return', 0.0)),
        'risk_free': float(features.get('risk_free', 0.03)),
        'n_sectors': n_sectors
    }


def _get_default_mac_state(n_sectors=4):
    """بازگرداندن State پیش‌فرض برای MAC با تعداد بخش‌های مشخص"""
    return {
        'sector_returns': np.zeros(n_sectors, dtype=np.float32),
        'sector_vols': np.ones(n_sectors, dtype=np.float32) * 0.01,
        'corr_matrix': np.eye(n_sectors, dtype=np.float32),
        'confidences': np.ones(n_sectors, dtype=np.float32) * 0.5,
        'vix': 20.0,
        'market_return': 0.0,
        'risk_free': 0.03,
        'n_sectors': n_sectors
    }


def build_mac_state(date, split="train", mac_dir="mac_features", n_sectors=None):
    """
    ساخت بردار State برای MAC از فایل‌های از پیش محاسبه‌شده.
    اگر n_sectors مشخص شده باشد، تعداد بخش‌ها به آن مقدار محدود می‌شود.
    """
    mac_data = load_mac_features(date, split, mac_dir)
    actual_n = mac_data.get('n_sectors', len(mac_data['sector_returns']))
    
    # اگر n_sectors مشخص شده بود، ابعاد را تنظیم کن
    if n_sectors is not None:
        if actual_n > n_sectors:
            # برش به تعداد بخش‌های مورد نظر
            mac_data['sector_returns'] = mac_data['sector_returns'][:n_sectors]
            mac_data['sector_vols'] = mac_data['sector_vols'][:n_sectors]
            mac_data['confidences'] = mac_data['confidences'][:n_sectors]
            mac_data['corr_matrix'] = mac_data['corr_matrix'][:n_sectors, :n_sectors]
            mac_data['n_sectors'] = n_sectors
        elif actual_n < n_sectors:
            # padding با مقادیر پیش‌فرض
            pad_len = n_sectors - actual_n
            mac_data['sector_returns'] = np.concatenate([mac_data['sector_returns'], np.zeros(pad_len, dtype=np.float32)])
            mac_data['sector_vols'] = np.concatenate([mac_data['sector_vols'], np.ones(pad_len, dtype=np.float32) * 0.01])
            mac_data['confidences'] = np.concatenate([mac_data['confidences'], np.ones(pad_len, dtype=np.float32) * 0.5])
            # ماتریس همبستگی
            new_corr = np.eye(n_sectors, dtype=np.float32)
            new_corr[:actual_n, :actual_n] = mac_data['corr_matrix']
            mac_data['corr_matrix'] = new_corr
            mac_data['n_sectors'] = n_sectors
    else:
        n_sectors = actual_n
    
    # تلفیق همه‌ی مؤلفه‌ها
    state = np.concatenate([
        mac_data['sector_returns'],
        mac_data['sector_vols'],
        mac_data['corr_matrix'].flatten(),
        np.array([mac_data['vix'], mac_data['market_return'], mac_data['risk_free']], dtype=np.float32),
        mac_data['confidences']
    ])
    return state.astype(np.float32)


# ============================================================================
# 4. نرمال‌سازی (Z-Score) - برای استفاده در build_sector_state
# ============================================================================

_NORM_STATS = None

def load_normalization_stats(mac_dir="mac_features"):
    """بارگذاری آمار نرمال‌سازی از فایل ذخیره‌شده"""
    global _NORM_STATS
    if _NORM_STATS is not None:
        return _NORM_STATS
    
    stats_path = os.path.join(mac_dir, "normalization_stats.csv")
    if not os.path.exists(stats_path):
        print("⚠️ Normalization stats not found. Using raw values without normalization.")
        return None
    
    df = pd.read_csv(stats_path, index_col=0)
    _NORM_STATS = df.to_dict(orient='index')
    return _NORM_STATS


def normalize_feature(value, feature_name, stats):
    """نرمال‌سازی Z-Score یک مقدار"""
    if stats is None or feature_name not in stats:
        return value
    mean = stats[feature_name]['mean']
    std = stats[feature_name]['std']
    if std < 1e-8:
        return 0.0
    return (value - mean) / std


# ============================================================================
# 5. ساخت State برای عامل‌های بخشی (Sector Agents)
# ============================================================================

def build_sector_state(date, sector_tickers, features_df, vix_series, args):
    """
    ساخت بردار State برای یک عامل بخشی خاص.
    ترکیبی از:
      - داده‌های بازار (مومنتوم، نوسان، حجم) برای هر سهم
      - بردارهای Node2Vec (ساختار گراف)
      - زمینه‌ی کلی بازار (VIX، نرخ بهره، بازده شاخص)
    """
    # 1. داده‌های بازار (قبلاً نرمال‌سازی شده‌اند)
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
    
    # 2. Node2Vec Embeddings
    node2vec_dict = load_node2vec_embeddings(sector_tickers, date)
    node2vec_vectors = []
    for t in sector_tickers:
        vec = node2vec_dict.get(t)
        if vec is None:
            vec = np.zeros(64, dtype=np.float32)
        else:
            vec = np.asarray(vec, dtype=np.float32).flatten()
            if vec.shape[0] != 64:
                vec = np.zeros(64, dtype=np.float32)
        node2vec_vectors.append(vec)
    node2vec_vec = np.concatenate(node2vec_vectors)
    
    # 3. زمینه‌ی کلی بازار (با نرمال‌سازی VIX)
    vix_raw = float(vix_series.loc[date]) if date in vix_series.index else 20.0
    risk_free = float(getattr(args, "risk_free_rate", 0.03))
    
    # نرمال‌سازی VIX
    stats = load_normalization_stats("mac_features")
    if stats and 'vix' in stats:
        vix_norm = normalize_feature(vix_raw, 'vix', stats)
    else:
        vix_norm = vix_raw / 30.0  # fallback ساده
    
    market_return = 0.0  # می‌توانید از benchmark واقعی استفاده کنید
    global_context = np.array([vix_norm, market_return, risk_free], dtype=np.float32)
    
    # 4. تلفیق
    market_data_array = np.array(market_data, dtype=np.float32)
    state = np.concatenate([market_data_array, node2vec_vec, global_context])
    return state.astype(np.float32)


# ============================================================================
# 6. توابع کمکی برای نگاشت بخش‌ها
# ============================================================================

def get_sector_map_from_file(sector_map_path="sp500_panel_data/sp500_sectors_map.csv"):
    """بارگذاری نگاشت بخش‌ها از فایل CSV"""
    if not os.path.exists(sector_map_path):
        raise FileNotFoundError(f"Sector map file not found: {sector_map_path}")
    df = pd.read_csv(sector_map_path)
    sector_map = {}
    for _, row in df.iterrows():
        sector = row['GICS Sector']
        ticker = row['Symbol']
        sector_map.setdefault(sector, []).append(ticker)
    return sector_map


def get_sector_map_from_basket(basket_tickers, sector_map_full):
    """فیلتر کردن نگاشت بخش‌ها بر اساس سبد انتخابی"""
    filtered_map = {}
    for sector, tickers in sector_map_full.items():
        filtered = [t for t in tickers if t in basket_tickers]
        if filtered:
            filtered_map[sector] = filtered
    return filtered_map
