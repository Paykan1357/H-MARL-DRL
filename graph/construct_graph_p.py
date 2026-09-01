

import os
import numpy as np
import pandas as pd
import pywt
from scipy.signal import convolve2d
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. WAVELET COHERENCE HELPER FUNCTIONS (Optimized)
# ============================================================================

def compute_cwt(signal, scales, wavelet='cmor1.5-1.0', dt=1):
    """Compute Continuous Wavelet Transform for a signal."""
    if len(signal) < 10:
        return None
    signal = np.nan_to_num(signal)
    if np.std(signal) < 1e-8:
        return None
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    coefs, _ = pywt.cwt(signal, scales, wavelet, dt)
    return coefs


def compute_wavelet_coherence_from_cwt(Wx, Wy, scales, time_smooth_window=5, scale_smooth_window=3):
    """Compute coherence |R|^2 from two CWT matrices."""
    if Wx is None or Wy is None:
        return None
    
    # Cross-wavelet spectrum
    Wxy = Wx * np.conj(Wy)
    
    # Smoothing kernels
    kernel_time = np.ones((1, time_smooth_window)) / time_smooth_window
    kernel_scale = np.ones((scale_smooth_window, 1)) / scale_smooth_window
    
    smooth_Wxy = convolve2d(np.abs(Wxy)**2, kernel_time, mode='same')
    smooth_Wxy = convolve2d(smooth_Wxy, kernel_scale, mode='same')
    
    smooth_Wx = convolve2d(np.abs(Wx)**2, kernel_time, mode='same')
    smooth_Wx = convolve2d(smooth_Wx, kernel_scale, mode='same')
    
    smooth_Wy = convolve2d(np.abs(Wy)**2, kernel_time, mode='same')
    smooth_Wy = convolve2d(smooth_Wy, kernel_scale, mode='same')
    
    denominator = smooth_Wx * smooth_Wy
    denominator[denominator < 1e-10] = 1e-10
    coherence = smooth_Wxy / denominator
    
    return np.mean(coherence, axis=1)


# ============================================================================
# 2. GRAPH CONSTRUCTION FOR A SINGLE DATE (PARALLEL PER STOCK)
# ============================================================================

def build_graph_for_date_parallel(target_date, price_matrix, sector_map, vix_series,
                                  scales, K=10, gamma=0.25, mu=20, k=0.5, n_jobs=-1):
    """
    Parallel version: for each stock, compute its edges independently.
    Returns: (edge_df, adj_matrix) or (None, None).
    """
    # --- 1. Get rolling window ---
    try:
        end_idx = price_matrix.index.get_loc(target_date)
    except KeyError:
        return None, None
    
    start_idx = max(0, end_idx - 126)
    window_prices = price_matrix.iloc[start_idx:end_idx+1]
    
    if len(window_prices) < 50:
        return None, None
    
    # Drop stocks with too many NaNs
    valid_stocks = window_prices.columns[window_prices.isna().sum() < 10].tolist()
    if len(valid_stocks) < 10:
        return None, None
    
    window_prices = window_prices[valid_stocks]
    n_stocks = len(valid_stocks)
    
    # --- 2. Get VIX safely ---
    try:
        vix_t = vix_series.loc[target_date]
        if pd.isna(vix_t) or not isinstance(vix_t, (float, int)):
            if target_date in vix_series.index:
                vix_t = vix_series.ffill().loc[target_date]
            else:
                available = vix_series.index[vix_series.index <= target_date]
                vix_t = vix_series.loc[available[-1]] if len(available) > 0 else 20.0
        vix_t = float(vix_t)
    except (KeyError, TypeError):
        vix_t = vix_series.ffill().iloc[-1] if len(vix_series) > 0 else 20.0
        vix_t = float(vix_t)
    
    vix_t = max(vix_t, 10.0)
    
    # --- 3. VIX-adaptive weights ---
    alpha_ST = 1 / (1 + np.exp(k * (vix_t - mu)))
    alpha_LT = 1 - alpha_ST
    alpha_MT = 0.5
    
    # --- 4. Pre-compute CWT for each stock (cache) ---
    cwt_cache = {}
    for ticker in valid_stocks:
        ts = window_prices[ticker].values
        if len(ts) >= 50:
            cwt_cache[ticker] = compute_cwt(ts, scales)
    
    if len(cwt_cache) < 5:
        return None, None
    
    # --- 5. Define function to compute edges for a single stock ---
    def compute_edges_for_stock(i):
        ticker_i = valid_stocks[i]
        if ticker_i not in cwt_cache or cwt_cache[ticker_i] is None:
            return []
        
        W_i = cwt_cache[ticker_i]
        stock_edges = []
        
        for j in range(i+1, n_stocks):
            ticker_j = valid_stocks[j]
            if ticker_j not in cwt_cache or cwt_cache[ticker_j] is None:
                continue
            
            W_j = cwt_cache[ticker_j]
            coherence_scales = compute_wavelet_coherence_from_cwt(W_i, W_j, scales)
            if coherence_scales is None:
                continue
            
            # Extract bands
            st_idx = (scales >= 2) & (scales <= 8)
            mt_idx = (scales > 8) & (scales <= 32)
            lt_idx = (scales > 32) & (scales <= 128)
            
            C_ST = np.nanmean(coherence_scales[st_idx]) if np.any(st_idx) else 0.0
            C_MT = np.nanmean(coherence_scales[mt_idx]) if np.any(mt_idx) else 0.0
            C_LT = np.nanmean(coherence_scales[lt_idx]) if np.any(lt_idx) else 0.0
            
            C_ST = np.clip(C_ST, 0, 1)
            C_MT = np.clip(C_MT, 0, 1)
            C_LT = np.clip(C_LT, 0, 1)
            
            # Raw fused weight
            A_tilde = alpha_ST * C_ST + alpha_MT * C_MT + alpha_LT * C_LT
            A_tilde = np.clip(A_tilde, 1e-6, 1.0)
            
            # Sector bonus
            sector_i = sector_map.get(ticker_i, 'Unknown')
            sector_j = sector_map.get(ticker_j, 'Unknown')
            delta = 1 if sector_i == sector_j and sector_i != 'Unknown' else 0
            weight = A_tilde * (1 + gamma * delta)
            
            stock_edges.append((ticker_i, ticker_j, weight))
        
        return stock_edges
    
    # --- 6. Parallel execution across stocks ---
    try:
        # Use 'loky' backend which is more robust
        all_edges = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
            delayed(compute_edges_for_stock)(i) for i in range(n_stocks)
        )
    except Exception:
        # Fallback to sequential if parallel fails
        all_edges = []
        for i in range(n_stocks):
            all_edges.append(compute_edges_for_stock(i))
    
    # Flatten the list
    edges = [edge for sublist in all_edges for edge in sublist]
    
    if not edges:
        return None, None
    
    # --- 7. Build adjacency matrix ---
    adj_full = pd.DataFrame(0.0, index=valid_stocks, columns=valid_stocks)
    for src, tgt, w in edges:
        adj_full.loc[src, tgt] = w
        adj_full.loc[tgt, src] = w
    
    # --- 8. Top-K sparsification ---
    adj_sparse = pd.DataFrame(0.0, index=valid_stocks, columns=valid_stocks)
    for stock in valid_stocks:
        row = adj_full.loc[stock]
        top_k = row.nlargest(K)
        for neighbor, w in top_k.items():
            adj_sparse.loc[stock, neighbor] = w
    
    # --- 9. Ensure symmetry ---
    for i in range(n_stocks):
        for j in range(i+1, n_stocks):
            s1 = valid_stocks[i]
            s2 = valid_stocks[j]
            if adj_sparse.loc[s1, s2] > 0 or adj_sparse.loc[s2, s1] > 0:
                avg_w = (adj_sparse.loc[s1, s2] + adj_sparse.loc[s2, s1]) / 2
                adj_sparse.loc[s1, s2] = avg_w
                adj_sparse.loc[s2, s1] = avg_w
    
    # --- 10. Row-normalization ---
    row_sums = adj_sparse.sum(axis=1)
    row_sums = row_sums.replace(0, 1e-6)
    adj_norm = adj_sparse.div(row_sums, axis=0)
    
    # --- 11. Edge list for saving ---
    edge_list = []
    for src in valid_stocks:
        for tgt in valid_stocks:
            w = adj_norm.loc[src, tgt]
            if w > 1e-8 and src != tgt:
                edge_list.append({'Source': src, 'Target': tgt, 'Weight': w})
    
    return pd.DataFrame(edge_list), adj_norm


# ============================================================================
# 3. MAIN PIPELINE WITH SUBSET SELECTION
# ============================================================================

def run_graph_construction_fast(
    data_dir="sp500_panel_data",
    split="train",
    vix_path="sp500_panel_data/vix_train.csv",
    output_dir="graphs",
    top_n=200,          # Set to e.g., 100 for fast testing, None for all stocks
    K=10,
    gamma=0.25,
    mu=20,
    k=0.5,
    n_jobs=-1
):
    """
    Main function with optional subset selection.
    """
    print(f"\n🚀 Building graphs for {split.upper()} set...")
    
    # --- 1. Load data ---
    panel_path = os.path.join(data_dir, f"{split}_panel.csv")
    features_path = os.path.join(data_dir, f"{split}_features.csv")
    
    if not os.path.exists(panel_path) or not os.path.exists(features_path):
        print(f"❌ Data files not found for {split}.")
        return
    
    panel = pd.read_csv(panel_path, parse_dates=['Date'])
    
    # --- 2. Optional: Subset to Top N most liquid stocks ---
    if top_n is not None and top_n > 0:
        print(f"   Selecting top {top_n} stocks by average volume...")
        avg_volume = panel.groupby('Ticker')['Volume'].mean().sort_values(ascending=False)
        top_tickers = avg_volume.head(top_n).index.tolist()
        panel = panel[panel['Ticker'].isin(top_tickers)]
        print(f"   Reduced to {len(top_tickers)} stocks.")
    
    # --- 3. Create price matrix ---
    price_matrix = panel.pivot(index='Date', columns='Ticker', values='Close')
    price_matrix = price_matrix.sort_index()
    print(f"   Price matrix: {price_matrix.shape}")
    
    # --- 4. Load sector map ---
    sector_path = os.path.join(data_dir, "sp500_sectors_map.csv")
    if not os.path.exists(sector_path):
        print(f"❌ Sector map not found.")
        return
    sector_df = pd.read_csv(sector_path)
    sector_map = dict(zip(sector_df['Symbol'], sector_df['GICS Sector']))
    
    # --- 5. Load VIX ---
    if not os.path.exists(vix_path):
        print(f"❌ VIX file not found.")
        return
    vix_df = pd.read_csv(vix_path, parse_dates=['date'])
    vix_df = vix_df.set_index('date')['vix_close']
    vix_df = pd.to_numeric(vix_df, errors='coerce')
    vix_df = vix_df.ffill()
    print(f"   VIX data: {len(vix_df)} days")
    
    # --- 6. Scales ---
    scales = np.geomspace(2, 128, num=25)  # Reduced to 25 for speed
    print(f"   Scales: {len(scales)} (2 to 128 days)")
    
    # --- 7. Output directory ---
    output_dir_split = os.path.join(output_dir, split)
    os.makedirs(output_dir_split, exist_ok=True)
    
    # --- 8. Common dates ---
    price_dates = pd.to_datetime(price_matrix.index).normalize()
    print(f"   Price dates: {len(price_dates)} (from {price_dates[0]} to {price_dates[-1]})")
    
    # Filter to dates with at least 126 days of history
    min_date = price_dates[0] + pd.Timedelta(days=126)
    valid_dates = [d for d in price_dates if d >= min_date]
    print(f"   Valid dates after warmup: {len(valid_dates)}")
    
    if len(valid_dates) == 0:
        print("⚠️ No dates with sufficient history.")
        return
    
        
    stats_list = []
    
    # --- 10. Process each date ---
    for current_date in tqdm(valid_dates, desc="Building graphs"):
        try:
            edge_df, adj_mat = build_graph_for_date_parallel(
                target_date=current_date,
                price_matrix=price_matrix,
                sector_map=sector_map,
                vix_series=vix_df,
                scales=scales,
                K=K,
                gamma=gamma,
                mu=mu,
                k=k,
                n_jobs=n_jobs
            )
            
            if edge_df is None or edge_df.empty:
                continue
            
            # Save edge list
            date_str = current_date.strftime('%Y%m%d')
            edge_df.to_csv(os.path.join(output_dir_split, f"edges_{date_str}.csv"), index=False)
            
            # Statistics
            n_nodes = len(adj_mat.index)
            if n_nodes > 2:
                degrees = adj_mat.sum(axis=1)
                stats_list.append({
                    'Date': current_date,
                    'Num_Stocks': n_nodes,
                    'Num_Edges': len(edge_df),
                    'Avg_Degree': degrees.mean(),
                    'Avg_Edge_Weight': edge_df['Weight'].mean(),
                    'VIX': vix_df.loc[current_date] if current_date in vix_df.index else np.nan
                })
                
        except Exception as e:
            print(f"   Error on {current_date}: {e}")
            continue
    
    # --- 11. Save stats ---
    if stats_list:
        stats_df = pd.DataFrame(stats_list)
        stats_df.to_csv(os.path.join(output_dir_split, "graph_stats.csv"), index=False)
        print(f"\n✅ {split.upper()} graphs saved to: {output_dir_split}")
        print(f"   Processed {len(stats_list)} days.")
        print(f"   Average stocks per day: {stats_df['Num_Stocks'].mean():.0f}")
    else:
        print(f"\n⚠️ No graphs built for {split.upper()}.")


# ============================================================================
# 4. EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("HIGH-PERFORMANCE GRAPH CONSTRUCTION")
    print("=" * 70)
    
    # ===============================
    # CONFIGURATION
    # ===============================
    DATA_DIR = "sp500_panel_data"
    OUTPUT_DIR = "sp500_graphs_fast"
    
    
    TOP_N = 200   
    
    K = 25                 # K=10
    GAMMA = 0.25
    MU = 20
    K_SIGMOID = 0.5
    N_JOBS = -1   # Use all CPU cores
    
    print(f"   🔹 Top N stocks: {TOP_N if TOP_N else 'ALL (503)'}")
    print(f"   🔹 Parallel jobs: {N_JOBS} (all cores)")
    print("=" * 70)
    
    # ===============================
    # RUN FOR ALL SPLITS
    # ===============================
    for split in ["train", "val", "test"]:
        vix_file = os.path.join(DATA_DIR, f"vix_{split}.csv")
        if not os.path.exists(vix_file):
            print(f"⚠️ Skipping {split}: VIX file not found.")
            continue
        
        run_graph_construction_fast(
            data_dir=DATA_DIR,
            split=split,
            vix_path=vix_file,
            output_dir=OUTPUT_DIR,
            top_n=TOP_N,
            K=K,
            gamma=GAMMA,
            mu=MU,
            k=K_SIGMOID,
            n_jobs=N_JOBS
        )
    
    print("\n" + "=" * 70)
    print("🎉 ALL GRAPHS BUILT SUCCESSFULLY!")
    print(f"📂 Output: {OUTPUT_DIR}")
    print("=" * 70)
