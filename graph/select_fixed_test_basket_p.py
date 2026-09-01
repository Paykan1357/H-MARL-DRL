

import os
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

DATA_DIR = "sp500_panel_data"
GRAPHS_DIR = "sp500_graphs_fast"
OUTPUT_DIR = "selected_stocks"

TRAIN_YEARS = list(range(2013, 2023))  # 2013 تا 2022 (9 سال)
TOP_K = 20

# ============================================================================
# 2. HELPER FUNCTIONS
# ============================================================================

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

def compute_degree_centrality(edge_df):
    if edge_df.empty:
        return None
    try:
        G = nx.from_pandas_edgelist(edge_df, 'Source', 'Target', 'Weight')
        n_nodes = G.number_of_nodes()
        if n_nodes < 5:
            return None
        deg_centrality = dict(G.degree(weight='Weight'))
        max_deg = max(deg_centrality.values()) if deg_centrality else 1.0
        if max_deg > 0:
            deg_centrality = {node: val / max_deg for node, val in deg_centrality.items()}
        else:
            deg_centrality = {node: 1.0 / n_nodes for node in G.nodes()}
        return deg_centrality
    except Exception:
        return None

def compute_structural_score_degree(deg_centrality):
    if not deg_centrality:
        return {}
    all_tickers = sorted(deg_centrality.keys())
    n = len(all_tickers)
    if n == 0:
        return {}
    values = [deg_centrality.get(t, 0.0) for t in all_tickers]
    sorted_indices = np.argsort(values)[::-1]
    ranks = np.zeros(n)
    for rank, idx in enumerate(sorted_indices, 1):
        ranks[idx] = rank
    if n > 1:
        rank_scores = 1 - (ranks - 1) / (n - 1)
    else:
        rank_scores = np.ones(n)
    S_struct = {ticker: rank_scores[i] for i, ticker in enumerate(all_tickers)}
    return S_struct

def compute_quality_score(features_df):
    Q = {}
    required_cols = ['Ticker', 'Momentum_scaled', 'Volatility_scaled', 'NormVolume_scaled']
    if not all(col in features_df.columns for col in required_cols):
        return {row['Ticker']: 1.0 for _, row in features_df.iterrows()}
    for _, row in features_df.iterrows():
        ticker = row['Ticker']
        mom_score = sigmoid(row['Momentum_scaled'])
        vol_score = 1.0 / sigmoid(row['Volatility_scaled'])
        vol_score = max(0.0, min(vol_score, 10.0))
        norm_vol_score = sigmoid(row['NormVolume_scaled'])
        Q[ticker] = mom_score * vol_score * norm_vol_score
    if Q:
        max_q = max(Q.values())
        min_q = min(Q.values())
        if max_q > min_q:
            Q = {k: (v - min_q) / (max_q - min_q) for k, v in Q.items()}
        else:
            Q = {k: 1.0 for k in Q}
    return Q

# ============================================================================
# 3. MAIN FUNCTION (با ترکیب Train+Val)
# ============================================================================

def select_fixed_test_basket(aggregation='median'):
    print("=" * 70)
    print("📊 SELECTING FIXED BASKET FOR TEST SET (DEGREE-ONLY)")
    print("=" * 70)
    print(f"📅 Training years: {TRAIN_YEARS[0]} to {TRAIN_YEARS[-1]} (expanded window)")
    print(f"📈 Aggregation method: {aggregation.upper()}")
    print(f"📊 Selecting top-{TOP_K} stocks based on aggregate Psi.")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 1. Load Features: Train + Val ---
    features_path = os.path.join(DATA_DIR, "train_features.csv")
    if not os.path.exists(features_path):
        print(f"❌ Features not found: {features_path}")
        return
    features_df = pd.read_csv(features_path, parse_dates=['Date'])
    features_df['Date'] = pd.to_datetime(features_df['Date']).dt.date

    # اضافه کردن val_features برای 2021-2022
    val_features_path = os.path.join(DATA_DIR, "val_features.csv")
    if os.path.exists(val_features_path):
        val_features = pd.read_csv(val_features_path, parse_dates=['Date'])
        val_features['Date'] = pd.to_datetime(val_features['Date']).dt.date
        features_df = pd.concat([features_df, val_features], ignore_index=True)
        print(f"✅ Added val features: {len(val_features)} rows")
    else:
        print("⚠️ val_features.csv not found.")

    print(f"✅ Total features: {len(features_df)} rows, {features_df['Ticker'].nunique()} tickers")

    # --- 2. Load VIX: Train + Val ---
    vix_path = os.path.join(DATA_DIR, "vix_train.csv")
    if not os.path.exists(vix_path):
        print(f"❌ VIX not found: {vix_path}")
        return
    vix_df = pd.read_csv(vix_path, parse_dates=['date'])
    vix_df['date'] = pd.to_datetime(vix_df['date']).dt.date
    vix_series = vix_df.set_index('date')['vix_close']
    vix_series = pd.to_numeric(vix_series, errors='coerce').ffill()

    # اضافه کردن val VIX برای 2021-2022
    val_vix_path = os.path.join(DATA_DIR, "vix_val.csv")
    if os.path.exists(val_vix_path):
        val_vix = pd.read_csv(val_vix_path, parse_dates=['date'])
        val_vix['date'] = pd.to_datetime(val_vix['date']).dt.date
        val_vix_series = val_vix.set_index('date')['vix_close']
        val_vix_series = pd.to_numeric(val_vix_series, errors='coerce').ffill()
        vix_series = pd.concat([vix_series, val_vix_series])
        vix_series = vix_series[~vix_series.index.duplicated(keep='first')]
        print(f"✅ Added val VIX: {len(val_vix_series)} days")

    vix_max = vix_series.max()
    print(f"✅ VIX loaded. Max VIX: {vix_max:.2f}")

    # --- 3. Graph directory ---
    graphs_dir = os.path.join(GRAPHS_DIR, "train")
    if not os.path.exists(graphs_dir):
        print(f"❌ Graphs not found: {graphs_dir}")
        return

    edge_files = sorted([f for f in os.listdir(graphs_dir) if f.startswith('edges_') and f.endswith('.csv')])
    print(f"✅ Found {len(edge_files)} graph files")

    # Filter only files within the TRAIN_YEARS (2013-2022)
    filtered_edge_files = []
    for f in edge_files:
        date_str = f.replace('edges_', '').replace('.csv', '')
        try:
            if '-' in date_str:
                year = int(date_str[:4])
            else:
                year = int(date_str[:4])
            if year in TRAIN_YEARS:
                filtered_edge_files.append((date_str, f))
        except Exception:
            continue
    print(f"✅ Filtered to {len(filtered_edge_files)} files within training years.")

    if not filtered_edge_files:
        print("❌ No graph files found for the specified training years.")
        return

    # --- 4. Process all days and accumulate Psi per ticker ---
    psi_accumulator = {}
    processed_days = 0
    skipped_days = 0

    for idx, (date_str, edge_file) in enumerate(filtered_edge_files):
        try:
            current_date = pd.to_datetime(date_str).date()
            edge_path = os.path.join(graphs_dir, edge_file)
            edge_df = pd.read_csv(edge_path)
            if edge_df.empty:
                skipped_days += 1
                continue

            deg_cent = compute_degree_centrality(edge_df)
            if deg_cent is None:
                skipped_days += 1
                continue

            S_struct = compute_structural_score_degree(deg_cent)
            if not S_struct:
                skipped_days += 1
                continue

            # Quality score for this date
            date_features = features_df[features_df['Date'] == current_date]
            if date_features.empty:
                skipped_days += 1
                continue
            Q = compute_quality_score(date_features)
            if not Q:
                Q = {ticker: 1.0 for ticker in S_struct.keys()}

            for ticker in set(S_struct.keys()) & set(Q.keys()):
                psi_val = S_struct[ticker] * Q[ticker]
                if ticker not in psi_accumulator:
                    psi_accumulator[ticker] = []
                psi_accumulator[ticker].append(psi_val)

            processed_days += 1

        except Exception:
            skipped_days += 1
            continue

        if (idx + 1) % 50 == 0:
            print(f"\n   Progress: {idx+1}/{len(filtered_edge_files)} days (✓{processed_days} ✗{skipped_days})", end='')

    print(f"\n\n✅ Completed: ✓{processed_days} ✗{skipped_days}")
    print(f"✅ Accumulated Psi for {len(psi_accumulator)} unique tickers.")

    if not psi_accumulator:
        print("❌ No Psi values accumulated. Exiting.")
        return

    # --- 5. Compute aggregate score per ticker ---
    aggregate_scores = {}
    for ticker, values in psi_accumulator.items():
        if aggregation == 'median':
            aggregate_scores[ticker] = np.median(values)
        elif aggregation == 'mean':
            aggregate_scores[ticker] = np.mean(values)
        else:
            raise ValueError("aggregation must be 'median' or 'mean'")

    # --- 6. Sort and select top-k ---
    sorted_tickers = sorted(aggregate_scores.items(), key=lambda x: x[1], reverse=True)
    selected = sorted_tickers[:TOP_K]

    print(f"\n🏆 Top {TOP_K} stocks for the fixed test basket (based on {aggregation} of Psi):")
    for rank, (ticker, score) in enumerate(selected, 1):
        print(f"   {rank:2d}. {ticker:6s}  (Score: {score:.6f})")

    selected_tickers = [ticker for ticker, _ in selected]

    # --- 7. Save to CSV ---
    output_df = pd.DataFrame({
        'Ticker': selected_tickers,
        f'Aggregate_Psi_{aggregation}': [score for _, score in selected]
    })

    summary_df = pd.DataFrame({
        'Basket_Type': ['Fixed_Test_Basket'],
        'Aggregation': [aggregation],
        'Training_Years': [f"{TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}"],
        'Selected_Tickers': ['|'.join(selected_tickers)],
        'Num_Stocks': [TOP_K]
    })

    detailed_path = os.path.join(OUTPUT_DIR, "fixed_test_basket_detailed_scores.csv")
    output_df.to_csv(detailed_path, index=False)
    print(f"\n✅ Detailed scores saved to: {detailed_path}")

    summary_path = os.path.join(OUTPUT_DIR, "fixed_test_basket_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"✅ Summary saved to: {summary_path}")

    print("\n" + "=" * 70)
    print("🎉 FIXED TEST BASKET SELECTION COMPLETED!")
    print(f"📅 Basket selected based on training years: {TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}")
    print("=" * 70)

# ============================================================================
# 4. EXECUTION
# ============================================================================

if __name__ == "__main__":
    select_fixed_test_basket(aggregation='median')
