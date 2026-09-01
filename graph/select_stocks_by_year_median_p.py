


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
OUTPUT_DIR = "selected_stocks_by_year"

TRAIN_YEARS = list(range(2013, 2023))

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

def select_stocks_per_year(split='train', years=TRAIN_YEARS, output_dir=OUTPUT_DIR, top_k=20):
    print(f"\n🚀 Processing {split.upper()} data by year (episodes)...")
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Load Features: Train + Val ---
    features_path = os.path.join(DATA_DIR, f"{split}_features.csv")
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
        print(f"   ✅ Added val features: {len(val_features)} rows")
    else:
        print("   ⚠️ val_features.csv not found.")

    # --- 2. Load VIX: Train + Val ---
    vix_path = os.path.join(DATA_DIR, f"vix_{split}.csv")
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
        print(f"   ✅ Added val VIX: {len(val_vix_series)} days")

    vix_max = vix_series.max()
    print(f"   📊 VIX max: {vix_max:.2f}")

    # --- 3. Graph directories: Train + Val ---
    graphs_dir = os.path.join(GRAPHS_DIR, split)
    if not os.path.exists(graphs_dir):
        print(f"❌ Graphs not found: {graphs_dir}")
        return

    search_dirs = [graphs_dir]
    if split == "train":
        val_dir = os.path.join(GRAPHS_DIR, "val")
        if os.path.exists(val_dir):
            search_dirs.append(val_dir)
            print(f"   🔍 Also searching in val directory for 2021-2022...")

    edge_files = []
    for dir_path in search_dirs:
        for f in os.listdir(dir_path):
            if f.startswith('edges_') and f.endswith('.csv'):
                edge_files.append((dir_path, f))

    edge_files = sorted(edge_files, key=lambda x: x[1])
    print(f"   📁 Found {len(edge_files)} graph files")

    # Group files by year
    years_dict = {}
    for dir_path, f in edge_files:
        date_str = f.replace('edges_', '').replace('.csv', '')
        try:
            if '-' in date_str:
                dt = pd.to_datetime(date_str)
            else:
                dt = pd.to_datetime(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}")
            year = dt.year
            years_dict.setdefault(year, []).append((dir_path, f, date_str))
        except Exception:
            continue

    selected_records = []

    for year in sorted(years_dict.keys()):
        if year not in years:
            continue
        print(f"\n📅 Processing year {year}...")
        year_files = years_dict[year]
        print(f"   Days in {year}: {len(year_files)}")

        psi_per_ticker = {}
        processed_days = 0
        skipped_days = 0

        for idx, (dir_path, edge_file, date_str) in enumerate(year_files):
            try:
                current_date = pd.to_datetime(date_str).date()

                # Load edge list
                edge_path = os.path.join(dir_path, edge_file)
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
                    psi_per_ticker.setdefault(ticker, []).append(psi_val)

                processed_days += 1

            except Exception as e:
                skipped_days += 1
                print(f"\n   ⚠️ Error on {date_str}: {e}")
                continue

            if (idx + 1) % 10 == 0:
                print(f"\n   Progress: {idx+1}/{len(year_files)} days (✓{processed_days} ✗{skipped_days})", end='')

        print(f"\n   ✅ Completed: ✓{processed_days} ✗{skipped_days}")

        if not psi_per_ticker:
            print(f"   ⚠️ No data for year {year}. Skipping.")
            continue

        median_psi = {ticker: np.median(vals) for ticker, vals in psi_per_ticker.items()}
        sorted_tickers = sorted(median_psi.items(), key=lambda x: x[1], reverse=True)
        selected = [ticker for ticker, _ in sorted_tickers[:top_k]]

        print(f"\n   🏆 Top {top_k} stocks for {year}:")
        for i, (ticker, score) in enumerate(sorted_tickers[:top_k], 1):
            print(f"      {i:2d}. {ticker:6s} (Score: {score:.6f})")

        selected_records.append({
            'Year': year,
            'Selected_Tickers': '|'.join(selected)
        })

    if selected_records:
        out_df = pd.DataFrame(selected_records)
        out_path = os.path.join(output_dir, f"selected_stocks_{split}_by_year.csv")
        out_df.to_csv(out_path, index=False)
        print(f"\n✅ Saved selected stocks per year to: {out_path}")
    else:
        print("\n⚠️ No records saved.")

# ============================================================================
# 4. EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ENSEMBLE CENTRALITY – DEGREE-ONLY (FIXED FOR 2021-2022)")
    print("=" * 70)
    print(f"📅 Training years: {TRAIN_YEARS}")
    print("=" * 70)

    select_stocks_per_year(split='train', years=TRAIN_YEARS, top_k=20)

    print("\n" + "=" * 70)
    print("🎉 SELECTION COMPLETED SUCCESSFULLY!")
    print(f"📂 Output saved in: {OUTPUT_DIR}")
    print("=" * 70)
