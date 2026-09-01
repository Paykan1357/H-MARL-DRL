


"""
compute_node2vec.py

This script computes Node2Vec embeddings for each year (or period) from the
existing graph edge files and saves them as numpy arrays for later use.
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

GRAPHS_DIR = "sp500_graphs_fast"
EMBEDDINGS_DIR = "embeddings"
SPLIT = "train"          # "train", "val", "test"
YEARS = list(range(2013, 2021))  # برای آموزش

# پارامترهای Node2Vec
DIMENSIONS = 64
WALK_LENGTH = 30
NUM_WALKS = 200
WINDOW = 10
MIN_COUNT = 1
WORKERS = 4

# ============================================================================
# 2. FUNCTIONS
# ============================================================================

def load_graph_for_date(date_str):
    """بارگذاری گراف یک روز خاص از فایل لبه‌ها"""
    edge_path = os.path.join(GRAPHS_DIR, SPLIT, f"edges_{date_str}.csv")
    if not os.path.exists(edge_path):
        return None
    edge_df = pd.read_csv(edge_path)
    if edge_df.empty:
        return None
    G = nx.from_pandas_edgelist(edge_df, 'Source', 'Target', 'Weight')
    return G

def compute_node2vec_for_graph(G):
    """محاسبه‌ی Embedding با Node2Vec برای یک گراف"""
    if G is None or G.number_of_nodes() < 5:
        return None
    node2vec = Node2Vec(
        G,
        dimensions=DIMENSIONS,
        walk_length=WALK_LENGTH,
        num_walks=NUM_WALKS,
        workers=WORKERS
    )
    model = node2vec.fit(window=WINDOW, min_count=MIN_COUNT)
    # استخراج Embedding برای هر گره
    embeddings = {node: model.wv[node] for node in G.nodes()}
    return embeddings

def compute_embeddings_for_year(year):
    """محاسبه‌ی Embedding برای یک سال (با میانگین گرفتن از تمام روزهای آن سال)"""
    # پیدا کردن تمام فایل‌های گراف مربوط به سال
    graph_dir = os.path.join(GRAPHS_DIR, SPLIT)
    all_files = [f for f in os.listdir(graph_dir) if f.startswith("edges_") and f.endswith(".csv")]
    year_files = []
    for f in all_files:
        date_str = f.replace("edges_", "").replace(".csv", "")
        if date_str.startswith(str(year)):
            year_files.append(date_str)
    
    if not year_files:
        print(f"⚠️ No graph files found for year {year}.")
        return None
    
    print(f"   Processing year {year} with {len(year_files)} days...")
    
    # جمع‌آوری تمام لبه‌های سال (برای ساخت یک گراف تجمیعی)
    all_edges = []
    for date_str in tqdm(year_files, desc=f"Loading graphs for {year}"):
        edge_path = os.path.join(graph_dir, f"edges_{date_str}.csv")
        edge_df = pd.read_csv(edge_path)
        if not edge_df.empty:
            all_edges.append(edge_df)
    
    if not all_edges:
        return None
    
    # ترکیب تمام لبه‌ها (میانگین وزنی)
    combined_df = pd.concat(all_edges, ignore_index=True)
    # میانگین گرفتن از وزن لبه‌ها برای هر جفت (Source, Target)
    combined_df = combined_df.groupby(['Source', 'Target'], as_index=False)['Weight'].mean()
    
    # ساخت گراف
    G = nx.from_pandas_edgelist(combined_df, 'Source', 'Target', 'Weight')
    if G.number_of_nodes() < 5:
        return None
    
    # محاسبه‌ی Node2Vec
    embeddings = compute_node2vec_for_graph(G)
    return embeddings

def save_embeddings(embeddings, year):
    """ذخیره‌ی Embedding در فایل numpy"""
    if embeddings is None:
        return
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    # تبدیل به ماتریس (برای ذخیره‌سازی آسان)
    tickers = sorted(embeddings.keys())
    embedding_matrix = np.array([embeddings[t] for t in tickers])
    # ذخیره به‌صورت دوتایی (numpy)
    np.savez_compressed(
        os.path.join(EMBEDDINGS_DIR, f"embeddings_{SPLIT}_{year}.npz"),
        tickers=tickers,
        embeddings=embedding_matrix
    )
    print(f"✅ Saved embeddings for year {year} with {len(tickers)} stocks.")

# ============================================================================
# 3. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("📊 COMPUTING NODE2VEC EMBEDDINGS (YEARLY AGGREGATED)")
    print("="*70)
    print(f"📁 Graph dir: {GRAPHS_DIR}/{SPLIT}")
    print(f"📁 Output dir: {EMBEDDINGS_DIR}")
    print(f"📅 Years: {YEARS}")
    print(f"🔧 Node2Vec params: dim={DIMENSIONS}, walk_len={WALK_LENGTH}, walks={NUM_WALKS}")
    print("="*70)
    
    for year in YEARS:
        print(f"\n📅 Computing for year {year}...")
        embeddings = compute_embeddings_for_year(year)
        if embeddings:
            save_embeddings(embeddings, year)
        else:
            print(f"   ❌ No embeddings for year {year}.")
    
    print("\n" + "="*70)
    print("🎉 Node2Vec embeddings computed and saved successfully!")
    print(f"📂 Files saved in: {EMBEDDINGS_DIR}")
    print("="*70)
