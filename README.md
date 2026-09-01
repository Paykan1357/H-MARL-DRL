# H-MARL: Hierarchical Multi-Agent Reinforcement Learning for Portfolio Optimization

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

This repository contains the official implementation of the paper:

> **"H-MARL: A Hierarchical Multi-Agent Reinforcement Learning Framework for Dynamic Portfolio Optimization with Graph-Based Stock Selection"**  
> 
---

## 📖 Abstract

We propose a novel hierarchical multi-agent reinforcement learning (MARL) framework for dynamic portfolio optimization. The framework integrates:

- **Regime-adaptive graph construction** using wavelet coherence with VIX‑modulated fusion weights (short‑, medium‑, and long‑term frequency bands).
- **Ensemble centrality‑based stock selection** (Degree, Eigenvector, Betweenness, PageRank) with VIX‑adaptive weighting to select a structurally diversified universe of 20 stocks.
- **Node2Vec structural embeddings** (64‑dim) that capture each stock's topological role (hub, bridge, peripheral node) and augment the state space of lower‑tier agents.
- **Two‑tier hierarchical MARL architecture**: lower‑tier sector‑specialist agents trained with **PPO** managing intra‑sector allocation, and an upper‑tier **Meta‑Adaptive Controller (MAC)** trained with **SAC** that dynamically orchestrates inter‑sector weights and cash allocation.

The model achieves a cumulative return of **56.68%** (Sharpe ratio **1.30**, max drawdown **‑11.05%**) on a fixed 20‑stock universe over the out‑of‑sample period 2023–2025, substantially outperforming the equal‑weight (1/N) baseline (**15.45%** return, Sharpe **1.17**) by **41.23 percentage points**. The framework also demonstrates robustness to transaction costs (Sharpe **1.41** at 0.5% cost) and effective risk reduction during the 2023 SVB banking crisis (volatility reduced by **12.3%** vs. S&P 500).

---

## 📁 Repository Structure
