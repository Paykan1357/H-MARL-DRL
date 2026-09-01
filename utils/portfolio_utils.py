
import numpy as np
import scipy.optimize as spo

EPS = 1e-8


def compute_turnover(weights_old, weights_new):
    """Compute turnover as sum of absolute weight changes."""
    return np.sum(np.abs(weights_new - weights_old))


def compute_sharpe_ratio(returns, risk_free_rate=0.0):
    """Compute annualized Sharpe ratio from a series of daily returns."""
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free_rate / 252
    return np.mean(excess) / (np.std(excess) + EPS) * np.sqrt(252)


def compute_max_drawdown(portfolio_values):
    """Compute maximum drawdown from a series of portfolio values."""
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / (peak + EPS)
    return np.max(drawdown)


def compute_sector_returns(weights_sector, returns_sector):
    """Compute weighted return of a sector's sub‑portfolio."""
    return np.dot(weights_sector, returns_sector)


def compute_portfolio_return(weights_all, returns_all, cash_return):
    """
    Compute total portfolio return including cash.
    weights_all: list/array of weights for [cash, stock1, stock2, ...]
    returns_all: list/array of returns for [cash_return, stock1_ret, stock2_ret, ...]
    """
    return np.dot(weights_all, returns_all)


# ============================================================
# Legacy MPT functions (kept for reference, may be removed)
# ============================================================
def get_opt_portfolio(state, objective, trans_coef, to_logret=True):
    # … (keep as is, but it’s not used in the new MARL pipeline)
    pass


def get_sharpe_ratio(weights_new, weights_old, rets, trans_coef):
    # … (kept for compatibility)
    pass
