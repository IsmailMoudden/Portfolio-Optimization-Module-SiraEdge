import numpy as np
import pandas as pd


def annualized_return(returns: pd.Series) -> float:
    return float((1 + returns).prod() ** (252 / max(len(returns), 1)) - 1)


def annualized_volatility(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(252))


def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    mu = annualized_return(returns) - rf
    vol = annualized_volatility(returns) + 1e-12
    return float(mu / vol)


def downside_std(returns: pd.Series, threshold: float = 0.0) -> float:
    downside = returns[returns < threshold]
    return float(downside.std(ddof=0) * np.sqrt(252)) if len(downside) else 0.0


def sortino_ratio(returns: pd.Series, rf: float = 0.0, threshold: float = 0.0) -> float:
    mu = annualized_return(returns) - rf
    dstd = downside_std(returns - rf / 252, threshold)
    return float(mu / (dstd + 1e-12))


def max_drawdown(cum: pd.Series) -> float:
    running_max = cum.cummax()
    dd = 1 - cum / (running_max + 1e-12)
    return float(dd.max())


def calmar_ratio(returns: pd.Series) -> float:
    mu = annualized_return(returns)
    mdd = max_drawdown((1 + returns).cumprod()) + 1e-12
    return float(mu / mdd)


def turnover(w_prev: np.ndarray, w_new: np.ndarray) -> float:
    return float(np.abs(w_new - w_prev).sum())


def weight_stability(weights_over_time: np.ndarray) -> float:
    # Higher is more stable: inverse of average absolute change
    if weights_over_time.shape[0] < 2:
        return 1.0
    diffs = np.abs(np.diff(weights_over_time, axis=0))
    avg_change = float(diffs.mean()) + 1e-12
    return float(1.0 / avg_change)



