import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from data_utils import download_prices, ensure_figures_dir, DEFAULT_TICKERS


def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    mu = returns.mean() * 252 - rf
    vol = returns.std() * np.sqrt(252)
    return float(mu / (vol + 1e-8))


def max_drawdown(cum_returns: pd.Series) -> float:
    rolling_max = cum_returns.cummax()
    drawdowns = 1 - cum_returns / (rolling_max + 1e-12)
    return float(drawdowns.max())


def stability_score(returns: pd.Series) -> float:
    # 1 / variance des variations de rendement (proxy de stabilité)
    diffs = returns.diff().dropna()
    return float(1.0 / (np.var(diffs) + 1e-8))


def objective(weights: np.ndarray, returns: pd.DataFrame, lam: float, gamma: float) -> float:
    weights = np.maximum(weights, 0)
    weights /= weights.sum() + 1e-12
    port_rets = returns @ weights
    sharpe = sharpe_ratio(port_rets)
    cum = (1 + port_rets).cumprod()
    mdd = max_drawdown(cum)
    stab = stability_score(port_rets)
    score = sharpe - lam * mdd + gamma * stab
    return -score


def run_custom_optimization(tickers=None, start="2020-01-01", end="2023-12-31", lam=2.0, gamma=0.01) -> pd.Series:
    if tickers is None:
        tickers = DEFAULT_TICKERS
    prices = download_prices(tickers, start, end)
    rets = prices.pct_change().dropna()
    n = rets.shape[1]
    x0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    res = minimize(objective, x0=x0, args=(rets, lam, gamma), method="SLSQP", bounds=bounds, constraints=cons)
    if not res.success:
        raise RuntimeError(res.message)
    w = np.maximum(res.x, 0)
    w /= w.sum()
    weights = pd.Series(w, index=rets.columns)
    fig_dir = ensure_figures_dir()
    plt.figure(figsize=(8, 4))
    weights.sort_values(ascending=False).plot(kind="bar")
    plt.title("Poids — Optimisation personnalisée (Sharpe - lambda*Drawdown + gamma*Stabilite)")
    plt.ylabel("Poids")
    out_path = os.path.join(fig_dir, "custom_opt_weights.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return weights


if __name__ == "__main__":
    print(run_custom_optimization().round(4))

