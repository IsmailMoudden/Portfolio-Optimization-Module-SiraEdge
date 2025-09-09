import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from .data_utils import download_prices, ensure_figures_dir, DEFAULT_TICKERS


def risk_contribution(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    portfolio_var = weights.T @ cov @ weights
    marginal_contrib = cov @ weights
    return weights * marginal_contrib / np.sqrt(portfolio_var)


def objective_equal_risk(weights: np.ndarray, cov: np.ndarray) -> float:
    rc = risk_contribution(weights, cov)
    return np.sum((rc - rc.mean()) ** 2)


def risk_parity_weights(cov: np.ndarray) -> np.ndarray:
    n = cov.shape[0]
    x0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    res = minimize(objective_equal_risk, x0=x0, args=(cov,), method="SLSQP", bounds=bounds, constraints=cons)
    if not res.success:
        raise RuntimeError(res.message)
    return res.x


def run_risk_parity(tickers=None, start="2020-01-01", end="2023-12-31") -> pd.Series:
    if tickers is None:
        tickers = DEFAULT_TICKERS
    prices = download_prices(tickers, start, end)
    rets = np.log(prices / prices.shift(1)).dropna()
    cov = np.cov(rets.values.T) * 252
    w = risk_parity_weights(cov)
    weights = pd.Series(w, index=prices.columns)

    fig_dir = ensure_figures_dir()
    plt.figure(figsize=(8, 4))
    weights.sort_values(ascending=False).plot(kind="bar")
    plt.title("Risk Parity — Poids")
    plt.ylabel("Poids")
    out_path = os.path.join(fig_dir, "risk_parity_weights.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return weights


if __name__ == "__main__":
    print(run_risk_parity().round(4))


