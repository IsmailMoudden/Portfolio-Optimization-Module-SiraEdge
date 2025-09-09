import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .data_utils import download_prices, ensure_figures_dir, DEFAULT_TICKERS


def simulate_portfolios(returns: pd.DataFrame, n_portfolios: int = 10000, seed: int = 42):
    rng = np.random.default_rng(seed)
    mu = returns.mean() * 252
    cov = returns.cov() * 252
    means = []
    vols = []
    for _ in range(n_portfolios):
        w = rng.random(len(mu))
        w /= w.sum()
        port_mu = float(np.dot(w, mu))
        port_vol = float(np.sqrt(w.T @ cov.values @ w))
        means.append(port_mu)
        vols.append(port_vol)
    return np.array(means), np.array(vols)


def run_monte_carlo(tickers=None, start="2020-01-01", end="2023-12-31"):
    if tickers is None:
        tickers = DEFAULT_TICKERS
    prices = download_prices(tickers, start, end)
    returns = np.log(prices / prices.shift(1)).dropna()
    means, vols = simulate_portfolios(returns)

    fig_dir = ensure_figures_dir()
    plt.figure(figsize=(8, 5))
    plt.scatter(vols, means, s=6, alpha=0.35)
    plt.xlabel("Volatilité annualisée")
    plt.ylabel("Rendement annualisé")
    plt.title("10,000 portefeuilles aléatoires")
    out_path = os.path.join(fig_dir, "monte_carlo_cloud.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    run_monte_carlo()


