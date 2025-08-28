import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_utils import download_prices, ensure_figures_dir, DEFAULT_TICKERS


def random_portfolios(mu: np.ndarray, cov: np.ndarray, n: int = 20000, seed: int = 42):
    rng = np.random.default_rng(seed)
    k = len(mu)
    W = rng.random((n, k))
    W = W / W.sum(axis=1, keepdims=True)
    rets = W @ mu
    vols = np.sqrt((W @ cov * W).sum(axis=1))
    return W, rets, vols


def run_markowitz(tickers=None, start="2020-01-01", end="2023-12-31") -> pd.Series:
    if tickers is None:
        tickers = DEFAULT_TICKERS
    prices = download_prices(tickers, start, end)
    returns = np.log(prices / prices.shift(1)).dropna()
    mu = returns.mean().values * 252
    cov = (returns.cov().values) * 252

    W, rets, vols = random_portfolios(mu, cov)
    sharpe = rets / (vols + 1e-12)
    idx = int(np.nanargmax(sharpe))
    w_best = W[idx]

    # Approximate frontier by binning volatility and taking max return per bin
    bins = np.linspace(vols.min(), vols.max(), 60)
    idxs = np.digitize(vols, bins)
    frontier_x, frontier_y = [], []
    for b in np.unique(idxs):
        m = rets[idxs == b]
        v = vols[idxs == b]
        if len(m) == 0:
            continue
        j = np.argmax(m)
        frontier_x.append(v[j])
        frontier_y.append(m[j])

    fig_dir = ensure_figures_dir()
    plt.figure(figsize=(8, 5))
    plt.scatter(vols, rets, s=5, alpha=0.2, label="Portefeuilles aléatoires")
    if frontier_x:
        order = np.argsort(frontier_x)
        plt.plot(np.array(frontier_x)[order], np.array(frontier_y)[order], c="black", lw=2, label="Frontière approx.")
    plt.scatter(vols[idx], rets[idx], marker="*", s=140, c="red", label="Max Sharpe")
    plt.title("Frontière efficiente (approx.)")
    plt.xlabel("Volatilité annualisée")
    plt.ylabel("Rendement annualisé")
    plt.legend()
    out_path = os.path.join(fig_dir, "markowitz_frontier.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return pd.Series(w_best, index=returns.columns)


if __name__ == "__main__":
    w = run_markowitz()
    print(w[w > 0].sort_values(ascending=False))

