import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .data_utils import download_prices, ensure_figures_dir, DEFAULT_TICKERS


def run_black_litterman(tickers=None, start="2020-01-01", end="2023-12-31") -> pd.Series:
    # Lightweight illustrative BL: blend market-implied (equal-cap proxy) with a simple view
    if tickers is None:
        tickers = DEFAULT_TICKERS
    prices = download_prices(tickers, start, end)
    returns = np.log(prices / prices.shift(1)).dropna()
    S = returns.cov().values * 252
    k = len(tickers)
    market_pi = returns.mean().values * 252  # proxy prior
    # View: QQQ outperform SPY by 2%
    P = np.zeros((1, k))
    i_spy = tickers.index("SPY") if "SPY" in tickers else 0
    i_qqq = tickers.index("QQQ") if "QQQ" in tickers else 1
    P[0, i_qqq] = 1.0
    P[0, i_spy] = -1.0
    Q = np.array([0.02])
    tau = 0.05
    # BL posterior (simplified): pi_post = pi + tau S P^T (P tau S P^T + Omega)^{-1} (Q - P pi)
    Omega = np.array([[0.0025]])
    tauS = tau * S
    middle = np.linalg.inv(P @ tauS @ P.T + Omega)
    pi_post = market_pi + (tauS @ P.T @ middle @ (Q - P @ market_pi))
    cov_post = S + tauS - tauS @ P.T @ middle @ P @ tauS

    # Tangency weights proxy
    w = np.linalg.pinv(cov_post) @ pi_post
    w = np.clip(w, 0, None)
    w = w / (w.sum() + 1e-12)
    weights = pd.Series(w, index=tickers)

    fig_dir = ensure_figures_dir()
    plt.figure(figsize=(8, 4))
    weights.sort_values(ascending=False).plot(kind="bar")
    plt.title("Poids Black–Litterman (vue simple)")
    plt.ylabel("Poids")
    out_path = os.path.join(fig_dir, "black_litterman_weights.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return weights


if __name__ == "__main__":
    print(run_black_litterman().round(4))

