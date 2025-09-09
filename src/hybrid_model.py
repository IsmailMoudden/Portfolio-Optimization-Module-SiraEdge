import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    from .data_utils import download_prices, compute_indicators, ensure_figures_dir, DEFAULT_TICKERS
    from .risk_parity import risk_parity_weights
except ImportError:
    from data_utils import download_prices, compute_indicators, ensure_figures_dir, DEFAULT_TICKERS
    from risk_parity import risk_parity_weights
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def compute_ml_scores(prices: pd.DataFrame) -> pd.Series:
    indicators = compute_indicators(prices)
    scores = {}
    for ticker, feats in indicators.items():
        y = np.log(prices[ticker] / prices[ticker].shift(1)).shift(-1).reindex(feats.index)
        df = feats.join(y.rename("target")).dropna()
        if len(df) < 100:
            continue
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=5.0))
        ])
        pipe.fit(df[["momentum", "volatility", "rsi"]], df["target"])
        scores[ticker] = float(np.maximum(pipe.score(df[["momentum", "volatility", "rsi"]], df["target"]), 0))
    s = pd.Series(scores)
    if s.sum() == 0:
        s = pd.Series(1.0, index=prices.columns)
    return s / s.sum()


def run_hybrid(tickers=None, start="2020-01-01", end="2023-12-31") -> pd.Series:
    if tickers is None:
        tickers = DEFAULT_TICKERS
    prices = download_prices(tickers, start, end)
    rets = np.log(prices / prices.shift(1)).dropna()
    cov = rets.cov().values * 252
    w_rp = risk_parity_weights(cov)
    rp = pd.Series(w_rp, index=rets.columns)
    ml_scores = compute_ml_scores(prices.loc[rets.index])
    ml_scores = ml_scores.reindex(rp.index).fillna(ml_scores.mean())
    w = (rp * ml_scores)
    w /= w.sum()
    fig_dir = ensure_figures_dir()
    plt.figure(figsize=(8, 4))
    w.sort_values(ascending=False).plot(kind="bar")
    plt.title("Poids Hybrides: Risk Parity × Score ML")
    plt.ylabel("Poids")
    out_path = os.path.join(fig_dir, "hybrid_weights.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return w


if __name__ == "__main__":
    print(run_hybrid().round(4))


