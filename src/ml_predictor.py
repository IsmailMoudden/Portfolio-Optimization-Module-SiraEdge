import os
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from .data_utils import download_prices, compute_indicators, ensure_figures_dir, DEFAULT_TICKERS


def prepare_ml_dataset(start="2020-01-01", end="2023-12-31", tickers=None) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    if tickers is None:
        tickers = DEFAULT_TICKERS
    prices = download_prices(tickers, start, end)
    indicators = compute_indicators(prices)
    # Construct panel-like dataset
    rows = []
    for ticker, feats in indicators.items():
        r = np.log(prices[ticker] / prices[ticker].shift(1)).shift(-1)
        df = feats.join(r.rename("target")).dropna()
        df["asset"] = ticker
        rows.append(df)
    data = pd.concat(rows)
    X = data[["momentum", "volatility", "rsi"]]
    y = data["target"]
    assets = data["asset"]
    return X, y, assets


def fit_ridge_by_asset(X: pd.DataFrame, y: pd.Series, assets: pd.Series, alpha=1.0) -> pd.Series:
    coefs = {}
    for asset in assets.unique():
        Xa = X[assets == asset]
        ya = y[assets == asset]
        if len(Xa) < 100:
            continue
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha))
        ])
        pipe.fit(Xa, ya)
        coefs[asset] = float(np.linalg.norm(pipe.named_steps["ridge"].coef_))
    return pd.Series(coefs)


def run_ml_ridge():
    X, y, assets = prepare_ml_dataset()
    coef_norms = fit_ridge_by_asset(X, y, assets, alpha=5.0)
    fig_dir = ensure_figures_dir()
    plt.figure(figsize=(8, 4))
    coef_norms.sort_values(ascending=False).plot(kind="bar")
    plt.title("Importance (norme des coefficients Ridge)")
    plt.ylabel("||w||")
    out_path = os.path.join(fig_dir, "ml_coefficients.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return coef_norms


if __name__ == "__main__":
    print(run_ml_ridge().round(4))


