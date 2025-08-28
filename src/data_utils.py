import os
import argparse
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import yfinance as yf


DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "EFA", "GLD", "SLV", "USO", "DBA"]


def ensure_figures_dir() -> str:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    figures_dir = os.path.join(base_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


def _simulate_correlated_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    dates = pd.bdate_range(start=start, end=end, freq="C")
    num_days = len(dates)
    num_assets = len(tickers)
    # Target annual volatilities (approx): equities ~20%, gold 15%, silver 25%, oil 35%, agri 12%
    ann_vol = np.array([0.20, 0.22, 0.28, 0.18, 0.15, 0.25, 0.35, 0.12])[:num_assets]
    daily_vol = ann_vol / np.sqrt(252)
    # Coherent correlation matrix: high among equities, mild with commodities
    corr = np.eye(num_assets)
    for i in range(num_assets):
        for j in range(i+1, num_assets):
            if i <= 3 and j <= 3:
                corr[i, j] = corr[j, i] = 0.7 - 0.1*abs(i-j)
            else:
                corr[i, j] = corr[j, i] = 0.2 if (i <= 3) ^ (j <= 3) else 0.3
    cov = np.outer(daily_vol, daily_vol) * corr
    rng = np.random.default_rng(123)
    rets = rng.multivariate_normal(mean=np.zeros(num_assets), cov=cov, size=num_days)
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    df = pd.DataFrame(prices, index=dates, columns=tickers)
    return df


def download_prices(tickers: List[str], start: str, end: str, mode: str = "auto") -> pd.DataFrame:
    """Download adjusted close prices. If mode='synthetic', generate coherent synthetic prices.
    If mode='auto', try download, then fallback to synthetic on failure or empty data."""
    if mode == "synthetic":
        return _simulate_correlated_prices(tickers, start, end)
    try:
        data = yf.download(tickers=tickers, start=start, end=end, progress=False, auto_adjust=True)
        if isinstance(data, pd.DataFrame) and "Close" in data.columns:
            prices = data["Close"].dropna(how="all")
        else:
            prices = data.dropna(how="all")
        if prices is None or prices.empty:
            raise ValueError("Empty download; switching to synthetic")
        return prices
    except Exception:
        return _simulate_correlated_prices(tickers, start, end)


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    log_returns = np.log(prices / prices.shift(1)).dropna(how="all")
    return log_returns


def compute_indicators(prices: pd.DataFrame, window_mom: int = 20, window_vol: int = 20, window_rsi: int = 14) -> Dict[str, pd.DataFrame]:
    indicators: Dict[str, pd.DataFrame] = {}
    # Lazy import to avoid hard dependency when not needed
    try:
        from ta.momentum import RSIIndicator  # type: ignore
        def rsi_fn(series: pd.Series, window: int) -> pd.Series:
            return RSIIndicator(close=series, window=window).rsi()
    except Exception:
        def rsi_fn(series: pd.Series, window: int) -> pd.Series:
            # Fallback RSI approximation using exponential averages of gains/losses
            delta = series.diff()
            gain = delta.clip(lower=0.0)
            loss = -delta.clip(upper=0.0)
            avg_gain = gain.ewm(alpha=1.0/window, min_periods=window, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1.0/window, min_periods=window, adjust=False).mean()
            rs = avg_gain / (avg_loss + 1e-12)
            return 100.0 - (100.0 / (1.0 + rs))
    for ticker in prices.columns:
        series = prices[ticker].dropna()
        if series.empty:
            continue
        momentum = series.pct_change(window_mom)
        volatility = series.pct_change().rolling(window_vol).std() * np.sqrt(252)
        rsi = rsi_fn(series, window_rsi)
        feats = pd.DataFrame({
            "momentum": momentum,
            "volatility": volatility,
            "rsi": rsi
        }).dropna()
        indicators[ticker] = feats
    return indicators


def align_features_and_targets(prices: pd.DataFrame, indicators: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    log_returns = compute_log_returns(prices)
    next_day_returns = log_returns.shift(-1)
    frames = []
    y_list = []
    asset_list = []
    for ticker, feats in indicators.items():
        common_index = feats.index.intersection(next_day_returns.index)
        X = feats.loc[common_index]
        y = next_day_returns.loc[common_index, ticker]
        asset = pd.Series([ticker] * len(common_index), index=common_index, name="asset")
        frames.append(X)
        y_list.append(y)
        asset_list.append(asset)
    if not frames:
        raise ValueError("No features were computed; check tickers and date range.")
    X_all = pd.concat(frames, keys=[s.name for s in asset_list], names=["asset"], axis=0)
    y_all = pd.concat(y_list, axis=0)
    asset_index = X_all.index.get_level_values(0)
    return X_all.reset_index(level=0, drop=True), y_all, asset_index


def cli():
    parser = argparse.ArgumentParser(description="Download data and compute indicators")
    parser.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--end", type=str, default="2023-12-31")
    parser.add_argument("--mode", type=str, choices=["auto", "synthetic", "download"], default="auto")
    args = parser.parse_args()

    prices = download_prices(args.tickers, args.start, args.end, mode=args.mode)
    indicators = compute_indicators(prices)
    _ = ensure_figures_dir()
    print(f"Downloaded prices shape: {prices.shape}")
    print("Computed indicators for:", list(indicators.keys()))


if __name__ == "__main__":
    cli()

