import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Callable, Tuple

from .data_utils import download_prices, ensure_figures_dir, DEFAULT_TICKERS
from .metrics_utils import sharpe_ratio, calmar_ratio, sortino_ratio, weight_stability, turnover


def compute_weights_markowitz(prices: pd.DataFrame) -> np.ndarray:
    # Simple heuristic: inverse-variance portfolio as proxy (to avoid heavy deps here)
    returns = np.log(prices / prices.shift(1)).dropna()
    iv = 1.0 / (returns.var() + 1e-12)
    w = iv / iv.sum()
    return w.values


def rebalance_simplex(weights: np.ndarray) -> np.ndarray:
    w = np.maximum(weights, 0)
    s = w.sum()
    return w / s if s > 0 else np.ones_like(w) / len(w)


def walk_forward_backtest(
    prices: pd.DataFrame,
    lookback_days: int = 252,
    rebalance_freq: str = "M",
    weight_fn: Callable[[pd.DataFrame], np.ndarray] = compute_weights_markowitz,
    transaction_cost_bps: float = 5.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    rets_daily = prices.pct_change().dropna()
    rebal_dates = rets_daily.resample(rebalance_freq).first().index
    weights_list = []
    port_rets = []
    w_prev = np.ones(prices.shape[1]) / prices.shape[1]
    tc = transaction_cost_bps / 10000.0
    for date in rebal_dates:
        window_start = date - pd.tseries.offsets.BDay(lookback_days)
        hist = prices.loc[window_start:date]
        if len(hist) < lookback_days // 2:
            continue
        w_new = weight_fn(hist)
        w_new = rebalance_simplex(w_new)
        # Transaction cost penalty applied on rebal day
        tw = turnover(w_prev, w_new)
        day_ret = rets_daily.loc[date:date].values
        if day_ret.size == 0:
            continue
        daily_ret = float(w_new @ day_ret[0] - tc * tw)
        port_rets.append(pd.Series([daily_ret], index=[date]))
        weights_list.append(pd.Series(w_new, index=prices.columns, name=date))
        w_prev = w_new
    port_rets = pd.concat(port_rets).reindex(rets_daily.index, method="ffill").fillna(0.0)
    weights_df = pd.DataFrame(weights_list)
    metrics = {
        "Sharpe": sharpe_ratio(port_rets),
        "Calmar": calmar_ratio(port_rets),
        "Sortino": sortino_ratio(port_rets),
        "Stability": weight_stability(weights_df.values) if not weights_df.empty else 1.0,
        "TurnoverAvg": float(np.abs(np.diff(weights_df.values, axis=0)).sum(axis=1).mean()) if len(weights_df) > 1 else 0.0,
    }
    return weights_df, port_rets, metrics


def main():
    prices = download_prices(DEFAULT_TICKERS, "2020-01-01", "2023-12-31", mode="auto")
    weights_df, port_rets, metrics = walk_forward_backtest(prices)
    out_dir = ensure_figures_dir()
    metrics_path = os.path.join(out_dir, "walkforward_metrics.csv")
    weights_path = os.path.join(out_dir, "walkforward_weights.csv")
    port_path = os.path.join(out_dir, "walkforward_returns.csv")
    weights_df.to_csv(weights_path)
    port_rets.to_csv(port_path, header=["ret"]) 
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print("Walk-forward metrics:", metrics)


if __name__ == "__main__":
    main()



