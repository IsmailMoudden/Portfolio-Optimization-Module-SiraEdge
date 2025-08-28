HYPERPARAMS = {
    "data": {
        "start": "2020-01-01",
        "end": "2023-12-31",
        "mode": "auto",
    },
    "features": {
        "momentum_window": 20,
        "volatility_window": 20,
        "rsi_window": 14,
    },
    "ridge": {
        "alpha": 5.0,
    },
    "black_litterman": {
        "tau": 0.05,
        "delta": 2.5,
        "view_QQQ_minus_SPY": 0.02,
    },
    "monte_carlo": {
        "n_portfolios": 10000,
        "seed": 42,
    },
    "walkforward": {
        "lookback_days": 252,
        "rebalance_freq": "M",
        "transaction_cost_bps": 5.0,
    },
}

if __name__ == "__main__":
    import json
    print(json.dumps(HYPERPARAMS, indent=2))


