import os
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

try:
    from .data_utils import (
        download_prices,
        compute_indicators,
        align_features_and_targets,
        ensure_figures_dir,
        DEFAULT_TICKERS,
    )
except ImportError:
    from data_utils import (
        download_prices,
        compute_indicators,
        align_features_and_targets,
        ensure_figures_dir,
        DEFAULT_TICKERS,
    )


def compute_sign_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return np.nan
    return np.mean(np.sign(y_true) == np.sign(y_pred))


def rolling_ridge_training(
    X: pd.DataFrame,
    y: pd.Series,
    asset_index,
    alphas: List[float],
    min_train: int = 252,
    step: int = 21,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological rolling training per asset. At each step, fit RidgeCV on data up to t,
    evaluate on the next "step" days. Returns three DataFrames indexed by datetime:
    - metrics_df: columns [asset, r2, mae, sign_acc, alpha]
    - preds_df: predicted returns
    - coefs_df: model coefficients per feature aggregated by asset (last fit)
    """
    features: List[str] = list(X.columns)
    # Ensure asset_index is a Series aligned to X
    if not isinstance(asset_index, pd.Series):
        asset_index = pd.Series(asset_index, index=X.index, name="asset")
    dates: List[pd.Timestamp] = sorted(pd.Index(X.index).unique())

    rows_metrics = []
    preds_records = []
    last_coefs: Dict[Tuple[str, str], float] = {}

    for start_idx in range(min_train, len(dates) - step, step):
        train_end_date = dates[start_idx]
        test_end_idx = min(start_idx + step, len(dates))
        test_end_date = dates[test_end_idx - 1]

        # Build boolean masks instead of label slicing to handle non-unique indices
        train_mask = pd.Index(X.index) <= train_end_date
        test_mask = (pd.Index(X.index) > train_end_date) & (pd.Index(X.index) <= test_end_date)

        X_train = X.loc[train_mask]
        y_train = y.loc[train_mask]
        X_test = X.loc[test_mask]
        y_test = y.loc[test_mask]
        assets_test = asset_index.loc[test_mask]

        # Fit one model per asset to respect cross-sectional heterogeneity
        for asset in assets_test.unique():
            idx_train = (asset_index.loc[train_mask] == asset)
            idx_test = (assets_test == asset)
            if idx_train.sum() < min_train // 4 or idx_test.sum() == 0:
                continue

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", RidgeCV(alphas=np.array(alphas))),
            ])

            pipe.fit(X_train[idx_train], y_train[idx_train])
            y_pred = pipe.predict(X_test[idx_test])

            r2 = r2_score(y_test[idx_test], y_pred)
            mae = mean_absolute_error(y_test[idx_test], y_pred)
            sacc = compute_sign_accuracy(y_test[idx_test].values, y_pred)
            alpha_chosen: float = float(pipe.named_steps["ridge"].alpha_)

            rows_metrics.append({
                "date": test_end_date,
                "asset": asset,
                "r2": r2,
                "mae": mae,
                "sign_acc": sacc,
                "alpha": alpha_chosen,
            })

            preds_records.append(pd.Series(y_pred, index=y_test[idx_test].index, name=asset))

            # Save coefficients of the last fit window per asset
            ridge = pipe.named_steps["ridge"]
            scaler = pipe.named_steps["scaler"]
            coefs = ridge.coef_ / (scaler.scale_ + 1e-12)
            for f, c in zip(features, coefs):
                last_coefs[(asset, f)] = c

    metrics_df = pd.DataFrame(rows_metrics)
    # Optional: predictions export can be heavy and indices may duplicate; keep empty for report
    preds_df = pd.DataFrame()
    if last_coefs:
        coefs_df = (
            pd.Series(last_coefs)
            .rename("coef")
            .reset_index()
            .rename(columns={"level_0": "asset", "level_1": "feature"})
        )
    else:
        coefs_df = pd.DataFrame(columns=["asset", "feature", "coef"])
    return metrics_df, preds_df, coefs_df


def plot_metrics(metrics_df: pd.DataFrame, figures_dir: str) -> None:
    if metrics_df.empty:
        return
    # Aggregate across assets per date
    agg = metrics_df.groupby("date").agg({"r2": "mean", "sign_acc": "mean"}).reset_index()

    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax[0].plot(agg["date"], agg["r2"], label="R^2 (moyenne actifs)")
    ax[0].axhline(0.0, color="gray", lw=0.8, ls="--")
    ax[0].set_ylabel("R^2")
    ax[0].legend()

    ax[1].plot(agg["date"], agg["sign_acc"], color="tab:green", label="Accuracy signe")
    ax[1].axhline(0.5, color="gray", lw=0.8, ls="--", label="Hasard 50%")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_xlabel("Date")
    ax[1].legend()

    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "ml_accuracy_evolution.png"), dpi=160)
    plt.close(fig)


def plot_coefs(coefs_df: pd.DataFrame, figures_dir: str) -> None:
    if coefs_df.empty:
        return
    # Bar plot per feature averaged over assets
    mean_coefs = coefs_df.groupby("feature")["coef"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    mean_coefs.plot(kind="barh", ax=ax, color=["tab:blue" if v >= 0 else "tab:red" for v in mean_coefs.values])
    ax.set_title("Coefficients moyens (dernière fenêtre)")
    ax.set_xlabel("Amplitude")
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "ml_coefficients.png"), dpi=160)
    plt.close(fig)


def save_hyperparams(metrics_df: pd.DataFrame, figures_dir: str) -> None:
    if metrics_df.empty:
        return
    # Most frequent alpha per asset
    hp = (
        metrics_df.groupby("asset")["alpha"]
        .agg(lambda s: s.value_counts().idxmax())
        .rename("alpha_chosen")
        .reset_index()
    )
    hp.to_csv(os.path.join(figures_dir, "ml_hyperparams.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description="Train ML predictor and export figures/metrics")
    parser.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--end", type=str, default="2023-12-31")
    parser.add_argument("--mode", type=str, choices=["auto", "synthetic", "download"], default="auto")
    parser.add_argument("--alphas", type=float, nargs="*", default=[1e-3, 1e-2, 1e-1, 1.0, 10.0])
    parser.add_argument("--min-train", type=int, default=252)
    parser.add_argument("--step", type=int, default=21)
    args = parser.parse_args()

    prices = download_prices(args.tickers, args.start, args.end, mode=args.mode)
    indicators = compute_indicators(prices)
    X, y, asset_index = align_features_and_targets(prices, indicators)

    figures_dir = ensure_figures_dir()

    metrics_df, preds_df, coefs_df = rolling_ridge_training(
        X, y, asset_index, alphas=args.alphas, min_train=args.min_train, step=args.step
    )

    # Persist artifacts
    metrics_df.to_csv(os.path.join(figures_dir, "ml_training_metrics.csv"), index=False)
    preds_df.to_csv(os.path.join(figures_dir, "ml_predictions.csv"))
    coefs_df.to_csv(os.path.join(figures_dir, "ml_coefficients_lastwindow.csv"), index=False)

    # Plots
    plot_metrics(metrics_df, figures_dir)
    plot_coefs(coefs_df, figures_dir)
    save_hyperparams(metrics_df, figures_dir)

    print(f"Saved metrics and figures to: {figures_dir}")


if __name__ == "__main__":
    main()


