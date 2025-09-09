"""
SiraEdge Portfolio Optimization Module

This package contains implementations of various portfolio optimization models:
- Markowitz (Mean-Variance) optimization
- Risk Parity optimization
- Monte Carlo simulation
- Black-Litterman model
- Machine Learning predictor
- Hybrid model
- Custom metrics optimization
- Walk-forward backtesting

All modules are designed to work together for comprehensive portfolio analysis.
"""

__version__ = "1.0.0"
__author__ = "Ismail Moudden"
__email__ = "siraedge.service@gmail.com"

# Import main functions for easy access
from .markowitz import markowitz_weights
from .risk_parity import risk_parity_weights
from .monte_carlo import monte_carlo_simulation
from .black_litterman import black_litterman_weights
from .ml_predictor import ml_predictor
from .hybrid_model import hybrid_weights
from .custom_metrics_opt import custom_metrics_weights
from .walkforward_backtest import walkforward_backtest

__all__ = [
    "markowitz_weights",
    "risk_parity_weights", 
    "monte_carlo_simulation",
    "black_litterman_weights",
    "ml_predictor",
    "hybrid_weights",
    "custom_metrics_weights",
    "walkforward_backtest"
]
