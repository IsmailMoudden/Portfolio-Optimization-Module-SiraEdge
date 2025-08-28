# SiraEdge - Portfolio Optimization Module

> 🇫🇷 Ce README est aussi disponible en [français](README.fr.md).

## Overview

This project presents the **portfolio optimization module** developed for the **SiraEdge** platform — an innovative solution designed to democratize access to professional-grade asset management tools.

## Mission of SiraEdge

SiraEdge was born from a simple yet ambitious vision: **to democratize access to professional portfolio management tools**.  
Our mission is to make finance accessible to everyone by combining mathematical rigor with an intuitive user experience.

### Core Values
- **Accessibility**: Democratizing finance through pedagogy  
- **Innovation**: Leveraging the latest advances in quantitative finance  
- **Excellence**: Robust code, optimal performance, and reliability  

## Implemented Optimization Models

This module implements and analyzes **7 portfolio optimization models**:

### 1. **Markowitz (Mean-Variance)**
- Foundation of modern portfolio theory  
- Maximization of the Sharpe ratio  
- Balancing risk and return  

### 2. **Risk Parity**
- Equalizing risk contributions across assets  
- Robustness to return uncertainty  
- Natural diversification  

### 3. **Monte Carlo (10,000 Portfolios)**
- Random exploration of the solution space  
- Visualization of the efficient frontier  
- Validation of candidate solutions  

### 4. **Black-Litterman**
- Combining market equilibrium with investor views  
- Adjusted equilibrium portfolio  
- Institutional-grade approach  

### 5. **Machine Learning (Ridge Regression)**
- Forecasting returns using AI  
- Analysis of technical patterns (RSI, momentum, volatility)  
- Objective quantitative signals  

### 6. **Hybrid Model**
- Combination of Risk Parity + ML signals  
- Stable foundation + smart opportunism  
- Core-satellite approach  

### 7. **Custom Metrics**
- Tailor-made objectives (Sharpe + Stability + MDD + Turnover)  
- Adapted to investor preferences  
- Multi-objective optimization  

## 🏗️ Project Structure

```
SiraEdge_Optimisation/
├── src/                          # Python source code
│   ├── data_utils.py            # Data utilities and indicators
│   ├── markowitz.py             # Markowitz optimization
│   ├── risk_parity.py           # Risk parity
│   ├── monte_carlo.py           # Monte Carlo simulation
│   ├── black_litterman.py       # Black-Litterman model
│   ├── ml_predictor.py          # ML predictor
│   ├── hybrid_model.py          # Hybrid model
│   ├── custom_metrics_opt.py    # Custom metrics
│   ├── metrics_utils.py         # Financial metrics calculations
│   ├── walkforward_backtest.py  # Walk-forward backtesting
│   └── hyperparams.py           # Centralized hyperparameters
├── figures/                      # Charts and visualizations
│   ├── markowitz_frontier.png
│   ├── risk_parity_weights.png
│   ├── monte_carlo_cloud.png
│   ├── black_litterman_weights.png
│   ├── ml_coefficients.png
│   ├── hybrid_weights.png
│   └── custom_opt_weights.png
├── rapport/                      # LaTeX report sources
│   ├── sources/                  # LaTeX source files
│   │   ├── rapport_siraedge_fr.tex
│   │   ├── rapport_siraedge_en.tex
│   │   ├── portfolio_summary_included.tex
│   │   └── walkforward_results_simulated.tex
│   ├── pdf/                      # Compiled PDF reports
│   │   ├── rapport_siraedge_fr.pdf
│   │   └── rapport_siraedge_en.pdf
│   ├── tables/                   # Tabular data
│   ├── compile_reports.sh        # Automated compilation script
│   └── README.md                 # Report organization guide
├── README.md                     # This file
├── requirements.txt              # Python dependencies
└── make_report.sh               # Automated generation script
```
## 🛠️ Installation & Setup

### Requirements
- Python 3.8+
- pip or conda

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/SiraEdge/portfolio-optimization.git
cd portfolio-optimization
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Verify installation**
```bash
python -c "import pandas, numpy, matplotlib, yfinance; print('Installation réussie!')"
```

## 🚀 Usage

### 1. **Generate Figures**
```bash
# Générer toutes les figures
python src/data_utils.py
python src/markowitz.py
python src/risk_parity.py
python src/monte_carlo.py
python src/black_litterman.py
python src/ml_predictor.py
python src/hybrid_model.py
python src/custom_metrics_opt.py
python src/walkforward_backtest.py
```

### 2. **Compile Report**
```bash
# Use automatic script
chmod +x make_report.sh
./make_report.sh

# Or manual compilation
cd rapport/sources
tectonic rapport_siraedge_fr.tex
tectonic rapport_siraedge_en.tex

# Or use the automated compilation script
cd rapport
./compile_reports.sh
```

### 3. **Interactive Usage**
```python
# Usage exemple of the Risk Parity module
import numpy as np
from src.risk_parity import risk_parity_weights
from src.data_utils import download_prices

# collect prices 
prices = download_prices(["SPY", "QQQ", "GLD", "DBA"], "2020-01-01", "2023-12-31", mode="auto")
rets = np.log(prices / prices.shift(1)).dropna()
cov = (rets.cov().values) * 252

# Calculate the Risk Parity weigth
weights = risk_parity_weights(cov)
print("Poids optimaux:", weights)
```

## 📈 Data & Assets

### Asset Universe
- **ETFs** : SPY, QQQ, IWM, EFA
- **Commodities** : GLD, SLV, USO, DBA

### Analysis Period
- **Window** : 2020-2023
- **Fréquency** : Daily data
- **Rebalancing** : Monthly (walk-forward)

### Technical Indicators
- RSI (Relative Strength Index)
- Momentum (20-day returns)
- Rolling volatility (20-day window)
- Dynamic correlations

## Walk-Forward Analysis

The module includes a robust walk-forward analysis:

- **Rolling window:** 252 days (1 trading year)
- **Rebalancing:** Monthly
- **Transaction costs:** 5 bps
- **Metrics:** Sharpe, Calmar, Sortino, Stability, Turnover

## Performance Metrics

### Classic Metrics
- **Sharpe Ratio** – risk-adjusted return
- **Sortino Ratio** – downside risk-adjusted return
- **Calmar Ratio** – return vs. max drawdown
- **Volatility** – annualized risk

### Advanced Metrics
- **Weight stability** – allocation consistency
- **Turnover** – trading activity
- **Effective diversification** – equivalent number of assets
- **Average correlation** – portfolio cohesion

## 🎓 Educational Purpose

This project is designed to be **educational** and **accessible**:

- Detailed explanations of each concept
- Concrete examples with numbers
- Simple analogies for complex ideas
- Commented and documented code
- End-to-end workflow from start to finish

## 🔧 Customization

### Adjustable Hyperparameters
- Estimation windows (126, 252, 504 days)
- ML signal intensity (alpha)
- Confidence in Black-Litterman views (tau)
- Weights for custom metrics

### Configurable Constraints
- Asset weight limits
- Sector/asset class constraints
- Turnover penalties
- Leverage constraints

## 📚 Documentation

### LaTeX Report
The full reports are available in both French and English:
- **French Report**: `rapport/pdf/rapport_siraedge_fr.pdf`
- **English Report**: `rapport/pdf/rapport_siraedge_en.pdf`

Both reports include:
- Detailed theoretical explanations
- Mathematical formulations
- Comparative analyses
- Results and interpretations
- Practical limitations and considerations

### Source Code
- Complete implementations of the models
- Lightweight comments and docstrings for key functions
- Data fallback: automatic switch to synthetic series if download fails

## Integration with SiraEdge

This module integrates directly into the SiraEdge platform to allow users to:

1. Test different optimization models
2. Compare performance in real time
3. Understand risk–return trade-offs
4. Learn by doing, with immediate feedback (educational focus)

> **Scope:** this module is intended for educational and comparative purposes only.  
> It does not constitute investment advice or performance guarantees.

## 📄 License

This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for details.

## 📞 Contact

- **SiraEdge Team:** [siraedge.service@gmail.com](mailto:siraedge.service@gmail.com)
- **GitHub Repository:** https://github.com/SiraEdge/portfolio-optimization
- **SiraEdge Platform:** https://siraedge.com

**SiraEdge** – Making finance accessible, transparent, and innovative for everyone. 🚀