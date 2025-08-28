#!/bin/zsh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "[1/2] Génération des figures..."
python src/data_utils.py || true
python src/markowitz.py || true
python src/risk_parity.py || true
python src/monte_carlo.py || true
python src/black_litterman.py || true
python src/ml_predictor.py || true
python src/hybrid_model.py || true
python src/custom_metrics_opt.py || true

echo "[2/2] Compilation du rapport (Tectonic)..."
cd rapport
if command -v tectonic >/dev/null 2>&1; then
  tectonic rapport_siraedge_optimisation.tex
else
  echo "Tectonic introuvable. Essaie: brew install tectonic" >&2
  exit 1
fi


