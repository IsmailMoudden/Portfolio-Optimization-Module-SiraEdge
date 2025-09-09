# SiraEdge - Module d'Optimisation de Portefeuille

> 🇬🇧 This README is also available in [English](README.md).

## Vue d'ensemble

Ce projet présente le module d'optimisation de portefeuille développé pour la plateforme **SiraEdge**, une solution innovante visant à démocratiser l'accès aux outils professionnels de gestion d'actifs.

## Mission de SiraEdge

SiraEdge est né d'une vision simple mais ambitieuse : **démocratiser l'accès aux outils professionnels de gestion de portefeuille**. Notre mission est de rendre la finance accessible à tous, en combinant rigueur mathématique et expérience utilisateur intuitive.

### Valeurs Fondamentales
- **Accessibilité** : Démocratisation de la finance et pédagogie
- **Innovation** : Intégration des dernières avancées en finance quantitative
- **Excellence** : Code robuste, performance optimale et fiabilité

## Modèles d'Optimisation Implémentés

Ce module implémente et analyse **7 modèles d'optimisation** de portefeuille :

### 1. **Markowitz (Mean-Variance)**
- Fondement de l'optimisation moderne
- Maximisation du ratio de Sharpe
- Gestion du compromis risque-rendement

### 2. **Risk Parity**
- Égalisation des contributions au risque
- Robustesse face aux incertitudes de rendement
- Diversification naturelle

### 3. **Monte Carlo (10 000 portefeuilles)**
- Exploration aléatoire de l'espace des solutions
- Visualisation de la frontière efficiente
- Validation des solutions candidates

### 4. **Black-Litterman**
- Fusion des vues du marché et convictions personnelles
- Portefeuille d'équilibre ajusté
- Approche professionnelle institutionnelle

### 5. **Machine Learning (Ridge)**
- Prédiction des rendements via IA
- Analyse de patterns techniques (RSI, momentum, volatilité)
- Signaux quantitatifs objectifs

### 6. **Modèle Hybride**
- Combinaison Risk Parity + scores ML
- Stabilité de base + opportunisme intelligent
- Approche core-satellite

### 7. **Métriques Personnalisées**
- Objectifs sur mesure (Sharpe + Stabilité + MDD + Turnover)
- Adaptation aux préférences de l'investisseur
- Optimisation multi-critères

## 🏗️ Structure du Projet

```
SiraEdge_Optimisation/
├── src/                          # Code source Python
│   ├── data_utils.py            # Utilitaires de données et indicateurs
│   ├── markowitz.py             # Optimisation Markowitz
│   ├── risk_parity.py           # Parité de risque
│   ├── monte_carlo.py           # Simulation Monte Carlo
│   ├── black_litterman.py       # Modèle Black-Litterman
│   ├── ml_predictor.py          # Prédicteur ML
│   ├── hybrid_model.py          # Modèle hybride
│   ├── custom_metrics_opt.py    # Métriques personnalisées
│   ├── metrics_utils.py         # Calculs de métriques financières
│   ├── walkforward_backtest.py  # Backtest walk-forward
│   └── hyperparams.py           # Hyperparamètres centralisés
├── figures/                      # Graphiques et visualisations
│   ├── markowitz_frontier.png
│   ├── risk_parity_weights.png
│   ├── monte_carlo_cloud.png
│   ├── black_litterman_weights.png
│   ├── ml_coefficients.png
│   ├── hybrid_weights.png
│   └── custom_opt_weights.png
├── rapport/                      # Sources LaTeX du rapport
│   ├── sources/                  # Fichiers sources LaTeX
│   │   ├── rapport_siraedge_fr.tex
│   │   ├── rapport_siraedge_en.tex
│   │   ├── portfolio_summary_included.tex
│   │   └── walkforward_results_simulated.tex
│   ├── pdf/                      # Rapports PDF compilés
│   │   ├── rapport_siraedge_fr.pdf
│   │   └── rapport_siraedge_en.pdf
│   ├── tables/                   # Données tabulaires
│   ├── compile_reports.sh        # Script de compilation automatisée
│   └── README.md                 # Guide d'organisation du rapport
├── README.md                     # Ce fichier
├── requirements.txt              # Dépendances Python
└── make_report.sh               # Script de génération automatique
```

## 🛠️ Installation et Configuration

### Prérequis
- Python 3.8+
- pip ou conda

### Installation

1. **Cloner le repository**
```bash
git clone https://github.com/IsmailMoudden/Portfolio-Optimization-Module-SiraEdge.git
cd Portfolio-Optimization-Module-SiraEdge
```

2. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

3. **Vérifier l'installation**
```bash
python -c "import pandas, numpy, matplotlib, yfinance; print('Installation réussie!')"
```

## 🚀 Utilisation

### 1. **Génération des Figures**
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

### 2. **Compilation du Rapport**
```bash
# Utiliser le script automatique
chmod +x make_report.sh
./make_report.sh

# Ou compiler manuellement
cd rapport/sources
tectonic rapport_siraedge_fr.tex
tectonic rapport_siraedge_en.tex

# Ou utiliser le script de compilation automatisé
cd rapport
./compile_reports.sh
```

### 3. **Utilisation Interactive**
```python
# Exemple d'utilisation du module Risk Parity
import numpy as np
from src.risk_parity import risk_parity_weights
from src.data_utils import download_prices

# Récupérer les prix (mode auto: télécharge ou bascule en données synthétiques)
prices = download_prices(["SPY", "QQQ", "GLD", "DBA"], "2020-01-01", "2023-12-31", mode="auto")
rets = np.log(prices / prices.shift(1)).dropna()
cov = (rets.cov().values) * 252

# Calculer les poids Risk Parity
weights = risk_parity_weights(cov)
print("Poids optimaux:", weights)
```

## 📈 Données et Actifs

### Univers d'Actifs
- **ETFs** : SPY, QQQ, IWM, EFA
- **Commodities** : GLD, SLV, USO, DBA

### Période d'Analyse
- **Fenêtre** : 2020-2023
- **Fréquence** : Données quotidiennes
- **Rebalancing** : Mensuel (walk-forward)

### Indicateurs Techniques
- RSI (Relative Strength Index)
- Momentum (retour sur 20 jours)
- Volatilité rolling (fenêtre 20 jours)
- Corrélations dynamiques

## Analyse Walk-Forward

Le module inclut une analyse walk-forward robuste :

- **Rolling window** : 252 jours (1 année de trading)
- **Rebalancing** : Mensuel
- **Coûts de transaction** : 5 bps
- **Métriques** : Sharpe, Calmar, Sortino, Stabilité, Turnover

## Métriques de Performance

### Métriques Classiques
- **Ratio de Sharpe** : Rendement ajusté au risque
- **Ratio de Sortino** : Rendement ajusté au risque baissier
- **Ratio de Calmar** : Rendement vs Maximum Drawdown
- **Volatilité** : Risque annualisé

### Métriques Avancées
- **Stabilité des poids** : Constance de l'allocation
- **Turnover** : Activité de trading
- **Diversification effective** : Nombre d'actifs équivalents
- **Corrélation moyenne** : Cohésion du portefeuille

## 🎓 Aspect Pédagogique

Ce projet est conçu pour être **éducatif** et **accessible** :

- **Explications détaillées** de chaque concept
- **Exemples concrets** avec chiffres
- **Analogies simples** pour concepts complexes
- **Code commenté** et documenté
- **Workflow complet** de A à Z

## 🔧 Personnalisation

### Hyperparamètres Modifiables
- Fenêtres d'estimation (126, 252, 504 jours)
- Intensité des signaux ML (alpha)
- Confiance des vues Black-Litterman (tau)
- Poids des métriques personnalisées

### Contraintes Configurables
- Limites de poids par actif
- Contraintes de secteur/classe d'actifs
- Pénalités de turnover
- Contraintes de levier

## 📚 Documentation

### Rapport LaTeX
Les rapports complets sont disponibles en français et en anglais :
- **Rapport Français** : `rapport/pdf/rapport_siraedge_fr.pdf`
- **Rapport Anglais** : `rapport/pdf/rapport_siraedge_en.pdf`

Les deux rapports contiennent :
- Explications théoriques détaillées
- Formulations mathématiques
- Analyses comparatives
- Résultats et interprétations
- Limites et considérations pratiques

### Code Source
- **Implémentations complètes** des modèles listés dans le rapport
- **Commentaires et docstrings légers** pour l'essentiel des fonctions
- **Fallback de données**: bascule automatique vers des séries synthétiques si le téléchargement échoue

## Intégration SiraEdge

Ce module s'intègre directement dans la plateforme SiraEdge pour permettre aux utilisateurs de :

1. **Tester différents modèles** d'optimisation
2. **Comparer leurs performances** en temps réel
3. **Comprendre les trade-offs** risque-rendement
4. **Apprendre par la pratique** avec feedback immédiat (cadre pédagogique)

> Portée: ce module a une vocation éducative et comparative; il ne constitue pas une recommandation d'investissement ni une promesse de performance.



## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 📞 Contact

- **Équipe SiraEdge** : [siraedge.service@gmail.com](mailto:siraedge.service@gmail.com)
- **Repository GitHub** : [https://github.com/IsmailMoudden/Portfolio-Optimization-Module-SiraEdge](https://github.com/IsmailMoudden/Portfolio-Optimization-Module-SiraEdge)
- **Plateforme SiraEdge** : [https://siraedge.com](https://siraedge.com)



**SiraEdge** - Rendre la finance accessible, transparente et innovante pour tous. 🚀
