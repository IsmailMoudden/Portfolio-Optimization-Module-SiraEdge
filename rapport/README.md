# Dossier Rapport - SiraEdge Portfolio Optimization

## 📁 Structure des Dossiers

```
rapport/
├── sources/           # Sources LaTeX des rapports
├── pdf/              # Rapports PDF compilés
├── tables/           # Données tabulaires
├── compile_reports.sh # Script de compilation
└── README.md         # Ce fichier
```

## 📄 Fichiers Sources (sources/)

### Rapports Principaux
- **`rapport_siraedge_fr.tex`** - Version française du rapport (58.2 KB)
- **`rapport_siraedge_en.tex`** - Version anglaise du rapport (53.0 KB)

### Données Incluses
- **`portfolio_summary_included.tex`** - Tableau de synthèse des performances
- **`walkforward_results_simulated.tex`** - Résultats du backtest walk-forward

## 📊 Rapports Compilés (pdf/)

### PDFs Finaux
- **`rapport_siraedge_fr.pdf`** - Rapport français compilé (1.30 MB)
- **`rapport_siraedge_en.pdf`** - Rapport anglais compilé (1.30 MB)

## 📋 Données Tabulaires (tables/)

Ce dossier est prêt pour recevoir des données tabulaires supplémentaires si nécessaire.

## 🚀 Compilation

### Compilation Automatique
```bash
./compile_reports.sh
```

### Compilation Manuelle
```bash
cd sources
tectonic rapport_siraedge_fr.tex
tectonic rapport_siraedge_en.tex
```

## 📝 Notes Importantes

- **Traduction Complète** : Les deux versions (FR/EN) contiennent exactement le même contenu
- **Organisation Propre** : Structure claire avec séparation des sources, PDFs et données
- **Script de Compilation** : Automatisation de la génération des rapports
- **Gestion des Erreurs** : Le script gère les erreurs de compilation de manière gracieuse

## 🔧 Dépendances

- **Tectonic** : Compilateur LaTeX moderne
- **Python** : Pour les scripts de génération de données (si nécessaire)

## 📚 Contenu des Rapports

Les rapports couvrent :
1. Introduction et concepts de base
2. Modèles d'optimisation (Markowitz, Risk Parity, Monte Carlo, etc.)
3. Machine Learning et modèles hybrides
4. Métriques de performance avancées
5. Backtest walk-forward
6. Intégration dans SiraEdge

## 🎯 Utilisation

1. **Consultation** : Ouvrir les PDFs dans le dossier `pdf/`
2. **Modification** : Éditer les fichiers `.tex` dans `sources/`
3. **Recompilation** : Exécuter `./compile_reports.sh`
4. **Organisation** : Maintenir la structure des dossiers

Pour toute question sur l'organisation ou la compilation, consulter le README principal du projet.
