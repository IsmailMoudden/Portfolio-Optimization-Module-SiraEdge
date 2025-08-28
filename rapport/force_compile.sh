#!/bin/bash

echo "🚀 Compilation forcée des rapports..."

cd sources

echo "🔧 Compilation du rapport français (mode forcé)..."
# Utiliser --keep-intermediates pour forcer la compilation même avec des erreurs
tectonic --keep-intermediates rapport_siraedge_fr.tex 2>/dev/null || true

if [ -f "rapport_siraedge_fr.pdf" ]; then
    echo "✅ PDF français généré (même avec des erreurs)"
    mv rapport_siraedge_fr.pdf ../pdf/
else
    echo "⚠️  Échec de la génération du PDF français"
fi

echo "🔧 Compilation du rapport anglais (mode forcé)..."
tectonic --keep-intermediates rapport_siraedge_en.tex 2>/dev/null || true

if [ -f "rapport_siraedge_en.pdf" ]; then
    echo "✅ PDF anglais généré (même avec des erreurs)"
    mv rapport_siraedge_en.pdf ../pdf/
else
    echo "⚠️  Échec de la génération du PDF anglais"
fi

echo "🎯 Compilation forcée terminée !"
echo "📁 Vérifiez les PDFs dans le dossier pdf/"
