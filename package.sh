#!/bin/bash
# Package the project for submission.
# Run from the project root: bash package.sh
#
# Before running:
#   1. Run the Kaggle notebook to generate figures
#   2. Download fig_*.pdf from /kaggle/working/ into report/figures/
#   3. Compile the report: cd report && pdflatex report && bibtex report && pdflatex report && pdflatex report

set -e

OUT="Tabbane_Soukane_OOD.zip"
rm -f "$OUT"

# Check figures exist
for fig in report/figures/fig_seed_comparison.pdf report/figures/fig_tsne_features.pdf report/figures/fig_confidence_hist.pdf; do
    if [ ! -f "$fig" ]; then
        echo "WARNING: $fig not found. Run the notebook first and download figures."
    fi
done

# Check report PDF exists
if [ ! -f "report/report.pdf" ]; then
    echo "WARNING: report/report.pdf not found. Compile the report first."
fi

zip -r "$OUT" \
    README.md \
    Experiments.md \
    train_ood_kaggle.ipynb \
    report/report.tex \
    report/report.pdf \
    report/references.bib \
    report/midl.cls \
    report/orcid.png \
    report/figures/ \
    -x "report/figures/.gitkeep"

echo ""
echo "Created $OUT ($(du -h "$OUT" | cut -f1))"
echo "Contents:"
unzip -l "$OUT" | tail -n +4 | head -n -2
