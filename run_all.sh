#!/usr/bin/env bash
# Reproduce every result, table and figure in the manuscript and supplement.
# Usage: bash run_all.sh <TCGA_dir> <GEO_dir>
#   TCGA_dir: extracted uvm_tcga_pan_can_atlas_2018/ from cBioPortal
#   GEO_dir : GSE22138.txt.gz, GSE44295.txt.gz (series matrix), GPL570.annot.gz, GPL6883.annot.gz
# Long steps checkpoint per fold and resume if interrupted. Total ~7 h on 2 cores.
set -euo pipefail
TCGA=${1:?TCGA dir}; GEO=${2:?GEO dir}
export CNA_TXT="$TCGA/data_cna.txt"                                   # cytobands for the positional prior
python src/preprocess.py "$TCGA" processed                             # TCGA cleaning
python src/run_cv.py processed results 10                              # TCGA CV: baselines, RL selector (original), ablations
python src/run_cv_plus.py processed results 10                         # TCGA CV: multi-agent RL variants (supplement)
python src/run_v2.py processed results 0 4 boot cv                     # RL selector with fresh splits + policy ensemble, repeats 1-5
python src/run_v2.py processed results 5 9 boot                        # same, repeats 6-10
python src/compare_v2.py results 0 9
python src/pooled.py processed "$GEO" results loco                     # leave-one-cohort-out
python src/pooled.py processed "$GEO" results cv 0 9                   # pooled CV, 199 patients
python src/ablation_alasso.py processed "$GEO" results                 # direct control (TCGA, pooled)
python src/review_extra.py processed "$GEO" results "$CNA_TXT"         # positional prior, 15-gene proxy, 5-seed LOCO control
python src/pooled_summary.py results "Anchor-weighted adaptive LASSO"
python src/analyze.py results figures "Anchor-weighted adaptive LASSO" # Figs S1-S3
python src/stability.py results 11802 figures                          # Table 4, Fig 3 (Nogueira Phi)
python src/fig_three_designs.py results figures                        # Table 2, Fig 2
python src/external.py processed "$GEO" results_external               # fixed-model external evaluation
python src/external_stats.py results_external "$GEO" figures "Anchored adaptive LASSO"   # Tables 5, 6, S2; Fig 4
python src/external_extra.py processed results_external "$GEO" figures  # calibration (Table S6, Fig S4), early events, markers + score, scoring sheet
python src/review2.py processed "$GEO" results "$CNA_TXT"              # permuted/rescaled controls, sensitivity, markers + score, stratified Cox
python src/review2_summary.py results                                  # Tables 3-4, TCGA paired tests
python src/simulation.py processed results "$CNA_TXT" 50               # semi-synthetic simulation (~55 min on one core)
python src/fig_simulation.py results figures                           # Fig 5
python src/runtimes.py processed results                               # wall-clock times
python src/fig1_schematic.py figures                                   # Fig 1
