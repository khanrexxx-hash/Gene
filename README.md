# Omics-anchored gene priors for uveal melanoma survival signatures

Code and results for *"Omics-Anchored Gene Priors for Compact Uveal Melanoma Survival Signatures:
A Controlled Multi-Cohort Evaluation"*.

Each gene gets an **anchoring score** — the strongest correlation of its expression with its own copy
number or methylation in the training cohort. The score sets the penalty weights of an adaptive LASSO
Cox model; the deployed signature needs expression only. The study compares it with the identical
model without anchoring (direct control), a positional prior (genes on chromosome 3, 8q, 1p), a
15-gene-class proxy, standard penalised models, a random survival forest, a univariate screen and a
reinforcement-learning selector (supplement). Controls for weight spread (permuted anchoring, rescaled prior), cohort-stratified Cox models and a semi-synthetic simulation are included.

Everything runs on a laptop CPU; no GPU needed.

## Data (public, de-identified)

| Cohort | Role | Source |
|---|---|---|
| TCGA-UVM PanCancer Atlas (79) | training, anchoring | cBioPortal `uvm_tcga_pan_can_atlas_2018` |
| GSE22138 (Affymetrix GPL570, 63) | pooled / held-out / external | GEO series matrix + `GPL570.annot.gz` |
| GSE44295 (Illumina GPL6883, 57 tumours; endpoint metastatic death) | pooled / held-out / external (supporting) | GEO series matrix + `GPL6883.annot.gz` |

## Reproduce

```bash
pip install -r requirements.txt
bash run_all.sh /path/to/uvm_tcga_pan_can_atlas_2018 /path/to/geo_dir
```

## Main findings

| | Anchored | No anchoring (control) | Permuted anchoring | Rescaled prior | Positional prior |
|---|---|---|---|---|---|
| TCGA 10×5 CV | 0.775 | 0.763 | 0.770 | 0.765 | 0.774 |
| Pooled 10×5 CV (199) | 0.784 | 0.769 | 0.771 | 0.783 | 0.780 |
| Pooled, cohort-stratified Cox | 0.783 | 0.779 | – | – | 0.783 |
| Leave-one-cohort-out mean | 0.766 | 0.759 | – | 0.766 | 0.768 |
| External GSE22138 / GSE44295 | 0.734 / 0.780 | 0.717 / 0.708 | – | – | 0.745 / 0.780 |
| Stability Φ (TCGA) | 0.292 | 0.247 | – | – | 0.275 |

- Anchored vs control, pooled: +0.015, corrected p = 0.22 (minimum detectable 0.035). Permuted anchoring removes the gain and a variance-matched prior keeps it, so the gain is not an artefact of weight spread. With cohort-stratified fitting it shrinks to +0.004.
- The positional prior matches anchoring everywhere: in UM anchoring rediscovers the known risk arms.
- Simulation on the real omics: anchoring helps (up to +0.016) only when prognostic cis-regulated genes lie outside the known arms; no harm otherwise.
- Molecular markers alone (chr3, 8q, BAP1) = 0.770 in TCGA CV; adding the score does not help (0.751).
- External calibration slopes 0.57 / 0.66 (predictions too extreme); Cox-LASSO and elastic net calibration not estimable (apparent TCGA C ≈ 1).
- 4-gene signature (SLC25A38, JPH1, CDCA7L, KLHL21): HR 2.45 per SD adjusted for monosomy 3 (p = 0.001). Single-patient scoring sheet: `results_external/single_patient_scoring_sheet.csv`.

## Code

| File | Purpose |
|---|---|
| `src/preprocess.py` | TCGA cleaning |
| `src/rl_agent.py`, `src/rl_agent2.py` | Ridge-Cox solver, C-index, anchoring score, reinforcement-learning selectors (supplement; originally named OASIS = Omics-Anchored, Stability-rewarded Interpretable Selection) |
| `src/strat_lasso.py` | Cohort-stratified weighted LASSO Cox (FISTA), stratified screen and C-index |
| `src/run_cv.py`, `run_cv_plus.py`, `run_v2.py` | TCGA cross-validation |
| `src/pooled.py` | Pooled CV and leave-one-cohort-out |
| `src/ablation_alasso.py`, `review_extra.py` | Direct control, positional prior, 15-gene-class proxy |
| `src/review2.py`, `review2_summary.py` | Permuted / rescaled controls, positional + 6p, sensitivity table, markers + score, stratified Cox, paired tests |
| `src/simulation.py`, `fig_simulation.py` | Semi-synthetic simulation on the real TCGA omics (Fig 5, Table S7) |
| `src/stability.py` | Nogueira Φ and Jaccard (Table 5, Fig 3) |
| `src/external.py`, `external_stats.py`, `external_extra.py` | Fixed-model external evaluation: Uno's C, td-AUC, Brier, KM, monosomy-3 adjustment, calibration slope / in-the-large / KM-null Brier, early-event sensitivity, markers + score, scoring sheet |
| `src/runtimes.py` | Wall-clock time per fit |
| `src/analyze.py`, `compare_v2.py`, `pooled_summary.py`, `fig_three_designs.py`, `fig1_schematic.py` | Statistics, tables, figures |

## Data not included

Raw TCGA and GEO files are not redistributed here; download them from the sources above.
The `processed/` folder is created by `src/preprocess.py`.

## Citation

If you use this code, please cite the paper (citation and DOI will be added on publication).

## License

MIT, see `LICENSE`.
