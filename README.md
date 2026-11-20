# Stress-testing BBB-permeability classifiers under structure-novel splits

**Question.** How much of the accuracy reported for BBB-permeability models on the DeePred-BBB dataset (Kumar et al., Front. Neurosci. 2022) survives when the test set is structurally novel, and does a graph model (GIN) change the answer compared with fingerprint/descriptor baselines?

This is an evaluation study, not a rebuttal. It uses the 3,605-compound dataset from the DeePred-BBB paper (Table 1) but **not** the authors' exact train/test partition, which is not public (the repository holds unlabelled PaDEL features and no split indices). Numbers here are therefore not directly comparable with the paper's 98.07% accuracy.

## Data
| Item | Value |
|---|---|
| Source | DeePred-BBB Table 1 (Roy 2019, Zhao 2007, Shen 2010) |
| Rows / parse failures | 3,605 / 37 |
| Clean unique molecules | 3,482 (72.6% permeable) |
| Distinct structures ignoring stereochemistry | 1,913 |
| Murcko scaffolds | 1,010 (357 singletons) |
| Label conflicts after standardisation | 0 |
| Overlap with B3DB / Martins | 91.5% / 36.3% (so neither is an independent test) |

Standardisation: RDKit largest fragment + neutralisation, canonical SMILES; exact duplicates removed.

## Splits (about 25% test, 5 seeds each)
| Split | Median max-Tanimoto test→train | Test compounds with ≥0.9 neighbour | Test compounds whose flat structure is in train |
|---|---|---|---|
| random_rows (plain random) | 1.00 | 66% | 65% |
| random_groups (isomers kept together) | 0.60 | 1.4% | 0% |
| scaffold_rand (whole scaffolds held out) | 0.47 | 0.7% | 0% |
| cluster_butina (Butina clusters, distance cut-off 0.6) | 0.39 | 0% | 0% |

Source-held-out splits were computed but are not used for model comparison: after removing structures shared across sources the Roy and Shen folds had only 94 and 38 compounds.

## Models
Logistic regression (descriptors), random forest (ECFP4; descriptors+ECFP4), LightGBM (descriptors), XGBoost (default and Optuna, 20 trials inside each outer fold), an MLP (descriptors+MACCS), and a GIN (graph only, and graph+descriptors; grid of 4 configurations chosen on an inner grouped validation split). Threshold fixed at 0.5. Features: RDKit 2D descriptors, ECFP4 (2048 bits), MACCS.

## Main results (mean over 5 seeds; MCC)
| Model | random_rows | random_groups | scaffold_rand | cluster_butina |
|---|---|---|---|---|
| LightGBM (desc) | 0.949 | 0.869 | 0.790 | 0.793 |
| RF (desc+ECFP4) | 0.935 | 0.861 | 0.792 | 0.787 |
| MLP (desc+MACCS) | 0.931 | 0.851 | 0.783 | 0.779 |
| LR (desc) | 0.851 | 0.839 | 0.782 | 0.772 |
| GIN | 0.895 | 0.822 | 0.768 | 0.710 |
| GIN + desc | 0.904 | 0.836 | 0.786 | 0.773 |

Accuracy of always predicting "permeable" is about 0.73. GIN numbers are from a full `run_all.py` re-run; the GIN is not bit-reproducible (thread scheduling), and two runs differed by up to about 0.01 in MCC and in AUROC. All other models reproduced exactly. Full tables: `baseline_summary.csv`, `xgb_results_per_run.csv`, `gin_results_per_run.csv`, paired comparisons in `xgb_paired_vs_baselines.csv` and `gin_paired_vs_lgbm.csv`.

## Findings
1. A plain random split places a stereoisomer of the same flat structure in training for 65% of test compounds, so it is a weak test of generalisation.
2. Under scaffold- and cluster-held-out splits, accuracy falls by about 6–8 points and MCC by about 0.15, but accuracy stays near 0.92, so the signal is real.
3. Tuned XGBoost did not improve on library defaults, and neither beat LightGBM; the graph-only GIN trailed the tree models, with the gap growing on harder splits (AUROC −0.008 on random rows, −0.041 on the cluster split). Adding descriptors to the GIN closed most of the gap.
4. Error rate falls with similarity to the training set: about 17% when max Tanimoto < 0.3, 8.6% for 0.3–0.4, and 0.2% at ≥ 0.9 (`error_by_similarity_bin.csv`).

## Limitations
- Five seeds per split; differences of about 0.01 between tree models are within noise, and no formal significance tests were run.
- The exact DeePred-BBB split and its PaDEL-feature DNN were not evaluated; PaDEL (Java) was not run.
- Fixed 0.5 threshold; no threshold tuning or calibration.
- GIN: small grid, at most 80 epochs, selected model trained on 80% of the training fold.
- The cluster cut-off and the 25% test fraction are design choices; results depend on them.
- Pooled error-by-similarity counts reuse compounds across seeds, so they overstate the effective sample size.
- Similarity is ECFP4 Tanimoto only.

## Reproduce
```
pip install -r requirements.txt     # or the conda env used here
python run_all.py                   # from this folder; add `--skip 05 06` to omit the slow steps
```
Needs the DeePred-BBB Table 1 spreadsheet (`Table 1.XLSX`, from the paper's supplementary material) next to `run_all.py` or one level up, or set `TABLE1_PATH`. It is not included in this repository. Scripts are `01_audit.py` … `08_summaries.py`; they read `data/` and `results/` relative to the working directory.

## Licence
Code in this repository is released under the MIT licence (see `LICENSE`). The licence covers the code only. The DeePred-BBB dataset (Table 1 of Kumar et al., 2022), BBBP and B3DB have their own terms; they are not distributed here.
