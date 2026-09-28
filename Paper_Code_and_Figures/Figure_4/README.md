# Figure 4 — Highly specific cell type-specific FGESs

Two parts: the step-by-step refinement of the macrophage FGES (panels A–D) and the sorted-cell
benchmark of the internal cell type-specific FGESs against public gene sets (panels E–I).

## Macrophage FGES refinement (4A–D, S2)

| Notebook | Panels | Content |
|---|---|---|
| [`Macrophage_FGES_cells_and_TCGA.ipynb`](Macrophage_FGES_cells_and_TCGA.ipynb) | 4A, 4B, 3B, S2A–B, S2D–E | TCGA expression violins of the original (Bindea et al. 2013) and refined macrophage FGES (4A); unscaled ssGSEA scores in sorted cells, Mann–Whitney U (4B, full comparisons in Table S4); gene correlation heatmaps (3B); good vs poor gene examples *VSIG4* / *SCG5* / *SCARB2* in TCGA, cancer cell lines and sorted cells (S2) |
| [`Macrophage_FGES_scRNA.ipynb`](Macrophage_FGES_scRNA.ipynb) | 4C, 4D, S2C | UMAPs annotated by cell population and by original / refined FGES scores (4C); normalized (0.2–1.2) ssGSEA scores per cell type in five scRNA-seq datasets (4D, p-values in Table S5): GSE178341, GSE144735, GSE103322, GSE158803 and the internal NSCLC dataset (Table 1) |

The signature is defined in a *Definition of custom signature* cell, so the same notebooks can be
re-run for another FGES (e.g. the Treg FGES of Supplementary Figure S3).

## Cell type-specific FGES benchmark (4E–I)

Folder: [`Cell_type_FGES_comparison/`](Cell_type_FGES_comparison/) · notebook:
[`mapping_ssgseas/Signatures comparison NEW.ipynb`](Cell_type_FGES_comparison/mapping_ssgseas/Signatures%20comparison%20NEW.ipynb)

Internal FGESs are compared with public gene sets (MSigDB, Bindea et al. 2013, Nirmal et al. 2018,
xCell / Aran et al. 2017) and random FGESs on sorted-cell bulk RNA-seq (Methods, *Comparison of FGES
performance*). Logic lives in `signature_validation.benchmark`.

| Panel | Content |
|---|---|
| 4E | Row-normalized median ssGSEA scores of the top-10 macrophage FGESs by F-score |
| 4F | Bootstrap F-score and CV of macrophage FGESs (95% CI); random-prediction and random-FGES reference lines |
| 4G | F-score and CV averaged by FGES source across all cell types |
| 4H | Median unscaled ssGSEA scores of the internal FGESs across sorted cell types |
| 4I | Unscaled ssGSEA scores in GOI vs control samples, stratified by FGES source |

Also produces the per-cell-type FGES performance tables (Supplementary Table S6) and the sorted-cell
dataset inventory with scored sample counts (Supplementary Table S8.1; named `S6.1` in the notebook
code).

How the benchmark is set up:

- **GOI / Control / Deleted_controls.** For every FGES, the GOI cell type(s), control cell types
  and *deleted controls* — parental or descendant cell types biologically overlapping with the GOI,
  excluded from the control pool — are defined in `benchmark/cohorts.py`. The three groups are
  never merged.
- **Held-out by sample.** Samples already present in the train cohort used to develop the
  signatures are dropped before scoring.
- **Rare cell types.** Cell types with too few held-out samples are backfilled with
  cross-validated scores from the train cohort and marked in the plots (`*` / `†` / `‡`), never
  silently pooled.
- **Sample counts** in tables are counts of *scored* samples, not annotation rows.
- **`RECOMPUTE` toggle.** `True` recomputes ssGSEA and metrics (needs expression data, slow);
  `False` loads cached pickles and only re-plots.
- `detect_fges_source` reads `./data/msigdb.v2023.1.Hs.symbols.gmt` relative to the working
  directory, so that call runs from `Cell_type_FGES_comparison/`.

New outputs go to `Cell_type_FGES_comparison/plots/new_cohort/` (git-ignored); the committed SVGs in
`Cell_type_FGES_comparison/plots/` are the published versions.
