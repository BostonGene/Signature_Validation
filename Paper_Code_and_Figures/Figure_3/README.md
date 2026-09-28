# Figure 3 — Concordance of gene expression and FGES specificity

Notebook: [`correlation.ipynb`](correlation.ipynb)

Tests the gene inter-correlation criterion on the public macrophage FGES (Bindea et al. 2013)
(Methods, *Gene correlation analysis*).

| Panel | Content | Code |
|---|---|---|
| 3A | Cohen's d between ssGSEA scores of macrophages (GOI) and M1 macrophages / monocytes / fibroblasts while lowly correlated genes are removed one by one, least correlated first | `correlation.ipynb` → `Figures/Macrophages_corr.svg` |
| 3B | Spearman correlation heatmaps (TCGA, n = 9,863) of the original, the trimmed and the refined macrophage FGES | [`../Figure_4/Macrophage_FGES_cells_and_TCGA.ipynb`](../Figure_4/Macrophage_FGES_cells_and_TCGA.ipynb), section *Check if there are anticorrelating genes* |

Inputs: sorted-cell RNA-seq expressions and annotation (train part of the public sorted-cell
collection), TCGA expressions.
