# Figure 5 — Process-describing FGESs

Validation of the hypoxia, senescence, EMT, melanoma-specific EMT (EMT-SKCM) and metastasis FGESs
and their comparison with public gene sets (Methods, *Process-describing FGESs*). Each notebook
covers train and test cohorts; scores are compared by Mann–Whitney U with FDR correction.

| Notebook | Panels | Content |
|---|---|---|
| [`Hypoxia_Senescence/Hypoxia_FGES.ipynb`](Hypoxia_Senescence/Hypoxia_FGES.ipynb) | 5A (hypoxia), 5B, 5F, S4B, S4D, S4K / S5C | Cell lines and sorted cells with in vitro induced hypoxia vs controls: gene correlation clustermap, train (S4D) and test (5B) scores, comparison with public hypoxia gene sets (5F, S4K / S5C), performance vs FGES length (S4B) |
| [`Hypoxia_Senescence/Senescence_FGES.ipynb`](Hypoxia_Senescence/Senescence_FGES.ipynb) | 5A (senescence), 5C, 5G, S4A, S4B, S4C, S4J / S5B | Cells with induced senescence (radiation, Ras, doxorubicin, replicative) vs controls: clustermap, train (S4C, GSE130727) and test (5C) scores, comparison with public senescence gene sets (5G, S4J / S5B), Venn diagram of gene overlap (S4A), performance vs FGES length (S4B) |
| [`EMT_Met_compare/EMT_Met_compare.ipynb`](EMT_Met_compare/EMT_Met_compare.ipynb) | 5D, 5E, S4E–I, S5A | EMT, EMT-SKCM and metastasis FGESs by tumor grade, TCGA-SKCM pigmentation score and GSE48091 metastasis status (5D); median-score differences vs Hallmark EMT and published metastasis signatures (5E, waterfall plots); train / test boxplots (S4E–I, S5A) |

Main cohorts: TCGA-LUAD and basal-like TCGA-BRCA with internal histological grades (EMT training),
TCGA-SKCM with depigmentation score, GSE103584 (NSCLC grades), GSE48091 (breast cancer metastasis),
plus the sorted-cell / cell-line datasets listed in Supplementary Table S8.3.

The progression-free survival forest plots (Supplementary Figure S5D) are not part of this
repository.
