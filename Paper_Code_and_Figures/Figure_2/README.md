# Figure 2 — Gene expression noise and its impact on FGESs

Notebook: [`noise.ipynb`](noise.ipynb)

Implements the noise criterion of the pipeline (Methods, *Gene expression noise calculation* and
*Assessment of FGES reproducibility*). Technical noise of each gene is estimated on the pan-cancer
TCGA cohort from expression level, gene length and sample coverage; a gene is called **noisy** if
its noise CI includes values below 0 TPM in ≥ 50% of TCGA samples.

| Panel | Content | Notebook section |
|---|---|---|
| 2A | Expression distribution (TPM) of a noisy, a lowly expressed and a highly expressed gene | *Display noisy and non-noisy genes* |
| 2B | Observed expression vs noise level (TPM) on TCGA for the same genes | *Display noisy and non-noisy genes* |
| 2C | FGES score reproducibility as the share of noisy / lowly / highly expressed genes grows (10 samples with technical replicates) | *Noisy genes vs ssGSEA score STD* → `plots/noise_gene_data_*` |
| 2E | ssGSEA score variation across replicates for a 0%-noisy (Macrophage) and a 40%-noisy (Th2) FGES | *Cytokines vs Macrophages on technical replicates* |
| 2F–G | Permutation test over replicate combinations (20,000 of 442,368) and Pearson correlation of sample ranks | *Cytokines vs Macrophages on technical replicates* |
| S1A | Gene ranks raised to the power 0.25 vs expression (TPM) for a TCGA sample | *TPM vs rank\*\*1/4* |

Panel 2D is a schematic (no code).

Inputs: TCGA expressions and raw counts, gene lengths (`../../Data/gene_length_values.tsv`), and
internal samples with 3–5 technical replicates (available upon request).
