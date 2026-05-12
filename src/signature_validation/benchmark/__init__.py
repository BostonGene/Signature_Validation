"""Cell-type signature benchmark on the new sorted-cell test cohort (CHESS-1333 / OD-128).

The pipeline mirrors the published Figure 4 notebook
(``Paper_Code_and_Figures/Figure_4/Cell_type_FGES_comparison/mapping_ssgseas/Signatures comparison.ipynb``)
but reads the new test cohort delivered with Jira OD-128 and reuses the v1
GMT (``msigdb_gmt.pkl``) so chess_db / MSigDb / public / random gene lists are
byte-identical to the published run; only the ssGSEA scoring is rerun on the
new cohort.
"""

from signature_validation.benchmark.cohorts import (
    CONTROLS_ORDER,
    CONTROLS_TO_DELETE,
    EXCLUDED_FGES_RARE,
    MAP_RAW,
    RENAME_NEW_TO_OLD,
    build_mapping,
    intersect_controls_with_cohort,
    load_new_cohort_annotation,
    load_new_cohort_expressions,
)
from signature_validation.benchmark.scoring import (
    compute_mapping_ssgseas,
    compute_out_table,
    fdr_correct_out,
)
from signature_validation.benchmark.signatures import (
    count_random_fges,
    harmonize_gmt_to_index,
    load_v1_msigdb_gmt,
    select_msigdb_gmt_subset,
)
from signature_validation.benchmark.splits import (
    aggregate_score_over_splits,
    stratified_holdout_indices,
)

__all__ = [
    "CONTROLS_ORDER",
    "CONTROLS_TO_DELETE",
    "EXCLUDED_FGES_RARE",
    "MAP_RAW",
    "RENAME_NEW_TO_OLD",
    "aggregate_score_over_splits",
    "build_mapping",
    "compute_mapping_ssgseas",
    "compute_out_table",
    "count_random_fges",
    "fdr_correct_out",
    "harmonize_gmt_to_index",
    "intersect_controls_with_cohort",
    "load_new_cohort_annotation",
    "load_new_cohort_expressions",
    "load_v1_msigdb_gmt",
    "select_msigdb_gmt_subset",
    "stratified_holdout_indices",
]
