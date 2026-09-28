"""Cross-validation backfill for rare cell types in the validation cohort.

The true validation cohort (new sorted cells) has too few — or zero — samples
for several GOI / control cell types (e.g. ``Th17_cells``, ``Endothelium_lymph``,
``Plasma_B_cells``). Rather than excluding those FGES from the analysis, per-cell-
type scores below a sample-count threshold are filled in from the pre-computed
cross-validated scores of the **original** cohort (10x stratified 75/25 holdout,
mean-aggregated across folds — the same recipe the retired rare-types notebook used
for just four FGES, now precomputed for all of them and stored in
``mapping_ssgseas_crossval.pkl``).

Two fill-in strategies, chosen with ``mode``:

``"merge"`` (default)
    Test and train samples are **pooled** and the score is taken over all of them,
    so a short cell type keeps whatever real holdout data it has instead of having
    it thrown away.
``"substitute"``
    The historical behaviour — the true-cohort frame is replaced outright.

Every fill-in is recorded in a provenance dict so downstream figures can label each
cell type as ``"true_holdout"`` (genuine new-cohort samples), ``"cross_validation"``
(entirely from the original-cohort rerun) or ``"merged"`` (both pooled).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Literal, Tuple, Union

import pandas as pd
import s3fs
from loguru import logger

ProvenanceLabel = Literal["true_holdout", "cross_validation", "merged"]
Provenance = Dict[str, Dict[str, Dict[str, ProvenanceLabel]]]

DEFAULT_MIN_N = 20


def load_crossval_ssgseas(
    path: Union[str, Path] = "<PATH_TO_CROSSVAL_MAPPING_SSGSEAS_PKL>",
) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
    """Load the pre-computed cross-validated original-cohort ssGSEA scores.

    Parameters
    ----------
    path : str or Path
        ``s3://...`` URI or local filesystem path. Same nested shape as
        :func:`signature_validation.benchmark.scoring.compute_mapping_ssgseas`'s
        output: ``{Main4_*: {Goi|Control|Deleted_controls: {cell_type: DataFrame}}}``.

    Returns
    -------
    dict
    """
    path_str = str(path)
    if path_str.startswith("s3://"):
        fs = s3fs.S3FileSystem()
        with fs.open(path_str, "rb") as handle:
            crossval = pickle.load(handle)
    else:
        with open(path_str, "rb") as handle:
            crossval = pickle.load(handle)
    logger.info(
        "loaded crossval ssGSEA scores from {p}: {n} FGES",
        p=path,
        n=len(crossval),
    )
    return crossval


def _merge_score_frames(true_df: pd.DataFrame, cv_df: pd.DataFrame) -> pd.DataFrame:
    """Pool already-computed ssGSEA scores of the test and train cohorts.

    Sound only because ssGSEA is a single-sample method — ``expressions.rank()``
    runs per column, so a sample's score never depends on which other samples are
    in the matrix. Pooling the two score frames is therefore arithmetically the
    same as rescoring the union of raw expressions, which is what the crossval
    pickle (scores only, no expressions) could not otherwise support.

    Columns are intersected rather than assumed identical: the crossval pickle's
    generating script is not in this repo, so a gene-set version drift would
    otherwise corrupt the pooled medians silently instead of raising a warning.
    Duplicate sample IDs resolve in favour of the test cohort — keeping both would
    give one donor double weight in the median.
    """
    shared = true_df.columns.intersection(cv_df.columns)
    n_dropped = len(true_df.columns) - len(shared)
    if n_dropped:
        logger.warning(
            "merge: {n} of {t} signatures missing from the crossval frame — dropped",
            n=n_dropped,
            t=len(true_df.columns),
        )
    if shared.empty:
        logger.error(
            "merge: the two cohorts share no signature columns; keeping test-only data"
        )
        return true_df

    pooled = pd.concat([true_df[shared], cv_df[shared]])
    return pooled[~pooled.index.duplicated(keep="first")]


def backfill_rare_cell_types(
    mapping_ssgseas: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
    crossval_ssgseas: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
    annotation: pd.DataFrame,
    min_n: int = DEFAULT_MIN_N,
    mode: Literal["merge", "substitute"] = "merge",
) -> Tuple[Dict[str, Dict[str, Dict[str, pd.DataFrame]]], Provenance]:
    """Fill in crossval scores for any cell type with ``< min_n`` true-cohort samples.

    For each ``(FGES, group, cell_type)``, the true-validation-cohort frame is kept
    as-is when the validation annotation has at least ``min_n`` QC-passing samples
    of that cell type *and* :func:`compute_mapping_ssgseas` actually scored it.
    Below that threshold the crossval frame comes into play, in one of two ways.

    Parameters
    ----------
    mapping_ssgseas : dict
        Output of :func:`signature_validation.benchmark.scoring.compute_mapping_ssgseas`
        run on the true validation cohort.
    crossval_ssgseas : dict
        Output of :func:`load_crossval_ssgseas`.
    annotation : pd.DataFrame
        True validation annotation (post-QC), indexed by sample, with a
        ``Cell_type`` column — used only to count samples per cell type.
    min_n : int
        Minimum true-cohort sample count required to keep the true-cohort frame.
    mode : {"merge", "substitute"}
        ``"merge"`` pools the test and train samples and scores over all of them,
        so the few genuine holdout samples are not discarded; the cell type is
        labelled ``"merged"``. ``"substitute"`` replaces the test frame outright,
        the historical behaviour. Cell types absent from the test cohort entirely
        are ``"cross_validation"`` either way — there is nothing to merge with.

    Returns
    -------
    (dict, dict)
        The merged ``mapping_ssgseas`` (same nested shape) and a provenance dict
        ``{FGES: {group: {cell_type: "true_holdout" | "cross_validation" | "merged"}}}``.
    """
    counts = annotation["Cell_type"].value_counts()
    merged: Dict[str, Dict[str, Dict[str, pd.DataFrame]]] = {}
    provenance: Provenance = {}
    n_backfilled = 0
    n_merged = 0
    merged_cell_types: set = set()
    n_unavailable: List[str] = []

    all_signs = sorted(set(mapping_ssgseas) | set(crossval_ssgseas))
    for sign in all_signs:
        true_groups = mapping_ssgseas.get(sign, {})
        cv_groups = crossval_ssgseas.get(sign, {})
        merged[sign] = {}
        provenance[sign] = {}
        for group in ("Goi", "Control", "Deleted_controls"):
            true_cts = true_groups.get(group, {})
            cv_cts = cv_groups.get(group, {})
            merged_group: Dict[str, pd.DataFrame] = {}
            group_provenance: Dict[str, str] = {}
            for ct in sorted(set(true_cts) | set(cv_cts)):
                n_true = int(counts.get(ct, 0))
                if n_true >= min_n and ct in true_cts:
                    merged_group[ct] = true_cts[ct]
                    group_provenance[ct] = "true_holdout"
                elif ct in cv_cts and ct in true_cts and mode == "merge":
                    # Too few test samples to stand alone, but they are real
                    # holdout data — pool them with train instead of discarding.
                    merged_group[ct] = _merge_score_frames(true_cts[ct], cv_cts[ct])
                    group_provenance[ct] = "merged"
                    n_merged += 1
                    merged_cell_types.add(ct)
                elif ct in cv_cts:
                    merged_group[ct] = cv_cts[ct]
                    group_provenance[ct] = "cross_validation"
                    n_backfilled += 1
                elif ct in true_cts:
                    # Below min_n but no crossval fallback available; keep what
                    # the true cohort has rather than dropping the cell type.
                    merged_group[ct] = true_cts[ct]
                    group_provenance[ct] = "true_holdout"
                    n_unavailable.append(f"{sign}/{group}/{ct} (n={n_true})")
            merged[sign][group] = merged_group
            provenance[sign][group] = group_provenance

    logger.info(
        "crossval backfill (mode={md}, min_n={m}): {n} frames substituted, "
        "{k} frames merged test+train",
        md=mode,
        m=min_n,
        n=n_backfilled,
        k=n_merged,
    )
    if merged_cell_types:
        logger.info(
            "merged cell types (test+train pooled, marked on figures): {c}",
            c=sorted(merged_cell_types),
        )
    if n_unavailable:
        logger.warning(
            "below min_n with no crossval fallback available, kept true-cohort data: {ls}",
            ls=n_unavailable,
        )
    return merged, provenance
