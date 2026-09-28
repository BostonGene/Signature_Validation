"""Cohort assembly for the new sorted-cell test cohort.

Functions here load the new annotation / expressions and
build the v1-style ``mapping`` dict (GOI / Control / Deleted_controls per FGES),
restricted to the 16 in-scope FGES and to the cell types present in the new
cohort.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

from signature_validation.utils.utils import read_dataset, read_expressions

# New annotation cell-type labels → existing pipeline cell-type names.
RENAME_NEW_TO_OLD: Dict[str, str] = {
    "Follicular_T_helper": "Follicular_T_helpers",
    "T_helper_1": "Th1_cells",
    "T_helper_2": "Th2_cells",
    "Natural_killer_cells": "NK_cells",
    "Plasma_cells": "Plasma_B_cells",
    "Regulatory_CD4_T_cells": "Tregs",
}

# Cell-type labels that differ between the published (train) cohort and the new
# (holdout) one but name the same population. Only needed where the two cohorts
# are put side by side — the per-dataset inventory of Supplement S6.1 — because
# there the raw labels would list one population twice under two names.
# ``Follicular_T_helper_tonsil`` is the v1 label; ``Follicular_T_helper`` is the
# raw new-annotation label that :data:`RENAME_NEW_TO_OLD` already normalises.
HARMONIZE_TRAIN_TO_NEW: Dict[str, str] = {
    "Follicular_T_helper_tonsil": "Follicular_T_helpers",
    "Follicular_T_helper": "Follicular_T_helpers",
}

# Sample-level annotation behind the published (train) cohort — the table the v1
# notebook built ``public_cells_annot`` from. Needed to attribute train samples to
# their source dataset in Supplement S6.1.
TRAIN_ANNOT_PATH: Path = Path("<PATH_TO_TRAIN_CELLS_ANNOTATION_TSV>")

# FGES with no (or very few) new-cohort samples for their GOI cell type. Not
# excluded from the validation pipeline: the crossval backfill
# (see benchmark.crossval) sources these entirely from the cross-validated
# original-cohort pickle instead, tagged "cross_validation" in provenance.
EXCLUDED_FGES_RARE: Set[str] = {
    "Main4_Th17_signature",
    "Main4_Lymphatic_endothelium",
    "Main4_Eosinophil_signature",
    "Main4_Plasma_cells",
}

# GOI / control-deletion definitions for the four FGES above (v1 cell 36-37),
# merged into MAP_RAW / CONTROLS_TO_DELETE by the validation pipeline so all
# 19 FGES flow through one uniform <20-sample crossval-backfill rule instead
# of a separate rare-types rerun.
MAP_RAW_RARE: Dict[str, List[str]] = {
    "Main4_Th17_signature": ["Th17_cells"],
    "Main4_Lymphatic_endothelium": ["Endothelium_lymph"],
    "Main4_Eosinophil_signature": ["Eosinophils"],
    "Main4_Plasma_cells": ["Plasma_B_cells", "Plasmablasts"],
}

CONTROLS_TO_DELETE_RARE: Dict[str, List[str]] = {
    "Main4_Th17_signature": [
        "T_cells",
        "CD4_T_cells",
        "CD4_T_helpers",
        "Memory_CD4_T_cells",
    ],
    "Main4_Lymphatic_endothelium": ["Endothelium"],
    "Main4_Eosinophil_signature": ["Myeloid_cells"],
    "Main4_Plasma_cells": ["B_cells"],
}

# Per-FGES list of GOI cell types (v1 cell 36, restricted to the 16 in-scope FGES,
# with Follicular_T_helper_tonsil → Follicular_T_helpers and CD4_T_helpers
# dropped because the new cohort does not provide them).
MAP_RAW: Dict[str, List[str]] = {
    "Main4_Th1_signature": ["Th1_cells"],
    "Main4_CD8_T_cells": ["CD8_T_cells"],
    "Main4_Treg": ["Tregs"],
    "Main4_Neutrophil_signature": ["Neutrophils"],
    "Main4_Mast_cell_signature": ["Mast_cells"],
    "Main4_Effector_cells": ["CD8_T_cells", "NK_cells"],
    "Main4_Follicular_helper_T_cells": ["Follicular_T_helpers"],
    "Main4_B_cells": ["B_cells"],
    "Main4_Endothelium": ["Endothelium"],
    "Main4_Pan_macrophage_signature": ["Macrophages"],
    "Main4_NK_cells": ["NK_cells"],
    "Main4_M2_signature": ["Macrophages_M2"],
    "Main4_T_cells": ["T_cells"],
    "Main4_CD4_T_cells": ["CD4_T_cells"],
    "Main4_Monocyte": ["Monocytes"],
}

# v1 cell 36 controls_order — the canonical column / x-axis order for figures.
# `intersect_controls_with_cohort` filters this down to types actually present
# in the new annotation.
CONTROLS_ORDER: List[str] = [
    "T_cells",
    "CD4_T_helpers",
    "CD4_T_cells",
    "PD1_CD4_T_cells",
    "Memory_CD4_T_cells",
    "Th1_cells",
    "Th2_cells",
    "Th2",
    "Th17_cells",
    "Follicular_T_helper_tonsil",
    "Tregs",
    "CD8_T_cells",
    "Memory_CD8_T_cells",
    "CD8_T_cells_PD1_high",
    "NK_cells",
    "B_cells",
    "Plasma_B_cells",
    "Plasmablasts",
    "Non_plasma_B_cells",
    "Myeloid_cells",
    "Neutrophils",
    "Eosinophils",
    "Mast_cells",
    "Monocytes",
    "Macrophages",
    "Macrophages_M1",
    "Macrophages_M2",
    "Monocytic_DC",
    "Dendritic_cells",
    "Fibroblasts",
    "Cardiac_myofibroblasts",
    "Endothelium",
    "Endothelium_lymph",
    "Hepatocytes",
    "Astrocytes",
    "Bronchial_cells",
    "Epithelium",
    "Fibroblast_line",
    "Follicular_T_helper",
    "Keratinocytes",
    "MAIT_cells",
    "MSC",
    "Neurons",
    "Pancreatic_cells",
    "iPSC",
]

# v1 cell 37: per-FGES exclusions of cognate cell types from the control set.
CONTROLS_TO_DELETE: Dict[str, List[str]] = {
    "Main4_Th1_signature": [
        "T_cells",
        "CD4_T_cells",
        "CD4_T_helpers",
        "Memory_CD4_T_cells",
    ],
    "Main4_CD8_T_cells": [
        "T_cells",
        "Memory_CD8_T_cells",
        "CD8_T_cells_PD1_high",
        "MAIT_cells",
    ],
    "Main4_Treg": ["T_cells", "CD4_T_cells", "Memory_CD4_T_cells"],
    "Main4_Neutrophil_signature": ["Myeloid_cells"],
    "Main4_Mast_cell_signature": ["Myeloid_cells"],
    "Main4_Effector_cells": ["T_cells"],
    "Main4_Follicular_helper_T_cells": [
        "T_cells",
        "CD4_T_cells",
        "CD4_T_helpers",
        "Memory_CD4_T_cells",
    ],
    "Main4_B_cells": ["Plasma_B_cells", "Non_plasma_B_cells", "Plasmablasts"],
    "Main4_Endothelium": ["Endothelium_lymph"],
    "Main4_NK_cells": [],
    "Main4_M2_signature": ["Macrophages", "Myeloid_cells", "Monocytes"],
    "Main4_T_cells": [
        "CD4_T_helpers",
        "CD4_T_cells",
        "PD1_CD4_T_cells",
        "Memory_CD4_T_cells",
        "Th1_cells",
        "Th17_cells",
        "Th2_cells",
        "Th2",
        "Follicular_T_helper_tonsil",
        "Tregs",
        "CD8_T_cells",
        "Memory_CD8_T_cells",
        "CD8_T_cells_PD1_high",
    ],
    "Main4_CD4_T_cells": [
        "PD1_CD4_T_cells",
        "T_cells",
        "Th1_cells",
        "Th17_cells",
        "Follicular_T_helper_tonsil",
        "Tregs",
    ],
    "Main4_Pan_macrophage_signature": [
        "Macrophages_M1",
        "Macrophages_M2",
        "Myeloid_cells",
        "Monocytes",
    ],
    "Main4_Monocyte": [
        "Macrophages",
        "Myeloid_cells",
        "Macrophages_M1",
        "Macrophages_M2",
        "Monocytic_DC",
    ],
}

# v1 Scater_plots cell 16: for each parent FGES, the daughter FGES whose
# sub-signature columns must be stripped out of the parent's GOI frames before
# scoring/ranking (a signature shared by parent and daughter would otherwise be
# double-counted, and — in ``compute_out_table`` — silently overwritten by the
# last FGES processed). Daughters not present in the in-scope run are skipped by
# the cleaner. Kept identical to v1 so macrophage/monocyte rows match the paper.
PARENT_TO_DAUGHTER: Dict[str, List[str]] = {
    "Main4_T_cells": [
        "Main4_Th17_signature",
        "Main4_Th1_signature",
        "Main4_CD8_T_cells",
        "Main4_Treg",
        "Main4_Effector_cells",
        "Main4_CD4_T_cells",
        "Main4_Follicular_helper_T_cells",
    ],
    "Main4_CD4_T_cells": [
        "Main4_Th17_signature",
        "Main4_Th1_signature",
        "Main4_Follicular_helper_T_cells",
        "Main4_Treg",
    ],
    "Main4_B_cells": ["Main4_Plasma_cells"],
    "Main4_Pan_macrophage_signature": ["Main4_M2_signature"],
    "Main4_Monocyte": ["Main4_Pan_macrophage_signature", "Main4_M2_signature"],
    "Main4_Endothelium": ["Main4_Lymphatic_endothelium"],
}


def load_new_cohort_annotation(
    path: Union[str, Path],
    rename_map: Optional[Dict[str, str]] = None,
    apply_rename: bool = True,
) -> pd.DataFrame:
    """Load the new sorted-cell annotation TSV.

    The file is already filtered by ``Technical_QC == True`` and
    ``Decision_deconvolution_without_parent != False`` and the rename to
    pipeline names is already applied; the rename pass here
    is idempotent and defensive. ``Cell_type`` is stripped of leading/trailing
    whitespace before renaming, since raw labels can carry stray whitespace
    (e.g. ``"T_helper_1 "``) that silently breaks an exact-match ``.replace()``.

    Parameters
    ----------
    path : str or Path
        Path to ``sorted_cells_to_check_all_annot.tsv``.
    rename_map : dict, optional
        Override for :data:`RENAME_NEW_TO_OLD`.
    apply_rename : bool
        Apply the rename map to ``Cell_type``. No-op when labels already match.

    Returns
    -------
    pd.DataFrame
        Sample-indexed annotation with at least ``Cell_type`` and ``Dataset``.

    Raises
    ------
    ValueError
        If required columns ``Cell_type`` or ``Dataset`` are missing.
    """
    annot = read_dataset(path)
    if "Sample" in annot.columns and annot.index.name != "Sample":
        annot = annot.set_index("Sample")
    if "Cell_type" not in annot.columns:
        raise ValueError(f"new-cohort annotation at {path} lacks 'Cell_type' column")
    if "Dataset" not in annot.columns:
        raise ValueError(
            f"new-cohort annotation at {path} lacks 'Dataset' column "
            "(required by signature_validation.utils.utils.read_expressions)"
        )
    # The raw TSV carries a few malformed rows with an empty ``Sample`` cell
    # (shifted columns, ``Technical_QC == False``). They collapse into duplicate
    # NaN index labels, which later breaks ``annot.Cell_type.reindex(...)`` in
    # get_strat_cell_type with "cannot reindex on an axis with duplicate labels".
    missing_id = annot.index.isna()
    if missing_id.any():
        logger.warning(
            "dropping {n} rows with an empty Sample id", n=int(missing_id.sum())
        )
        annot = annot[~missing_id]
    duplicated_id = annot.index.duplicated()
    if duplicated_id.any():
        logger.warning(
            "dropping {n} rows with duplicate Sample ids", n=int(duplicated_id.sum())
        )
        annot = annot[~duplicated_id]

    annot = annot.copy()
    annot["Cell_type"] = annot["Cell_type"].str.strip()
    if apply_rename:
        annot["Cell_type"] = annot["Cell_type"].replace(rename_map or RENAME_NEW_TO_OLD)
    logger.info(
        "loaded {n} samples across {k} cell types from {p}",
        n=len(annot),
        k=annot["Cell_type"].nunique(),
        p=path,
    )
    return annot


def load_new_cohort_expressions(
    annotation: pd.DataFrame,
    path: Union[str, Path],
    log2: bool = True,
) -> pd.DataFrame:
    """Load expressions matching ``annotation`` and apply the pipeline log2(TPM+1).

    Parameters
    ----------
    annotation : pd.DataFrame
        Sample-indexed annotation produced by
        :func:`load_new_cohort_annotation`.
    path : str or Path
        Root path or S3 prefix passed to
        :func:`signature_validation.utils.utils.read_expressions`.
    log2 : bool
        Apply ``log2(TPM + 1)``.

    Returns
    -------
    pd.DataFrame
        Gene × sample expression matrix.
    """
    expr = read_expressions(annotation, path=path)
    if log2:
        expr = np.log2(expr + 1)
    logger.info(
        "loaded expressions: {g} genes × {s} samples (log2={l})",
        g=expr.shape[0],
        s=expr.shape[1],
        l=log2,
    )
    return expr


def intersect_controls_with_cohort(
    controls_order: List[str],
    annotation: pd.DataFrame,
    min_n: int = 1,
) -> List[str]:
    """Drop cell types absent from ``annotation`` (or below ``min_n`` samples).

    Parameters
    ----------
    controls_order : list of str
        Authoritative axis order from :data:`CONTROLS_ORDER`.
    annotation : pd.DataFrame
        Indexed by sample, must have a ``Cell_type`` column.
    min_n : int
        Minimum sample count to keep a cell type.

    Returns
    -------
    list of str
        Subset of ``controls_order`` whose cell types meet the count threshold.
    """
    counts = annotation["Cell_type"].value_counts()
    return [ct for ct in controls_order if counts.get(ct, 0) >= min_n]


def build_mapping(
    map_raw: Optional[Dict[str, List[str]]] = None,
    controls_order: Optional[List[str]] = None,
    controls_to_delete: Optional[Dict[str, List[str]]] = None,
    annotation: Optional[pd.DataFrame] = None,
) -> Dict[str, Dict[str, List[str]]]:
    """Build the v1-style mapping dict scoped to in-scope FGES and the new cohort.

    Parameters
    ----------
    map_raw : dict, optional
        Override for :data:`MAP_RAW`.
    controls_order : list, optional
        Override for :data:`CONTROLS_ORDER`.
    controls_to_delete : dict, optional
        Override for :data:`CONTROLS_TO_DELETE`.
    annotation : pd.DataFrame, optional
        New-cohort annotation; when given, ``controls_order`` is filtered to the
        cell types present in ``annotation`` via
        :func:`intersect_controls_with_cohort`.

    Returns
    -------
    dict
        ``{Main4_*: {'Goi': [...], 'Control': [...], 'Deleted_controls': [...]}}``
    """
    map_raw = map_raw or MAP_RAW
    controls_order = controls_order or CONTROLS_ORDER
    controls_to_delete = controls_to_delete or CONTROLS_TO_DELETE

    if annotation is not None:
        controls_order = intersect_controls_with_cohort(controls_order, annotation)

    gois_flat = sorted({g for gois in map_raw.values() for g in gois})
    controls_universe = sorted(set(controls_order) | set(gois_flat))

    mapping: Dict[str, Dict[str, List[str]]] = {}
    for sign, gois in map_raw.items():
        controls = [ct for ct in controls_universe if ct not in gois]
        deletes = controls_to_delete.get(sign, [])
        kept = [ct for ct in controls if ct not in deletes]
        deleted = [ct for ct in deletes if ct in controls]
        mapping[sign] = {
            "Goi": list(gois),
            "Control": kept,
            "Deleted_controls": deleted,
        }
    return mapping


# Published v1 cohort — the training set for everything Figure 4 reports. Any
# sample of the new cohort that also appears here is not held out.
OLD_TRAIN_SSGSEAS_PATH: Path = Path("<PATH_TO_TRAIN_MAPPING_SSGSEAS_PKL>")


def collect_sample_ids(mapping_ssgseas: Dict) -> Set[str]:
    """Every sample ID appearing anywhere in a ``mapping_ssgseas`` structure.

    The structure nests ``{FGES: {group: {cell_type: DataFrame}}}`` with sample IDs
    on the frame index, and the same sample recurs across FGES and groups, so the
    union over all frames is the cohort's true sample universe — i.e. exactly the
    samples that got an ssGSEA score and therefore reached the figures.

    Parameters
    ----------
    mapping_ssgseas : dict
        Either cohort's scores: the output of
        :func:`signature_validation.benchmark.scoring.compute_mapping_ssgseas` or
        the published pickle loaded from disk.

    Returns
    -------
    set of str
    """
    samples: Set[str] = set()
    for groups in mapping_ssgseas.values():
        if not isinstance(groups, dict):
            continue
        for cell_types in groups.values():
            if not isinstance(cell_types, dict):
                continue
            for frame in cell_types.values():
                samples.update(map(str, frame.index))
    return samples


def load_train_sample_ids(path: Union[str, Path] = OLD_TRAIN_SSGSEAS_PATH) -> Set[str]:
    """Every sample ID appearing anywhere in the published (train) ssGSEA pickle."""
    with open(path, "rb") as handle:
        train_ssgseas = pickle.load(handle)

    samples = collect_sample_ids(train_ssgseas)
    logger.info("train cohort: {n} unique sample IDs from {p}", n=len(samples), p=path)
    return samples


def load_train_annotation(
    train_samples: Set[str],
    path: Union[str, Path] = TRAIN_ANNOT_PATH,
    fallback_annotation: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Resolve ``Dataset`` / ``Cell_type`` for the published (train) cohort's samples.

    The v1 cohort was assembled from ``cells_all_annotation.tsv`` plus a handful of
    samples added by hand from Google Sheets (tonsillar Tfh, mast cells), so that
    table alone does not cover every scored train sample. Those stragglers do appear
    in the new-cohort annotation, which is why ``fallback_annotation`` exists: pass
    the validation annotation and the two sources together cover the cohort.

    Parameters
    ----------
    train_samples : set of str
        Sample IDs to resolve, e.g. from :func:`load_train_sample_ids`.
    path : str or Path
        Sample-level annotation of the sorted-cell database (:data:`TRAIN_ANNOT_PATH`).
    fallback_annotation : pd.DataFrame, optional
        Sample-indexed annotation consulted for IDs missing from ``path``.

    Returns
    -------
    pd.DataFrame
        Indexed by sample ID, columns ``Dataset`` and ``Cell_type``. Rows resolved
        by neither source carry ``Dataset = "Unknown"`` rather than being dropped,
        so the sample counts of downstream tables stay complete.

    Raises
    ------
    ValueError
        If the primary annotation lacks ``Dataset`` or ``Cell_type``.
    """
    annot = read_dataset(path)
    if "Sample" in annot.columns and annot.index.name != "Sample":
        annot = annot.set_index("Sample")
    missing_cols = [c for c in ("Dataset", "Cell_type") if c not in annot.columns]
    if missing_cols:
        raise ValueError(
            f"train annotation at {path} lacks column(s): {', '.join(missing_cols)}"
        )
    annot = annot[~annot.index.isna()]
    annot = annot[~annot.index.duplicated()]
    annot.index = annot.index.astype(str)

    wanted = {str(s) for s in train_samples}
    resolved = annot.loc[sorted(wanted & set(annot.index)), ["Dataset", "Cell_type"]]

    still_missing = sorted(wanted - set(resolved.index))
    if still_missing and fallback_annotation is not None:
        fb = fallback_annotation.copy()
        fb.index = fb.index.astype(str)
        fb = fb[~fb.index.duplicated()]
        from_fb = sorted(set(still_missing) & set(fb.index))
        if from_fb:
            resolved = pd.concat(
                [resolved, fb.loc[from_fb, ["Dataset", "Cell_type"]]]
            )
            logger.info(
                "train annotation: {n} sample(s) resolved from the fallback annotation",
                n=len(from_fb),
            )
        still_missing = sorted(set(still_missing) - set(from_fb))

    if still_missing:
        logger.warning(
            "train annotation: {n} sample(s) unresolved, Dataset='Unknown': {ids}",
            n=len(still_missing),
            ids=still_missing[:20],
        )
        resolved = pd.concat(
            [
                resolved,
                pd.DataFrame(
                    {"Dataset": "Unknown", "Cell_type": np.nan},
                    index=pd.Index(still_missing),
                ),
            ]
        )

    logger.info(
        "train annotation: {n}/{t} sample(s) resolved to a dataset",
        n=int((resolved["Dataset"] != "Unknown").sum()),
        t=len(wanted),
    )
    return resolved


def drop_train_samples(
    annotation: pd.DataFrame,
    train_ids: Set[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Remove samples that already belong to the published train cohort.

    Without this the "held-out test cohort" partly re-measures the data the
    signatures were selected on. The overlap is small and concentrated, so the
    report matters more than the count: it names which cell types shrink, and by
    how much, which is what decides whether a GOI cohort drops under the
    crossval-backfill threshold.

    Returns
    -------
    (annotation, report)
        The annotation without the overlapping samples, and a
        ``cell_type | n_before | n_dropped | n_after`` frame covering only the
        cell types that actually lost samples.
    """
    overlap = annotation.index.astype(str).isin(train_ids)
    n_dropped = int(overlap.sum())
    if not n_dropped:
        logger.info("no train samples found in the annotation — nothing dropped")
        return annotation, pd.DataFrame(
            columns=["cell_type", "n_before", "n_dropped", "n_after"]
        )

    before = annotation["Cell_type"].value_counts()
    dropped = annotation.loc[overlap, "Cell_type"].value_counts()
    kept = annotation.loc[~overlap]
    after = kept["Cell_type"].value_counts()

    report = pd.DataFrame(
        {
            "cell_type": dropped.index,
            "n_before": [int(before.get(ct, 0)) for ct in dropped.index],
            "n_dropped": [int(dropped[ct]) for ct in dropped.index],
            "n_after": [int(after.get(ct, 0)) for ct in dropped.index],
        }
    ).sort_values("n_dropped", ascending=False, ignore_index=True)

    logger.info(
        "dropped {n} train samples ({p:.2f}% of the annotation) across {c} cell types",
        n=n_dropped,
        p=100.0 * n_dropped / max(len(annotation), 1),
        c=len(report),
    )
    for row in report.itertuples(index=False):
        logger.info(
            "  {ct}: {b} → {a} (−{d})",
            ct=row.cell_type,
            b=row.n_before,
            a=row.n_after,
            d=row.n_dropped,
        )
    return kept, report
