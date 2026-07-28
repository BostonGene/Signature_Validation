"""Cohort assembly for the new sorted-cell test cohort (Jira OD-128, CHESS-1333).

Functions here load the new annotation / expressions delivered with OD-128 and
build the v1-style ``mapping`` dict (GOI / Control / Deleted_controls per FGES),
restricted to the 16 in-scope FGES and to the cell types present in the new
cohort.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set, Union

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

# FGES not covered by this rerun. Th17 / Endothelium_lymph / Eosinophils have
# no new samples; Plasma_cells has only 3 samples for both Plasma_B_cells and
# Plasmablasts. They are deferred to a separate rare-types notebook that uses
# 75/25 holdouts on the original cohort.
EXCLUDED_FGES_RARE: Set[str] = {
    "Main4_Th17_signature",
    "Main4_Lymphatic_endothelium",
    "Main4_Eosinophil_signature",
    "Main4_Plasma_cells",
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


def load_new_cohort_annotation(
    path: Union[str, Path],
    rename_map: Optional[Dict[str, str]] = None,
    apply_rename: bool = True,
) -> pd.DataFrame:
    """Load the new sorted-cell annotation TSV.

    The file is already filtered by ``Technical_QC == True`` and
    ``Decision_deconvolution_without_parent != False`` and the rename to
    pipeline names is already applied (per OD-128 ticket); the rename pass here
    is idempotent and defensive.

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
    if apply_rename:
        annot = annot.copy()
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
