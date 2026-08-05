"""Supplementary summary tables for the new-cohort cell-type FGES benchmark.

Two deliverables (CHESS-1333 / OD-128):

1. :func:`build_fges_performance_tables` — one per-FGES ranking table (paper
   Supplement S4) that couples the classification metrics (mean over seeds of
   ``fges_metrics``) with the pooled GOI / Control ssGSEA means and BH-corrected
   Mann–Whitney p-values, one row per sub-signature.
2. :func:`build_dataset_list_table` — a compact dataset inventory (dataset,
   sample count, cell types) for the methods section.

Both write tab-separated files and return the assembled DataFrame(s).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

from signature_validation.benchmark.cohorts import EXCLUDED_FGES_RARE, MAP_RAW
from signature_validation.ssgsea_calc.ssgsea_calc import detect_fges_source

# fges_metrics inner-dict key → output column. Kept explicit so a rename in the
# scoring layer surfaces here rather than silently producing NaN columns.
_METRIC_KEY_MAP: Dict[str, str] = {
    "F_score": "F1",
    "Accuracy": "Accuracy",
    "PR_AUC": "PR_AUC",
    "ROC_AUC": "ROC_AUC",
    "GOI_CV": "goi_cv",
    "Control_CV": "control_cv",
}

_PERFORMANCE_COLUMNS: List[str] = [
    "Top",
    "Cell_type",
    "FGES_Name",
    "Source",
    "F_score",
    "Accuracy",
    "PR_AUC",
    "ROC_AUC",
    "GOI_CV",
    "Mean_ssGSEA_GOI",
    "adjusted_p_wilcoxon_GOI",
    "Mean_ssGSEA_Control",
    "adjusted_p_wilcoxon_Control",
    "Control_CV",
]


def _safe_mannwhitneyu(x: pd.Series, y: pd.Series) -> float:
    """Run Mann–Whitney U, returning NaN on degenerate inputs (constant/empty)."""
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    try:
        _, p = mannwhitneyu(x, y)
    except ValueError:
        # Raised e.g. when all values in the pooled sample are identical.
        return float("nan")
    return float(p)


def _classify_source(signat: str) -> str:
    """Classify a sub-signature name via ``detect_fges_source``, NaN-safe on I/O.

    ``detect_fges_source`` reads a MSigDb .gmt from a relative path; off-cluster
    that file may be absent. Fall back to ``"Other"`` rather than crashing the
    whole table build.
    """
    try:
        return detect_fges_source(signat)
    except (FileNotFoundError, OSError):
        logger.warning(
            "detect_fges_source could not read its .gmt for {s}; source='Other'",
            s=signat,
        )
        return "Other"


def _mean_over_seeds(seed_map: Dict[int, Dict[str, float]], metric_key: str) -> float:
    """Mean of ``metric_key`` across seeds; NaN when absent everywhere."""
    if not seed_map:
        return float("nan")
    vals = [
        seed_map[seed][metric_key]
        for seed in seed_map
        if metric_key in seed_map[seed]
    ]
    if not vals:
        return float("nan")
    return float(np.nanmean(vals))


def _bh_adjust(pvals: pd.Series) -> pd.Series:
    """BH-FDR-correct a Series of p-values, preserving NaN positions."""
    adjusted = pd.Series(np.nan, index=pvals.index, dtype=float)
    valid = pvals.dropna()
    if valid.empty:
        return adjusted
    _, corrected, _, _ = multipletests(valid.values, method="fdr_bh")
    adjusted.loc[valid.index] = corrected
    return adjusted


def _pool_group(
    group_frames: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Concatenate per-cell-type ssGSEA frames and drop duplicate sample rows."""
    frames = list(group_frames.values())
    if not frames:
        return pd.DataFrame()
    pooled = pd.concat(frames)
    # Duplicate sample indices can appear when a sample maps to several cell
    # types in the pool; keep the first occurrence for a deterministic mean.
    return pooled[~pooled.index.duplicated(keep="first")]


def build_fges_performance_tables(
    mapping_ssgseas: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
    fges_metrics: Dict[str, Dict[int, Dict[str, float]]],
    mapping: Dict[str, Dict[str, List[str]]],
    msigdb_gmt: Dict[str, Dict[str, Any]],
    save_dir: Union[str, Path],
    prefix: str = "S4",
    exclude: Iterable[str] = EXCLUDED_FGES_RARE,
) -> Dict[str, pd.DataFrame]:
    """Build one per-FGES sub-signature ranking table (paper Supplement S4).

    Each in-scope FGES yields a table whose rows are its candidate sub-signatures
    (from ``msigdb_gmt[fges_key]``) that were actually scored. Classification
    metrics are averaged over seeds; the pooled GOI / Control ssGSEA means and
    the BH-corrected Mann–Whitney p-values (sub-signature vs. the internal "ours"
    column ``fges_key``) come from ``mapping_ssgseas``. Rows are ranked by
    ``F_score`` descending (NaN last) and numbered in ``Top``.

    Parameters
    ----------
    mapping_ssgseas : dict
        Nested ``{fges_key: {Goi|Control|Deleted_controls: {cell_type: DataFrame}}}``
        of ssGSEA scores (sample × sub-signature), from
        :func:`signature_validation.benchmark.scoring.compute_mapping_ssgseas`.
    fges_metrics : dict
        ``{sub_signature: {seed: {"F1", "Accuracy", "PR_AUC", "ROC_AUC",
        "goi_cv", "control_cv"}}}`` classification metrics per seed.
    mapping : dict
        ``{fges_key: {"Goi": [...], "Control": [...], "Deleted_controls": [...]}}``
        from :func:`signature_validation.benchmark.cohorts.build_mapping`; used to
        resolve the FGES cell type.
    msigdb_gmt : dict
        ``{fges_key: {sub_signature: gene_set}}`` — its keys enumerate the
        candidate sub-signatures per FGES.
    save_dir : str or Path
        Directory for the per-FGES ``{prefix}.{k}_{cell_type}.tsv`` files.
    prefix : str
        Filename prefix (paper table label). Default ``"S4"``.
    exclude : iterable of str
        FGES keys to skip (default :data:`EXCLUDED_FGES_RARE`).

    Returns
    -------
    dict
        ``{fges_key: DataFrame}`` for every emitted table.

    Raises
    ------
    ValueError
        If ``mapping`` and ``MAP_RAW`` both lack a cell type for an in-scope FGES.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    exclude_set = set(exclude)

    tables: Dict[str, pd.DataFrame] = {}
    emitted = 0
    for fges_key in mapping_ssgseas:
        if fges_key in exclude_set:
            continue
        goi_pool = _pool_group(mapping_ssgseas[fges_key].get("Goi", {}))
        control_pool = _pool_group(mapping_ssgseas[fges_key].get("Control", {}))
        if goi_pool.empty or control_pool.empty:
            logger.info("{f}: empty GOI or Control pool; skipping", f=fges_key)
            continue

        cell_type = _resolve_cell_type(fges_key, mapping)

        scored_cols = set(goi_pool.columns) | set(control_pool.columns)
        sub_signatures = [
            signat for signat in msigdb_gmt.get(fges_key, {}) if signat in scored_cols
        ]
        if not sub_signatures:
            logger.info("{f}: no scored sub-signatures; skipping", f=fges_key)
            continue

        rows = [
            _build_row(
                signat=signat,
                fges_key=fges_key,
                cell_type=cell_type,
                goi_pool=goi_pool,
                control_pool=control_pool,
                fges_metrics=fges_metrics,
            )
            for signat in sub_signatures
        ]
        table = pd.DataFrame(rows)

        # BH-correct the raw MWU p-values across this FGES's sub-signatures.
        table["adjusted_p_wilcoxon_GOI"] = _bh_adjust(
            table["adjusted_p_wilcoxon_GOI"]
        ).values
        table["adjusted_p_wilcoxon_Control"] = _bh_adjust(
            table["adjusted_p_wilcoxon_Control"]
        ).values

        table = table.sort_values(
            "F_score", ascending=False, na_position="last"
        ).reset_index(drop=True)
        table["Top"] = np.arange(1, len(table) + 1)
        table = table[_PERFORMANCE_COLUMNS]

        emitted += 1
        out_path = save_dir / f"{prefix}.{emitted}_{cell_type}.tsv"
        table.to_csv(out_path, sep="\t", index=False)
        logger.info(
            "wrote {n} sub-signatures for {f} → {p}",
            n=len(table),
            f=fges_key,
            p=out_path,
        )
        tables[fges_key] = table

    return tables


def _resolve_cell_type(
    fges_key: str, mapping: Dict[str, Dict[str, List[str]]]
) -> str:
    """Return the FGES GOI cell type from ``mapping`` or fall back to ``MAP_RAW``."""
    goi = mapping.get(fges_key, {}).get("Goi", [])
    if goi:
        return goi[0]
    raw = MAP_RAW.get(fges_key, [])
    if raw:
        return raw[0]
    raise ValueError(f"no GOI cell type found for FGES '{fges_key}'")


def _build_row(
    signat: str,
    fges_key: str,
    cell_type: str,
    goi_pool: pd.DataFrame,
    control_pool: pd.DataFrame,
    fges_metrics: Dict[str, Dict[int, Dict[str, float]]],
) -> Dict[str, Any]:
    """Assemble one sub-signature row (raw MWU p-values; BH applied by caller)."""
    seed_map = fges_metrics.get(signat, {})
    row: Dict[str, Any] = {
        "Cell_type": cell_type,
        "FGES_Name": signat,
        "Source": _classify_source(signat),
        "F_score": _mean_over_seeds(seed_map, _METRIC_KEY_MAP["F_score"]),
        "Accuracy": _mean_over_seeds(seed_map, _METRIC_KEY_MAP["Accuracy"]),
        "PR_AUC": _mean_over_seeds(seed_map, _METRIC_KEY_MAP["PR_AUC"]),
        "ROC_AUC": _mean_over_seeds(seed_map, _METRIC_KEY_MAP["ROC_AUC"]),
        "GOI_CV": _mean_over_seeds(seed_map, _METRIC_KEY_MAP["GOI_CV"]),
        "Control_CV": _mean_over_seeds(seed_map, _METRIC_KEY_MAP["Control_CV"]),
        "Mean_ssGSEA_GOI": (
            float(goi_pool[signat].mean())
            if signat in goi_pool.columns
            else float("nan")
        ),
        "Mean_ssGSEA_Control": (
            float(control_pool[signat].mean())
            if signat in control_pool.columns
            else float("nan")
        ),
    }
    # Raw MWU vs. the internal "ours" column (fges_key); NaN when either the
    # sub-signature or the "ours" reference is missing from the pool.
    if signat in goi_pool.columns and fges_key in goi_pool.columns:
        row["adjusted_p_wilcoxon_GOI"] = _safe_mannwhitneyu(
            goi_pool[signat], goi_pool[fges_key]
        )
    else:
        row["adjusted_p_wilcoxon_GOI"] = float("nan")
    if signat in control_pool.columns and fges_key in control_pool.columns:
        row["adjusted_p_wilcoxon_Control"] = _safe_mannwhitneyu(
            control_pool[signat], control_pool[fges_key]
        )
    else:
        row["adjusted_p_wilcoxon_Control"] = float("nan")
    return row


def build_dataset_list_table(
    annotation: pd.DataFrame,
    save_path: Union[str, Path],
) -> pd.DataFrame:
    """Summarise the cohort as a per-dataset inventory table.

    Parameters
    ----------
    annotation : pd.DataFrame
        Sample-indexed annotation with at least ``Dataset`` and ``Cell_type``.
    save_path : str or Path
        Destination TSV path.

    Returns
    -------
    pd.DataFrame
        Columns ``Dataset, N_samples, Cell_types`` sorted by ``N_samples``
        descending then ``Dataset`` ascending.

    Raises
    ------
    ValueError
        If ``Dataset`` or ``Cell_type`` columns are missing.
    """
    missing = [c for c in ("Dataset", "Cell_type") if c not in annotation.columns]
    if missing:
        raise ValueError(
            f"annotation is missing required column(s): {', '.join(missing)}"
        )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for dataset, group in annotation.groupby("Dataset"):
        cell_types = sorted(group["Cell_type"].dropna().unique())
        rows.append(
            {
                "Dataset": dataset,
                "N_samples": len(group),
                "Cell_types": ", ".join(cell_types),
            }
        )

    table = pd.DataFrame(rows, columns=["Dataset", "N_samples", "Cell_types"])
    table = table.sort_values(
        ["N_samples", "Dataset"], ascending=[False, True]
    ).reset_index(drop=True)
    table.to_csv(save_path, sep="\t", index=False)
    logger.info(
        "wrote dataset inventory: {n} datasets → {p}", n=len(table), p=save_path
    )
    return table
