"""ssGSEA scoring orchestration and the per-signature × cell-type stats table.

The original v1 notebook's cells 45 (mapping_ssgseas), 52-54 (out table) and 56
(FDR correction) are refactored here so the new-cohort notebook is a thin caller.
The FDR pass is deterministic (uses an explicit ``controls_order`` argument
instead of the ``set()``-based ordering of v1).
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from signature_validation.ssgsea_calc.ssgsea_calc import GeneSet, ssgsea_formula
from signature_validation.utils.utils import cohen_d


def compute_mapping_ssgseas(
    public_cells_expr: pd.DataFrame,
    public_cells_annot: pd.DataFrame,
    mapping: Dict[str, Dict[str, List[str]]],
    msigdb_gmt: Dict[str, Dict[str, GeneSet]],
) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
    """Score every (signature × cohort group × cell type) combination via ssGSEA.

    Equivalent to v1 cell 45.

    Parameters
    ----------
    public_cells_expr : pd.DataFrame
        Gene × sample expression matrix (log2 TPM).
    public_cells_annot : pd.DataFrame
        Sample-indexed annotation with a ``Cell_type`` column.
    mapping : dict
        Output of :func:`signature_validation.benchmark.cohorts.build_mapping`.
    msigdb_gmt : dict
        Output of :func:`signature_validation.benchmark.signatures.harmonize_gmt_to_index`.

    Returns
    -------
    dict
        ``{Main4_*: {Goi|Control|Deleted_controls: {cell_type: DataFrame[sample × signature]}}}``.
    """
    out: Dict[str, Dict[str, Dict[str, pd.DataFrame]]] = {}
    for sign, bucket in tqdm(mapping.items(), desc="ssGSEA per FGES"):
        out[sign] = {"Goi": {}, "Control": {}, "Deleted_controls": {}}
        for group in ("Goi", "Control", "Deleted_controls"):
            for ct in bucket[group]:
                samples_of_ct = public_cells_annot.index[
                    public_cells_annot["Cell_type"] == ct
                ]
                if len(samples_of_ct) == 0:
                    continue
                samples_in_expr = public_cells_expr.columns.intersection(samples_of_ct)
                if len(samples_in_expr) == 0:
                    logger.warning(
                        "{sign}/{group}/{ct}: no samples with expressions; skipping",
                        sign=sign,
                        group=group,
                        ct=ct,
                    )
                    continue
                part = public_cells_expr[samples_in_expr]
                scores = ssgsea_formula(part, msigdb_gmt[sign]).T
                out[sign][group][ct] = scores
    return out


def clean_parent_daughter_goi(
    mapping_ssgseas: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
    parent_to_daughter: Dict[str, List[str]],
) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
    """Strip daughter-FGES sub-signature columns out of each parent's GOI frames.

    Port of v1 ``Scater_plots.ipynb`` cell 16. A sub-signature that qualifies for
    both a parent FGES (e.g. ``Main4_Pan_macrophage_signature``) and one of its
    daughters (e.g. ``Main4_M2_signature``) appears under the same column name in
    both; leaving it in the parent's GOI double-counts it in the metric loop and,
    in :func:`compute_out_table`, lets the last-processed FGES overwrite the
    shared row. Removing it from the parent resolves both. Daughters absent from
    ``mapping_ssgseas`` (rare/out-of-scope FGES) are skipped.

    The input is not mutated; a new nested dict is returned (immutability rule).

    Parameters
    ----------
    mapping_ssgseas : dict
        Output of :func:`compute_mapping_ssgseas`.
    parent_to_daughter : dict
        ``{parent_FGES: [daughter_FGES, ...]}``, e.g.
        :data:`signature_validation.benchmark.cohorts.PARENT_TO_DAUGHTER`.

    Returns
    -------
    dict
        A copy of ``mapping_ssgseas`` with parent GOI frames pruned of daughter
        sub-signature columns.
    """
    cleaned: Dict[str, Dict[str, Dict[str, pd.DataFrame]]] = {
        sign: {group: dict(frames) for group, frames in groups.items()}
        for sign, groups in mapping_ssgseas.items()
    }
    for parent, daughters in parent_to_daughter.items():
        if parent not in cleaned:
            continue
        daughter_columns: set[str] = set()
        for daughter in daughters:
            if daughter not in mapping_ssgseas:
                continue
            for frame in mapping_ssgseas[daughter]["Goi"].values():
                daughter_columns.update(frame.columns)
        if not daughter_columns:
            continue
        for ct, frame in cleaned[parent]["Goi"].items():
            keep = [c for c in frame.columns if c not in daughter_columns]
            cleaned[parent]["Goi"][ct] = frame[keep]
        logger.info(
            "cleaned {parent}: removed {n} daughter columns from GOI",
            parent=parent,
            n=len(daughter_columns),
        )
    return cleaned


def compute_out_table(
    mapping_ssgseas: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
    mapping: Dict[str, Dict[str, List[str]]],
    msigdb_gmt: Dict[str, Dict[str, GeneSet]],
    controls_order: List[str],
) -> pd.DataFrame:
    """Build the per-(signature × cell type) stats DataFrame.

    Equivalent to v1 cells 52 + 54, with deterministic column ordering driven
    by ``controls_order`` (the v1 ``set()``-based order is non-reproducible
    across runs).

    Parameters
    ----------
    mapping_ssgseas : dict
        Output of :func:`compute_mapping_ssgseas`.
    mapping : dict
        Output of :func:`signature_validation.benchmark.cohorts.build_mapping`.
    msigdb_gmt : dict
        Output of :func:`signature_validation.benchmark.signatures.harmonize_gmt_to_index`.
    controls_order : list of str
        Authoritative control-cell-type order; defines the column order.

    Returns
    -------
    pd.DataFrame
        Indexed by sub-signature name; columns
        ``{Median, Mean, Std, Cohen's D from ours, MW from ours, FDR} × {GOI, *controls}``.
    """
    metric_suffixes = (
        "Median",
        "Mean",
        "Std",
        "Cohen's D from ours",
        "MW from ours",
        "FDR",
    )
    cols: List[str] = []
    for prefix in ("GOI",) + tuple(controls_order):
        cols.extend(f"{m} in {prefix}" for m in metric_suffixes)
    all_signs: List[str] = []
    seen: set[str] = set()
    for sign in mapping:
        for sub in msigdb_gmt[sign]:
            if sub not in seen:
                all_signs.append(sub)
                seen.add(sub)
    out_df = pd.DataFrame(index=all_signs, columns=cols, dtype=float)

    for sign in mapping_ssgseas:
        goi_dfs = list(mapping_ssgseas[sign]["Goi"].values())
        if not goi_dfs:
            continue
        goi_df = pd.concat(goi_dfs)
        for signat in goi_df.columns:
            x = goi_df[sign]
            y = goi_df[signat]
            out_df.loc[signat, "Median in GOI"] = float(y.median())
            out_df.loc[signat, "Mean in GOI"] = float(y.mean())
            out_df.loc[signat, "Std in GOI"] = float(np.std(y))
            out_df.loc[signat, "Cohen's D from ours in GOI"] = float(cohen_d(x, y))
            mw_p = _safe_mannwhitneyu(x, y)
            out_df.loc[signat, "MW from ours in GOI"] = mw_p
            for ct, ct_df in mapping_ssgseas[sign]["Control"].items():
                if signat not in ct_df.columns:
                    continue
                xc = ct_df[sign]
                yc = ct_df[signat]
                out_df.loc[signat, f"Median in {ct}"] = float(yc.median())
                out_df.loc[signat, f"Mean in {ct}"] = float(yc.mean())
                out_df.loc[signat, f"Std in {ct}"] = float(np.std(yc))
                out_df.loc[signat, f"Cohen's D from ours in {ct}"] = float(
                    cohen_d(xc, yc)
                )
                out_df.loc[signat, f"MW from ours in {ct}"] = _safe_mannwhitneyu(
                    xc, yc
                )
    return out_df


def fdr_correct_out(out_df: pd.DataFrame, controls_order: List[str]) -> pd.DataFrame:
    """BH-correct ``MW from ours in {ct}`` columns into ``FDR in {ct}``.

    Parameters
    ----------
    out_df : pd.DataFrame
        Output of :func:`compute_out_table`.
    controls_order : list of str
        Same order as passed to :func:`compute_out_table`. Determines the
        deterministic concatenation order for multiple-testing correction.

    Returns
    -------
    pd.DataFrame
        Copy of ``out_df`` with ``FDR in {ct}`` columns populated.
    """
    out_df = out_df.copy()
    flats: List[pd.Series] = []
    for ct in controls_order:
        col = f"MW from ours in {ct}"
        if col in out_df.columns:
            flats.append(out_df[col].dropna())
    if not flats:
        return out_df
    flat = pd.concat(flats)
    _, corrected, _, _ = multipletests(flat.values, method="fdr_bh")
    flat_corr = pd.Series(corrected, index=flat.index)
    cursor = 0
    for ct in controls_order:
        col_p = f"MW from ours in {ct}"
        col_fdr = f"FDR in {ct}"
        if col_p not in out_df.columns:
            continue
        present_idx = out_df[col_p].dropna().index
        n = len(present_idx)
        if n == 0:
            continue
        out_df.loc[present_idx, col_fdr] = flat_corr.iloc[cursor : cursor + n].values
        cursor += n
    return out_df


def _safe_mannwhitneyu(x: pd.Series, y: pd.Series) -> float:
    """Run Mann–Whitney U returning NaN on degenerate inputs (constant arrays)."""
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    try:
        _, p = mannwhitneyu(x, y)
    except ValueError:
        return float("nan")
    return float(p)
