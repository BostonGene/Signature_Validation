"""FGES quality metrics + F1-vs-CV scatter figures for the new cohort (OD-128).

Faithful port of the reference ``Scater_plots.ipynb`` cells:

* cell 20 → :func:`compute_fges_metrics` — per-FGES F-score / rank-deviation
  metrics under repeated stratified subsampling of the control set.
* cells 23 + 25 → :func:`plot_f1_cv_scatters` — per-BG-FGES and aggregated
  scatter plots of weighted F1 against the GOI rank coefficient of variation.

The notebook relied on module-level globals (``mapping_ssgseas``, ``msigdb_gmt``,
``public_cells_annot``, ``ranked_expr``, ``pipeline_genes``). Here they are
injected as keyword arguments so the code runs on the new-cohort objects and off
the BostonGene cluster.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import signature_validation.plotting.plotting as pl_mod
from signature_validation.benchmark.cohorts import EXCLUDED_FGES_RARE
from signature_validation.utils.fges_utils import (
    derive_rank_deviation,
    get_metric_for_signature,
    get_strat_cell_type,
)

# FGES-source → colour, from Scater_plots cell 7. The plotting module ships a
# LIST under the same name (used only to size a distinctipy palette), which the
# scatter functions wrongly index like a dict; :func:`plot_f1_cv_scatters`
# monkeypatches this dict in before calling them. See the module docstring.
DEFAULT_SIGNATURE_PALETTE: Dict[str, str] = {
    "Internal": "#ff0000",
    "Random_FGES": "black",
    "xCell": "#12ff1b",
    "Bindea": "#ff8000",
    "Nirmal": "#804080",
    "Gene_Ontology": "#40fbc9",
    "KEGG": "gold",
    "BioCarta": "#fe7cc3",
    "WikiPathways": "#0080ff",
    "Reactome": "#258103",
    "Pathway_Interaction_Database": "#83e9ff",
    "Human_Phenotype_Ontology": "#0500f5",
    "MSigDb_Dif_Expression": "#93ae8b",
    "MSigDb_Single_Cell": "#df7ffe",
    "MSigDb_Other": "#008080",
}

# Minimum sizing constants from the reference subsampling logic (cell 20).
_GOI_FLOOR: int = 50
_CONTROL_FLOOR: int = 100


def compute_fges_metrics(
    mapping_ssgseas: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
    msigdb_gmt: Dict[str, Dict[str, Any]],
    public_cells_annot: pd.DataFrame,
    ranked_expr: pd.DataFrame,
    pipeline_genes: List[str],
    n_iter: int = 10,
    exclude: Iterable[str] = EXCLUDED_FGES_RARE,
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Compute F-score and rank-deviation metrics per FGES over repeated seeds.

    Port of ``Scater_plots.ipynb`` cell 20. For every in-scope Main4 signature
    the GOI and Control ssGSEA blocks are concatenated, then for each candidate
    FGES column and each seed the control set is stratified-subsampled (via
    :func:`get_strat_cell_type` + :func:`sklearn.model_selection.train_test_split`)
    and the GOI is bootstrap-resampled to a size derived from the block sizes.
    Classification metrics come from :func:`get_metric_for_signature` and rank
    coefficients of variation from :func:`derive_rank_deviation`.

    Parameters
    ----------
    mapping_ssgseas : dict
        ``{Main4_signature: {"Goi"|"Control"|"Deleted_controls":
        {cell_type: ssGSEA-DataFrame}}}``. Only ``"Goi"`` and ``"Control"`` are
        read here.
    msigdb_gmt : dict
        ``{Main4_signature: {sub_signature: GeneSet}}`` mapping, forwarded to
        :func:`derive_rank_deviation`.
    public_cells_annot : pd.DataFrame
        Sample-indexed annotation with a ``Cell_type`` column, forwarded to
        :func:`get_strat_cell_type` for stratification.
    ranked_expr : pd.DataFrame
        Gene × sample matrix of percentile ranks, forwarded to
        :func:`derive_rank_deviation`.
    pipeline_genes : list of str
        Genes available in the expression pipeline, forwarded to
        :func:`derive_rank_deviation`.
    n_iter : int
        Number of subsampling seeds per FGES (reference default 10).
    exclude : iterable of str
        Main4 signature names to skip (rare types deferred to a separate
        notebook). Defaults to
        :data:`signature_validation.benchmark.cohorts.EXCLUDED_FGES_RARE`.

    Returns
    -------
    dict
        ``{fges_name: {seed: {metric_name: value}}}`` combining the
        classification metrics with ``goi_cv`` / ``control_cv`` /
        ``goi_std`` / ``control_std``.
    """
    exclude_set = set(exclude)
    fges_metrics: Dict[str, Dict[int, Dict[str, float]]] = {}

    # Reverse order mirrors the reference cell (`[::-1]`) for byte-identical seeds.
    for sign in list(mapping_ssgseas.keys())[::-1]:
        if sign in exclude_set:
            continue

        goi_blocks = mapping_ssgseas[sign].get("Goi", {})
        control_blocks = mapping_ssgseas[sign].get("Control", {})
        if not goi_blocks or not control_blocks:
            logger.debug("skipping {s}: empty Goi or Control block", s=sign)
            continue

        control = pd.concat(list(control_blocks.values()))
        goi = pd.concat(list(goi_blocks.values()))
        control = control[~control.index.duplicated()]
        goi = goi[~goi.index.duplicated()]
        if control.empty or goi.empty:
            logger.debug("skipping {s}: empty concatenated Goi or Control", s=sign)
            continue

        for fges in tqdm(goi.columns, desc=f"{sign}", leave=False):
            fges_metrics[fges] = {}
            for seed in range(0, n_iter):
                new_control, cell_types = get_strat_cell_type(
                    control, seed, public_cells_annot=public_cells_annot
                )

                # GOI bootstrap target size.
                if len(goi) > len(new_control) and len(goi) > _GOI_FLOOR:
                    goi_size = len(new_control)
                elif len(goi) < _GOI_FLOOR:
                    goi_size = _GOI_FLOOR
                else:
                    goi_size = len(goi)

                # Control subsample target size.
                if len(goi) <= _CONTROL_FLOOR or len(new_control) <= _CONTROL_FLOOR:
                    control_size = _CONTROL_FLOOR
                elif len(goi) > len(new_control) and len(new_control) > _CONTROL_FLOOR:
                    control_size = len(new_control)
                else:
                    control_size = len(goi)

                sample_perc = control_size / (len(new_control) + len(goi))
                _, control_subsample = train_test_split(
                    new_control,
                    test_size=sample_perc,
                    stratify=cell_types,
                    random_state=seed,
                )
                goi_subsample = goi.sample(n=goi_size, replace=True, random_state=seed)

                labels = pd.concat(
                    [
                        pd.Series(0, index=control_subsample.index),
                        pd.Series(1, index=goi_subsample.index),
                    ]
                )
                df = pd.concat([control_subsample, goi_subsample])

                m_dict = get_metric_for_signature(
                    df[fges], labels, verbose=False, youden_thr=False
                )
                dev_dict = derive_rank_deviation(
                    control_subsample,
                    goi,
                    sign,
                    fges,
                    msigdb_gmt=msigdb_gmt,
                    ranked_expr=ranked_expr,
                    pipeline_genes=pipeline_genes,
                )
                m_dict.update(dev_dict)
                fges_metrics[fges][seed] = m_dict

    logger.info("computed metrics for {n} FGES", n=len(fges_metrics))
    return fges_metrics


def plot_f1_cv_scatters(
    fges_metrics: Dict[str, Dict[int, Dict[str, float]]],
    msigdb_gmt: Dict[str, Dict[str, Any]],
    save_dir: Union[str, Path],
    signature_palette: Optional[Dict[str, str]] = None,
    exclude: Iterable[str] = EXCLUDED_FGES_RARE,
) -> None:
    """Emit per-BG-FGES and aggregated F1-vs-CV scatter figures.

    Port of ``Scater_plots.ipynb`` cells 23 (one plot per BG signature, saved
    under ``save_dir/svg_pictures_F1_cv/``) and 25 (aggregated plot saved under
    ``save_dir``). Both underlying plot functions save ``path/(title + ".svg")``.

    Parameters
    ----------
    fges_metrics : dict
        Output of :func:`compute_fges_metrics`
        (``{fges_name: {seed: {metric: value}}}``).
    msigdb_gmt : dict
        ``{Main4_signature: {sub_signature: GeneSet}}``; its inner keys select
        which FGES enter each per-signature plot.
    save_dir : str or Path
        Root output directory. The per-signature SVGs land in the
        ``svg_pictures_F1_cv`` subdirectory; the aggregated SVG lands here.
    signature_palette : dict, optional
        FGES-source → colour override. Defaults to
        :data:`DEFAULT_SIGNATURE_PALETTE`.
    exclude : iterable of str
        Main4 signature names to skip. Defaults to
        :data:`signature_validation.benchmark.cohorts.EXCLUDED_FGES_RARE`.

    Returns
    -------
    None
    """
    exclude_set = set(exclude)
    save_dir = Path(save_dir)
    svg_dir = save_dir / "svg_pictures_F1_cv"
    save_dir.mkdir(parents=True, exist_ok=True)
    svg_dir.mkdir(parents=True, exist_ok=True)

    # CRITICAL: the plot functions index the module-level `signature_palette` as
    # a dict, but plotting.py defines it as a LIST. Inject the correct dict here.
    pl_mod.signature_palette = signature_palette or DEFAULT_SIGNATURE_PALETTE

    x_label = "Coeffient of variation of ranks in GOI"
    y_label = "Weighted F1-score (GOI vs Controls)"

    for bg_sign in msigdb_gmt.keys():
        if bg_sign in exclude_set:
            continue
        alt_signs = msigdb_gmt[bg_sign].keys()
        plot_dict = {i: fges_metrics[i] for i in alt_signs if i in fges_metrics}
        if not plot_dict:
            logger.debug("skipping {s}: no FGES present in metrics", s=bg_sign)
            continue
        pl_mod.plot_scatter_with_ci(
            plot_dict,
            title=f"Comparison for {bg_sign}",
            path=svg_dir,
            xlabel=x_label,
            ylabel=y_label,
        )

    pl_mod.plot_scatter_with_ci_agg(
        fges_metrics,
        title="Comparison for FGES types",
        path=save_dir,
        xlabel=x_label,
        ylabel=y_label,
    )
    logger.info("saved F1-vs-CV scatters under {d}", d=save_dir)
