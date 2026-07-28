"""FGES selection and quality-metric helpers.

Originally relied on module-level reads from ``/internal_data`` and an undefined
``p`` variable, which broke ``import signature_validation.utils.fges_utils`` off
the BostonGene cluster. The data are now loaded lazily on first use, so the
module imports cleanly anywhere.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn import metrics
from tqdm import tqdm

from signature_validation.utils.utils import (
    read_dataset,
    read_expressions,
    scale_series,
)

_INTERNAL_DATA_DEFAULT = Path("/internal_data")

_msigdb_gmt_cache: Optional[Dict[str, Dict[str, Any]]] = None
_public_cells_cache: Optional[
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]
] = None


def _load_msigdb_gmt(
    internal_data_dir: Path = _INTERNAL_DATA_DEFAULT,
) -> Dict[str, Dict[str, Any]]:
    """Lazy: load and cache the MSigDb GMT pickle from the internal-data mount."""
    global _msigdb_gmt_cache
    if _msigdb_gmt_cache is None:
        with open(internal_data_dir / "msigdb_gmt.pkl", "rb") as handle:
            _msigdb_gmt_cache = pickle.load(handle)
    return _msigdb_gmt_cache


def _load_public_cells(
    internal_data_dir: Path = _INTERNAL_DATA_DEFAULT,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """Lazy: load (annot, log2 expr, pct ranks, gene list) from the internal mount."""
    global _public_cells_cache
    if _public_cells_cache is None:
        annot = read_dataset(internal_data_dir / "public_cells_annot.tsv.gz")
        expr = read_expressions(annot)
        expr = np.log2(expr + 1)
        ranked = expr.rank(pct=True)
        _public_cells_cache = (annot, expr, ranked, expr.index.to_list())
    return _public_cells_cache


def select_runs(
    goi_labels_good: pd.Series,
    goi_runs: pd.Series,
    goi_cell_types: pd.Series,
    goi_min_runs: pd.Series,
    n: Optional[int] = None,
) -> pd.Series:
    """
    Select representative runs for each label–cell type combination.

    Parameters
    ----------
    goi_labels_good : pd.Series
        Labels (e.g., gene or cluster assignments) for the samples of interest.
        Used for grouping.
    goi_runs : pd.Series
        Run identifiers for each sample (index-aligned with `goi_labels_good`).
    goi_cell_types : pd.Series
        Cell type annotation for each sample (index-aligned).
    goi_min_runs : pd.Series
        Minimal set of run identifiers per sample (index-aligned).
    n : int, optional
        If given, randomly sample `n` runs per label–cell type combination
        (with replacement). If None, use all available runs.

    Returns
    -------
    pd.Series
        Mapping of selected sample indices to their corresponding label.
        Index corresponds to selected sample IDs, values to label types.

    Notes
    -----
    - Sampling is reproducible (fixed random_state=42).
    - A progress bar is displayed with `tqdm`.
    - Total number of updates in the progress bar is estimated as:
      ``len(unique_labels) * len(unique_cell_types) * n`` if `n` is set,
      otherwise ``len(unique_labels) * len(goi_min_runs)``.
    """
    total = (
        len(goi_labels_good.unique()) * len(goi_cell_types.unique()) * n
        if n
        else len(goi_labels_good.unique()) * len(goi_min_runs)
    )
    pbar = tqdm(total=total, desc="Selecting runs...", position=0, leave=True)
    grouped = goi_labels_good.groupby(goi_labels_good)
    selected_labels = {}
    for typ, group in grouped:
        run_part = goi_runs.reindex(group.index)
        for ct in goi_cell_types.unique():
            if n:
                runs = (
                    goi_min_runs.reindex(goi_cell_types[goi_cell_types == ct].index)
                    .dropna()
                    .sample(n, replace=True, random_state=42)
                )
            else:
                runs = goi_min_runs.reindex(
                    goi_cell_types[goi_cell_types == ct].index
                ).dropna()
            for run in runs:
                sample = run_part[run_part == run].sample(1, replace=False)
                selected_labels[sample.index[0]] = typ
                pbar.update(1)
    pbar.close()
    return pd.Series(selected_labels)


def get_ls_and_ss(
    ser: pd.Series,
    fges_type: str,
    fges: str,
) -> Tuple[pd.Series, pd.Series]:
    """
    Construct label series (ls) and score series (ss) with new indexed names.

    Parameters
    ----------
    ser : pd.Series
        Input series of values to transform. Index should represent sample IDs
        or features that will be expanded.
    fges_type : str
        Type or category label (e.g., "EMT", "Metastasis").
    fges : str
        Specific FGES signature identifier.

    Returns
    -------
    ls : pd.Series
        Series with the same length as `ser`.
        Index: renamed as "<original_index>_<fges_type>_<fges>".
        Values: constant string `fges_type`.
    ss : pd.Series
        Series with the same length as `ser`.
        Index: same as `ls`.
        Values: original values from `ser`.

    Examples
    --------
    >>> import pandas as pd
    >>> ser = pd.Series([0.1, 0.2], index=["geneA", "geneB"])
    >>> ls, ss = get_ls_and_ss(ser, "EMT", "BG")
    >>> ls
    geneA_EMT_BG    EMT
    geneB_EMT_BG    EMT
    dtype: object
    >>> ss
    geneA_EMT_BG    0.1
    geneB_EMT_BG    0.2
    dtype: float64
    """
    new_ind = ser.index.map(lambda x: f"{x}_{fges_type}_{fges}")
    ls = pd.Series(index=new_ind, data=[fges_type] * len(new_ind))
    ss = pd.Series(index=new_ind, data=ser.values)
    return ls, ss


def get_metric_for_signature(
    series: pd.Series,
    labels: pd.Series,
    verbose: bool = False,
    youden_thr: bool = False,
    sign: str = "",
) -> Dict[str, float]:
    """
    Calculate metrics for a given signature.

    Parameters
    ----------
    series : pandas Series
        Expression values of the signature genes.
    labels : pandas Series
        Binary labels for the cell types.
    verbose : bool, optional
        Whether to print the results. Default is False.
    youden_thr : bool, optional
        Whether to use Youden's index to determine the threshold. Default is False.
    sign : str, optional
        The name of the signature. Default is an empty string.

    Returns
    -------
    metrics_dict : dict
        A dictionary with the following keys:
            - F1
            - Accuracy
            - Precision_score
            - Average_precision
            - ROC_AUC
            - PR_AUC
            - Recall_score
    """
    y_test_bin = labels
    y_pred = scale_series(series)
    fpr, tpr, thresholds = metrics.roc_curve(y_test_bin, y_pred)
    if youden_thr:
        thr = thresholds[np.argmax(tpr - fpr)]
    else:
        thr = thresholds[np.argmin(np.sqrt((0 - fpr) ** 2 + (1 - tpr) ** 2))]
    roc_auc = metrics.auc(fpr, tpr)
    # `thr` can be `np.inf` (sklearn's synthetic "reject everything" threshold,
    # `roc_curve`'s `thresholds[0]`) when that's the optimal operating point on a
    # degenerate ROC curve (e.g. tiny bootstrap samples) — a fixed-bin `pd.cut`
    # then breaks on non-monotonic bins. Thresholding directly handles any `thr`.
    y_pred_bin = (y_pred > thr).astype(int)
    f1 = metrics.f1_score(y_test_bin, y_pred_bin, average="weighted")
    accuracy = metrics.accuracy_score(y_test_bin, y_pred_bin)
    recall_score = metrics.recall_score(y_test_bin, y_pred_bin)
    precision_score = metrics.precision_score(y_test_bin, y_pred_bin)
    av_precision = metrics.average_precision_score(y_test_bin, y_pred)
    precision, recall, _ = metrics.precision_recall_curve(y_test_bin, y_pred)
    pr_auc = metrics.auc(recall, precision)
    if verbose:
        print(sign, "\t", "recall_score", "\t", f"{recall_score:.4f}")
        print(sign, "\t", "precision_score", "\t", f"{precision_score:.4f}")
        print(sign, "\t", "accuracy", "\t", f"{accuracy:.4f}")
        print(sign, "\t", "f1", "\t", f"{f1:.4f}")
        print(sign, "\t", "roc auc", "\t", f"{roc_auc:.4f}")
        print(sign, "\t", "average precision", "\t", f"{av_precision:.4f}")
        print(sign, "\t", "pr auc", "\t", f"{pr_auc:.4f}")
    m_dict = {
        "F1": f1,
        "Accuracy": accuracy,
        "Precision_score": precision_score,
        "Average_precision": av_precision,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "Recall_score": recall_score,
    }
    return m_dict


def derive_rank_deviation(
    control: pd.Series,
    goi: pd.Series,
    sign: str,
    fges: str,
    msigdb_gmt: Optional[Dict[str, Dict[str, Any]]] = None,
    ranked_expr: Optional[pd.DataFrame] = None,
    pipeline_genes: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Calculate the rank deviation for control and goi groups.

    Parameters
    ----------
    control : pd.Series
        Series of control sample names.
    goi : pd.Series
        Series of group of interest sample names.
    sign : str
        Signature name.
    fges : str
        Functional gene expression signature (FGES) name.
    msigdb_gmt : dict, optional
        ``{Main4_*: {sub_signature: GeneSet}}`` mapping. Loaded lazily from the
        internal-data mount when None.
    ranked_expr : pd.DataFrame, optional
        Gene × sample matrix of percentile ranks. Loaded lazily from the
        internal-data mount when None.
    pipeline_genes : list of str, optional
        Genes available in the expression pipeline. Loaded lazily when None.

    Returns
    -------
    dev_dict : dict
        Dictionary with rank deviation metrics for control and goi groups.
    """
    if msigdb_gmt is None:
        msigdb_gmt = _load_msigdb_gmt()
    if ranked_expr is None or pipeline_genes is None:
        _, _, _ranked, _genes = _load_public_cells()
        ranked_expr = ranked_expr if ranked_expr is not None else _ranked
        pipeline_genes = pipeline_genes if pipeline_genes is not None else _genes
    gs = [i for i in msigdb_gmt[sign][fges].genes if i in pipeline_genes]
    goi_ranked_df = ranked_expr[goi.index].loc[gs].T
    control_ranked_df = ranked_expr[control.index].loc[gs].T
    goi_cv = (goi_ranked_df.std() / goi_ranked_df.mean()).mean()
    control_cv = (control_ranked_df.std() / control_ranked_df.mean()).mean()
    goi_std = (goi_ranked_df.std()).mean()
    control_std = (control_ranked_df.std()).mean()
    return {
        "goi_cv": goi_cv,
        "control_cv": control_cv,
        "goi_std": goi_std,
        "control_std": control_std,
    }


def get_strat_cell_type(
    control: pd.Series,
    seed: int,
    public_cells_annot: Optional[pd.DataFrame] = None,
) -> Tuple[pd.Series, pd.Series]:
    """
    Sample a subset of control samples, such that each cell type is
    represented by at least min_samples samples.

    Parameters
    ----------
    control : pd.Series
        Series of control sample names.
    seed : int
        Random seed for sampling.
    public_cells_annot : pd.DataFrame, optional
        Annotation indexed by sample with at least a ``Cell_type`` column.
        Loaded lazily from the internal-data mount when None.

    Returns
    -------
    new_control : pd.Series
        Series of control sample names, with each cell type represented
        by at least min_samples samples.
    cell_types : pd.Series
        Series of cell type labels for the samples in new_control.
    """
    if public_cells_annot is None:
        public_cells_annot, _, _, _ = _load_public_cells()
    cell_types = public_cells_annot.Cell_type.reindex(control.index).dropna()
    min_samples = cell_types.value_counts().min()
    sampled_indices: List[Any] = []
    for cell_type in cell_types.unique():
        cell_type_indices = cell_types[cell_types == cell_type].index
        np.random.seed(seed)
        sampled_indices.extend(
            np.random.choice(cell_type_indices, min_samples, replace=False)
        )
    cell_types = cell_types.loc[sampled_indices]
    new_control = control.reindex(cell_types.index).dropna()
    return new_control, cell_types
