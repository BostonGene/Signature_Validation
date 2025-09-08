import pickle
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn import metrics
from tqdm import tqdm

from signature_validation.utils.utils import (
    read_dataset,
    read_expressions,
    scale_series,
)

with open(p / "msigdb_gmt.pkl", "rb") as handle:
    msigdb_gmt = pickle.load(handle)

public_cells_annot = read_dataset(
    "/internal_data/public_cells_annot.tsv.gz"
)  # Sharing by request
public_cells_expr = read_expressions(public_cells_annot)
public_cells_expr = np.log2(public_cells_expr + 1)
pipeline_genes = public_cells_expr.index.to_list()
ranked_expr = public_cells_expr.rank(pct=True)


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
) -> pd.Series:
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
    y_pred_bin = pd.cut(y_pred, bins=[-1, thr, 777], labels=[0, 1])
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
    control: pd.Series, goi: pd.Series, sign: str, fges: str
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

    Returns
    -------
    dev_dict : dict
        Dictionary with rank deviation metrics for control and goi groups.
    """
    gs = [i for i in msigdb_gmt[sign][fges].genes if i in pipeline_genes]
    goi_ranked_df = ranked_expr[goi.index].loc[gs].T
    control_ranked_df = ranked_expr[control.index].loc[gs].T
    goi_cv = (goi_ranked_df.std() / goi_ranked_df.mean()).mean()
    control_cv = (control_ranked_df.std() / control_ranked_df.mean()).mean()
    goi_std = (goi_ranked_df.std()).mean()
    control_std = (control_ranked_df.std()).mean()
    dev_dict = {
        "goi_cv": goi_cv,
        "control_cv": control_cv,
        "goi_std": goi_std,
        "control_std": control_std,
    }
    return dev_dict


def get_strat_cell_type(control: pd.Series, seed: int) -> Tuple[pd.Series, pd.Series]:
    """
    Sample a subset of control samples, such that each cell type is
    represented by at least min_samples samples. The samples are
    chosen randomly with the given seed.

    Parameters
    ----------
    control : pd.Series
        Series of control sample names.
    seed : int
        Random seed for sampling.

    Returns
    -------
    new_control : pd.Series
        Series of control sample names, with each cell type represented
        by at least min_samples samples.
    cell_types : pd.Series
        Series of cell type labels for the samples in new_control.
    """
    cell_types = public_cells_annot.Cell_type.reindex(control.index).dropna()
    min_samples = cell_types.value_counts().min()
    sampled_indices = []
    for cell_type in cell_types.unique():
        cell_type_indices = cell_types[cell_types == cell_type].index
        np.random.seed(seed)
        sampled_indices.extend(
            np.random.choice(cell_type_indices, min_samples, replace=False)
        )
    cell_types = cell_types.loc[sampled_indices]
    new_control = control.reindex(cell_types.index).dropna()
    return new_control, cell_types
