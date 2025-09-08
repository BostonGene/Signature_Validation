from typing import Any, Dict, Optional

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csc_matrix
from scipy.stats import mannwhitneyu

from signature_validation.plotting.plotting import boxplot_with_pvalue, cells_p
from signature_validation.ssgsea_calc.ssgsea_calc import (
    GeneSet,
    gmt_genes_alt_names,
    ssgsea_formula,
)
from signature_validation.utils.utils import median_scale


def get_expression_dataframe(adata: anndata.AnnData) -> pd.DataFrame:
    """
    Convert an AnnData object to a Pandas DataFrame.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to convert.

    Returns
    -------
    expr_df : pd.DataFrame
        The Pandas DataFrame, with genes in the index and cells in the columns.

    Notes
    -----
    - Filter out genes with zero counts.
    - Use the `sparse` module to avoid loading the full matrix into memory.
    """
    adata.X = csc_matrix(adata.X)
    expr_df = pd.DataFrame.sparse.from_spmatrix(
        adata.X, index=adata.obs_names, columns=adata.var_names
    ).T
    row_sums = expr_df.sum(axis=1)
    row_sums = row_sums[row_sums > 0]
    expr_df = expr_df.loc[row_sums.index]
    return expr_df


def calculate_and_plot_ssgsea(
    adata: anndata.AnnData,
    expr_df: pd.DataFrame,
    geneset_of_interest: GeneSet,
    signature_name: str,
    celltype_col: str = "Cell_type",
    y_max: float | None = None,
    y_min: float | None = None,
    palette: dict[str, tuple[float, float, float, float]] = cells_p,
) -> tuple[pd.DataFrame, pd.Series[float] | pd.DataFrame, pd.Series[float], Any]:
    """
    Calculate single-sample gene set enrichment analysis (ssGSEA) scores for a given signature_name
    and plot the results as a boxplot.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data object containing expression, cell type information and other annotations.
    expr_df : pandas.DataFrame
        A pandas dataframe containing gene expression information.
    geneset_of_interest : list
        A list of genes that correspond to the signature_name.
    signature_name : str
        The name of the signature or process for which the ssGSEA scores should be calculated.
    celltype_col : str, optional
        The column name in adata.obs that contains cell type information. Defaults to "Cell_type".
    y_max : float, optional
        The maximum value for the y-axis of the boxplot. Defaults to None.
    y_min : float, optional
        The minimum value for the y-axis of the boxplot. Defaults to None.
    palette : list, optional
        A list of colors to use for the boxplot. Defaults to cells_p.

    Returns
    -------
    cprocesses : pandas.DataFrame
        A pandas dataframe containing the ssGSEA scores for each cell and each signature.
    cprocesses_sc : pandas.DataFrame
        A pandas dataframe containing the scaled ssGSEA scores for each cell and each signature.
    df_median : pandas.Series
        A pandas series containing the median ssGSEA score for each cell type.
    ax : matplotlib.Axes
        A matplotlib axes object containing the boxplot.

    """
    cprocesses = ssgsea_formula(
        expr_df,
        gmt_genes_alt_names(geneset_of_interest, expr_df.index, verbose=False),
        rank_method="average",
    ).T
    cprocesses_sc = median_scale(cprocesses)
    cprocesses["Cell_types"] = adata.obs[celltype_col].tolist()
    cprocesses["Cell_name"] = cprocesses_sc.index
    cprocesses_sc["Cell_types"] = adata.obs[celltype_col].tolist()
    cprocesses_sc["Cell_name"] = cprocesses_sc.index
    ctypes_median = pd.Series(data="NA", index=set(adata.obs[celltype_col].tolist()))
    for type_name in cprocesses["Cell_types"]:
        ctypes_median[type_name] = cprocesses.loc[
            cprocesses["Cell_types"] == type_name
        ][signature_name].median()
    df_median = ctypes_median.sort_values(ascending=False)
    ax = boxplot_with_pvalue(
        cprocesses_sc[signature_name],
        cprocesses_sc["Cell_types"],
        order=df_median.index,
        swarm=False,
        figsize=[10, 6],
        y_max=y_max,
        y_min=y_min,
        palette=palette,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    return cprocesses, cprocesses_sc, df_median, ax


def calculate_pvalues(
    cprocesses: pd.DataFrame,
    cprocesses_new: pd.DataFrame,
    populations: list,
    signature_name: str,
    signature_name_new: str,
    dataset: str,
) -> pd.DataFrame:
    """
    Calculate pvalues for a given signature against a reference signature for different cell populations.

    Parameters
    ----------
    cprocesses : pd.DataFrame
        A pandas dataframe containing the scaled ssGSEA scores for each cell and each signature.
    cprocesses_new : pd.DataFrame
        A pandas dataframe containing the scaled ssGSEA scores for each cell and each signature.
    populations : list
        A list of strings representing the different cell populations to be analyzed.
    signature_name : str
        The name of the original signature.
    signature_name_new : str
        The name of the new signature.
    dataset : str
        The name of the dataset.

    Returns
    -------
    pvalues : pd.DataFrame
        A pandas dataframe containing the pvalues for each population and each signature.
    """

    pvalues_greater = pd.Series(index=populations)
    pvalues_less = pd.Series(index=populations)
    for i in populations:
        samples_g1 = cprocesses_new[cprocesses_new.Cell_types == i].index
        samples_g2 = cprocesses[cprocesses.Cell_types == i].index
        try:
            if len(samples_g1) and len(samples_g2):
                pv_g = mannwhitneyu(
                    cprocesses_new.loc[samples_g1][signature_name_new],
                    cprocesses.loc[samples_g2][signature_name],
                    alternative="greater",
                ).pvalue
                pv_l = mannwhitneyu(
                    cprocesses_new.loc[samples_g1][signature_name_new],
                    cprocesses.loc[samples_g2][signature_name],
                    alternative="less",
                ).pvalue
            else:
                pv_g = 1
                pv_l = 1
        except ValueError:
            pv_g = 1
            pv_l = 1
        pvalues_greater[i] = pv_g
        pvalues_less[i] = pv_l
    pvalues_greater = pvalues_greater * (len(populations) - 1)
    for i in pvalues_greater.index:
        if pvalues_greater[i] > 1:
            pvalues_greater[i] = 1
    pvalues_less = pvalues_less * (len(populations) - 1)
    for i in pvalues_less.index:
        if pvalues_less[i] > 1:
            pvalues_less[i] = 1
    pvalues = pd.DataFrame()
    pvalues[f"{signature_name}_{dataset}"] = pvalues_less
    pvalues[f"{signature_name_new}_{dataset}"] = pvalues_greater
    return pvalues


def process_scrna_dataset(
    adata: anndata.AnnData,
    geneset_of_interest: GeneSet,
    signature_name: str,
    geneset_of_interest_new: GeneSet,
    signature_name_new: str,
    dataset: str,
    celltype_col: str = "Cell_type",
    y_max: Optional[float] = None,
    y_min: Optional[float] = None,
    palette: Dict[str, str] = cells_p,
    save_orig: Optional[str] = None,
    save_tuned: Optional[str] = None,
) -> None:
    """
    Process a single-cell RNA-seq dataset by calculating ssGSEA scores, plotting heatmaps and calculating pvalues.

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing the single-cell RNA-seq data.
    geneset_of_interest : dict
        A dictionary containing the genesets of interest.
    signature_name : str
        The name of the signature of interest.
    geneset_of_interest_new : dict
        A dictionary containing the genesets of interest for the refined signature.
    signature_name_new : str
        The name of the refined signature.
    dataset : str
        The name of the dataset.
    celltype_col : str
        The column name in the adata.obs DataFrame containing the cell types.
    y_max : float
        The maximum y value for the plot.
    y_min : float
        The minimum y value for the plot.
    palette : dict
        A dictionary containing the colors for each cell type.
    save_orig : str
        The filename to save the original UMAP plot to.
    save_tuned : str
        The filename to save the tuned UMAP plot to.

    Returns
    -------
    ax : matplotlib.Axes
        A matplotlib axes object containing the boxplot.
    ax1 : matplotlib.Axes
        A matplotlib axes object containing the boxplot for the refined signature.
    scs : pd.DataFrame
        A pandas DataFrame containing the median ssGSEA scores for each cell type.
    pvalues : pd.DataFrame
        A pandas DataFrame containing the pvalues for each population and each signature.
    """
    expr_df = get_expression_dataframe(adata)
    cprocesses, cprocesses_sc, df_median, ax = calculate_and_plot_ssgsea(
        adata,
        expr_df,
        geneset_of_interest,
        signature_name,
        celltype_col=celltype_col,
        y_max=y_max,
        y_min=y_min,
        palette=palette,
    )
    cprocesses_new, cprocesses_sc_new, df_median_new, ax1 = calculate_and_plot_ssgsea(
        adata,
        expr_df,
        geneset_of_interest_new,
        signature_name_new,
        celltype_col=celltype_col,
        y_max=y_max,
        y_min=y_min,
        palette=palette,
    )
    sc.pl.umap(adata, color=[celltype_col])
    adata.obs[signature_name] = cprocesses_sc[signature_name]
    sc.pl.umap(
        adata,
        color=[signature_name],
        title="Macrophage signature, Bindea et al.",
        save=save_orig,
    )
    adata.obs[signature_name_new] = cprocesses_sc_new[signature_name_new]
    sc.pl.umap(
        adata,
        color=[signature_name_new],
        title="Macrophage signature, tuned",
        save=save_tuned,
    )

    try:
        sc.pl.umap(
            adata, color=np.intersect1d(geneset_of_interest, expr_df.index).tolist()
        )
    except:
        print("UMAP for signature genes can not be plotted")
    try:
        sc.pl.umap(
            adata,
            color=np.intersect1d(geneset_of_interest_new, expr_df.index).tolist(),
        )
    except:
        print("UMAP for refined signature genes can not be plotted")
    scs = pd.DataFrame(index=df_median_new.index)
    scs[f"{signature_name}_{dataset}"] = df_median
    scs[f"{signature_name_new}_{dataset}"] = df_median_new
    pvalues = calculate_pvalues(
        cprocesses,
        cprocesses_new,
        scs.index,
        signature_name,
        signature_name_new,
        dataset,
    )
    return ax, ax1, scs, pvalues
