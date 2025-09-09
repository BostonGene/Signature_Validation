import itertools
import warnings
from pathlib import Path
from typing import Any, Dict, Generator, List, Literal, Optional, Tuple, Union

import distinctipy
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import linregress, mannwhitneyu, pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error

from signature_validation.ssgsea_calc.ssgsea_calc import detect_fges_source
from signature_validation.utils.utils import (
    get_pvalue_string,
    item_series,
    sort_by_terms_order,
    to_common_samples,
)

default_cmap = matplotlib.cm.coolwarm


signature_palette = [
    "Internal",
    "Random_FGES",
    "xCell",
    "Bindea",
    "Nirmal",
    "Gene_Ontology",
    "KEGG",
    "BioCarta",
    "WikiPathways",
    "Reactome",
    "Human_Phenotype_Ontology",
    "MSigDb_Dif_Expression",
    "MSigDb_Single_Cell",
    "MSigDb_Other",
    "Pathway_Interaction_Database",
]
colors = distinctipy.get_colors(len(signature_palette), n_attempts=1000)

cells_p = {
    "B_cells": "#004283",
    "Plasma_B_cells": "#0054A8",
    "Non_plasma_B_cells": "#0066CC",
    "Mature_B_cells": "#3889DB",
    "Naive_B_cells": "#78B0E9",
    "T_cells": "#285A51",
    "CD8_T_cells": "#31685E",
    "CD8_T_cells_PD1_high": "#3C776C",
    "CD8_T_cells_PD1_low": "#3C776C",
    "CD4_T_cells": "#61A197",
    "Th": "#70B0A5",
    "Th1_cells": "#7FBEB3",
    "Th2_cells": "#8FCCC2",
    "Th17_cells": "#9DD4C9",
    "Naive_T_helpers": "#ACDCD3",
    "Tregs": "#CBEBE6",
    "NK_cells": "#6181A1",
    "Cytotoxic_NK_cells": "#7F9EBE",
    "Regulatory_NK_cells": "#9DB8D4",
    "Myeloid_cells": "#8C0021",
    "Monocytes": "#6A3C77",
    "Macrophages": "#865494",
    "Macrophages_M1": "#A370B0",
    "Macrophages_M2": "#BF8FCC",
    "Microglia": "#6B4F73",
    "MDSC": "#9F86A6",
    "Granulocytes": "#D93158",
    "Eosinophils": "#B7002B",
    "Neutrophils": "#EC849C",
    "Basophils": "#854855",
    "Mast_cells": "#B0707D",
    "Dendritic_cells": "#50285B",
    "Endothelium": "#DCB7AC",
    "Vascular_endothelium_cells": "#DCB7AC",
    "Lymphatic_endothelium_cells": "#998078",
    "Stromal_cells": "#CC7A00",
    "Fibroblasts": "#FF9500",
    "iCAF": "#FFB341",
    "myCAF": "#FFCD83",
    "Follicular_dendritic_cells": "#D2871E",
    "Adypocytes": "#ECDAA7",
    "Fibroblastic_reticular_cells": "#995B00",
    "Other": "#C2C1C7",
    "Epithelial_cells": "#DFD3CF",
    "Muscles": "#DF714B",
    "Bones": "#96A4B3",
    "Epithelium": "#DFD3CF",
    "Pericytes_and_smooth_muscle": "#CC7A00",
    "Glial_cells": "#995B00",
}


def complementary_color(hex_color: str) -> str:
    """
    Returns complementary RGB color
    """
    c_hex_color = hex_color.lstrip("#")
    rgb = (c_hex_color[0:2], c_hex_color[2:4], c_hex_color[4:6])
    return "#{}".format("".join(["%02X" % (255 - int(a, 16)) for a in rgb]))


def simple_scatter(x, y, ax=None, title="", color="b", figsize=(5, 5), s=20, **kwargs):
    """
    Plot a scatter for 2 vectors. Only samples with common indexes are plotted.
    If color is a pd.Series - it will be used to color the dots
    :param x: pd.Series, numerical values
    :param y: pd.Series, numerical values
    :param ax: matplotlib axis, axis to plot on
    :param title: str, plot title
    :param color: str, color to use for points
    :param figsize: (float, float), figure size in inches
    :param s: float, size of points
    :param alpha: float, alpha of points
    :param marker: str, marker to use for points
    :param linewidth: float, width of marker borders
    :param edgecolor: str, color of marker borders
    :return: matplotlib axis
    """

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    try:
        c_x, c_y, c_color = to_common_samples([x, y, color])
    except Exception:
        c_x, c_y = to_common_samples([x, y])
        c_color = color

    ax.set_title(title)

    ax.scatter(c_x, c_y, color=c_color, s=s, **kwargs)

    if hasattr(x, "name"):
        ax.set_xlabel(x.name)
    if hasattr(y, "name"):
        ax.set_ylabel(y.name)

    return ax


def simple_palette_scatter(
    x: pd.Series,
    y: pd.Series,
    grouping: pd.Series,
    palette: Optional[Dict[Any, str]] = None,
    order: List[Any] = None,
    centroids: bool = False,
    legend: bool = "out",
    patch_size: int = 10,
    centroid_complement_color: bool = True,
    ax: Optional[matplotlib.axes.Axes] = None,
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Plot a scatter for 2 vectors, coloring by grouping.
    Only samples with common indexes are plotted.

    See Also
    --------------------
    plotting.simple_scatter

    Parameters
    --------------------
    x: pd.Series
        numerical values
    y: pd.Series
        numerical values
    grouping: pd.Series
        which group each sample belongs to
    palette: dict
        palette for plotting. Keys are unique values from groups, entries are color hexes
    order: list
        order to plot the entries in. Contains ordered unique values from grouping
    centroids: bool
        whether to plot centroids of each group
    legend: bool
        whether to plot legend
    patch_size: float
        size of legend
    centroid_complement_color: bool
        whether to plot centroids in complement color
    ax: plt
        axes to plot on

    Returns
    --------------------
    matplotlib axis
    """

    if palette is None:
        palette = lin_colors(grouping)

    if order is None:
        order = np.sort(list(palette.keys()))

    c_grouping, c_x, c_y = to_common_samples(
        [grouping[sort_by_terms_order(grouping, order)], x, y]
    )

    patch_location = 2
    if "loc" in kwargs:
        patch_location = kwargs.pop("loc")

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(kwargs.get("figsize", (4, 4))))

    kwargs["marker"] = kwargs.get("marker", "o")
    kwargs["edgecolor"] = kwargs.get("edgecolor", "black")
    kwargs["linewidth"] = kwargs.get("linewidth", 0)

    for label in order:
        samps = c_grouping[c_grouping == label].index
        simple_scatter(c_x[samps], c_y[samps], color=palette[label], ax=ax, **kwargs)
        handles = [mpatches.Patch(color=palette[label], label=label) for label in order]

    if centroids:
        c_color = "black"
        for label in order:
            samps = c_grouping[c_grouping == label].index
            mean_x = c_x[samps].mean()
            mean_y = c_y[samps].mean()
            if centroid_complement_color:
                c_color = complementary_color(palette[label])

            if "s" in kwargs:
                s = kwargs["s"]
            else:
                s = 20
            ax.scatter(
                mean_x,
                mean_y,
                marker="*",
                lw=1.5,
                s=s * 10,
                edgecolor=c_color,
                color=palette[label],
            )

    if legend:
        ax.legend(
            bbox_to_anchor=(1, 1) if legend == "out" else None,
            handles=handles,
            loc=patch_location,
            prop={"size": patch_size},
            borderaxespad=0.1,
        )

    return ax


def matrix_projection_plot(
    data: pd.DataFrame,
    grouping: Optional[pd.Series] = None,
    order: Optional[List[Any]] = None,
    n_components: int = 2,
    ax: Optional[matplotlib.axes.Axes] = None,
    palette: Optional[Dict[Any, str]] = None,
    confidence_level: bool = False,
    centroids: bool = False,
    centroid_complement_color: bool = False,
    random_state: int = 42,
    figsize: Tuple[float, float] = (5, 5),
    title: str = "",
    return_model: bool = False,
    legend: Union[str, None] = "out",
    plot_limits: bool = False,
    label_samples: bool = False,
    kwargs_scatter: Optional[Dict[Any, Any]] = None,
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Performs a data dimensionality reduction using p_model
    and then plots as a scatter plot colored by grouping.
    If n_components > 2, then first two components will be chosen for plotting.

    Usually the function is not called directly,
    but is called via pca_plot(), tsne_plot() ot umap_plot().

    :param data: Samples as indexes and features as columns.
    :param grouping: Series of samples to group correspondence.
    :param order: Groups plotting order (useful for limited groups displaying)
    :param n_components: int, default: 2 The number of dimensions to reduce the data to.
    :param ax: axes to plot
    :param palette: Colors corresponding to the groups. If None -> lin_colors will be applied.
    :param confidence_level: {80, 90, 95, 99} or False, defaut: False
        Confidence level, based on pearson correlation and standard deviation
    :param return_model: bool, default: False
        If True -> return Tuple[ax, transformed_data, model]
    :param alpha: plotting option
    :param random_state: 42
    :param s: point size
    :param figsize: if ax=None a new axis with the figsize will be created
    :param title: plot title
    :param legend: {'in', 'out'} or None, default: 'in'
        'in' - plots the legend inside the plot, 'out' - outside. Otherwise - no legend
    :param plot_limits: limits axes size respect to a plot
    :param label_samples: bool, default: False
        Whether to subscribe samples' names on plot.
    :param kwargs_scatter: dict
        Dict with various params for ax.scatter - marker, linewidth, edgecolor.
    :param kwargs: kwargs for projection model, n_jobs is set to 4 by default for UMAP and TSNE
    :return: matplotlib.axes.Axes
    """
    kwargs.setdefault("n_jobs", 4)

    if grouping is None:
        grouping = item_series("*", data)

    # Common samples
    c_data, c_grouping = to_common_samples([data, grouping])

    if order:
        group_order = order
    else:
        group_order = np.sort(c_grouping.unique())

    if palette is None:
        cur_palette = lin_colors(c_grouping)
    else:
        cur_palette = palette

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Get model and transform
    n_components = min(n_components, len(c_data.columns))

    kwargs.pop("n_jobs", "None")  # PCA does not support n_jobs
    model = PCA(n_components=n_components, random_state=random_state, **kwargs)

    data_tr = pd.DataFrame(model.fit_transform(c_data), index=c_data.index)

    label_1 = "PCA 1 component {}% variance explained".format(
        int(model.explained_variance_ratio_[0] * 100)
    )
    label_2 = "PCA 2 component {}% variance explained".format(
        int(model.explained_variance_ratio_[1] * 100)
    )

    kwargs_scatter = kwargs_scatter or {}
    simple_palette_scatter(
        x=data_tr[0],
        y=data_tr[1],
        grouping=c_grouping,
        order=group_order,
        palette=cur_palette,
        ax=ax,
        legend=legend,
        confidence_level=confidence_level,
        centroids=centroids,
        centroid_complement_color=centroid_complement_color,
        **kwargs_scatter,
    )

    if plot_limits:
        x_lim_min = data_tr[0].min()
        x_lim_max = data_tr[0].max()
        y_lim_min = data_tr[1].min()
        y_lim_max = data_tr[1].max()
        if label_samples:
            delta_x = (x_lim_max + x_lim_min) / 2
            delta_y = (y_lim_max + y_lim_min) / 2
        else:
            delta_x = (x_lim_max + x_lim_min) / 20
            delta_y = (y_lim_max + y_lim_min) / 20

        ax.set_xlim([x_lim_min - delta_x, x_lim_max + delta_x])
        ax.set_ylim([y_lim_min - delta_y, y_lim_max + delta_x])

    if label_samples:
        texts = []
        sample_names = list(c_grouping.index)
        X = list(data_tr[0])
        Y = list(data_tr[1])
        for i, name in enumerate(sample_names):
            texts.append(plt.text(X[i], Y[i], s=str(name), fontsize=8))
        adjust_text(
            texts, x=X, y=Y, arrowprops=dict(arrowstyle="-", color="black", lw=0.1)
        )

    ax.set_title(title)
    ax.set_xlabel(label_1)
    ax.set_ylabel(label_2)

    if return_model:
        return ax, data_tr, model
    return ax


def pca_plot(
    data: pd.DataFrame,
    grouping: Optional[pd.Series] = None,
    n_components: int = 2,
    title: str = "",
    ax: Optional[matplotlib.axes.Axes] = None,
    order: Optional[List[Any]] = None,
    palette: Optional[Dict[Any, str]] = None,
    confidence_level: bool = False,
    centroids: bool = False,
    centroid_complement_color: bool = False,
    return_model: bool = False,
    legend: Union[str, None] = "out",
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Perform dimensionality reduction using Principal Component Analysis
    and plot results as scatter plot.

    See Also
    ------------------
    plotting.matrix_projection_plot

    :param data: pd.DataFrame
        Samples as indexes and features as columns.
    :param grouping: pd.Series
        Series of samples to group correspondence.
    :param n_components: int, default: 2
        The number of components, that will be calculated.
    :param title: str, title of plot
    :param ax: axes to plot
    :param order: list
        Groups plotting order (useful for limited groups displaying)
    :param palette: dict
        Colors corresponding to the groups.
        If None -> lin_colors will be applied.
    :param confidence_level: {80, 90, 95, 99} or False, defaut: False
        Confidence level, based on pearson correlation and standard deviation
    :param return_model: bool, default: False
        If True -> return Tuple[ax, transformed_data, model]
    :param legend: {'in', 'out'} or None, default: 'in'
        'in' - plots the legend inside the plot, 'out' - outside. Otherwise - no legend
    :param kwargs:
    :return: matplotlib.axes.Axes

    Example
    ------------------------
    # If we want to get four components from PCA
    ax, transformed_data, model = pca_plot(data=expressions, grouping=subtype, return_model=True, n_components=4)

    # If we want to annotate outliers
    pca_plot(data=expressions, grouping=subtype, label_samples=True)
    """

    kwargs_scatter = dict()
    kwargs_scatter["linewidth"] = kwargs.pop("linewidth", 0)
    kwargs_scatter["marker"] = kwargs.pop("marker", "o")
    kwargs_scatter["edgecolor"] = kwargs.pop("edgecolor", "black")
    kwargs_scatter["s"] = kwargs.pop("s", 20)
    kwargs_scatter["alpha"] = kwargs.pop("alpha", 1)

    return matrix_projection_plot(
        data=data,
        grouping=grouping,
        n_components=n_components,
        title=title,
        ax=ax,
        order=order,
        palette=palette,
        return_model=return_model,
        legend=legend,
        confidence_level=confidence_level,
        centroids=centroids,
        centroid_complement_color=centroid_complement_color,
        kwargs_scatter=kwargs_scatter,
        **kwargs,
    )


def genes_heatmap(
    expr: pd.DataFrame,
    annot: pd.Series,
    genes: list,
    row_cluster: bool = True,
    col_cluster: bool = False,
    z_scores: Optional[Union[int, None]] = 0,
    standard_scale: Optional[bool] = None,
    show_expression: bool = True,
    show_indices: bool = False,
    decimals: int = 0,
    highlight_genes: Optional[list] = None,
    gene_highlight_color: str = "black",
    gene_highlight_lw: int = 2,
    highlight_columns: Optional[list] = None,
    col_highlight_color: str = "black",
    col_highlight_lw: int = 2,
    x_axis_annot: bool = True,
    font_scale: int = 2,
    figsize: tuple = (30, 30),
    yticklabels: int = 1,
    cmap: str = "Spectral_r",
    row_dendrogram: bool = False,
    col_dendrogram: bool = False,
    colormap_legend: bool = False,
    fmt: str = "g",
    annot_size: int = 12,
):
    """
    Heatmap for checking on expression signatures of RNA-seq samples.
    :param expr: pd.DataFrame with genes as rows and samples as columns and TPM values in the cells
    :param annot: pd.Series with the desired order of samples as indices and annotations as values
    :param genes: a list of plotted genes
    :param row_cluster: boolean, cluster heatmap by rows
    :param col_cluster: boolean, cluster heatmap by columns
    :param z_scores: 0 (genes), 1 (samples) or None:  z = (x - mean)/std, by default genes
    :param standard_scale: 0 (rows), 1 (columns) or None. For each row/column subtract the minimum
                            and divide each by its maximum. mutually exclusive with z_scores
    :param show_expression: boolean, show rounded expression values in the cells
    :param show_indices: boolean, creates the second x axis with indices as labels above the heatmap
    :param decimals: int, the number of decimals after the comma while rounding
    :param highlight_genes: list, the genes to highlight, default is None
    :param gene_highlight_color: the color of the border, default is black
    :param gene_highlight_lw: int, line weight of the border, default is 2
    :param highlight_columns: list, the samples to highlight, default is None
    :param col_highlight_color: the color of the border, default is black
    :param col_highlight_lw: int, line weight of the border, default is 2
    :param x_axis_annot: if True, shows sample ids as x axis labels instead of annotation values
    :param font_scale: specify font size
    :param figsize: figsize if ax=None
    :param yticklabels: If True, plot row names of the dataframe. If an integer, use row names but plot
                        only every n label. If “auto”, try to densely plot non-overlapping labels.
    :param cmap: colormap for plotting
    :param row_dendrogram: boolean, display row dendrogram
    :param col_dendrogram: boolean, display column dendrogram
    :param colormap_legend: boolean, display colormap legend
    :param fmt: string formatting code to use when adding annotations
    :param annot_size: font size of the annotation
    :returns: ax
    """
    ann_name = annot.name
    annotation = pd.DataFrame({ann_name: annot})
    inds = annotation.index
    # subssetting expr dataframe to plotting values
    expression = expr.loc[genes, inds]
    # dropping genes in case if all 0
    expression = expression[(expression.T != 0).any()]

    sns.set(font_scale=font_scale)
    if show_expression is True:
        g = sns.clustermap(
            np.log(1 + expression),
            row_cluster=row_cluster,
            col_cluster=col_cluster,
            z_score=z_scores,
            annot=round(expression, decimals),
            fmt=fmt,
            annot_kws={"size": annot_size},
            standard_scale=standard_scale,
            yticklabels=yticklabels,
            figsize=figsize,
            cmap=cmap,
        )
    else:
        g = sns.clustermap(
            np.log(1 + expression),
            row_cluster=row_cluster,
            col_cluster=col_cluster,
            z_score=z_scores,
            standard_scale=standard_scale,
            yticklabels=yticklabels,
            figsize=figsize,
            cmap=cmap,
        )
    ax = g.ax_heatmap
    if show_indices:
        ax2 = ax.twiny()
        ax2.set_xticks(np.arange(0.5, expression.shape[1]))
        ax2.set_xlim([0, expression.shape[1]])
        ax2.grid(False)
    if highlight_genes is not None:
        expr_length = expression.shape[1]
        for gene in highlight_genes:
            if row_cluster:
                k = g.dendrogram_row.reordered_ind.index(expression.index.get_loc(gene))
            else:
                k = expression.index.get_loc(gene)
            ax.add_patch(
                Rectangle(
                    (0, k),
                    expr_length,
                    1,
                    fill=False,
                    edgecolor=gene_highlight_color,
                    lw=gene_highlight_lw,
                )
            )
    if highlight_columns is not None:
        col_len = expression.shape[0]
        for col in highlight_columns:
            if col_cluster:
                k = g.dendrogram_col.reordered_ind.index(
                    expression.columns.get_loc(col)
                )
            else:
                k = expression.columns.get_loc(col)
            ax.add_patch(
                Rectangle(
                    (k, 0),
                    1,
                    col_len,
                    fill=False,
                    edgecolor=col_highlight_color,
                    lw=col_highlight_lw,
                )
            )
    g.ax_row_dendrogram.set_visible(row_dendrogram)
    g.ax_col_dendrogram.set_visible(col_dendrogram)
    g.cax.set_visible(colormap_legend)
    if x_axis_annot:
        if col_cluster:
            ax.set_xticklabels(
                annotation[ann_name].iloc[g.dendrogram_col.reordered_ind]
            )
            if show_indices:
                ax2.set_xticklabels(
                    annotation.iloc[g.dendrogram_col.reordered_ind].index, rotation=90
                )
        else:
            ax.set_xticklabels(annotation.loc[inds, ann_name])
            if show_indices:
                ax2.set_xticklabels(annotation.index, rotation=90)

    ax.set_ylabel("")
    ax.set_xlabel(ann_name)
    return ax


def boxplot_with_pvalue_new(
    data: pd.Series,
    categories: pd.Series,
    pvalues: pd.Series,
    columns_list: List[str],
    signs: pd.Series,
    palette: Optional[Dict[str, str]] = None,
    figsize: Tuple[float, float] = (12, 6),
    s: int = 1,
    p_fontsize: int = 16,
    p_digits: int = 3,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    stars: bool = True,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
):
    """
    Creates boxplots with swarmplots for selected columns, grouped by categories with p-values annotated.

    :param data: pd.Series, numerical data.
    :param categories: pd.Series, categorical data.
    :param p_values: pd.Series, p-values for each column.
    :param columns_list: list, list of column names to plot.
    :param signs: pd.Series, categorical data for labeling x-axis.
    :param palette: dict, colors for categories.
    :param figsize: tuple, size of the figure.
    :param s: int, size of the dots in swarmplot.
    :param p_fontsize: int, font size for p-value annotations.
    """
    plt.rcParams.update({"font.size": 20})
    if palette is None:
        palette = {categories.unique()[0]: "white", categories.unique()[1]: "blue"}

    # Setup the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Create box and swarm plots
    sns.boxplot(x=signs, y=data, hue=categories, ax=ax, palette=palette, dodge=True)
    sns.swarmplot(
        x=signs,
        y=data,
        hue=categories,
        ax=ax,
        color="black",
        dodge=True,
        edgecolor="gray",
        linewidth=1,
        size=s,
    )

    y_max = y_max or data.max()
    y_min = y_min or data.min()
    effective_size = y_max - y_min
    plot_y_limits = (y_min - effective_size * 0.15, y_max + effective_size * 0.2)

    if p_digits > 0:
        pvalue_line_y_1 = y_max + effective_size * 0.05
        if figsize is None:
            figsize = define_ax_figsize(ax)
        pvalue_text_y_1 = pvalue_line_y_1 + 0.25 * effective_size / figsize[1]
        unique_signs = signs.unique()
        middle_pos = -0.5
        for i, sign in enumerate(unique_signs):
            middle_pos = -0.5 + i
            p_val = p_values.loc[sign]
            pvalue_str = get_pvalue_string(p_val, p_digits, stars=stars)
            pvalue_text_y_1_local = pvalue_text_y_1
            if pvalue_str == "-":
                pvalue_text_y_1_local += 0.1 * effective_size / figsize[1]
            bar_fraction = str(0.25 / 2.0 / (figsize[0] / float(len(order))))

            pos = middle_pos
            ax.annotate(
                "",
                xy=(pos + 0.2, pvalue_line_y_1),
                xycoords="data",
                xytext=(pos + 0.8, pvalue_line_y_1),
                textcoords="data",
                arrowprops=dict(
                    arrowstyle="-",
                    ec="#000000",
                    connectionstyle="bar,fraction={}".format(bar_fraction),
                ),
            )
            ax.text(
                pos + 0.5,
                pvalue_text_y_1_local,
                pvalue_str,
                fontsize=p_fontsize,
                horizontalalignment="center",
                verticalalignment="center",
            )

    ax.set_title(title)
    ax.set_ylim(plot_y_limits)
    ax.set_xlabel(xlabel)
    # Remove duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    ax.set_xticklabels(order, rotation=90)
    ax.legend(
        handles[: int(len(handles) / 2)],
        categories.unique(),
        title="Category",
        loc="upper right",
        bbox_to_anchor=(1.2, 1),
    )


def plot_waterfall(
    df_means: pd.DataFrame,
    df_p_values: pd.DataFrame,
    df_stdev: pd.DataFrame,
    main_column: Optional[str] = None,
    title: Optional[str] = None,
    labels: Optional[pd.Series] = None,
    save: bool = True,
    path_to_save: str = "~/waterfall.",
    format: str = "svg",
    sort_by_list: Optional[List[str]] = False,
) -> None:
    """
    Plots a waterfall plot for the difference between mean scores.

    Parameters:
        - df_means (pd.DataFrame): DataFrame with mean scores for each group (columns) and signature (rows).
        - df_p_values (pd.DataFrame): DataFrame with p-values for comparison between groups, signatures in rows, one column.
        - df_stdev (pd.DataFrame): DataFrame with standard deviation of scores for each group (columns) and signature (rows).
        - main_column (str): Name of the column from which to subtract the other column. For example, if main_column is a target, the plot will show the difference between the target and non-target.
        - title (str): Title of the plot.
        - labels (pd.Series): Series with names for each signature, if you want to add other names.
        - save (bool): Save the plot. Default True.
        - path_to_save (str): Path to save the file, if not specified, saves to "~/waterfall_plot.svg"
        - format (str): Format of the saved file. Default svg.

    """

    groups = df_means.columns
    dif = (
        df_means.T.loc[[main_column]].iloc[0]
        - df_means.T.loc[[i for i in groups if i != main_column]].iloc[0]
    )

    if sort_by_list:
        dif = dif.loc[sort_by_list]
    else:
        dif = dif.sort_values(ascending=False)

    lower_bounds = df_stdev.min(axis=1)
    upper_bounds = df_stdev.max(axis=1)
    error_bars = upper_bounds - lower_bounds
    error_bars = error_bars.where(error_bars <= 500, np.nan)

    sns.set_style("ticks")
    plt.rcParams.update({"font.size": 20})
    p = ["#a8e6cf" if x > 0 else "#ff8b94" for x in dif.values]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(
        range(len(dif)),
        dif,
        color=p,
        yerr=error_bars.loc[dif.index],
        error_kw={"ecolor": "lightgray"},
    )
    if not labels:
        labels = pd.Series(index=dif.index, data=dif.index)
    plt.xticks(range(len(dif)), labels=labels.loc[dif.index], rotation=90)
    ax.set_xlabel("Signatures")
    ax.set_ylabel("Difference")
    ax.set_title(title)

    for i, value in enumerate(dif.values):
        p_value = df_p_values.loc[dif.index[i]][0]
        if p_value <= 0.05:
            if p_value <= 0.001:
                pv = "***"
            elif p_value <= 0.01:
                pv = "**"
            else:
                pv = "*"
        else:
            pv = "n/s"
        if value < 0:
            y = 10
        else:
            y = value + 10
        if pv != "n/s":
            ax.annotate(pv, (i, y), ha="center", va="center", size=17, c="black")
        else:
            ax.annotate(pv, (i, y), ha="center", va="bottom", size=13, c="black")
    plt.ylim(dif.min() - 100, dif.max() + 100)
    if title:
        plt.title(title)
    if save:
        plt.rcParams["svg.fonttype"] = "none"
        if "." + format in path_to_save.split("/")[-1]:
            path_to_save = path_to_save
        else:
            path_to_save = path_to_save + format
        plt.savefig(path_to_save, format=format, dpi=300)


def get_hclust_order(
    data, metric="correlation", method="average", optimal_ordering=True, **kwargs
):
    """
    Return an order of observations that corresponds to the order of leaf nodes in a dendrogram of hierarchical clustering
    performed by clustering_method algorithm based on pairwise distance_metric

    REQUIRED
    :param data: pd.DataFrame, index - observations, columns - features.
                 Alternatively, ndarray, an m by n array of m original observations in an n-dimensional space

    OPTIONAL
    :param distance_metric: str, distance metric to use. See scipy.spatial.distance.pdist documentation for options
    :param clustering_method: str, linkage algorithm to use. See scipy.cluster.hierarchy.linkage documentation for options
    :param optimal_ordering: bool, if True, the linkage matrix will be reordered so that the distance between successive leaves is minimal.
                             See scipy.cluster.hierarchy.linkage for the full description.

    :return: list, a list of labels corresponding to the leaf nodes
    """

    if metric in ["spearman", "kendall", "pearson"]:
        similarity_matrix = data.dropna().T.corr(method=metric)
        dissimilarity_matrix = 1 - similarity_matrix
        hclust_linkage = linkage(
            squareform(dissimilarity_matrix), method=method, optimal_ordering=True
        )

    else:
        dissimilarity_matrix = pdist(data.dropna(), metric=metric)
        hclust_linkage = linkage(
            dissimilarity_matrix, method=method, optimal_ordering=True
        )

    R = dendrogram(Z=hclust_linkage, labels=data.index, no_plot=True, **kwargs)

    leaf_order = R["ivl"]

    return leaf_order


def get_ranks(df: pd.DataFrame) -> Generator[pd.Series, None, None]:
    """
    Generator of ranks of genes in df

    Yields
    ------
    pd.Series
        rank of genes in df
    """
    for gene in df.index:
        yield df.loc[gene].rank()


def gene_corr_plot(genes: List[str], df: pd.DataFrame, diag: str) -> None:
    #     display(genes)
    """
    Plot correlation heatmap of genes, heatmap of rank sums and violin plot of expression ranks and mean expression.

    Parameters
    ----------
    genes : list
        List of genes to plot
    df : pd.DataFrame
        DataFrame with expression data, index - genes, columns - samples
    diag : str
        Diagnosis to plot
    """
    clustered_heatmap(df.reindex(genes), title=diag, metric="spearman", vmin=0, vmax=1)
    df_ranks = pd.concat(list(get_ranks(df)), axis=1).T
    df_ranks = df_ranks.reindex(genes)
    order = df_ranks.sum().sort_values(ascending=False).index
    df_ranks = df_ranks.reindex(order, axis=1)
    _, ax = plt.subplots(figsize=(15, 4))
    sns.heatmap(
        df_ranks.reindex(get_hclust_order(df_ranks, metric="spearman")),
        cbar=False,
        cmap="coolwarm",
        xticklabels=False,
        ax=ax,
    )
    df = df.reindex(genes)
    data = pd.melt(df.T)
    data["KDE"] = "Gene"

    data2 = pd.Series(
        data=list(
            itertools.chain.from_iterable(itertools.repeat(df.mean(), len(df.index)))
        ),
        index=list(
            itertools.chain.from_iterable(
                itertools.repeat(x, len(df.columns)) for x in df.index
            )
        ),
        name="value",
    )

    data2 = data2.reset_index().rename(columns={"index": "Gene"})

    data2["KDE"] = "Mean"

    _, ax = plt.subplots(figsize=(len(df.index) * 1.5, 5))

    ax2 = sns.violinplot(
        x="Gene",
        y="value",
        hue="KDE",
        data=pd.concat([data, data2]),
        palette="Set2",
        split=True,
        scale="count",
        inner="box",
        ax=ax,
    )
    ax2.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.show()


def genes_expression_chart_(
    genes: List[str],
    num_to_show: int = 30,
    axes: Optional[Tuple[matplotlib.axes.Axes, ...]] = None,
    figsize: Tuple[int, int] = (12, 8),
    source: Optional[Literal["lines", "cells", "tissues", "all"]] = None,
    path_to_cells: str = "/internal_data/Deconvolution/Data/Median_Cells_by_sample-kallisto-Xena-gene-TPM_without_noncoding",
    path_to_lines: str = "/internal_data/Deconvolution/Data/Median_Cancer_cell_lines_by_sample-kallisto-Xena-gene-TPM_without_noncoding",
    path_to_tissues: str = "/internal_data/Deconvolution/Data/Mean_Tissues_by_sample-kallisto-Xena-gene-TPM_without_noncoding.tsv",
) -> pd.DataFrame:
    """
    Function makes chart of genes expression from all sources
    :param genes: list of genes
    :param num_to_show: number of cells / lines with highest expression level to show (top-k)
    :param axes: axes to plot
    :param figsize: size of a figure
    :param source: 'lines', 'cells' or 'all'
    :param path_to_cells: path to cells expression data
    :param path_to_lines: path to lines expression data
    :param path_to_tissues: path to tissues expression data
    :returns: tables with genes expression
    """

    if not (
        source == "lines"
        or source == "cells"
        or source == "tissues"
        or source == "all"
        or source == None
    ):
        raise ValueError(
            "Incorrect source type! Only cells, lines, tissues or all are available."
        )

    cells = pd.read_csv(path_to_cells, sep=",", index_col=0)
    lines = pd.read_csv(path_to_lines, sep=",", index_col=0)
    tissues = pd.read_csv(path_to_tissues, sep="\t", index_col=0)

    for common_name in set(lines.columns).intersection(set(cells.columns)):
        lines.rename(columns={common_name: common_name + "_line"}, inplace=True)
        cells.rename(columns={common_name: common_name + "_sample"}, inplace=True)
    for common_name in set(cells.columns).intersection(set(tissues.columns)):
        cells.rename(columns={common_name: common_name + "_sample"}, inplace=True)
        tissues.rename(columns={common_name: common_name + "_tissue"}, inplace=True)
    for common_name in set(lines.columns).intersection(set(tissues.columns)):
        lines.rename(columns={common_name: common_name + "_line"}, inplace=True)
        tissues.rename(columns={common_name: common_name + "_tissue"}, inplace=True)

    for gene in genes:
        if gene not in cells.index:
            print("Gene {} is not found".format(gene))
            genes.remove(gene)

    if axes is not None:
        assert len(genes) == len(
            axes.flat
        ), "Length of genes list should be equal to the number of subplots!"
    else:
        fig, axes = plt.subplots(
            len(genes), 1, figsize=(figsize[0], figsize[1] * len(genes))
        )

    for ax, gene in zip(axes.flat, genes):
        if source == "lines":
            exp = lines.loc[gene].sort_values(ascending=False)[:num_to_show]
        elif source == "cells":
            exp = cells.loc[gene].sort_values(ascending=False)[:num_to_show]
        elif source == "tissues":
            exp = tissues.loc[gene].sort_values(ascending=False)[:num_to_show]
        else:
            exp = pd.concat([cells.loc[gene], lines.loc[gene]]).sort_values(
                ascending=False
            )[:num_to_show]
        palette = lin_colors(exp.index)
        ax.set_title(gene, size=20)
        ax.set_xticklabels(exp.index, rotation="vertical", fontsize=12)
        exp.plot(kind="bar", ax=ax, color=list(exp.index.map(palette)), grid=False)
    plt.tight_layout()
    plt.show()

    return axes, exp


def resize_clustermap_annotation(
    g: sns.matrix.ClusterGrid,
    col_colors_h_ratio: float = 1,
    row_colors_w_ratio: float = 1,
) -> sns.matrix.ClusterGrid:
    """
    Function changes size of clustermap's row/col_colors annotation
    g: seaborn.matrix.ClusterGrid object. It is returned by sns.clustermap() function
    col_colors_h_ratio: parameters of new clustermaps annotation size
    row_colors_w_ratio: parameters of new clustermaps annotation size
    returns:
    seaborn.matrix.ClusterGrid object (clustermap plot) with resized annotations
    """

    if g.ax_col_colors is not None:
        col = g.ax_col_colors.get_position()
        g.ax_col_colors.set_position(
            [col.x0, col.y0, col.width, col.height * col_colors_h_ratio]
        )

        col_d = g.ax_col_dendrogram.get_position()
        g.ax_col_dendrogram.set_position(
            [
                col_d.x0,
                col_d.y0 + col.height * (col_colors_h_ratio - 1),
                col_d.width,
                col_d.height,
            ]
        )

    if g.ax_row_colors is not None:
        row = g.ax_row_colors.get_position()
        g.ax_row_colors.set_position(
            [
                row.x0 - row.width * (row_colors_w_ratio - 1),
                row.y0,
                row.width * row_colors_w_ratio,
                row.height,
            ]
        )

        row_d = g.ax_row_dendrogram.get_position()
        g.ax_row_dendrogram.set_position(
            [
                row_d.x0 - row.width * (row_colors_w_ratio - 1),
                row_d.y0,
                row_d.width,
                row_d.height,
            ]
        )
    return g


def clustered_heatmap(
    ds: pd.DataFrame,
    title: str = "",
    corr: str = "spearman",
    method: str = "average",
    yl: bool = True,
    xl: bool = True,
    cmap: str = "coolwarm",
    figsize: Tuple[float, float] = (15, 15),
    annot: bool = True,
    vmin: float = 0.5,
    vmax: float = 1,
    col_colors_h_ratio: float = 1,
    row_colors_w_ratio: float = 1,
    **kwargs: Dict[str, Any],
) -> None:
    """
    Plot correlation heatmap for rows after performing clustering on dataset "ds"
    :param ds: pd.DataFrame, numerical data only

    """

    similarity_matrix = ds.dropna().T.corr(method=corr)
    dissimilarity_matrix = 1 - similarity_matrix
    hclust_linkage = linkage(squareform(dissimilarity_matrix), method=method)

    g = sns.clustermap(
        similarity_matrix.round(3),
        method=method,
        row_linkage=hclust_linkage,
        col_linkage=hclust_linkage,
        annot=annot,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        yticklabels=yl,
        xticklabels=xl,
        figsize=figsize,
        **kwargs,
    )

    g.fig.suptitle(title)

    return resize_clustermap_annotation(g, col_colors_h_ratio, row_colors_w_ratio)


def rgb_to_hex(rgb_tuple: Tuple[float, float, float]) -> str:
    """
    Convert an RGB tuple to a hex color string.

    Parameters
    ----------
    rgb_tuple : tuple of 3 floats between 0 and 1
        RGB values to convert

    Returns
    -------
    hex_color : str
        Hex color string in the format #RRGGBB
    """
    r, g, b = [round(x * 255) for x in rgb_tuple]
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


def patch_plot(
    patches: pd.Series,
    ax: Optional[matplotlib.axes.Axes] = None,
    order: Union[str, List[str]] = "sort",
    w: float = 0.25,
    h: float = 0,
    vertical: bool = False,
    legend_right: bool = True,
    show_ticks: bool = False,
) -> Tuple[matplotlib.axes.Axes, Optional[List[matplotlib.patches.Rectangle]]]:
    """
    Plots given palette (dict key:color) as a pretty legend
    :param patches: Series with keys - labels, and values - colors
    :param ax: ax to plot
    :param order: list with order of labels
    :param w: int, width
    :param h: int, 0 - auto determine
    :param vertical: bool, whether to make vertical plot instead of horizontal
    :param legend_right: bool, whether to plot legend on the right
    :param show_ticks: book, whether to show ticks on plot
    :return: Axes with plot, and list of rectangles (if legend_right)
    """

    cur_patches = pd.Series(patches)

    if order == "sort":
        order = list(np.sort(cur_patches.index))

    if vertical:
        data = pd.Series([1] * len(order), index=order)
        if ax is None:
            if h == 0:
                h = 0.3 * len(patches)
            _, ax = plt.subplots(figsize=(h, w))
        data.plot(
            kind="bar", color=[cur_patches[x] for x in data.index], width=1, ax=ax
        )
        ax.set_yticks([])
    else:
        data = pd.Series([1] * len(order), index=order[::-1])
        if ax is None:
            if h == 0:
                h = 0.3 * len(patches)
            _, ax = plt.subplots(figsize=(w, h))

        data.plot(
            kind="barh", color=[cur_patches[x] for x in data.index], width=1, ax=ax
        )
        ax.set_xticks([])
        if legend_right:
            ax.yaxis.tick_right()

    sns.despine(offset={"left": -2}, ax=ax)

    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if not show_ticks:
        ax.tick_params(length=0)

    return ax


def lin_colors(
    factors_vector: pd.Series,
    cmap: Union[str, LinearSegmentedColormap] = "default",
    sort: bool = True,
    min_v: float = 0,
    max_v: float = 1,
    linspace: bool = True,
    lighten_color: Optional[float] = None,
) -> Dict[str, str]:
    """
    Return dictionary of unique features of "factors_vector" as keys and color hexes as entries
    :param factors_vector: pd.Series
    :param cmap: matplotlib.colors.LinearSegmentedColormap, which colormap to base the returned dictionary on
        default - matplotlib.cmap.hsv with min_v=0, max_v=.8, lighten_color=.9
    :param sort: bool, whether to sort the unique features
    :param min_v: float, for continuous palette - minimum number to choose colors from
    :param max_v: float, for continuous palette - maximum number to choose colors from
    :param linspace: bool, whether to spread the colors from "min_v" to "max_v"
        linspace=False can be used only in discrete cmaps
    :param lighten_color: float, from 0 to +inf: 0 - very dark (just black), 1 - original color, >1 - brighter color
    :return: dict
    """

    unique_factors = factors_vector.dropna().unique()
    if sort:
        unique_factors = np.sort(unique_factors)

    if cmap == "default":
        cmap = matplotlib.cm.rainbow
        max_v = 0.92

    if linspace:
        cmap_colors = cmap(np.linspace(min_v, max_v, len(unique_factors)))
    else:
        cmap_colors = np.array(cmap.colors[: len(unique_factors)])

    if lighten_color is not None:
        cmap_colors = [x * lighten_color for x in cmap_colors]
        cmap_colors = np.array(cmap_colors).clip(0, 1)

    return dict(
        list(zip(unique_factors, [matplotlib.colors.to_hex(x) for x in cmap_colors]))
    )


def define_ax_figsize(ax) -> Tuple[float, float]:
    """
    Function calculates figsize for given ax.
    Calculations are quite tricky when ax come from figure with multiple subplots

    :param ax: amatplotlib axis
    :return: (float, float), calculated figure size
    """
    full_fig_size = list(ax.figure.get_size_inches())
    subplots_num = ax.get_subplotspec().get_geometry()[:2]
    current_ax_num = ax.get_subplotspec().get_geometry()[2]

    current_ax_x = current_ax_num % subplots_num[1]
    current_ax_y = current_ax_num // subplots_num[1]

    height_ratios = ax.get_subplotspec().get_gridspec().get_height_ratios()
    width_ratios = ax.get_subplotspec().get_gridspec().get_width_ratios()

    if height_ratios is None:
        # in this case all ratios are equal
        height_ratios = [1] * subplots_num[0]

    if width_ratios is None:
        # in this case all ratios are equal
        width_ratios = [1] * subplots_num[1]

    return (
        width_ratios[current_ax_x] / float(sum(width_ratios)) * full_fig_size[0],
        height_ratios[current_ax_y] / float(sum(height_ratios)) * full_fig_size[1],
    )


def boxplot_with_pvalue(
    data: pd.Series,
    grouping: pd.Series,
    title: str = "",
    ax: Optional[matplotlib.axes.Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    swarm: bool = True,
    p_digits: int = 3,
    stars: bool = True,
    violin: bool = False,
    palette: Optional[Dict[str, str]] = None,
    order: Optional[List[str]] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    s: float = 7,
    p_fontsize: float = 16,
    xlabel: Optional[str] = None,
    n_samples: bool = True,
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Plots boxplot or violin plot with pairwise comparisons
    :param data: pd.Series, series with numerical data
    :param grouping: pd.Series, series with categorical data
    :param title: str, plot title
    :param ax: matplotlib axis, axis to plot on
    :param figsize: (float, float), figure size in inches
    :param swarm: bool, whether to plot a swarm in addition to boxes
    :param p_digits: int, number of digits to round p value to
    :param stars: bool, whether to plot star notation instead of number for p value
    :param violin: bool, whether to do a violin plot
    :param palette: dict, palette for plotting. Keys are unique values from groups, entries are color hexes
    :param order: list, order to plot the entries in. Contains ordered unique values from "grouping"
    :param y_min: float, vertical axis minimum
    :param y_max:float, vertical axis maximum
    :param s: float, size of dots in swarmplot
    :param p_fontsize: float, font size for p value labels
    :param n_samples: bool, whether to annotate each group label with the number of samples
    :param kwargs:
    :return: matplotlib axis
    """

    if data.index.duplicated().any() | grouping.index.duplicated().any():
        raise Exception("Indexes contain duplicates")

    if not data.apply(lambda x: isinstance(x, (int, float)) or pd.isna(x)).all():
        warnings.warn("Data contains non-numeric values", UserWarning)

    cdata, cgrouping = to_common_samples([data.dropna(), grouping.dropna()])

    if len(cgrouping.dropna().unique()) < 2:
        raise Exception(
            "Less from 2 classes provided: {}".format(len(cgrouping.unique()))
        )

    if order is None:
        order = cgrouping.dropna().unique()

    if ax is None:
        if figsize is None:
            figsize = (1.2 * len(order), 4)
        _, ax = plt.subplots(figsize=figsize)

    if not violin:
        sns.boxplot(
            y=cdata,
            x=cgrouping,
            ax=ax,
            palette=palette,
            order=order,
            fliersize=0,
            **kwargs,
        )
    else:
        sns.violinplot(
            y=cdata, x=cgrouping, ax=ax, palette=palette, order=order, **kwargs
        )

        # Ignoring swarm setting since violin performs same function
        swarm = False

    if swarm:
        sns.swarmplot(y=cdata, x=cgrouping, ax=ax, color=".25", order=order, s=s)

    pvalues = []
    for g1, g2 in zip(order[:-1], order[1:]):
        samples_g1 = cgrouping[cgrouping == g1].index
        samples_g2 = cgrouping[cgrouping == g2].index
        try:
            if len(samples_g1) and len(samples_g2):
                pv = mannwhitneyu(
                    cdata.loc[samples_g1],
                    cdata.loc[samples_g2],
                    alternative="two-sided",
                ).pvalue
            else:
                pv = 1
        except ValueError:
            pv = 1
        pvalues.append(pv)

    y_max = y_max or max(cdata)
    y_min = y_min or min(cdata)
    effective_size = y_max - y_min
    plot_y_limits = (y_min - effective_size * 0.15, y_max + effective_size * 0.2)

    if p_digits > 0:

        pvalue_line_y_1 = y_max + effective_size * 0.05
        if figsize is None:
            figsize = define_ax_figsize(ax)
        pvalue_text_y_1 = pvalue_line_y_1 + 0.25 * effective_size / figsize[1]

        for pos, pv in enumerate(pvalues):
            pvalue_str = get_pvalue_string(pv, p_digits, stars=stars)
            pvalue_text_y_1_local = pvalue_text_y_1

            if pvalue_str == "-":
                pvalue_text_y_1_local += 0.1 * effective_size / figsize[1]

            bar_fraction = str(0.25 / 2.0 / (figsize[0] / float(len(order))))

            ax.annotate(
                "",
                xy=(pos + 0.1, pvalue_line_y_1),
                xycoords="data",
                xytext=(pos + 0.9, pvalue_line_y_1),
                textcoords="data",
                arrowprops=dict(
                    arrowstyle="-",
                    ec="#000000",
                    connectionstyle="bar,fraction={}".format(bar_fraction),
                ),
            )
            ax.text(
                pos + 0.5,
                pvalue_text_y_1_local,
                pvalue_str,
                fontsize=p_fontsize,
                horizontalalignment="center",
                verticalalignment="center",
            )

    ax.set_title(title)
    ax.set_ylim(plot_y_limits)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if n_samples:
        new_labels = [
            f"{label}\nn = {len(cgrouping.loc[cgrouping == label])}"
            for label in [elem.get_text() for elem in ax.get_xticklabels()]
        ]
        ax.set_xticklabels(new_labels)

    return ax


def calculate_and_plot_correlations(
    series1: pd.Series,
    series2: pd.Series,
    name1: str = "Predicted",
    name2: str = "True",
    verbose: bool = True,
    ret: bool = False,
    plot: bool = True,
    title: Optional[str] = False,
) -> Optional[Dict[str, float]]:
    """
    Calculate and plot correlations between two series.

    Parameters
    ----------
    series1 : pd.Series
        Predicted values
    series2 : pd.Series
        True values
    name1 : str
        Name of the first series
    name2 : str
        Name of the second series
    verbose : bool
        Print the metrics to the console
    ret : bool
        Return the metrics as a dictionary
    plot : bool
        Plot the data
    title : str
        Title of the plot

    Returns
    -------
    metrics : dict
        Dictionary of metrics if ret is True
    """

    series1, series2 = to_common_samples((series1.dropna(), series2.dropna()))
    df = pd.DataFrame({name1: series1, name2: series2})
    pearson_corr, pearson_p = pearsonr(series1, series2)
    spearman_corr, spearman_p = spearmanr(series1, series2)
    mse = mean_squared_error(series1, series2)
    mae = mean_absolute_error(series1, series2)
    try:
        slope, intercept, r_value, p_value, std_err = linregress(series1, series2)
        concordance_corr_coef = (2 * r_value * np.std(series1) * np.std(series2)) / (
            np.var(series1)
            + np.var(series2)
            + (np.mean(series1) - np.mean(series2)) ** 2
        )
    except:
        concordance_corr_coef = 0
    text = [
        f"Pearson correlation: {pearson_corr:.4f}, p-value: {pearson_p:.1e}",
        f"Spearman correlation: {spearman_corr:.4f}, p-value: {spearman_p:.1e}",
        f"MSE: {mse:.4f}; MAE: {mae:.4f}",
        f"CCC: {concordance_corr_coef:.4f}",
        f"Number of samples: {df.dropna().shape[0]}",
    ]
    if verbose:
        print("\n".join(text))
    if plot:
        g = sns.JointGrid(data=df, x=name1, y=name2, space=0)
        g.plot_joint(sns.scatterplot, alpha=0.6, color="teal")
        sns.regplot(x=name1, y=name2, data=df, ax=g.ax_joint, scatter=False, color="r")
        g.plot_marginals(sns.histplot, kde=True, color="teal")
        if title:
            text += [title]
        textstr = "\n".join(text)
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        g.ax_joint.text(
            0.05,
            0.95,
            textstr,
            transform=g.ax_joint.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
            linespacing=1.5,
        )
        plt.show()
        plt.close()
    if ret:
        metrics = {
            "MSE": mse,
            "MAE": mae,
            "Spearman": spearman_corr,
            "CCC": concordance_corr_coef,
            "Pearson": pearson_corr,
        }
        return metrics


def plot_scatter_with_ci(
    data: Dict[str, Dict[str, float]],
    metric_x: str = "goi_cv",
    metric_y: str = "F1",
    title: Optional[str] = None,
    random_score: float = 0.5,
    path: Optional[Path] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
):
    """
    Plot scatter plot with CI for aggregated data.

    Parameters
    ----------
    data : dict
        dictionary where keys are fges names and values are dictionaries with
        metrics values.
    metric_x : str, optional
        metric to use for x axis, by default "goi_cv"
    metric_y : str, optional
        metric to use for y axis, by default "F1"
    title : str, optional
        title of the plot, by default None
    random_score : float, optional
        random score line, by default 0.5
    path : Path, optional
        path to save the plot, by default None
    xlabel : str, optional
        x axis label, by default metric_x
    ylabel : str, optional
        y axis label, by default metric_y

    """
    means_x = []
    means_y = []
    stds_x = []
    stds_y = []
    labels = []

    label_counts = {}
    for key in data.keys():
        fges_type = detect_fges_source(key)
        label_counts[fges_type] = label_counts.get(fges_type, 0) + 1

    for key, values in data.items():
        x_values = [v[metric_x] for v in values.values()]
        y_values = [v[metric_y] for v in values.values()]

        mean_x = np.mean(x_values)
        mean_y = np.mean(y_values)
        std_x = np.std(x_values)
        std_y = np.std(y_values)

        means_x.append(mean_x)
        means_y.append(mean_y)
        stds_x.append(std_x)
        stds_y.append(std_y)
        labels.append(key)

    zipped = zip(means_x, means_y, stds_x, stds_y, labels)
    sort_order = [
        "Internal",
        "Nirmal",
        "Bindea",
        "xCell",
        "KEGG",
        "Gene_Ontology",
        "Human_Phenotype_Ontology",
        "BioCarta",
        "Reactome",
        "WikiPathways",
        "Pathway_Interaction_Database",
        "MSigDb_Single_Cell",
        "MSigDb_Other",
        "MSigDb_Dif_Expression",
        "Random_FGES",
    ]
    order_dict = {key: index for index, key in enumerate(sort_order)}
    zipped_sorted = sorted(
        zipped, key=lambda x: label_counts[detect_fges_source(x[4])], reverse=True
    )

    fig, ax = plt.subplots(figsize=(6, 3.5))
    label_to_handle = {}
    seen_labels = set()

    for mean_x, mean_y, std_x, std_y, label in zipped_sorted:
        fges_type = detect_fges_source(label)
        color = signature_palette[fges_type]
        if fges_type not in seen_labels:
            handle = ax.errorbar(
                mean_x,
                mean_y,
                yerr=std_y,
                fmt="o",
                label=fges_type,
                c=color,
                ecolor="#B6B6B4",
                capsize=2,
                markersize=4,
                elinewidth=1,
            )
            label_to_handle[fges_type] = handle
            seen_labels.add(fges_type)
        else:
            ax.errorbar(
                mean_x,
                mean_y,
                yerr=std_y,
                fmt="o",
                label="_nolegend_",
                c=color,
                ecolor="#B6B6B4",
                capsize=2,
                markersize=4,
                elinewidth=1,
            )
    if xlabel:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel(metric_x)
    if ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel(metric_y)

    y_coord = random_score
    x_coord = min(means_x) + 0.001
    plt.axhline(y=y_coord, color="crimson", linestyle="--", alpha=0.5, linewidth=1)
    plt.text(
        x_coord,
        y_coord + 0.003,
        "Random\nPrediction",
        color="crimson",
        fontsize=6.5,
        ha="left",
    )

    random_means_y = [
        mean_y
        for _, mean_y, _, _, label in zipped_sorted
        if detect_fges_source(label) == "Random_FGES"
    ]
    y_coord = max(random_means_y)
    x_coord = min(means_x) + 0.001
    plt.axhline(y=y_coord, color="crimson", linestyle="--", alpha=0.5, linewidth=1)
    plt.text(
        x_coord,
        y_coord + 0.003,
        "Random\nFGES",
        color="crimson",
        fontsize=6.5,
        ha="left",
    )
    handles, labels = [], []
    for label in sort_order:
        if label in label_to_handle:
            handles.append(label_to_handle[label])
            labels.append(label)
    ax.legend(
        handles,
        labels,
        bbox_to_anchor=(1.05, 1.0),
        loc="upper left",
        bbox_transform=ax.transAxes,
    )

    # plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', bbox_transform=ax.transAxes)
    plt.tight_layout(pad=0.5)
    if title:
        plt.title(title)
    name = title + ".svg"
    plt.savefig(path / name, format="svg")
    plt.show()
    plt.close()


def plot_scatter_with_ci_agg(
    data: Dict[str, Dict[str, float]],
    metric_x: str = "goi_cv",
    metric_y: str = "F1",
    title: Optional[str] = None,
    random_score: float = 0.5,
    path: Optional[Path] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
):
    """
    Plot scatter plot with CI for aggregated data.

    Parameters
    ----------
    data : dict
        dictionary where keys are fges names and values are dictionaries with
        metrics values.
    metric_x : str, optional
        metric to use for x axis, by default "goi_cv"
    metric_y : str, optional
        metric to use for y axis, by default "F1"
    title : str, optional
        title of the plot, by default None
    random_score : float, optional
        random score line, by default 0.5
    path : Path, optional
        path to save the plot, by default None
    xlabel : str, optional
        x axis label, by default metric_x
    ylabel : str, optional
        y axis label, by default metric_y

    """
    sort_order = [
        "Internal",
        "Nirmal",
        "Bindea",
        "xCell",
        "Gene_Ontology",
        "Human_Phenotype_Ontology",
        "WikiPathways",
        "MSigDb_Single_Cell",
        "MSigDb_Other",
        "MSigDb_Dif_Expression",
        "Random_FGES",
    ]
    aggregated_data = {key: [] for key in sort_order}

    for key, values in data.items():
        fges_type = detect_fges_source(key)

        x_values = [v[metric_x] for v in values.values()]
        y_values = [v[metric_y] for v in values.values()]
        if fges_type in sort_order:
            aggregated_data[fges_type].extend(list(zip(x_values, y_values)))

    means_x = []
    means_y = []
    stds_x = []
    stds_y = []
    labels = []

    for key, values in aggregated_data.items():
        x_values, y_values = zip(*values)

        mean_x = np.mean(x_values)
        mean_y = np.mean(y_values)
        std_x = np.std(x_values)
        std_y = np.std(y_values)

        means_x.append(mean_x)
        means_y.append(mean_y)
        stds_x.append(std_x)
        stds_y.append(std_y)
        labels.append(key)

    zipped_sorted = sorted(
        zip(means_x, means_y, stds_x, stds_y, labels),
        key=lambda x: len(aggregated_data[x[4]]),
        reverse=True,
    )

    fig, ax = plt.subplots(figsize=(6, 3.5))
    label_to_handle = {}
    for mean_x, mean_y, std_x, std_y, label in zipped_sorted:
        color = signature_palette[label]
        handle = ax.errorbar(
            mean_x,
            mean_y,
            xerr=std_x,
            yerr=std_y,
            fmt="o",
            label=label,
            c=color,
            ecolor="#B6B6B4",
            capsize=3,
            markersize=6,
            elinewidth=1,
            capthick=1,
        )
        label_to_handle[label] = handle

    if xlabel:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel(metric_x)
    if ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel(metric_y)

    y_coord = 0.5
    x_coord = min(means_x) - 0.05
    plt.axhline(y=y_coord, color="crimson", linestyle="--", alpha=0.5)
    plt.text(
        x_coord,
        y_coord + 0.005,
        "Random Prediction",
        color="crimson",
        fontsize=6.5,
        ha="left",
    )
    handles, labels = [], []
    for label in sort_order:
        if label in label_to_handle:
            handles.append(label_to_handle[label])
            labels.append(label)
    ax.legend(
        handles,
        labels,
        bbox_to_anchor=(1.05, 1.0),
        loc="upper left",
        bbox_transform=ax.transAxes,
    )
    plt.tight_layout(pad=0.5)
    if title:
        plt.title(title)
    name = title + ".svg"
    plt.savefig(path / name, format="svg")
    plt.show()
    plt.close()


def grouped_boxplot_with_pvalue(
    data: pd.Series,
    grouping: pd.Series,
    title: str = "",
    ax: Optional[matplotlib.axes.Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    swarm: bool = True,
    p_digits: int = 3,
    stars: bool = True,
    violin: bool = False,
    hue: Optional[pd.Series] = None,
    hue_order: Optional[List[Any]] = None,
    palette: Optional[Dict[str, str]] = None,
    order: Optional[List[str]] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    s: float = 7,
    p_fontsize: float = 16,
    xlabel: Optional[str] = None,
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Plots boxplot or violin plot with pairwise comparisons
    :param data: pd.Series, series with numerical data
    :param grouping: pd.Series, series with categorical data
    :param title: str, plot title
    :param ax: matplotlib axis, axis to plot on
    :param figsize: (float, float), figure size in inches
    :param swarm: bool, whether to plot a swarm in addition to boxes
    :param p_digits: int, number of digits to round p value to
    :param stars: bool, whether to plot star notation instead of number for p value
    :param violin: bool, whether to do a violin plot
    :param palette: dict, palette for plotting. Keys are unique values from groups, entries are color hexes
    :param order: list, order to plot the entries in. Contains ordered unique values from "grouping"
    :param y_min: float, vertical axis minimum
    :param y_max:float, vertical axis maximum
    :param s: float, size of dots in swarmplot
    :param p_fontsize: float, font size for p value labels
    :param kwargs:
    :return: matplotlib axis
    """

    from scipy.stats import mannwhitneyu

    if data.index.duplicated().any() | grouping.index.duplicated().any():
        raise Exception("Indexes contain duplicates")

    cdata, cgrouping = to_common_samples([data.dropna(), grouping.dropna()])

    if len(cgrouping.dropna().unique()) < 2:
        raise Exception(
            "Less from 2 classes provided: {}".format(len(cgrouping.unique()))
        )

    if order is None:
        order = cgrouping.dropna().unique()

    if hue is not None and hue_order is None:
        hue_order = []
        for i in hue.dropna().unique():
            hue_order += [i]

    if hue is not None and palette is None:
        palette = dict()
        x = 0
        for h in hue_order:
            palette[h] = sns.color_palette()[x]
            x += 1

    if ax is None:
        if figsize is None:
            figsize = (1.2 * len(order), 4)
        _, ax = plt.subplots(figsize=figsize)

    if hue is None:
        dodge = False
    else:
        dodge = True

    if not violin:
        a = sns.boxplot(
            y=cdata,
            x=cgrouping,
            ax=ax,
            palette=palette,
            order=order,
            fliersize=0,
            **kwargs,
            hue=hue,
            hue_order=hue_order,
            dodge=dodge,
        )
    else:
        sns.violinplot(
            y=cdata, x=cgrouping, ax=ax, palette=palette, order=order, **kwargs
        )

        # Ignoring swarm setting since violin performs same function
        swarm = False

    if swarm:
        sns.swarmplot(
            y=cdata,
            x=cgrouping,
            ax=ax,
            color=".25",
            order=order,
            s=s,
            hue=hue,
            hue_order=hue_order,
            dodge=dodge,
        )

    pvalues = []

    if hue is None:
        for g1, g2 in zip(order[:-1], order[1:]):
            samples_g1 = cgrouping[cgrouping == g1].index
            samples_g2 = cgrouping[cgrouping == g2].index
            try:
                if len(samples_g1) and len(samples_g2):
                    pv = mannwhitneyu(
                        cdata.loc[samples_g1],
                        cdata.loc[samples_g2],
                        alternative="two-sided",
                    ).pvalue
                else:
                    pv = 1
            except ValueError:
                pv = 1
            pvalues.append(pv)
    else:
        for x in order:
            for g1, g2 in zip(hue_order[:-1], hue_order[1:]):
                samples_g1 = cgrouping[grouping == x][hue == g1].index
                samples_g2 = cgrouping[grouping == x][hue == g2].index
                try:
                    if len(samples_g1) and len(samples_g2):
                        pv = mannwhitneyu(
                            cdata.loc[samples_g1],
                            cdata.loc[samples_g2],
                            alternative="two-sided",
                        ).pvalue
                    else:
                        pv = 1
                except ValueError:
                    pv = 1
                pvalues.append(pv)

    y_max = y_max or max(cdata)
    y_min = y_min or min(cdata)
    effective_size = y_max - y_min
    plot_y_limits = (y_min - effective_size * 0.15, y_max + effective_size * 0.2)

    if p_digits > 0:

        pvalue_line_y_1 = y_max + effective_size * 0.05
        if figsize is None:
            figsize = define_ax_figsize(ax)
        pvalue_text_y_1 = pvalue_line_y_1 + 0.25 * effective_size / figsize[1]

        for pos, pv in enumerate(pvalues):
            pvalue_str = get_pvalue_string(pv, p_digits, stars=stars)
            pvalue_text_y_1_local = pvalue_text_y_1

            if pvalue_str == "-":
                pvalue_text_y_1_local += 0.1 * effective_size / figsize[1]

            bar_fraction = str(0.25 / 2.0 / (figsize[0] / float(len(order))))

            if hue is None:
                ax.annotate(
                    "",
                    xy=(pos + 0.1, pvalue_line_y_1),
                    xycoords="data",
                    xytext=(pos + 0.9, pvalue_line_y_1),
                    textcoords="data",
                    arrowprops=dict(
                        arrowstyle="-",
                        ec="#000000",
                        connectionstyle="bar,fraction={}".format(bar_fraction),
                    ),
                )
                ax.text(
                    pos + 0.5,
                    pvalue_text_y_1_local,
                    pvalue_str,
                    fontsize=p_fontsize,
                    horizontalalignment="center",
                    verticalalignment="center",
                )
            else:
                ax.annotate(
                    "",
                    xy=(pos - 0.25, pvalue_line_y_1),
                    xycoords="data",
                    xytext=(pos + 0.25, pvalue_line_y_1),
                    textcoords="data",
                    arrowprops=dict(
                        arrowstyle="-",
                        ec="#000000",
                        connectionstyle="bar,fraction={}".format(bar_fraction),
                    ),
                )
                ax.text(
                    pos + 0.0,
                    pvalue_text_y_1_local,
                    pvalue_str,
                    fontsize=p_fontsize,
                    horizontalalignment="center",
                    verticalalignment="center",
                )

    ax.set_title(title)
    ax.set_ylim(plot_y_limits)

    if hue is not None:
        handl = []
        for x in hue_order:
            handl += [mpatches.Patch(color=palette[x])]
        ax.legend(
            handles=handl, labels=hue_order, bbox_to_anchor=(1, 1), loc="upper left"
        )

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    return ax
