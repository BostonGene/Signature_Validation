import warnings
from math import floor, log10
from pathlib import Path
from typing import Iterable, List, Literal, Optional, Tuple, Union

import gspread
import numpy as np
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu
from sklearn.preprocessing import MinMaxScaler
from statsmodels.robust.scale import mad
from statsmodels.stats.multitest import multipletests

RANDOM_SEED = 42


def sort_by_terms_order(
    data: pd.Series, t_order: list, vector: Optional[pd.Series] = None
) -> np.ndarray:
    """
    Sort "data" into blocks with values from "t_order". If "vector" is provided, sort each block by corresponding
    values in "vector"
    :param data: pd.Series
    :param t_order: list, values for blocks to sort "data" into
    :param vector: pd.Series, same index as data, which values to sort each block by
    :return: np.array, 1 dimensional
    """

    x = []
    for term in t_order:
        indices = data[data == term].index

        if len(indices):
            if vector is not None:
                x.append(vector.reindex(indices).dropna().sort_values().index)
            else:
                x.append(indices)

    return np.concatenate(x)


def to_quantiles(data: pd.Series, qs: List[float] = [0.5]) -> pd.Series:
    """
    Annotates samples of a numeric series by its quantile. Change amount of groups and thresholds by modifying qs arg
    :param data: pd.Series with numeric-like data
    :param qs: quantiles marks to assosiate sumples
    :return:
    """
    ann = []
    tr = 0

    for tr2 in list(np.sort(qs)) + [1]:
        if tr == 0:
            x = list(
                data[
                    np.logical_and(
                        data >= data.quantile(tr), data <= data.quantile(tr2)
                    )
                ].index
            )
        else:
            x = list(
                data[
                    np.logical_and(data > data.quantile(tr), data <= data.quantile(tr2))
                ].index
            )

        ann.append(pd.Series(["{}q<x<{}q".format(tr, tr2)] * len(x), index=x))
        tr = tr2
    return pd.concat(ann)


def process_generic_dataset(
    df_scores: pd.DataFrame, df_metadata: pd.DataFrame, group_col: str, score_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Processes the generic dataset to compute the mean, standard deviation, and perform the Mann-Whitney U test for each column.

    Parameters:
        - df_scores (pandas.DataFrame): The dataframe containing the scores.
        - df_metadata (pandas.DataFrame): The dataframe containing metadata information.
        - group_col (str): The column name in df_metadata representing the groups.
        - score_col (str): The column name in df_scores representing the scores.

    Returns:
        - out (dict): A nested dictionary with the mean values for each column and group.
        - out_stdev (dict): A nested dictionary with the standard deviation values for each column and group.
        - out_mw (pandas.DataFrame): A dataframe with the p-values resulting from the Mann-Whitney U test.
    """
    unique_groups = df_metadata[group_col].unique()
    num_groups = len(unique_groups)

    out = {group: {} for group in unique_groups}
    out_stdev = {group: {} for group in unique_groups}
    out_mw = {group: {} for group in unique_groups}

    for group in unique_groups:
        group_data = (
            df_scores[score_col]
            .reindex(df_metadata[df_metadata[group_col] == group].index)
            .dropna()
        )

        out[group][score_col] = group_data.mean()
        out_stdev[group][score_col] = group_data.std()

        for other_group in unique_groups:
            if other_group != group:
                other_group_data = (
                    df_scores[score_col]
                    .reindex(df_metadata[df_metadata[group_col] == other_group].index)
                    .dropna()
                )
                _, p_value = mannwhitneyu(
                    group_data, other_group_data, alternative="two-sided"
                )
                out_mw[group][other_group] = p_value

    out_mw = pd.DataFrame(out_mw)

    return out, out_stdev, out_mw


def get_expression_table(cohort, log=True):
    """
    Reads expression table from S3 and returns it as a pandas DataFrame.

    Parameters
    ----------
    cohort : str
        Name of the cohort.
    log : bool
        Whether to take the logarithm of the expression values. Default to True.

    Returns
    -------
    expression_table : pandas.DataFrame
        A DataFrame with samples as columns and genes as rows.
    """
    if log:
        return np.log2(
            read_dataset(f"/internal_data/mvp-data/cohorts/{cohort}/expressions.tsv")
            + 1
        ).T
    else:
        return read_dataset(
            f"/internal_data/mvp-data/cohorts/{cohort}/expressions.tsv"
        ).T


def get_anno(cohort):
    """
    Reads annotation data from S3 and returns it as a pandas DataFrame.

    Parameters
    ----------
    cohort : str
        Name of the cohort.

    Returns
    -------
    annotation : pandas.DataFrame
        A DataFrame with samples as columns and annotation features as rows.
    """
    return read_dataset(f"/internal_data/mvp-data/cohorts/{cohort}/annotation.tsv").T


def get_deconv(cohort):
    """
    Reads deconvolution data from S3 and returns it as a pandas DataFrame.

    Parameters
    ----------
    cohort : str
        Name of the cohort.

    Returns
    -------
    deconvolution : pandas.DataFrame
        A DataFrame with samples as columns and cell types as rows.
    """
    return read_dataset(
        f"/internal_data/mvp-data/cohorts/{cohort}/deconvolution/cells_deconvolution_rna_percent.tsv"
    ).T


def prepare_sample_annot(annot: pd.DataFrame) -> pd.DataFrame:
    """
    Function for handling filtering and aggregation of sample annotation data.
    :param annot: A pandas DataFrame with SRRs as rows and "Sample", "Readcounts", and "QC_score" columns.
    :return: an aggregated pandas DataFrame with samples as rows
    """
    annot = annot.loc[:, ~annot.columns.str.contains("Unnamed")]
    annot = annot.loc[~annot["Sample"].isna()]
    annot["Readcounts"] = [str(x).replace(",", ".") for x in annot["Readcounts"]]
    annot["Readcounts"] = annot["Readcounts"].astype(float)
    annot["QC_score"] = annot["QC_score"].astype(float)
    readcounts_df = annot[["Sample", "Readcounts"]].groupby(["Sample"]).sum()
    annot = annot.drop(["Readcounts", "Run_tiledb_id"], axis=1)
    annot = (
        annot.sort_values("QC_score", ascending=False)
        .drop_duplicates("Sample", keep="last")
        .sort_index()
    )
    annot = readcounts_df.merge(annot, left_index=True, right_on="Sample", how="outer")

    annot.index = annot["Sample"]
    annot.index.name = None

    return annot


def cohen_d(x: Iterable[float], y: Iterable[float]) -> float:
    """
    Calculate the Cohen's d value for two datasets.

    Cohen's d is a measure of the difference between two datasets.
    It is the difference between the two means, divided by the pooled standard deviation.

    :param x: The first dataset
    :param y: The second dataset
    :return: The Cohen's d value

    """

    return (np.mean(x) - np.mean(y)) / np.sqrt(
        (np.std(x, ddof=1) ** 2 + np.std(y, ddof=1) ** 2) / 2.0
    )


def get_gspread(table, sheet, index_col=0, path_to_json=None):
    """
    Access Google Sheets via python

    :param path_to_json: absolute path to JSON containing creditials for Google API
    :param table: Annotation calculated, for instance
    :param sheet: Blood, for instance
    :param index_col: column number
    :return: pandas DataFrame
    """

    if path_to_json == None:
        path_to_json = "/uftp/Blood/google_secret.json"

    credentials = ServiceAccountCredentials.from_json_keyfile_name(
        path_to_json
    )  # Your json file here
    gc = gspread.authorize(credentials)

    # open table
    wks = gc.open(table)
    # select sheet
    sheet = wks.worksheet(sheet)
    data = pd.DataFrame(sheet.get_all_records())
    data = data.set_index(data.columns[index_col])
    return data


def df_fisher_chi2(
    clusters: pd.Series,
    response: pd.Series,
    df: Optional[pd.DataFrame] = None,
    R: str = "R",
    NR: str = "NR",
):
    """
    Calculate fisher exact test p-value and chi-squared test p-value between cluster labels and response variable.

    Parameters
    ----------
    df : pandas.DataFrame, optional
        Contingency table of clusters and response variable. If None, it will be calculated using `pd.crosstab`.
    clusters : pandas.Series
        Cluster labels.
    response : pandas.Series
        Response variable.
    R : str, optional
        Name of the column in df which represents the positive response. Default is "R".
    NR : str, optional
        Name of the column in df which represents the negative response. Default is "NR".

    Returns
    -------
    pandas.DataFrame
        Contingency table with additional columns "Fisher_pv" and "Chi2_pv" which represent the p-values of fisher exact test and chi-squared test respectively. The p-values are corrected for multiple testing using FDR method.
    """
    try:
        if not df:
            df = pd.crosstab(clusters, response)
    except:
        df = df

    df.insert(0, "Fisher_pv", 1)
    df.insert(1, "Chi2_pv", 1)

    for i in df.index:
        nr, r = df[R].loc[i], df[NR].loc[i]
        nrj = df[R].sum() - nr
        rj = df[NR].sum() - r
        oddsratio, pvalue = fisher_exact([[nr, r], [nrj, rj]])
        if pvalue > 1:
            pvalue = 1
        df.at[i, "Fisher_pv"] = pvalue

    for i in df.index:
        nr, r = df[R].loc[i], df[NR].loc[i]
        nrj = df[R].sum() - nr
        rj = df[NR].sum() - r
        chi, pvalue, dof, exp = chi2_contingency([[r, nr], [rj, nrj]])
        if pvalue > 1:
            pvalue = 1
        df.at[i, "Chi2_pv"] = pvalue
    _, df["Fisher_pv"], _, _ = multipletests(df["Fisher_pv"], method="fdr_bh")
    _, df["Chi2_pv"], _, _ = multipletests(df["Chi2_pv"], method="fdr_bh")
    return df


def print_95ci_of_mean(data: Iterable, r: bool = True, bounds: bool = False) -> None:
    """
    Print 95% confidence interval for the mean of given data.

    Parameters
    ----------
    data : iterable
        The data to calculate the mean and standard error for.
    r : bool, optional
        Whether to print the mean and standard error with rounded values (default is True).
    bounds : bool, optional
        Whether to print the 95% confidence interval bounds (default is False).
    """
    sem = data.std() / (len(data) ** 0.5)
    mean = data.mean()
    z = 1.96
    lower_bound = mean - z * sem
    upper_bound = mean + z * sem
    if r:
        print(f"Mean: {round(mean,2)}±{round(z * sem, 3)}")
    else:
        print(f"Mean: {mean}±{z * sem}")
    if bounds:
        print(f"95% CI: ({lower_bound}, {upper_bound})")


def scale_series(series: pd.Series, feature_range: tuple = (0, 1)) -> pd.Series:
    """
    Scale a pandas Series to a given range.

    Parameters
    ----------
    series : pandas Series
        The series to be scaled.
    feature_range : tuple, optional
        Desired range of transformed data. Default is (0, 1).

    Returns
    -------
    scaled_series : pandas Series
        Scaled series.
    """

    name = series.name
    scaler = MinMaxScaler(feature_range=feature_range)
    series_2d = series.values.reshape(-1, 1)
    scaled_series_2d = scaler.fit_transform(series_2d)
    scaled_series = pd.Series(scaled_series_2d.flatten(), index=series.index)
    scaled_series.name = name
    return scaled_series


def read_expressions(
    annotation: pd.DataFrame,
    gene_subset: Union[None, List[str]] = None,
    suf: str = "-kallisto-Xena-gene-TPM_without_noncoding.tsv",
    sample_type: str = "Sample",
    path: Union[str, Path] = "/interna_data/Databases/Deconvolution/",
) -> pd.DataFrame:
    """
    Function for reading expressions from a database directory on and performing filtering and aggregation based on sample annotations.
    :param annotation: A pandas DataFrame with samples as rows and "Dataset" column.
    :param gene_subset: A list of gene names to include in the analysis. If None (default), all genes will be included in the output.
    :param suf: A suffix string that represents the file format of the gene expression data files in the database directory.
    :param sample_type: A string indicating the type of sample annotation, default "Sample".
    :param path: The path to the database directory, default "/interna_data/Deconvolution/" --sharing by request
    :return: A pd.DataFrame containing the aggregated gene expression data for the specified samples.
    """
    path = Path(path)
    missing_datasets = []
    samples_expr = []
    for dataset in annotation["Dataset"].unique():
        if sample_type == "Sample":
            sample_path = path / dataset / (dataset + "_by_sample" + suf)
        else:
            sample_path = path / dataset / (dataset + suf)

        if sample_path.is_file():
            sample_ex = pd.read_csv(sample_path, sep="\t", index_col=0)
            if "Gene" in sample_ex.index:
                sample_ex = sample_ex.drop("Gene")
            if gene_subset:
                sample_ex = sample_ex.loc[gene_subset]
            samples_expr.append(sample_ex)
        else:
            print("no " + dataset + " expression")
            missing_datasets.append(dataset)

    samples_expr = pd.concat(samples_expr, axis=1)
    samples_with_exp = list(
        set(annotation.index).intersection(set(samples_expr.columns))
    )
    samples_expr = samples_expr[samples_with_exp]
    return samples_expr


def to_common_samples(df_list: Iterable[Union[pd.DataFrame, pd.Series]] = ()):
    """
    Accepts a list of dataframe/series-s. Returns all dataframe/series-s with intersected indexes in the same order
    :param df_list: list of pd.DataFrame/pd.Series
    :return: pd.DataFrame
    """
    cs = set(df_list[0].index)
    for i in range(1, len(df_list)):
        cs = cs.intersection(df_list[i].index)

    if len(cs) < 1:
        warnings.warn("No common samples!")
    return [df_list[i].loc[list(cs)] for i in range(len(df_list))]


def star_pvalue(
    pvalue: float, lev1: float = 0.05, lev2: float = 0.01, lev3: float = 0.001
) -> Literal["***", "**", "*", "-"]:
    """
    Return star notation for p value
    :param pvalue: float
    :param lev1: float, '*' threshold
    :param lev2: float, '**' threshold
    :param lev3: float, '***' threshold
    :return: str
    """
    if pvalue < lev3:
        return "***"
    if pvalue < lev2:
        return "**"
    if pvalue < lev1:
        return "*"
    return "-"


def round_to_1(x) -> float:
    """
    Round "x" to first significant digit
    :param x: float
    :return: float
    """

    return round(x, -int(floor(log10(abs(x)))))


def get_pvalue_string(p, p_digits=3, stars=False, prefix="p-value") -> str:
    """
    Return string with p-value, rounded to p_digits
    :param stars: Display pvalue as stars
    :param p: float, p-value
    :param p_digits: int, default 4, number of digits to round p-value
    :param prefix:
    :return: str, p-value info string
    """
    significant_pvalue = 10 ** (-p_digits)
    if stars:
        pvalue_str = star_pvalue(p, lev3=10 ** (-p_digits))
    else:
        if p < significant_pvalue:
            if len(prefix):
                prefix += " < "
            pvalue_str = f"{prefix}{significant_pvalue}"
        else:
            if len(prefix):
                prefix += " = "
            if p < 0.00001:
                pvalue_str = f"{prefix}{round_to_1(p):.0e}"
            else:
                pvalue_str = f"{prefix}{round_to_1(p)}"
    return pvalue_str


def read_dataset(
    file: str,
    sep: str = "\t",
    header: Optional[int] = 0,
    index_col: Optional[int] = 0,
    comment: Optional[str] = None,
) -> pd.DataFrame:
    """
    Read a dataset from a file using pandas' `read_csv` function, with custom settings for delimiter,
    header row, index column, and comments.

    Parameters
    ----------
    file : str
        The file path or buffer to read data from.
    sep : str, optional
        The delimiter to use. Default is '\t' (tab).
    header : int, optional
        The row number to use as column names. Defaults to 0 (first line).
    index_col : int, optional
        The column number to use as the row labels of the DataFrame. Defaults to 0 (first column).
    comment : str, optional
        Indicates remainder of line should not be parsed. If found at the beginning of a line,
        the line will be ignored altogether. This parameter must be a single character. Default is None.

    Returns
    -------
    pd.DataFrame
        A DataFrame of the read data.
    """

    return pd.read_csv(
        file,
        sep=sep,
        header=header,
        index_col=index_col,
        na_values=["Na", "NA", "NAN"],
        comment=comment,
    )


def median_scale(
    data: pd.DataFrame | pd.Series,
    clip: Optional[float] = None,
    c: float = 1.0,
    exclude: Optional[pd.Series] = None,
    axis: Literal[0, 1] = 0,
) -> pd.Series[float] | pd.DataFrame:
    """
    Scale using median and mad over any axis (over columns by default).
    Removes the median and scales the data according to the median absolute deviation.

    To calculate Median Absolute Deviation (MAD) - function "mad" from statsmodels.robust.scale is used
    with arguments "c" equals to 1, hence no normalization is performed on MAD
    [please see: https://www.statsmodels.org/stable/generated/statsmodels.robust.scale.mad.html]

    :param data: pd.DataFrame of pd.Series
    :param clip: float, symmetrically clips the scaled data to the value
    :param c: float, coefficient of normalization used in calculation of MAD
    :param exclude: pd.Series, samples to exclude while calculating median and mad
    :param axis: int, default=0, axis to be applied on: if 0, scale over columns, otherwise (if 1) scale over rows

    :return: pd.DataFrame
    """

    if exclude is not None:
        data_filtered = data.reindex(data.index & exclude[~exclude].index)
    else:
        data_filtered = data

    median = 1.0 * data_filtered.median(axis=axis)

    if isinstance(data, pd.Series):
        madv = 1.0 * mad(data_filtered.dropna(), c=c)
        c_data = data.sub(median).div(madv)
    else:
        inv_axis = (axis + 1) % 2  # Sub and div are performed by the other axis
        madv = 1.0 * data_filtered.apply(lambda x: mad(x.dropna(), c=c), axis=axis)
        c_data = data.sub(median, axis=inv_axis).div(madv, axis=inv_axis)

    if clip is not None:
        return c_data.clip(-clip, clip)
    return c_data
