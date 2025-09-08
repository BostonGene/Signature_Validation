from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from signature_validation.utils.utils import read_dataset


def calculate_sample_noise(
    sample: Union[pd.Series, pd.DataFrame],
    readcounts: Union[float, pd.Series],
    gene_lengths_path=Path("/uftp/gene_data_common/" "gene_length_values.tsv"),
    alpha=2.05,
) -> Union[pd.Series, pd.DataFrame]:
    """
    Calculation of sigma describing gene expression noise in technical replicates.    :param sample: pandas dataframe with expressions in TPM with samples as indexes
    :param readcounts: sample readcounts (sum for every patient) in mln
    :param gene_lengths_path: gene lehgths in bp
    :returns: sigma describing gene expression noise in technical replicates
    """
    gene_length = read_dataset(gene_lengths_path)
    # gene lengths are to be turned to kbp
    gene_length = gene_length / 1000
    noise_sigma = (
        alpha
        * np.sqrt(
            sample.T.divide(gene_length.loc[sample.columns, "length"], axis="index")
            / readcounts
        ).T
    )
    return noise_sigma


def generate_noise(sample: pd.Series, noise_sigma: pd.Series) -> pd.Series:
    """
    Method adds Poisson noise (very close approximation) and uniform noise for expressions in TPM.
    Uniform noise - proportional to gene expressions noise from a normal distribution.
    :param sample: pandas series with expressions in TPM
    :param noise_sigma: sigma describing gene expression noise in technical replicates from calculate_sample_noise
    :returns: dataframe data with added noise
    """
    return (sample + noise_sigma * np.random.normal(size=sample.shape)).clip(lower=0)
