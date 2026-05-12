"""Stratified holdout splits for the rare-types rerun.

Per Q5 of the OD-128 plan: 10 repeated holdouts with stratification by
(signature score relative to its median) × (GOI / Control). For rare GOIs
(Th17, Endothelium_lymph, Eosinophils, Plasma_B/Plasmablasts) the test fold
is the only one that feeds the figures, and per-fold scores are averaged.
"""

from __future__ import annotations

from typing import List, Literal, Sequence

import numpy as np
import pandas as pd

from signature_validation.utils.utils import RANDOM_SEED


def stratified_holdout_indices(
    samples: Sequence[str],
    score: pd.Series,
    cohort_label: pd.Series,
    n_splits: int = 10,
    test_size: float = 0.25,
    random_state: int = RANDOM_SEED,
) -> List[pd.Index]:
    """Repeated stratified holdouts.

    Stratification key is ``(score >= median(score), cohort_label)``. Within
    each stratum, ``test_size`` fraction of samples is sampled without
    replacement; the test fold is the union across strata. Each split uses an
    independent ``random_state + split_id`` seed so the 10 folds are
    decorrelated but reproducible.

    Parameters
    ----------
    samples : sequence of str
        Sample identifiers to split.
    score : pd.Series
        Per-sample score used for the median-split stratum (e.g. BG-FGES ssGSEA).
    cohort_label : pd.Series
        Per-sample cohort label (e.g. ``"GOI"`` / ``"Control"``).
    n_splits : int
        Number of holdouts to generate.
    test_size : float
        Fraction of each stratum to include in the test fold.
    random_state : int
        Base seed for split-level RNGs.

    Returns
    -------
    list of pd.Index
        ``n_splits`` test-fold index objects.
    """
    samples_idx = pd.Index(samples)
    score = score.reindex(samples_idx)
    cohort_label = cohort_label.reindex(samples_idx)

    median = float(score.median())
    above = (score >= median).fillna(False).astype(bool)
    strata = pd.Series(
        list(zip(above, cohort_label.fillna("__nan__"))),
        index=samples_idx,
    )

    splits: List[pd.Index] = []
    for split_id in range(n_splits):
        rng = np.random.default_rng(random_state + split_id)
        test: List[str] = []
        for _, group in strata.groupby(strata):
            n_test = max(1, int(round(test_size * len(group))))
            chosen = rng.choice(group.index.values, size=n_test, replace=False)
            test.extend(chosen.tolist())
        splits.append(pd.Index(test, name=f"test_split_{split_id}"))
    return splits


def aggregate_score_over_splits(
    per_split_scores: List[pd.DataFrame],
    how: Literal["mean", "median"] = "mean",
) -> pd.DataFrame:
    """Reduce a list of per-split sample × signature frames to one frame.

    Parameters
    ----------
    per_split_scores : list of pd.DataFrame
        Each frame is indexed by sample IDs (the split's test fold) with
        identical columns (signatures).
    how : {'mean', 'median'}
        Reducer applied across splits per sample.

    Returns
    -------
    pd.DataFrame
        Reduced sample × signature frame.

    Raises
    ------
    ValueError
        If ``how`` is unknown.
    """
    if not per_split_scores:
        return pd.DataFrame()
    stacked = pd.concat(per_split_scores, keys=range(len(per_split_scores)))
    if how == "mean":
        return stacked.groupby(level=1).mean()
    if how == "median":
        return stacked.groupby(level=1).median()
    raise ValueError(f"unknown reducer: {how!r}")
