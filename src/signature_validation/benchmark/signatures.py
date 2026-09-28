"""Load and assemble per-FGES gene-set buckets from v1's persisted GMT.

The v1 notebook builds ``msigdb_gmt`` in cells 12-33: prompt-regex search
across MSigDB, public-signature pool (Bindea / Petitprez / Nirmal / xCell),
deduplication, 10 randomly sampled FGES per signature, plus the internal
``chess_db`` BG signature. The resulting object is pickled to
``msigdb_gmt.pkl`` and reused here so the gene lists stay byte-identical to
the published run.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Dict, Iterable, Union

import pandas as pd
from loguru import logger

from signature_validation.ssgsea_calc.ssgsea_calc import GeneSet, gmt_genes_alt_names

RANDOM_FGES_PREFIX_RE = re.compile(r"^RANDOM_FGES\d+_")


def load_v1_msigdb_gmt(path: Union[str, Path]) -> Dict[str, Dict[str, GeneSet]]:
    """Load the persisted ``msigdb_gmt`` dict from a v1 notebook run.

    Parameters
    ----------
    path : str or Path
        Path to ``msigdb_gmt.pkl``.

    Returns
    -------
    dict
        ``{Main4_*: {sub_signature_name: GeneSet}}``. Includes the BG signature
        (key equals the FGES key), MSigDb / Bindea / Petitprez / Nirmal / xCell
        hits, and 10 ``RANDOM_FGES{1..10}_*`` per FGES.

    Raises
    ------
    TypeError
        If the pickled object is not the expected nested dict.
    """
    with open(path, "rb") as handle:
        gmt = pickle.load(handle)
    if not isinstance(gmt, dict):
        raise TypeError(f"expected dict in {path}, got {type(gmt).__name__}")
    if not gmt:
        raise TypeError(f"{path} contained an empty dict")
    sample_key = next(iter(gmt))
    if not isinstance(gmt[sample_key], dict):
        raise TypeError(
            f"{path} top-level value is {type(gmt[sample_key]).__name__}; "
            "expected dict of GeneSets"
        )
    logger.info(
        "v1 GMT loaded: {n} FGES, {m} sub-signatures total",
        n=len(gmt),
        m=sum(len(v) for v in gmt.values()),
    )
    return gmt


def select_msigdb_gmt_subset(
    full_gmt: Dict[str, Dict[str, GeneSet]],
    keep_fges: Iterable[str],
) -> Dict[str, Dict[str, GeneSet]]:
    """Restrict ``full_gmt`` to the FGES keys in ``keep_fges``.

    Parameters
    ----------
    full_gmt : dict
        Output of :func:`load_v1_msigdb_gmt`.
    keep_fges : iterable of str
        FGES keys to retain.

    Returns
    -------
    dict
        Subset of ``full_gmt`` containing only the requested keys.

    Raises
    ------
    KeyError
        If any element of ``keep_fges`` is missing from ``full_gmt``.
    """
    keep = list(keep_fges)
    missing = [k for k in keep if k not in full_gmt]
    if missing:
        raise KeyError(f"v1 GMT missing FGES keys: {sorted(missing)}")
    return {k: full_gmt[k] for k in keep}


def harmonize_gmt_to_index(
    gmt_per_fges: Dict[str, Dict[str, GeneSet]],
    expression_index: pd.Index,
) -> Dict[str, Dict[str, GeneSet]]:
    """Apply :func:`gmt_genes_alt_names` per FGES bucket against ``expression_index``.

    Parameters
    ----------
    gmt_per_fges : dict
        Output of :func:`select_msigdb_gmt_subset` (or any sub-dict structure).
    expression_index : pd.Index
        Gene index of the expression matrix to harmonise against.

    Returns
    -------
    dict
        New dict; original ``gmt_per_fges`` is not mutated.
    """
    return {
        sign: gmt_genes_alt_names(bucket, expression_index)
        for sign, bucket in gmt_per_fges.items()
    }


def count_random_fges(bucket: Dict[str, GeneSet]) -> int:
    """Count ``RANDOM_FGES{1..10}_*`` entries inside a single FGES bucket.

    Parameters
    ----------
    bucket : dict
        ``{sub_signature_name: GeneSet}`` for one FGES.

    Returns
    -------
    int
    """
    return sum(1 for name in bucket if RANDOM_FGES_PREFIX_RE.match(name))
