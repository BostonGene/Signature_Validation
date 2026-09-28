"""Reuse-what-is-on-disk guards for the notebook pipelines.

The validation notebooks write a handful of expensive artefacts: the ~220 MB
``mapping_ssgseas`` pickle behind the ssGSEA sweep, the weighted-F1 benchmark
table, the per-signature metrics brick of section M. A kernel restart used to
recompute every one of them from scratch, which is how the 2026-07-30 container
restart cost 19 minutes of the section M sweep.

These helpers make a cell reuse the artefact it already has:

.. code-block:: python

    if is_cached(MAPPING_PATH, force=FORCE_RECOMPUTE, label="mapping_ssgseas"):
        mapping_ssgseas = load_pickle(MAPPING_PATH)
    else:
        mapping_ssgseas = compute_mapping_ssgseas(...)
        dump_pickle(mapping_ssgseas, MAPPING_PATH)

The guard must always bind the same variables the compute branch binds -
downstream cells read them, and a bare ``skip`` leaves them undefined.
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Union

import pandas as pd
from loguru import logger

PathLike = Union[str, Path]


def is_cached(path: PathLike, *, force: bool = False, label: str = "") -> bool:
    """Whether ``path`` is already on disk and may be reused.

    Logs the decision either way, with size and mtime, so a notebook run makes
    it obvious which stages were skipped and how old the reused artefact is.
    """
    path = Path(path)
    name = label or path.name
    if force:
        logger.info("{}: FORCE_RECOMPUTE is set, recomputing {}", name, path)
        return False
    if not path.exists():
        logger.info("{}: {} is absent, computing it", name, path)
        return False
    stat = path.stat()
    logger.info(
        "{}: reusing {} ({:.1f} MB, written {})",
        name,
        path,
        stat.st_size / 1024**2,
        datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
    )
    return True


def load_pickle(path: PathLike) -> Any:
    """Read a pickled artefact."""
    with open(Path(path), "rb") as handle:
        return pickle.load(handle)


def dump_pickle(obj: Any, path: PathLike) -> Path:
    """Write ``obj`` to ``path``, creating the parent directory if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(obj, handle, pickle.HIGHEST_PROTOCOL)
    logger.info("wrote {} ({:.1f} MB)", path, path.stat().st_size / 1024**2)
    return path


def load_table(path: PathLike, **read_csv_kwargs: Any) -> pd.DataFrame:
    """Read a TSV artefact. Pass ``index_col=0`` for tables written with an index."""
    return pd.read_csv(Path(path), sep="\t", **read_csv_kwargs)


def dump_table(df: pd.DataFrame, path: PathLike, **to_csv_kwargs: Any) -> Path:
    """Write ``df`` as TSV, creating the parent directory if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", **to_csv_kwargs)
    logger.info("wrote {} ({} rows)", path, len(df))
    return path
