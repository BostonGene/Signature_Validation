"""Supplementary median-ssGSEA heatmaps for the validation cohort.

Two figures that :func:`signature_validation.benchmark.plotting.plot_signature_heatmap`
does not cover, both built on **per-cell-type medians of raw (unscaled) ssGSEA**:

1. :func:`plot_top_f1_heatmap` — top-``k`` sub-signatures per FGES ranked by
   bootstrap F1, colour = **z-score down each column**. Answers "within one cell
   type, which signature scores highest?", so the normalisation has to be per
   cell type. Untouched reproduction path for the published figure.
2. :func:`plot_top_metric_heatmaps` — one file per FGES, rows chosen either by a
   bootstrap metric or by :data:`SEPARATION_KEY` ("highest in the GOI, lowest in
   the controls").
3. :func:`plot_internal_specificity_heatmap` — the internal (BG) sub-signature of
   every FGES.

Both supplementary figures (2, 3) colour cells by the **raw** median ssGSEA on a
scale shared across every FGES, with square cells and no printed numbers: the
colorbar carries the value the cell text used to.

Columns follow :data:`SUPP_COLUMN_ORDER` verbatim; every cell type outside that
list is pooled into a single trailing ``Other controls`` column.

Cell types backfilled from the cross-validated original cohort
(:mod:`signature_validation.benchmark.crossval`) are marked on the figure — column
labels get ``*`` when every row of that column is backfilled, row labels get ``†``
when the whole signature comes from the crossval pickle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize, SymLogNorm

from signature_validation.benchmark.plotting import (
    DEFAULT_CMAP,
    OTHER_CONTROLS_LABEL,
    YTICK_FGES_LABEL,
    _msigdb_yticklabel,
)

# x-axis order requested for the supplementary figures, verbatim and strict.
SUPP_COLUMN_ORDER: Tuple[str, ...] = (
    "T_cells",
    "CD4_T_cells",
    "Th1_cells",
    "Th2_cells",
    "Th17_cells",
    "Follicular_T_helpers",
    "Tregs",
    "CD8_T_cells",
    "NK_cells",
    "B_cells",
    "Plasma_B_cells",
    "Plasmablasts",
    "Neutrophils",
    "Eosinophils",
    "Mast_cells",
    "Monocytes",
    "Macrophages",
    "Macrophages_M1",
    "Macrophages_M2",
    "Endothelium",
    "Endothelium_lymph",
)

# Cell types pooled into the trailing "Other controls" column: everything the
# validation cohort actually scored that is not one of the columns above.
OTHER_CONTROLS_MEMBERS: Tuple[str, ...] = (
    "Astrocytes",
    "Bronchial_cells",
    "CD4_T_helpers",
    "Dendritic_cells",
    "Epithelium",
    "Fibroblast_line",
    "Fibroblasts",
    "Hepatocytes",
    "Keratinocytes",
    "MAIT_cells",
    "MSC",
    "Memory_CD4_T_cells",
    "Memory_CD8_T_cells",
    "Monocytic_DC",
    "Myeloid_cells",
    "Neurons",
    "Non_plasma_B_cells",
    "Pancreatic_cells",
    "iPSC",
)

# The original cohort labelled tonsillar Tfh differently; the validation
# annotation renames them to Follicular_T_helpers (cohorts.RENAME_NEW_TO_OLD).
# Crossval-sourced frames still carry the old labels, so fold them in rather than
# leaving the Tfh column empty for crossval-only signatures.
COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    "Follicular_T_helpers": ("Follicular_T_helper", "Follicular_T_helper_tonsil"),
}

CROSSVAL_MARK = "†"
BACKFILLED_COLUMN_MARK = "*"
# Cell types whose scores pool the true holdout together with the crossval
# original cohort (crossval.backfill_rare_cell_types with mode="merge").
MERGED_MARK = "‡"

# Colour shown for cells with no samples at all in either cohort.
MISSING_CELL_COLOR = "#DDDDDD"

_GROUPS: Tuple[str, ...] = ("Goi", "Control", "Deleted_controls")

# Selection metrics where a larger value is better. These are exactly the keys
# fges_utils.get_metric_for_signature writes into fges_metrics.
HIGHER_IS_BETTER_METRICS: Tuple[str, ...] = (
    "F1",
    "Accuracy",
    "Precision_score",
    "Recall_score",
    "Average_precision",
    "ROC_AUC",
    "PR_AUC",
)

# Coefficient of variation of GOI expression ranks: *lower* is better, so it can
# never enter a composite score on the same footing as the metrics above —
# see :func:`composite_scores`.
CV_METRIC_KEY: str = "goi_cv"

# Row selection that reads no bootstrap metric at all: "highest in the GOI,
# lowest in the controls", scored on the plotted ssGSEA itself. Passed to
# :func:`plot_top_metric_heatmaps` in place of a metric name; it also names the
# output folder, so it stays a plain identifier.
SEPARATION_KEY: str = "separation"


def fges_order_by_cell_type(
    map_raw: Dict[str, Sequence[str]],
    column_order: Sequence[str] = SUPP_COLUMN_ORDER,
) -> List[str]:
    """Order FGES so their GOI cell type follows the x-axis order.

    Without this the row blocks follow ``MAP_RAW`` insertion order and the
    "hot" GOI cells scatter across the figure instead of running down a diagonal.
    FGES whose GOI is absent from ``column_order`` are appended, in input order.
    """
    positions = {ct: i for i, ct in enumerate(column_order)}
    fallback = len(column_order)

    def sort_key(item: Tuple[int, str]) -> Tuple[int, int]:
        index, fges = item
        ranks = [positions.get(ct, fallback) for ct in map_raw[fges]]
        return (min(ranks) if ranks else fallback, index)

    return [fges for _, fges in sorted(enumerate(map_raw), key=sort_key)]


def row_key(fges: str, signature: str) -> str:
    """Unique row identifier — the same MSigDb set can sit under two FGES."""
    return f"{fges}||{signature}"


def pretty_cell_type(cell_type: str) -> str:
    """``Macrophages_M1`` → ``Macrophages M1``; the "Other controls" label passes through."""
    return cell_type.replace("_", " ")


def _column_members(column: str, other_members: Sequence[str]) -> Tuple[str, ...]:
    """Cell types whose samples feed ``column``."""
    if column == OTHER_CONTROLS_LABEL:
        return tuple(other_members)
    return (column, *COLUMN_ALIASES.get(column, ()))


def _provenance_code(labels: Sequence[str]) -> str:
    """Collapse the provenance labels feeding one cell into a single code.

    A figure column can pool several cell types ("Other controls"), so one cell
    may draw on more than one provenance. ``"merged"`` wins over everything: once
    holdout and crossval samples share a median, that is the honest description.
    ``"mixed"`` marks the remaining case — a pooled column that silently blends
    two cohorts — which used to be invisible because the old boolean flag only
    recorded "every contributor is crossval".
    """
    unique = set(labels)
    if not unique:
        return ""
    if "merged" in unique:
        return "merged"
    if unique == {"cross_validation"}:
        return "cross_validation"
    if unique == {"true_holdout"}:
        return "true_holdout"
    return "mixed"


def _pooled_scores(
    frames_by_cell_type: Dict[str, Tuple[pd.DataFrame, str]],
    signature: str,
    members: Sequence[str],
) -> Tuple[Optional[np.ndarray], str]:
    """Pool ``signature`` scores across ``members``.

    Returns ``(scores, provenance_code)``; ``scores`` is ``None`` when no member
    cell type is scored for this signature.
    """
    chunks: List[np.ndarray] = []
    labels: List[str] = []
    for cell_type in members:
        entry = frames_by_cell_type.get(cell_type)
        if entry is None:
            continue
        frame, label = entry
        if signature not in frame.columns:
            continue
        chunks.append(frame[signature].to_numpy(dtype=float))
        labels.append(label)
    if not chunks:
        return None, ""
    return np.concatenate(chunks), _provenance_code(labels)


def _flatten_fges(
    mapping_ssgseas: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
    fges: str,
    provenance: Optional[Dict[str, Dict[str, Dict[str, str]]]] = None,
) -> Dict[str, Tuple[pd.DataFrame, str]]:
    """One frame per cell type for a single FGES, with its provenance label.

    A cell type can appear under two groups of the same FGES once the crossval
    backfill has run — ``backfill_rare_cell_types`` fills ``Goi`` / ``Control`` /
    ``Deleted_controls`` independently, so e.g. ``Main4_Pan_macrophage_signature``
    ends up with true-cohort ``Macrophages`` in ``Goi`` *and* crossval
    ``Macrophages`` in ``Deleted_controls``. Concatenating those would pool two
    cohorts into one median, so exactly one frame is kept: a merged frame beats a
    true-cohort frame beats crossval, then ``Goi`` beats ``Control`` beats
    ``Deleted_controls``.

    A ``"merged"`` frame already contains both cohorts by construction, so it has
    to outrank the true-holdout frame — otherwise the merge would be silently
    discarded here and the extra samples would never reach the median.
    """
    group_priority = {group: rank for rank, group in enumerate(_GROUPS)}
    label_priority = {"merged": 0, "true_holdout": 1, "cross_validation": 2}
    best: Dict[str, Tuple[Tuple[int, int], pd.DataFrame, str]] = {}
    dropped: List[str] = []

    for group in _GROUPS:
        for cell_type, frame in mapping_ssgseas[fges].get(group, {}).items():
            label = (
                (provenance or {})
                .get(fges, {})
                .get(group, {})
                .get(cell_type, "true_holdout")
            )
            rank = (label_priority.get(label, 2), group_priority[group])
            if cell_type not in best:
                best[cell_type] = (rank, frame, label)
                continue
            dropped.append(f"{fges}/{cell_type}")
            if rank < best[cell_type][0]:
                best[cell_type] = (rank, frame, label)

    if dropped:
        logger.debug(
            "{f}: {n} cell types scored under more than one group; kept one frame each ({d})",
            f=fges,
            n=len(dropped),
            d=", ".join(sorted(set(dropped))),
        )
    return {cell_type: (frame, label) for cell_type, (_, frame, label) in best.items()}


def build_median_matrix(
    mapping_ssgseas: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
    rows: Sequence[Tuple[str, str]],
    column_order: Sequence[str] = SUPP_COLUMN_ORDER,
    other_members: Sequence[str] = OTHER_CONTROLS_MEMBERS,
    provenance: Optional[Dict[str, Dict[str, Dict[str, str]]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Median raw ssGSEA per (sub-signature, cell-type column).

    Parameters
    ----------
    mapping_ssgseas : dict
        ``{FGES: {"Goi"|"Control"|"Deleted_controls": {cell_type: DataFrame}}}``,
        after the crossval backfill.
    rows : sequence of (str, str)
        ``(fges, sub_signature)`` pairs, in the row order wanted on the figure.
        ``fges`` selects which block's ssGSEA frames the sub-signature is read
        from; the same MSigDb set may appear under two FGES.
    column_order : sequence of str
        Cell types for the x-axis. ``Other controls`` is appended automatically.
    other_members : sequence of str
        Cell types pooled into the ``Other controls`` column.
    provenance : dict, optional
        ``{FGES: {group: {cell_type: "true_holdout"|"cross_validation"}}}`` from
        :func:`signature_validation.benchmark.crossval.backfill_rare_cell_types`.

    Returns
    -------
    (medians, counts, provenance_codes)
        Three row × column frames: median raw ssGSEA, pooled sample count, and a
        per-cell provenance code — one of ``"true_holdout"``, ``"cross_validation"``,
        ``"merged"``, ``"mixed"`` or ``""`` (no samples). Rows are indexed by
        ``"<fges>||<sub_signature>"`` because the same MSigDb set can appear under
        two FGES and a duplicated index would make positional lookups write to
        both rows.
    """
    columns = [*column_order, OTHER_CONTROLS_LABEL]
    row_keys = [row_key(fges, signature) for fges, signature in rows]

    medians = np.full((len(rows), len(columns)), np.nan, dtype=float)
    counts = np.zeros((len(rows), len(columns)), dtype=int)
    codes = np.full((len(rows), len(columns)), "", dtype=object)

    frames_cache: Dict[str, Dict[str, Tuple[pd.DataFrame, str]]] = {}
    for i, (fges, signature) in enumerate(rows):
        if fges not in frames_cache:
            frames_cache[fges] = _flatten_fges(mapping_ssgseas, fges, provenance)
        frames = frames_cache[fges]

        for j, column in enumerate(columns):
            members = _column_members(column, other_members)
            scores, code = _pooled_scores(frames, signature, members)
            if scores is None or scores.size == 0:
                continue
            medians[i, j] = float(np.median(scores))
            counts[i, j] = int(scores.size)
            codes[i, j] = code

    return (
        pd.DataFrame(medians, index=row_keys, columns=columns),
        pd.DataFrame(counts, index=row_keys, columns=columns),
        pd.DataFrame(codes, index=row_keys, columns=columns),
    )


def select_top_f1_rows(
    fges_metrics: Dict[str, Dict[int, Dict[str, float]]],
    msigdb_gmt: Dict[str, Dict[str, object]],
    fges_order: Sequence[str],
    top_k: int = 10,
    pin_internal: bool = False,
) -> Tuple[List[Tuple[str, str]], pd.DataFrame]:
    """Top-``top_k`` sub-signatures per FGES by mean bootstrap F1.

    ``fges_metrics`` is keyed by the bare sub-signature name, so an MSigDb set
    shared by two FGES keeps only the F1 of whichever block was scored last. Such
    rows are flagged ``ambiguous_f1`` in the returned table.

    Returns
    -------
    (rows, ranking)
        ``rows`` is the ``(fges, sub_signature)`` list for :func:`build_median_matrix`;
        ``ranking`` is the selected-row F1 table.
    """
    name_owners: Dict[str, int] = {}
    for fges in fges_order:
        for signature in msigdb_gmt.get(fges, {}):
            name_owners[signature] = name_owners.get(signature, 0) + 1

    records: List[dict] = []
    rows: List[Tuple[str, str]] = []
    for fges in fges_order:
        candidates = []
        for signature in msigdb_gmt.get(fges, {}):
            seed_map = fges_metrics.get(signature)
            if not seed_map:
                continue
            values = [
                seed_map[seed]["F1"]
                for seed in seed_map
                if "F1" in seed_map[seed] and np.isfinite(seed_map[seed]["F1"])
            ]
            if not values:
                continue
            candidates.append((signature, float(np.mean(values))))
        if not candidates:
            logger.warning("{f}: no sub-signature has a finite F1; skipped", f=fges)
            continue

        candidates.sort(key=lambda pair: pair[1], reverse=True)
        selected = candidates[:top_k]
        if pin_internal and fges not in [name for name, _ in selected]:
            internal = next((pair for pair in candidates if pair[0] == fges), None)
            if internal is not None:
                selected = [internal, *selected[: top_k - 1]]

        for rank, (signature, f1) in enumerate(selected, start=1):
            rows.append((fges, signature))
            records.append(
                {
                    "FGES": fges,
                    "Signature": signature,
                    "Rank": rank,
                    "F1": f1,
                    "is_internal": signature == fges,
                    "ambiguous_f1": name_owners.get(signature, 0) > 1,
                }
            )

    ranking = pd.DataFrame.from_records(records)
    n_ambiguous = int(ranking["ambiguous_f1"].sum()) if not ranking.empty else 0
    if n_ambiguous:
        logger.warning(
            "{n} of {t} selected rows share their name with another FGES block, so "
            "their F1 was measured on that block's GOI/control split",
            n=n_ambiguous,
            t=len(ranking),
        )
    return rows, ranking


def composite_scores(
    fges_metrics: Dict[str, Dict[int, Dict[str, float]]],
    signatures: Sequence[str],
    metric: str = "F1",
    cv_key: Optional[str] = CV_METRIC_KEY,
) -> pd.DataFrame:
    """Rank-based selection score for the sub-signatures of one FGES.

    Both quantities are first averaged over the bootstrap seeds, then turned into
    percentile ranks oriented "larger is better" — the metric as it stands, the CV
    negated, because a smaller CV means a steadier signal::

        score = rank_pct(metric_mean) + rank_pct(-cv_mean)          # 0..2

    Ranks rather than ``metric / cv``: on the real data ``1 / goi_cv`` spans
    1.5–16.8 while F1 spans 0.03–0.99, so a multiplicative composite would order
    signatures by whoever happens to have the smallest CV instead of by quality.
    Ranks give the two terms equal weight by construction and are immune to a
    future near-zero CV.

    Ranks are computed *within* ``signatures``, i.e. within one FGES, because that
    is the pool the top-k selection actually chooses from.

    Ties use ``method="average"`` so the ordering cannot depend on dict insertion
    order, which would make the selection irreproducible between runs.

    Returns
    -------
    pd.DataFrame
        Indexed by signature, columns ``metric_mean``, ``cv_mean``,
        ``metric_rank_pct``, ``cv_rank_pct``, ``score``. Empty when no signature
        has a usable value. With ``cv_key=None`` the CV columns are ``NaN`` and
        ``score`` is the metric's own percentile rank.
    """
    records: Dict[str, Tuple[float, float]] = {}
    for signature in signatures:
        seed_map = fges_metrics.get(signature)
        if not seed_map:
            continue
        metric_values = [
            seed[metric]
            for seed in seed_map.values()
            if metric in seed and np.isfinite(seed[metric])
        ]
        if not metric_values:
            continue
        # A non-positive or non-finite CV would make the reciprocal meaningless,
        # so such seeds are ignored rather than propagated as inf.
        cv_values = (
            [
                seed[cv_key]
                for seed in seed_map.values()
                if cv_key in seed and np.isfinite(seed[cv_key]) and seed[cv_key] > 0
            ]
            if cv_key is not None
            else []
        )
        if cv_key is not None and not cv_values:
            continue
        records[signature] = (
            float(np.mean(metric_values)),
            float(np.mean(cv_values)) if cv_values else float("nan"),
        )

    frame = pd.DataFrame.from_dict(
        records, orient="index", columns=["metric_mean", "cv_mean"]
    )
    if frame.empty:
        return frame.assign(metric_rank_pct=[], cv_rank_pct=[], score=[])

    frame["metric_rank_pct"] = frame["metric_mean"].rank(pct=True, method="average")
    if cv_key is None:
        frame["cv_rank_pct"] = float("nan")
        frame["score"] = frame["metric_rank_pct"]
    else:
        frame["cv_rank_pct"] = (-frame["cv_mean"]).rank(pct=True, method="average")
        frame["score"] = frame["metric_rank_pct"] + frame["cv_rank_pct"]
    return frame


def select_top_rows(
    fges_metrics: Dict[str, Dict[int, Dict[str, float]]],
    msigdb_gmt: Dict[str, Dict[str, object]],
    fges_order: Sequence[str],
    metric: str = "F1",
    cv_key: Optional[str] = CV_METRIC_KEY,
    top_k: int = 10,
    pin_internal: bool = False,
) -> Tuple[List[Tuple[str, str]], pd.DataFrame]:
    """Top-``top_k`` sub-signatures per FGES by the rank composite of ``metric`` and CV.

    The metric-agnostic counterpart of :func:`select_top_f1_rows`, which stays as
    the untouched reproduction path for the published F1-only figure.

    ``fges_metrics`` is keyed by the bare sub-signature name, so an MSigDb set
    shared by two FGES keeps only the metrics of whichever block was scored last.
    Such rows are flagged ``ambiguous_metric`` in the returned table.

    Returns
    -------
    (rows, ranking)
        ``rows`` is the ``(fges, sub_signature)`` list for :func:`build_median_matrix`;
        ``ranking`` is the selected-row score table.
    """
    name_owners: Dict[str, int] = {}
    for fges in fges_order:
        for signature in msigdb_gmt.get(fges, {}):
            name_owners[signature] = name_owners.get(signature, 0) + 1

    records: List[dict] = []
    rows: List[Tuple[str, str]] = []
    for fges in fges_order:
        scores = composite_scores(
            fges_metrics, list(msigdb_gmt.get(fges, {})), metric=metric, cv_key=cv_key
        )
        if scores.empty:
            logger.warning(
                "{f}: no sub-signature has a usable {m}; skipped", f=fges, m=metric
            )
            continue

        ordered = scores.sort_values("score", ascending=False)
        selected = list(ordered.index[:top_k])
        if pin_internal and fges in ordered.index and fges not in selected:
            selected = [fges, *selected[: top_k - 1]]

        for rank, signature in enumerate(selected, start=1):
            rows.append((fges, signature))
            records.append(
                {
                    "FGES": fges,
                    "Signature": signature,
                    "Rank": rank,
                    "Metric": metric,
                    "Metric_mean": float(ordered.at[signature, "metric_mean"]),
                    "CV_mean": float(ordered.at[signature, "cv_mean"]),
                    "Metric_rank_pct": float(ordered.at[signature, "metric_rank_pct"]),
                    "CV_rank_pct": float(ordered.at[signature, "cv_rank_pct"]),
                    "Score": float(ordered.at[signature, "score"]),
                    "is_internal": signature == fges,
                    "ambiguous_metric": name_owners.get(signature, 0) > 1,
                }
            )

    ranking = pd.DataFrame.from_records(records)
    n_ambiguous = int(ranking["ambiguous_metric"].sum()) if not ranking.empty else 0
    if n_ambiguous:
        logger.warning(
            "{n} of {t} selected rows share their name with another FGES block, so "
            "their {m} was measured on that block's GOI/control split",
            n=n_ambiguous,
            t=len(ranking),
            m=metric,
        )
    return rows, ranking


def separation_scores(
    mapping_ssgseas: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
    fges: str,
    signatures: Sequence[str],
) -> pd.DataFrame:
    """"Highest in the GOI, lowest in the controls" score for one FGES's signatures.

    Scored directly on the ssGSEA the figure itself plots — GOI samples against
    the pooled ``Control`` samples of the same FGES — rather than on the
    bootstrap metrics in ``fges_metrics``. Two consequences that matter: the
    crossval backfill is included (``fges_metrics`` predates it), and a MSigDb
    set shared by two FGES is scored separately under each, so it can no longer
    inherit the other block's GOI/control split.

    ``Deleted_controls`` are left out on purpose. They are the cell types each
    FGES excludes as biologically overlapping (``CONTROLS_TO_DELETE``) — e.g.
    Monocytes and M1/M2 for the pan-macrophage FGES — so counting them as
    controls would punish a signature for being right.

    ``gap`` — the raw median distance between the two cohorts — is the score
    :func:`select_top_separation_rows` ranks on, because it is the separation
    the reader actually sees on the figure: the cells are coloured by raw
    medians, so the widest gap is the widest colour swing. Cohen's d is computed
    alongside and reported, but not ranked on; it divides by the pooled SD and
    so demotes a signature purely for a broad within-GOI spread, which the
    figure does not show. The trade-off of ranking on ``gap`` is the opposite
    one: sub-signatures of a single FGES sit on different ssGSEA scales, so a
    large-magnitude gene set can win on gap without separating any better in
    relative terms — read ``d`` in the ranking table when that matters.

    Returns
    -------
    pd.DataFrame
        Indexed by signature, columns ``goi_median``, ``control_median``,
        ``gap`` (the score), ``d``, ``worst_control``, ``worst_control_median``,
        ``n_goi``, ``n_control``. ``worst_control*`` describe the single loudest
        control cell type — reported for inspection, not ranked on. Empty when
        no signature has both cohorts.
    """
    columns = [
        "goi_median",
        "control_median",
        "gap",
        "d",
        "worst_control",
        "worst_control_median",
        "n_goi",
        "n_control",
    ]
    groups = mapping_ssgseas.get(fges, {})

    records: Dict[str, Dict[str, object]] = {}
    for signature in signatures:
        goi_chunks = [
            frame[signature].to_numpy(dtype=float)
            for frame in groups.get("Goi", {}).values()
            if signature in frame.columns
        ]
        control_by_type = {
            cell_type: frame[signature].to_numpy(dtype=float)
            for cell_type, frame in groups.get("Control", {}).items()
            if signature in frame.columns
        }
        if not goi_chunks or not control_by_type:
            continue

        goi = np.concatenate(goi_chunks)
        control = np.concatenate(list(control_by_type.values()))
        goi = goi[np.isfinite(goi)]
        control = control[np.isfinite(control)]
        if goi.size < 2 or control.size < 2:
            continue

        # Pooled SD, the standard Cohen's d denominator. A zero SD would send d
        # to infinity, so such a signature is dropped rather than ranked first.
        pooled_var = (
            (goi.size - 1) * np.var(goi, ddof=1)
            + (control.size - 1) * np.var(control, ddof=1)
        ) / (goi.size + control.size - 2)
        pooled_sd = float(np.sqrt(pooled_var))
        if not np.isfinite(pooled_sd) or pooled_sd <= 0:
            continue

        control_medians = {
            cell_type: float(np.median(values[np.isfinite(values)]))
            for cell_type, values in control_by_type.items()
            if np.isfinite(values).any()
        }
        worst = max(control_medians, key=control_medians.get) if control_medians else ""

        records[signature] = {
            "goi_median": float(np.median(goi)),
            "control_median": float(np.median(control)),
            "gap": float(np.median(goi) - np.median(control)),
            "d": float(np.mean(goi) - np.mean(control)) / pooled_sd,
            "worst_control": worst,
            "worst_control_median": control_medians.get(worst, float("nan")),
            "n_goi": int(goi.size),
            "n_control": int(control.size),
        }

    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame.from_dict(records, orient="index")[columns]


def select_top_separation_rows(
    mapping_ssgseas: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
    msigdb_gmt: Dict[str, Dict[str, object]],
    fges_order: Sequence[str],
    top_k: int = 10,
    pin_internal: bool = False,
) -> Tuple[List[Tuple[str, str]], pd.DataFrame]:
    """Top-``top_k`` sub-signatures per FGES by :func:`separation_scores`.

    The counterpart of :func:`select_top_rows` for the "highest in the GOI,
    lowest in the controls" selection. It reads no ``fges_metrics``, so it also
    has no ``ambiguous_metric`` failure mode.

    Rows are ordered by ``gap``, the raw median distance between the GOI and the
    pooled controls — the same quantity the cell colours encode. Cohen's d rides
    along in the ranking table for inspection; see :func:`separation_scores` for
    what each of the two rewards.

    Returns
    -------
    (rows, ranking)
        ``rows`` is the ``(fges, sub_signature)`` list for
        :func:`build_median_matrix`; ``ranking`` is the selected-row score table.
    """
    reported = (
        "gap",
        "d",
        "goi_median",
        "control_median",
        "worst_control",
        "worst_control_median",
        "n_goi",
        "n_control",
    )

    records: List[dict] = []
    rows: List[Tuple[str, str]] = []
    for fges in fges_order:
        scores = separation_scores(mapping_ssgseas, fges, list(msigdb_gmt.get(fges, {})))
        if scores.empty:
            logger.warning(
                "{f}: no sub-signature has both GOI and control scores; skipped",
                f=fges,
            )
            continue

        ordered = scores.sort_values("gap", ascending=False)
        selected = list(ordered.index[:top_k])
        if pin_internal and fges in ordered.index and fges not in selected:
            selected = [fges, *selected[: top_k - 1]]

        for rank, signature in enumerate(selected, start=1):
            rows.append((fges, signature))
            records.append(
                {
                    "FGES": fges,
                    "Signature": signature,
                    "Rank": rank,
                    "Rank_in_fges": int(list(ordered.index).index(signature)) + 1,
                    "Pool": len(ordered),
                    **{key: ordered.at[signature, key] for key in reported},
                    "is_internal": signature == fges,
                }
            )

    return rows, pd.DataFrame.from_records(records)


def zscore(frame: pd.DataFrame, axis: int) -> pd.DataFrame:
    """Z-score along ``axis`` (0 = down each column, 1 = across each row), NaN-safe."""
    values = frame.to_numpy(dtype=float)
    mean = np.nanmean(values, axis=axis, keepdims=True)
    std = np.nanstd(values, axis=axis, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        scaled = (values - mean) / std
    scaled[~np.isfinite(scaled)] = np.nan
    return pd.DataFrame(scaled, index=frame.index, columns=frame.columns)


def median_color_norm(
    medians: pd.DataFrame,
    scale: str = "linear",
    linthresh: float = 1000.0,
) -> Normalize:
    """Colour scale for **raw** median ssGSEA, spanning the data as it stands.

    With the cell text gone the colour has to carry the number itself, so the
    range is taken from the data instead of the fixed ±3 of the z-scored
    figures, and the colorbar ticks read in ssGSEA units.

    ``scale="symlog"`` switches to a signed log10 scale (linear inside
    ±``linthresh``, log10 outside), which is the only log form these data admit:
    a plain log10 is undefined for the ~8% of medians that are negative. It is
    not the default because it hurts here — the medians span barely one decade
    (99.5% of cells fall in 10³–10⁴), so log10 squeezes the informative contrast
    (e.g. 8688 in Macrophages vs 2955 in NK cells → 3.94 vs 3.47) into a twelfth
    of the bar while spending the rest on the near-empty region around zero.
    """
    values = medians.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return Normalize(vmin=0.0, vmax=1.0)

    vmin, vmax = float(np.min(finite)), float(np.max(finite))
    if scale == "symlog":
        limit = max(abs(vmin), abs(vmax))
        return SymLogNorm(linthresh=linthresh, vmin=-limit, vmax=limit, base=10)
    if scale != "linear":
        raise ValueError(f"unknown scale {scale!r}; expected 'linear' or 'symlog'")
    return Normalize(vmin=vmin, vmax=vmax)


def _square_figsize(
    nrows: int,
    ncols: int,
    cell: float = 0.32,
    margin_w: float = 7.0,
    margin_h: float = 4.0,
) -> Tuple[float, float]:
    """Figure size that fits ``nrows × ncols`` square cells plus label margins.

    ``square=True`` pins the axes aspect, so a figsize with the wrong aspect
    would only pad the figure with whitespace instead of resizing the cells.
    The margins hold the long MSigDb row labels and the rotated column labels.
    """
    return (cell * ncols + margin_w, cell * nrows + margin_h)


def save_colorbar(
    save_path: Union[str, Path],
    label: str,
    norm: Normalize,
    cmap=DEFAULT_CMAP,
    orientation: str = "vertical",
) -> Path:
    """Write a standalone colorbar SVG matching one of the heatmaps."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    vertical = orientation == "vertical"
    fig = plt.figure(figsize=(1.1, 4.0) if vertical else (4.0, 1.1))
    rect = [0.12, 0.08, 0.28, 0.84] if vertical else [0.08, 0.45, 0.84, 0.28]
    cax = fig.add_axes(rect)
    bar = ColorbarBase(cax, cmap=cmap, norm=norm, orientation=orientation)
    bar.set_label(label)
    fig.savefig(save_path, format=save_path.suffix.lstrip(".") or "svg", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("colorbar → {p}", p=save_path)
    return save_path


def _log_column_pools(counts: pd.DataFrame) -> None:
    """Report the sample pool behind each column as min–max across rows.

    The pool is per (FGES, cell type), so a column is only as comparable across
    rows as this range is narrow — a single row's count would hide that.
    """
    summary = {
        column: (
            str(int(counts[column].min()))
            if counts[column].min() == counts[column].max()
            else f"{int(counts[column].min())}-{int(counts[column].max())}"
        )
        for column in counts.columns
    }
    logger.info("pooled samples per column (min-max across rows): {s}", s=summary)


def _row_label(fges: str, signature: str, row_codes: pd.Series) -> str:
    """Y label plus provenance marks: ``†`` fully crossval, ``‡`` contains merged."""
    base = YTICK_FGES_LABEL.get(signature, _msigdb_yticklabel(signature))
    marks = ""
    if (row_codes == "cross_validation").all():
        marks += f" {CROSSVAL_MARK}"
    if (row_codes == "merged").any():
        marks += f" {MERGED_MARK}"
    return f"{base}{marks}"


def _column_labels(codes: pd.DataFrame) -> List[str]:
    """X labels plus provenance marks: ``*`` fully crossval, ``‡`` contains merged."""
    labels: List[str] = []
    for column in codes.columns:
        values = codes[column]
        mark = ""
        if (values == "cross_validation").all():
            mark = BACKFILLED_COLUMN_MARK
        elif (values == "merged").any():
            mark = MERGED_MARK
        labels.append(f"{pretty_cell_type(column)}{mark}")
    return labels


def _log_mixed_pools(codes: pd.DataFrame) -> None:
    """Warn about columns that silently blend two cohorts inside one median.

    A pooled column ("Other controls") can mix true-holdout and crossval cell
    types. That never earns a figure mark — the marks are all-or-nothing by
    design — so it would otherwise be invisible.
    """
    mixed = [column for column in codes.columns if (codes[column] == "mixed").any()]
    if mixed:
        logger.warning(
            "{n} column(s) pool more than one cohort into a single median: {c}",
            n=len(mixed),
            c=", ".join(mixed),
        )


def _draw(
    scaled: pd.DataFrame,
    yticklabels: Sequence[str],
    xticklabels: Sequence[str],
    norm: Normalize,
    figsize: Tuple[float, float],
    annot: Optional[np.ndarray],
    annot_fontsize: float,
    ytick_fontsize: float,
    block_boundaries: Sequence[int],
    title: str,
    square: bool = True,
):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(MISSING_CELL_COLOR)
    sns.heatmap(
        scaled,
        cmap=DEFAULT_CMAP,
        # ``norm`` carries the limits, so vmin/vmax must stay None — matplotlib
        # rejects both being set at once.
        vmin=None,
        vmax=None,
        norm=norm,
        cbar=False,
        ax=ax,
        annot=annot,
        fmt="",
        annot_kws={"fontsize": annot_fontsize},
        linewidths=0.2,
        linecolor="white",
        square=square,
        xticklabels=list(xticklabels),
        yticklabels=list(yticklabels),
    )
    for boundary in block_boundaries:
        ax.axhline(boundary, color="black", linewidth=1.0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=ytick_fontsize)
    ax.tick_params(length=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title, fontsize=10, pad=12)
    fig.tight_layout()
    return fig


def plot_top_f1_heatmap(
    mapping_ssgseas: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
    fges_metrics: Dict[str, Dict[int, Dict[str, float]]],
    msigdb_gmt: Dict[str, Dict[str, object]],
    fges_order: Sequence[str],
    save_path: Union[str, Path],
    colorbar_path: Optional[Union[str, Path]] = None,
    provenance: Optional[Dict[str, Dict[str, Dict[str, str]]]] = None,
    column_order: Sequence[str] = SUPP_COLUMN_ORDER,
    other_members: Sequence[str] = OTHER_CONTROLS_MEMBERS,
    top_k: int = 10,
    pin_internal: bool = False,
    vmin: float = -3.0,
    vmax: float = 3.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Top-``top_k``-by-F1 signatures per FGES, coloured by **column** z-score.

    Returns ``(medians, ranking)`` — the raw median matrix behind the figure and
    the F1 ranking table used to pick the rows.
    """
    rows, ranking = select_top_f1_rows(
        fges_metrics, msigdb_gmt, fges_order, top_k=top_k, pin_internal=pin_internal
    )
    medians, counts, codes = build_median_matrix(
        mapping_ssgseas, rows, column_order, other_members, provenance
    )
    scaled = zscore(medians, axis=0).clip(vmin, vmax)

    yticklabels = [
        _row_label(fges, signature, codes.iloc[i])
        for i, (fges, signature) in enumerate(rows)
    ]
    boundaries: List[int] = []
    seen: List[str] = []
    for index, (fges, _) in enumerate(rows):
        if fges not in seen:
            seen.append(fges)
            if index:
                boundaries.append(index)

    fig = _draw(
        scaled,
        yticklabels=yticklabels,
        xticklabels=_column_labels(codes),
        norm=Normalize(vmin=vmin, vmax=vmax),
        figsize=(11.0, max(6.0, 0.17 * len(rows) + 3.0)),
        annot=None,
        annot_fontsize=4.0,
        ytick_fontsize=5.0,
        block_boundaries=boundaries,
        # Reproduction path for the published figure: its cells stay rectangular.
        square=False,
        title=(
            f"Top-{top_k} signatures per FGES by bootstrap F1 — median ssGSEA, "
            "z-scored within each cell type"
        ),
    )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # Honour the extension instead of hardcoding a format, so a ".svg" path can
    # never silently receive a PNG payload.
    fig.savefig(save_path, format=save_path.suffix.lstrip(".") or "svg", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(
        "heatmap → {p} ({r} rows × {c} columns)",
        p=save_path,
        r=len(rows),
        c=medians.shape[1],
    )

    if colorbar_path is not None:
        save_colorbar(
            colorbar_path,
            "Median ssGSEA, z-score by cell type",
            Normalize(vmin=vmin, vmax=vmax),
        )

    _log_column_pools(counts)
    _log_mixed_pools(codes)
    return medians, ranking


def plot_internal_specificity_heatmap(
    mapping_ssgseas: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
    fges_order: Sequence[str],
    save_path: Union[str, Path],
    colorbar_path: Optional[Union[str, Path]] = None,
    provenance: Optional[Dict[str, Dict[str, Dict[str, str]]]] = None,
    column_order: Sequence[str] = SUPP_COLUMN_ORDER,
    other_members: Sequence[str] = OTHER_CONTROLS_MEMBERS,
    color_scale: str = "linear",
) -> pd.DataFrame:
    """Internal (BG) signatures only, coloured by the **raw** median ssGSEA.

    The colour now carries the number that used to be printed inside each cell,
    so the colorbar reads in ssGSEA units and the cells stay clean. Row-wise
    z-scoring is gone with the annotation: normalising per row while the bar
    claims absolute units would make the two disagree.

    Returns the raw median matrix behind the figure.
    """
    rows = [(fges, fges) for fges in fges_order if fges in mapping_ssgseas]
    medians, counts, codes = build_median_matrix(
        mapping_ssgseas, rows, column_order, other_members, provenance
    )
    norm = median_color_norm(medians, scale=color_scale)

    yticklabels = [
        _row_label(fges, signature, codes.iloc[i])
        for i, (fges, signature) in enumerate(rows)
    ]

    fig = _draw(
        medians,
        yticklabels=yticklabels,
        xticklabels=_column_labels(codes),
        norm=norm,
        figsize=_square_figsize(len(rows), medians.shape[1]),
        annot=None,
        annot_fontsize=5.0,
        ytick_fontsize=8.0,
        block_boundaries=[],
        title="Internal (BG) FGES specificity — colour: median raw ssGSEA",
    )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # Honour the extension instead of hardcoding a format, so a ".svg" path can
    # never silently receive a PNG payload.
    fig.savefig(save_path, format=save_path.suffix.lstrip(".") or "svg", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(
        "heatmap → {p} ({r} rows × {c} columns)",
        p=save_path,
        r=len(rows),
        c=medians.shape[1],
    )

    if colorbar_path is not None:
        save_colorbar(colorbar_path, "Median raw ssGSEA", norm)

    _log_column_pools(counts)
    _log_mixed_pools(codes)
    return medians


def plot_top_metric_heatmaps(
    mapping_ssgseas: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
    fges_metrics: Dict[str, Dict[int, Dict[str, float]]],
    msigdb_gmt: Dict[str, Dict[str, object]],
    fges_order: Sequence[str],
    save_dir: Union[str, Path],
    metric: str = "F1",
    cv_key: Optional[str] = CV_METRIC_KEY,
    top_k: int = 10,
    provenance: Optional[Dict[str, Dict[str, Dict[str, str]]]] = None,
    column_order: Sequence[str] = SUPP_COLUMN_ORDER,
    other_members: Sequence[str] = OTHER_CONTROLS_MEMBERS,
    pin_internal: bool = False,
    color_scale: str = "linear",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """One SVG per FGES: its top-``top_k`` sub-signatures for a single selection metric.

    Replaces the single stacked figure of :func:`plot_top_f1_heatmap` with one file
    per FGES, written into ``save_dir / metric /``. Cell colour is the **raw**
    median ssGSEA on a shared scale, so the colorbar reads in ssGSEA units and
    the cells need no printed numbers.

    Pass ``metric=SEPARATION_KEY`` to select rows by "highest in the GOI, lowest
    in the controls" (:func:`select_top_separation_rows`) instead of by a
    bootstrap metric from ``fges_metrics``; ``cv_key`` is then unused.

    The colour limits are deliberately taken from the **full** matrix of every
    FGES and only afterwards sliced into files. Rescaling each FGES block on its
    own would make the colours of two files incomparable — which is exactly the
    comparison these figures exist to support. One colorbar per selection, not
    per file, for the same reason.

    Returns
    -------
    (medians, ranking)
        The full raw median matrix behind all files, and the selection table.
    """
    save_dir = Path(save_dir) / metric
    save_dir.mkdir(parents=True, exist_ok=True)

    if metric == SEPARATION_KEY:
        rows, ranking = select_top_separation_rows(
            mapping_ssgseas,
            msigdb_gmt,
            fges_order,
            top_k=top_k,
            pin_internal=pin_internal,
        )
        selection_label = "median gap, GOI vs pooled controls"
    else:
        rows, ranking = select_top_rows(
            fges_metrics,
            msigdb_gmt,
            fges_order,
            metric=metric,
            cv_key=cv_key,
            top_k=top_k,
            pin_internal=pin_internal,
        )
        selection_label = metric + ("" if cv_key is None else f" / {cv_key}")
    if not rows:
        logger.warning("{m}: no FGES produced any row; nothing written", m=metric)
        return pd.DataFrame(), ranking

    medians, counts, codes = build_median_matrix(
        mapping_ssgseas, rows, column_order, other_members, provenance
    )
    norm = median_color_norm(medians, scale=color_scale)

    written = 0
    for fges in fges_order:
        positions = [i for i, (block, _) in enumerate(rows) if block == fges]
        if not positions:
            continue

        fig = _draw(
            medians.iloc[positions],
            yticklabels=[
                _row_label(fges, rows[i][1], codes.iloc[i]) for i in positions
            ],
            xticklabels=_column_labels(codes.iloc[positions]),
            norm=norm,
            figsize=_square_figsize(len(positions), medians.shape[1]),
            annot=None,
            annot_fontsize=5.0,
            ytick_fontsize=8.0,
            block_boundaries=[],  # one FGES per file — nothing left to separate
            title=(
                f"{fges} — top {len(positions)} signatures by {selection_label}\n"
                "colour: median raw ssGSEA"
            ),
        )
        fig.savefig(save_dir / f"{fges}.svg", format="svg", dpi=200, bbox_inches="tight")
        plt.close(fig)
        written += 1

    save_colorbar(
        save_dir / "_colorbar.svg",
        f"Median raw ssGSEA — top {top_k} by {selection_label}",
        norm,
    )
    logger.info(
        "{m}: {n} per-FGES heatmaps ({r} rows total) → {p}",
        m=metric,
        n=written,
        r=len(rows),
        p=save_dir,
    )

    _log_column_pools(counts)
    _log_mixed_pools(codes)
    return medians, ranking
