"""Figure builders for the Figure-4 signature comparison plots.

Direct ports of v1 cells 60-68 (violins), 71-86 (heatmap) and 90-105
(sens/spec scatter), parametrised so the new-cohort notebook only passes a
``mapping_ssgseas`` plus a couple of palette overrides. Output filenames carry
a ``suffix`` (default ``"_new_cohort"``) so v1 SVGs are never overwritten.

For the rare-types rerun (separate notebook), :data:`RARE_FGES_KEYS` and
:data:`RARE_CELL_TYPES` drive an asterisk decoration on the relevant axis
labels — but in the new-cohort notebook those FGES are excluded entirely, so
the asterisks only fire when the rare-types pipeline reuses these helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, wilcoxon

from signature_validation.benchmark.cohorts import (
    CONTROLS_ORDER,
    EXCLUDED_FGES_RARE,
    MAP_RAW,
)
from signature_validation.plotting.plotting import (
    axis_matras,
    boxplot_with_pvalue,
    cells_color,
    cells_p,
    line_palette_annotation_plot,
    patch_plot,
)
from signature_validation.ssgsea_calc.ssgsea_calc import GeneSet
from signature_validation.utils.utils import (
    get_pvalue_string,
    median_scale,
    sort_by_terms_order,
    to_common_samples,
)

DEFAULT_CMAP = matplotlib.cm.coolwarm

# Trailing group that lumps every non-cell-of-interest into one annotation block.
OTHER_CONTROLS_LABEL = "Other controls"
OTHER_CONTROLS_COLOR = "#B0B0B0"

RARE_FGES_KEYS: Tuple[str, ...] = tuple(sorted(EXCLUDED_FGES_RARE))
RARE_CELL_TYPES: Tuple[str, ...] = (
    "Th17_cells",
    "Endothelium_lymph",
    "Eosinophils",
    "Plasma_B_cells",
    "Plasmablasts",
)

# Source palette for the per-source violin and the sens/spec scatter (v1 cell 96).
DIF_SOURCES_PAL: Dict[str, str] = {
    "BG": cells_color.navy,
    "Nirmal": cells_color.maroon2,
    "Bindea": cells_color.darkorange,
    "xCell": cells_color.forestgreen,
    "MSigDb": cells_color.dark_silver,
    "Random": "black",
    "KEGG": "gold",
    "GOBP": cells_color.electric_violet,
    "BioCarta": "#00c0ff",
}

# Per-source GOI/Control palette for the violin plot (v1 cell 66).
SOURCE_VIOLIN_PAL: Dict[str, str] = {
    "BG_CONTROL": "#000080",
    "BG_GOI": "#000080",
    "BINDEA_CONTROL": "violet",
    "BINDEA_GOI": "violet",
    "BIOCARTA_CONTROL": "#0054ff",
    "BIOCARTA_GOI": "#0054ff",
    "GOBP_CONTROL": "#00c0ff",
    "GOBP_GOI": "#00c0ff",
    "KEGG_CONTROL": "#39ffbe",
    "KEGG_GOI": "#39ffbe",
    "MSIG_CONTROL": "#90ff66",
    "MSIG_GOI": "#90ff66",
    "NIRMAL_CONTROL": "#e7ff0f",
    "NIRMAL_GOI": "#e7ff0f",
    "PETITPREZ_CONTROL": "#ffa300",
    "PETITPREZ_GOI": "#ffa300",
    "RANDOM_CONTROL": "#ff3f00",
    "RANDOM_GOI": "#ff3f00",
    "XCELL_CONTROL": "#bb0000",
    "XCELL_GOI": "#bb0000",
}

# Friendly labels for the violin x-axis (v1 cell 67).
VIOLIN_SOURCE_ORDER: Tuple[str, ...] = (
    "BG",
    "NIRMAL",
    "XCELL",
    "BINDEA",
    "BIOCARTA",
    "KEGG",
    "GOBP",
    "MSIG",
    "RANDOM",
)
VIOLIN_PRETTY: Tuple[str, ...] = (
    "BG",
    "Nirmal",
    "xCell",
    "Bindea",
    "BioCarta",
    "KEGG",
    "GO",
    "Other MSigDb",
    "Random",
)

# Combined GOI+Control box plot (v2 of plot_violin_per_source): GOI boxes are
# solid, Control boxes of the same source share the same fill colour but carry
# this hatch instead of a different colour.
CONTROL_HATCH = "//"

# Everything below is the Okabe-Ito colourblind-safe set, checked all-pairs with
# a validator rather than picked by eye: the worst pair is #D55E00 vs #009E73 at
# deuteranopic ΔE 11.0 (OKLab x100), comfortably over the ΔE 8 target, and the
# worst normal-vision pair is #D55E00 vs #E69F00 at ΔE 15.6.

# Combined box plot fill by *population*, not by signature source: the figure's
# question is GOI vs Control, so that is what the fill encodes. The orange sits
# below 3:1 against a white surface, which is why the Control boxes also keep
# CONTROL_HATCH and a black edge — texture, not colour alone, carries them into
# print and forced-colour rendering.
GOI_BOX_COLOR = "#0072B2"
CONTROL_BOX_COLOR = "#E69F00"

# BG-signature median reference lines: both red, as the figure calls for. Two
# red lines on one axis can only be told apart by their dash pattern, so here
# the pattern — not the colour — is what marks which population each one is.
BG_MEDIAN_COLOR = "#D55E00"
BG_GOI_MEDIAN_STYLE = "--"
BG_CONTROL_MEDIAN_STYLE = "-."

# Bracket colour encodes the TEST, not the population: Mann-Whitney (GOI vs
# Control within one source) in black at the bottom, paired Wilcoxon (adjacent
# sources) in green at the top. The Wilcoxon row is drawn twice — once across
# the GOI boxes, once across the Control boxes — and both rows are green because
# they are the same test; their x-position is what says which population.
MW_BRACKET_COLOR = "#000000"
WILCOXON_BRACKET_COLOR = "#009E73"

# Marker area for the internal (BG) FGES star in the sens/spec scatter. Set
# independently of the CV-derived sizes used for the external-source circles
# (32-144 pt^2), so the star stays legible on top of them.
INTERNAL_STAR_SIZE = 450.0

YTICK_FGES_LABEL: Dict[str, str] = {
    "Main4_Th1_signature": "Th1 cells Fges",
    "Main4_CD8_T_cells": "CD8+ T cells Fges",
    "Main4_Treg": "Treg cells Fges",
    "Main4_Neutrophil_signature": "Neutrophils Fges",
    "Main4_Effector_cells": "Effector cells Fges",
    "Main4_Eosinophil_signature": "Eosinophils Fges",
    "Main4_B_cells": "B cells Fges",
    "Main4_Endothelium": "Endothelial cells Fges",
    "Main4_Pan_macrophage_signature": "Macrophages Fges",
    "Main4_NK_cells": "NK cells Fges",
    "Main4_M2_signature": "M2 Macrophages Fges",
    "Main4_Mast_cell_signature": "Mast cells Fges",
    "Main4_Follicular_helper_T_cells": "Tfh cells Fges",
    "Main4_T_cells": "T cells Fges",
    "Main4_CD4_T_cells": "CD4+ T cells Fges",
    "Main4_Lymphatic_endothelium": "Lymphatic endothelium Fges",
    "Main4_Th17_signature": "Th17 cells Fges",
    "Main4_Plasma_cells": "Plasma cells Fges",
    "Main4_Monocyte": "Monocyte Fges",
}


def _classify_signature(name: str) -> str:
    """Classify a sub-signature name into one of :data:`VIOLIN_SOURCE_ORDER`."""
    upper = name.upper()
    for key in ("XCELL", "PETITPREZ", "BINDEA", "NIRMAL", "BIOCARTA", "KEGG", "GOBP", "RANDOM"):
        if key in upper:
            return key
    return "MSIG"


def _scatter_source_for_signature(name: str) -> str:
    """Map a sub-signature name to one of :data:`DIF_SOURCES_PAL`."""
    upper = name.upper()
    for key in ("Nirmal", "Bindea", "xCell", "KEGG", "GOBP", "BioCarta", "Random"):
        if key.upper() in upper:
            return key
    return "MSigDb"


def _star_label(label: str, mark: bool) -> str:
    return f"{label} *" if mark else label


def plot_violin_per_source(
    mapping_ssgseas: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
    save_dir: Union[str, Path],
    suffix: str = "_new_cohort",
    rare_cell_types: Sequence[str] = RARE_CELL_TYPES,
    rare_fges: Sequence[str] = RARE_FGES_KEYS,
    mark_rare: bool = False,
) -> None:
    """Per-source ssGSEA violin plot, separately for GOI and Control samples.

    Saves ``violin_comparison_nonscaled{suffix}_GOI.svg`` and
    ``..._control.svg`` under ``save_dir``. Direct port of v1 cells 60-68 with
    asterisks added to source labels whose pool intersects rare cell types
    (driven by sample-level naming) and rare FGES (driven by signature names).

    Parameters
    ----------
    mapping_ssgseas : dict
        Output of :func:`signature_validation.benchmark.scoring.compute_mapping_ssgseas`.
    save_dir : str or Path
    suffix : str
    rare_cell_types : sequence of str
    rare_fges : sequence of str
    mark_rare : bool
        When ``True`` decorate rare-source x-labels with ``*``; when ``False``
        (default) the rare/CV asterisks are stripped from the figure.
    """
    save_dir = Path(save_dir)
    rare_cell_types_set = set(rare_cell_types)
    rare_fges_set = set(rare_fges)

    bg_goi: List[pd.Series] = []
    bg_ctrl: List[pd.Series] = []
    other_goi: List[pd.Series] = []
    other_ctrl: List[pd.Series] = []

    for sign, groups in mapping_ssgseas.items():
        if not groups["Goi"] or not groups["Control"]:
            continue
        goi_df = pd.concat(groups["Goi"].values())
        control_df = pd.concat(groups["Control"].values())
        goi_rare = bool(rare_fges_set.intersection({sign})) or bool(
            rare_cell_types_set.intersection(groups["Goi"])
        )
        ctrl_rare = bool(rare_cell_types_set.intersection(groups["Control"]))
        for signat in goi_df.columns:
            is_bg = signat == sign
            tag = "_rare" if (goi_rare or signat in rare_fges_set) else ""
            x_goi = goi_df[signat].copy()
            x_goi.index = x_goi.index.map(lambda s: f"{s}_{signat}_{sign}_goi{tag}")
            x_ctrl = control_df[signat].copy()
            ctrl_tag = "_rare" if ctrl_rare else ""
            x_ctrl.index = x_ctrl.index.map(
                lambda s: f"{s}_{signat}_{sign}_control{ctrl_tag}"
            )
            if is_bg:
                bg_goi.append(x_goi)
                bg_ctrl.append(x_ctrl)
            else:
                other_goi.append(x_goi)
                other_ctrl.append(x_ctrl)

    if not bg_goi:
        return

    bg_goi_s = pd.concat(bg_goi)
    other_goi_s = pd.concat(other_goi) if other_goi else pd.Series(dtype=float)
    bg_ctrl_s = pd.concat(bg_ctrl)
    other_ctrl_s = pd.concat(other_ctrl) if other_ctrl else pd.Series(dtype=float)

    bg_goi_s.index = bg_goi_s.index.map(lambda s: f"{s}_BG_Goi")
    other_goi_s.index = other_goi_s.index.map(lambda s: f"{s}_Oth_Goi")
    bg_ctrl_s.index = bg_ctrl_s.index.map(lambda s: f"{s}_BG_Cont")
    other_ctrl_s.index = other_ctrl_s.index.map(lambda s: f"{s}_Oth_Cont")

    labels = pd.concat(
        [
            pd.Series(index=bg_goi_s.index, data="BG_GOI"),
            pd.Series(index=other_goi_s.index, data="MSIG_GOI"),
            pd.Series(index=bg_ctrl_s.index, data="BG_CONTROL"),
            pd.Series(index=other_ctrl_s.index, data="MSIG_CONTROL"),
        ]
    )
    sign_data = pd.concat([bg_goi_s, other_goi_s, bg_ctrl_s, other_ctrl_s])

    labels = labels[~labels.index.duplicated()]
    sign_data = sign_data[~sign_data.index.duplicated()].astype("float32")

    for source in ("XCELL", "PETITPREZ", "BINDEA", "NIRMAL", "BIOCARTA", "KEGG", "GOBP", "RANDOM"):
        goi_mask = labels.index.to_series().str.contains(source) & labels.index.to_series().str.contains("Goi")
        ctrl_mask = labels.index.to_series().str.contains(source) & labels.index.to_series().str.contains("Cont")
        labels.loc[goi_mask] = f"{source}_GOI"
        labels.loc[ctrl_mask] = f"{source}_CONTROL"

    rare_marked_goi = {
        cat
        for idx, cat in labels.items()
        if "_rare" in idx and cat.endswith("_GOI")
    }
    rare_marked_ctrl = {
        cat
        for idx, cat in labels.items()
        if "_rare" in idx and cat.endswith("_CONTROL")
    }

    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(20, 5))
    order_goi = [f"{s}_GOI" for s in VIOLIN_SOURCE_ORDER]
    pretty_goi = [
        _star_label(f"{p} Fges,\ncell type of interest", mark_rare and f"{s}_GOI" in rare_marked_goi)
        for s, p in zip(VIOLIN_SOURCE_ORDER, VIOLIN_PRETTY)
    ]
    boxplot_with_pvalue(
        sign_data,
        labels,
        palette=SOURCE_VIOLIN_PAL,
        ax=ax,
        title="Comparison of cell type Fges, GOI",
        violin=True,
        order=order_goi,
    )
    ax.set_xticklabels(pretty_goi, rotation=90)
    if (labels == "BG_GOI").any():
        ax.axhline(
            y=sign_data[labels == "BG_GOI"].median(),
            color="r",
            linestyle="--",
            alpha=0.2,
        )
    if (labels == "BG_CONTROL").any():
        ax.axhline(
            y=sign_data[labels == "BG_CONTROL"].median(),
            color="r",
            linestyle="--",
            alpha=0.2,
        )
    ax.set_ylabel("Unscaled ssGSEA score")
    plt.rcParams["svg.fonttype"] = "none"
    fig.savefig(save_dir / f"violin_comparison_nonscaled{suffix}_GOI.svg", format="svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(20, 5))
    order_ctrl = [f"{s}_CONTROL" for s in VIOLIN_SOURCE_ORDER]
    pretty_ctrl = [
        _star_label(f"{p} Fges,\ncontrol types", mark_rare and f"{s}_CONTROL" in rare_marked_ctrl)
        for s, p in zip(VIOLIN_SOURCE_ORDER, VIOLIN_PRETTY)
    ]
    boxplot_with_pvalue(
        sign_data,
        labels,
        palette=SOURCE_VIOLIN_PAL,
        ax=ax,
        title="Comparison of cell type Fges, Control types",
        violin=True,
        order=order_ctrl,
    )
    ax.set_xticklabels(pretty_ctrl, rotation=90)
    if (labels == "BG_GOI").any():
        ax.axhline(
            y=sign_data[labels == "BG_GOI"].median(),
            color="r",
            linestyle="--",
            alpha=0.2,
        )
    if (labels == "BG_CONTROL").any():
        ax.axhline(
            y=sign_data[labels == "BG_CONTROL"].median(),
            color="r",
            linestyle="--",
            alpha=0.2,
        )
    ax.set_ylabel("Unscaled ssGSEA score")
    fig.savefig(save_dir / f"violin_comparison_nonscaled{suffix}_control.svg", format="svg")
    plt.close(fig)


def _source_sample_means(
    mapping_ssgseas: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
    group: str,
) -> Dict[str, Dict[str, pd.Series]]:
    """Per-sample score of every source, averaged over that source's sub-signatures.

    A source contributes many sub-signatures per FGES, so "source A vs. source
    B on the same samples" is only well defined once each source is reduced to
    one value per (FGES, sample) — otherwise the pairing would have to pick an
    arbitrary sub-signature on each side, and pooling the full cross product
    would duplicate each sample once per sub-signature (pseudo-replication).

    Parameters
    ----------
    mapping_ssgseas : dict
    group : str
        ``"Goi"`` or ``"Control"`` — which sample population to reduce.

    Returns
    -------
    dict
        ``{source: {fges: Series indexed by sample}}``.
    """
    out: Dict[str, Dict[str, pd.Series]] = {}
    for sign, groups in mapping_ssgseas.items():
        frames = groups.get(group, {})
        if not frames:
            continue
        df = pd.concat(frames.values())
        df = df[~df.index.duplicated(keep="first")]
        by_source: Dict[str, List[str]] = {}
        for signat in df.columns:
            source = "BG" if signat == sign else _classify_signature(signat)
            by_source.setdefault(source, []).append(signat)
        for source, cols in by_source.items():
            series = df[cols].mean(axis=1).dropna()
            if not series.empty:
                out.setdefault(source, {})[sign] = series
    return out


def _paired_neighbour_pvalues(
    mapping_ssgseas: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
    group: str,
    sources: Sequence[str],
) -> Dict[Tuple[str, str], float]:
    """Paired-Wilcoxon p-value for each *adjacent* pair in ``sources``.

    Samples are matched within every FGES (the ``to_common_samples`` +
    ``wilcoxon`` pattern of the Section T stats table) on the per-source means
    of :func:`_source_sample_means`, then pooled across FGES. Only neighbouring
    sources are tested: the all-vs-BG variant produced one bracket per source,
    which stacked into a bracket tower taller than the data panel itself.

    Parameters
    ----------
    mapping_ssgseas : dict
    group : str
        ``"Goi"`` or ``"Control"``.
    sources : sequence of str
        Sources in plotting order; consecutive entries are compared.

    Returns
    -------
    dict
        ``{(left, right): pvalue}`` for every adjacent pair with paired samples.
    """
    means = _source_sample_means(mapping_ssgseas, group)
    pvalues: Dict[Tuple[str, str], float] = {}
    for left, right in zip(sources, sources[1:]):
        xs: List[pd.Series] = []
        ys: List[pd.Series] = []
        for fges, x_full in means.get(left, {}).items():
            y_full = means.get(right, {}).get(fges)
            if y_full is None:
                continue
            x, y = to_common_samples((x_full, y_full))
            if len(x) == 0:
                continue
            xs.append(x)
            ys.append(y)
        if not xs:
            continue
        try:
            pvalues[(left, right)] = wilcoxon(pd.concat(xs), pd.concat(ys)).pvalue
        except ValueError:
            pvalues[(left, right)] = 1.0
    return pvalues


def _draw_bracket(
    ax: matplotlib.axes.Axes,
    x1: float,
    x2: float,
    y: float,
    pvalue: float,
    color: str,
    tick: float,
    above: bool = True,
    fontsize: float = 9,
    inset: float = 0.12,
) -> None:
    """Draw one p-value bracket as a plain three-segment line.

    The horizontal bar sits at ``y`` with two short legs of length ``tick``
    (data units) pointing *towards* the boxes and the label on the far side;
    ``above`` flips the whole thing for brackets drawn below the boxes.
    ``inset`` shortens the bar at both ends so that a chain of adjacent-pair
    brackets sharing endpoints (source 1-2, 2-3, 3-4 ...) reads as separate
    brackets rather than one long line.

    The first implementation used ``connectionstyle="bar,fraction=..."``, whose
    bar height scales with the x-distance between the compared boxes — brackets
    spanning the whole axis ended up several data units tall and ran off the
    figure.
    """
    direction = 1.0 if above else -1.0
    x_left, x_right = min(x1, x2) + inset, max(x1, x2) - inset
    ax.plot(
        [x_left, x_left, x_right, x_right],
        [y - direction * tick, y, y, y - direction * tick],
        lw=1.0,
        color=color,
        solid_capstyle="butt",
        zorder=5,
    )
    ax.text(
        (x_left + x_right) / 2,
        y + direction * tick * 0.4,
        get_pvalue_string(pvalue, p_digits=3, stars=True),
        fontsize=fontsize,
        color=color,
        ha="center",
        va="bottom" if above else "top",
        zorder=6,
    )


def plot_combined_box_per_source(
    mapping_ssgseas: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
    save_dir: Union[str, Path],
    suffix: str = "_new_cohort",
    provenance: Optional[Dict[str, Dict[str, Dict[str, str]]]] = None,
    swarm: bool = False,
) -> Optional[Dict[str, str]]:
    """One combined GOI+Control box plot per signature source.

    "Combined" is meant in two senses. Vertically, it replaces the two separate
    ``plot_violin_per_source`` panels (one for GOI samples, one for Control
    samples) with a single panel: for every source in
    :data:`VIOLIN_SOURCE_ORDER` (BG first, then the external sources), a GOI
    box sits immediately next to a Control box. Fill encodes the population, not
    the source — :data:`GOI_BOX_COLOR` vs :data:`CONTROL_BOX_COLOR`, the latter
    also hatched (:data:`CONTROL_HATCH`) so the pair survives greyscale and
    colour-vision deficiency. Horizontally, one box pools *every* FGES and
    *every* sub-signature attributed to that source — this is a per-source
    summary, not a per-FGES one.

    Each *pair* carries one x-label naming the source and both counts: ``N``
    distinct samples and, in parentheses, the number of plotted points (roughly
    samples × sub-signatures of that source, summed over the FGES that have
    any), for the GOI and the Control box in turn.

    Two independent significance tests are drawn on the same axis:

    - Mann-Whitney U, GOI vs. Control of the *same* source (different samples,
      same signature family) — one bracket per source, below the boxes.
    - Paired Wilcoxon between *adjacent* sources on the **GOI** samples, matched
      sample-by-sample within each FGES on the per-source means of
      :func:`_source_sample_means` — one bracket per neighbouring pair, above
      the boxes (e.g. internal vs WikiPathways). Neighbours only: comparing
      every source against BG needed a bracket tower taller than the data panel.

    The BG signature's own GOI and Control medians are drawn as two red dashed
    reference lines (:data:`BG_MEDIAN_COLOR`), told apart by dash pattern:
    :data:`BG_GOI_MEDIAN_STYLE` vs :data:`BG_CONTROL_MEDIAN_STYLE`.

    Parameters
    ----------
    mapping_ssgseas : dict
        Output of :func:`signature_validation.benchmark.scoring.compute_mapping_ssgseas`
        (optionally crossval-backfilled).
    save_dir : str or Path
    suffix : str
    provenance : dict, optional
        Output of :func:`signature_validation.benchmark.crossval.backfill_rare_cell_types`.
        When given, the cell types substituted from the cross-validation
        pickle are listed in a caption under the plot so the figure's data
        provenance is explicit per cell type.
    swarm : bool
        Overlay individual sample points. Off by default — this plot already
        pools every FGES per source, so a swarm is usually too dense to read.

    Returns
    -------
    dict or None
        ``{cell_type: "true_holdout" | "cross_validation"}`` derived from
        ``provenance`` (``None`` if ``provenance`` was not given).
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    bg_goi: List[pd.Series] = []
    bg_ctrl: List[pd.Series] = []
    other_goi: List[pd.Series] = []
    other_ctrl: List[pd.Series] = []
    # Distinct sample ids behind each box. The plotted point count is
    # samples x sub-signatures, so it is not a sample size — keep both.
    box_samples: Dict[str, set] = {}

    for sign, groups in mapping_ssgseas.items():
        if not groups["Goi"] or not groups["Control"]:
            continue
        goi_df = pd.concat(groups["Goi"].values())
        control_df = pd.concat(groups["Control"].values())
        for signat in goi_df.columns:
            is_bg = signat == sign
            source = "BG" if is_bg else _classify_signature(signat)
            box_samples.setdefault(f"{source}_GOI", set()).update(goi_df.index)
            box_samples.setdefault(f"{source}_CONTROL", set()).update(control_df.index)
            x_goi = goi_df[signat].copy()
            x_goi.index = x_goi.index.map(lambda s: f"{s}_{signat}_{sign}_goi")
            x_ctrl = control_df[signat].copy()
            x_ctrl.index = x_ctrl.index.map(lambda s: f"{s}_{signat}_{sign}_control")
            if is_bg:
                bg_goi.append(x_goi)
                bg_ctrl.append(x_ctrl)
            else:
                other_goi.append(x_goi)
                other_ctrl.append(x_ctrl)

    if not bg_goi:
        return None

    bg_goi_s = pd.concat(bg_goi)
    other_goi_s = pd.concat(other_goi) if other_goi else pd.Series(dtype=float)
    bg_ctrl_s = pd.concat(bg_ctrl)
    other_ctrl_s = pd.concat(other_ctrl) if other_ctrl else pd.Series(dtype=float)

    bg_goi_s.index = bg_goi_s.index.map(lambda s: f"{s}_BG_Goi")
    other_goi_s.index = other_goi_s.index.map(lambda s: f"{s}_Oth_Goi")
    bg_ctrl_s.index = bg_ctrl_s.index.map(lambda s: f"{s}_BG_Cont")
    other_ctrl_s.index = other_ctrl_s.index.map(lambda s: f"{s}_Oth_Cont")

    labels = pd.concat(
        [
            pd.Series(index=bg_goi_s.index, data="BG_GOI"),
            pd.Series(index=other_goi_s.index, data="MSIG_GOI"),
            pd.Series(index=bg_ctrl_s.index, data="BG_CONTROL"),
            pd.Series(index=other_ctrl_s.index, data="MSIG_CONTROL"),
        ]
    )
    sign_data = pd.concat([bg_goi_s, other_goi_s, bg_ctrl_s, other_ctrl_s])

    labels = labels[~labels.index.duplicated()]
    sign_data = sign_data[~sign_data.index.duplicated()].astype("float32")

    for source in ("XCELL", "PETITPREZ", "BINDEA", "NIRMAL", "BIOCARTA", "KEGG", "GOBP", "RANDOM"):
        goi_mask = labels.index.to_series().str.contains(source) & labels.index.to_series().str.contains("Goi")
        ctrl_mask = labels.index.to_series().str.contains(source) & labels.index.to_series().str.contains("Cont")
        labels.loc[goi_mask] = f"{source}_GOI"
        labels.loc[ctrl_mask] = f"{source}_CONTROL"

    sources_present = [s for s in VIOLIN_SOURCE_ORDER if (labels == f"{s}_GOI").any() or (labels == f"{s}_CONTROL").any()]
    order: List[str] = []
    palette: Dict[str, str] = {}
    # One label per *pair* of boxes, centred between them, carrying both sample
    # counts — two half-labels under two adjacent boxes read as four unrelated
    # columns and hid that the pair is one comparison.
    pretty_labels: List[str] = []
    for s, p in zip(VIOLIN_SOURCE_ORDER, VIOLIN_PRETTY):
        if s not in sources_present:
            continue
        counts: Dict[str, Tuple[int, int]] = {}
        for suf, tag in (("_GOI", "GOI"), ("_CONTROL", "Control")):
            key = f"{s}{suf}"
            order.append(key)
            palette[key] = GOI_BOX_COLOR if suf == "_GOI" else CONTROL_BOX_COLOR
            counts[tag] = (
                len(box_samples.get(key, ())),
                int((labels == key).sum()),
            )
        pretty_labels.append(
            f"{p}\n"
            f"N GOI={counts['GOI'][0]:,} ({counts['GOI'][1]:,} pts)\n"
            f"N Control={counts['Control'][0]:,} ({counts['Control'][1]:,} pts)"
        )

    fig, ax = plt.subplots(figsize=(1.6 * len(order), 8))
    sns.boxplot(
        y=sign_data,
        x=labels,
        ax=ax,
        palette=palette,
        order=order,
        fliersize=0,
    )
    if swarm:
        sns.swarmplot(y=sign_data, x=labels, ax=ax, color=".25", order=order, s=3)
    for pos, key in enumerate(order):
        if key.endswith("_CONTROL"):
            ax.patches[pos].set_hatch(CONTROL_HATCH)
            ax.patches[pos].set_edgecolor("black")
    # Pin the ticks before relabelling: seaborn leaves an auto locator, and
    # set_xticklabels alone then warns that the labels may end up mislabelled.
    # One tick per pair, sitting between the two boxes it names (0.5, 2.5, ...).
    ax.set_xticks([2 * i + 0.5 for i in range(len(pretty_labels))])
    ax.set_xticklabels(pretty_labels, rotation=90, fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("Unscaled ssGSEA score")
    ax.set_title(
        "Cell type Fges, GOI vs Control — combined over all FGES, per signature source\n"
        "one box = every sub-signature of that source in every FGES; "
        "N = distinct samples, (n pts) = plotted values; "
        "bracket colour = statistical test",
        fontsize=11,
    )

    if (labels == "BG_GOI").any():
        ax.axhline(
            y=sign_data[labels == "BG_GOI"].median(),
            color=BG_MEDIAN_COLOR,
            linestyle=BG_GOI_MEDIAN_STYLE,
            alpha=0.8,
            zorder=0,
        )
    if (labels == "BG_CONTROL").any():
        ax.axhline(
            y=sign_data[labels == "BG_CONTROL"].median(),
            color=BG_MEDIAN_COLOR,
            linestyle=BG_CONTROL_MEDIAN_STYLE,
            alpha=0.8,
            zorder=0,
        )

    y_min, y_max = ax.get_ylim()
    effective_size = y_max - y_min
    tick = effective_size * 0.015

    # Bottom brackets: Mann-Whitney U, GOI vs Control of the same source. These
    # already join neighbouring boxes, so a single row is enough.
    mw_y = y_min - effective_size * 0.05
    for s in sources_present:
        i_goi, i_ctrl = order.index(f"{s}_GOI"), order.index(f"{s}_CONTROL")
        goi_vals = sign_data[labels == f"{s}_GOI"]
        ctrl_vals = sign_data[labels == f"{s}_CONTROL"]
        try:
            pv = mannwhitneyu(goi_vals, ctrl_vals, alternative="two-sided").pvalue if len(goi_vals) and len(ctrl_vals) else 1.0
        except ValueError:
            pv = 1.0
        _draw_bracket(ax, i_goi, i_ctrl, mw_y, pv, MW_BRACKET_COLOR, tick, above=False)

    # Top brackets: paired Wilcoxon between neighbouring sources (e.g. internal
    # vs WikiPathways), one row across the GOI boxes and one across the Control
    # boxes. Both rows are WILCOXON_BRACKET_COLOR — colour names the *test*, so
    # one test never shows up in two colours; the rows are told apart by their
    # x-offset, GOI boxes sitting at 0-2-4... and Control at 1-3-5...
    top_step = effective_size * 0.09
    top_base = y_max + effective_size * 0.03
    for row, (group, suf) in enumerate((("Goi", "_GOI"), ("Control", "_CONTROL"))):
        pvalues = _paired_neighbour_pvalues(mapping_ssgseas, group, sources_present)
        for (left, right), pv in pvalues.items():
            keys = (f"{left}{suf}", f"{right}{suf}")
            if not all(k in order for k in keys):
                continue
            _draw_bracket(
                ax,
                order.index(keys[0]),
                order.index(keys[1]),
                top_base + row * top_step,
                pv,
                WILCOXON_BRACKET_COLOR,
                tick,
                above=True,
            )

    ax.set_ylim(mw_y - effective_size * 0.06, top_base + 2 * top_step)

    # Legend grouped the way the encodings are: first what a box is, then what a
    # bracket colour means, then the reference lines.
    legend_handles = [
        mpatches.Patch(facecolor=GOI_BOX_COLOR, edgecolor="black", label="GOI samples"),
        mpatches.Patch(
            facecolor=CONTROL_BOX_COLOR,
            edgecolor="black",
            hatch=CONTROL_HATCH,
            label="Control samples",
        ),
        plt.Line2D(
            [0], [0],
            color=MW_BRACKET_COLOR, linewidth=2,
            label="Mann-Whitney U — GOI vs Control, same source",
        ),
        plt.Line2D(
            [0], [0],
            color=WILCOXON_BRACKET_COLOR, linewidth=2,
            label="Paired Wilcoxon — adjacent sources (upper row: GOI, lower: Control)",
        ),
        plt.Line2D(
            [0], [0],
            color=BG_MEDIAN_COLOR, linestyle=BG_GOI_MEDIAN_STYLE,
            label="Internal (BG) GOI median",
        ),
        plt.Line2D(
            [0], [0],
            color=BG_MEDIAN_COLOR, linestyle=BG_CONTROL_MEDIAN_STYLE,
            label="Internal (BG) Control median",
        ),
    ]
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)

    ct_provenance: Optional[Dict[str, str]] = None
    if provenance:
        ct_provenance = {}
        for _, groups in provenance.items():
            for group in ("Goi", "Control", "Deleted_controls"):
                for ct, prov in groups.get(group, {}).items():
                    ct_provenance[ct] = prov
        cv_cts = sorted(ct for ct, prov in ct_provenance.items() if prov == "cross_validation")
        merged_cts = sorted(ct for ct, prov in ct_provenance.items() if prov == "merged")
        caption = (
            "Cross-validation-derived cell types (no samples in the true validation cohort): "
            + (", ".join(cv_cts) if cv_cts else "none")
            + ". Cell types pooling test and train samples (<20 true-cohort samples): "
            + (", ".join(merged_cts) if merged_cts else "none")
            + ". All other cell types: true holdout."
        )
        fig.text(0.01, -0.02, caption, fontsize=9, ha="left", va="top", wrap=True)

    plt.tight_layout(pad=0.4)
    fig.savefig(save_dir / f"box_comparison_combined{suffix}.svg", format="svg", bbox_inches="tight")
    plt.close(fig)
    return ct_provenance


def _build_short_df_index(
    out_df: pd.DataFrame,
    mapping: Dict[str, Dict[str, List[str]]],
    msigdb_gmt: Dict[str, Dict[str, GeneSet]],
    controls_to_skip_per_fges: Optional[Dict[str, Iterable[str]]] = None,
    top_k: int = 5,
) -> List[str]:
    """Pick top-k signatures per FGES by aggregated Cohen's d (v1 cell 75)."""
    controls_to_skip_per_fges = controls_to_skip_per_fges or {}
    indices: List[str] = []
    for sign in mapping:
        sub_signs = list(msigdb_gmt[sign].keys())
        goi = mapping[sign]["Goi"][0] if mapping[sign]["Goi"] else None
        skip = set(controls_to_skip_per_fges.get(sign, [])) | ({goi} if goi else set())
        cts = [ct for ct in mapping[sign]["Control"] if ct not in skip]
        cohen_cols = [f"Cohen's D from ours in {ct}" for ct in cts]
        cohen_cols = [c for c in cohen_cols if c in out_df.columns]
        cohen_cols.append("Cohen's D from ours in GOI")
        present = [s for s in sub_signs if s in out_df.index]
        if not present:
            continue
        part = out_df.loc[present, cohen_cols].copy()
        for col in cohen_cols[:-1]:
            part[col] = part[col] * -1
        part = part.dropna(how="all")
        part = part.sort_values(by=cohen_cols, ascending=True)
        indices.extend(list(part.index[:top_k]))
    return indices


def _slice_panel_boundaries(yticks: Sequence[str]) -> List[int]:
    """Return panel boundaries derived from FGES-label rows in ``yticks``."""
    label_set = set(YTICK_FGES_LABEL.values())
    boundaries = [i for i, y in enumerate(yticks) if y in label_set]
    return boundaries + [len(yticks)]


def plot_signature_heatmap(
    mapping_ssgseas: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
    out_df: pd.DataFrame,
    mapping: Dict[str, Dict[str, List[str]]],
    msigdb_gmt: Dict[str, Dict[str, GeneSet]],
    annotation: pd.DataFrame,
    controls_order: Sequence[str],
    palette: Optional[Dict[str, str]] = None,
    save_path: Union[str, Path] = "signature_heatmap.svg",
    short: bool = True,
    rare_fges: Sequence[str] = RARE_FGES_KEYS,
    main_cell_types: Optional[Sequence[str]] = None,
    mark_rare: bool = False,
) -> None:
    """Median-scaled signature × cell-type heatmap with cell-type annotation strip.

    Direct port of v1 cells 71-83. Each FGES contributes its top-5 signatures
    by Cohen's d; rare-FGES rows are starred in ``yticks`` when ``mark_rare``.

    The annotation strip keeps only the "cells of interest"
    (``main_cell_types``) as individual column blocks and lumps every other
    cell type into one trailing :data:`OTHER_CONTROLS_LABEL` block.

    Parameters
    ----------
    mapping_ssgseas : dict
    out_df : pd.DataFrame
    mapping : dict
    msigdb_gmt : dict
    annotation : pd.DataFrame
        Sample-indexed, ``Cell_type`` column.
    controls_order : sequence of str
        Retained for downstream use; on-figure grouping follows the
        ``main_cell_types`` / "Other controls" scheme instead.
    palette : dict, optional
        Cell-type → colour. Defaults to :data:`signature_validation.plotting.plotting.cells_p`.
    save_path : str or Path
    short : bool
        Use top-5-by-Cohen's-d slice (True) or all signatures (False).
    rare_fges : sequence of str
    main_cell_types : sequence of str, optional
        Cell types kept as individual column blocks. When ``None``, derived as
        the ordered union of :data:`MAP_RAW` GOI cell types present in the
        cohort, ordered by their index in :data:`CONTROLS_ORDER` (types absent
        from ``CONTROLS_ORDER`` go last, in first-seen order).
    mark_rare : bool
        When ``True`` star rare-FGES rows in ``yticks``; when ``False``
        (default) the rare/CV asterisks are stripped from the figure.
    """
    palette = palette or {ct: cells_p.get(ct, "#777777") for ct in controls_order}
    rare_fges_set = set(rare_fges)

    labels_acc: List[pd.Series] = []
    df_acc: List[pd.DataFrame] = []
    for sign, groups in mapping_ssgseas.items():
        per_fges_frames: List[pd.DataFrame] = []
        for group in ("Goi", "Control", "Deleted_controls"):
            for ct, frame in groups[group].items():
                lbl = pd.Series(index=frame.index, data=ct)
                labels_acc.append(lbl)
                per_fges_frames.append(frame)
        if per_fges_frames:
            stacked = pd.concat(per_fges_frames)
            stacked = stacked[~stacked.index.duplicated(keep="first")]
            df_acc.append(stacked)
    if not df_acc:
        return
    sample_cell_type = pd.concat(labels_acc)
    sample_cell_type = sample_cell_type[~sample_cell_type.index.duplicated(keep="first")]
    df_full = pd.concat(df_acc, axis=1)

    if short:
        sub_index = _build_short_df_index(out_df, mapping, msigdb_gmt)
        sub_index = [s for s in sub_index if s in df_full.columns]
        df_used = df_full[sub_index]
    else:
        df_used = df_full

    present_set = set(sample_cell_type.tolist())

    if main_cell_types is None:
        goi_union: List[str] = []
        for gois in MAP_RAW.values():
            for ct in gois:
                if ct not in goi_union:
                    goi_union.append(ct)
        candidates = [ct for ct in goi_union if ct in present_set]
        in_order = [ct for ct in CONTROLS_ORDER if ct in candidates]
        extras = [ct for ct in candidates if ct not in CONTROLS_ORDER]
        main_cell_types = in_order + extras

    main_present = [ct for ct in main_cell_types if ct in present_set]
    main_set = set(main_present)

    # Lump every non-cell-of-interest into a single trailing "Other controls".
    grouped_cell_type = sample_cell_type.where(
        sample_cell_type.isin(main_set), other=OTHER_CONTROLS_LABEL
    )
    has_other = bool((~sample_cell_type.isin(main_set)).any())

    column_order = list(main_present)
    if has_other:
        column_order.append(OTHER_CONTROLS_LABEL)

    heatmap_palette = {
        ct: (palette.get(ct) or cells_p.get(ct, "#777777")) for ct in main_present
    }
    if has_other:
        heatmap_palette[OTHER_CONTROLS_LABEL] = OTHER_CONTROLS_COLOR

    so = sort_by_terms_order(grouped_cell_type, column_order)

    data = median_scale(df_used.T).clip(-2, 2)
    yticks = [
        _star_label(
            YTICK_FGES_LABEL[i] if i in YTICK_FGES_LABEL else _msigdb_yticklabel(i),
            mark_rare and i in rare_fges_set,
        )
        for i in data.index
    ]
    data.index = yticks
    slices = _slice_panel_boundaries(yticks)
    if not slices or slices == [len(yticks)]:
        slices = [0, len(yticks)]
    elif slices[0] != 0:
        slices = [0, *slices]

    panel_heights = [0.3] + [
        0.23 * (slices[i + 1] - slices[i]) for i in range(len(slices) - 1)
    ]
    af = axis_matras(panel_heights, x_len=15)
    ax = next(af)
    line_palette_annotation_plot(grouped_cell_type[so], heatmap_palette, ax=ax)
    ax.set_ylabel("Cell\ntypes")

    for i in range(len(slices) - 1):
        sl = data.iloc[slices[i] : slices[i + 1]][so]
        ax = next(af)
        sns.heatmap(
            sl.clip(-2.5, 2.5),
            cmap=DEFAULT_CMAP,
            xticklabels=False,
            yticklabels=True,
            ax=ax,
            cbar=False,
        )
        ax.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )

    plt.tight_layout(pad=0.2)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, format="png", dpi=300)
    plt.close()


def _msigdb_yticklabel(name: str) -> str:
    parts = name.split("_")
    if len(parts) >= 2:
        return f"MSigDB's {parts[0]}...{parts[-2]}_{parts[-1]}"
    return name


def plot_sens_spec_scatter(
    mapping_ssgseas: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
    msigdb_gmt: Dict[str, Dict[str, GeneSet]],
    mapping: Dict[str, Dict[str, List[str]]],
    save_dir: Union[str, Path],
    suffix: str = "_new_cohort",
    rare_fges: Sequence[str] = RARE_FGES_KEYS,
    mark_rare: bool = False,
) -> Dict[str, Any]:
    """Per-FGES sens/spec scatter + per-source averaged scatter.

    Direct port of v1 cells 90-105 (per-FGES) and 105 (averaged). Returns the
    averaged sens/spec dict so callers can save additional aggregates.

    Parameters
    ----------
    mapping_ssgseas : dict
    msigdb_gmt : dict
    mapping : dict
    save_dir : str or Path
    suffix : str
    rare_fges : sequence of str
    mark_rare : bool
        When ``True`` append the rare-FGES ``(*)`` title suffix; when ``False``
        (default) that suffix is suppressed.

    Returns
    -------
    dict
        Aggregated structure keyed by FGES → {Sensitivity|Specificity → source → Series}.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    rare_fges_set = set(rare_fges)
    sources = ("BG", "Nirmal", "Bindea", "xCell", "MSigDb", "Random", "KEGG", "GOBP", "BioCarta")

    averaged: Dict[str, Dict[str, Dict[str, pd.Series]]] = {
        sign: {"Sensitivity": {s: [] for s in sources}, "Specificity": {s: [] for s in sources}}
        for sign in mapping_ssgseas
    }

    for sign, groups in mapping_ssgseas.items():
        if not groups["Goi"] or not groups["Control"]:
            continue
        goi_part = pd.concat(groups["Goi"].values())
        control_part = pd.concat(groups["Control"].values())
        goi_part = goi_part.T[~goi_part.T.index.duplicated(keep="first")].T
        goi_part = goi_part[~goi_part.index.duplicated(keep="first")]
        control_part = control_part.T[~control_part.T.index.duplicated(keep="first")].T
        control_part = control_part[~control_part.index.duplicated(keep="first")]
        goi_mean = goi_part.mean()
        ctrl_mean = control_part.mean()
        sensitivity = goi_mean.rank(pct=True)
        specificity = 1 - (ctrl_mean.rank(pct=True) / goi_mean.rank(pct=True)).rank(pct=True)
        sensitivity = sensitivity[~sensitivity.index.duplicated()]
        specificity = specificity[~specificity.index.duplicated()]
        sizes = goi_part.std() / (goi_part.mean() + goi_part.mean().mean())
        try:
            sizes_q = pd.qcut(
                sizes,
                q=[0, 0.25, 0.5, 0.75, 1],
                labels=[80 * f for f in (1.8, 1.1, 0.8, 0.4)],
            ).astype(float)
        except ValueError:
            sizes_q = pd.Series(80.0, index=sizes.index)

        plt.rcParams.update({"font.size": 12})
        fig, (ax0, ax, ax2) = plt.subplots(
            1, 3, figsize=(7.5, 5), gridspec_kw={"width_ratios": [2, 15, 0.5]}
        )

        internal_points: List[Tuple[float, float]] = []
        for signat in sensitivity.index[::-1]:
            if signat == sign:
                # internal (BG) FGES is drawn separately below with its own size
                internal_points.append(
                    (float(specificity.loc[signat]), float(sensitivity.loc[signat]))
                )
            else:
                ax.scatter(
                    specificity.loc[signat],
                    sensitivity.loc[signat],
                    s=sizes_q.get(signat, 80.0),
                    c=DIF_SOURCES_PAL[_scatter_source_for_signature(signat)],
                    marker="o",
                    edgecolors="white",
                    linewidths=1,
                    alpha=0.7,
                )
            src = "BG" if signat == sign else _scatter_source_for_signature(signat)
            averaged[sign]["Sensitivity"][src].append((signat, float(sensitivity.loc[signat])))
            averaged[sign]["Specificity"][src].append((signat, float(specificity.loc[signat])))

        if internal_points:
            ax.scatter(
                [p[0] for p in internal_points],
                [p[1] for p in internal_points],
                s=INTERNAL_STAR_SIZE,
                c=DIF_SOURCES_PAL["BG"],
                marker="*",
                edgecolors="black",
                linewidths=0.8,
                alpha=0.95,
                zorder=5,
            )

        ax.set_ylabel("Sensitivity: normalized ssGSEA score in GOI")
        ax.set_xlabel(
            "Specificity: 1 - normalized ratio of ssGSEA score in control cell types to GOI"
        )
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlim(-0.02, 1.05)
        patch_plot(DIF_SOURCES_PAL, ax=ax2, order=list(sources))
        ax2.set_ylabel("FGES sources")

        signname = sign.replace("Main4", "").replace("_", " ")
        gois = mapping.get(sign, {}).get("Goi", [])
        if len(gois) > 1:
            gnames = "s are " + ", ".join(gois)
        elif len(gois) == 1:
            gnames = " is " + gois[0]
        else:
            gnames = ""
        gnames = gnames.replace("_", " ")
        title = f"{signname} FGESs' specificity and sensitivity comparison\nGOI{gnames}"
        if mark_rare and sign in rare_fges_set:
            title += " (*)"
        ax.set_title(title)

        ax0.set_xticks([])
        ax0.set_ylabel("CV of ssGSEA scores in GOIs")
        plt.tight_layout(pad=0.5)
        fig.savefig(save_dir / f"sens_spec_comparison{suffix}_for_{sign}.svg", format="svg")
        plt.close(fig)

    averaged_series: Dict[str, Dict[str, Dict[str, pd.Series]]] = {}
    for sign, two in averaged.items():
        averaged_series[sign] = {"Sensitivity": {}, "Specificity": {}}
        for axis_name in ("Sensitivity", "Specificity"):
            for src, pairs in two[axis_name].items():
                if not pairs:
                    continue
                idx = [p[0] for p in pairs]
                vals = [p[1] for p in pairs]
                averaged_series[sign][axis_name][src] = pd.Series(vals, index=idx)

    plot_averaged_sens_spec(averaged_series, save_dir, suffix)
    return averaged_series


def plot_averaged_sens_spec(
    averaged: Dict[str, Dict[str, Dict[str, pd.Series]]],
    save_dir: Union[str, Path],
    suffix: str = "_new_cohort",
) -> None:
    """Averaged scatter across FGES, one point per source (v1 cells 103-105)."""
    save_dir = Path(save_dir)
    sens_per_source: Dict[str, List[pd.Series]] = {}
    spec_per_source: Dict[str, List[pd.Series]] = {}
    for sign, two in averaged.items():
        for src, ser in two["Sensitivity"].items():
            sens_per_source.setdefault(src, []).append(ser)
        for src, ser in two["Specificity"].items():
            spec_per_source.setdefault(src, []).append(ser)
    if not sens_per_source:
        return

    sens_concat = {src: pd.concat(parts) for src, parts in sens_per_source.items()}
    spec_concat = {src: pd.concat(parts) for src, parts in spec_per_source.items()}

    cv = pd.Series(
        {src: float(ser.std() / ser.mean()) if ser.mean() else float("nan") for src, ser in sens_concat.items()}
    ).dropna()
    if cv.empty:
        return
    sizes = pd.Series(index=cv.sort_values().index, data=[500 * v for v in cv.sort_values(ascending=False).values])

    plt.rcParams.update({"font.size": 12})
    fig, (ax0, ax, ax2) = plt.subplots(
        1, 3, figsize=(7, 5), gridspec_kw={"width_ratios": [2, 15, 0.5]}
    )
    for src in sens_concat:
        if src not in DIF_SOURCES_PAL:
            continue
        ax.scatter(
            float(spec_concat[src].mean()),
            float(sens_concat[src].mean()),
            s=float(sizes.get(src, 200)),
            c=DIF_SOURCES_PAL[src],
            edgecolors="white",
            linewidths=1,
            alpha=0.8,
        )

    patch_plot(DIF_SOURCES_PAL, ax=ax2, order=list(DIF_SOURCES_PAL.keys()))
    ax2.set_ylabel("FGES sources")
    ax.set_title("FGESs' specificity and sensitivity comparison\nAveraged by GOI")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylabel("Averaged sensitivity:\nnormalized ssGSEA score in GOIs")
    ax.set_xlabel(
        "Averaged specificity:\n1 - normalized ratio of ssGSEA score in control cell types to GOIs"
    )
    ax0.set_xticks([])
    ax0.set_ylabel("CV of sensitivity scores across GOIs")
    plt.tight_layout(pad=0.5)
    fig.savefig(save_dir / f"sens_spec_comparison{suffix}_averaged_wo_dbs.svg", format="svg")
    plt.close(fig)
