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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from signature_validation.benchmark.cohorts import EXCLUDED_FGES_RARE
from signature_validation.plotting.plotting import (
    axis_matras,
    boxplot_with_pvalue,
    cells_color,
    cells_p,
    line_palette_annotation_plot,
    patch_plot,
)
from signature_validation.ssgsea_calc.ssgsea_calc import GeneSet
from signature_validation.utils.utils import median_scale, sort_by_terms_order

DEFAULT_CMAP = matplotlib.cm.coolwarm

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
        _star_label(f"{p} Fges,\ncell type of interest", f"{s}_GOI" in rare_marked_goi)
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
        _star_label(f"{p} Fges,\ncontrol types", f"{s}_CONTROL" in rare_marked_ctrl)
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
) -> None:
    """Median-scaled signature × cell-type heatmap with cell-type annotation strip.

    Direct port of v1 cells 71-83. Each FGES contributes its top-5 signatures
    by Cohen's d; rare-FGES rows are starred in ``yticks``.

    Parameters
    ----------
    mapping_ssgseas : dict
    out_df : pd.DataFrame
    mapping : dict
    msigdb_gmt : dict
    annotation : pd.DataFrame
        Sample-indexed, ``Cell_type`` column.
    controls_order : sequence of str
    palette : dict, optional
        Cell-type → colour. Defaults to :data:`signature_validation.plotting.plotting.cells_p`.
    save_path : str or Path
    short : bool
        Use top-5-by-Cohen's-d slice (True) or all signatures (False).
    rare_fges : sequence of str
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

    so = sort_by_terms_order(sample_cell_type, list(controls_order))

    data = median_scale(df_used.T).clip(-2, 2)
    yticks = [
        _star_label(
            YTICK_FGES_LABEL[i] if i in YTICK_FGES_LABEL else _msigdb_yticklabel(i),
            i in rare_fges_set,
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
    line_palette_annotation_plot(sample_cell_type[so], palette, ax=ax)
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
    plt.savefig(save_path, format=save_path.suffix.lstrip(".") or "svg")
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

        for signat in sensitivity.index[::-1]:
            color = DIF_SOURCES_PAL["BG"] if signat == sign else DIF_SOURCES_PAL[
                _scatter_source_for_signature(signat)
            ]
            marker = "*" if signat == sign else "o"
            ax.scatter(
                specificity.loc[signat],
                sensitivity.loc[signat],
                s=sizes_q.get(signat, 80.0),
                c=color,
                marker=marker,
                edgecolors="white",
                linewidths=1,
                alpha=0.7,
            )
            src = "BG" if signat == sign else _scatter_source_for_signature(signat)
            averaged[sign]["Sensitivity"][src].append((signat, float(sensitivity.loc[signat])))
            averaged[sign]["Specificity"][src].append((signat, float(specificity.loc[signat])))

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
        if sign in rare_fges_set:
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
