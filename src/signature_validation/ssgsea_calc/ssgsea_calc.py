import re
import warnings
from typing import Dict, List, Literal, Set

import mygene
import pandas as pd


def read_gene_sets(gmt_file: str) -> dict:
    """
    Return dict {geneset_name : GeneSet object}

    :param gmt_file: str, path to .gmt file
    :return: dict
    """
    gene_sets = dict()
    with open(gmt_file) as handle:
        for line in handle:
            items = line.strip().split("\t")
            name = items[0].strip()
            description = items[1].strip()
            genes = set([gene.strip() for gene in items[2:]])
            gene_sets[name] = GeneSet(name, description, genes)

    return gene_sets


class GeneSet(object):
    """ """

    def __init__(self, name, descr, genes):
        """

        :param name:
        :param descr:
        :param genes:
        """
        self.name = name
        self.descr = descr
        self.genes = set(genes)
        self.genes_ordered = list(genes)

    def __str__(self):
        """

        :return:
        """
        return "{}\t{}\t{}".format(self.name, self.descr, "\t".join(self.genes))


def query_genes_by_symbol(genes: List[str], verbose: bool = False) -> pd.DataFrame:
    """
    :param genes:
    :param verbose:
    :return:
    """

    mg = mygene.MyGeneInfo()
    q = mg.querymany(
        genes,
        species="human",
        as_dataframe=True,
        verbose=verbose,
        df_index=True,
        scopes=["symbol"],
        fields="all",
    )
    try:
        q.dropna(subset=["HGNC"], inplace=True)
        q.dropna(subset=["type_of_gene"], inplace=True)
        q.dropna(subset=["map_location"], inplace=True)
    except Exception:
        warnings.warn("Output lacks map_location or type_of_gene")

    return q


def update_gene_names(genes_old: set, genes_cur: set, verbose: bool = False) -> dict:
    """
    Takes a set of gene names genes_old and matches it with genes_cur.
    For all not found tries to match with known aliases using mygene.
    All not matched are returned as is.
    Returns a dict with matching rule. No duplicates will be in output
    :param genes_old:
    :param genes_cur:
    :param verbose:
    :return:
    """
    c_genes = set(genes_cur)
    old_genes = set(genes_old)

    missing = set()

    common_genes = c_genes.intersection(old_genes)
    if verbose:
        print("Matched: {}".format(len(common_genes)))

    converting_genes = old_genes.difference(c_genes)
    rest_genes = c_genes.difference(old_genes)
    match_rule = {cg: cg for cg in common_genes}

    if len(converting_genes):

        if verbose:
            print(
                "Trying to find new names for {} genes in {} known".format(
                    len(converting_genes), len(rest_genes)
                )
            )

        else:
            qr = query_genes_by_symbol(list(converting_genes), verbose=verbose)
        if hasattr(qr, "alias"):
            cg_ann = qr.alias.dropna()
        else:
            cg_ann = pd.DataFrame()

        for cg in converting_genes:
            if cg in cg_ann.index:
                if (isinstance(cg_ann.loc[cg], list)) | (
                    isinstance(cg_ann.loc[cg], pd.core.series.Series)
                ):
                    al_set = set(cg_ann[cg])
                else:
                    al_set = set([cg_ann.loc[cg]])

                hits = al_set.intersection(rest_genes)
                if len(hits) == 1:
                    match_rule[cg] = list(hits)[0]
                    rest_genes.remove(match_rule[cg])
                elif len(hits) > 1:
                    warnings.warn("{} hits for gene {}".format(len(hits), cg))
                    match_rule[cg] = list(hits)[0]
                    rest_genes.remove(match_rule[cg])
                else:
                    missing.add(cg)
                    match_rule[cg] = cg
            else:
                missing.add(cg)
                match_rule[cg] = cg
        if verbose and len(missing):
            print("{} genes were not converted".format(len(missing)))
    return match_rule


def gmt_genes_alt_names(
    gmt: Dict[str, GeneSet],
    genes: Set[str],
    verbose: bool = False,
    report_missing: bool = False,
    **kwargs,
) -> Dict[str, GeneSet]:
    """
    Updates gmt with genes aliases found in genes
    :param gmt: read_gene_sets() function result
    :param genes: list/set of genes available for current platform
    :param verbose: if True then prints all mismatched genes
    :param report_missing: report additional set with genes failed to convert
    :return: read_gene_sets() return like with updated or removed genes
    """
    s_genes = set(genes)
    alt_gmt = {}
    gmt_genes = set()

    for geneset in gmt:
        gmt_genes.update(gmt[geneset].genes)

    match_rule = update_gene_names(
        genes_old=gmt_genes, genes_cur=s_genes, verbose=verbose, **kwargs
    )

    missing_genes = set()
    for geneset in gmt:
        new_set = set()
        for gene in gmt[geneset].genes:
            if match_rule[gene] != gene or gene in s_genes:
                new_set.add(match_rule[gene])
            else:
                missing_genes.add(gene)
        alt_gmt[geneset] = GeneSet(
            name=gmt[geneset].name, descr=gmt[geneset].descr, genes=new_set
        )

    if report_missing:
        return alt_gmt, missing_genes
    return alt_gmt


def ssgsea_score(ranks: pd.DataFrame, genes: List) -> pd.Series:
    """
    Calculates single sample GSEA score based on vector of gene expression ranks.
    Only overlapping genes will be analyzed.
    The original article describing the ssGSEA formula: https://doi.org/10.1038/nature08460.
    We use adapted fast function. Result is the same as in analogous packages (like GSVA).

    :param ranks: DataFrame with gene expression ranks; samples in columns and genes in rows
    :param genes: list or set, genes of interest
    :return: Series with ssGSEA scores for samples
    """

    # Finding common_genes
    # Note: List is needed here because pandas can not do .loc with sets
    common_genes = list(set(genes).intersection(set(ranks.index)))

    # If not intersections were found
    if not common_genes:
        return pd.Series([0.0] * len(ranks.columns), index=ranks.columns)

    # Ranks of genes inside signature
    sranks = ranks.loc[common_genes]

    return (sranks**1.25).sum() / (sranks**0.25).sum() - (len(ranks.index) + 1) / 2


def ssgsea_formula(
    expressions: pd.DataFrame,
    gmt: Dict,
    rank_method: Literal["average", "min", "max", "first", "dense"] = "max",
) -> pd.DataFrame:
    """
    Return DataFrame with ssGSEA scores for gene signatures from gmt (dict of GeneSets)
    Only overlapping genes will be analyzed
    The original article describing the ssGSEA formula: https://doi.org/10.1038/nature08460

    :param expressions: DataFrame with gene expressions; samples in columns and genes in rows
    :param gmt: keys - signature names, values  GeneSet
    :param rank_method: {'min', 'max', etc} how to rank genes that have the same expression value
    :return: DataFrame with ssGSEA scores, index - signatures, columns - samples
    """

    # Calculate ranks
    ranks = expressions.rank(method=rank_method, na_option="bottom")

    # Calculate ssGSEA scores for all signatures - at the same moment merging it in DataFrame
    return pd.DataFrame(
        {
            gs_name: ssgsea_score(ranks, geneset.genes)
            for gs_name, geneset in gmt.items()
        }
    ).T


def detect_fges_source(fges: str) -> Literal[
    "Internal",
    "Random_FGES",
    "MSigDb_Single_Cell",
    "xCell",
    "Bindea",
    "Nirmal",
    "Gene_Ontology",
    "KEGG",
    "BioCarta",
    "WikiPathways",
    "Reactome",
    "Pathway_Interaction_Database",
    "Human_Phenotype_Ontology",
    "MSigDb_Dif_Expression",
    "MSigDb_Other",
    "Other",
]:
    """
    Detect the source of a Functional Gene Set (FGES) based on its name.

    Parameters
    ----------
    fges : str
        The name of the FGES.

    Returns
    -------
    source : str
        The source of the FGES. The possible sources are:
        - Internal
        - Random_FGES
        - MSigDb_Single_Cell
        - xCell
        - Bindea
        - Nirmal
        - Gene_Ontology
        - KEGG
        - BioCarta
        - WikiPathways
        - Reactome
        - Pathway_Interaction_Database
        - Human_Phenotype_Ontology
        - MSigDb_Dif_Expression
        - MSigDb_Other
        - Other
    """
    all_msigdb_gmt = read_gene_sets("./data/msigdb.v2023.1.Hs.symbols.gmt")

    sc_source = [
        "HE_LIM_SUN_FETAL_LUNG_",
        "DESCARTES_",
        "TRAVAGLINI_LUNG_",
        "FAN_EMBRYONIC_",
        "FAN_OVARY_",
        "GAUTAM_EYE_",
        "HAY_BONE_MARROW_",
        "DURANTE_ADULT_OLFACTORY_NEUROEPITHELIUM_",
        "CUI_DEVELOPING_HEART_",
        "LAKE_ADULT_KIDNEY_",
        "RUBENSTEIN_SKELETAL_MUSCLE_",
    ]
    if re.search("^Main4", fges):
        return "Internal"
    elif re.search("^RANDOM", fges):
        return "Random_FGES"
    elif [None] * len(sc_source) != [
        re.search(f"^{prefix}", fges) for prefix in sc_source
    ]:
        return "MSigDb_Single_Cell"
    elif "XCELL" in fges:
        return "xCell"
    elif "BINDEA" in fges:
        return "Bindea"
    elif "NIRMAL" in fges:
        return "Nirmal"
    elif (
        re.search("^GOBP_", fges)
        or re.search("^GOCC_", fges)
        or re.search("^GOMF_", fges)
    ):
        return "Gene_Ontology"
    elif re.search("^KEGG_", fges):
        return "KEGG"
    elif re.search("^BIOCARTA_", fges):
        return "BioCarta"
    elif re.search("^WP_", fges):
        return "WikiPathways"
    elif re.search("^REACTOME_", fges):
        return "Reactome"
    elif re.search("^PID_", fges):
        return "Pathway_Interaction_Database"
    elif re.search("^HP_", fges):
        return "Human_Phenotype_Ontology"
    elif (
        re.search(".*_VS_.*UP$", fges)
        or re.search(".*_VS_.*DN$", fges)
        or re.search("^(?!.*VS).*UP$", fges)
    ):
        return "MSigDb_Dif_Expression"
    elif fges in all_msigdb_gmt.keys():
        return "MSigDb_Other"
    else:
        return "Other"
