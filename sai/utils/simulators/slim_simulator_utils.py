import msprime
import pyslim
import tskit
import random
import copy
from typing import List, Tuple, Dict, Optional


def remove_mutations(ts, start, end, proportion):
    """
    Removes mutations in the specified regions [start, end] with given probability.
    Ensures SLiM metadata is preserved or substituted to stay schema-compliant.

    Parameters:
        ts : tskit.TreeSequence
            Tree sequence with mutations.
        start : list of float
            Region starts.
        end : list of float
            Region ends (must be same length as `start`).
        proportion : float
            Probability of removing a mutation in specified regions.

    Returns:
        tskit.TreeSequence
            A new tree sequence with retained mutations and valid metadata.
    """
    tables = ts.dump_tables()
    tables.sites.clear()
    tables.mutations.clear()

    for tree in ts.trees():
        for site in tree.sites():
            if len(site.mutations) != 1:
                print("multiple mutations at site!")
                continue  # Skip if multiple mutations at site

            mut = site.mutations[0]

            # Decide whether to keep the mutation
            keep_mutation = True
            for i in range(len(start)):
                assert start[i] < end[i]
                if start[i] <= site.position < end[i]:
                    keep_mutation = random.uniform(0, 1) > proportion
                    break  # no need to check further

            if keep_mutation:
                site_id = tables.sites.add_row(
                    position=site.position, ancestral_state=site.ancestral_state
                )

                # If original mutation has metadata, keep it; otherwise, add SLiM-compliant dummy metadata
                if mut.metadata and len(mut.metadata.get("mutation_list", [])) > 0:
                    metadata = mut.metadata
                else:
                    metadata = {
                        "mutation_list": [
                            {
                                "mutation_type": 0,
                                "selection_coeff": 0.0,
                                "subpopulation": 0,
                                "slim_time": 1,
                                "nucleotide": -1,
                            }
                        ]
                    }

                tables.mutations.add_row(
                    site=site_id,
                    node=mut.node,
                    derived_state=mut.derived_state,
                    metadata=metadata,
                )

    return tables.tree_sequence()


def remove_mutations_non_slim(ts, start, end, proportion):
    """
    Removes *non-SLiM* mutations in the specified regions [start, end) with given probability.
    SLiM mutations are always preserved.
    Ensures SLiM metadata is preserved or substituted to stay schema-compliant.

    Parameters:
        ts : tskit.TreeSequence
            Tree sequence with mutations.
        start : list of float
            Region starts.
        end : list of float
            Region ends (must be same length as `start`).
        proportion : float
            Probability of removing a *non-SLiM* mutation in specified regions.

    Returns:
        tskit.TreeSequence
            A new tree sequence with retained mutations and valid metadata.
    """
    tables = ts.dump_tables()
    tables.sites.clear()
    tables.mutations.clear()

    for tree in ts.trees():
        for site in tree.sites():
            if len(site.mutations) != 1:
                print("multiple mutations at site!")
                continue

            mut = site.mutations[0]

            # Check if this mutation is a SLiM mutation
            is_slim_mut = (
                mut.metadata is not None
                and "mutation_list" in mut.metadata
                and len(mut.metadata["mutation_list"]) > 0
            )

            # Decide whether to keep the mutation
            keep_mutation = True
            if not is_slim_mut:
                for i in range(len(start)):
                    assert start[i] < end[i]
                    if start[i] <= site.position < end[i]:
                        keep_mutation = random.uniform(0, 1) > proportion
                        break

            if keep_mutation:
                site_id = tables.sites.add_row(
                    position=site.position,
                    ancestral_state=site.ancestral_state,
                )

                if is_slim_mut:
                    metadata = mut.metadata
                else:
                    metadata = {
                        "mutation_list": [
                            {
                                "mutation_type": 0,
                                "selection_coeff": 0.0,
                                "subpopulation": 0,
                                "slim_time": 1,
                                "nucleotide": -1,
                            }
                        ]
                    }

                tables.mutations.add_row(
                    site=site_id,
                    node=mut.node,
                    derived_state=mut.derived_state,
                    metadata=metadata,
                )

    return tables.tree_sequence()


def generate_synonymous_mutations(
    ts: tskit.TreeSequence,
    mu: float,
    start: List[float],
    end: List[float],
    scaling_factor_syn: float = 3.31,
    remove_prior_mutations: bool = True
) -> tskit.TreeSequence:
    """
    Generate synonymous mutations in a tree sequence within a specified genomic region.

    Parameters
    ----------
    ts : tskit.TreeSequence
        The input tree sequence representing ancestral recombination graphs.
    mu : float
        The original mutation rate per base per generation.
    start : list of float
        The list of start coordinates for existing genomic intervals.
    end : list of float
        The list of end coordinates for existing genomic intervals.
    scaling_factor_syn : float, optional
        Factor by which to reduce the original mutation rate to estimate synonymous mutations,
        by default 3.31.
    remove_prior_mutations : bool, optional
        Whether to remove prior mutations outside newly introduced synonymous mutation regions,
        by default True.

    Returns
    -------
    tskit.TreeSequence
        A new tree sequence with added synonymous mutations.
    """
    mu_syn = 1 / scaling_factor_syn * mu
    ts_syn_all = msprime.mutate(ts, rate=mu_syn, keep=False)
    first = ts_syn_all.first().interval[0]
    last = ts_syn_all.last().interval[1]
    new_start = [first] + end
    new_end = start + [last]
    if remove_prior_mutations:
        ts_syn = remove_mutations_non_slim(ts_syn_all, new_start, new_end, 1.0)
    else:
        ts_syn = ts_syn_all
    return ts_syn


def generate_noncoding_mutations(
    ts: tskit.TreeSequence,
    rate: float,
    start: List[float],
    end: List[float],
    remove_prior_mutations: bool = True
) -> tskit.TreeSequence:
    """
    Generate noncoding mutations in a tree sequence across specified genomic regions.

    Parameters
    ----------
    ts : tskit.TreeSequence
        The input tree sequence to which noncoding mutations will be added.
    rate : float
        The mutation rate per base per generation for noncoding regions.
    start : list of float
        The list of start coordinates for the regions where existing mutations may be removed.
    end : list of float
        The list of end coordinates corresponding to `start`, defining the regions.
    remove_prior_mutations : bool, optional
        Whether to remove mutations previously present in the specified regions,
        by default True.

    Returns
    -------
    tskit.TreeSequence
        A new tree sequence with noncoding mutations added.
    """
    ts_nc_all = msprime.mutate(ts, rate=rate, keep=False)
    if remove_prior_mutations:
        ts_nc = remove_mutations_non_slim(ts_nc_all, start, end, 1.0)
    else:
        ts_nc = ts_nc_all
    return ts_nc


def combine_mutations(
    ts_ns: tskit.TreeSequence,
    ts_syn: tskit.TreeSequence,
    ts_nc: tskit.TreeSequence
) -> Tuple[tskit.TreeSequence, Dict[int, str]]:
    """
    Combine nonsynonymous, synonymous, and noncoding mutations from separate tree sequences
    into a single tree sequence, annotating mutation types appropriately.

    Parameters
    ----------
    ts_ns : tskit.TreeSequence
        Tree sequence containing nonsynonymous mutations (typically from SLiM).
    ts_syn : tskit.TreeSequence
        Tree sequence containing synonymous mutations.
    ts_nc : tskit.TreeSequence
        Tree sequence containing noncoding mutations.

    Returns
    -------
    tuple of (tskit.TreeSequence, dict of int to str)
        - The combined tree sequence containing all mutations from the three sources.
        - A dictionary mapping site IDs to their mutation type labels ("NS", "SYN", "NC").

    Notes
    -----
    - Mutation metadata is modified or generated to include a `mutation_type` field:
        * Nonsynonymous: preserved from SLiM metadata if available.
        * Synonymous: assigned a new mutation type ID greater than the max existing one.
        * Noncoding: assigned another new mutation type ID.
    - Only sites with a single mutation are included; sites with multiple mutations are skipped.
    - The resulting site order is sorted by position, as required by `tskit`.
    """

    tables = ts_ns.dump_tables()
    tables.sites.clear()
    tables.mutations.clear()

    mut_type_map = {}

    # Detect all existing mutation_type values from SLiM
    existing_types = set()
    for site in ts_ns.sites():
        for mut in site.mutations:
            if mut.metadata and "mutation_list" in mut.metadata:
                for m in mut.metadata["mutation_list"]:
                    if "mutation_type" in m:
                        existing_types.add(m["mutation_type"])

    # unused mutation_type values for SYN and NC
    max_existing = max(existing_types) if existing_types else 0
    MUTATION_TYPE_SYN = max_existing + 1
    MUTATION_TYPE_NC = max_existing + 2

    # Step 3: Helper to assign metadata
    def get_metadata(original_metadata, label):
        if original_metadata and "mutation_list" in original_metadata:
            return copy.deepcopy(original_metadata)

        if label == "SYN":
            mut_type = MUTATION_TYPE_SYN
        elif label == "NC":
            mut_type = MUTATION_TYPE_NC
        else:
            mut_type = 0  # fallback

        return {
            "mutation_list": [
                {
                    "mutation_type": mut_type,
                    "selection_coeff": 0.0,
                    "subpopulation": 0,
                    "slim_time": 1,
                    "nucleotide": -1,
                }
            ]
        }

    # collect all mutation data
    all_sites = []

    def collect_sites(ts_part, label):
        for site in ts_part.sites():
            if len(site.mutations) != 1:
                continue  # Skip if multiple mutations at site
            mut = site.mutations[0]
            all_sites.append((site.position, site.ancestral_state, mut, label))

    collect_sites(ts_ns, "NS")
    collect_sites(ts_syn, "SYN")
    collect_sites(ts_nc, "NC")

    # sort by position (required by tskit)
    all_sites.sort(key=lambda x: x[0])

    # new tables
    for i, (position, ancestral_state, mut, label) in enumerate(all_sites):
        site_id = tables.sites.add_row(
            position=position, ancestral_state=ancestral_state
        )

        metadata = get_metadata(mut.metadata, label)

        tables.mutations.add_row(
            site=site_id,
            node=mut.node,
            derived_state=mut.derived_state,
            metadata=metadata,
        )

        mut_type_map[site_id] = label

    return tables.tree_sequence(), mut_type_map


def write_annotated_vcf(
    ts: tskit.TreeSequence,
    filename: str,
    mut_type_map: Dict[int, str],
    chrom: str = "1"
) -> None:
    """
    Writes an annotated VCF file from a tree sequence, adding mutation type annotations.

    Parameters
    ----------
    ts : tskit.TreeSequence
        The tree sequence to export as VCF.
    filename : str
        Base filename for output files. Two files will be written:
        - `{filename}_raw.vcf`: raw VCF output from the tree sequence.
        - `{filename}_combined.vcf`: annotated VCF with mutation types.
    mut_type_map : dict[int, str]
        A mapping from site position (0-based) to a mutation type string, used for annotation.
    chrom : str, optional
        Chromosome name used in the VCF (default: "1").

    Returns
    -------
    None

    Notes
    -----
    - Adds a custom INFO field (`TT`) to the VCF header and populates it per mutation site.
    - Assumes 1-based positions in the VCF (adjusted by subtracting 1 for indexing into `mut_type_map`).
    """
    raw_path = f"{filename}_raw.vcf"
    out_path = f"{filename}_combined.vcf"

    with open(raw_path, "w") as f:
        ts.write_vcf(f, contig_id=chrom,allow_position_zero=True)

    with open(raw_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            if line.startswith("##"):
                fout.write(line)
            elif line.startswith("#CHROM"):
                fout.write(
                    '##INFO=<ID=TT,Number=A,Type=String,Description="Annotation">\n'
                )
                fout.write(line)
            elif line.strip():
                fields = line.strip().split("\t")
                site_id = int(fields[1]) - 1  # VCF POS is 1-based
                tt = mut_type_map.get(site_id, "NA")
                fields[7] = f"TT={tt}"
                fout.write("\t".join(fields) + "\n")


def combine_mutation_trees(
    ts_sample: tskit.TreeSequence,
    mu: float,
    start: float,
    end: float,
    filename: Optional[str] = None,
    nc_mu: float = 1.8e-8,
    nc_scaling: float = 1.0,
    write_vcf: bool = False,
    save_trees: bool = False,
) -> tskit.TreeSequence:
    """
    Generates synonymous and noncoding mutations on a tree sequence and combines them.

    Parameters
    ----------
    ts_sample : tskit.TreeSequence
        The original sample tree sequence to which mutations will be added.
    mu : float
        The mutation rate for synonymous mutations (e.g., coding regions).
    start : float
        The start position (in genomic coordinates) for adding mutations.
    end : float
        The end position (in genomic coordinates) for adding mutations.
    filename : str, optional
        Base filename to use for output files (VCF and tree file). Required if `write_vcf` or `save_trees` is True.
    nc_mu : float, optional
        Mutation rate for noncoding regions (default is 1.8e-8).
    nc_scaling : float, optional
        Scaling factor applied to `nc_mu` (default is 1.0).
    write_vcf : bool, optional
        If True, writes the combined mutation tree sequence as a VCF file with annotations.
    save_trees : bool, optional
        If True, saves the combined tree sequence to disk with a `.trees` extension.

    Returns
    -------
    tskit.TreeSequence
        A new tree sequence with both synonymous and noncoding mutations added and combined.

    Raises
    ------
    ValueError
        If `write_vcf` or `save_trees` is True but `filename` is not provided.

    Notes
    -----
    - Assumes the presence of three external helper functions:
        `generate_synonymous_mutations`, `generate_noncoding_mutations`, and `combine_mutations`.
    - Writes output files only if `filename` is specified and the corresponding flags are True.
    """
    

    # syn mutations
    ts_syn = generate_synonymous_mutations(ts_sample, mu, start, end)

    # nc mutations
    ts_nc = generate_noncoding_mutations(ts_sample, nc_mu * nc_scaling, start, end)

    # all mutations
    ts_combined, mut_type_map = combine_mutations(ts_sample, ts_syn, ts_nc)

    if write_vcf and filename:
        write_annotated_vcf(ts_combined, filename, mut_type_map)

    if save_trees:
        tree_path = f"{filename}_combined.trees"
        ts_combined.dump(tree_path)

    return ts_combined


def only_add_maladapt_mutations(
    ts_sample: tskit.TreeSequence,
    mu: float,
    start: float,
    end: float,
    filename: Optional[str] = None,
    nc_mu: float = 1.8e-8,
    nc_scaling: float = 1.0,
    scaling_factor_syn: float = 3.31,
    write_vcf: bool = False,
    save_trees: bool = False,
    chromosome: str = "1",
) -> tskit.TreeSequence:
    """
    Adds maladaptive synonymous and noncoding mutations to a tree sequence without removing existing mutations.

    Parameters
    ----------
    ts_sample : tskit.TreeSequence
        The input tree sequence to which mutations will be added.
    mu : float
        Mutation rate for synonymous (possibly maladaptive) mutations.
    start : float
        Genomic start position for applying mutations.
    end : float
        Genomic end position for applying mutations.
    filename : str, optional
        Base filename for output files (required if `write_vcf` or `save_trees` is True).
    nc_mu : float, optional
        Mutation rate for noncoding regions (default: 1.8e-8).
    nc_scaling : float, optional
        Scaling multiplier applied to `nc_mu` (default: 1.0).
    scaling_factor_syn : float, optional
        Scaling factor for synonymous mutations, e.g., to represent deleterious pressure (default: 3.31).
    write_vcf : bool, optional
        Whether to write the resulting tree sequence to a VCF file.
    save_trees : bool, optional
        Whether to save the mutated tree sequence to disk.
    chromosome : str, optional
        Chromosome name to use in VCF output (default: "1").

    Returns
    -------
    tskit.TreeSequence
        The mutated tree sequence containing added maladaptive synonymous and noncoding mutations.

    Raises
    ------
    ValueError
        If `write_vcf` or `save_trees` is True and `filename` is not provided.

    Notes
    -----
    - This function **adds** mutations without removing prior ones.
    - Assumes the existence of `generate_synonymous_mutations` and `generate_noncoding_mutations`.
    """

    ts_sample = generate_synonymous_mutations(
        ts_sample,
        mu,
        start,
        end,
        scaling_factor_syn=scaling_factor_syn,
        remove_prior_mutations=False,
    )
    ts_sample = generate_noncoding_mutations(
        ts_sample, nc_mu * nc_scaling, start, end, remove_prior_mutations=False
    )

    if write_vcf and filename:
        with open(filename, "w") as f:
            ts_sample.write_vcf(f, contig_id=chromosome, allow_position_zero=True)

    if save_trees:
        tree_path = f"{filename}_added.trees"
        ts_sample.dump(tree_path)

    return ts_sample
