import msprime
import pyslim
import tskit
import random
import copy
import csv
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Union, Set, DefaultDict


def _create_ref_tgt_file_slim(
    ref_ind_file: str,
    tgt_ind_file: str,
    all_p_ref: List[int],
    all_p_tgt: List[int],
    pops: Dict[str, str],
    identifier: str = "tsk_",
    src_ind_file: Optional[str] = None,
    all_p_src: Optional[List[int]] = None,
) -> None:
    """
    Writes individual population assignment files for use in SLiM simulations.

    Parameters
    ----------
    ref_ind_file : str
        File path to write reference population individual IDs.
    tgt_ind_file : str
        File path to write target population individual IDs.
    all_p_ref : list of int
        List of reference population individual indices.
    all_p_tgt : list of int
        List of target population individual indices.
    pops : dict of str
        Dictionary mapping population roles ("ref", "tgt", optionally "src")
        to population IDs used in SLiM.
    identifier : str, optional
        Prefix string to prepend to individual IDs (default is "tsk_").
    src_ind_file : str, optional
        File path to write source population individual IDs (if "src" in pops).
    all_p_src : list of int, optional
        List of source population individual indices (only used if "src" in pops).

    Returns
    -------
    None
        Writes files to disk, no return value.
    """

    with open(ref_ind_file, "w") as fref:
        for ind in all_p_ref:
            fref.write(f"{pops['ref']}\t{identifier}{ind}\n")

    with open(tgt_ind_file, "w") as ftgt:
        for ind in all_p_tgt:
            ftgt.write(f"{pops['tgt']}\t{identifier}{ind}\n")

    if "src" in pops:
        with open(src_ind_file, "w") as fsrc:
            for ind in all_p_src:
                fsrc.write(f"{pops['src']}\t{identifier}{ind}\n")


def sample_and_simplify(
    ts: tskit.TreeSequence,
    pops: Dict[str, str],
    sample_sizes: Dict[str, int],
    archaic_sample_time: Optional[
        Union[
            int,
            float,
            List[Union[int, float]],
            Tuple[Union[int, float], Union[int, float]],
            str,
        ]
    ] = None,
    seed: Optional[int] = None,
) -> Tuple[tskit.TreeSequence, List[int], List[int], List[int], List[int]]:
    """
    Simplifies a SLiM-generated tree sequence by sampling individuals from specified
    populations and optionally filtering archaic individuals by sampling time.

    Parameters
    ----------
    ts : tskit.TreeSequence
        A tskit TreeSequence object from SLiM with metadata.
    pops : dict of str to str
        Mapping of logical population labels ('ref', 'src', 'tgt') to SLiM population names (e.g., "p1", "p2", "p4").
    sample_sizes : dict of str to int
        Number of individuals to sample from each population, keyed by 'ref', 'src', 'tgt'.
    archaic_sample_time : int, float, list, tuple, or str, optional
        Time or condition for selecting archaic source individuals. Can be:
            - an exact time (int or float),
            - a list of times,
            - a (min_time, max_time) tuple,
            - "youngest" to select most recent archaic individuals,
            - "oldest" to select most ancient archaic individuals,
            - or None to include all.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    tuple
        A 5-tuple containing:
            - ts_simplified : tskit.TreeSequence
                The simplified tree sequence.
            - all_ps_simplified : list of int
                List of all individual IDs retained across populations.
            - all_p_ref : list of int
                Sampled reference population individual IDs.
            - all_p_tgt : list of int
                Sampled target population individual IDs.
            - all_p_src : list of int
                Sampled source population individual IDs.

    Raises
    ------
    ValueError
        If the requested sample size exceeds available individuals in any population.

    Notes
    -----
    - This method uses SLiM and pyslim-specific metadata and flags.
    - Archaic individuals are sampled based on node times according to the
    `archaic_sample_time` specification.
    - The simplified TreeSequence retains selected individuals and nodes across populations.
    """
    if seed is not None:
        random.seed(seed)

    pops_ids = copy.deepcopy(pops)
    for pop in pops:

        pop_value = pops[pop]

        pop_id = get_pop_id_by_name(ts, pop_value)

        pops_ids[pop] = pop_id

    indivs = list(ts.individuals())

    sample_node_ids = ts.samples()

    # Get the set of individual ids associated with sample nodes
    sample_indiv_ids = {
        ts.node(n).individual
        for n in sample_node_ids
        if ts.node(n).individual != tskit.NULL
    }

    # Now filter the individuals list
    sample_indivs = [ind for ind in ts.individuals() if ind.id in sample_indiv_ids]

    indivs = sample_indivs

    # Get alive individuals in pop 1 / ref and 4 / tgt
    alive_p_ref = [
        ind.id
        for ind in indivs
        if ind.population == pops_ids["ref"] and ind.flags & pyslim.INDIVIDUAL_ALIVE
    ]
    alive_p_tgt = [
        ind.id
        for ind in indivs
        if ind.population == pops_ids["tgt"] and ind.flags & pyslim.INDIVIDUAL_ALIVE
    ]

    if not archaic_sample_time:
        # no time filtering:  get all individuals from the source population
        p_src = [ind.id for ind in indivs if ind.population == pops_ids["src"]]

    elif isinstance(archaic_sample_time, list):
        # archaic_sample_time is a list of specific times
        p_src = [
            ind.id
            for ind in indivs
            if ind.population == pops_ids["src"]
            and any(ts.node(n).time in archaic_sample_time for n in ind.nodes)
        ]

    elif isinstance(archaic_sample_time, tuple) and len(archaic_sample_time) == 2:
        # archaic_sample_time is a (min_time, max_time) tuple
        min_time, max_time = archaic_sample_time
        p_src = [
            ind.id
            for ind in indivs
            if ind.population == pops_ids["src"]
            and any(min_time <= ts.node(n).time <= max_time for n in ind.nodes)
        ]

    elif archaic_sample_time == "youngest":
        # Youngest individuals
        source_inds = [ind for ind in indivs if ind.population == pops_ids["src"]]

        # Get the minimum node time
        ind_min_times = {
            ind.id: min(ts.node(n).time for n in ind.nodes) for ind in source_inds
        }

        min_time = min(ind_min_times.values())

        p_src = [ind_id for ind_id, t in ind_min_times.items() if t == min_time]

    elif archaic_sample_time == "oldest":
        # Oldest individuals
        source_inds = [ind for ind in indivs if ind.population == pops_ids["src"]]

        # Get the minimum node time
        ind_min_times = {
            ind.id: min(ts.node(n).time for n in ind.nodes) for ind in source_inds
        }

        max_time = max(ind_min_times.values())

        p_src = [ind_id for ind_id, t in ind_min_times.items() if t == max_time]

    # if integer
    else:

        # default: archaic_sample_time is a single time, given as integer
        p_src = [
            ind.id
            for ind in indivs
            if ind.population == pops_ids["src"]
            and any(ts.node(n).time == archaic_sample_time for n in ind.nodes)
        ]

    if (
        sample_sizes["ref"] > len(alive_p_ref)
        or sample_sizes["tgt"] > len(alive_p_tgt)
        or sample_sizes["src"] > len(p_src)
    ):
        raise ValueError("Sample size exceeds available individuals.")

    sample_p_ref = random.sample(alive_p_ref, sample_sizes["ref"])
    sample_p_tgt = random.sample(alive_p_tgt, sample_sizes["tgt"])

    if len(sample_sizes) > 2:

        sample_p_src = random.sample(p_src, sample_sizes["src"])
    else:
        sample_p_src = p_src

    sample_p_ref.sort()
    sample_p_tgt.sort()
    sample_p_src.sort()

    all_individuals = sample_p_ref + sample_p_tgt + sample_p_src

    # Get nodes from sampled individuals
    nodes_p_ref = [n for ind in indivs if ind.id in sample_p_ref for n in ind.nodes]
    nodes_p_tgt = [n for ind in indivs if ind.id in sample_p_tgt for n in ind.nodes]

    nodes_p_src = [n for ind in indivs if ind.id in sample_p_src for n in ind.nodes]

    # Combine all sampled nodes and simplify
    all_sampled_nodes = nodes_p_ref + nodes_p_tgt + nodes_p_src

    ts_simplified = ts.simplify(
        samples=all_sampled_nodes,
        keep_unary=True,
        keep_input_roots=True,
        filter_populations=False,
        filter_individuals=True,
        filter_nodes=True,
    )

    indivs_simplified = list(ts_simplified.individuals())

    sample_node_ids_simplified = ts_simplified.samples()

    # Get the set of individual ids associated with sample nodes
    sample_indiv_ids_simplified = {
        ts_simplified.node(n).individual
        for n in sample_node_ids_simplified
        if ts_simplified.node(n).individual != tskit.NULL
    }

    # filter the individuals list
    sample_indivs_simplified = [
        ind
        for ind in ts_simplified.individuals()
        if ind.id in sample_indiv_ids_simplified
    ]

    indivs_simplified = sample_indivs_simplified

    all_p_ref = [
        ind.id
        for ind in indivs_simplified
        if ind.population == pops_ids["ref"] and ind.flags
    ]
    all_p_src = [
        ind.id
        for ind in indivs_simplified
        if ind.population == pops_ids["src"] and ind.flags
    ]
    all_p_tgt = [
        ind.id
        for ind in indivs_simplified
        if ind.population == pops_ids["tgt"] and ind.flags
    ]

    all_nodes_p_ref = [
        n for ind in indivs_simplified if ind.id in all_p_ref for n in ind.nodes
    ]
    all_nodes_p_src = [
        n for ind in indivs_simplified if ind.id in all_p_src for n in ind.nodes
    ]
    all_nodes_p_tgt = [
        n for ind in indivs_simplified if ind.id in all_p_tgt for n in ind.nodes
    ]

    all_ps_simplified = all_p_ref + all_p_tgt + all_p_src

    all_nodes_simplified = all_nodes_p_ref + all_nodes_p_tgt + all_nodes_p_src

    return ts_simplified, all_ps_simplified, all_p_ref, all_p_tgt, all_p_src


def count_individuals(ts: tskit.TreeSequence) -> Dict[int, int]:
    """
    Counts the number of individuals per population in a tree sequence.

    Parameters
    ----------
    ts : tskit.TreeSequence
        The tree sequence containing individuals and population metadata.

    Returns
    -------
    dict of int to int
        A dictionary mapping population IDs to the number of individuals
        associated with each population.

    Notes
    -----
    - An individual is counted only if it has one or more associated nodes.
    - The function prints the population name (if available in metadata) and count.
    - Population names are retrieved from metadata under the "name" key.
    """
    population_counts = {}
    for ind in ts.individuals():
        if len(ind.nodes) > 0:
            pop_id = ts.node(ind.nodes[0]).population
            population_counts[pop_id] = population_counts.get(pop_id, 0) + 1

    print("Individuals per population:")
    for pop_id, count in population_counts.items():
        pop = ts.population(pop_id)
        if isinstance(pop.metadata, dict):
            name = pop.metadata.get("name", f"Population {pop_id}")
        else:
            name = f"Population {pop_id}"
        print(f"  {name}: {count}")

    return population_counts


def get_pop_id_by_name(ts: tskit.TreeSequence, name: str) -> Optional[int]:
    """
    Retrieves the population ID corresponding to a given population name.

    Parameters
    ----------
    ts : tskit.TreeSequence
        The tree sequence containing population metadata.
    name : str
        The name of the population to look up.

    Returns
    -------
    int or None
        The integer ID of the population if found, otherwise None.

    Notes
    -----
    This function assumes that population metadata is a dictionary and contains
    a `"name"` field that matches the input `name`.

    If multiple populations have the same name, the first match will be returned.
    """
    for pop in ts.populations():
        if pop.metadata and isinstance(pop.metadata, dict):
            if pop.metadata.get("name") == name:
                return pop.id
    return None


def get_inds_by_pop(ts, pop_id):
    return [ind for ind in ts.individuals() if ind.population == pop_id]


def get_inds_ids_by_pop(ts, pop_id):
    return [ind.id for ind in ts.individuals() if ind.population == pop_id]


def sample_times(
    ts: tskit.TreeSequence, individuals: List[tskit.Individual]
) -> List[float]:
    """
    Computes the sampling time for each individual based on the times of their associated nodes.

    Parameters
    ----------
    ts : tskit.TreeSequence
        The tree sequence containing the individuals and node information.
    individuals : list of tskit.Individual
        A list of individuals whose sample times are to be computed.

    Returns
    -------
    list of float
        A list of sampling times (in generations before present), one per individual.
        Each time is calculated as the maximum of the associated node times.

    Notes
    -----
    Node times typically represent the time before present in generations.
    The method currently uses the maximum node time per individual, but this can
    be modified to use the minimum or average depending on analysis needs.
    """
    individual_times = []
    for one_ind in individuals:

        # individual
        individual = ts.individual(one_ind.id)

        # Get times of their nodes
        node_times = [ts.node(n).time for n in individual.nodes]

        #  max, min, or average time
        individual_time = max(node_times)  # or min(node_times), depending on use case

        individual_times.append(individual_time)

    return individual_times


def simplify_by_population(
    ts: tskit.TreeSequence,
    pop_names_to_keep: List[str],
    filter_populations: bool = False,
    filter_individuals: bool = False,
    filter_sites: bool = False,
    filter_nodes: bool = False,
) -> tskit.TreeSequence:
    """
    Simplifies a tree sequence by retaining only samples from specified populations.

    Parameters
    ----------
    ts : tskit.TreeSequence
        The input tree sequence to simplify.
    pop_names_to_keep : list of str
        Names of the populations whose sample nodes should be retained.
    filter_populations : bool, optional
        Whether to remove populations not referenced by retained nodes (default: False).
    filter_individuals : bool, optional
        Whether to remove individuals not referenced by retained nodes (default: False).
    filter_sites : bool, optional
        Whether to remove sites that are not ancestral to retained nodes (default: False).
    filter_nodes : bool, optional
        Whether to remove nodes that are not ancestral to retained nodes (default: False).

    Returns
    -------
    tskit.TreeSequence
        A simplified tree sequence containing only the specified population samples.

    Notes
    -----
    This method uses `get_pop_id_by_name` to resolve population names to numeric IDs.
    Only sample nodes from the specified populations are retained in the simplified sequence.
    """

    populations_to_keep = []
    for pop_name in pop_names_to_keep:
        populations_to_keep.append(get_pop_id_by_name(ts, pop_name))

    nodes_to_keep = [
        n.id
        for n in ts.nodes()
        if ts.node(n.id).population in populations_to_keep and n.is_sample()
    ]

    ts_simplified = ts.simplify(
        nodes_to_keep,
        filter_populations=filter_populations,
        filter_individuals=filter_individuals,
        filter_sites=filter_sites,
        filter_nodes=filter_nodes,
    )

    return ts_simplified


def simplify_by_source_target(
    ts: tskit.TreeSequence,
    source_pop_name: str = "p2",
    target_pop_name: str = "p4",
) -> tskit.TreeSequence:
    """
    Simplifies a tree sequence to retain only samples from the source and target populations.

    Parameters
    ----------
    ts : tskit.TreeSequence
        The input tree sequence to simplify.
    source_pop_name : str, optional
        The name of the source population to retain (default is "p2").
    target_pop_name : str, optional
        The name of the target population to retain (default is "p4").

    Returns
    -------
    tskit.TreeSequence
        A simplified tree sequence containing only nodes from the source and target populations.
        All populations, individuals, sites, and nodes are retained in metadata even if unused.

    """
    pop_names_to_keep = [source_pop_name, target_pop_name]
    populations_to_keep = []
    for pop_name in pop_names_to_keep:
        populations_to_keep.append(get_pop_id_by_name(ts, pop_name))

    nodes_to_keep = [
        n.id
        for n in ts.nodes()
        if ts.node(n.id).population in populations_to_keep and n.is_sample()
    ]

    ts_simplified = ts.simplify(
        nodes_to_keep,
        filter_populations=False,
        filter_individuals=False,
        filter_sites=False,
        filter_nodes=False,
    )

    return ts_simplified


def get_individuals_with_mutations(
    ts: tskit.TreeSequence,
    target_mut_type_ids: Set[int] = {2},
    target_pop_name: str = "p4",
    target_pop_id: Optional[int] = None,
    only_alive: bool = True,
    print_results: bool = False,
) -> Dict[int, Dict[int, List[Tuple[float, int]]]]:
    """
    Identifies individuals in a population who carry specific mutation types.

    Parameters
    ----------
    ts : tskit.TreeSequence
        The tree sequence containing individuals and mutations.
    target_mut_type_ids : set of int, optional
        Mutation type IDs to search for (default is {2}).
    target_pop_name : str, optional
        Population name to search in, used if `target_pop_id` is not provided (default is "p4").
    target_pop_id : int, optional
        Population ID to filter individuals (if provided, `target_pop_name` is ignored).
    only_alive : bool, optional
        Whether to restrict to individuals marked as alive (`pyslim.INDIVIDUAL_ALIVE`) (default is True).
    print_results : bool, optional
        If True, prints the individuals and mutations found (default is False).

    Returns
    -------
    indiv_hap_mut_data : dict of dict
        A nested dictionary of the form:
        {
            individual_id: {
                haplotype_index: [(mutation_position, mutation_type), ...]
            }
        }

    Raises
    ------
    ValueError
        If no population name or ID matches are found in the tree sequence.

    Notes
    -----
    The function scans mutations in the tree sequence and maps them to individuals
    in the target population if their haplotype nodes are descendants of the node
    carrying the mutation.

    The `haplotype_index` corresponds to the 0 or 1 chromosome/haplotype of a diploid individual.
    """

    # Get population ID if not provided
    if target_pop_id is None and target_pop_name:
        for pop in ts.populations():
            if pop.metadata and isinstance(pop.metadata, dict):
                if pop.metadata.get("name", "") == target_pop_name:
                    target_pop_id = pop.id
                    break

    # Get all alive individuals in the target population
    if only_alive:
        found_inds = [
            ind
            for ind in ts.individuals()
            if ind.population == target_pop_id and ind.flags & pyslim.INDIVIDUAL_ALIVE
        ]
    else:
        found_inds = [
            ind for ind in ts.individuals() if ind.population == target_pop_id
        ]

    # Collect relevant mutations by type
    matching_mutations = []
    for mut in ts.mutations():
        if mut.metadata and "mutation_list" in mut.metadata:
            for m in mut.metadata["mutation_list"]:
                if m["mutation_type"] in target_mut_type_ids:
                    matching_mutations.append((mut, m["mutation_type"]))
                    break

    # Map of individual ID -> hap_idx -> list of (position, mutation_type)
    indiv_hap_mut_data = {}

    for mut, mut_type in matching_mutations:
        mutation_node = mut.node
        site_pos = ts.site(mut.site).position
        tree = ts.at(site_pos)

        for ind in found_inds:
            for hap_idx, node in enumerate(ind.nodes):
                if tree.is_descendant(node, mutation_node):
                    if ind.id not in indiv_hap_mut_data:
                        indiv_hap_mut_data[ind.id] = {}
                    if hap_idx not in indiv_hap_mut_data[ind.id]:
                        indiv_hap_mut_data[ind.id][hap_idx] = []
                    indiv_hap_mut_data[ind.id][hap_idx].append((site_pos, mut_type))

    # Optional result printing
    if print_results:
        print(
            f"\nAlive individuals in population '{target_pop_name}' (ID={target_pop_id}) with mutations of type(s) {target_mut_type_ids}:"
        )
        for ind_id in sorted(indiv_hap_mut_data):
            ind = ts.individual(ind_id)
            pop_id = ind.population
            pop_name = (
                ts.population(pop_id).metadata.get("name", f"ID={pop_id}")
                if ts.population(pop_id).metadata
                else f"ID={pop_id}"
            )

            for hap_idx, mut_list in sorted(indiv_hap_mut_data[ind_id].items()):
                pos_str = ", ".join(
                    f"{p:.1f} (type {mt})" for p, mt in sorted(mut_list)
                )
                print(
                    f" - Individual {ind_id}, Haplotype {hap_idx} | Population: {pop_name} | Mutations at: {pos_str}"
                )

    return indiv_hap_mut_data


def get_introgressed_segments(
    ts: tskit.TreeSequence,
    source_pop_name: str = "p2",
    target_pop_name: str = "p4",
    print_results: bool = False,
) -> DefaultDict[int, DefaultDict[int, List[Tuple[float, float]]]]:
    """
    Identifies and returns genomic segments in the target population that have
    been introgressed from a source population in a tree sequence.

    Parameters
    ----------
    ts : tskit.TreeSequence
        A tree sequence object with population and individual metadata.
    source_pop_name : str, optional
        The name of the source population (default is "p2").
    target_pop_name : str, optional
        The name of the target population (default is "p4").
    print_results : bool, optional
        Whether to print the results to stdout (default is False).

    Returns
    -------
    merged_introgression : defaultdict
        Nested dictionary where:
            - keys are individual IDs from the target population,
            - values are dictionaries mapping haplotype index (0 or 1)
            to a list of (start, end) tuples indicating introgressed genomic segments.

    Raises
    ------
    ValueError
        If the source or target population cannot be found by name.

    Notes
    -----
    A genomic segment is considered introgressed if, in any tree, a sample node
    from the target population descends from a parent node in the source population.
    This is assessed per haplotype (node) of each individual.

    Contiguous or overlapping introgressed segments are merged for clarity.
    """

    # Get population ID by name
    def get_pop_id_by_name(ts, name):
        for pop in ts.populations():
            if pop.metadata and isinstance(pop.metadata, dict):
                if pop.metadata.get("name") == name:
                    return pop.id
        return None

    source_pop_id = get_pop_id_by_name(ts, source_pop_name)
    target_pop_id = get_pop_id_by_name(ts, target_pop_name)

    if source_pop_id is None or target_pop_id is None:
        raise ValueError("Could not find source or target population by name.")

    # Precompute node populations
    node_pop = [node.population for node in ts.nodes()]
    sample_set = set(ts.samples())

    # Precompute target individuals
    target_inds = [
        (ind.id, ind.nodes)
        for ind in ts.individuals()
        if ind.population == target_pop_id
    ]

    # Store introgressed segments: [ind_id][hap_index] = list of (start, end)
    introgressed_segments = defaultdict(lambda: defaultdict(list))

    for tree in ts.trees():
        start, end = tree.interval
        get_parent = tree.parent

        for ind_id, nodes in target_inds:
            for hap_index, node_id in enumerate(nodes):
                if node_id not in sample_set:
                    continue

                current = node_id
                while True:
                    parent = get_parent(current)
                    if parent == tskit.NULL:
                        break

                    if (
                        node_pop[current] == target_pop_id
                        and node_pop[parent] == source_pop_id
                    ):
                        introgressed_segments[ind_id][hap_index].append((start, end))
                        break

                    current = parent

    # Merge contiguous/overlapping intervals
    def merge_intervals(intervals):
        if not intervals:
            return []
        intervals.sort()
        merged = []
        current_start, current_end = intervals[0]
        for start, end in intervals[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        merged.append((current_start, current_end))
        return merged

    # Merge results
    merged_introgression = defaultdict(lambda: defaultdict(list))
    for ind_id, hap_data in introgressed_segments.items():
        for hap_index, intervals in hap_data.items():
            merged_introgression[ind_id][hap_index] = (
                merge_intervals(intervals) if len(intervals) > 1 else intervals
            )

    if print_results:
        print(
            f"\n--- Introgressed segments from {source_pop_name} into {target_pop_name} ---"
        )
        for ind_id, hap_data in merged_introgression.items():
            print(f"Individual {ind_id}:")
            for hap, segs in hap_data.items():
                if segs:
                    for start, end in segs:
                        print(f"  Haplotype {hap}: {start:.1f} - {end:.1f}")
                else:
                    print(f"  Haplotype {hap}: No introgressed segments detected.")

        return merged_introgression


def write_introgressed_segments_to_tsv(merged_introgression, output_path, chrom="1"):
    """
    Writes merged introgressed segments to a tab-separated file.

    Parameters:
    - merged_introgression: dict[individual_id][haplotype] -> list of (start, end)
    - chrom: int, chromosome number
    - output_path: str, path to the output TSV file
    """
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for ind_id, hap_data in merged_introgression.items():
            for hap_index, intervals in hap_data.items():
                label = f"{ind_id}_{hap_index}"
                for start, end in intervals:
                    writer.writerow([chrom, int(start), int(end), label])


def write_filtered_mutation_sites_to_tsv(
    phased_mutations,
    output_path,
    target_mut_types=None,
    target_positions=None,
    chrom="1",
):
    """
    Writes filtered mutation sites to a tab-separated file.

    Parameters:
    - phased_mutations: dict[individual_id][haplotype] -> list of (position, mutation_type)
    - chrom: int, chromosome number to write in the output
    - output_path: str, path to the output TSV file
    - target_mut_types: optional set of ints. Only mutations of these types will be written.
    - target_positions: optional set of floats or ints. Only mutations at these positions will be written.
    """
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for ind_id, hap_data in phased_mutations.items():
            for hap_index, mut_list in hap_data.items():
                label = f"{ind_id}_{hap_index}"
                for pos, mut_type in mut_list:
                    if (
                        target_mut_types is not None
                        and mut_type not in target_mut_types
                    ):
                        continue
                    if target_positions is not None and pos not in target_positions:
                        continue
                    writer.writerow([chrom, int(pos), mut_type, label])
