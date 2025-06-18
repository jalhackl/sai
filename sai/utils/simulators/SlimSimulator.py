# Copyright 2024 Xin Huang
#
# GNU General Public License v3.0
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, please see
#
#    https://www.gnu.org/licenses/gpl-3.0.en.html


import demes
import msprime
import os
import tskit
from copy import deepcopy
from sai.utils.simulators import DataSimulator
import random
import subprocess
import csv
from collections import defaultdict
from typing import Dict, Any, List, Optional, Union, Tuple, DefaultDict, Set

from sai.utils.simulators.slim_simulator_utils import *


class SlimSimulator(DataSimulator):
    """
    A simulator for generating genetic data using SLiM within a tskit-based workflow.

    This subclass of `DataSimulator` configures parameters specific to SLiM simulations,
    particularly for maladaptive or uniform selection models. It manages simulation output,
    mutation parameters, and metadata relevant to downstream genetic analysis.

    Parameters
    ----------
    nref : int, optional
        Number of reference individuals, by default 108.
    ntgt : int, optional
        Number of target individuals, by default 99.
    ref_id : str, optional
        Population ID for the reference population, by default "p1".
    tgt_id : str, optional
        Population ID for the target population, by default "p4".
    src_id : str, optional
        Population ID for the source population, by default "p2".
    seq_len : int, optional
        Length of the simulated genomic sequence, by default 5,000,000.
    output_prefix : str, optional
        Prefix for output files, by default "slim_sim".
    output_dir : str, optional
        Directory to store simulation results, by default "slim_res".
    is_phased : bool, optional
        Whether output data should be phased, by default True.
    scaling_factor : int or float, optional
        Factor used to scale mutation rates, by default 10.
    basic_mut_rate : float, optional
        Base mutation rate before scaling, by default 1.5e-8.
    archaic_sample_time : int, optional
        Time in generations for archaic sampling, by default 152.
    nsrc : int, optional
        Number of source population individuals, by default 2.
    out_id : str or None, optional
        Population ID for any additional output group, by default None.
    nout : int or None, optional
        Number of individuals in the output group, by default None.
    create_tracts_file : bool, optional
        Whether to generate a tracts file, by default False.
    create_mutation_check_file : bool, optional
        Whether to generate a mutation check file, by default True.
    create_all_mutation_output : bool, optional
        Whether to output all mutation data, by default False.
    slim_mutation_nr : int, optional
        Number of SLiM mutation types, by default 2.
    maladapt_mutations_preprocess : bool, optional
        Whether to apply preprocessing for maladaptive mutations, by default True.
    extra_mutations : bool, optional
        Whether to add extra mutations in postprocessing, by default False.
    settings : str, optional
        Simulation scenario to use, "maladapt" or "uniform", by default "maladapt".
    resample : int, optional
        Whether and how often to resample the SLiM simulations, by default 0.
    chromosome : str, optional
        Chromosome name used in simulation setup, by default "1".

    Raises
    ------
    Exception
        If the provided settings are not "maladapt" or "uniform".
    """

    def __init__(
        self,
        nref: int = 108,
        ntgt: int = 99,
        ref_id: str = "p1",
        tgt_id: str = "p4",
        src_id: str = "p2",
        seq_len: int = 5000000,
        output_prefix: str = "slim_sim",
        output_dir: str = "slim_res",
        is_phased: bool = True,
        scaling_factor=10,
        basic_mut_rate=1.5e-8,
        archaic_sample_time: int = 152,
        nsrc: int = 2,
        out_id: str = None,
        nout: int = None,
        create_tracts_file: bool = False,
        create_mutation_check_file: bool = True,
        create_all_mutation_output: bool = False,
        slim_mutation_nr: int = 2,
        maladapt_mutations_preprocess: bool = True,
        extra_mutations: bool = False,
        settings: str = "maladapt",
        resample: int = 0,
        chromosome: str = "1",
    ):

        super().__init__(
            demo_model_file=None,
            nref=nref,
            ntgt=ntgt,
            ref_id=ref_id,
            tgt_id=tgt_id,
            src_id=src_id,
            ploidy=None,
            seq_len=seq_len,
            mut_rate=None,
            rec_rate=None,
            output_prefix=output_prefix,
            output_dir=output_dir,
        )
        self.test_tgt_id = self.tgt_id
        self.is_phased = is_phased

        self.nsrc = nsrc
        self.out_id = out_id
        self.nout = nout

        self.settings = settings

        if self.settings == "maladapt":
            self.dominances = ["recessive", "additive", "partial"]
            self.filenames = {
                "recessive": "maladapt_repl_adaptive_recessive_exon_repl.slim",
                "additive": "maladapt_repl_adaptive_additive_w_muteff_exons_original.slim",
                "partial": "maladapt_repl_adaptive_partial_plus_muteff_exons_repl.slim",
            }
            self.slim_script_folder = os.path.join("examples", "slim", "maladapt")
        elif self.settings == "uniform":
            self.dominances = ["uniform"]
            self.filenames = {
                "uniform": "simpler_adapt_v1_plus_params.slim",
            }
            self.slim_script_folder = os.path.join("examples", "slim")
        else:
            raise Exception(
                "Model settings currently not supported (only maladapt and uniform)!"
            )

        self.create_tracts_file = create_tracts_file
        self.create_mutation_check_file = create_mutation_check_file
        self.create_all_mutation_output = create_all_mutation_output

        self.identifier = "tsk_"

        self.scaling_factor = scaling_factor
        self.basic_mut_rate = basic_mut_rate
        self.mu = self.basic_mut_rate * self.scaling_factor

        self.slim_mutation_nr = slim_mutation_nr

        self.extra_mutations = extra_mutations
        self.maladapt_mutations_preprocess = maladapt_mutations_preprocess

        self.archaic_sample_time = archaic_sample_time
        self.pops = {"ref": ref_id, "src": src_id, "tgt": tgt_id}
        self.sample_sizes = {"ref": nref, "src": nsrc, "tgt": ntgt}

        self.seq_len = seq_len

        self.exon_file = os.path.join(
            self.slim_script_folder, "sim_seq_info_chr3region.txt"
        )

        self.resample = resample

        self.chromosome = chromosome

    def run(self, rep: int = None, seed: int = None) -> list[dict[str, str]]:
        """
        Executes the simulation with optional runtime arguments.

        Outputs multiple files including simulation results and metadata.

        Parameters
        ----------
        rep : int or None
            Used to specify the replicate number for the simulation. This attribute is not set
            in the constructor but should be assigned before running simulations that require
            tracking or distinguishing between multiple replicates.
        seed : int or None
            Seed for the random number generator to ensure reproducibility of the simulations.
            Similar to `rep`, this is not directly set in the constructor but should be specified
            to ensure that simulations can be reproduced exactly.

        Returns
        -------
        list[dict[str, str]]
            A list of a dictionary containing file paths for the simulated data.

        """
        output_dir = (
            self.output_dir if rep is None else os.path.join(self.output_dir, str(rep))
        )
        output_prefix = (
            self.output_prefix if rep is None else f"{self.output_prefix}.{rep}"
        )

        adm_range = [0.01, 0.02, 0.05, 0.1]

        dominance = random.choice(self.dominances)
        adm_time = random.choice(range(8697, 8747))
        adm_amount = random.choice(adm_range)
        sel_time = random.choice(range(0, 61))
        slim_script_name = self.filenames[dominance]

        slim_script = os.path.join(self.slim_script_folder, slim_script_name)

        os.makedirs(output_dir, exist_ok=True)
        ts_file = os.path.join(output_dir, f"{output_prefix}_{dominance}.ts")
        txt_file = os.path.join(output_dir, f"{output_prefix}_{dominance}.txt")
        vcf_file = os.path.join(output_dir, f"{output_prefix}_{dominance}.vcf")
        bed_file = os.path.join(output_dir, f"{output_prefix}.true.tracts.bed")
        ref_ind_file = os.path.join(output_dir, f"{output_prefix}.ref.ind.list")
        tgt_ind_file = os.path.join(output_dir, f"{output_prefix}.tgt.ind.list")

        if self.nsrc is not None and self.nsrc > 0:
            src_ind_file = os.path.join(output_dir, f"{output_prefix}.src.ind.list")
        else:
            src_ind_file = None
        if self.nout is not None and self.nout > 0:
            out_ind_file = os.path.join(output_dir, f"{output_prefix}.out.ind.list")
        else:
            out_ind_file = None

        mut_tgt_file = os.path.join(output_dir, f"{output_prefix}.tgt.mut.list")

        file_paths = {
            "ts_file": ts_file,
            "vcf_file": vcf_file,
            "bed_file": bed_file,
            "ref_ind_file": ref_ind_file,
            "tgt_ind_file": tgt_ind_file,
            "src_ind_file": src_ind_file,
            "txt_file": txt_file,
        }
        if self.create_mutation_check_file:
            file_paths["mut_file"] = mut_tgt_file

        all_file_paths = []
        if self.resample == 0:
            all_file_paths.append(file_paths)

        simulation = self.perform_slim_simulation(
            file_paths, slim_script, adm_time, sel_time, adm_amount, dominance, rep
        )

        ts_path = simulation["output_file"]
        ts = tskit.load(ts_path)

        # get simuation tree sequence (ts)

        if self.resample > 0:
            print("currently not implemented!")
            pass

        # simplify and sample part
        ts_simplified, all_ps_simplified, all_p_ref, all_p_tgt, all_p_src = (
            self.sample_and_simplify(
                ts,
                pops=self.pops,
                sample_sizes=self.sample_sizes,
                archaic_sample_time=self.archaic_sample_time,
                seed=seed,
            )
        )

        if self.maladapt_mutations_preprocess:
            start, end = self.read_exon_file(self.exon_file)
            if not self.extra_mutations:
                ts_simplified = combine_mutation_trees(
                    ts_simplified,
                    self.mu,
                    start,
                    end,
                    filename=None,
                    write_vcf=False,
                    save_trees=False,
                )
            else:
                ts_simplified = only_add_maladapt_mutations(
                    ts_sample=ts_simplified,
                    mu=self.mu,
                    start=start,
                    end=end,
                    filename=None,
                )

        if self.extra_mutations and not self.maladapt_mutations_preprocess:
            ts_simplified = msprime.mutate(ts_simplified, rate=self.mu, keep=True)

        # write vcf
        with open(file_paths["vcf_file"], "w") as vcf_file:
            ts_simplified.write_vcf(
                vcf_file,
                individuals=all_ps_simplified,
                individual_names=[self.identifier + str(x) for x in all_ps_simplified],
                allow_position_zero=True
            )

        # create sample files
        self._create_ref_tgt_file_slim(
            ref_ind_file=file_paths["ref_ind_file"],
            tgt_ind_file=file_paths["tgt_ind_file"],
            all_p_ref=all_p_ref,
            all_p_tgt=all_p_tgt,
            pops=self.pops,
            identifier="tsk_",
            src_ind_file=file_paths["src_ind_file"],
            all_p_src=all_p_src,
        )

        if self.create_tracts_file:
            ts_only_src_tgt = self.simplify_by_source_target(
                ts_simplified, source_pop_name=self.src_id, target_pop_name=self.tgt_id
            )
            fragments = self.get_introgressed_segments(
                ts_only_src_tgt,
                source_pop_name=self.src_id,
                target_pop_name=self.tgt_id,
                print_results=False,
            )
            self.write_introgressed_segments_to_tsv(
                fragments, file_paths["bed_file"], chrom="1"
            )

        if self.create_mutation_check_file:
            phased_mutations_list = self.get_individuals_with_mutations(
                ts_simplified,
                target_mut_type_ids={self.slim_mutation_nr},
                target_pop_name=self.tgt_id,
                target_pop_id=None,
                only_alive=True,
                print_results=False,
            )
            self.write_filtered_mutation_sites_to_tsv(
                phased_mutations=phased_mutations_list,
                output_path=mut_tgt_file,
                target_mut_types={self.slim_mutation_nr},
                target_positions=None,
                chrom=self.chromosome,
            )

        return all_file_paths

    def perform_slim_simulation(
        self,
        file_paths: Dict[str, str],
        slim_script: str,
        adm_time: int,
        sel_time: int,
        adm_amount: float,
        dominance: str,
        rep: int,
    ) -> Dict[str, Any]:
        """
        Executes a SLiM simulation with specified admixture and selection parameters.

        Parameters
        ----------
        file_paths : dict of str
            Dictionary containing output file paths. Must include:
            - "ts_file": Path to the tree sequence output.
            - "txt_file": Path to auxiliary text output.
        slim_script : str
            Path to the SLiM simulation script to run.
        adm_time : int
            Time (in generations) at which admixture occurs.
        sel_time : int
            Time (in generations) at which selection starts.
        adm_amount : float
            Proportion of admixture from the source population.
        dominance : str
            Dominance model label (e.g., "recessive", "additive", etc.).
        rep : int
            Replicate identifier or counter.

        Returns
        -------
        dict
            A dictionary containing:
            - 'adm_time': int, admixture time
            - 'sel_time': int, selection time
            - 'adm_amount': float, admixture proportion
            - 'dominance': str, dominance model used
            - 'slim_script': str, path to the SLiM script
            - 'output_file': str, path to the tree sequence output
            - 'txt_file': str, path to the auxiliary output
            - 'mutation': bool, True if mutation type 2 occurred, False otherwise
            - 'output': CompletedProcess, result from subprocess.run
            - 'rep': int, replicate number
        """
        simulation = {}

        simulation["adm_time"] = adm_time
        simulation["sel_time"] = sel_time
        simulation["adm_amount"] = adm_amount
        simulation["dominance"] = dominance

        output_file = file_paths["ts_file"]
        txt_file = file_paths["txt_file"]

        simulation["slim_script"] = slim_script
        simulation["output_file"] = output_file
        simulation["txt_file"] = txt_file

        sub_output = subprocess.run(
            [
                "slim",
                "-d",
                f'output_path="{output_file}"',
                "-d",
                f'txt_path="{txt_file}"',
                "-d",
                f"adm_time={adm_time}",
                "-d",
                f"sel_time={sel_time}",
                "-d",
                f"adm_amount={adm_amount}",
                slim_script,
            ],
            capture_output=True,
            text=True,
        )

        if "Finish due to no mutation of type 2" in sub_output.stdout:
            simulation["mutation"] = False
        else:
            simulation["mutation"] = True

        simulation["output"] = sub_output
        simulation["rep"] = rep

        return simulation

    def _create_ref_tgt_file_slim(
        self,
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

    def count_individuals(self, ts: tskit.TreeSequence) -> Dict[int, int]:
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

    def get_pop_id_by_name(self, ts: tskit.TreeSequence, name: str) -> Optional[int]:
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

    def get_inds_by_pop(self, ts, pop_id):
        return [ind for ind in ts.individuals() if ind.population == pop_id]

    def get_inds_ids_by_pop(self, ts, pop_id):
        return [ind.id for ind in ts.individuals() if ind.population == pop_id]

    def sample_times(
        self, ts: tskit.TreeSequence, individuals: List[tskit.Individual]
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
            individual_time = max(
                node_times
            )  # or min(node_times), depending on use case

            individual_times.append(individual_time)

        return individual_times

    def simplify_by_population(
        self,
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
            populations_to_keep.append(self.get_pop_id_by_name(ts, pop_name))

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
        self,
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

        Notes
        -----
        Population names must match those assigned in the tree sequence metadata.
        This method relies on `self.get_pop_id_by_name` to resolve population names to IDs.
        """
        pop_names_to_keep = [source_pop_name, target_pop_name]
        populations_to_keep = []
        for pop_name in pop_names_to_keep:
            populations_to_keep.append(self.get_pop_id_by_name(self, ts, pop_name))

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

    def read_exon_file(self, exon_file: str) -> Tuple[List[int], List[int]]:
        """
        Reads exon region information from a formatted exon annotation file.

        The file is expected to contain lines beginning with the keyword 'exon',
        followed by a start and end coordinate separated by spaces.
        Taken directly from MalAdapt immplementation

        Parameters
        ----------
        exon_file : str
            Path to the exon annotation file. Each relevant line should start with
            'exon' and be followed by two integers (start and end positions).

        Returns
        -------
        tuple of list of int
            Two lists containing the start and end coordinates of the exon regions, respectively.

        Notes
        -----
        This method is based on logic from the `maladapt` pipeline. Lines not starting
        with 'exon' are ignored.
        """

        lines = open(exon_file, "r").readlines()
        lines = [x for x in lines if x.startswith("exon")]
        lines = [x.strip("\n").split(" ") for x in lines]
        annot, start, end = zip(*lines)
        start = [int(x) for x in start]
        end = [int(x) for x in end]

        return start, end

    def sample_and_simplify(
        self,
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

        pops_ids = deepcopy(pops)
        for pop in pops:

            pop_value = pops[pop]

            pop_id = self.get_pop_id_by_name(ts, pop_value)

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
            p_src = [ind.id for ind in indivs if ind.population == pops_ids["src"]]
        else:
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

        # Get archaic individuals sampled at a specific time

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
        # if integer
        else:
            # default: rchaic_sample_time is a single time, given as integer
            p_src = [
                ind.id
                for ind in indivs
                if ind.population == pops_ids["src"]
                and any(ts.node(n).time == archaic_sample_time for n in ind.nodes)
            ]

        nodes_p_src = [n for ind in indivs if ind.id in p_src for n in ind.nodes]

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

    def get_introgressed_segments(
        self,
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
                            introgressed_segments[ind_id][hap_index].append(
                                (start, end)
                            )
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

    def write_introgressed_segments_to_tsv(
        self, merged_introgression, output_path, chrom="1"
    ):
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
        self,
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

    def remove_mutations(self, ts, start, end, proportion):
        """
        This function will return a new tree sequence the same as the input,
        but after removing each non-SLiM mutation within regions specified in lists
        start and end with probability `proportion`, independently. So then, if we
        want to add neutral mutations with rate 1.0e-8 within the regions and 0.7e-8
        outside the regions, we could do
        ts = pyslim.load("my.trees")
        first_mut_ts = msprime.mutate(ts, rate=1e-8)
        mut_ts = remove_mutations(first_mut_ts, start, end, 0.3)
        :param float proportion: The proportion of mutations to remove.
        """
        tables = ts.dump_tables()
        tables.sites.clear()
        tables.mutations.clear()
        for tree in ts.trees():
            for site in tree.sites():
                assert len(site.mutations) == 1
                mut = site.mutations[0]
                keep_mutation = True
                for i in range(len(start)):
                    left = start[i]
                    right = end[i]
                    assert left < right
                    if i > 0:
                        assert end[i - 1] <= left
                    if left <= site.position < right:
                        keep_mutation = random.uniform(0, 1) > proportion

                if keep_mutation:
                    site_id = tables.sites.add_row(
                        position=site.position, ancestral_state=site.ancestral_state
                    )
                    tables.mutations.add_row(
                        site=site_id,
                        node=mut.node,
                        derived_state=mut.derived_state,
                        metadata=mut.metadata,
                    )

        return tables.tree_sequence()

    def get_individuals_with_mutations(
        self,
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
                if ind.population == target_pop_id
                and ind.flags & pyslim.INDIVIDUAL_ALIVE
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
