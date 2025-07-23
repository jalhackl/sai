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
from sai.utils.simulators import DataSimulator
import random
import subprocess
from typing import Dict, Any, List, Optional, Union, Tuple, DefaultDict, Set

from sai.utils.simulators.slim_simulator_utils import (
    _create_ref_tgt_file_slim,
    sample_and_simplify,
    simplify_by_source_target,
    get_introgressed_segments,
    write_introgressed_segments_to_tsv,
    get_individuals_with_mutations,
    write_filtered_mutation_sites_to_tsv,
)

from sai.utils.simulators.slim_simulator_maladapt_utils import combine_mutation_trees, only_add_maladapt_mutations


class SlimSimulatorMaladapt(DataSimulator):
    """
    A simulator for generating genetic data using SLiM within a tskit-based workflow, following the model and specifications of Maladapt.

    This subclass of `DataSimulator` configures parameters specific to SLiM simulations,
    different settings (e.g. maladapt treatment of mutations and uniform mutation rate). It manages simulation output,
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
            sample_and_simplify(
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
        _create_ref_tgt_file_slim(
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
            ts_only_src_tgt = simplify_by_source_target(
                ts_simplified, source_pop_name=self.src_id, target_pop_name=self.tgt_id
            )
            fragments = get_introgressed_segments(
                ts_only_src_tgt,
                source_pop_name=self.src_id,
                target_pop_name=self.tgt_id,
                print_results=False,
            )
            write_introgressed_segments_to_tsv(
                fragments, file_paths["bed_file"], chrom="1"
            )

        if self.create_mutation_check_file:
            phased_mutations_list = get_individuals_with_mutations(
                ts_simplified,
                target_mut_type_ids={self.slim_mutation_nr},
                target_pop_name=self.tgt_id,
                target_pop_id=None,
                only_alive=True,
                print_results=False,
            )
            write_filtered_mutation_sites_to_tsv(
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
