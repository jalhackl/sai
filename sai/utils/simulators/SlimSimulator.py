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
from sai.utils.simulators.slim_simulator_utils import (
    _create_ref_tgt_file_slim,
    sample_and_simplify,
    simplify_by_source_target,
    get_introgressed_segments,
    write_introgressed_segments_to_tsv,
    get_individuals_with_mutations,
    write_filtered_mutation_sites_to_tsv,
)
import tskit
import pyslim
from sai.utils.simulators import DataSimulator
import subprocess
from typing import Dict, Any, List, Optional, Union, Tuple, DefaultDict, Set


class SlimSimulator(DataSimulator):
    """
        A simulator for generating genetic data using SLiM within a tskit-based workflow.

        This subclass of `DataSimulator` configures parameters specific to SLiM simulations,
        particularly for maladaptive or uniform selection models. It manages simulation output,
        mutation parameters, and metadata relevant to downstream genetic analysis.

        Parameters
        ----------
    slim_script_folder : str, optional
            Path to the folder containing SLiM scripts. Default is 'examples/slim/racimo'.
        slim_script_name : str, optional
            Filename of the SLiM script. Default is 'racimo_default.slim'.
        nref : int, optional
            Number of reference individuals. Default is 108.
        ntgt : int, optional
            Number of target individuals. Default is 99.
        ref_id : str, optional
            Population ID for the reference population. Default is "p1".
        tgt_id : str, optional
            Population ID for the target population. Default is "p4".
        src_id : str, optional
            Population ID for the source population. Default is "p2".
        seq_len : int, optional
            Length of the simulated genomic sequence. Default is 40,000.
        adm_amount : float, optional
            Proportion of admixture. Default is 0.02.
        output_prefix : str, optional
            Prefix for output files. Default is "slim_sim".
        output_dir : str, optional
            Directory to store simulation results. Default is "slim_res".
        is_phased : bool, optional
            Whether output data should be phased. Default is True.
        scaling_factor : int or float, optional
            Factor used to scale mutation and recombination rates. Default is 1.
        basic_mut_rate : float, optional
            Base mutation rate before scaling. Default is 1.5e-8.
        neutral_mut_rate : float, optional
            Neutral mutation rate before scaling. Default is 1.5e-8.
        archaic_sample_time : int or str, optional
            Time in generations for archaic sampling or "oldest". Default is "oldest".
        nsrc : int, optional
            Number of source population individuals. Default is 1.
        create_tracts_file : bool, optional
            Whether to generate a tracts file. Default is False.
        create_mutation_check_file : bool, optional
            Whether to generate a mutation check file. Default is True.
        slim_mutation_nr : int, optional
            Number of SLiM mutation types. Default is 2.
        extra_mutations : bool, optional
            Whether to add extra mutations in postprocessing. Default is False.
        resample : int, optional
            Number of resamples. Default is 0.
        chromosome : str, optional
            Chromosome name used in simulation setup. Default is "1".
        return_finished_trees : bool, optional
            Whether to save finished tree sequences. Default is False.
        simplify_input_trees : bool, optional
            Whether to simplify input trees. Default is True.
        selection_coefficient : float, optional
            Selection coefficient for mutations. Default is 0.1.
        recombination_rate : float, optional
            Recombination rate per site. Default is 1e-8.
        ancestral_Ne : int, optional
            Ancestral effective population size. Default is 10,000.
        initial_sweep_frequency : float, optional
            Initial sweep frequency for selected mutations. Default is 0.
        recapitate_slim_sim : bool, optional
            Whether to recapitate the SLiM simulation. Default is True.
        src_sample_generations : int, optional
            Number of generations since source sampling. Default is 1500.

    """

    def __init__(
        self,
        slim_script_folder=os.path.join("examples", "slim", "racimo"),
        slim_script_name="racimo_default.slim",
        nref: int = 108,
        ntgt: int = 99,
        ref_id: str = "p1",
        tgt_id: str = "p4",
        src_id: str = "p2",
        seq_len: int = 40000,
        adm_amount: float = 0.02,
        output_prefix: str = "slim_sim",
        output_dir: str = "slim_res",
        is_phased: bool = True,
        scaling_factor=1,
        basic_mut_rate=1.5e-8,
        neutral_mut_rate=1.5e-8,
        archaic_sample_time="oldest",
        nsrc: int = 1,
        create_tracts_file: bool = False,
        create_mutation_check_file: bool = True,
        slim_mutation_nr: int = 2,
        extra_mutations: bool = False,
        resample: int = 0,
        chromosome: str = "1",
        return_finished_trees: bool = False,
        simplify_input_trees: bool = True,
        selection_coefficient: float = 0.1,
        recombination_rate: float = 1e-8,
        ancestral_Ne: int = 10000,
        initial_sweep_frequency: float = 0,
        recapitate_slim_sim: bool = True,
        src_sample_generations: int = 1500,
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
        self.nsrc = nsrc

        self.slim_script_folder = slim_script_folder
        self.slim_script_name = slim_script_name

        self.return_finished_trees = return_finished_trees
        self.simplify_input_trees = simplify_input_trees

        self.adm_amount = adm_amount
        self.initial_sweep_frequency = initial_sweep_frequency

        self.test_tgt_id = self.tgt_id
        self.is_phased = is_phased

        self.create_tracts_file = create_tracts_file
        self.create_mutation_check_file = create_mutation_check_file

        self.identifier = "tsk_"

        self.scaling_factor = scaling_factor
        self.basic_mut_rate = basic_mut_rate
        self.mu = self.basic_mut_rate * self.scaling_factor

        self.neutral_mut_rate = neutral_mut_rate
        self.neutral_mut_rate_scaled = neutral_mut_rate * self.scaling_factor

        self.recombination_rate = recombination_rate
        self.recombination_rate_scaled = 0.5 * (
            1 - (1 - 2 * self.recombination_rate) ** self.scaling_factor
        )

        self.ancestral_Ne = ancestral_Ne / self.scaling_factor

        self.initial_sweep_frequency = initial_sweep_frequency
        self.selection_coefficient = selection_coefficient

        self.slim_mutation_nr = slim_mutation_nr

        self.extra_mutations = extra_mutations

        self.archaic_sample_time = archaic_sample_time

        self.pops = {"ref": ref_id, "src": src_id, "tgt": tgt_id}
        self.sample_sizes = {"ref": nref, "src": nsrc, "tgt": ntgt}

        self.resample = resample

        self.chromosome = chromosome

        self.return_finished_trees = return_finished_trees

        self.bottleneck_factor = 1
        self.expansion_factor = 1

        self.recapitate_slim_sim = recapitate_slim_sim

        self.src_sample_generations = src_sample_generations

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

        slim_script_name = self.slim_script_name

        slim_script = os.path.join(self.slim_script_folder, slim_script_name)

        os.makedirs(output_dir, exist_ok=True)
        ts_file = os.path.join(output_dir, f"{output_prefix}.ts")
        if self.return_finished_trees:
            ts_file_finished = os.path.join(output_dir, f"{output_prefix}_finished.ts")
        txt_file = os.path.join(output_dir, f"{output_prefix}.txt")
        vcf_file = os.path.join(output_dir, f"{output_prefix}.vcf")
        bed_file = os.path.join(output_dir, f"{output_prefix}.true.tracts.bed")
        ref_ind_file = os.path.join(output_dir, f"{output_prefix}.ref.ind.list")
        tgt_ind_file = os.path.join(output_dir, f"{output_prefix}.tgt.ind.list")

        slim_vcf_file = os.path.join(output_dir, f"{output_prefix}_slim.vcf")

        if self.nsrc is not None and self.nsrc > 0:
            src_ind_file = os.path.join(output_dir, f"{output_prefix}.src.ind.list")
        else:
            src_ind_file = None

        mut_tgt_file = os.path.join(output_dir, f"{output_prefix}.tgt.mut.list")

        file_paths = {
            "ts_file": ts_file,
            "vcf_file": vcf_file,
            "bed_file": bed_file,
            "ref_ind_file": ref_ind_file,
            "tgt_ind_file": tgt_ind_file,
            "src_ind_file": src_ind_file,
            "txt_file": txt_file,
            "slim_vcf_file": slim_vcf_file,
        }
        if self.return_finished_trees:
            file_paths["finished_ts_file"] = ts_file_finished
        if self.create_mutation_check_file:
            file_paths["mut_file"] = mut_tgt_file

        all_file_paths = []
        if self.resample == 0:
            all_file_paths.append(file_paths)

        simulation = self.perform_slim_simulation(
            file_paths, slim_script, rep, seed=seed
        )

        ts_path = simulation["output_file"]
        ts = tskit.load(ts_path)

        # get simuation tree sequence (ts)

        if self.resample > 0:
            print("currently not implemented!")
            pass

        if self.recapitate_slim_sim:
            ts = pyslim.recapitate(
                ts,
                ancestral_Ne=self.ancestral_Ne,
                recombination_rate=self.recombination_rate_scaled,
            )

        # simplify and sample part
        if self.sample_sizes:
            ts_simplified, all_ps_simplified, all_p_ref, all_p_tgt, all_p_src = (
                sample_and_simplify(
                    ts,
                    pops=self.pops,
                    sample_sizes=self.sample_sizes,
                    archaic_sample_time=self.archaic_sample_time,
                    seed=seed,
                )
            )
        else:
            ts_simplified = ts

        if self.extra_mutations:
            ts_simplified = msprime.mutate(
                ts_simplified, rate=self.neutral_mut_rate_scaled, keep=True
            )

        # write vcf
        with open(file_paths["vcf_file"], "w") as vcf_file:
            ts_simplified.write_vcf(
                vcf_file,
                individuals=all_ps_simplified,
                individual_names=[self.identifier + str(x) for x in all_ps_simplified],
                allow_position_zero=True,
            )

        if self.return_finished_trees:
            ts_simplified.dump(ts_file_finished)

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
        self, file_paths: Dict[str, str], slim_script: str, rep: int, seed: int = None
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

        simulation["seq_length"] = self.seq_len
        simulation["bottleneck_factor"] = self.bottleneck_factor
        simulation["expansion_factor"] = self.expansion_factor
        simulation["initial_sweep_frequency"] = self.initial_sweep_frequency
        simulation["selection_coefficient"] = self.selection_coefficient
        simulation["mutation_rate"] = self.basic_mut_rate
        simulation["recombination_rate"] = self.recombination_rate

        output_file = file_paths["ts_file"]
        txt_file = file_paths["txt_file"]

        simulation["slim_script"] = slim_script
        simulation["output_file"] = output_file
        simulation["txt_file"] = txt_file

        slim_vcf_file = file_paths["slim_vcf_file"]

        slim_args = [
            "slim",
            "-d",
            f'output_path="{output_file}"',
            "-d",
            f'txt_path="{txt_file}"',
            "-d",
            f'output_vcf="{slim_vcf_file}"',
            "-d",
            f"seq_length={self.seq_len}",
            "-d",
            f"initial_sweep_frequency={self.initial_sweep_frequency}",
            "-d",
            f"selection_coefficient={self.selection_coefficient}",
            "-d",
            f"mutation_rate={self.basic_mut_rate}",
            "-d",
            f"recombination_rate={self.recombination_rate}",
            "-d",
            f"scaling_factor={self.scaling_factor}",
            "-d",
            f"adm_amount={self.adm_amount}",
            "-d",
            f"slim_archaic_sample_generations_unscaled={self.src_sample_generations}",
        ]

        if seed is not None:
            slim_args.extend(["-d", f"seed={seed}"])

        slim_args.append(slim_script)

        sub_output = subprocess.run(
            slim_args,
            capture_output=True,
            text=True,
        )

        simulation["output"] = sub_output
        simulation["rep"] = rep

        return simulation
