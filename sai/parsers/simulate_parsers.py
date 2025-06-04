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


import argparse
import sys
from sai.parsers.argument_validation import positive_int
from sai.parsers.argument_validation import positive_number
from sai.parsers.argument_validation import existed_file


def _run_simulationSlim(args: argparse.Namespace) -> None:

    from sai.simulate import simulate_slim_data

    simulate_slim_data(
        nref=args.nref,
        ntgt=args.ntgt,
        ref_id=args.ref_id,
        tgt_id=args.tgt_id,
        src_id=args.src_id,
        seq_len=args.seq_len,
        output_prefix=args.output_prefix,
        output_dir=args.output_dir,
        is_phased=args.is_phased,
        scaling_factor=args.scaling_factor,
        basic_mut_rate=args.basic_mut_rate,
        archaic_sample_time=args.archaic_sample_time,
        nsrc=args.nsrc,
        out_id=args.out_id,
        nout=args.nout,
        create_tracts_file=args.create_tracts_file,
        create_mutation_check_file=args.create_mutation_check_file,
        create_all_mutation_output=args.create_all_mutation_output,
        slim_mutation_nr=args.slim_mutation_nr,
        maladapt_mutations_preprocess=args.maladapt_mutations_preprocess,
        extra_mutations=args.extra_mutations,
        settings=args.settings,
        nrep=args.nrep,
        seed=args.seed,
        nprocess=args.nprocess,
        resample=args.resample
    )


def _run_simulation(args: argparse.Namespace) -> None:
    """
    Executes a simulation process with specified parameters.

    Parameters
    ----------
    args : argparse.Namespace
        A namespace object obtained from argparse, containing simulation parameters:
        - demes: File path to the demographic model specification in YAML format.
        - replicate: Number of simulation replicates.
        - nref: Size of the reference population.
        - ntgt: Size of the target population.
        - ref_id: Identifier for the reference population.
        - tgt_id: Identifier for the target population.
        - src_id: Identifier for the source population.
        - ploidy: Ploidy of the organisms being simulated.
        - is_phased: Indicates if the simulated data should be phased.
        - seq_len: Length of the sequence to simulate.
        - mut_rate: Mutation rate to use in the simulation.
        - rec_rate: Recombination rate to use in the simulation.
        - nprocess: Number of processes to use for parallel simulations.
        - output_prefix: Prefix for output files.
        - output_dir: Directory where output files will be saved.
        - seed: Random seed for reproducibility.
        - nfeature: Number of features to simulate.
        - num_polymorphisms: Number of polymorphisms in each genotype matrix to simulate.
        - num_upsamples: Number of samples after upsampling.
        - output_h5: Boolean flag to save output in HDF5 format.
        - is_sorted: Boolean flag to indicate whether to sort the genotype matrices.
        - only_intro: Boolean flag to simulate only introgressed fragments.
        - only_non_intro: Boolean flag to simulate only non-introgressed fragments.
        - force_balanced: Boolean flag to ensure a balanced distribution of introgressed and
                          non-introgressed classes in the training data.
        - keep_sim_data: Boolean flag to keep or discard simulation data.
        - chunk_size: integer for HDF chunk size


    """

    import demes
    from sai.simulate import simulate_test_data

    demog = demes.load(args.demes)
    pops = [d.name for d in demog.demes]
    if args.ref_id not in pops:
        print(
            f"gaia simulate: error: argument --ref_id: Population {args.ref_id} is not found in the demographic model file {args.demes}"
        )
        sys.exit(1)
    if args.tgt_id not in pops:
        print(
            f"gaia simulate: error: argument --tgt_id: Population {args.tgt_id} is not found in the demographic model file {args.demes}"
        )
        sys.exit(1)
    if args.src_id not in pops:
        print(
            f"gaia simulate: error: argument --src_id: Population {args.src_id} is not found in the demographic model file {args.demes}"
        )
        sys.exit(1)

    simulate_test_data(
        demo_model_file=args.demes,
        nrep=args.replicate,
        nref=args.nref,
        ntgt=args.ntgt,
        ref_id=args.ref_id,
        tgt_id=args.tgt_id,
        src_id=args.src_id,
        ploidy=args.ploidy,
        is_phased=args.phased,
        seq_len=args.seq_len,
        mut_rate=args.mut_rate,
        rec_rate=args.rec_rate,
        nprocess=args.nprocess,
        output_prefix=args.output_prefix,
        output_dir=args.output_dir,
        seed=args.seed,
        nsrc=args.nsrc,
        out_id=args.out_id,
        nout=args.nout,
    )


def add_simulate_parsers(subparsers: argparse.ArgumentParser) -> None:
    """
    Initializes and configures the command-line interface parser
    for simultating data.

    Parameters
    ----------
    subparsers : argparse.ArgumentParser
        A command-line interface parser to be configured.

    """

    parser = subparsers.add_parser(
        "simulate",
        help="Simulate and Label data ready for processing with the implemented models, e.g. for testing",
    )
    # unet_subparsers = unet_parsers.add_subparsers(dest="unet_subparsers")

    # Arguments for the general simulate command
    # parser = unet_subparsers.add_parser("simulate", help="simulate data for training")
    parser.add_argument(
        "--demes",
        type=existed_file,
        required=True,
        help="demographic model in the DEMES format",
    )
    parser.add_argument(
        "--nref",
        type=positive_int,
        required=True,
        help="number of samples in the reference population",
    )
    parser.add_argument(
        "--ntgt",
        type=positive_int,
        required=True,
        help="number of samples in the target population",
    )
    parser.add_argument(
        "--ref-id",
        type=str,
        required=True,
        help="name of the reference population in the demographic model",
        dest="ref_id",
    )
    parser.add_argument(
        "--tgt-id",
        type=str,
        required=True,
        help="name of the target population in the demographic model",
        dest="tgt_id",
    )
    parser.add_argument(
        "--src-id",
        type=str,
        required=True,
        help="name of the source population in the demographic model",
        dest="src_id",
    )
    parser.add_argument(
        "--seq-len",
        type=positive_int,
        required=True,
        help="length of the simulated genomes",
        dest="seq_len",
    )
    parser.add_argument(
        "--ploidy",
        type=positive_int,
        default=2,
        help="ploidy of the simulated genomes; default: 2",
    )
    parser.add_argument(
        "--phased",
        action="store_true",
        help="enable to use phased genotypes; default: False",
    )
    parser.add_argument(
        "--mut-rate",
        type=positive_number,
        default=1e-8,
        help="mutation rate per base pair per generation for the simulation; default: 1e-8",
        dest="mut_rate",
    )
    parser.add_argument(
        "--rec-rate",
        type=positive_number,
        default=1e-8,
        help="recombination rate per base pair per generation for the simulation; default: 1e-8",
        dest="rec_rate",
    )

    parser.add_argument(
        "--replicate",
        type=positive_int,
        default=1,
        help="total number of simulations, i.e. how many vcf-files with simulations are created; default: 1",
    )

    parser.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="prefix of the output file name",
        dest="output_prefix",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="directory of the output files",
        dest="output_dir",
    )

    parser.add_argument(
        "--nprocess",
        type=positive_int,
        default=1,
        help="number of processesfor the simulation, i.e. how many vcf-files with simulations are processed in parallel; default: 1",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for the simulation; default: None",
    )

    parser.add_argument(
        "--nsrc",
        type=positive_int,
        required=False,
        default=None,
        help="number of samples in the source population",
    )

    parser.add_argument(
        "--nout",
        type=positive_int,
        required=False,
        default=None,
        help="number of samples in the outgroup population",
    )
    parser.add_argument(
        "--out-id",
        type=str,
        required=False,
        default=None,
        help="name of the outgroup population in the demographic model",
        dest="out_id",
    )

    parser.set_defaults(runner=_run_simulation)


def add_simulate_parsers(subparsers: argparse.ArgumentParser) -> None:
    """
    Initializes and configures the command-line interface parser
    for simultating data.

    Parameters
    ----------
    subparsers : argparse.ArgumentParser
        A command-line interface parser to be configured.

    """

    parser = subparsers.add_parser(
        "simulate_slim",
        help="Simulate data using Slim",
    )

    parser.add_argument(
        "--nref", type=int, default=108, help="Number of reference individuals"
    )
    parser.add_argument(
        "--ntgt", type=int, default=99, help="Number of target individuals"
    )
    parser.add_argument(
        "--ref_id", type=str, default="p1", help="Population ID for reference"
    )
    parser.add_argument(
        "--tgt_id", type=str, default="p4", help="Population ID for target"
    )
    parser.add_argument(
        "--src_id", type=str, default="p2", help="Population ID for source"
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=5000000,
        help="Sequence length to simulate (Currently not used).",
    )
    parser.add_argument(
        "--output_prefix", type=str, default="slim_sim.", help="Prefix for output files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Slim_Output",
        help="Directory to save output files",
    )
    parser.add_argument(
        "--is_phased",
        action="store_true",
        help="Indicate if the data should be phased (default: True)",
    )
    parser.set_defaults(is_phased=True)
    parser.add_argument(
        "--scaling_factor",
        type=int,
        default=10,
        help="Scaling factor for the simulation",
    )
    parser.add_argument(
        "--basic_mut_rate", type=float, default=1.5e-8, help="Basic mutation rate"
    )
    parser.add_argument(
        "--archaic_sample_time",
        type=int,
        default=152,
        help="Time at which to sample archaic population",
    )
    parser.add_argument(
        "--nsrc", type=int, default=2, help="Number of source individuals"
    )
    parser.add_argument(
        "--out_id", type=str, default=None, help="Population ID for outgroup (optional)"
    )
    parser.add_argument(
        "--nout",
        type=int,
        default=None,
        help="Number of outgroup individuals (optional)",
    )
    parser.add_argument(
        "--create_tracts_file",
        action="store_true",
        help="Whether to create a tracts file",
    )
    parser.set_defaults(create_tracts_file=False)
    parser.add_argument(
        "--create_mutation_check_file",
        action="store_true",
        help="Whether to create mutation check file",
    )
    parser.set_defaults(create_mutation_check_file=True)
    parser.add_argument(
        "--create_all_mutation_output",
        action="store_true",
        help="Whether to output all mutations",
    )
    parser.set_defaults(create_all_mutation_output=False)
    parser.add_argument(
        "--slim_mutation_nr", type=int, default=2, help="Number of SLiM mutation types"
    )
    parser.add_argument(
        "--no_maladapt_mutations_preprocess",
        action="store_false",
        help="Do NOT apply Maladapt processing (exon information for removal and adding of msprime mutations)",
        dest="maladapt_mutations_preprocess"
    )
    parser.set_defaults(maladapt_mutations_preprocess=True)
    parser.add_argument(
        "--extra_mutations",
        action="store_true",
        help="Include extra mutations in the simulation",
    )
    parser.set_defaults(extra_mutations=False)
    parser.add_argument(
        "--settings",
        type=str,
        required=False,
        default="maladapt",
        help="Slim script and settings to be used, currently 'maladapt' and 'uniform' are supported",
        dest="settings",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for the simulation; default: None",
    )

    parser.add_argument(
        "--nrep",
        type=positive_int,
        default=1,
        help="total number of simulations, i.e. how many vcf-files with simulations are created; default: 1",
    )

    parser.add_argument(
        "--nprocess",
        type=positive_int,
        default=1,
        help="number of processesfor the simulation, i.e. how many vcf-files with simulations are processed in parallel; default: 1",
    )

    parser.add_argument(
        "--resample",
        type=int,
        default=0,
        help="Nr of resamplings from same simulation.",
    )

    parser.set_defaults(runner=_run_simulationSlim)
