import argparse
import sys
from sai.parsers.argument_validation import positive_int
from sai.parsers.argument_validation import positive_number
from sai.parsers.argument_validation import existed_file

from sai.preprocess import Sai_lr_process_folder


def add_preprocess_parsers(subparsers: argparse.ArgumentParser) -> None:
    """
    Initializes and configures the command-line interface parser
    for preprocessing data for SAI feature extraction.

    Parameters
    ----------
    subparsers : argparse.ArgumentParser
        A command-line interface parser to be configured.

    """

    parser = subparsers.add_parser(
        "preprocess",
        help="Preprocess data for SAI feature extraction",
    )

    parser.add_argument(
        "--vcf_location",
        type=str,
        help="Directory of input files",
    )

    parser.add_argument(
        "--feature_config",
        type=str,
        help="File with feature configuration to be computed",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="Slim_Preprocess_Output",
        help="Directory to save output files",
    )

    parser.add_argument(
        "--vcf_ending",
        type=str,
        default=".vcf.gz",
        help="suffix of vcf-files, usually .vcf or .vcf.gz",
    )

    parser.add_argument(
        "--nprocess",
        type=positive_int,
        default=1,
        help="number of processesfor the simulation, i.e. how many vcf-files with simulations are processed in parallel; default: 1",
    )

    parser.set_defaults(runner=Sai_lr_process_folder)
