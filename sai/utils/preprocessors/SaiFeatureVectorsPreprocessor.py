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


import yaml
import numpy as np
from typing import Any
from sai.utils import parse_ind_file
from sai.utils.preprocessors import DataPreprocessor

from sai.stats.features import *
import sai.utils.gaia_utils


class SaiFeatureVectorsPreprocessor(DataPreprocessor):
    """
    A preprocessor subclass for generating feature vectors from genomic data.

    This class extends DataPreprocessor to include additional functionality for creating
    feature vectors based on genomic variants, reference, source and target individual genotypes,
    and window-based genomic statistics.

    """

    def __init__(
        self,
        ref_ind_file: str,
        tgt_ind_file: str,
        src_ind_file: str,
        feature_config: str,
    ):
        """
        Initializes a new instance of FeatureVectorsPreprocessor with specific parameters.

        Parameters:
        -----------
        ref_ind_file : str
            Path to the file listing reference individual identifiers.
        tgt_ind_file : str
            Path to the file listing target individual identifiers.
        src_ind_file : str
            Path to the file listing source individual identifiers.
        feature_config : str
            Path to the configuration file specifying the features to be computed.

        Raises
        ------
        FileNotFoundError
            If the feature configuration file is not found.
        ValueError
            If the feature configuration file is incorrectly formatted or does not contain any features.

        """
        try:
            with open(feature_config, "r") as f:
                features = yaml.safe_load(f)
            # self.features = features.get("Features", {})
            self.features = features
            if not self.features:
                raise ValueError("No features found in the configuration.")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Feature configuration file {feature_config} not found."
            )
        except yaml.YAMLError as exc:
            raise ValueError(f"Error parsing feature configuration: {exc}")

        ref_samples = sai.utils.gaia_utils.parse_ind_file(ref_ind_file)
        tgt_samples = sai.utils.gaia_utils.parse_ind_file(tgt_ind_file)

        src_samples = sai.utils.gaia_utils.parse_ind_file(src_ind_file)
        self.samples = {"Ref": ref_samples, "Tgt": tgt_samples, "Src": src_samples}

    def run(
        self,
        chr_name: str,
        start: int,
        end: int,
        ploidy: int,
        is_phased: bool,
        ref_gts: np.ndarray,
        tgt_gts: np.ndarray,
        pos: np.ndarray,
        src_gts: np.ndarray = None,
    ) -> list[dict[str, Any]]:
        """
        Executes the feature vector generation process for a specified genomic window.

        Parameters
        ----------
        chr_name : str
            Name of the chromosome.
        start : int
            Start position of the genomic window.
        end : int
            End position of the genomic window.
        ploidy : int
            Ploidy of the samples, typically 2 for diploid organisms.
        is_phased : bool
            Indicates whether the genomic data is phased.
        ref_gts : np.ndarray
            Genotype array for the reference individuals.
        tgt_gts : np.ndarray
            Genotype array for the target individuals.
        pos : np.ndarray
            Array of variant positions within the genomic window.

        Returns
        -------
        list
            A list of dictionaries containing the formatted feature vectors for the genomic window.

        """
        variants_not_in_ref = np.sum(ref_gts, axis=1) == 0
        sub_ref_gts = ref_gts[variants_not_in_ref]
        sub_tgt_gts = tgt_gts[variants_not_in_ref]
        sub_pos = pos[variants_not_in_ref]

        import inspect

        basic_params = {
            "is_phased": is_phased,
            "ploidy": ploidy,
            "ref_gts": ref_gts,
            "tgt_gts": tgt_gts,
            "src_gts": src_gts,
            "pos": pos,
            "src_gts_list": [src_gts],
            "y_list": [(">", 0.5)],
        }

        items = {}
        items["Chromosome"] = chr_name
        items["Start"] = start
        items["End"] = end
        items["Samples"] = self.samples
        for func_name, yaml_params in self.features["Features"].items():
            func = globals().get(func_name)
            if not callable(func):
                continue

            sig = inspect.signature(func)
            call_args = {}

            if isinstance(yaml_params, dict):
                call_args.update(yaml_params)
            elif isinstance(yaml_params, bool):
                if not yaml_params:
                    continue
                # Else: proceed with just context-based parameters
            else:
                raise ValueError(
                    f"Unsupported value type for feature '{func_name}': {type(yaml_params)}"
                )

            for param in sig.parameters.values():
                name = param.name

                if name in call_args:
                    continue
                if name == "gts" and "tgt_gts" in basic_params:
                    call_args["gts"] = basic_params["tgt_gts"]
                elif name in basic_params:
                    call_args[name] = basic_params[name]

            res = func(**call_args)

            if isinstance(res, tuple):
                for i, value in enumerate(res):
                    key = f"{func_name}|val{i}|{yaml_params}"
                    items[key] = value
            else:
                key = f"{func_name}|{yaml_params}"
                items[key] = res

        # number of snps as column
        items["nsnps"] = tgt_gts.shape[0]

        items["tgt_gts_shape"] = tgt_gts.shape
        items["ref_gts_shape"] = ref_gts.shape
        items["src_gts_shape"] = src_gts.shape

        stat_results = items

        return stat_results
