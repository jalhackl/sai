import numpy as np
from typing import Any
from sai.utils.preprocessors import DataPreprocessor
import yaml

from sai.stats.features import *


class SaiFeaturePreprocessor(DataPreprocessor):
    """
    A preprocessor subclass for generating feature vectors from genomic data.

    This class extends DataPreprocessor to include additional functionality for creating
    feature vectors based on genomic variants, reference and target individual genotypes,
    and window-based genomic statistics.
    """

    def __init__(
        self,
        output_file: str,
        feature_config: str,
        anc_allele_available: bool = False,
        # additional phased information
        is_phased: bool = True,
    ):
        """
        Initializes FeatureVectorsPreprocessor with specific frequency thresholds
        and output file for storing generated feature vectors.

        Parameters
        ----------

        output_file : str
            Path to the output file to save processed feature vectors.
        feature_config: str,
            Specifies the config file with the statistics to compute.
        anc_allele_available: bool, optional
            If True, ancestral allele information is available.
            If False, ancestral allele information is unavailable.
            Default is False.
        is_phased: bool, optional
            whether data is / should be treated as phased, default=True
        """

        self.output_file = output_file
        self.feature_config = feature_config
        self.anc_allele_available = anc_allele_available

        self.is_phased = is_phased

        try:
            with open(feature_config, "r") as f:
                features = yaml.safe_load(f)
            # self.features = features.get("Features", {})
            self.features = features
            if not self.features:
                raise ValueError("No features found in the configuration.")

            print(self.features)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Feature configuration file {feature_config} not found."
            )
        except yaml.YAMLError as exc:
            raise ValueError(f"Error parsing feature configuration: {exc}")

    def run(
        self,
        chr_name: str,
        ref_pop: str,
        tgt_pop: str,
        src_pop_list: list[str],
        start: int,
        end: int,
        pos: np.ndarray,
        ref_gts: np.ndarray,
        tgt_gts: np.ndarray,
        src_gts_list: list[np.ndarray],
        ploidy: int,
    ) -> list[dict[str, Any]]:
        """
        Generates feature vectors for a specified genomic window.

        Parameters
        ----------
        chr_name : str
            Chromosome name.
        ref_pop : str
            Reference population name.
        tgt_pop : str
            Target population name.
        src_pop_list : list[str]
            List of source population names.
        start : int
            Start position of the genomic window.
        end : int
            End position of the genomic window.
        pos : np.ndarray
            A 1D numpy array where each element represents the genomic position.
        ref_gts : np.ndarray
            Genotype data for the reference population.
        tgt_gts : np.ndarray
            Genotype data for the target population.
        src_gts_list : list[np.ndarray]
            List of genotype arrays for each source population.
        ploidy: int
            Ploidy of the genome.

        Returns
        -------
        list[dict[str, Any]]
            A list containing a dictionary of calculated feature vectors for the genomic window.
        """

        if (
            (ref_gts is None or len(ref_gts) == 0)
            or (tgt_gts is None or len(tgt_gts) == 0)
            or (src_gts_list is None)
            or (ploidy is None)
        ):
            return None
            # items["statistic"] = np.nan
            # items["candidates"] = np.array([])

        all_items = []
        for src_pop, src_gts in zip(src_pop_list, src_gts_list):

            items = {
                "chr_name": chr_name,
                "start": start,
                "end": end,
                "ref_pop": ref_pop,
                "tgt_pop": tgt_pop,
                "src_pop": src_pop,
                "src_pop_list": src_pop_list,
                "nsnps": len(pos),
            }

            basic_params = {
                "phased": self.is_phased,
                "is_phased": self.is_phased,
                "ploidy": ploidy,
                "ref_gts": ref_gts,
                "tgt_gts": tgt_gts,
                "src_gts": src_gts,
                "pos": pos,
                "src_gts_list": [src_gts],
                "y_list": [(">", 0.5)],
            }

            import inspect

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
                        continue  # Skip if it's explicitly False
                    # Else: proceed with just context-based parameters
                else:
                    raise ValueError(
                        f"Unsupported value type for feature '{func_name}': {type(yaml_params)}"
                    )

                for param in sig.parameters.values():
                    name = param.name

                    if name in call_args:
                        continue
                    # features which only require one genotype matrix get the tgt matrix - usually this is the desired behaviour
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

                items["tgt_gts_shape"] = tgt_gts.shape
                items["ref_gts_shape"] = ref_gts.shape
                items["src_gts_shape"] = src_gts.shape

            all_items.append(items)

        return all_items

    def process_items(self, items: list[dict[str, Any]]) -> None:
        """
        Processes and writes a single dictionary of feature vectors to the output file.

        Parameters
        ----------
        items : dict[str, Any]
            A dictionary containing feature vectors for a genomic window.
        """

        with open(
            self.output_file, "a"
        ) as f:  # Open in append mode for continuous writing
            lines = []
            for item in items:
                src_pop_str = ",".join(item["src_pop_list"])
                candidates = (
                    "NA"
                    if item["candidates"].size == 0
                    else ",".join(
                        f"{item['chr_name']}:{pos}" for pos in item["candidates"]
                    )
                )

                line = (
                    f"{item['chr_name']}\t{item['start']}\t{item['end']}\t"
                    f"{item['ref_pop']}\t{item['tgt_pop']}\t{src_pop_str}\t"
                    f"{item['nsnps']}\t{item['statistic']}\t{candidates}\n"
                )
                lines.append(line)

            f.writelines(lines)
