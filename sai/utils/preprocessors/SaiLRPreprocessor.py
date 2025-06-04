import os
import pandas as pd

from sai.utils.multiprocessing import mp_manager
from .SaiFeatureVectorsPreprocessor import SaiFeatureVectorsPreprocessor
from sai.utils.generators import SaiWindowDataGenerator


class SaiLRPreprocessor:
    """
    Preprocess genomic data to generate feature vectors for machine learning models.

    This class orchestrates the preprocessing pipeline by initializing a genomic data generator
    and a feature vector preprocessor. It utilizes multiprocessing to efficiently process large
    genomic datasets, generating feature vectors based on the specified configuration.
    """

    def __init__(
        self,
        feature_config: str,
        output_dir: str,
        chr_name: str = "1",
        ref_ind_file: str = None,
        tgt_ind_file: str = None,
        win_len: int = 50000,
        win_step: int = 10000,
        output_prefix: str = "lr",
        nprocess: int = 1,
        ploidy: int = 2,
        is_phased: bool = True,
        anc_allele_file: str = None,
        src_ind_file: str = None,
        label_data: bool = False,
    ):

        self.chr_name = chr_name
        self.ref_ind_file = ref_ind_file
        self.tgt_ind_file = tgt_ind_file
        self.win_len = win_len
        self.win_step = win_step
        self.feature_config = feature_config
        self.output_dir = output_dir
        self.output_prefix = output_prefix
        self.nprocess = nprocess
        self.ploidy = ploidy
        self.is_phased = is_phased
        self.anc_allele_file = anc_allele_file
        self.src_ind_file = src_ind_file

        self.label_data = label_data

    def run(self, file_info: dict):
        vcf_file = file_info["vcf_file"]
        rep = file_info.get("rep", None)

        # Fall back to constructor values if keys are missing (only_one_type=True case)
        self.ref_ind_file = file_info.get("ref_ind_file", self.ref_ind_file)
        self.tgt_ind_file = file_info.get("tgt_ind_file", self.tgt_ind_file)
        self.src_ind_file = file_info.get("src_ind_file", self.src_ind_file)
        self.anc_allele_file = file_info.get("mut_file", self.anc_allele_file)

        if self.nprocess <= 0:
            raise ValueError("Number of processes must be greater than 0.")

        generator = SaiWindowDataGenerator(
            vcf_file=vcf_file,
            src_ind_file=self.src_ind_file,
            ref_ind_file=self.ref_ind_file,
            tgt_ind_file=self.tgt_ind_file,
            anc_allele_file=self.anc_allele_file,
            ploidy=self.ploidy,
            is_phased=self.is_phased,
            chr_name=self.chr_name,
            win_len=self.win_len,
            win_step=self.win_step,
        )

        preprocessor = SaiFeatureVectorsPreprocessor(
            ref_ind_file=self.ref_ind_file,
            tgt_ind_file=self.tgt_ind_file,
            src_ind_file=self.src_ind_file,
            feature_config=self.feature_config,
        )

        res = mp_manager(
            job=preprocessor, data_generator=generator, nprocess=self.nprocess
        )

        if res == "error":
            raise SystemExit("Some errors occurred, stopping the program ...")

        res.sort(key=lambda x: (x["Chromosome"], x["Start"], x["End"]))
        df_res = pd.DataFrame(res)

        os.makedirs(self.output_dir, exist_ok=True)
        output_file = os.path.join(self.output_dir, f"{self.output_prefix}.features")
        df_res.to_csv(output_file, sep="\t", index=False)

        df_res["file"] = vcf_file
        return df_res
