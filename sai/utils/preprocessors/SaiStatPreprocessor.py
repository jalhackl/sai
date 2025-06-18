import os
import pandas as pd
from sai.utils.generators import ChunkGenerator
from sai.utils.preprocessors import SaiChunkPreprocessor


class SaiStatPreprocessor:
    def __init__(
        self,
        feature_config: str,
        output_dir: str,
        vcf_file: str = None,
        chr_name: str = "1",
        ref_ind_file: str = None,
        tgt_ind_file: str = None,
        src_ind_file: str = None,
        win_len: int = 40000,
        win_step: int = 10000,
        num_src: int = 1,
        anc_allele_file: str = None,
        output_prefix: str = "saifeatures",
        num_workers: int = 1,
        ploidy: int = 2,
        is_phased: bool = True,
        mut_file: str = None,
    ):
        """
        Initialize the StatProcessor with all required parameters.
        """
        self.vcf_file = vcf_file
        self.chr_name = chr_name
        self.ref_ind_file = ref_ind_file
        self.tgt_ind_file = tgt_ind_file
        self.src_ind_file = src_ind_file
        self.win_len = win_len
        self.win_step = win_step
        self.num_src = num_src
        self.anc_allele_file = anc_allele_file
        self.output_prefix = output_prefix
        self.feature_config = feature_config
        self.num_workers = num_workers
        self.ploidy = ploidy
        self.is_phased = is_phased
        self.mut_file = mut_file

        self.output_dir = output_dir

        output_file = os.path.join(self.output_dir, f"{self.output_prefix}.features")
        self.output_file = output_file

    def run(self, **kwargs):
        """
        Processes and scores genomic data by generating windowed data and feature vectors.
        """
        file_info = kwargs

        self.vcf_file = file_info["vcf_file"]
        rep = file_info.get("rep", None)

        # Fall back to constructor values if keys are missing (only_one_type=True case)
        self.ref_ind_file = file_info.get("ref_ind_file", self.ref_ind_file)
        self.tgt_ind_file = file_info.get("tgt_ind_file", self.tgt_ind_file)
        self.src_ind_file = file_info.get("src_ind_file", self.src_ind_file)
        self.mut_file = file_info.get("mut_file", self.mut_file)

        generator = ChunkGenerator(
            vcf_file=self.vcf_file,
            chr_name=self.chr_name,
            window_size=self.win_len,
            step_size=self.win_step,
            num_chunks=self.num_workers,
        )

        preprocessor = SaiChunkPreprocessor(
            vcf_file=self.vcf_file,
            ref_ind_file=self.ref_ind_file,
            tgt_ind_file=self.tgt_ind_file,
            src_ind_file=self.src_ind_file,
            win_len=self.win_len,
            win_step=self.win_step,
            output_file=self.output_file,
            feature_config=self.feature_config,
            anc_allele_file=self.anc_allele_file,
            num_src=self.num_src,
            ploidy=self.ploidy,
            is_phased=self.is_phased,
            mut_file=self.mut_file,
        )

        items = []
        for params in generator.get():

            items.extend(preprocessor.run(**params))

        res = items

        # res.sort(key=lambda x: (x["Chromosome"], x["Start"], x["End"]))
        res.sort(key=lambda x: (x["chr_name"], x["start"], x["end"]))

        df_res = pd.DataFrame(res)

        os.makedirs(self.output_dir, exist_ok=True)

        df_res.to_csv(self.output_file, sep="\t", index=False)

        df_res["file"] = self.vcf_file

        return df_res
