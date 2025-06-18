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


import os
from sai.utils.generators import DataGenerator
from sai.utils import collect_files


class FileGenerator(DataGenerator):
    """
    Generates file paths for processing from a given file or directory.

    This class collects all VCF (.vcf) files from a specified location (either a single file or
    all files within a directory) and provides them one by one via the `get` method.

    Attributes
    ----------
    file_list : list of str
        A list containing the paths of all discovered VCF files.
    """

    def __init__(
        self,
        file_location: str,
        file_ending: str = ".vcf",
        only_one_type=True,
        different_names=True,
        return_rep=True,
        sample_file_suffix="ind.list"
    ):
        """
        Initializes the FileGenerator instance by collecting valid VCF files.

        Parameters
        ----------
        file_location : str
            The path to a single VCF file or a directory containing VCF files.

        Raises
        ------
        SystemExit
            If the specified path is neither a file nor a directory.
        ValueError
            If no VCF files are found in the given location.
        """

        self.file_list = collect_files(file_location, file_ending)
        self.only_one_type = only_one_type
        self.different_names = different_names
        self.return_rep = return_rep
        self.sample_file_suffix = sample_file_suffix

        if len(self.file_list) == 0:
            raise ValueError(
                f"No files with extension '{file_ending}' found in '{file_location}'."
            )

    def _find_matching_file(self, folder, suffix):
        """Finds the first file in the folder that ends with the given suffix."""
        for fname in os.listdir(folder):
            if fname.endswith(suffix):
                return os.path.join(folder, fname)
        return None

    def get(self):
        """
        Yields dictionaries containing the file index and file path.

        Yields
        ------
        dict
            A dictionary with keys:
            - 'rep' : int, the index of the file.
            - 'vcf_file' : str, the path to the VCF file.
        """

        if self.only_one_type:

            for ifile, file in enumerate(self.file_list):
                if self.return_rep:
                    yield {"rep": ifile, "vcf_file": file}
                else:

                    yield {"vcf_file": file}

        else:

            if not self.different_names:
                for idx, vcf_path in enumerate(self.file_list):
                    base_path = os.path.splitext(vcf_path)[0]  # remove .vcf
                    if self.return_rep:
                        result = {
                            "rep": idx,
                            "vcf_file": vcf_path,
                            "ref_ind_file": (
                                base_path + ".ref." + self.sample_file_suffix
                                if os.path.exists(base_path + ".ref." + self.sample_file_suffix)
                                else None
                            ),
                            "src_ind_file": (
                                base_path + ".src.ind.list"
                                if os.path.exists(base_path + ".src." + self.sample_file_suffix)
                                else None
                            ),
                            "tgt_ind_file": (
                                base_path + ".tgt." + self.sample_file_suffix
                                if os.path.exists(base_path + ".tgt." + self.sample_file_suffix)
                                else None
                            ),
                            "mut_file": (
                                base_path + ".tgt.mut.list"
                                if os.path.exists(base_path + ".tgt.mut.list")
                                else None
                            ),
                        }
                    else:
                        result = {
                            "vcf_file": vcf_path,
                            "ref_ind_file": (
                                base_path + ".ref." + self.sample_file_suffix
                                if os.path.exists(base_path + ".ref." + self.sample_file_suffix)
                                else None
                            ),
                            "src_ind_file": (
                                base_path + ".src." + self.sample_file_suffix
                                if os.path.exists(base_path + ".src." + self.sample_file_suffix)
                                else None
                            ),
                            "tgt_ind_file": (
                                base_path + ".tgt." + self.sample_file_suffix
                                if os.path.exists(base_path + ".tgt." + self.sample_file_suffix)
                                else None
                            ),
                            "mut_file": (
                                base_path + ".tgt.mut.list"
                                if os.path.exists(base_path + ".tgt.mut.list")
                                else None
                            ),
                        }

                    yield result

            else:

                for idx, vcf_path in enumerate(self.file_list):
                    folder = os.path.dirname(vcf_path)
                    if self.return_rep:
                        result = {
                            "rep": idx,
                            "vcf_file": vcf_path,
                            "ref_ind_file": self._find_matching_file(
                                folder, ".ref." + self.sample_file_suffix
                            ),
                            "src_ind_file": self._find_matching_file(
                                folder, ".src." + self.sample_file_suffix
                            ),
                            "tgt_ind_file": self._find_matching_file(
                                folder, ".tgt." + self.sample_file_suffix
                            ),
                            "mut_file": self._find_matching_file(
                                folder, ".tgt.mut.list"
                            ),
                        }
                    else:
                        result = {
                            "vcf_file": vcf_path,
                            "ref_ind_file": self._find_matching_file(
                                folder, ".ref." + self.sample_file_suffix
                            ),
                            "src_ind_file": self._find_matching_file(
                                folder, ".src." + self.sample_file_suffix
                            ),
                            "tgt_ind_file": self._find_matching_file(
                                folder, ".tgt." + self.sample_file_suffix
                            ),
                            "mut_file": self._find_matching_file(
                                folder, ".tgt.mut.list"
                            ),
                        }

                    yield result

    def __len__(self):
        return len(self.file_list)
