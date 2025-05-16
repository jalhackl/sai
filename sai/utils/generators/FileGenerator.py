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

    def __init__(self, file_location: str, file_ending: str = ".vcf"):
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

        if len(self.file_list) == 0:
            raise ValueError(f"No files with extension '{file_ending}' found in '{file_location}'.")

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

        for ifile, file in enumerate(self.file_list):
            yield {"rep": ifile, "vcf_file": file}

    def __len__(self):
        return len(self.file_list)
