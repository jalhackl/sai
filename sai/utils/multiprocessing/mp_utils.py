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


import h5py
import multiprocessing
import numpy as np
import pandas as pd


def rearrange_for_hdf(
    data_dict: dict, stepsize: int = 192, is_phased: bool = True
) -> dict:
    """
    Rearranges a dictionary for HDF storage by ensuring proper formatting of start/end positions and sample entries.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing genomic data with keys 'Start', 'End', 'Ref_sample', and 'Tgt_sample'.
        - 'Start' can be 'Random' or an integer.
        - 'End' is an integer indicating the end position.
        - 'Ref_sample' and 'Tgt_sample' contain lists of sample identifiers in the format 'sample_index_haplo'.
    stepsize : int, optional
        Step size to use when 'Start' is 'Random'. Defaults to 192.
    is_phased : bool, optional
        Indicates whether the data is phased. If True, haplotype numbers are preserved; if False, they are set to 0.
        Defaults to True.

    Returns
    -------
    dict
        The modified data dictionary with correctly formatted 'StartEnd', 'Ref_sample', and 'Tgt_sample' entries.
    """
    # for random, we need nonetheless integers - so we assert that the window starts at 0
    if data_dict["Start"] == "Random":
        data_dict["Start"] = 0
        data_dict["End"] = data_dict["Start"] + stepsize

    # we need one field with start and end, as it is expected by the HDF creation function
    data_dict["StartEnd"] = [data_dict["Start"], data_dict["End"]]

    # Convert sample entries to integer lists
    # Similarly, ref_sample and tgt_sample entries have to be integers (in the format [ind_nr, haplo_nr]; in case of unphased data, haplo_nr is always 0)
    if is_phased:
        data_dict["Ref_sample"] = [
            [int(entry.split("_")[1]), int(entry.split("_")[2]) - 1]
            for entry in data_dict["Ref_sample"]
        ]
        data_dict["Tgt_sample"] = [
            [int(entry.split("_")[1]), int(entry.split("_")[2]) - 1]
            for entry in data_dict["Tgt_sample"]
        ]
    else:
        data_dict["Ref_sample"] = [
            [int(entry.split("_")[1]), 0] for entry in data_dict["Ref_sample"]
        ]
        data_dict["Tgt_sample"] = [
            [int(entry.split("_")[1]), 0] for entry in data_dict["Tgt_sample"]
        ]

    return data_dict


def data_dict_to_list_for_hdf(data_dict: dict) -> list:
    """
    Converts a dictionary into a structured list for HDF storage.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing genomic data with keys representing different data categories.

    Returns
    -------
    list
        A nested list where data is organized in a specific order suitable for HDF creation.
    """


    # get the important values from the dict - in the correct order/concatenation for HDF creation
    key_groups = [
        ["Ref_genotype", "Tgt_genotype"],
        ["Label"],
        ["Ref_sample", "Tgt_sample"],
        ["StartEnd"],
        ["End"],
        ["Replicate"],
        ["Position"],
        ["Forward_relative_position"],
        ["Backward_relative_position"],
    ]

    # Extract values from the dictionary in the specified order
    value_list = [[data_dict[key] for key in group] for group in key_groups]

    return value_list


def hdf_gaia(
    hdf_file: str,
    input_entries: list,
    lock: multiprocessing.Lock = multiprocessing.Lock(),
    start_nr: int = None,  # default is None, because the start_nr is detected automatically / set by an attribute
    x_name: str = "x_0",
    y_name: str = "y",
    ind_name: str = "indices",
    pos_name: str = "pos",
    ix_name: str = "ix",
    chunk_size: int = 1,
    fwbw: bool = True,
    set_attributes: bool = True,
    vcf_file: str = None,
    file_name: str = "file",
    only_one_file: bool = False,
    store_all_positions: bool = False,
    all_pos_name: str = "positions"
) -> int:
    """
    Stores genomic data entries into an HDF5 file in a structured format.

    Parameters
    ----------
    hdf_file : str
        Path to the HDF5 file.
    input_entries : list
        List of tuples containing genomic data.
    lock : multiprocessing.Lock, optional
        Lock for synchronizing multiprocessing operations. Defaults to a new lock instance.
    start_nr : int, optional
        Starting index for dataset storage. Defaults to 0.
    x_name : str, optional
        Name of the dataset for feature data (matrices containing genotype information, usually 4 channels, 1 reference, 1 target, foward and backward differences). Defaults to "x_0".
    y_name : str, optional
        Name of the dataset for label data (matrices containing introgression information). Defaults to "y".
    ind_name : str, optional
        Name of the dataset for the individuals (two integers per individual, for individual nr and haplotype). Defaults to "indices".
    pos_name : str, optional
        Name of the dataset for positions. Defaults to "pos".
    ix_name : str, optional
        Name of the dataset for index tracking. Defaults to "ix".
    chunk_size : int, optional
        Number of entries processed at a time. Defaults to 1.
    fwbw : bool, optional
        Whether to include forward/backward information. Defaults to True.
    set_attributes : bool, optional
        Whether to set the attribute according the start_nr. Defaults to True.

    Returns
    -------
    int
        The updated starting index for the next group.
    """

    # currently chunk_size=1, because always only one new entry is added to the hdf-file

    additional_x_features = 2 if fwbw else 0

    # a - create if not existent, otherwise add entries


    with lock:
        with h5py.File(hdf_file, "a") as h5f:

            if start_nr is None:

                start_nr = h5f.attrs.get("last_index", 0)  # Get stored last index

            for i in range(0, len(input_entries) - chunk_size + 1, chunk_size):

                entry = input_entries[i]

                # create new group index, this index has to be unique for the full dataset
                group_id = start_nr + i

                act_shape0, act_shape1, act_shape2, act_shape3 = (
                    np.array(entry[0]).shape,
                    np.array(entry[1]).shape,
                    np.array(entry[2]).shape,
                    np.array(entry[3]).shape,
                )

                dset1 = h5f.create_dataset(
                    f"{i + start_nr}/{x_name}",
                    (
                        chunk_size,
                        act_shape0[0] + additional_x_features,
                        act_shape0[1],
                        act_shape0[2],
                    ),
                    compression="lzf",
                    dtype=np.uint32,
                )
                dset2 = h5f.create_dataset(
                    f"{i + start_nr}/{y_name}",
                    (chunk_size, 1, act_shape1[1], act_shape1[2]),
                    compression="lzf",
                    dtype=np.uint8,
                )
                dset3 = h5f.create_dataset(
                    f"{i + start_nr}/{ind_name}",
                    (chunk_size, act_shape2[0], act_shape2[1], act_shape2[2]),
                    compression="lzf",
                    dtype=np.uint32,
                )
                dset4 = h5f.create_dataset(
                    f"{i + start_nr}/{pos_name}",
                    (chunk_size, 1, act_shape3[0], act_shape3[1]),
                    compression="lzf",
                    dtype=np.uint32,
                )
                dset5 = h5f.create_dataset(
                    f"{i + start_nr}/{ix_name}",
                    (chunk_size, 1, 1),
                    compression="lzf",
                    dtype=np.uint32,
                )

                if store_all_positions:
                    dset6 = h5f.create_dataset(
                    f"{i + start_nr}/{all_pos_name}",
                    (chunk_size, 1, act_shape1[1], act_shape1[2]),
                    compression="lzf",
                    dtype=np.uint32,
                    )


                if vcf_file and not only_one_file:
                    dset7 = h5f.create_dataset(
                    f"{i + start_nr}/{file_name}",
                    (chunk_size, 1, 1),
                    compression="lzf",
                    dtype=h5py.string_dtype(encoding='utf-8'),
                    )

                # fill the datasets
                for k in range(chunk_size):
                    entry = input_entries[i + k]

                    # if fwbw, add two channels for forward and backward information, otherwise only 2 channels (ref/tgt) stored in entry[0]
                    features = (
                        np.concatenate([entry[0], entry[-2], entry[-1]])
                        if fwbw
                        else entry[0]
                    )
                    labels = entry[1]

                    dset1[k] = features
                    dset2[k] = labels

                    # indices
                    dset3[k] = entry[2]
                    # start and end position
                    dset4[k] = entry[3]
                    # ix
                    dset5[k] = entry[5]

                    # all positions (necessary for test data to determine tracts)
                    if store_all_positions:
                        dset6[k] = entry[-3]
                    # filename (necessary to assign corresponding test data file easily)
                    if vcf_file and not only_one_file:
                        dset7[k] = [vcf_file]

            # return the current group number

            if set_attributes:
                h5f.attrs["last_index"] = group_id + 1  # write index for next iteration
                h5f.flush()

                if vcf_file and only_one_file:
                    h5f.attrs["file_name"] = vcf_file

            # also return current index (although not needed)
            return group_id + 1




def write_tsv(file_name: str, data_dict: dict, lock: multiprocessing.Lock) -> None:
    """
    Writes the data dictionary to a TSV file.

    Parameters
    ----------
    file_name : str
        Path to the TSV file.
    data_dict : dict
        Dictionary containing the data to be written to the TSV file.
    lock : multiprocessing.Lock
        Lock for synchronizing multiprocessing operations.

    """

    converted_dict = {}

    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            array_list = value.tolist()
            converted_dict[key] = array_list
        else:
            converted_dict[key] = value

    df = pd.DataFrame([converted_dict])

    with lock:

        with open(file_name, "a") as f:
            df.to_csv(f, header=False, index=False, sep="\t")
