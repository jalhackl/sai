# Copyright 2025 Xin Huang
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


import numpy as np


def calc_freq(gts: np.ndarray, ploidy: int = 1) -> np.ndarray:
    """
    Calculates allele frequencies, supporting both phased and unphased data.

    Parameters
    ----------
    gts : np.ndarray
        A 2D numpy array where each row represents a locus and each column represents an individual.
    ploidy : int, optional
        Ploidy level of the organism. If ploidy=1, the function assumes phased data and calculates
        frequency by taking the mean across individuals. For unphased data, it calculates frequency by
        dividing the sum across individuals by the total number of alleles. Default is 1.

    Returns
    -------
    np.ndarray
        An array of allele frequencies for each locus.
    """
    if ploidy == 1:
        return np.mean(gts, axis=1)
    else:
        return np.sum(gts, axis=1) / (gts.shape[1] * ploidy)
