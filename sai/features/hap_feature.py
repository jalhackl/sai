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
from scipy.spatial import distance_matrix
from sai.registries.feature_registry import FEATURE_REGISTRY


@FEATURE_REGISTRY.register("n-ton")
def calc_n_ton(gt: np.ndarray, ploidy: int = 1) -> np.ndarray:
    """
    Calculates per-sample frequency spectra based on observed values
    across genomic sites.

    For each sample, the function counts how many times the sample
    is involved in a site with total allele count equal to k.
    Sites where the sample has 0 are ignored for that sample.

    Parameters
    ----------
    gt : np.ndarray
        Genotype matrix of shape (num_sites, num_samples).
        Values are assumed to be integer counts; allele coding is not assumed.
    ploidy : int, optional
        Ploidy level of the genome. Set to 1 for phased data; >1 for unphased data. Default: 1.

    Returns
    -------
    np.ndarray
        A 2D array of shape (num_samples, K), where each row is the frequency
        spectrum for a sample, and entry [i, k] counts the number of sites
        where sample i is nonzero and the total site count equals k.

        The first column (index 0) is zeroed out to ignore sites where the sample
        had no contribution.
    """
    mut_num, sample_num = gt.shape
    counts = (gt > 0) * gt.sum(axis=1, keepdims=True)

    spectra = np.array(
        [
            np.bincount(
                counts[:, idx].astype("int64"), minlength=sample_num * ploidy + 1
            )
            for idx in range(sample_num)
        ]
    )

    # Exclude non-segregating sites
    spectra[:, 0] = 0

    return spectra


@FEATURE_REGISTRY.register("dist")
def calc_dist(gt1: np.ndarray, gt2: np.ndarray) -> np.ndarray:
    """
    Calculates pairwise Euclidean distances between individuals
    from two genotype matrices.

    Each column in `gt1` and `gt2` represents an individual, and
    each row represents a variant site. The function computes all
    pairwise distances between individuals in `gt1` and `gt2`, and
    returns a sorted distance matrix (ascending along each row).

    Parameters
    ----------
    gt1 : np.ndarray
        A 2D genotype matrix of shape (num_sites, num_individuals_1).
    gt2 : np.ndarray
        A 2D genotype matrix of shape (num_sites, num_individuals_2).

    Returns
    -------
    np.ndarray
        A 2D array of shape (num_individuals_2, num_individuals_1),
        containing sorted pairwise Euclidean distances. Each row
        corresponds to one individual in `gt2`, and contains its
        distances to all individuals in `gt1`, sorted in ascending order.
    """
    dists = distance_matrix(gt2.T, gt1.T)  # shape: (n2, n1)
    dists.sort(axis=1)  # sort distances for each row (i.e., each gt2 sample)

    return dists
