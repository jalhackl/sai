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
from typing import Tuple, Optional, Union
from sai.features.feature_utils import calc_freq
from sai.registries.feature_registry import FEATURE_REGISTRY


@FEATURE_REGISTRY.register("D")
def calc_d(
    ref_gts: np.ndarray,
    tgt_gts: np.ndarray,
    src_gts: np.ndarray,
    out_gts: Optional[np.ndarray] = None,
    ploidy: int = 1,
) -> float:
    """
    Calculates Patterson's D-statistic (ABBA-BABA statistic) for detecting admixture between populations.

    Parameters
    ----------
    ref_gts : np.ndarray
        A 2D numpy array representing genotypes of population 1 (reference / sister group).
    tgt_gts : np.ndarray
        A 2D numpy array representing genotypes of population 2 (target / recipient group).
    src_gts : np.ndarray
        A 2D numpy array representing genotypes of population 3 (source / donor group).
    out_gts : np.ndarray, optional
        A 2D numpy array representing genotypes of population 4 (outgroup).
        If not provided, it is assumed that the ancestral allel is always present in the outgroup, and thus the frequency of the derived allel (p4_freq) is 0.
        Default: None,
    ploidy : int, optional
        Ploidy level of the genome. Default: 1 (phased data).

    Returns
    -------
    float
        The D statistic, indicating the degree of allele sharing bias.
    """
    ref_freq, tgt_freq, src_freq, out_freq = _calc_four_pops_freq(
        ref_gts, tgt_gts, src_gts, out_gts, ploidy
    )

    abba = _calc_pattern_sum(ref_freq, tgt_freq, src_freq, out_freq, "abba")
    baba = _calc_pattern_sum(ref_freq, tgt_freq, src_freq, out_freq, "baba")

    return (abba - baba) / (abba + baba) if (abba + baba) != 0 else np.nan


@FEATURE_REGISTRY.register("fd")
def calc_fd(
    ref_gts: np.ndarray,
    tgt_gts: np.ndarray,
    src_gts: np.ndarray,
    out_gts: Optional[np.ndarray] = None,
    ploidy: int = 1,
) -> float:
    """
    Calculates fd statistic for detecting admixture between populations (Martin et al. 2015).

    Parameters
    ----------
    ref_gts : np.ndarray
        A 2D numpy array representing genotypes of population 1 (reference / sister group).
    tgt_gts : np.ndarray
        A 2D numpy array representing genotypes of population 2 (target / recipient group).
    src_gts : np.ndarray
        A 2D numpy array representing genotypes of population 3 (source / donor group).
    out_gts : np.ndarray
        A 2D numpy array representing genotypes of population 4 (outgroup).
        If not provided, it is assumed that the ancestral allel is always present in the outgroup, and thus the frequency of the derived allel (p4_freq) is 0.
    ploidy : int, optional
        Ploidy level of the genome. Default: 1 (phased data).

    Returns
    -------
    float
        The fd statistic.
    """
    ref_freq, tgt_freq, src_freq, out_freq = _calc_four_pops_freq(
        ref_gts, tgt_gts, src_gts, out_gts, ploidy
    )

    abba_n = _calc_pattern_sum(ref_freq, tgt_freq, src_freq, out_freq, "abba")
    baba_n = _calc_pattern_sum(ref_freq, tgt_freq, src_freq, out_freq, "baba")

    dnr_freq = np.maximum(tgt_freq, src_freq)

    abba_d = _calc_pattern_sum(ref_freq, dnr_freq, dnr_freq, out_freq, "abba")
    baba_d = _calc_pattern_sum(ref_freq, dnr_freq, dnr_freq, out_freq, "baba")

    return (abba_n - baba_n) / (abba_d - baba_d) if (abba_d - baba_d) != 0 else np.nan


@FEATURE_REGISTRY.register("fhom")
def calc_fhom(
    ref_gts: np.ndarray,
    tgt_gts: np.ndarray,
    src_gts: np.ndarray,
    out_gts: Optional[np.ndarray] = None,
    ploidy: int = 1,
) -> float:
    """
    Calculates fhom statistic for detecting admixture between populations (Martin et al. 2015).

    Parameters
    ----------
    ref_gts : np.ndarray
        A 2D numpy array representing genotypes of population 1 (reference / sister group).
    tgt_gts : np.ndarray
        A 2D numpy array representing genotypes of population 2 (target / recipient group).
    src_gts : np.ndarray
        A 2D numpy array representing genotypes of population 3 (source / donor group).
    out_gts : np.ndarray
        A 2D numpy array representing genotypes of population 4 (outgroup).
        If not provided, it is assumed that the ancestral allel is always present in the outgroup, and thus the frequency of the derived allel (p4_freq) is 0.
    ploidy : int, optional
        Ploidy level of the genome. Default: 1 (phased data).

    Returns
    -------
    float
        The fhom statistic.
    """
    ref_freq, tgt_freq, src_freq, out_freq = _calc_four_pops_freq(
        ref_gts, tgt_gts, src_gts, out_gts, ploidy
    )

    abba_n = _calc_pattern_sum(ref_freq, tgt_freq, src_freq, out_freq, "abba")
    baba_n = _calc_pattern_sum(ref_freq, tgt_freq, src_freq, out_freq, "baba")

    dnr_freq = tgt_freq

    abba_d = _calc_pattern_sum(ref_freq, dnr_freq, dnr_freq, out_freq, "abba")
    baba_d = _calc_pattern_sum(ref_freq, dnr_freq, dnr_freq, out_freq, "baba")

    return (abba_n - baba_n) / (abba_d - baba_d) if (abba_d - baba_d) != 0 else np.nan


@FEATURE_REGISTRY.register("Dplus")
def calc_d_plus(
    ref_gts: np.ndarray,
    tgt_gts: np.ndarray,
    src_gts: np.ndarray,
    out_gts: Optional[np.ndarray] = None,
    ploidy: int = 1,
) -> float:
    """
    Calculates the D+ statistic for detecting admixture between populations (Fang et al. 2024).

    Parameters
    ----------
    ref_gts : np.ndarray
        A 2D numpy array representing genotypes of population 1 (reference / sister group).
    tgt_gts : np.ndarray
        A 2D numpy array representing genotypes of population 2 (target / recipient group).
    src_gts : np.ndarray
        A 2D numpy array representing genotypes of population 3 (source / donor group).
    out_gts : np.ndarray, optional
        A 2D numpy array representing genotypes of population 4 (outgroup).
        If not provided, it is assumed that the ancestral allel is always present in the outgroup, and thus the frequency of the derived allel (p4_freq) is 0.
        Default: None,
    ploidy : int, optional
        Ploidy level of the genome. Default: 1 (phased data).

    Returns
    -------
    float
        The D+ statistic.
    """
    ref_freq, tgt_freq, src_freq, out_freq = _calc_four_pops_freq(
        ref_gts, tgt_gts, src_gts, out_gts, ploidy
    )

    abba = _calc_pattern_sum(ref_freq, tgt_freq, src_freq, out_freq, "abba")
    baba = _calc_pattern_sum(ref_freq, tgt_freq, src_freq, out_freq, "baba")
    baaa = _calc_pattern_sum(ref_freq, tgt_freq, src_freq, out_freq, "baaa")
    abaa = _calc_pattern_sum(ref_freq, tgt_freq, src_freq, out_freq, "abaa")

    return (
        (abba - baba + baaa - abaa) / (abba + baba + baaa + abaa)
        if (abba + baba + baaa + abaa) != 0
        else np.nan
    )


@FEATURE_REGISTRY.register("Danc")
def calc_d_anc(
    ref_gts: np.ndarray,
    tgt_gts: np.ndarray,
    src_gts: np.ndarray,
    out_gts: Optional[np.ndarray] = None,
    ploidy: int = 1,
) -> float:
    """
    Calculates the D ancestral statistic for detecting admixture between populations (Fang et al. 2024).

    Parameters
    ----------
    ref_gts : np.ndarray
        A 2D numpy array representing genotypes of population 1 (reference / sister group).
    tgt_gts : np.ndarray
        A 2D numpy array representing genotypes of population 2 (target / recipient group).
    src_gts : np.ndarray
        A 2D numpy array representing genotypes of population 3 (source / donor group).
    out_gts : np.ndarray, optional
        A 2D numpy array representing genotypes of population 4 (outgroup).
        If not provided, it is assumed that the ancestral allel is always present in the outgroup, and thus the frequency of the derived allel (p4_freq) is 0.
        Default: None,
    ploidy : int, optional
        Ploidy level of the genome. Default: 1 (phased data).

    Returns
    -------
    float
        The D ancestral statistic.
    """
    ref_freq, tgt_freq, src_freq, out_freq = _calc_four_pops_freq(
        ref_gts, tgt_gts, src_gts, out_gts, ploidy
    )

    baaa = _calc_pattern_sum(ref_freq, tgt_freq, src_freq, out_freq, "baaa")
    abaa = _calc_pattern_sum(ref_freq, tgt_freq, src_freq, out_freq, "abaa")

    return (baaa - abaa) / (baaa + abaa) if (baaa + abaa) != 0 else np.nan


def _calc_four_pops_freq(
    ref_gts: np.ndarray,
    tgt_gts: np.ndarray,
    src_gts: np.ndarray,
    out_gts: Union[np.ndarray, None],
    ploidy: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates allele frequencies for four populations given their genotype matrices.

    Parameters
    ----------
    ref_gts : np.ndarray
        Genotype matrix for the reference population.
    tgt_gts : np.ndarray
        Genotype matrix for the target population.
    src_gts : np.ndarray
        Genotype matrix for the source population.
    out_gts : np.ndarray or None
        Genotype matrix for the outgroup. If None, the outgroup frequency is assumed to be 0 at all loci.
    ploidy : int, optional
        Ploidy level of the genomes. Default: 1 (phased data).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Allele frequencies for (ref, tgt, src, out) populations.
    """
    ref_freq = calc_freq(ref_gts, ploidy)
    tgt_freq = calc_freq(tgt_gts, ploidy)
    src_freq = calc_freq(src_gts, ploidy)
    if out_gts is None:
        out_freq = np.zeros_like(ref_freq)
    else:
        out_freq = calc_freq(out_gts, ploidy)

    return ref_freq, tgt_freq, src_freq, out_freq


def _calc_pattern_sum(
    ref_freq: np.ndarray,
    tgt_freq: np.ndarray,
    src_freq: np.ndarray,
    out_freq: np.ndarray,
    pattern: str,
) -> float:
    """
    Applies an ABBA-like pattern and returns the sum over loci of the transformed frequency products.

    Parameters
    ----------
    ref_freq:
        Allele frequencies for the reference population (no introgression) across loci.
    tgt_freq:
        Allele frequencies for the target population (receive introgression) across loci.
    src_freq:
        Allele frequencies for the source population (provide introgression) across loci.
    out_freq:
        Allele frequencies for the outgroup across loci.
    pattern : str
        A 4-character pattern string (e.g., 'abba'), where:
        - 'a': use 1 - freq
        - 'b': use freq

    Returns
    -------
    float
        Sum over loci of the product defined by the pattern.

    Raises
    ------
    ValueError
        - If the pattern string is not exactly four characters long.
        - If the pattern contains characters other than 'a' or 'b'.
    """
    if len(pattern) != 4:
        raise ValueError("Pattern must be a four-character string.")

    freqs = [ref_freq, tgt_freq, src_freq, out_freq]
    product = np.ones_like(ref_freq)

    for i, c in enumerate(pattern.lower()):
        if c == "a":
            product *= 1 - freqs[i]
        elif c == "b":
            product *= freqs[i]
        else:
            raise ValueError(
                f"Invalid character '{c}' in pattern. Only 'a' and 'b' allowed."
            )

    return float(np.sum(product))
