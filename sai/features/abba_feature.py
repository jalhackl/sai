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
from sai.features.feature_utils import calc_freq
from sai.registries.feature_registry import FEATURE_REGISTRY


def S_ABBA_BABA_calc(
    pop1_freq: np.ndarray,
    pop2_freq: np.ndarray,
    pop3_freq: np.ndarray,
    outgroup_freq: float = 0,
    return_numerator_for_S: bool = False,
) -> float:
    """
    Calculate the ABBA-BABA statistic, which is used in population genetics to test
    for asymmetry in allele sharing between populations and an outgroup. The function
    computes either the D-statistic (ABBA - BABA) / (ABBA + BABA) or the numerator
    (ABBA - BABA) based on the specified parameters.

    Parameters
    ----------
    pop1_freq : np.ndarray
        An array of allele frequencies for population 1 at each locus. (Usually the reference population)

    pop2_freq : np.ndarray
        An array of allele frequencies for population 2 at each locus. (Usually the target population)

    pop3_freq : np.ndarray
        An array of allele frequencies for population 3 at each locus. (Usually the source population)

    outgroup_freq : float, optional, default=0
        The allele frequency in the outgroup population. The default is 0.

    return_numerator_for_S : bool, optional, default=False
        If True, the function returns the numerator (ABBA - BABA) instead of
        the D-statistic. The default is False.

    Returns
    -------
    float
        The D-statistic (if `return_numerator_for_S` is False) or the numerator
        (ABBA - BABA) if `return_numerator_for_S` is True. Returns `nan` if the
        denominator is zero.

    Notes
    -----
    The ABBA-BABA statistic is commonly used in studies of introgression or
    gene flow between populations, where a high value of the D-statistic suggests
    gene flow from population 1 to 2, and a low value suggests gene flow in the
    opposite direction.
    """

    abbavec = (1.0 - pop1_freq) * pop2_freq * pop3_freq * (1 - outgroup_freq)
    babavec = pop1_freq * (1.0 - pop2_freq) * pop3_freq * (1 - outgroup_freq)

    # Summing up across loci
    abba = np.sum(abbavec)
    baba = np.sum(babavec)

    if not return_numerator_for_S:
        # Compute D-statistic
        if (abba + baba) > 0:
            return (abba - baba) / (abba + baba)
        else:
            return float("nan")
    else:
        return abba - baba


def compute_ABBA_BABA_D(
    src_gts: np.ndarray,
    ref_gts: np.ndarray,
    tgt_gts: np.ndarray,
    out_gts: np.ndarray = None,
    ploidy: int = 1,
) -> float:
    """
    Computes Patterson's D-statistic (ABBA-BABA statistic) for detecting admixture between populations.

    Parameters
    ----------
    src_gts : np.ndarray
        A 2D numpy array representing haplotypes of population 1 (source).
    ref_gts : np.ndarray
        A 2D numpy array representing haplotypes of population 2 (reference / sister group).
    tgt_gts : np.ndarray
        A 2D numpy array representing haplotypes of population 3 (target).
    out_gts : np.ndarray
        A 2D numpy array representing haplotypes of population 4 (outgroup).
        If not provided, it is assumed that the ancestral allel is always present in the outgroup, and thus the frequency of the derived allel (p4_freq) is 0.

    Returns
    -------
    float
        The D-statistic value, indicating the degree of allele sharing bias.
    """

    # Compute allele frequencies using the provided calc_freq function
    src_freq = calc_freq(src_gts, ploidy=ploidy)
    ref_freq = calc_freq(ref_gts, ploidy=ploidy)
    tgt_freq = calc_freq(tgt_gts, ploidy=ploidy)
    if out_gts is not None:
        out_freq = calc_freq(out_gts, ploidy=ploidy)
    else:
        out_freq = 0

    D_stat = S_ABBA_BABA_calc(
        ref_freq,
        tgt_freq,
        src_freq,
        outgroup_freq=out_freq,
        return_numerator_for_S=False,
    )

    return D_stat


def compute_fd(
    src_gts: np.ndarray,
    ref_gts: np.ndarray,
    tgt_gts: np.ndarray,
    out_gts: np.ndarray = None,
    ploidy: int = 1,
    use_hom: bool = False,
) -> float:
    """
    Computes fD-statistic for detecting admixture between populations (Martin 2015).

    Parameters
    ----------

    src_gts : np.ndarray
        A 2D numpy array representing haplotypes of population 3 (source).
    ref_gts : np.ndarray
        A 2D numpy array representing haplotypes of population 1 (reference / sister group).
    tgt_gts : np.ndarray
        A 2D numpy array representing haplotypes of population 2 (target).
    out_gts : np.ndarray
        A 2D numpy array representing haplotypes of population 4 (outgroup).
        If not provided, it is assumed that the ancestral allel is always present in the outgroup, and thus the frequency of the derived allel (p4_freq) is 0.
    use_hom: boolean
        If true, compute fhom instead of fd (Martin 2015, p.254)

    Returns
    -------
    float
        The fd-statistic value.
    """

    # Compute allele frequencies using the provided calc_freq function
    src_freq = calc_freq(src_gts)  # 3
    ref_freq = calc_freq(ref_gts)  # 1
    tgt_freq = calc_freq(tgt_gts)  # 2
    if out_gts is not None:
        out_freq = calc_freq(out_gts, ploidy=ploidy)
    else:
        out_freq = 0

    fd_nominator = S_ABBA_BABA_calc(
        ref_freq,
        tgt_freq,
        src_freq,
        outgroup_freq=out_freq,
        return_numerator_for_S=True,
    )

    if not use_hom:
        donor_pop_freq = np.maximum(src_freq, tgt_freq)
        fd_denumerator = S_ABBA_BABA_calc(
            ref_freq,
            donor_pop_freq,
            donor_pop_freq,
            outgroup_freq=out_freq,
            return_numerator_for_S=True,
        )
    else:
        fd_denumerator = S_ABBA_BABA_calc(
            ref_freq,
            tgt_freq,
            tgt_freq,
            outgroup_freq=out_freq,
            return_numerator_for_S=True,
        )

    fd = fd_nominator / fd_denumerator
    return fd


def compute_D_plus(
    src_gts: np.ndarray,
    ref_gts: np.ndarray,
    tgt_gts: np.ndarray,
    out_gts: np.ndarray = None,
    ploidy: int = 1,
    compute_D_ancestral: bool = False,
) -> float:
    """
    Computes the D+-statistic for detecting admixture between populations (Lopez-Fang 2024).

    Parameters
    ----------
    src_gts : np.ndarray
        A 2D numpy array representing haplotypes of population 1 (source).
    ref_gts : np.ndarray
        A 2D numpy array representing haplotypes of population 2 (reference / sister group).
    tgt_gts : np.ndarray
        A 2D numpy array representing haplotypes of population 3 (target).
    out_gts : np.ndarray
        A 2D numpy array representing haplotypes of population 4 (outgroup).
        If not provided, it is assumed that the ancestral allel is always present in the outgroup, and thus the frequency of the derived allel (p4_freq) is 0.
    compute_D_ancestral : bool
        compute D_ancestral (which basically consists of the D+-terms w/o the standard D-terms, see Lopez-Fang 2024)

    Returns
    -------
    float
        The D+-statistic value (Lopez-Fang 2024).
    """

    # Compute allele frequencies using the provided calc_freq function
    src_freq = calc_freq(src_gts, ploidy=ploidy)
    ref_freq = calc_freq(ref_gts, ploidy=ploidy)
    tgt_freq = calc_freq(tgt_gts, ploidy=ploidy)
    if out_gts is not None:
        out_freq = calc_freq(out_gts, ploidy=ploidy)
    else:
        out_freq = 0

    # Compute ABBA and BABA site patterns
    abbavec = (1.0 - ref_freq) * tgt_freq * src_freq * (1.0 - out_freq)
    babavec = ref_freq * (1.0 - tgt_freq) * src_freq * (1.0 - out_freq)

    # Compute BAAA and ABAA site patterns
    baaavec = ref_freq * (1.0 - tgt_freq) * (1.0 - src_freq) * (1.0 - out_freq)
    abaavec = (1.0 - ref_freq) * tgt_freq * (1.0 - src_freq) * (1.0 - out_freq)

    # Summing up across loci
    abba = np.sum(abbavec)
    baba = np.sum(babavec)

    baaa = np.sum(baaavec)
    abaa = np.sum(abaavec)

    baaa_abaa_difference = baaa - abaa
    baaa_abaa_addition = baaa + abaa
    if compute_D_ancestral:
        return baaa_abaa_difference / baaa_abaa_addition

    abba_baba_difference = abba - baba
    abba_baba_addition = abba + baba

    D_plus = (abba_baba_difference + baaa_abaa_difference) / (
        abba_baba_addition + baaa_abaa_addition
    )
    return D_plus
