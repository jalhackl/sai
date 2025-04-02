def return_segsite_occurrence_vector(
    tgt_gt, ploidy: int = 1, remove_non_segregating: bool = False
):
    """
    Compute the number of segregating sites where the mutant allele occurs exactly i times in the sample.

    Arguments:
        tgt_gt (numpy.ndarray): Genotype matrix (mutations x samples).
        remove_non_segregating (bool): if True, return a vector which only proviedes the counts for at least one segregating site, i.e. the first entry is removed

    Returns:
        seg_sites (numpy.ndarray): A vector where index i represents the count of sites where
                                   the mutant allele appears exactly i times.

    """
    # Compute the sum of mutant alleles per mutation (sum across rows)
    mutant_counts = np.sum(tgt_gt, axis=1)

    seg_sites = np.bincount(mutant_counts, minlength=tgt_gt.shape[1] * ploidy)
    if remove_non_segregating:
        seg_sites = seg_sites[1:]

    return seg_sites


def num_segregating_sites(
    gts: np.ndarray, return_frequencies: bool = False, ploidy: int = 1
) -> Union[int, tuple[int, np.ndarray]]:
    """
    Computes the number of segregating sites in a genotype matrix.

    A site (locus) is considered segregating if it contains at least two different alleles (i.e.,
    the allele frequency is between 0 and 1).

    Parameters
    ----------
    gts : np.ndarray
        A 2D NumPy array where each row represents a locus and each column represents an individual.
    return_frequencies : bool, optional
        If True, also returns the allele frequencies at segregating sites. Default is False.
    ploidy : int, optional
        The ploidy level of the organism (default is 1).

    Returns
    -------
    int
        The number of segregating sites.
    tuple[int, np.ndarray], optional
        If `return_frequencies` is True, returns a tuple containing the number of segregating sites
        and an array of their frequencies.
    """
    # Compute allele frequencies for each locus
    gts_freq = calc_freq(gts, ploidy=ploidy)

    # Identify segregating sites (where 0 < p < 1)
    segregating_mask = (gts_freq > 0) & (gts_freq < 1)

    num_S = np.sum(segregating_mask)

    if return_frequencies:
        return num_S, gts_freq[segregating_mask]
    return num_S


def theta_W(gts: np.ndarray, ploidy: int = 1) -> float:
    """
    Calculates Watterson's Theta from genotype data.

    Parameters
    ----------
    gts : np.ndarray
        A 2D numpy array where rows represent genetic sites and columns represent individuals.
        Each element in the array represents the allele at a given site for an individual, typically
        0 (reference allele) or 1 (alternate allele).

    Returns
    -------
    float
        theta_W
    """
    S = num_segregating_sites(gts)
    individuals = gts.shape[1] * ploidy
    harmonic_number = sum(1 / i for i in range(1, individuals + 1))
    theta_W = S / harmonic_number
    return theta_W


def theta_pi(gts: np.ndarray, ploidy: int = 1, metric: str = "cityblock") -> float:
    """
    Calculates theta_pi from genotype data,
    following Zeng et al. 2006: Statistical Tests for Detecting Positive Selection by Utilizing High-Frequency Variants
    Should in principle (using cityblock metric) also work in a reasonable way for unphased data

    Parameters
    ----------
    gts : np.ndarray
        A 2D numpy array where rows represent genetic sites and columns represent individuals.
    ploidy : int, optional
        The number of chromosome copies per individual (default is 1 for haploid).
    metric: str, optional
        Metric to be used for sequence comparisons, default cityblock, i.e. counting number of differences

    Returns
    -------
    float
        theta_pi, estimator  of nucleotide diversity.
    """
    from math import comb
    from scipy.spatial.distance import pdist

    individuals = gts.shape[1] * ploidy
    combination_prefactor = 1 / (comb(individuals, 2))

    pairwise_differences = np.sum(pdist(gts.T, metric=metric))

    theta_pi_value = combination_prefactor * pairwise_differences
    return theta_pi_value


def theta_pi_v2(gts: np.ndarray, ploidy: int = 1) -> float:
    """
    Compute theta_pi from a genotype matrix,
    also following Zeng et al. 2006: Statistical Tests for Detecting Positive Selection by Utilizing High-Frequency Variants

    The function calculates θπ (nucleotide diversity) using a combination of the site frequency
    spectrum (SFS) and pairwise comparisons between individuals. It normalizes the result by the
    total number of possible pairwise comparisons, given the ploidy of the individuals.
    Is probably biased for unphased data.

    Parameters
    ----------
    gts : np.ndarray
        A 2D NumPy array of shape (mutations x samples) where each entry represents the number of
        mutant alleles (0, 1, or 2, ...) at each site for each individual.

    ploidy : int, optional, default=1
        The ploidy of the population. For example, ploidy = 2 for diploid organisms and ploidy = 1 for haploid organisms.

    Returns
    -------
    float
        The nucleotide diversity (θπ) for the given genotype matrix, considering the number of pairwise comparisons.

    """
    from math import comb

    individuals = gts.shape[1] * ploidy
    xi = return_segsite_occurrence_vector(
        gts, remove_non_segregating=False, ploidy=ploidy
    )

    combination_prefactor = 1 / (comb(individuals, 2))

    pi_sum = 0
    # the first term (all ancestral) is automatically 0, the last term (all derived) we remove from the calculation
    for ie, entry in enumerate(xi):
        pi_sum = pi_sum + ie * (individuals - ie) * entry

    theta_pi = pi_sum * combination_prefactor

    return theta_pi


def theta_pi_maladapt(gts: np.ndarray, ploidy: int = 1) -> float:
    """
    Calculates theta_pi from genotype data as in MaLAdapt.

    Parameters
    ----------
    gts : np.ndarray
        A 2D numpy array where rows represent genetic sites and columns represent individuals.
    ploidy : int, optional
        The number of chromosome copies per individual (default is 1 for haploid).

    Returns
    -------
    float
        theta_pi, estimator  of nucleotide diversity.
    """
    # Compute segregating sites (S) and allele frequencies (S_freq)
    S, S_freq = num_segregating_sites(gts, return_frequencies=True)

    # Total number of alleles
    individuals = gts.shape[1] * ploidy

    # Compute π using allele frequencies
    pi = sum(2 * S_freq * (1.0 - S_freq))

    # Adjust for sample size
    theta_pi_value = pi * individuals / (individuals - 1)

    return theta_pi_value


def compute_Fay_Wu_theta_h(gts: np.ndarray, ploidy: int = 1) -> float:
    """
    Computes Fay and Wu's theta_H based on genotype matrix (gts).

    This function calculates theta_H, a measure of nucleotide diversity that
    accounts for the number of segregating sites in a given genotype matrix.
    Theta_H is commonly used in population genetics to estimate genetic diversity.

    Parameters
    ----------
    gts : np.ndarray
        A 2D numpy array where rows represent loci and columns represent individuals.
        The matrix contains genotype information, where alleles are coded numerically.

    ploidy : int, optional, default=1
        The ploidy level of the individuals. For diploid organisms, ploidy would be 2.

    Returns
    -------
    float
        The calculated Fay and Wu's theta_H value.

    Notes
    -----
    Theta_H is calculated using the formula:
    theta_H = (2 * sum(x_vals^2 * xi[:-1])) / (n * (n - 1))
    where:
        - xi is the vector of derived allele counts,
        - x_vals is the range of indices for segregating sites,
        - n is the total haploid sample size.

    """

    individuals = gts.shape[1] * ploidy
    xi = return_segsite_occurrence_vector(
        gts, remove_non_segregating=False, ploidy=ploidy
    )

    x_vals = np.arange(len(xi) - 1)

    num = 2 * x_vals**2 * xi[:-1]
    denom = individuals * (individuals - 1)

    theta_h = np.sum(num) / denom
    return theta_h


def heterozygosity(tgt_gts: np.ndarray, ploidy: int = 1) -> float:
    """
    Computes the expected heterozygosity (H) for a given genotype matrix.

    Expected heterozygosity is calculated as:
        H = mean(2 * p * (1 - p)),
    where p is the allele frequency at each locus.

    Parameters
    ----------
    tgt_gts : np.ndarray
        A 2D numpy array where each row represents a locus and each column represents an individual.
    ploidy : int, optional
        The ploidy level of the organism (default is 1).

    Returns
    -------
    float
        The mean expected heterozygosity across all loci.
    """
    gts_freq = calc_freq(tgt_gts, ploidy=ploidy)
    hetvec = 2 * gts_freq * (1.0 - gts_freq)
    Het = np.mean(hetvec)
    return Het
