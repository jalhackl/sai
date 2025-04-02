def H1_H12_values(
    gts: np.ndarray,
    only_derived_homozygous: bool = False,
    compute_H123: bool = False,
    ploidy: int = 1,
) -> Union[Tuple[float, float], Tuple[float, float, float]]:
    """
    Computes H1, H12, and optionally H123 statistics to measure haplotype homozygosity.
    Garud 2015

    Parameters
    ----------
    gts : np.ndarray
        Genotype data (allele frequencies or haplotype counts).
    only_derived_homozygous : bool, optional
        If True, considers only derived homozygous frequencies (default is False).
    compute_H123 : bool, optional
        If True, also computes the H123 statistic (default is False).
    ploidy : int, optional
        Number of copies of each chromosome per individual (default is 1).

    Returns
    -------
    Tuple[float, float] or Tuple[float, float, float]
        H1 and H12 values; if compute_H123 is True, returns H1, H12, and H123 values.
    """
    freqs = calc_freq(gts, ploidy=ploidy)
    if not only_derived_homozygous:
        freqs[freqs < 0.5] = 1 - freqs[freqs < 0.5]
    freqs = np.sort(freqs, axis=0)[::-1]

    H1_value = np.sum(freqs**2)

    # alternative calculation
    # H12_value_alt = H1_value + 2 * freqs[0] * freqs[1]

    H12_value = ((freqs[0] + freqs[1]) ** 2) + np.sum(freqs[2:] ** 2)

    H123_value = ((freqs[0] + freqs[1] + freqs[2]) ** 2) + np.sum(freqs[3:] ** 2)

    if not compute_H123:
        return H1_value, H12_value
    else:
        return H1_value, H12_value, H123_value


def H2_value(
    gts: np.ndarray, only_derived_homozygous: bool = False, ploidy: int = 1
) -> float:
    """
    Computes the H2 statistic, which measures haplotype homozygosity excluding the most common haplotype.
    Garud 2015

    Parameters
    ----------
    gts : np.ndarray
        Genotype data (allele frequencies or haplotype counts).
    only_derived_homozygous : bool, optional
        If True, considers only derived homozygous frequencies (default is False).
    ploidy : int, optional
        Number of copies of each chromosome per individual (default is 1).

    Returns
    -------
    float
        The H2 value.
    """
    freqs = calc_freq(gts, ploidy=ploidy)
    if not only_derived_homozygous:
        freqs[freqs < 0.5] = 1 - freqs[freqs < 0.5]
    freqs = np.sort(freqs, axis=0)[::-1]

    H2 = ((freqs[1]) ** 2) + np.sum(freqs[2:] ** 2)
    return H2


def H2_H1_ratio(
    gts: np.ndarray, only_derived_homozygous: bool = False, ploidy: int = 1
) -> float:
    """
    Computes the H2/H1 ratio, a measure of haplotype diversity relative to the most common haplotype.
    Garud 2015

    Parameters
    ----------
    gts : np.ndarray
        Genotype data (allele frequencies or haplotype counts).
    only_derived_homozygous : bool, optional
        If True, considers only derived homozygous frequencies (default is False).
    ploidy : int, optional
        Number of copies of each chromosome per individual (default is 1).

    Returns
    -------
    float
        The H2/H1 ratio.
    """
    H2 = H2_value(gts, only_derived_homozygous=only_derived_homozygous, ploidy=ploidy)

    H1, H12 = H1_H12_values(
        gts, only_derived_homozygous=only_derived_homozygous, ploidy=ploidy
    )

    H2_H1 = H2 / H1
    return H2_H1


def Kellys_Zns(
    gts: np.ndarray, params_LD={"filter_unique": True, "maladapt_correction": False}
) -> float:
    """
    Computes the Zns metric, which quantifies the overall strength of
    linkage disequilibrium (LD) across SNPs using pairwise correlation
    coefficients (r) between SNPs in a genotypic dataset, based on the squared correlation of allelic identity between loci.

    Parameters
    ----------
    gts : np.ndarray
        A 2D numpy array representing genotypic data, where rows represent
        SNPs and columns represent individuals. The shape of the array is
        (num_snps, num_individuals).
    params_LD: dict
        Dictionary with extra parameters for the LD calculation function.

    Returns
    -------
    float
        The Zns value

    """
    ld_matrix, r_matrix = compute_LD_D(gts, compute_r=True, **params_LD)

    S = r_matrix.shape[0]

    # values above the diagonal
    values_above_diagonal = r_matrix[np.triu_indices(S, k=1)]
    r_squared = values_above_diagonal**2

    prefactor = 2 / (S * (S - 1))
    Zns = np.nansum(r_squared)
    Zns = prefactor * Zns
    return Zns


def tajimas_d(gts: np.ndarray, ploidy: int = 1) -> float:
    """
    Compute Tajima's D statistic for a given genotype matrix.

    D = (θπ - θW) / sqrt(Var(θπ - θW))
    Formulae for parameters from Tajima 1989.

    Parameters
    ----------
    gts : np.ndarray
        A 2D NumPy array of shape (mutations x samples) where each entry represents the number of
        mutant alleles (0, 1, or 2) at each site for each individual.

    ploidy : int, optional, default=1
        The ploidy of the population. For example, ploidy = 2 for diploid organisms and ploidy = 1 for haploid organisms.

    Returns
    -------
    float
        Tajima's D statistic
    """
    theta_pi_val = theta_pi(gts, ploidy=ploidy)
    theta_W_val = theta_W(gts, ploidy=ploidy)
    S = num_segregating_sites(gts)

    num_snps, individuals = gts.shape
    a1 = sum(1 / i for i in range(1, individuals + 1))
    a2 = sum(1 / (i**2) for i in range(1, individuals + 1))

    b1 = (individuals + 1) / (3 * (individuals - 1))
    b2 = (2 * (individuals**2 + individuals + 3)) / (
        9 * individuals * (individuals - 1)
    )

    c1 = b1 - (1 / a1)
    c2 = b2 - ((individuals + 2) / (a1**2 * individuals)) + a2 / (a1**2)

    e1 = c1 / a1
    e2 = c2 / (a1**2 + a2)

    denominator = np.sqrt(e1 * S + e2 * S * (S - 1))

    if denominator == 0:
        return float("nan")

    result = (theta_pi_val - theta_W_val) / denominator

    return result
