def compute_LD_D_estimate(gts, ploidy=1, phased=True, compute_r=True):
    """
    Computes the linkage disequilibrium coefficient D for all pairs of SNPs in phased data.
    Using the formulae from Ragsdale 2019: Unbiased Estimation of Linkage Disequilibrium from Unphased
    Data

    Parameters:
    gts (numpy.ndarray): A 2D numpy array where rows represent SNPs and columns represent haplotypes.


    Returns:
    numpy.ndarray: A symmetric matrix where entry (i, j) is the D value between SNP i and SNP j.
    """

    if phased or ploidy == 1:
        num_snps = gts.shape[0]

        # Initialize LD matrix with NaN
        ld_matrix = np.zeros((num_snps, num_snps))
        r_matrix = np.zeros((num_snps, num_snps))

        num_snps, num_individuals = gts.shape

        # Initialize LD matrix with NaN
        ld_matrix = np.full((num_snps, num_snps), np.nan)
        if compute_r:
            r_matrix = np.zeros((num_snps, num_snps))

        for i in range(num_snps):
            for j in range(i + 1, num_snps):  # Compute only upper triangle (symmetry)
                geno1 = gts[i, :]
                geno2 = gts[j, :]

                g1, g2, g3, g4 = 0, 0, 0, 0

                for ind in range(num_individuals):
                    loci1 = geno1[ind]
                    loci2 = geno2[ind]

                    if loci1 == 0 and loci2 == 0:
                        g4 += 1
                    elif loci1 == 0 and loci2 == 1:
                        g3 += 1
                    elif loci1 == 1 and loci2 == 0:
                        g2 += 1
                    elif loci1 == 1 and loci2 == 1:
                        g1 += 1

                num1 = g1 * g4
                num2 = g2 * g3

                # -1 used as in Ragsdale 2019
                denom = num_individuals * (num_individuals - 1)

                D_comp = (num1 - num2) / denom

                ld_matrix[i, j] = D_comp
                ld_matrix[j, i] = D_comp

                p = (g1 + g2) / num_individuals
                q = (g1 + g3) / num_individuals

                if compute_r:
                    r_value = D_comp / (np.sqrt(p * (1 - p) * q * (1 - q)))
                    r_matrix[i, j] = r_value
                    r_matrix[j, i] = r_value

        np.fill_diagonal(ld_matrix, np.nan)

    else:
        raise Exception("This function is only appropriate for phased/haploid data!")
    if not compute_r:
        return ld_matrix
    else:
        return ld_matrix, r_matrix


# LD specific


def compute_LD_D(
    gts: np.ndarray,
    ploidy: int = 1,
    filter_unique: bool = True,
    compute_r: bool = True,
    phased: bool = True,
    maladapt_correction: bool = False,
) -> np.ndarray:
    """
    Computes the linkage disequilibrium coefficient D for all pairs of SNPs in phased or unphased data.

    This function computes the coefficient D for pairs of SNPs either using phased or unphased data.

    Parameters
    ----------
    gts : numpy.ndarray
        A 2D numpy array where rows represent SNPs and columns represent haplotypes (genotypes).
        The array should have shape (n_snps, n_haplotypes), where `n_snps` is the number of SNPs and
        `n_haplotypes` is the number of haplotypes (usually 2 * n_individuals for diploid data).

    ploidy : int, optional, default=1
        The ploidy of the species (e.g., 1 for haploid or 2 for diploid). This is relevant for determining
        the method of computing linkage disequilibrium (LD).

    filter_unique : bool, optional, default=True
        If True, the function will filter out rows of `gts` that contain only unique values (i.e., no variation).

    compute_r : bool, optional, default=True
        If True, the function will also compute the r coefficient alongside the D coefficient. Otherwise,
        only the D coefficient is computed.

    phased : bool, optional, default=True
        If True, the data is considered to be phased. Otherwise, it is unphased.

    maladapt_correction : bool, optional, default=False
        If True, the function will calculate LD very similar to the implementation in MaLAdapt.

    Returns
    -------
    numpy.ndarray
        A symmetric matrix where entry (i, j) is the D value between SNP i and SNP j. The matrix will have
        shape (n_snps, n_snps), where `n_snps` is the number of SNPs in the `gts` array.

    Notes
    -----
    - The function dispatches to different methods based on the ploidy and phased status of the data.
    - The function assumes the input genotypic data (`gts`) is properly formatted and may raise errors if the
      data structure does not conform to expected shapes or values.
    """

    if filter_unique:
        gts = gts[np.array([len(np.unique(row)) > 1 for row in gts])]

    if (phased or ploidy == 1) and not maladapt_correction:
        return compute_LD_D_haploid(gts, ploidy=1, compute_r=compute_r, phased=True)

    elif ploidy == 2:
        # for unphased data
        return compute_ld_burrows(gts, compute_r=compute_r)
    else:
        return compute_LD_D_maladapt(
            gts, ploidy=1, compute_r=compute_r, maladapt_correction=maladapt_correction
        )


def compute_LD_D_haploid(gts, ploidy=1, compute_r=True, phased=True):
    """
    Computes the linkage disequilibrium coefficient D for all pairs of SNPs in phased/haploid data.

    Parameters:
    gts (numpy.ndarray): A 2D numpy array where rows represent SNPs and columns represent haplotypes.


    Returns:
    numpy.ndarray: A symmetric matrix where entry (i, j) is the D value between SNP i and SNP j.
    """

    if phased or ploidy == 1:
        num_snps = gts.shape[0]

        # Initialize LD matrix with NaN
        ld_matrix = np.zeros((num_snps, num_snps))
        r_matrix = np.zeros((num_snps, num_snps))

        for i in range(num_snps):
            for j in range(i + 1, num_snps):  # Compute only upper triangle (symmetry)
                hap1 = gts[i, :]
                hap2 = gts[j, :]

                # Compute allele frequencies
                pA = np.mean(hap1)  # Frequency of major allele at SNP i
                pB = np.mean(hap2)  # Frequency of major allele at SNP j

                # Compute haplotype frequency P_AB
                pAB = np.mean(hap1 * hap2)  # Joint probability

                # Compute LD coefficient D
                D = pAB - (pA * pB)

                ld_matrix[i, j] = D
                ld_matrix[j, i] = D

                if compute_r:
                    r_value = D / (np.sqrt(pA * (1 - pA) * pB * (1 - pB)))
                    r_matrix[i, j] = r_value
                    r_matrix[j, i] = r_value

        # Optionally set diagonal to 0
        np.fill_diagonal(ld_matrix, np.nan)

    else:
        raise Exception("This function works only with phased/haploid data correctly!")

    if not compute_r:
        return ld_matrix
    else:
        return ld_matrix, r_matrix


def compute_ld_burrows(
    gts: np.ndarray,
    compute_r: bool = True,
    ploidy: int = 2,
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Computes the linkage disequilibrium coefficient D for all pairs of SNPs in phased data.
    Using the formulae for D estimation from Ragsdale 2019: Unbiased Estimation of Linkage Disequilibrium from Unphased
    Data (Burrows composite covariance measure of LD).
    Should work ONLY for diploid unphased!!!

    Parameters
    ----------
    gts : numpy.ndarray
        A 2D numpy array where each row represents a SNP, and each column represents a genotype for an individual.
        The shape of the array is (n_snps, n_individuals), where `n_snps` is the number of SNPs and
        `n_individuals` is the number of individuals or haplotypes.

    ploidy : int, optional, default=1
    The ploidy level, indicating the number of chromosome sets.

    compute_r : bool, optional, default=True
        If True, the function will compute the r correlation coefficient in addition to the D coefficient.
        If False, only the D coefficient will be computed.

    Returns
    -------
    tuple of numpy.ndarray or numpy.ndarray
        - If `compute_r` is True, a tuple of two numpy arrays is returned:
        - `ld_matrix`: A symmetric matrix where entry (i, j) is the D value between SNP i and SNP j.
        - `r_matrix`: A symmetric matrix where entry (i, j) is the r value between SNP i and SNP j, or NaN if the r value
            cannot be computed.
        - If `compute_r` is False, only the `ld_matrix` is returned, which contains the D values.

    """

    if ploidy != 2:
        print(
            "Warning! This function for estimating LD is intended for diploid unphased data (encoded as 0,1,2)!"
        )

    num_snps, num_individuals = gts.shape

    # Initialize LD matrix with NaN
    ld_matrix = np.full((num_snps, num_snps), np.nan)
    if compute_r:
        r_matrix = np.zeros((num_snps, num_snps))

    for i in range(num_snps):
        for j in range(i + 1, num_snps):  # Compute only upper triangle (symmetry)
            geno1 = gts[i, :]
            geno2 = gts[j, :]

            g1, g2, g3, g4, g5, g6, g7, g8, g9 = 0, 0, 0, 0, 0, 0, 0, 0, 0

            for ind in range(num_individuals):
                loci1 = geno1[ind]
                loci2 = geno2[ind]

                if loci1 == 0 and loci2 == 0:
                    g9 += 1
                elif loci1 == 0 and loci2 < ploidy:
                    g8 += 1
                elif loci1 == 0 and loci2 == ploidy:
                    g7 += 1

                elif loci1 < ploidy and loci2 == 0:
                    g6 += 1
                elif loci1 < ploidy and loci2 < ploidy:
                    g5 += 1
                elif loci1 < ploidy and loci2 == ploidy:
                    g4 += 1

                elif loci1 == ploidy and loci2 == 0:
                    g3 += 1
                elif loci1 == ploidy and loci2 < ploidy:
                    g2 += 1
                elif loci1 == ploidy and loci2 == ploidy:
                    g1 += 1

            g1 = g1 / num_individuals
            g2 = g2 / num_individuals
            g3 = g3 / num_individuals
            g4 = g4 / num_individuals
            g5 = g5 / num_individuals
            g6 = g6 / num_individuals
            g7 = g7 / num_individuals
            g8 = g8 / num_individuals
            g9 = g9 / num_individuals

            # alternative computation
            """       
            xAB = g1 + (g2/2) + (g4/2) + (g5/4)
            xab = g9 + (g8/2) + (g6/2) + (g5/4)
            xAb = g3 + (g2/2) + (g5/4) + (g6/2)
            xaB = g7 + (g4/2) + (g5/4) + (g8/2)
            """

            p = (g1 + g2 + g3) + ((1 / 2) * (g4 + g5 + g6))
            q = (g1 + g4 + g7) + ((1 / 2) * (g2 + g5 + g8))

            D_comp = (2 * g1 + g2 + g4 + (1 / 2) * g5) - 2 * p * q
            # alternative computation
            # D_comp = 2 * (xAB*xab-xAb*xaB)

            ld_matrix[i, j] = D_comp
            ld_matrix[j, i] = D_comp

            if compute_r:
                r_value = D_comp / (np.sqrt(p * (1 - p) * q * (1 - q)))
                r_matrix[i, j] = r_value
                r_matrix[j, i] = r_value

    np.fill_diagonal(ld_matrix, np.nan)

    if not compute_r:
        return ld_matrix
    else:
        return ld_matrix, r_matrix


def compute_LD_D_maladapt(
    gts: np.ndarray,
    ploidy: int = 1,
    compute_r: bool = True,
    maladapt_correction: bool = True,
) -> Union[tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Computes the linkage disequilibrium coefficient D and optionally the correlation coefficient r for pairs of SNPs,
    with an option to apply maladaptation correction.

    Parameters
    ----------
    gts : numpy.ndarray
        A 2D numpy array where each row represents a SNP and each column represents a genotype (haplotype) for an individual.
        The shape of the array is (n_snps, n_individuals), where `n_snps` is the number of SNPs and `n_individuals` is
        the number of individuals or haplotypes.

    ploidy : int, optional, default=1
        The ploidy level, indicating the number of chromosome sets. Typically, ploidy is 1 for haploids and 2 for diploids.

    compute_r : bool, optional, default=True
        If True, the function will compute the r coefficient, which measures the strength of correlation between the SNPs.
        If False, only the D coefficient is computed.

    maladapt_correction : bool, optional, default=True
        If True, the minor allele will be used for D in case the frequency of the major allele (1) is zero.

    Returns
    -------
    tuple of numpy.ndarray or numpy.ndarray
        - If `compute_r` is True, a tuple of two numpy arrays is returned:
          - `ld_matrix`: A symmetric matrix where entry (i, j) is the D value between SNP i and SNP j.
          - `r_matrix`: A symmetric matrix where entry (i, j) is the r value between SNP i and SNP j, or NaN if the r value
            cannot be computed.
        - If `compute_r` is False, only the `ld_matrix` is returned, which is a symmetric matrix with the D values.

    """
    num_snps, num_individuals = gts.shape

    # Initialize LD matrix with NaN
    ld_matrix = np.full((num_snps, num_snps), np.nan)
    r_matrix = np.zeros((num_snps, num_snps))

    for i in range(num_snps):
        for j in range(i + 1, num_snps):  # Compute only upper triangle (symmetry)
            geno1 = gts[i, :]
            geno2 = gts[j, :]

            P_11 = 0
            P_22 = 0
            P_12 = 0
            P_21 = 0

            for ind in range(num_individuals):
                g1 = geno1[ind]
                g2 = geno2[ind]

                if g1 == 0 and g2 == 0:
                    P_11 += 1
                elif g1 == ploidy and g2 == ploidy:
                    P_22 += 1
                elif g1 == ploidy and g2 < ploidy:
                    P_21 += 1
                elif g1 < ploidy and g2 == ploidy:
                    P_12 += 1

            P_11 /= num_individuals
            P_22 /= num_individuals
            P_12 /= num_individuals
            P_21 /= num_individuals

            P_1x = P_11 + P_12
            P_2x = P_22 + P_21
            P_x1 = P_11 + P_21
            P_x2 = P_12 + P_22

            D_comp = (P_11 * P_22) - (P_12 * P_21)

            ld_matrix[i, j] = D_comp
            ld_matrix[j, i] = D_comp

            if compute_r:
                denominator = np.sqrt(P_2x * (1 - P_2x) * P_x2 * (1 - P_x2))
                if denominator == 0:
                    if maladapt_correction:
                        denominator = np.sqrt(P_1x * (1 - P_1x) * P_x1 * (1 - P_x1))

                        r_value = D_comp / denominator
                    else:
                        r_value = np.nan
                else:
                    r_value = D_comp / denominator
                r_matrix[i, j] = r_value
                r_matrix[j, i] = r_value

    np.fill_diagonal(ld_matrix, np.nan)

    if compute_r:
        return ld_matrix, r_matrix
    else:
        return ld_matrix
