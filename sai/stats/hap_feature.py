def cal_n_ton(tgt_gt, is_phased, ploidy):
    """
    Description:
        Calculates individual frequency spectra for samples.

    Arguments:
        tgt_gt numpy.ndarray: Genotype matrix from the target population.
        ploidy int: Ploidy of the genomes.

    Returns:
        spectra numpy.ndarray: Individual frequency spectra for haplotypes.
    """
    if is_phased:
        ploidy = 1
    mut_num, sample_num = tgt_gt.shape
    iv = np.ones((sample_num, 1))
    counts = (tgt_gt > 0) * np.matmul(tgt_gt, iv)
    spectra = np.array(
        [
            np.bincount(
                counts[:, idx].astype("int64"), minlength=sample_num * ploidy + 1
            )
            for idx in range(sample_num)
        ]
    )
    # ArchIE does not count non-segragating sites
    spectra[:, 0] = 0

    return spectra


def cal_dist(gt1, gt2):
    """
    Description:
        Calculates pairwise Euclidean distances between two genotype matrixes.

    Arguments:
        gt1 numpy.ndarray: Genotype matrix 1.
        gt2 numpy.ndarray: Genotype matrix 2.

    Returns:
        dists numpy.ndarray: Distances estimated.
    """

    from scipy.spatial import distance_matrix

    dists = distance_matrix(np.transpose(gt2), np.transpose(gt1))
    dists.sort()

    return dists
