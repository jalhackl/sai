def calc_rd(
    src_gts: np.ndarray,
    tgt_gts: np.ndarray,
    ref_gts: np.ndarray = None,
    metric: str = "cityblock",
) -> float:
    """
    Compute the average ratio of sequence divergence between an individual from the
    source population and an individual from the target (admixed) population, compared
    to the sequence divergence between an individual from the source population and
    an individual from the reference (non-admixed) population.

    This is done by comparing the pairwise distances between individuals in the source
    group (`src_gts`) with those in the target group (`tgt_gts`) and the reference group
    (`ref_gts`) using a chosen distance metric. The mean of the pairwise distances for
    each row in both matrices is computed, and the ratio of these means is averaged over
    all combinations of source-target and source-reference rows.

    Parameters
    ----------
    src_gts : np.ndarray
        A 2D numpy array where each row represents a locus and each column represents
        an individual in the source population.

    tgt_gts : np.ndarray
        A 2D numpy array where each row represents a locus and each column represents
        an individual in the target (admixed) population.

    ref_gts : np.ndarray, optional
        A 2D numpy array where each row represents a locus and each column represents
        an individual in the reference (non-admixed) population. If `None`, a zero array
        of the same shape as `tgt_gts` is used. Default is `None`.

    metric : str, optional
        The distance metric to use for the pairwise distance calculation. Default is "cityblock".

    Returns
    -------
    float
        The computed average ratio of sequence divergence between the source-target and
        source-reference populations.

    """

    # If ref_gts is None, set it to a zero matrix of the same shape as tgt_gts
    if ref_gts is None:
        ref_gts = np.zeros(tgt_gts.shape)

    # pairwise distances
    seq_divs_src_tgt = cdist(src_gts.T, tgt_gts.T, metric=metric)
    seq_divs_src_ref = cdist(src_gts.T, ref_gts.T, metric=metric)

    # mean of each row
    mean_src_tgt = np.mean(seq_divs_src_tgt, axis=1)
    mean_src_ref = np.mean(seq_divs_src_ref, axis=1)

    all_pairs = list(
        product(range(seq_divs_src_tgt.shape[0]), range(seq_divs_src_ref.shape[0]))
    )
    count = len(all_pairs)

    average_r = 0

    # Loop over all combinations
    for i, j in all_pairs:
        # Use precomputed row means
        row_tgt_mean = mean_src_tgt[i]
        row_ref_mean = mean_src_ref[j]

        # Accumulate the ratio of means
        if row_ref_mean != 0:
            average_r += row_tgt_mean / row_ref_mean
        else:
            print("Warning! An average ratio is zero!")

    average_r = average_r / count
    return average_r


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
