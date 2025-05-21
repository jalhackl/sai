import numpy as np
from typing import Callable, Tuple, Union, Optional, List, Dict
import numpy.typing as npt
import inspect
from numpy.lib.stride_tricks import sliding_window_view


class CalculatorTheoreticalSCCT:
    """
    Calculator for the theoretical SCCT (selection by conditional coalescent tree) ratio based on
    coalescent theory branch length expectations.

    This is a Python version of the Java implementation from:
    https://github.com/wavefancy/scct/blob/master/sources/TheoreticalRatio/version1.1/src/CalculatorV5.java

    Instead of Decimal float is used - it should be checked whether accuracy matters that much; if yes, one could also switch in Python to Decimal
    """

    def __init__(self, dn: Optional[int] = None, an: Optional[int] = None) -> None:
        """
        Initialize the CalculatorTheoreticalSCCT.

        Parameters
        ----------
        dn : int, optional
            Number of individuals carrying the derived allele.
        an : int, optional
            Number of individuals carrying the ancestral allele.
        """

        self.BT = None
        self.dn = dn
        self.an = an
        if dn is not None and an is not None:
            self.set_individual(dn, an)

    def set_individual(self, dn: int, an: int) -> None:
        """
        Set the number of individuals in the derived and ancestral groups
        and precompute branch length expectations.

        Parameters
        ----------
        dn : int
            Number of individuals with the derived allele.
        an : int
            Number of individuals with the ancestral allele.
        """

        self.dn = dn
        self.an = an
        self.BT = [[0.0 for _ in range(an + 1)] for _ in range(dn + 1)]
        # calculate expectation of branch length
        for i in range(1, dn + 1):
            for j in range(1, an + 1):
                if i == 1 and j == 1:
                    # not used
                    self.BT[i][j] = 1.0
                else:
                    total = i + j
                    self.BT[i][j] = 2.0 / (total * (total - 1))

    def normal_total_tree_len(self, n: int) -> float:
        """
        Compute the total tree length under the neutral coalescent model for n individuals.

        Parameters
        ----------
        n : int
            Total number of leaves.

        Returns
        -------
        float
            Total branch length of the tree.
        """

        total = 0.0
        for i in range(2, n + 1):
            total += 2.0 / (i - 1)
        return total

    def get_expectation(self, dn: int, an: int) -> float:
        """
        Compute the expected SCCT ratio given the number of derived and ancestral individuals.

        Parameters
        ----------
        dn : int
            Number of individuals with the derived allele.
        an : int
            Number of individuals with the ancestral allele.

        Returns
        -------
        float
            Expected SCCT ratio (branch length in derived / ancestral group).
            Returns NaN if input is invalid or ratio is undefined.
        """

        if dn < 1 or an < 1:  # monomorphic, invalid, ...
            return float("nan")
        self.set_individual(dn, an)
        return self._get_expectation()

    def _get_expectation(self) -> float:
        """
        Internal method that computes the branch length expectation ratio.

        Returns
        -------
        float
            Theoretical SCCT ratio.
        """

        dn, an = self.dn, self.an

        # tree length for iteration
        # perhaps it would be better to use numpy arrays etc.

        tl_D = 0.0
        tl_A = 0.0

        # test probability
        p = [[0.0 for _ in range(an + 1)] for _ in range(dn + 1)]
        p[dn][an] = 1.0

        # for the end condition is (1,a). (1,a+1) will not go to (1,a)
        for d in range(dn, 1, -1):
            for a in range(an, 0, -1):
                prob = p[d][a]
                if prob == 0.0:
                    continue
                total = d + a
                l = self.BT[d][a]

                tL = (prob * (d + 1)) / total if d - 1 >= 1 else 0.0
                tR = (prob * (a - 1)) / total if a - 1 >= 1 else 0.0

                if d - 1 >= 1:
                    p[d - 1][a] += tL
                if a - 1 >= 1:
                    p[d][a - 1] += tR

                temp = tL + tR
                tl_D += d * l * temp
                tl_A += a * l * temp

        # Compute the total branch length when reach (1,a)
        for a in range(1, an + 1):
            tl_A += self.normal_total_tree_len(a + 1) * p[1][a]

        if tl_A == 0.0:
            return float("inf")
        return tl_D / tl_A


def scct_counting(
    gts: np.ndarray,
    central_snp: int,
    return_counts: bool = False,
    use_log_ratio: bool = True,
    simple_log_ratio: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[int, int]]:
    """
    Count derived and ancestral informative SNPs relative to a central SNP using the SCCT framework.

    Parameters
    ----------
    gts : np.ndarray
        A binary matrix of shape (num_snps, num_samples), where rows are SNPs and columns are samples.
    central_snp : int
        Index of the SNP used as the reference (central) SNP.
    return_counts : bool, optional
        If True, return arrays of counts per SNP. If False, return summed counts.
    use_log_ratio : bool, optional
        Whether to apply a log-odds ratio correction for linkage inference.
    simple_log_ratio : bool, optional
        If True, use a simplified log-ratio approach. Ignored if use_log_ratio is False.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray] or Tuple[int, int]
        If `return_counts` is True, returns (count_D, count_C), arrays of counts per SNP row.
        If `return_counts` is False, returns (sum_D, sum_C), summed counts across all SNPs.

    Notes
    -----
    - This function differentiates SNPs that are linked to either the derived or ancestral allele
      at a central SNP based on mutation patterns.
    - Yates-corrected log-odds ratios are used to estimate linkage when `use_log_ratio=True`.
    - Assumes the genotype data is phased and binary (0: ancestral, 1: derived).
    """
    central_values = gts[central_snp, :]

    # this is in line with the original scct-implementation
    snp_mask = central_values == 1

    subgroup_D = np.where(snp_mask)[0]
    subgroup_C = np.where(~snp_mask)[0]

    other_snp_rows = np.delete(np.arange(gts.shape[0]), central_snp)

    submatrix_D = gts[other_snp_rows, :][:, subgroup_D]
    submatrix_C = gts[other_snp_rows, :][:, subgroup_C]

    count_D = np.sum(submatrix_D, axis=1)
    count_C = np.sum(submatrix_C, axis=1)

    # for implementation of odds ratio with Yates correction compare https://github.com/wavefancy/scct/blob/master/sources/CountTwoGroupMutations/src/FComputeWorker.java
    if use_log_ratio:

        if not simple_log_ratio:

            corr_count_D = 0
            corr_count_C = 0

            size_D = len(subgroup_D)
            size_C = len(subgroup_C)

            for d, c in zip(count_D, count_C):
                if c == 0 and d != 0:
                    # clear case: only in derived
                    if d < size_D:
                        corr_count_D += 1  # segregating in derived
                    else:
                        corr_count_C += 1  # fixed in derived, likely ancestral origin
                elif d == 0 and c != 0:
                    # clear case: only in ancestral
                    corr_count_C += 1
                elif d > 0 and c > 0:
                    # mutation in both ancestral and derived group. Check linkage: odds ratio with Yates correction
                    f11 = d + 0.5
                    f10 = size_D - d + 0.5
                    f01 = c + 0.5
                    f00 = size_C - c + 0.5

                    # 95% confidence interval
                    lor = np.log(f11) + np.log(f00) - np.log(f10) - np.log(f01)

                    # mutation with great linkage with derived allele
                    if lor > 0:
                        if d + c >= size_D:
                            # two mutations happened at the same ancestral branch
                            corr_count_C += 1
                        else:
                            corr_count_D += 1  # linked with derived
                    # linkage with ancestral allele.
                    elif lor < 0:
                        corr_count_C += 1  # linked with ancestral

            count_D = corr_count_D
            count_C = corr_count_C

        else:

            corr_count_D = 0
            corr_count_C = 0

            for d, c in zip(count_D, count_C):

                if d > 0 and c == 0:
                    corr_count_D += d
                elif c > 0 and d == 0:
                    corr_count_C += c
                elif d > 0 and c > 0:
                    log_ratio = np.log(d / c)
                    if log_ratio > 0:
                        corr_count_D += 1
                    else:
                        corr_count_C += 1

            count_D = corr_count_D
            count_C = corr_count_C

    if return_counts:
        return count_D, count_C

    sum_D = np.sum(count_D)
    sum_C = np.sum(count_C)

    return sum_D, sum_C


def sample_scct_phased(
    gts: np.ndarray,
    central_snp: Optional[int] = None,
    maf_threshold: float = 0.05,
    return_below_threshold: Union[str, float] = "nan",
    use_log_ratio: bool = False,
    theoretical: bool = True,
    gts_pos: Optional[np.ndarray] = None,
    full_vcf_gts: Optional[np.ndarray] = None,
    full_vcf_pos: Optional[np.ndarray] = None,
    simple_log_ratio: bool = False,
    set_alpha_1: bool = False,
) -> float:
    """
    Calculate the SCCT (Selection detection by Conditional Coalescent Tree) statistic for a given central SNP (if not given, the SNP in the middle of the window is chosen).

    Parameters
    ----------
    gts : np.ndarray
        Genotype matrix of shape (num_snps, num_samples), encoded as 0/1.
    central_snp : int, optional
        Index of the SNP to center the statistic on. If None, the middle SNP is used.
    maf_threshold : float, optional
        Minimum minor allele frequency (MAF) threshold. SNPs below this are ignored.
    return_below_threshold : str or float, optional
        Value to return if the central SNP is monomorphic or below the MAF threshold.
    use_log_ratio : bool, optional
        If True, uses log odds ratio as in https://github.com/wavefancy/scct/blob/master/sources/CountTwoGroupMutations/src/FComputeWorker.java
    theoretical : bool, optional
        If True, uses a theoretical expectation for ratio_alpha. Otherwise, uses empirical.
    gts_pos : np.ndarray, optional
        Positions corresponding to the SNPs in `gts`. Required if `theoretical` is False.
    full_vcf_gts : np.ndarray, optional
        Full genotype matrix used for empirical estimation. Required if `theoretical` is False.
    full_vcf_pos : np.ndarray, optional
        Full SNP positions for the full VCF. Required if `theoretical` is False.
    simple_log_ratio : bool, optional
        If True, in case use_log_ratio also is True, only a simple log-ratio is used.
    set_alpha_1 : bool, optional
        If True, sets ratio_alpha to 1 regardless of other parameters.

    Returns
    -------
    float
        The SCCT statistic `S`, or the specified fallback value (e.g., NaN) if MAF is too low.

    Notes
    -----
    - The function assumes phased binary genotype data (0/1).
    - Assumes the central SNP is biallelic.
    """

    # if index of central snp (for which the statistics is calculated) is not given, we choose the middle
    if not central_snp:
        central_snp = gts.shape[0] // 2

    central_values_count = np.unique(gts[central_snp, :], return_counts=True)
    if len(central_values_count[0]) < 2:
        print(
            "Error: There is either only the ancestral or the derived allel present at the central SNP, return nan"
        )
        return float(return_below_threshold)

    counts_dict = dict(zip(central_values_count[0], central_values_count[1]))
    central_values_ancestral = counts_dict.get(0, 0)
    central_values_derived = counts_dict.get(1, 0)

    # check MAF
    if maf_threshold:
        total = central_values_ancestral + central_values_derived
        maf = min(central_values_ancestral, central_values_derived) / total
        if maf < maf_threshold:
            return float(return_below_threshold)

    sum_D, sum_C = scct_counting(
        gts,
        central_snp,
        use_log_ratio=use_log_ratio,
        simple_log_ratio=simple_log_ratio,
        return_counts=False,
    )

    if set_alpha_1:
        ratio_alpha = 1

    elif theoretical:
        theoretical_calculator = CalculatorTheoreticalSCCT()
        ratio_alpha = theoretical_calculator.get_expectation(
            central_values_derived, central_values_ancestral
        )
        print(f"theoretical ratio: {ratio_alpha}")

    else:
        central_pos = gts_pos[central_snp]

        full_vcf_gts_filtered, full_indices_filtered = filter_zero_rows(
            full_vcf_gts, return_indices=True
        )

        full_vcf_pos_filtered = np.delete(full_vcf_pos, full_indices_filtered, axis=0)

        central_index_full = np.where(
            np.array(full_vcf_pos_filtered) == int(central_pos)
        )[0][0]

        sum_D_full, sum_C_full = scct_counting(
            full_vcf_gts_filtered,
            central_index_full,
            use_log_ratio=False,
            return_counts=False,
        )

        ratio_alpha = sum_D_full / sum_C_full

        print(f"empirical ratio: {ratio_alpha}")

    S = np.log(sum_D / (ratio_alpha * sum_C))

    return S


def load_full_vcf(
    vcf_file: str,
    chr_name: str,
    ref_ind_file: str,
    tgt_ind_file: str,
    anc_allele_file: Optional[str] = None,
    ploidy: int = 2,
    is_phased: bool = True,
    src_ind_file: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
    """
    Helper function:
    Load full genotype data from a VCF file for reference, target, and optionally source individuals.
    imports utils from gaia_utils

    It is used for calculation of the empirical scale ratio (for which the full genome is taken, in principle one could also use specific regions)

    Parameters
    ----------
    vcf_file : str
        Path to the VCF file.
    chr_name : str
        Chromosome name (e.g., "chr1") to extract data from.
    ref_ind_file : str
        File path to the list of reference individual IDs.
    tgt_ind_file : str
        File path to the list of target individual IDs.
    anc_allele_file : str, optional
        File path to ancestral allele information. Default is None.
    ploidy : int, optional
        Ploidy level of the individuals. Default is 2.
    is_phased : bool, optional
        Whether the genotype data is phased. Default is True.
    src_ind_file : str, optional
        File path to the list of source individual IDs. If not provided, source data is not returned.

    Returns
    -------
    ref_gts : np.ndarray
        Genotype array for reference individuals.
    src_gts : np.ndarray or None
        Genotype array for source individuals, if `src_ind_file` is provided. Otherwise, None.
    tgt_gts : np.ndarray
        Genotype array for target individuals.
    pos : np.ndarray
        Array of SNP positions for the specified chromosome.
    """

    import sai.utils.gaia_utils as gaia_utils

    ref_data, ref_samples, tgt_data, tgt_samples, src_data, src_samples = (
        gaia_utils.read_data_src(
            vcf_file,
            ref_ind_file,
            tgt_ind_file,
            src_ind_file,
            anc_allele_file,
            is_phased,
        )
    )

    pos = tgt_data[chr_name]["POS"]

    ref_gts = ref_data[chr_name]["GT"]
    tgt_gts = tgt_data[chr_name]["GT"]
    if src_ind_file:
        src_gts = src_data[chr_name]["GT"]

    return ref_gts, src_gts, tgt_gts, pos


def filter_zero_rows(
    gts: np.ndarray, return_indices: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Remove rows from a genotype matrix that contain only zeros.

    Parameters
    ----------
    gts : np.ndarray
        A 2D NumPy array representing genotype data (e.g., SNPs x individuals).
    return_indices : bool, optional
        If True, also return the indices of the rows that were removed. Default is False.

    Returns
    -------
    np.ndarray
        A filtered version of the input array with zero-only rows removed.
    tuple of (np.ndarray, np.ndarray)
        If `return_indices` is True, returns a tuple where the first element is the filtered array
        and the second is an array of indices of the removed rows.
    """

    zero_rows_mask = np.all(gts == 0, axis=1)

    zero_row_indices = np.where(zero_rows_mask)[0]

    filtered_arr = gts[~zero_rows_mask]

    if not return_indices:
        return filtered_arr
    else:
        return filtered_arr, zero_row_indices


def scct_windows_from_bpwindows(
    gts: np.ndarray,
    gts_pos: np.ndarray,
    snp_window_size: int = 51,
    full_vcf_gts: Optional[np.ndarray] = None,
    full_vcf_pos: Optional[np.ndarray] = None,
    function_to_apply: Callable = None,
    function_params: Optional[dict] = None,
    reduce_mode: Optional[
        str
    ] = None,  # 'max', 'mean', or None - in the latter case, full array is returned
) -> Union[np.ndarray, float]:
    """
    Apply SCCT statistic in sliding SNP windows (in case default function_to_apply is chosen) within a base pair window.

    Parameters
    ----------
    gts : np.ndarray
        Genotype matrix of shape (num_snps, num_samples), possibly containing rows with all zeros.
    gts_pos : np.ndarray
        1D array of genomic positions corresponding to rows in `gts`.
    snp_window_size : int, optional
        Size of the SNP window for applying the function. Default is 151.
    full_vcf_gts : np.ndarray, optional
        Full genotype matrix from the original VCF file, used for empirical alpha estimation.
    full_vcf_pos : np.ndarray, optional
        Position array corresponding to `full_vcf_gts`.
    function_to_apply : Callable, optional
        A function to apply to each SNP window. Must accept arguments compatible with `sample_scct_phased`.
    reduce_mode : str, optional
        Whether to return the a specific statistic (mean or max) value across all windows.
        Probably mean can be recommended for selection detection.
        Default is None.

    Returns
    -------
    np.ndarray or float
        If `reduce_mode` is None, returns an array of SCCT statistics for each window.
        If `reduce_mode` is str, returns the summary statistics of the SCCT values (or NaN if none).
    """

    # Get supported parameters of the function
    supported_params = set(inspect.signature(function_to_apply).parameters)

    # Defaults
    default_params = {
        "theoretical": True,
        "central_snp": None,
        "use_log_ratio": False,
        "set_alpha_1": False,
        "simple_log_ratio": False,
    }

    # Merge and filter to only supported ones
    merged_params = default_params.copy()
    if function_params:
        merged_params.update(function_params)
    filtered_params = {k: v for k, v in merged_params.items() if k in supported_params}

    gts_filtered, gts_indices = filter_zero_rows(gts, return_indices=True)
    # gts_pos_filtered = gts_pos[gts_indices]
    gts_pos_filtered = np.delete(gts_pos, gts_indices, axis=0)

    # Check if window size is too large
    if gts_filtered.shape[0] < snp_window_size:
        print(
            f"snp_window_size ({snp_window_size}) is larger than the number of SNPs ({gts_filtered.shape[0]}). Returning [0]."
        )
        return np.array([0])

    snp_slides = sliding_window_view(
        gts_filtered, window_shape=(snp_window_size, gts_filtered.shape[1])
    )
    pos_slides = sliding_window_view(gts_pos_filtered, window_shape=(snp_window_size))

    results = []
    for i in range(len(snp_slides)):
        snp_window = snp_slides[i][0]
        pos_window = pos_slides[i]

        result = function_to_apply(
            snp_window,
            gts_pos=pos_window,
            full_vcf_gts=full_vcf_gts,
            full_vcf_pos=full_vcf_pos,
            **filtered_params,
        )

        results.append(result)

    results_array = np.array(results)

    if reduce_mode == "max":
        finite_results = results_array[np.isfinite(results_array)]
        return np.nan if finite_results.size == 0 else np.nanmax(finite_results)

    elif reduce_mode == "mean":
        finite_results = results_array[np.isfinite(results_array)]
        return np.nan if finite_results.size == 0 else np.nanmean(finite_results)

    return results_array


# the functions below do not really belong to SCCT
# the idea was to get the frequencies instead of the counts - so one could perhaps also use a similar statistics for unphased data etc.


def frequency_by_snp_group(
    gts: npt.NDArray[np.number],
    central_snp: Optional[int] = None,
    ploidy: int = 2,
    is_phased: bool = True,
) -> Dict[np.number, npt.NDArray[np.floating]]:
    """
    Groups samples by central SNP value and computes frequency of 1s per SNP (excluding central SNP).

    Parameters:
    gts : ndarray of shape (n_snps, n_samples)
        A 2D array representing genotypes. Each element is a numeric genotype
        value (typically 0 or 1, possibly higher in polyploid organisms).

    central_snp : int, optional
        Index of the SNP to group by. If None, the SNP in the middle of the
        array (along axis 0) is used.

    ploidy : int, default=2
        Number of chromosome sets. Affects frequency calculation for unphased data.

    is_phased : bool, default=True
        Whether the genotype data is phased. If False, assumes diploid or polyploid
        genotypes and adjusts frequency calculation accordingly.

    Returns:
        dict of {scalar : ndarray}
        A dictionary mapping each unique central SNP value to a 1D array of
        allele frequencies for all other SNPs (excluding the central one).
    """
    if central_snp is None:
        central_snp = gts.shape[0] // 2

    central_values = gts[central_snp, :]
    unique_vals = np.unique(central_values)

    # Indices of all rows except the central SNP
    other_snp_rows = np.delete(np.arange(gts.shape[0]), central_snp)

    freq_by_value = {}

    for val in unique_vals:
        # Select samples (columns) where central SNP == val
        sample_indices = np.where(central_values == val)[0]

        # Extract relevant part of the matrix (excluding central SNP row)
        submatrix = gts[other_snp_rows, :][:, sample_indices]

        # Compute frequency
        if ploidy == 1 or is_phased:
            freq = np.mean(submatrix, axis=1)
        else:
            freq = np.sum(submatrix, axis=1) / (submatrix.shape[1] * ploidy)

        freq_by_value[val] = freq

    return freq_by_value


def counts_by_snp_group(
    gts: npt.NDArray[np.number],
    central_snp: Optional[int] = None,
    ploidy: int = 2,
    is_phased: bool = True,
) -> Dict[Union[int, float], Dict[str, npt.NDArray[np.integer]]]:
    """
    Groups samples by central SNP value and computes frequency of 1s per SNP (excluding central SNP).
    Group samples by the value of a central SNP and compute the counts of non-zero values
    (e.g., derived alleles) for each other SNP across these groups.

    Parameters
    ----------
    gts : ndarray of shape (n_snps, n_samples)
        A 2D array representing genotypes. Each entry is typically 0 or 1, or can be higher
        for polyploid or dosage data.

    central_snp : int, optional
        Index of the SNP to group by. If None, defaults to the center SNP.

    ploidy : int, default=2
        Number of chromosome sets (currently unused, included for interface consistency).

    is_phased : bool, default=True
        Whether genotype data is phased (currently unused, included for interface consistency).

    Returns
    -------
    dict
        A dictionary mapping each unique value of the central SNP to:
        - "per_snp_counts": array of total values per SNP (excluding central SNP),
        - "total_derived_mutations": scalar sum of all such values across SNPs and samples.
    """
    if central_snp is None:
        central_snp = gts.shape[0] // 2

    central_values = gts[central_snp, :]
    unique_vals = np.unique(central_values)

    # Indices of all rows except the central SNP
    other_snp_rows = np.delete(np.arange(gts.shape[0]), central_snp)

    # freq_by_value = {}
    result_by_value = {}

    for val in unique_vals:
        # Select samples (columns) where central SNP == val
        sample_indices = np.where(central_values == val)[0]

        # Extract relevant part of the matrix (excluding central SNP row)
        submatrix = gts[other_snp_rows, :][:, sample_indices]

        count = np.sum(submatrix, axis=1)

        total_derived = np.sum(count)

        result_by_value[val] = {
            "per_snp_counts": count,
            "total_derived_mutations": total_derived,
        }

    return result_by_value


def sample_scct_freqs(
    gts: npt.NDArray[np.integer], central_snp: Optional[int] = None
) -> Dict[Union[int, float], npt.NDArray[np.integer]]:
    """
    Group sample indices by the value of a central SNP.

    Parameters
    ----------
    gts : ndarray of shape (n_snps, n_samples)
        Genotype matrix where rows represent SNPs and columns represent samples.
        Each element typically contains integer genotype values (e.g., 0, 1, 2).

    central_snp : int, optional
        Index of the SNP used to group samples. If None, defaults to the middle SNP
        (i.e., gts.shape[0] // 2).

    Returns
    -------
    dict
        A dictionary mapping each unique value at the central SNP to an array of
        sample indices (column indices) that carry that value.
    """
    # if index of central snp (for which the statistics is calculated) is not given, we choose the middle
    if not central_snp:
        central_snp = gts.shape[0] // 2

    central_values = gts[central_snp, :]
    unique_values = np.unique(central_values)

    value_to_indices = {
        val: np.where(central_values == val)[0] for val in unique_values
    }

    return value_to_indices
