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


import pytest
import numpy as np
from sai.features.abba_baba_feature import _calc_four_pops_freq
from sai.features.abba_baba_feature import _calc_pattern_sum
from sai.features.abba_baba_feature import calc_d
from sai.features.abba_baba_feature import calc_fd
from sai.features.abba_baba_feature import calc_d_plus


def test_calc_four_pops_freq_basic():
    ref_gts = np.array([[0, 1], [1, 1]])  # freq = [0.5, 1.0]
    tgt_gts = np.array([[1, 0], [0, 0]])  # freq = [0.5, 0.0]
    src_gts = np.array([[1, 1], [1, 0]])  # freq = [1.0, 0.5]
    out_gts = np.array([[0, 0], [0, 1]])  # freq = [0.0, 0.5]

    ref, tgt, src, out = _calc_four_pops_freq(
        ref_gts, tgt_gts, src_gts, out_gts, ploidy=1
    )

    np.testing.assert_array_almost_equal(ref, np.array([0.5, 1.0]))
    np.testing.assert_array_almost_equal(tgt, np.array([0.5, 0.0]))
    np.testing.assert_array_almost_equal(src, np.array([1.0, 0.5]))
    np.testing.assert_array_almost_equal(out, np.array([0.0, 0.5]))


def test_calc_four_pops_freq_no_outgroup():
    ref_gts = np.array([[0, 1]])
    tgt_gts = np.array([[1, 0]])
    src_gts = np.array([[1, 1]])

    ref, tgt, src, out = _calc_four_pops_freq(
        ref_gts, tgt_gts, src_gts, out_gts=None, ploidy=1
    )

    np.testing.assert_array_equal(ref, np.array([0.5]))
    np.testing.assert_array_equal(tgt, np.array([0.5]))
    np.testing.assert_array_equal(src, np.array([1.0]))
    np.testing.assert_array_equal(out, np.array([0.0]))  # default to 0s


def test_calc_four_pops_freq_diploid():
    ref_gts = np.array([[0, 2]])
    tgt_gts = np.array([[1, 1]])
    src_gts = np.array([[2, 0]])
    out_gts = np.array([[1, 1]])

    # ploidy=2 → total alleles = 2 * n_samples
    # freq = sum / (2 * N)

    ref, tgt, src, out = _calc_four_pops_freq(
        ref_gts, tgt_gts, src_gts, out_gts, ploidy=2
    )

    np.testing.assert_array_equal(ref, np.array([0.5]))  # (0+2)/4
    np.testing.assert_array_equal(tgt, np.array([0.5]))  # (1+1)/4
    np.testing.assert_array_equal(src, np.array([0.5]))  # (2+0)/4
    np.testing.assert_array_equal(out, np.array([0.5]))  # (1+1)/4


def test_calc_pattern_sum_abba():
    ref = np.array([0.1, 0.8])
    tgt = np.array([0.9, 0.2])
    src = np.array([0.5, 0.5])
    out = np.array([0.0, 1.0])

    # pattern: 'abba'
    # site 0: (1-0.1)*0.9*0.5*(1-0.0) = 0.9*0.9*0.5*1 = 0.405
    # site 1: (1-0.8)*0.2*0.5*(1-1.0) = 0.2*0.2*0.5*0 = 0.0
    # sum = 0.405 + 0.0 = 0.405

    result = _calc_pattern_sum(ref, tgt, src, out, "abba")
    assert np.isclose(result, 0.405)


def test_calc_pattern_sum_baba():
    ref = np.array([0.1, 0.8])
    tgt = np.array([0.9, 0.2])
    src = np.array([0.5, 0.5])
    out = np.array([0.0, 1.0])

    # pattern: 'baba'
    # site 0: 0.1*(1-0.9)*0.5*(1-0.0) = 0.1*0.1*0.5*1 = 0.005
    # site 1: 0.8*(1-0.2)*0.5*0 = 0.8*0.8*0.5*0 = 0
    # sum = 0.005

    result = _calc_pattern_sum(ref, tgt, src, out, "baba")
    assert np.isclose(result, 0.005)


def test_calc_pattern_sum_baaa():
    ref = np.array([0.1, 0.8])
    tgt = np.array([0.9, 0.2])
    src = np.array([0.5, 0.5])
    out = np.array([0.0, 1.0])

    # pattern: 'baaa'
    # site 0: 0.1*(1-0.9)*(1-0.5)*(1-0.0) = 0.1*0.1*0.5*1 = 0.005
    # site 1: 0.8*(1-0.2)*(1-0.5)*0      = 0.8*0.8*0.5*0 = 0
    # sum = 0.005

    result = _calc_pattern_sum(ref, tgt, src, out, "baaa")
    assert np.isclose(result, 0.005)


def test_calc_pattern_sum_abaa():
    ref = np.array([0.1, 0.8])
    tgt = np.array([0.9, 0.2])
    src = np.array([0.5, 0.5])
    out = np.array([0.0, 1.0])

    # pattern: 'abaa'
    # site 0: (1-0.1)*0.9*(1-0.5)*(1-0.0) = 0.9*0.9*0.5*1 = 0.405
    # site 1: (1-0.8)*0.2*(1-0.5)*0       = 0.2*0.2*0.5*0 = 0
    # sum = 0.405

    result = _calc_pattern_sum(ref, tgt, src, out, "abaa")
    assert np.isclose(result, 0.405)


def test_invalid_pattern_length():
    ref = tgt = src = out = np.array([0.1, 0.2])
    with pytest.raises(ValueError, match="four-character"):
        _ = _calc_pattern_sum(ref, tgt, src, out, "ab")


def test_invalid_pattern_char():
    ref = tgt = src = out = np.array([0.1, 0.2])
    with pytest.raises(ValueError, match="Invalid character"):
        _ = _calc_pattern_sum(ref, tgt, src, out, "abxa")


def test_calc_d():
    # test input
    ref_gts = np.array([[0, 1], [1, 0], [0, 1]])  # Reference population
    tgt_gts = np.array([[1, 0], [0, 1], [1, 0]])  # Target population
    src_gts = np.array([[1, 1], [1, 1], [1, 1]])  # Source population
    out_gts = None  # No outgroup provided

    # ref_freq = [0.5, 0.5, 0.5]
    # tgt_freq = [0.5, 0.5, 0.5]
    # src_freq = [1, 1, 1]
    # out_freq = [0, 0, 0]

    # pattern: 'abba'
    # site 0: (1-0.5)*0.5*1*(1-0) = 0.25
    # site 1: (1-0.5)*0.5*1*(1-0) = 0.25
    # site 2: (1-0.5)*0.5*1*(1-0) = 0.25
    # sum = 0.75

    # pattern: 'baba'
    # site 0: 0.5*(1-0.5)*1*(1-0) = 0.25
    # site 1: 0.5*(1-0.5)*1*(1-0) = 0.25
    # site 2: 0.5*(1-0.5)*1*(1-0) = 0.25
    # sum = 0.75

    # abba - baba = 0
    # abba + baba = 0.75
    # (abba - baba) / (abba + baba) = 0

    # Call the function with the test input
    result = calc_d(ref_gts, tgt_gts, src_gts, out_gts)

    # Check the result
    expected_result = 0
    assert np.isclose(
        result, expected_result
    ), f"Expected {expected_result}, but got {result}"


def test_calc_fd():
    # test input
    ref_gts = np.array([[0, 1], [1, 0], [0, 1]])  # Reference population
    tgt_gts = np.array([[1, 0], [0, 1], [1, 0]])  # Target population
    src_gts = np.array([[1, 1], [1, 1], [1, 1]])  # Source population
    out_gts = None  # No outgroup provided

    # ref_freq = [0.5, 0.5, 0.5]
    # tgt_freq = [0.5, 0.5, 0.5]
    # src_freq = [1, 1, 1]
    # out_freq = [0, 0, 0]

    # abba_n - baba_n = 0 See test_calc_d()

    # use_hom = False
    # dnr_freq = src_freq = [1, 1, 1]
    # pattern: 'abba'
    # site 0: (1-0.5)*1*1*(1-0) = 0.5
    # site 1: (1-0.5)*1*1*(1-0) = 0.5
    # site 2: (1-0.5)*1*1*(1-0) = 0.5
    # sum = 1.5

    # pattern: 'baba'
    # site 0: 0.5*(1-1)*1*(1-0) = 0
    # site 1: 0.5*(1-1)*1*(1-0) = 0
    # site 2: 0.5*(1-1)*1*(1-0) = 0
    # sum = 0

    # abba_d - baba_d = 1.5

    # Call the function with the test input
    result = calc_fd(ref_gts, tgt_gts, src_gts, out_gts)

    # use_hom = True
    # dnr_freq = tgt_freq = [0.5, 0.5, 0.5]
    # pattern: 'abba'
    # site 0: (1-0.5)*0.5*0.5*(1-0) = 0.125
    # site 1: (1-0.5)*0.5*0.5*(1-0) = 0.125
    # site 2: (1-0.5)*0.5*0.5*(1-0) = 0.125
    # sum = 0.375

    # pattern: 'baba'
    # site 0: 0.5*(1-0.5)*0.5*(1-0) = 0.125
    # site 1: 0.5*(1-0.5)*0.5*(1-0) = 0.125
    # site 2: 0.5*(1-0.5)*0.5*(1-0) = 0.125
    # sum = 0.375

    # abba_d - baba_d = 0

    result_hom = calc_fd(ref_gts, tgt_gts, src_gts, out_gts, use_hom=True)

    # Check the result
    expected_result = 0
    assert np.isclose(
        result, expected_result
    ), f"Expected {expected_result}, but got {result}"

    assert np.isnan(result_hom), "Result should be NaN"


def test_calc_d_plus():
    # test input
    ref_gts = np.array([[0, 0], [0, 0], [1, 1]])  # Referece population
    tgt_gts = np.array([[1, 0], [0, 1], [0, 1]])  # Taget population
    src_gts = np.array([[0, 1], [1, 0], [1, 0]])  # Source population
    out_gts = None  # No outgroup provided

    # ref_freq = [0, 0, 1]
    # tgt_freq = [0.5, 0.5, 0.5]
    # src_freq = [0.5, 0.5, 0.5]
    # out_freq = [0, 0, 0]

    # pattern: 'abba'
    # site 0: (1-0)*0.5*0.5*(1-0) = 0.25
    # site 1: (1-0)*0.5*0.5*(1-0) = 0.25
    # site 2: (1-1)*0.5*0.5*(1-0) = 0
    # sum = 0.5

    # pattern: 'baba'
    # site 0: 0*(1-0.5)*0.5*(1-0) = 0
    # site 1: 0*(1-0.5)*0.5*(1-0) = 0
    # site 2: 1*(1-0.5)*0.5*(1-0) = 0.25
    # sum = 0.25

    # pattern: 'baaa'
    # site 0: 0*(1-0.5)*(1-0.5)*(1-0) = 0
    # site 1: 0*(1-0.5)*(1-0.5)*(1-0) = 0
    # site 2: 1*(1-0.5)*(1-0.5)*(1-0) = 0.25
    # sum = 0.25

    # pattern: 'abaa'
    # site 0: (1-0)*0.5*(1-0.5)*(1-0) = 0.25
    # site 1: (1-0)*0.5*(1-0.5)*(1-0) = 0.25
    # site 2: (1-1)*0.5*(1-0.5)*(1-0) = 0
    # sume = 0.5

    # abba - baba + baaa - abaa = 0.5 - 0.25 + 0.25 - 0.5 = 0
    # abba + baba + baaa + abaa = 0.5 + 0.25 + 0.25 + 0.5 = 1.5

    # D-ancestral
    # baaa - abaa = 0.25 - 0.5 = -0.25
    # baaa + abaa = 0.25 + 0.5 = 0.75
    # (baaa - abaa) / (baaa + abaa) = -1/3

    # Call the function with the test input
    result = calc_d_plus(ref_gts, tgt_gts, src_gts, out_gts)
    result_ancestral = calc_d_plus(ref_gts, tgt_gts, src_gts, out_gts, calc_d_ancestral=True)

    # Check the result
    expected_result = 0
    assert np.isclose(
        result, expected_result
    ), f"Expected {expected_result}, but got {result}"

    expected_result_ancestral = -1/3
    assert np.isclose(
        result, expected_result
    ), f"Expected {expected_result_ancestral}, but got {result_ancestral}"
