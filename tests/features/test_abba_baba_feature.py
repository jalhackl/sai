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
from sai.features.abba_baba_feature import _calc_abba_baba_components
from sai.features.abba_baba_feature import compute_ABBA_BABA_D
from sai.features.abba_baba_feature import compute_fd
from sai.features.abba_baba_feature import compute_D_plus


def test_calc_abba_baba_components_default_outgroup():
    pop1 = np.array([0.0, 1.0])
    pop2 = np.array([1.0, 0.0])
    pop3 = np.array([1.0, 1.0])

    # abba = (1 - 0.0)*1.0*1.0 + (1 - 1.0)*0.0*1.0 = 1.0 + 0 = 1.0
    # baba = 0.0*(1 - 1.0)*1.0 + 1.0*(1 - 0.0)*1.0 = 0 + 1.0 = 1.0
    # sum(abba - baba) = 0.0, sum(abba + baba) = 2.0

    diff, total = _calc_abba_baba_components(pop1, pop2, pop3)

    assert np.isclose(diff, 0.0)
    assert np.isclose(total, 2.0)


def test_calc_abba_baba_components_with_outgroup():
    pop1 = np.array([0.0, 1.0])
    pop2 = np.array([1.0, 0.0])
    pop3 = np.array([1.0, 1.0])
    outgroup = np.array([1.0, 0.0])

    # Only site 1 contributes:
    # abba = (1 - 1.0)*0.0*1.0*(1 - 0.0) = 0.0
    # baba = 1.0*(1 - 0.0)*1.0*(1 - 0.0) = 1.0

    diff, total = _calc_abba_baba_components(pop1, pop2, pop3, outgroup_freq=outgroup)

    assert np.isclose(diff, -1.0)
    assert np.isclose(total, 1.0)


def test_compute_ABBA_BABA_D():
    # test input
    src_gts = np.array([[1, 1], [1, 1], [1, 1]])  # Source population
    ref_gts = np.array([[0, 1], [1, 0], [0, 1]])  # Reference population
    tgt_gts = np.array([[1, 0], [0, 1], [1, 0]])  # Target population
    out_gts = None  # No outgroup provided

    # Call the function with the test input
    result = compute_ABBA_BABA_D(src_gts, ref_gts, tgt_gts, out_gts)

    # Check the result
    expected_result = 0
    assert np.isclose(
        result, expected_result
    ), f"Expected {expected_result}, but got {result}"


def test_compute_fd():
    # test input
    src_gts = np.array([[1, 1], [1, 1], [1, 1]])  # Source population
    ref_gts = np.array([[0, 1], [1, 0], [0, 1]])  # Reference population
    tgt_gts = np.array([[1, 0], [0, 1], [1, 0]])  # Target population
    out_gts = None  # No outgroup provided

    # Call the function with the test input
    result = compute_fd(src_gts, ref_gts, tgt_gts, out_gts)

    # Check the result
    expected_result = 0
    assert np.isclose(
        result, expected_result
    ), f"Expected {expected_result}, but got {result}"


def test_compute_D_plus():
    # test input
    src_gts = np.array([[0, 0], [0, 0], [1, 1]])  # Source population
    ref_gts = np.array([[1, 0], [0, 1], [0, 1]])  # Reference population
    tgt_gts = np.array([[0, 1], [1, 0], [1, 0]])  # Target population
    out_gts = None  # No outgroup provided

    # Call the function with the test input
    result = compute_D_plus(src_gts, ref_gts, tgt_gts, out_gts)

    # Check the result
    expected_result = 0
    assert np.isclose(
        result, expected_result
    ), f"Expected {expected_result}, but got {result}"
