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
from sai.features.abba_feature import compute_ABBA_BABA_D
from sai.features.abba_feature import compute_fd
from sai.features.abba_feature import compute_D_plus


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
