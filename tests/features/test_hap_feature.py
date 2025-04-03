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

from sai.features.hap_feature import calc_n_ton
from sai.features.hap_feature import calc_dist


def test_calc_n_ton_phased():
    gt = np.array(
        [
            [1, 0, 1],  # site 0
            [0, 1, 0],  # site 1
            [1, 1, 0],  # site 2
            [0, 0, 1],  # site 3
        ]
    )

    # Each site has mutation count:
    # site 0: 1 + 0 + 1 = 2
    # site 1: 0 + 1 + 0 = 1
    # site 2: 1 + 1 + 0 = 2
    # site 3: 0 + 0 + 1 = 1

    # For each sample:
    # sample 0: sites [0,2] with mutations (freq=2) → 2 times freq=2
    # sample 1: sites [1,2] → freq=1, freq=2 → 1 of each
    # sample 2: sites [0,3] → freq=2, freq=1 → 1 of each

    expected = np.array(
        [
            [0, 0, 2, 0],  # sample 0
            [0, 1, 1, 0],  # sample 1
            [0, 1, 1, 0],  # sample 2
        ]
    )

    result = calc_n_ton(gt)  # ploidy ignored
    np.testing.assert_array_equal(result, expected)


def test_calc_n_ton_unphased_diploid():
    gt = np.array(
        [
            [2, 0, 1],  # site 0
            [0, 2, 2],  # site 1
            [1, 1, 0],  # site 2
        ]
    )

    # For each site, total derived:
    # site 0 = 3, site 1 = 4, site 2 = 2

    # For each sample:
    # sample 0: sites [0,2] → counts 3, 2 → sum = 5 → freq bin 3 (x[3] += 1), x[2] += 1
    #           total per bin: [0, 0, 1, 1, 0, 0, 0]
    # sample 1: sites [1,2] → counts 4, 2 → freq bin 2, 4 → x[2] += 1, x[4] += 1
    # sample 2: sites [0,1] → counts 3, 4 → freq bin 3, 4 → x[3] += 1, x[4] += 1

    expected = np.array(
        [
            [0, 0, 1, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 0, 0],
        ]
    )

    result = calc_n_ton(gt, ploidy=2)
    np.testing.assert_array_equal(result, expected)


def test_calc_n_ton_zero_input():
    gt = np.zeros((4, 3))  # no mutations
    result = calc_n_ton(gt)
    assert result.shape[0] == 3
    assert np.all(result == 0)


def test_calc_dist_exact_sorted():
    gt1 = np.array([[0, 0], [1, 1]])  # → individuals: [0,1], [0,1]
    gt2 = np.array([[1, 0], [0, 1]])  # → individuals: [1,0], [0,1]

    # gt1.T = [[0,1], [0,1]]
    # gt2.T = [[1,0], [0,1]]

    # Distance matrix:
    # gt2[0] = [1, 0]
    #   → gt1[0] = [0, 1] → sqrt(1^2 + 1^2) = sqrt(2) ≈ 1.4142
    #   → gt1[1] = [0, 1] → sqrt(1^2 + 1^2) = sqrt(2) ≈ 1.4142
    #
    # gt2[1] = [0, 1]
    #   → gt1[0] = [0, 1] → sqrt(0) = 0
    #   → gt1[1] = [0, 1] → sqrt(0) = 0

    expected = np.array(
        [
            sorted([np.sqrt(2), np.sqrt(2)]),  # [1.4142, 1.4142]
            sorted([0.0, 0.0]),  # [0.0, 1.0]
        ]
    )

    result = calc_dist(gt1, gt2)
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_calc_dist_identity():
    gt = np.array([[0, 1, 2], [1, 1, 1]])  # 3 individuals: [0,1], [1,1], [2,1]

    result = calc_dist(gt, gt)

    # Distances:
    # each individual to itself = 0
    # These should be the smallest per row
    diag_min = np.min(result, axis=1)
    assert np.allclose(diag_min, 0.0)

    # Check sorted
    assert np.all(np.diff(result, axis=1) >= 0)

    # Shape
    assert result.shape == (3, 3)


def test_calc_dist_different_shapes():
    gt1 = np.array([[0, 1, 2], [0, 0, 1]])  # → 3 individuals

    gt2 = np.array([[1, 2], [1, 1]])  # → 2 individuals

    # gt1.T = [ [0,0], [1,0], [2,1] ]
    # gt2.T = [ [1,1], [2,1] ]

    # Row 0: [1,1]
    #   to [0,0] → sqrt(1^2 + 1^2) = sqrt(2)
    #   to [1,0] → sqrt(0^2 + 1^2) = 1
    #   to [2,1] → sqrt(1^2 + 0^2) = 1
    #
    # Row 1: [2,1]
    #   to [0,0] → sqrt(2^2 + 1^2) = sqrt(5)
    #   to [1,0] → sqrt(1^2 + 1^2) = sqrt(2)
    #   to [2,1] → sqrt(0^2 + 0^2) = 0

    row0 = sorted([np.sqrt(2), 1.0, 1.0])  # [1.0, 1.0, 1.4142]
    row1 = sorted([np.sqrt(5), np.sqrt(2), 0])  # [0.0, 1.4142, 2.236...]

    expected = np.array([row0, row1])

    result = calc_dist(gt1, gt2)
    np.testing.assert_allclose(result, expected, rtol=1e-6)
