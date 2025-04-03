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
from sai.features.feature_utils import calc_freq


def test_invalid_shape():
    gts = np.array([[1, 0, 1]])
    result = calc_freq(gts, ploidy=1)
    assert isinstance(result, np.ndarray)
    assert result.shape == (1,)


def test_phased_data():
    # Phased data, ploidy = 1
    gts = np.array([[1, 0, 0, 1], [0, 0, 0, 0], [1, 1, 1, 1]])
    expected_frequency = np.array([0.5, 0.0, 1.0])
    result = calc_freq(gts, ploidy=1)
    np.testing.assert_array_almost_equal(
        result, expected_frequency, decimal=6, err_msg="Phased data test failed."
    )


def test_unphased_diploid_data():
    # Unphased data, ploidy = 2 (diploid)
    gts = np.array([[1, 1], [0, 0], [2, 2]])
    expected_frequency = np.array([0.5, 0.0, 1.0])
    result = calc_freq(gts, ploidy=2)
    np.testing.assert_array_almost_equal(
        result,
        expected_frequency,
        decimal=6,
        err_msg="Unphased diploid data test failed.",
    )


def test_unphased_triploid_data():
    # Unphased data, ploidy = 3 (triploid)
    gts = np.array([[1, 2, 3], [0, 0, 0], [3, 3, 3]])
    expected_frequency = np.array([0.6667, 0.0, 1.0])
    result = calc_freq(gts, ploidy=3)
    np.testing.assert_array_almost_equal(
        result,
        expected_frequency,
        decimal=4,
        err_msg="Unphased triploid data test failed.",
    )


def test_unphased_tetraploid_data():
    # Unphased data, ploidy = 4 (tetraploid)
    gts = np.array([[2, 2, 2, 2], [1, 3, 0, 4], [0, 0, 0, 0]])
    expected_frequency = np.array([0.5, 0.5, 0.0])
    result = calc_freq(gts, ploidy=4)
    np.testing.assert_array_almost_equal(
        result,
        expected_frequency,
        decimal=6,
        err_msg="Unphased tetraploid data test failed.",
    )
