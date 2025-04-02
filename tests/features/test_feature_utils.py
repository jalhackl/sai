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


def test_calc_freq_phased():
    gts = np.array(
        [
            [0, 1, 1],  # freq = 2/3
            [1, 1, 1],  # freq = 1.0
            [0, 0, 0],  # freq = 0.0
        ]
    )
    expected = np.array([2 / 3, 1.0, 0.0])
    result = calc_freq(gts, ploidy=1)
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_calc_freq_unphased():
    gts = np.array(
        [
            [1, 2, 0],  # total derived = 3 / total alleles = 3*2 = 6 → freq = 0.5
            [0, 0, 0],  # freq = 0.0
            [2, 2, 2],  # freq = 1.0
        ]
    )
    expected = np.array([0.5, 0.0, 1.0])
    result = calc_freq(gts, ploidy=2)
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_invalid_shape():
    gts = np.array([[1, 0, 1]])
    result = calc_freq(gts, ploidy=1)
    assert isinstance(result, np.ndarray)
    assert result.shape == (1,)
