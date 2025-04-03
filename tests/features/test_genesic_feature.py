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
import pytest
from sai.features.genesic_feature import GenesicFeature


def dummy_func(gt1, gt2, scale=1):
    return np.sum(gt1) + np.sum(gt2) * scale


def test_genesic_feature_no_alias():
    feature = GenesicFeature(name="simple_sum", func=dummy_func, params={"scale": 2})

    result = feature.compute(gt1=np.array([1, 1]), gt2=np.array([2, 2]))
    assert result == 1 + 1 + (2 + 2) * 2  # 2 + 4 * 2 = 10


def test_genesic_feature_with_alias():
    feature = GenesicFeature(
        name="aliased_sum",
        func=dummy_func,
        params={"scale": 3},
        alias={"gt1": "ref", "gt2": "tgt"},
    )

    inputs = {
        "ref": np.array([1, 1]),
        "tgt": np.array([2, 0]),
    }

    result = feature.compute(**inputs)
    assert result == 2 + 2 * 3  # 2 + 6 = 8


def test_genesic_feature_missing_input_ignored():
    feature = GenesicFeature(name="test_partial", func=lambda x: x + 1)

    result = feature.compute(x=5, y=99)  # extra input "y" is ignored
    assert result == 6


def test_config_driven_genesic_feature():
    def calc_dist(gt1, gt2):
        return float(np.linalg.norm(gt1 - gt2))  # simple L2 norm

    # Simulate config
    config = {
        "name": "dist_feature",
        "function": calc_dist,
        "alias": {"gt1": "ref_gts", "gt2": "tgt_gts"},
        "params": {},
    }

    # Construct GenesicFeature
    feature = GenesicFeature(
        name=config["name"],
        func=config["function"],
        alias=config.get("alias"),
        params=config.get("params"),
    )

    # Input dictionary
    inputs = {"ref_gts": np.array([1.0, 2.0]), "tgt_gts": np.array([4.0, 6.0])}

    result = feature.compute(**inputs)
    assert np.isclose(result, 5.0)  # sqrt((3)^2 + (4)^2) = 5
