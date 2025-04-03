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


import inspect
from typing import Any, Callable, Optional


class GenesicFeature:
    """
    A genesic wrapper for genomic feature functions to provide unified interface,
    parameter injection, and flexible input mapping.

    This class allows a feature function (e.g., `calc_u`, `calc_dist`, etc.)
    to be used in a configuration-driven pipeline by explicitly registering
    the function, its parameters, and input aliases.

    Attributes
    ----------
    name : str
        Identifier name of the feature (e.g., 'U50', 'theta_pi').
    func : Callable
        Feature computation function.
    params : dict
        Predefined keyword parameters to be passed to the function.
    alias : dict
        Optional mapping from function parameter names to keys in the input dictionary.

    Examples
    --------
    >>> def calc_dist(gt1, gt2):
    ...     from scipy.spatial import distance_matrix
    ...     d = distance_matrix(gt2.T, gt1.T)
    ...     d.sort(axis=1)
    ...     return d

    >>> feature = GenesicFeature(
    ...     name="dist_ref_vs_tgt",
    ...     func=calc_dist,
    ...     alias={"gt1": "ref_gts", "gt2": "tgt_gts"}
    ... )

    >>> inputs = {"ref_gts": A, "tgt_gts": B}
    >>> result = feature.compute(**inputs)

    >>> def calc_u(ref_gts, tgt_gts, src_gts_list, pos, w, x, y_list):
    ...     return f"U with w={w}, x={x}, y={y_list}"

    >>> feature = GenesicFeature(
    ...     name="U50",
    ...     func=calc_u,
    ...     params={"w": 0.2, "x": 0.5, "y_list": [0.8]}
    ... )

    >>> inputs = {
    ...     "ref_gts": "REF",
    ...     "tgt_gts": "TGT",
    ...     "src_gts_list": ["SRC1", "SRC2"],
    ...     "pos": [1, 2, 3]
    ... }

    >>> feature.compute(**inputs)
    'U with w=0.2, x=0.5, y=[0.8]'
    """

    def __init__(
        self,
        name: str,
        func: Callable[..., Any],
        params: Optional[dict[str, Any]] = None,
        alias: Optional[dict[str, str]] = None,
    ):
        """
        Initialize a GenesicFeature instance.

        Parameters
        ----------
        name : str
            Name of the feature, used as its identifier and output key.
        func : Callable
            The function implementing the feature. It must accept keyword arguments.
        params : dict, optional
            Fixed keyword arguments to be passed to the function at every invocation.
            These are typically thresholds or quantile settings.
            Default is an empty dict.
        alias : dict, optional
            A mapping from function argument names to input dictionary keys.
            If not provided, argument names are assumed to match input keys.
            For example, {"gt1": "ref_gts"} means `gt1=inputs["ref_gts"]`.
            Default is empty (i.e., use identity).
        """
        self.name = name
        self.func = func
        self.params = params or {}
        self.alias = alias or {}

    def compute(self, **inputs: Any) -> Any:
        """
        Computes the feature value using mapped inputs and predefined parameters.

        Parameters
        ----------
        **inputs : Any
            All available named inputs (e.g., ref_gts, tgt_gts, etc.).

        Returns
        -------
        Any
            Result of feature function.
        """
        sig = inspect.signature(self.func)
        needed_keys = sig.parameters.keys()

        # Map inputs using alias if provided, fallback to identity
        args = {
            name: inputs[self.alias.get(name, name)]
            for name in needed_keys
            if self.alias.get(name, name) in inputs
        }

        return self.func(**args, **self.params)
