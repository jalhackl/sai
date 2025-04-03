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


from typing import Callable, Any


class GenesicFeature:
    """
    A genesic feature wrapper for registering and computing genomic features
    with optional parameter injection.

    This class allows feature functions (e.g., calc_u, theta_pi, etc.)
    to be wrapped in a consistent interface, with named identity and
    predefined parameters, to support configuration-driven computation.

    Attributes
    ----------
    name : str
        Feature name.
    func : Callable
        Feature implementation function.
    params : dict
        Default parameters to be passed to the function.
    """

    def __init__(
        self, name: str, func: Callable[..., Any], params: dict[str, Any] = None
    ):
        """
        Initializes a GenesicFeature instance.

        Parameters
        ----------
        name : str
            The name of the feature (e.g., "U50", "theta_pi").
            Used for identification and output labeling.
        func : Callable
            A function that implements the computation logic of the feature.
            Must accept (*args, **kwargs), with keys in `params` as kwargs.
        params : dict, optional
            A dictionary of parameters to be passed to the feature function.
            These will be injected as keyword arguments in each compute call.
            Defaults to an empty dict.
        """
        self.name: str = name
        self.func: Callable[..., Any] = func
        self.params: dict[str, Any] = params or {}

    def compute(self, *args: Any, **kwargs: Any) -> Any:
        """
        Executes the wrapped feature function with given inputs.

        Parameters
        ----------
        *args : Any
            Positional arguments to be passed to the feature function
            (e.g., genotype matrices, position arrays, etc.).
        **kwargs : Any
            Additional keyword arguments passed at runtime. Will override
            any keys in `self.params` if duplicated.

        Returns
        -------
        Any
            The output of the feature function (e.g., statistic value,
            site count, frequency vector, etc.).
        """
        all_kwargs = {**self.params, **kwargs}
        return self.func(*args, **all_kwargs)
