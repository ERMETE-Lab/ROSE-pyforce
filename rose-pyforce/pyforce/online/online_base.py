# Online Phase: base class for pyforce library
# Author: Stefano Riva, PhD Student, NRG, Politecnico di Milano
# Latest Code Update: 07 October 2025
# Latest Doc  Update: 07 October 2025

"""
Module for offline base class of offline phase in pyforce library.
This module provides the base class for the offline phase of the pyforce library.
"""

from abc import ABC, abstractmethod

from ..tools.functions_list import FunctionsList
from ..tools.backends import IntegralCalculator

import numpy as np
import pyvista as pv

class OnlineDDROM(ABC):
    """
    Abstract base class for the online phase of the pyforce library.

    This class defines the interface for the online phase, which is responsible
    for making predictions and (if possible) updating the model in real-time.
    """

    def __init__(self, grid, gdim=3, varname='u', **kwargs):
        """
        Initialize the OfflineDDROM class.
        
        Args:
            grid: The computational grid.
            gdim: The geometric dimension of the problem (default is 3).
            varname: The variable name to be processed (default is 'u').
        """

        self.grid = grid
        self.varname = varname
        self.gdim = gdim

        self.calculator = IntegralCalculator(grid, gdim)

    @property
    def basis(self):
        """Property to access the basis used in the online phase.
        This property returns a `FunctionsList` containing the basis functions used for the online phase of the model.

        Returns:
            FunctionsList: The basis functions used in the online phase.

        """
        return self._basis

    @abstractmethod
    def set_basis(self, basis: FunctionsList = None, path_folder: str = None, **kwargs):
        """Abstract method to load or assign the basis functions."""
        pass

    @abstractmethod
    def estimate(self, **kwargs):
        """Abstract method to estimate the state for a given input."""
        pass

    @abstractmethod
    def _reduce(self, **kwargs):
        """Abstract method to reduce the dimensionality of the data and obtain the true latent dynamics (backend method)."""
        pass

class SurrogateModelWrapper(ABC):
    """
    A wrapper class for surrogate models to predict coefficients based on input vector (it can be either test parameters or measurements).
    """

    @abstractmethod
    def predict(self, input_vector: np.ndarray, **kwargs) -> np.ndarray:
        r"""
        Predict coefficients based on the provided input vector of size :math:`(N_s, N_p)`, where :math:`N_s` is the number of samples and :math:`N_p` is the degrees of freedom of the input vector (for instance, either the number of parameters or the number of sensors).

        Parameters
        ----------
        input_vector : np.ndarray
            The input vector for which to predict the coefficients, shaped :math:`(N_s, N_p)`.

        Returns
        -------
        np.ndarray
            The predicted coefficients shaped :math:`(r, N_s)`.
        """
        pass

class OnlineSensors():
    r"""
    Base class for online sensors.
    This class provides the basic functionality for online sensors, focusing on the action of the sensor on a given function to obtain measurements.

    Parameters
    ----------
    grid: pv.UnstructuredGrid
        The computational grid.
    library: FunctionsList
        The library of sensors to be used.
    gdim: int, optional (default=3)
        The geometric dimension of the problem. 

    """

    def __init__(self, grid: pv.UnstructuredGrid, library: FunctionsList, gdim: int = 3):

        self.grid = grid
        self.gdim = gdim
        self.calculator = IntegralCalculator(grid, gdim)
        
        self._library = FunctionsList(dofs=library.fun_shape)
        self._library._list = library._list.copy()

    def __call__(self, func: FunctionsList | np.ndarray, **kwargs):
        """Compute the action of the sensor on a given function."""
        
        if isinstance(func, FunctionsList):
            return self.action(func, **kwargs)
        elif isinstance(func, np.ndarray):
            return self.action(func, **kwargs)
        else:
            raise TypeError("Input must be a FunctionsList or a numpy array.")
        
    def __len__(self):
        """Return the number of sensors in the library."""
        return len(self._library)

    def _action_single(self, func: np.ndarray, idx_sens: int):
        r"""
        Compute the action of a single sensor on a given function.

        Parameters
        ----------
        func: np.ndarray
            The function to be sensed.
        idx_sens: int
            The index of the sensor in the library.

        Returns
        -------
        action: float
            The action of the sensor on the function.
        """
        if self._library is None:
            raise ValueError("Sensor library is not created. Please call 'create_library' or `set_library` methods first.")

        return self.calculator.L2_inner_product(func, self._library[idx_sens])

    def action(self, func: FunctionsList | np.ndarray, M: int = None):
        r"""
        Compute the action of the sensor library on a given function (both single or matrix) or a list of functions.

        Parameters
        ----------
        func: FunctionsList | np.ndarray
            The function or list of functions to be sensed.
        M: int, optional (default=None)
            If provided, only the first M sensors in the library are used.

        Returns
        -------
        actions: np.ndarray
            The actions of the sensor library on the function(s). Shape (Nsensors, Ns).
        """

        if self._library is None:
            raise ValueError("Sensor library is not created. Please call 'create_library' or `set_library` methods first.")

        if M is None:
            M = len(self._library)
        else:
            assert M <= len(self._library), "M cannot be larger than the number of sensors in the library."

        if isinstance(func, FunctionsList):
            _measurements = list()
            for f in func:
                _measurements.append(
                                    np.array([self._action_single(f, i) for i in range(M)])
                                    )
            return np.array(_measurements).T # shape (Nsensors, Ns)
        
        elif isinstance(func, np.ndarray):

            if func.ndim == 1:
                func = np.atleast_2d(func).T # shape (N, 1)
            
            _measurements = list()
            for ii in range(func.shape[1]):
                _measurements.append(
                                    np.array([self._action_single(func[:, ii], i) for i in range(M)])
                                    )
            return np.array(_measurements).T # shape (Nsensors, Ns)
        