# Offline Phase: base class for pyforce library
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

class OfflineDDROM(ABC):
    """
    Abstract base class for the offline phase of the pyforce library.
    
    This class defines the interface for the offline phase, which is responsible
    for preparing data and models before the online phase begins.
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

    @abstractmethod
    def fit(self, train_snaps: FunctionsList, **kwargs):
        """Abstract method to fit the model using training snapshots."""
        pass

    @abstractmethod
    def reconstruct(self, **kwargs):
        """Abstract method to reconstruct the state for a given input."""
        pass

    @abstractmethod
    def reduce(self, **kwargs):
        """Abstract method to reduce the dimensionality of the data and obtain the latent dynamics."""
        pass

    @abstractmethod
    def save(self, path: str, **kwargs):
        """Abstract method to save the model to a specified path."""
        pass
