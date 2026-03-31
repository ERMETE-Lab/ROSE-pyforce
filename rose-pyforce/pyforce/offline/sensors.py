# Offline Phase: sensors classes
# Author: Stefano Riva, PhD Student, NRG, Politecnico di Milano
# Latest Code Update: 07 October 2025
# Latest Doc  Update: 07 October 2025

import numpy as np
from ..tools.functions_list import FunctionsList
from ..tools.backends import IntegralCalculator, LoopProgress
import pyvista as pv

from abc import ABC, abstractmethod

class SensorLibraryBase(ABC):
    r"""
    Abstract base class for sensors, mathematically modelled as linear functionals:

    .. math::
        v(u; \boldsymbol{\xi}) = \int_{\Omega} u(\mathbf{x}) \mathcal{K}(\mathbf{x}, \boldsymbol{\xi}) d\mathbf{x}

    where :math:`\mathcal{K}(\mathbf{x}, \boldsymbol{\xi})` is the kernel function and :math:`\boldsymbol{\xi}` are the sensor parameters (e.g., centre of mass and variance for Gaussian sensors).

    """

    @property
    def nodes(self):
        """Return the nodes where the sensors are placed."""
        return self._nodes

    @property
    def library(self):
        """Return the library of sensors."""
        return self._library
    
    def __len__(self):
        """Return the number of sensors in the library."""
        if self._library is None:
            return 0
        else:
            return len(self._library)

    @abstractmethod
    def _define(self, **kwargs):
        """Abstract method to define a sensor, given its parameters."""
        pass

    @abstractmethod
    def create_library(self, **kwargs):
        """Abstract method to create a library of sensors."""
        pass

    def __call__(self, func: FunctionsList | np.ndarray, **kwargs):
        """Compute the action of the sensor on a given function."""
        
        if isinstance(func, FunctionsList):
            _output = list()
            for f in func:
                _output.append(self.action(f, **kwargs))
            return np.array(_output).T # shape (Nsensors, Ns)
        elif isinstance(func, np.ndarray):
            return self.action(func, **kwargs).reshape(-1, 1) # shape (Nsensors, 1)
        else:
            raise TypeError("Input must be a FunctionsList or a numpy array.")

    def set_library(self, library: FunctionsList):
        r"""
        Set the sensor library to a given FunctionsList.

        Parameters
        ----------
        library: FunctionsList
            The library of sensors to be set.
        """
        self._library = FunctionsList(dofs = len(self.nodes))

        self._library._list = library._list.copy()

    def add_sensor(self, kernel: np.ndarray):
        r"""
        Add a sensor to the library.

        Parameters
        ----------
        kernel: np.ndarray
            The kernel function of the sensor to be added.
        """
        if self._library is None:
            self._library = FunctionsList(dofs = len(self.nodes))

        self._library.append(kernel)

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
        
class GaussianSensorLibrary(SensorLibraryBase):
    r"""
    A class implementing a library of Gaussian sensors.

    A Gaussian sensor is mathematically modelled as a linear functional, with a Gaussian kernel function with two parameters: centre of mass and variance:

    .. math::
        v(u(\mathbf{x}); \mathbf{x}_m, s) = C\cdot \int_{\Omega} u(\mathbf{x}) \exp\left(-\frac{||\mathbf{x} - \mathbf{x}_m||^2}{2s^2}\right) d\mathbf{x}

    where :math:`\mathbf{x}_m` is the centre of mass, :math:`s` is the standard deviation (variance), and :math:`C` is a normalization constant, such that :math:`v(1; \mathbf{x}_m, s) = 1` or equivalently that the :math:`L^1` norm of the kernel is equal to one.

    Parameters
    ----------
    grid: pyvista.UnstructuredGrid
        The computational grid.
    use_centroids: bool, optional (default=True)
        If True, the sensors are placed at the centroids of the grid cells. If False, the sensors are placed at the grid points.
    gdim: int, optional (default=3)
        The geometric dimension of the problem. Default is 3.

    """

    def __init__(self, grid: pv.UnstructuredGrid, use_centroids: bool = True, gdim: int = 3):

        self.grid = grid
        self.gdim = gdim
        self.calculator = IntegralCalculator(grid, gdim)
        
        self._library = None

        if use_centroids:
            self._nodes = grid.cell_centers().points
        else:
            self._nodes = grid.points

    def _define(self, xm: np.ndarray, s: float):
        r"""
        Define a Gaussian sensor given its parameters.

        Parameters
        ----------
        xm: np.ndarray
            The centre of mass of the Gaussian sensor.
        s: float
            The standard deviation (variance) of the Gaussian sensor.

        Returns
        -------
        kernel: function
            The kernel function of the Gaussian sensor.
        """

        
        def kernel(x):
            return np.exp(-np.linalg.norm(x - xm, axis=1)**2 / (2 * s**2))
        
        _kernel = kernel(self.nodes)

        # Normalization constant
        C = 1.0 / self.calculator.L1_norm(_kernel)

        return _kernel * C

    def create_library(self, s: float, xm_list: np.ndarray = None, 
                       verbose: bool = False):
        r"""
        Create a library of Gaussian sensors, given a variance and a list of centres of mass (if provided).
        
        Parameters
        ----------
        s: float
            The standard deviation (variance) of the Gaussian sensors.
        xm_list: np.ndarray, optional (default=None)
            A list of centres of mass for the Gaussian sensors. If None, the sensors are placed at the grid nodes.
        verbose: bool, optional (default=False)
            If True, print progress information.

        """

        if xm_list is None:
            xm_list = self.nodes.tolist()

        self.xm_list = xm_list
        
        self._library = FunctionsList(dofs = len(self.nodes))
        if verbose:
            progress = LoopProgress(msg="Creating Gaussian Sensor Library", final=len(xm_list))

        for xm in xm_list:

            # Define and append the sensor to the library
            self._library.append(
                self._define(xm=xm, s=s)
            )

            # Update progress bar
            if verbose:
                progress.update(1, percentage=True)
        
class ExponentialSensorLibrary(SensorLibraryBase):
    r"""
    A class implementing a library of Exponential sensors.

    An Exponential sensor is mathematically modelled as a linear functional, with an Exponential kernel function with two parameters: centre of mass and variance:

    .. math::
        v(u(\mathbf{x}); \mathbf{x}_m, s) = C\cdot \int_{\Omega} u(\mathbf{x}) \exp\left(-\frac{||\mathbf{x} - \mathbf{x}_m||}{s}\right) d\mathbf{x}

    where :math:`\mathbf{x}_m` is the centre of mass, :math:`s` is the standard deviation (variance), and :math:`C` is a normalization constant, such that :math:`v(1; \mathbf{x}_m, s) = 1` or equivalently that the :math:`L^1` norm of the kernel is equal to one.

    Parameters
    ----------
    grid: pyvista.UnstructuredGrid
        The computational grid.
    use_centroids: bool, optional (default=True)
        If True, the sensors are placed at the centroids of the grid cells. If False, the sensors are placed at the grid points.
    gdim: int, optional (default=3)
        The geometric dimension of the problem. Default is 3.

    """

    def __init__(self, grid: pv.UnstructuredGrid, use_centroids: bool = True,
                 gdim: int = 3):

        self.grid = grid
        self.gdim = gdim
        self.calculator = IntegralCalculator(grid, gdim)
        
        self._library = None

        if use_centroids:
            self._nodes = grid.cell_centers().points
        else:
            self._nodes = grid.points

    def _define(self, xm: np.ndarray, s: float):
        r"""
        Define an Exponential sensor given its parameters.

        Parameters
        ----------
        xm: np.ndarray
            The centre of mass of the Exponential sensor.
        s: float
            The standard deviation (variance) of the Exponential sensor.

        Returns
        -------
        kernel: function
            The kernel function of the Exponential sensor.
        """
        def kernel(x):
            return np.exp(-np.linalg.norm(x - xm, axis=1) / s)

        _kernel = kernel(self.nodes)

        # Normalization constant
        C = 1.0 / self.calculator.L1_norm(_kernel)

        return _kernel * C


    def create_library(self, s: float, xm_list: np.ndarray = None, 
                       verbose: bool = False):
        r"""
        Create a library of Exponential sensors, given a variance and a list of centres of mass (if provided).
        
        Parameters
        ----------
        s: float
            The standard deviation (variance) of the Exponential sensors.
        xm_list: np.ndarray, optional (default=None)
            A list of centres of mass for the Exponential sensors. If None, the sensors are placed at the grid nodes.
        verbose: bool, optional (default=False)
            If True, print progress information.

        """

        if xm_list is None:
            xm_list = self.nodes.tolist()

        self.xm_list = xm_list
        
        self._library = FunctionsList(dofs = len(self.nodes))
        if verbose:
            progress = LoopProgress(msg="Creating Exponential Sensor Library", final=len(xm_list))

        for xm in xm_list:

            # Define and append the sensor to the library
            self._library.append(
                self._define(xm=xm, s=s)
            )

            # Update progress bar
            if verbose:
                progress.update(1, percentage=True)

class IndicatorFunctionSensorLibrary(SensorLibraryBase):
    r"""
    A class implementing a library of Indicator Function sensors.

    An Indicator Function sensor is mathematically modelled as a linear functional, with an Indicator Function kernel:

    .. math::
        v(u(\mathbf{x}); \mathbf{x}_m, r) = \int_{\Omega} u(\mathbf{x}) \mathcal{I}(\mathbf{x}, \mathbf{x}_m, r) d\mathbf{x}

    where :math:`\mathcal{I}(\mathbf{x}, \mathbf{x}_m, r)` is the indicator function defined as:

    .. math::
        \mathcal{I}(\mathbf{x}, \mathbf{x}_m, r) = 
        \begin{cases}
        1 & \text{if } ||\mathbf{x} - \mathbf{x}_m|| \leq r \\
        0 & \text{otherwise}
        \end{cases}

    where :math:`\mathbf{x}_m` is the centre of mass and :math:`r` is the radius.
    
    Parameters
    ----------
    grid: pyvista.UnstructuredGrid
        The computational grid.
    use_centroids: bool, optional (default=True)    
        If True, the sensors are placed at the centroids of the grid cells. If False, the sensors are placed at the grid points.
    gdim: int, optional (default=3)
        The geometric dimension of the problem. Default is 3.
    """

    def __init__(self, grid: pv.UnstructuredGrid, 
                 use_centroids: bool = True,
                 gdim: int = 3):
        
        self.grid = grid
        self.gdim = gdim
        self.calculator = IntegralCalculator(grid, gdim)

        self._library = None    

        if use_centroids:
            self._nodes = grid.cell_centers().points
        else:
            self._nodes = grid.points   
    
    def _define(self, xm: np.ndarray, r: float):
        r"""
        Define an Indicator Function sensor given its parameters.

        Parameters
        ----------
        xm: np.ndarray
            The centre of mass of the Indicator Function sensor.
        r: float
            The radius of the Indicator Function sensor.

        Returns
        -------
        kernel: function
            The kernel function of the Indicator Function sensor.
        """
        def kernel(x):
            return np.where(np.linalg.norm(x - xm, axis=1) <= r, 1.0, 0.0)

        return kernel(self._nodes)
    
    def create_library(self, r: float, xm_list: np.ndarray = None,
                          verbose: bool = False):
          r"""
          Create a library of Indicator Function sensors, given a radius and a list of centres of mass (if provided).
          
          Parameters
          ----------
          r: float
                The radius of the Indicator Function sensors.
          xm_list: np.ndarray, optional (default=None)
                A list of centres of mass for the Indicator Function sensors. If None, the sensors are placed at the grid nodes.
          verbose: bool, optional (default=False)
                If True, print progress information.
    
          """
    
          if xm_list is None:
                xm_list = self._nodes.tolist()

          self.xm_list = xm_list
    
          self._library = FunctionsList(dofs = len(self._nodes))
          if verbose:
                progress = LoopProgress(msg="Creating Indicator Function Sensor Library", final=len(xm_list))
    
          for xm in xm_list:
    
                # Define and append the sensor to the library
                self._library.append(
                 self._define(xm=xm, r=r)
                )
    
                # Update progress bar
                if verbose:
                 progress.update(1, percentage=True)