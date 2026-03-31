# Fundamental tools
# Author: Stefano Riva, PhD Student, NRG, Politecnico di Milano
# Latest Code Update: 07 October 2025
# Latest Doc  Update: 07 October 2025

import pyvista as pv
import numpy as np
import time

# This class is used to compute integrals and norms on a PyVista UnstructuredGrid.
class IntegralCalculator():
    r"""
    Class to compute integrals and norms on a PyVista UnstructuredGrid.
    This class supports both 2D and 3D grids and computes cell sizes accordingly.
    It provides methods to compute integrals, averages, :math:`L^1` norms, :math:`L^2` inner products, and :math:`L^2` norms.
    
    Parameters
    ----------
    grid : pv.UnstructuredGrid
        The PyVista UnstructuredGrid on which the operations will be performed.
    gdim : int, optional (Default = 3)
        The geometric dimension of the grid. It can be either 2 or 3.
        If set to 2, the area of the cells will be computed; if set to 3, the volume of the cells will be computed. 

    Attributes
    ----------
    grid : pv.UnstructuredGrid
        The PyVista UnstructuredGrid on which the operations will be performed.
    gdim : int
        The geometric dimension of the grid (2 or 3).
    cell_sizes : np.ndarray
        The sizes of the cells in the grid, computed as areas for 2D grids or volumes for 3D grids.
    n_points : int
        The number of points in the grid.
    n_cells : int
        The number of cells in the grid.    
    
    """
    def __init__(self, grid: pv.UnstructuredGrid, gdim = 3):
        self.grid = grid
        self.gdim = gdim

        if gdim == 2:
            self.cell_sizes = grid.compute_cell_sizes()['Area']
        elif gdim == 3:
            self.cell_sizes = grid.compute_cell_sizes()['Volume']

        if np.isclose(np.sum(self.cell_sizes), 0):
            raise ValueError("The sum of cell sizes is zero, which is not valid. Check the gdim parameter for the adopted grid.")

        self.n_points = grid.n_points
        self.n_cells  = grid.n_cells
        
    def check_input(self, u):
        """
        Check if the input array u is valid for the grid.
        It should have the same number of elements as the number of points or cells in the grid.

        Parameters
        ----------
        u : np.ndarray
            The input array to be checked.  
        """
        
        if u.shape[0] == self.n_points:
            self.grid['u'] = u
            return self.grid.point_data_to_cell_data()['u']
        elif u.shape[0] == self.n_cells:
            return u
        elif u.shape[0] == self.n_points * self.gdim: # does this work for any vector field?
            return u.reshape(self.n_points, self.gdim)
        elif u.shape[0] == self.n_cells * self.gdim:
            return u.reshape(self.n_cells, self.gdim)
        else:
            raise ValueError(f"Input array must have shape {self.n_points} or {self.n_cells}, got {u.shape[0]}")

    def integral(self, u):
        r"""
        Computes the integral of a given scalar function `u` over the domain

        .. math::
            \int_\Omega u \,d\Omega 

        Parameters
        ----------
        u : `np.ndarray`
            Function belonging to the grid
            
        Returns
        -------
        value : float
            Integral over the domain
        """
        u = self.check_input(u)

        if len(u.shape) > 1: # vector field
            return np.sum(u.T * self.cell_sizes, axis=1)
        else:  # scalar field
            return np.sum(u * self.cell_sizes)
    
    def average(self, u):
        r"""
        Computes the integral average of a given **scalar** function `u` over the domain

        .. math::
            \langle u \rangle = \frac{1}{|\Omega|}\int_\Omega u \,d\Omega

        Parameters
        ----------
        u : np.ndarray
            Function belonging to the grid

        Returns
        -------
        ave_value : float
            Average over the domain
        """
        u = self.check_input(u)

        return self.integral(u) / np.sum(self.cell_sizes)
    
    def L1_norm(self, u):
        r""" 
        Computes the :math:`L^1` norm of a function `u` over the domain

        .. math::
            \|u\|_{L^1}=\int_\Omega |u| \,d\Omega

        Parameters
        ----------
        u : np.ndarray
            Function belonging to the grid

        Returns
        -------
        value : float
            :math:`L^1` norm of the function
        """
        u = self.check_input(u)

        return self.integral(np.abs(u))


    def L2_inner_product(self, u, v):
        r"""
        Compute the L2 inner product of two functions :math:`\left( u, v \right)` over the domain

        .. math::
            \left( u, v \right)_{L^2} = \int_\Omega u \cdot v \,d\Omega

        Parameters
        ----------
        u : np.ndarray
            First function belonging to the grid.
        v : np.ndarray
            Second function belonging to the grid.  

        Returns
        -------
        inner_product : float
            The L2 inner product of the two functions.

        """
        u = self.check_input(u)
        v = self.check_input(v)

        if len(u.shape) == 1 and len(v.shape) == 1:
            u = u.reshape(-1, 1)
            v = v.reshape(-1, 1)

        return self.integral((u * v).sum(axis=1))
    
    def L2_norm(self, u):
        r"""
        Computes the :math:`L^2` norm of a function `u` over the domain

        .. math::
            \|u\|_{L^2}=\sqrt{\left( u, u \right)_{L^2}}

        Parameters
        ----------
        u : np.ndarray
            Function belonging to the grid  

        Returns
        -------
        norm : float
            :math:`L^2` norm of the function 

        """
        return np.sqrt(self.L2_inner_product(u, u))
    
# Class to make progress bar using printing
class LoopProgress():
    r"""
    A class to make progress bar.

    Parameters
    ----------
    msg : str
        Message to be displayed
    final : float, optional (Default = 100)
        Maximum value for the iterations

    """
    def __init__(self, msg: str, final: float = 100):
        self.msg = msg
        self.final = final
        self.instant = 0.

        self.init_time  = time.time()
        self.comp_times = list()

        out =  self.msg+': '
        print (out, end="\r")

    def update(self, step: float, percentage: bool = False):
        r"""
        Update message to display and clears the previous one.
        
        Parameters
        ----------
        step : float
            Interger or float value to add at the counter.
        percentage : boolean, optional (Default = False)
            Indicates if the bar should be displayed in %.
        
        """

        # Compute average computational time
        self.comp_times.append(time.time() - self.init_time)        
        average_time = sum(self.comp_times) / len(self.comp_times)

        # Update instant
        self.instant += step

        # Write the message
        if percentage:
            printed_inst = '{:.3f}'.format(self.instant / self.final * 100)+' / 100.00%'
        else:
            printed_inst = '{:.3f}'.format(self.instant)+' / {:.2f}'.format(self.final)
        out =  self.msg+': '+printed_inst + ' - {:.6f}'.format(average_time)+' s/it'

        # Print output
        if np.isclose(self.instant, self.final):
            print (out)
        else:
            print (out, end="\r")

        # Update inital offset cpu time
        self.init_time  = time.time()

# Custom exception for Timer class
class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

# This class is used to measure the time taken for a specific operation.
class Timer:
    def __init__(self):
        self._start_time = None
        
    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.process_time()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.process_time() - self._start_time
        self._start_time = None

        return elapsed_time