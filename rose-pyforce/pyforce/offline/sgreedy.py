# Offline Phase: sensors classes
# Author: Stefano Riva, PhD Student, NRG, Politecnico di Milano
# Latest Code Update: 07 October 2025
# Latest Doc  Update: 07 October 2025

import numpy as np
import scipy
import scipy.linalg
import matplotlib.pyplot as plt
import os

from ..tools.functions_list import FunctionsList
from ..tools.backends import IntegralCalculator, LoopProgress
from .sensors import GaussianSensorLibrary, ExponentialSensorLibrary, IndicatorFunctionSensorLibrary, SensorLibraryBase

class SGREEDY():
    r"""
    A class to place sensors using the SGREEDY algorithm in order to maximize the information contained in the update space spanned by the Riesz representation of sensors.
    

    The algorithm is described in the work of `Haik et al. (2023) <https://www.sciencedirect.com/science/article/pii/S0045782522008246>`_.

    Parameters
    ----------
    grid : pyvista.UnstructuredGrid
        The computational grid.
    gdim : int, optional
        The geometric dimension of the problem (default is 3).
    varname : str, optional
        The variable name to be processed (default is 'u').
    sensors_type: str, optional (default='Gaussian')
        The type of sensors to use in the GEIM algorithm. It can be either 'Gaussian', 'Exponential', or 'IndicatorFunction'.
    
    """

    def __init__(self, grid, gdim=3, varname='u', sensors_type = 'Gaussian'):
        
        self.grid = grid
        self.varname = varname
        self.gdim = gdim

        self.calculator = IntegralCalculator(grid, gdim)

        _allowed_sensor_types = ['Gaussian', 'Exponential', 'IndicatorFunction']
        assert sensors_type in _allowed_sensor_types, f"sensors_type must be one of {_allowed_sensor_types}"
        self.sensors_type = sensors_type

    def fit(self, basis_functions: FunctionsList, Mmax: int, sensor_params: dict,
            Nmax: int = None, tol: float = 0.2, verbose: bool = True):
        r"""
        Selection of sensors position with a Riesz representation :math:`\{g_m\}_{m=1}^M` in :math:`L^2`.
        The positions of the sensors are either freely selected on the mesh or given as input with the 'sensors_params' dictionary.

        Parameters
        ----------
        basis_functions : FunctionsList
            The reduced basis functions.
        Mmax : int
            The maximum number of sensors to be placed (it must be greater or equal to the number of basis functions).
        sensors_params : dict
            Dictionary containing the parameters for the sensor library creation.
        Nmax : int, optional
            The maximum number of basis functions to be considered (default is None, which means all the basis functions are considered).
        tol : float, optional
            Tolerance to exit the stability loop (default is 0.2).
        verbose : bool, optional
            If True, prints the progress of the algorithm (default is True).

        """

        if Nmax is None:
            Nmax = len(basis_functions)

        assert Mmax >= Nmax, "Mmax must be greater or equal to Nmax."
        assert tol > 0, "tol must be greater than 0."

        if basis_functions.fun_shape == self.grid.n_cells:
            use_centroids = True
            self.Nh = self.grid.n_cells
        elif basis_functions.fun_shape == self.grid.n_points:
            use_centroids = False
            self.Nh = self.grid.n_points
        else:
            if basis_functions.fun_shape == self.grid.n_cells * self.gdim or basis_functions.fun_shape == self.grid.n_points * self.gdim:
                raise ValueError("The basis functions seem to be vector-valued. Please provide scalar-valued basis functions.")
            else:
                raise ValueError(f"The shape of the basis functions {basis_functions.fun_shape} does not match the number of cells {self.grid.n_cells} or points {self.grid.n_points} of the grid.")
            
        # Create the sensors library and initialize the sensors object
        self.sensor_centers = list()
        if self.sensors_type == 'Gaussian':
            self.sensors_library = GaussianSensorLibrary(self.grid, gdim=self.gdim, use_centroids=use_centroids)
            self.sensors_library.create_library(**sensor_params, verbose=verbose)
            self.sensors = GaussianSensorLibrary(self.grid, gdim=self.gdim, use_centroids=use_centroids)
        
        elif self.sensors_type == 'Exponential':
            self.sensors_library = ExponentialSensorLibrary(self.grid, gdim=self.gdim, use_centroids=use_centroids)
            self.sensors_library.create_library(**sensor_params, verbose=verbose)
            self.sensors = ExponentialSensorLibrary(self.grid, gdim=self.gdim, use_centroids=use_centroids)
        
        elif self.sensors_type == 'IndicatorFunction':
            self.sensors_library = IndicatorFunctionSensorLibrary(self.grid, gdim=self.gdim, use_centroids=use_centroids)
            self.sensors_library.create_library(**sensor_params, verbose=verbose)
            self.sensors = IndicatorFunctionSensorLibrary(self.grid, gdim=self.gdim, use_centroids=use_centroids)

        # Initialize the inf-sup constants list
        inf_sup_constants = list()

        # Define first point
        sens_idx = np.argmax(
            np.abs(self.sensors_library.action(basis_functions(0)))
        )
        self.sensors.add_sensor(
            self.sensors_library.library[sens_idx]
        )
        self.sensor_centers.append(
            self.sensors_library.xm_list[sens_idx]
        )

        mm = 1 # number of sensors placed

        # Stability loop
        while mm < Mmax:

            nn = np.min([mm, Nmax])

            # Compute the inf-sup constant and the least stable mode
            _inf_sup_constant, eigenvec = compute_inf_sup(
                self.sensors, basis_functions, self.calculator, N=nn, return_eigenvector=True
            )

            inf_sup_constants.append( _inf_sup_constant )

            # Print the progress
            if verbose:
                print(f'SGREEDY: m = {mm+0:02}, n = {nn+0:02} | beta_n,m = {inf_sup_constants[mm-1]:.6f}', end = "\r")

            # Compute the least stable mode
            w_inf = basis_functions.lin_combine(eigenvec)

            # Compute projection of w_inf onto the space spanned by the basis sensors
            _action_on_winf = self.sensors.action(w_inf)
            matrix_A = np.zeros((mm, mm))
            for ii in range(mm):
                for jj in range(mm):
                    if jj >= ii:
                        matrix_A[ii, jj] = self.calculator.L2_inner_product(
                            self.sensors.library(ii), self.sensors.library(jj)
                        )
                    else:
                        matrix_A[ii, jj] = matrix_A[jj, ii]
                        
            _coeffs = scipy.linalg.solve(matrix_A, _action_on_winf)
            residual = self.sensors.library.lin_combine(_coeffs) - w_inf

            # Select the next sensor as the one maximizing the residual
            _measure = np.abs(self.sensors_library.action(residual))
            sens_idx = np.argmax(_measure)
            self.sensors.add_sensor(
                self.sensors_library.library[sens_idx]
            )
            self.sensor_centers.append(
                self.sensors_library.xm_list[sens_idx]
            )
            
            # Update the number of sensors placed
            mm += 1

            # Check the stopping criterion
            if inf_sup_constants[-1] >= tol:
                if verbose:
                    print(f'SGREEDY: Stability loop exited at m = {mm:02} with inf-sup constant {inf_sup_constants[-1]:.6f} >= tol = {tol:.2f}.')
                break


        if mm < Mmax:
            if verbose: 
                print(f'SGREEDY: Maximum number of sensors Mmax = {Mmax} not reached yet -> Starting approximation loop')

            # Approximation loop
            _available_nodes = np.array(self.sensors_library.xm_list)

            while mm <= Mmax:

                min_xdofs = np.zeros((len(_available_nodes),))
                for ii, _node in enumerate(_available_nodes):
                    min_xdofs[ii] = np.min(np.linalg.norm(np.asarray(self.sensor_centers) - _node, axis=1))

                idx_max = np.argmax(min_xdofs)

                # Add the sensor
                self.sensors.add_sensor(
                    self.sensors_library.library[idx_max]
                )
                self.sensor_centers.append(
                    self.sensors_library.xm_list[idx_max]
                )

                # Update the number of sensors placed
                mm += 1

    def plot_sensors(self, M: int = None, view = 'xy',
                     cmap = 'jet', levels=50,
                     color_sens = 'black',
                     fontsize_sens = 10,
                     fig_length = 6,
                     show_ticks = False):
        r"""
        Plot the Riesz representation of the first M sensors, cumulated:

        .. math::
            G(\mathbf{x}) = \sum_{m=1}^M \frac{g_m(\mathbf{x})}{\max |g_m|}

        Parameters
        ----------
        M : int, optional (Default = None)
            Number of sensors to plot. If `None`, all available sensors are plotted.
        view : str, optional (Default = 'xy')
            View of the plot. It can be either 'xy', 'xz' or 'yz'.
        cmap : str, optional (Default = 'jet')
            Colormap to use for the plot.
        levels : int, optional (Default = 50)
            Number of levels to use for the contour plot.
        color_sens : str, optional (Default = 'black')
            Color of the sensor labels.
        fontsize_sens : int, optional (Default = 10)
            Font size of the sensor labels.
        fig_length : float, optional (Default = 6)
            Length of the figure in inches.
        show_ticks : bool, optional (Default = False)
            If `True`, show the ticks on the axes.
        """

        if M is None:
            M = len(self.sensors.library)

        assert M <= len(self.sensors.library), "M cannot be larger than the number of sensors available"

        if view == 'xy':
            length = fig_length
            width  = fig_length * (self.grid.bounds[3] - self.grid.bounds[2]) / (self.grid.bounds[1] - self.grid.bounds[0])

            nodes = self.sensors.nodes[:, :2]

        elif view == 'xz':
            length = fig_length
            width  = fig_length * (self.grid.bounds[5] - self.grid.bounds[4]) / (self.grid.bounds[1] - self.grid.bounds[0])

            nodes = self.sensors.nodes[:, [0, 2]]

        elif view == 'yz':
            length = fig_length
            width  = fig_length * (self.grid.bounds[5] - self.grid.bounds[4]) / (self.grid.bounds[3] - self.grid.bounds[2])

            nodes = self.sensors.nodes[:, 1:]

        else:
            raise ValueError("view must be one of 'xy', 'xz', 'yz'")
        
        fig, axs = plt.subplots(1, 1, figsize=(length, width))
    
        _cumulative = np.zeros((nodes.shape[0],))
        for mm in range(M):
            _sensor_fun = self.sensors.library(mm)
            _cumulative += _sensor_fun / np.max(np.abs(_sensor_fun))

        sc = axs.tricontourf(*nodes.T, _cumulative, cmap=cmap, levels=levels, show_edges=False)
        plt.colorbar(sc, ax=axs)

        _max_idx = np.argmax(self.sensors.library.return_matrix(), axis=0)[:M]

        # Add label at each sensor position
        for ii in range(len(_max_idx)):
            idx = _max_idx[ii]
            axs.text(nodes[idx, 0], nodes[idx, 1], str(ii + 1), color=color_sens, fontsize=fontsize_sens, 
                     ha='left', va='bottom')

        if not show_ticks:
            axs.set_xticks([])
            axs.set_yticks([])

        return fig
    

    def save(self, path_folder: str, **kwargs):
        r"""
        Save the magic functions and sensors to a specified path.

        Parameters
        ----------
        path_folder : str
            The folder path where the model will be saved.
        **kwargs : dict
            Additional keyword arguments for saving options.
        """

        os.makedirs(path_folder, exist_ok=True)
        
        # Save the basis sensors
        self.sensors.library.store( f'sgreedy_sens_{self.varname}',
                                    filename=os.path.join(path_folder, f'sgreedy_sens_{self.varname}'),
                                    **kwargs)

def compute_inf_sup(sensors: SensorLibraryBase, basis_functions: FunctionsList, calculator: IntegralCalculator, 
                              N: int, return_eigenvector: bool = False):
    r"""
    Compute the inf-sup constants for a given set of sensors and basis functions.

    Parameters
    ----------
    sensors : FunctionsList
        The list of sensors.
    basis_functions : FunctionsList
        The reduced basis functions.
    calculator : IntegralCalculator
        The integral calculator object.
    Nmax : int, optional
        The maximum number of basis functions to be considered (default is None, which means all the basis functions are considered).

    Returns
    -------
    inf_sup_constants : list
        The list of inf-sup constants at each iteration.
    """

    mm = len(sensors)
    assert N <= len(basis_functions), "N must be less or equal to the number of basis functions."

    # Build the matrices A, K, Z for the inf-sup constant computation
    matrix_A = np.zeros((mm, mm))
    matrix_K = np.zeros((mm, N))
    matrix_Z = np.zeros((N, N))

    # Fill the matrices A, K
    for ii in range(mm):
        
        # Fill matrix A
        for jj in range(mm):
            if jj >= ii:
                matrix_A[ii, jj] = calculator.L2_inner_product(
                    sensors.library(ii), sensors.library(jj)
                )
            else:
                matrix_A[ii, jj] = matrix_A[jj, ii]
        
        # Fill matrix K
        for kk in range(N):
            matrix_K[ii, kk] = sensors._action_single(
                basis_functions(kk), ii
            )

    # Fill matrix Z
    for ii in range(N):
        for jj in range(N):
            if jj >= ii:
                matrix_Z[ii, jj] = calculator.L2_inner_product(
                    basis_functions(ii), basis_functions(jj)
                )
            else:
                matrix_Z[ii, jj] = matrix_Z[jj, ii]

    # Assemble Schur complement
    schur_compl = np.linalg.multi_dot([
        matrix_K.T, np.linalg.inv(matrix_A), matrix_K
    ])

    # Solve the eigenvalue problem for the inf-sup constant (Schur complement is Hermitian)
    beta_squared, eigenvec = scipy.linalg.eigh(schur_compl, matrix_Z)       

    # Assign the inf-sup constant
    idx_min_eig = np.argmin(beta_squared)
    inf_sup_constant = np.sqrt(beta_squared[idx_min_eig])

    if not return_eigenvector:
        return inf_sup_constant
    else:
        return inf_sup_constant, eigenvec[:, idx_min_eig]