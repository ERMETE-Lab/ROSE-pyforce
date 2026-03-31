# Offline Phase: Generalised Empirical Interpolation Method (GEIM)
# Author: Stefano Riva, PhD Student, NRG, Politecnico di Milano
# Latest Code Update: 07 October 2025
# Latest Doc  Update: 07 October 2025

import numpy as np
import scipy
import matplotlib.pyplot as plt
from collections import namedtuple
import os

from ..tools.functions_list import FunctionsList
from ..tools.backends import IntegralCalculator, LoopProgress, Timer
from .offline_base import OfflineDDROM
from .sensors import GaussianSensorLibrary, ExponentialSensorLibrary, IndicatorFunctionSensorLibrary

class GEIM(OfflineDDROM):
    r"""
    A class implementing the Generalised Empirical Interpolation Method (GEIM) for dimensionality reduction and sensor placement.

    Parameters
    ----------
    grid: pv.UnstructuredGrid
        The computational grid.
    gdim: int, optional (default=3)
        The geometric dimension of the problem. It can be either 2 or 3.
    varname: str, optional (default='u')
        The name of the variable to analyse. Default is 'u'.
    sensors_type: str, optional (default='Gaussian')
        The type of sensors to use in the GEIM algorithm. It can be either 'Gaussian', 'Exponential', or 'IndicatorFunction'.
    """

    def __init__(self, grid, gdim=3, varname='u', sensors_type = 'Gaussian'):

        super().__init__(grid, gdim=gdim, varname=varname)

        _allowed_sensor_types = ['Gaussian', 'Exponential', 'IndicatorFunction']
        assert sensors_type in _allowed_sensor_types, f"sensors_type must be one of {_allowed_sensor_types}"
        self.sensors_type = sensors_type

    def fit(self, train_snaps: FunctionsList, Mmax: int, 
            sensor_params: dict,
            verbose= False):
        r"""
        The greedy algorithm chooses the magic functions and magic points by minimising the reconstruction error.
        
        Parameters
        ----------
        train_snaps : FunctionsList
            List of snapshots serving as training set.
        Mmax : int
            Integer input indicating the maximum number of functions and sensors to define
        sensor_params : dict
            Dictionary containing the parameters for the sensor library creation.
        verbose : boolean, optional (Default = False) 
            If `True`, print of the progress is enabled.
            
        Returns
        ----------
        maxAbsErr : np.ndarray
            Maximum absolute error measured in :math:`L^\infty` (max norm)
        maxRelErr : np.ndarray
            Maximum relative error measured in :math:`L^\infty` (max norm)
        beta_coeff : np.ndarray
            Matrix of the reduced coefficients :math:`\{ \beta_m \}_{m=1}^M`, obtained by greedy procedure
        """

        Ns = len(train_snaps)

        if train_snaps.fun_shape == self.grid.n_cells:
            use_centroids = True
            self.Nh = self.grid.n_cells
        elif train_snaps.fun_shape == self.grid.n_points:
            use_centroids = False
            self.Nh = self.grid.n_points
        else:
            if train_snaps.fun_shape == self.grid.n_cells * self.gdim or train_snaps.fun_shape == self.grid.n_points * self.gdim:
                raise ValueError("The input functions seem to be vector fields. Please provide scalar fields for GEIM.")
            else:
                raise ValueError(f"The shape of the input functions {train_snaps.fun_shape} does not match the number of cells {self.grid.n_cells} or points {self.grid.n_points} of the grid.")

        if self.sensors_type == 'Gaussian':
            self.sensors_library = GaussianSensorLibrary(self.grid, gdim=self.gdim, use_centroids=use_centroids)
            self.sensors_library.create_library(**sensor_params, verbose=verbose)
            self.magic_sensors = GaussianSensorLibrary(self.grid, gdim=self.gdim, use_centroids=use_centroids)
        
        elif self.sensors_type == 'Exponential':
            self.sensors_library = ExponentialSensorLibrary(self.grid, gdim=self.gdim, use_centroids=use_centroids)
            self.sensors_library.create_library(**sensor_params, verbose=verbose)
            self.magic_sensors = ExponentialSensorLibrary(self.grid, gdim=self.gdim, use_centroids=use_centroids)
        
        elif self.sensors_type == 'IndicatorFunction':
            self.sensors_library = IndicatorFunctionSensorLibrary(self.grid, gdim=self.gdim, use_centroids=use_centroids)
            self.sensors_library.create_library(**sensor_params, verbose=verbose)
            self.magic_sensors = IndicatorFunctionSensorLibrary(self.grid, gdim=self.gdim, use_centroids=use_centroids)

        # Initialize variables
        snaps_norms = list()
        self.magic_functions = FunctionsList(dofs = train_snaps.fun_shape)

        # Find first generating function: maximizing the L2 norm
        for f in train_snaps:
            snaps_norms.append(
                self.calculator.L2_norm(f)
            )
        
        generating_idx = np.argmax(snaps_norms)

        # Find first magic sensor
        _measure = self.sensors_library.action(train_snaps(generating_idx))
        sens_idx = np.argmax(np.abs(_measure))

        self.magic_sensors.add_sensor(
            self.sensors_library.library[sens_idx]
        )
        
        # First magic function
        self.magic_functions.append(
            train_snaps(generating_idx) / _measure[sens_idx]
        )

        # GEIM main loop

        beta_coeff = np.zeros((Mmax, Ns))
        
        maxAbsErr = np.zeros((Mmax,))
        maxRelErr = np.zeros((Mmax,))

        self.matrix_B = np.zeros((Mmax, Mmax))

        for mm in range(Mmax):
            
            # Compute matrix B
            for ii in range(mm+1):
                self.matrix_B[mm, ii] = self.magic_sensors._action_single(self.magic_functions(ii), mm)
            
            # Compute measures and coefficients of GEIM
            _measures = self.magic_sensors.action(train_snaps) # shape (mm+1, Ns)
            beta_coeff[:mm+1] = self._solve_geim_linear_system(_measures)

            # Compute residual fields as a matrix of size (N_h, Ns)
            resids = train_snaps.return_matrix() - self._reconstruct_from_coeffs(beta_coeff[:mm+1]).return_matrix()

            # Compute norms of the residuals
            _abs_norm_resid = list()
            for mu in range(Ns):
                _abs_norm_resid.append(
                    self.calculator.L2_norm(resids[:, mu])
                )

            # Find next generating function
            generating_idx = np.argmax(_abs_norm_resid)
            maxAbsErr[mm] = _abs_norm_resid[generating_idx]
            maxRelErr[mm] = maxAbsErr[mm] / snaps_norms[generating_idx]

            if mm < Mmax - 1:
                # Find next magic sensor
                _measure_resid = self.sensors_library.action(resids[:, generating_idx])

                sens_idx = np.argmax(np.abs(_measure_resid))
                self.magic_sensors.add_sensor(
                    self.sensors_library.library[sens_idx]
                )

                # Generate next magic function
                self.magic_functions.append(
                    resids[:, generating_idx] / _measure_resid[sens_idx]
                )

            # Progress print
            if verbose:
                print(f"GEIM Iteration {mm+1}/{Mmax} - MaxAbsErr: {maxAbsErr[mm]:.4e} - MaxRelErr: {maxRelErr[mm]:.4e}   ", end='\r')

        return maxAbsErr, maxRelErr, beta_coeff

    def _solve_geim_linear_system(self, measures: np.ndarray):
        r"""
        Computes the reduced coefficients :math:`\{\beta_m\}_{m=1}^{M}}` with as many magic functions/points (synthetic) as the input measures :math:`M`.
        
        .. math::
            y_m = u(\vec{x}_m) \qquad m = 1, \dots, M_{max}
        
        Parameters
        ----------
        measures : np.ndarray
            Measurements vector :math:`\{y_m\}_{m=1}^{M_{max}}` of the function `u` at the sensors locations.

        Returns
        ----------
        beta_coeff : np.ndarray
            Array of coefficients for the interpolant :math:`\{\beta_m\}_{m=1}^{M_{max}}`
        """

        assert measures.shape[0] <= len(self.magic_functions), "The number of measures must be equal to the number of magic functions"

        _m = measures.shape[0]
        beta_coeff = scipy.linalg.solve(self.matrix_B[:_m, :_m], measures[:_m], lower = True)

        return beta_coeff
    
    def _reconstruct_from_coeffs(self, coeffs: np.ndarray):
        r"""
        This method reconstructs the function `u` from the reduced coefficients :math:`\{\beta_m\}_{m=1}^M` using the magic functions :math:`\{q_m\}_{m=1}^M`.

        .. math::
            u(\cdot;\,\boldsymbol{\mu}) = \sum_{k=1}^N \beta_k(\boldsymbol{\mu}) q_k(\cdot)


        Parameters
        ----------
        coeffs : np.ndarray
            Reduced coefficients :math:`\{\beta_m\}_{m=1}^M`, shaped :math:`(M,N_s)`.

        Returns
        -------
        u : FunctionsList
            Reconstructed functions.
        """

        assert coeffs.shape[0] <= len(self.magic_functions), "The number of coefficients must be lower to the number of magic functions"
        coeffs = np.atleast_2d(coeffs)

        reconstructed_snaps = FunctionsList(dofs = self.Nh)

        for nn in range(coeffs.shape[1]):
            reconstructed_snaps.append(
                self.magic_functions.lin_combine(coeffs[:, nn])
            )

        return reconstructed_snaps

    def reconstruct(self, measures: np.ndarray):
        r"""
        This method reconstructs the state by obtaining the reduced coefficients :math:`\{\beta_m\}_{m=1}^M` from the input measures :math:`\{y_m\}_{m=1}^M` by solving the EIM linear system

        .. math::
            \mathbb{B} \boldsymbol{\beta} = \mathbf{y}

        where
        - :math:`\mathbb{B}` is the matrix of magic functions
        - :math:`\boldsymbol{\beta}` are the reduced coefficients
        - :math:`\mathbf{y}` are the input measures

        Parameters
        ----------
        measures : np.ndarray
            Input measures :math:`\{y_m\}_{m=1}^M`, shaped :math:`(M,N_s)`.

        Returns
        -------
        u : FunctionsList
            Reconstructed functions.
        """
        beta_coeff = self._solve_geim_linear_system(measures)
        return self._reconstruct_from_coeffs(beta_coeff)

    def _get_measures(self, snaps: FunctionsList | np.ndarray, M: int = None):
        r"""
        This method extracts the measures from the input functions at the magic points locations.

        Parameters
        ----------
        snaps : FunctionsList or np.ndarray
            Function object to extract the measures from.
        M : int, optional (Default = None)
            Number of magic points to use for the extraction. If `None`, all available magic points

        Returns
        -------
        measures : np.ndarray
            Measures :math:`\{y_m\}_{m=1}^M`, shaped :math:`(M,N_s)`.
        """

        if isinstance(snaps, FunctionsList):
            snaps = snaps.return_matrix()
        elif isinstance(snaps, np.ndarray):
            if snaps.ndim == 1:
                snaps = np.atleast_2d(snaps).T # shape (N, 1)
        else:
            raise TypeError("Input must be a FunctionsList or a numpy ndarray.")

        assert snaps.shape[0] == self.magic_functions.fun_shape, f"Input shape {snaps.shape[0]} must match the magic function shape {self.magic_functions.fun_shape}."

        if M is None:
            M = len(self.magic_functions)
        else:
            assert M <= len(self.magic_functions), "M cannot be larger than the number of magic functions available"

        return self.magic_sensors.action(snaps, M=M) # shape (M, Ns)

    def reduce(self, snaps: FunctionsList | np.ndarray, M: int = None):
        r"""
        This method can be used to generate the reduced coefficients based on the magic functions and points, by extracting the measures from the input functions and solve the associated EIM linear system.

        Parameters
        ----------
        snaps : FunctionsList or np.ndarray
            Function object to project onto the reduced space of dimension `M`.
        M : int, optional (Default = None)
            Number of magic functions/points to use for the projection. If `None`, all available magic functions/points are used.

        Returns
        -------
        beta_coeff : np.ndarray
            Reduced coefficients :math:`\{\beta_m\}_{m=1}^M`, shaped :math:`(M,N_s)`.

        """

        if M is None:
            M = len(self.magic_functions)
        else:
            assert M <= len(self.magic_functions), "M cannot be larger than the number of magic functions available"

        if isinstance(snaps, FunctionsList):
            snaps = snaps.return_matrix()
        elif isinstance(snaps, np.ndarray):
            if snaps.ndim == 1:
                snaps = np.atleast_2d(snaps).T # shape (Nh, 1)
        else:
            raise TypeError("Input must be a FunctionsList or a numpy ndarray.")

        assert snaps.shape[0] == self.magic_functions.fun_shape, f"Input shape {snaps.shape[0]} must match the magic function shape {self.magic_functions.fun_shape}."

        measures = self._get_measures(snaps, M=M) # shape (M, Ns)

        return self._solve_geim_linear_system(measures)
    
    def compute_errors(self, snaps: FunctionsList | np.ndarray, Mmax: int = None, verbose: bool = False):
        r"""
        Computes the errors between the original snapshots and the reconstructed ones.

        Parameters
        ----------
        snaps : FunctionsList or np.ndarray
            Original snapshots to compare with.
        Mmax : int, optional
            Maximum number of sensors and magic functions to use for the reconstruction. If None, all reduced basis is used.
        verbose : bool, optional
            If True, print progress messages. Default is False.

        Returns
        ----------
        mean_abs_err : np.ndarray
            Average absolute error measured in :math:`L^2`.
        mean_rel_err : np.ndarray
            Average relative error measured in :math:`L^2`.
        computational_time : dict
            Dictionary with the CPU time of the most relevant operations during the online phase.

        """

        if isinstance(snaps, FunctionsList):
            Ns = len(snaps)
            assert snaps.fun_shape == self.magic_functions.fun_shape, "The shape of the snapshots must match the shape of the magic functions."

            # Convert FunctionsList to numpy array for processing
            snaps = snaps.return_matrix()

        elif isinstance(snaps, np.ndarray):
            Ns = snaps.shape[1]
            assert snaps.shape[0] == self.magic_functions.fun_shape, "The shape of the snapshots must match the shape of the magic functions."

        else:
            raise TypeError("Input must be a FunctionsList or a numpy ndarray.")

        if Mmax is None:
            Mmax = len(self.magic_functions)
        else:
            assert Mmax <= len(self.magic_functions), f"Mmax={Mmax} must be less than or equal to the number of magic functions, {len(self.magic_functions)}."

        abs_err = np.zeros((Ns, Mmax))
        rel_err = np.zeros((Ns, Mmax))

        # Variables to store computational time
        computational_time = dict()
        computational_time['Measurements']    = np.zeros((Ns, Mmax))
        computational_time['StateEstimation'] = np.zeros((Ns, Mmax))
        computational_time['Errors']          = np.zeros((Ns, Mmax))

        timer = Timer()

        if verbose:
            print(f"Computing L2 norm of snapshot", end='\r')

        _snap_norm = list()
        for mu_i in range(Ns):

            timer.start()
            _snap_norm.append(
                self.calculator.L2_norm(snaps[:, mu_i])
            )
            
            computational_time['Errors'][mu_i, :] = timer.stop()
        
        if verbose: 
            progressBar = LoopProgress(msg = f"Computing errors - {self.varname}", final = Mmax)

        for mm in range(Mmax):

            if verbose: 
                progressBar.update(1, percentage = False)

            timer.start()
            _measures = self._get_measures(snaps, M=mm+1) # shape (M, Ns)
            computational_time['Measurements'][:, mm] = timer.stop()

            timer.start()
            reconstructed_snaps = self.reconstruct(_measures)
            computational_time['StateEstimation'][:, mm] = timer.stop()

            for mu_i in range(Ns):
                timer.start()
                _resid = snaps[:, mu_i] - reconstructed_snaps(mu_i)
                abs_err[mu_i, mm] = self.calculator.L2_norm(_resid)
                rel_err[mu_i, mm] = abs_err[mu_i, mm] / _snap_norm[mu_i]
                computational_time['Errors'][mu_i, mm] += timer.stop()

        Results = namedtuple('Results', ['mean_abs_err', 'mean_rel_err', 'computational_time'])
        _res = Results(mean_abs_err = abs_err.mean(axis = 0), mean_rel_err = rel_err.mean(axis = 0), computational_time = computational_time)

        return _res

    def compute_lebesgue_constant(self, verbose: bool = False):
        r"""
        This method computes the Lebesgue constant based on the magic functions and sensors, based on the procedure outlined in `Maday et al., 2015 <https://www.sciencedirect.com/science/article/pii/S0045782515000389>`_.
        
        Returns
        -------
        Lambda : float
            The Lebesgue constant.
        """

        if not hasattr(self, 'magic_functions'):
            raise ValueError("The model has not been fitted yet. Please call the 'fit' method before computing the Lebesgue constant.")

        M = len(self.magic_functions)

        # Initialize functions and sensors lists
        orth_magic_functions = FunctionsList(dofs = self.magic_functions.fun_shape)
        orth_magic_sensors   = FunctionsList(dofs = self.magic_sensors.library.fun_shape)

        if verbose:
            bar = LoopProgress(msg="Computing orthonormal basis", final=M)

        # Generate a set of orthonormal magic functions and sensors using Gram-Schmidt procedure
        for mm in range(M):

            _orth_fun = self.magic_functions(mm).copy()
            _orth_sen = self.magic_sensors.library(mm).copy()

            for jj in range(mm+1):
                if jj < mm:
                    _coeff_fun = self.calculator.L2_inner_product(_orth_fun, orth_magic_functions(jj)) / self.calculator.L2_norm(orth_magic_functions(jj))**2
                    _coeff_sen = self.calculator.L2_inner_product(_orth_sen, orth_magic_sensors(jj)) / self.calculator.L2_norm(orth_magic_sensors(jj))**2
                    
                    _orth_fun -= _coeff_fun * orth_magic_functions(jj)
                    _orth_sen -= _coeff_sen * orth_magic_sensors(jj)

            orth_magic_functions.append(_orth_fun / self.calculator.L2_norm(_orth_fun))
            orth_magic_sensors.append(_orth_sen / self.calculator.L2_norm(_orth_sen))

            if verbose:
                bar.update(1)

        if verbose:
            bar2  = LoopProgress(msg="Computing Lebesgue constant", final=M)

        # Compute the Lebesgue constant
        Lambda = list()

        for mm in range(M):
            _A = np.zeros((mm+1, mm+1))

            for ii in range(mm+1):
                for jj in range(mm+1):
                    _A[ii, jj] = self.calculator.L2_inner_product(orth_magic_functions(ii), orth_magic_sensors(jj))

            _eigvals = np.linalg.eigvals(_A.T @ _A)
            Lambda.append(
                1. / np.sqrt(np.min(_eigvals))
            )

            if verbose:
                bar2.update(1)

        return np.array(Lambda)
    
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
            M = len(self.magic_sensors.library)

        assert M <= len(self.magic_sensors.library), "M cannot be larger than the number of magic sensors available"

        if view == 'xy':
            length = fig_length
            width  = fig_length * (self.grid.bounds[3] - self.grid.bounds[2]) / (self.grid.bounds[1] - self.grid.bounds[0])

            nodes = self.magic_sensors.nodes[:, :2]

        elif view == 'xz':
            length = fig_length
            width  = fig_length * (self.grid.bounds[5] - self.grid.bounds[4]) / (self.grid.bounds[1] - self.grid.bounds[0])

            nodes = self.magic_sensors.nodes[:, [0, 2]]

        elif view == 'yz':
            length = fig_length
            width  = fig_length * (self.grid.bounds[5] - self.grid.bounds[4]) / (self.grid.bounds[3] - self.grid.bounds[2])

            nodes = self.magic_sensors.nodes[:, 1:]

        else:
            raise ValueError("view must be one of 'xy', 'xz', 'yz'")
        
        fig, axs = plt.subplots(1, 1, figsize=(length, width))
    
        _cumulative = np.zeros((nodes.shape[0],))
        for mm in range(M):
            _sensor_fun = self.magic_sensors.library(mm)
            _cumulative += _sensor_fun / np.max(np.abs(_sensor_fun))

        sc = axs.tricontourf(*nodes.T, _cumulative, cmap=cmap, levels=levels)
        plt.colorbar(sc, ax=axs)

        _max_idx = np.argmax(self.magic_sensors.library.return_matrix(), axis=0)[:M]

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

        # Save the magic functions
        self.magic_functions.store( f'mf_{self.varname}',
                                    filename=os.path.join(path_folder, f'mf_{self.varname}'),
                                    **kwargs)
        
        # Save the magic sensors
        self.magic_sensors.library.store( f'ms_{self.varname}',
                                          filename=os.path.join(path_folder, f'ms_{self.varname}'),
                                          **kwargs)