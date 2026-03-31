# Online Phase: Generalised Empirical Interpolation Method (EIM) - including regularized version
# Author: Stefano Riva, PhD Student, NRG, Politecnico di Milano
# Latest Code Update: 07 October 2025
# Latest Doc  Update: 07 October 2025

import numpy as np
import pyvista as pv
from collections import namedtuple
import os
import scipy

from ..tools.functions_list import FunctionsList
from ..tools.backends import IntegralCalculator, LoopProgress, Timer
from ..tools.write_read import ImportFunctionsList
from .online_base import OnlineDDROM, OnlineSensors

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class GEIM(OnlineDDROM):
    r"""
    A class to estimate the state using the GEIM.

    This class implements the generalized empirical interpolation method (GEIM) to estimate the state using measurements.
    
    Parameters
    ----------
    grid : pyvista.UnstructuredGrid
        The computational grid. It is used to define the spatial domain of the snapshots.
    gdim : int, optional (Default = 3)
        The geometric dimension of the grid. It can be either 2 or 3.
    varname : str, optional
        The name of the variable to be used. Default is 'u'.

    """

    def __init__(self, grid: pv.UnstructuredGrid, gdim: int = 3, varname: str = 'u'):

        super().__init__(grid, gdim=gdim, varname=varname)

        self.matrix_B = None
        self.tikhonov = None # Dictionary to store tikhonov regularization matrices
        self.train_beta_coeffs = None # Training reduced coefficients for Tikhonov regularization

    def set_basis(self, basis: FunctionsList = None, path_folder: str = None, **kwargs):
        """
        Assign the magic functions to the GEIM model either from a FunctionsList object or by loading from a folder.

        Parameters
        ----------
        basis : FunctionsList, optional
            An instance of `FunctionsList` containing the basis functions.
        path_folder : str, optional
            The path to the folder containing the basis functions.
        **kwargs : dict
            Additional keyword arguments to pass to the loading function.

        Returns
        -------
        None

        """

        if basis is not None:
            self._basis = basis
        elif path_folder is not None:
            _filename = os.path.join(path_folder, f'mf_{self.varname}')
            self._basis = ImportFunctionsList(_filename, **kwargs)
        else:
            raise ValueError("Either 'basis' or 'path_folder' must be provided.")
        
    def set_magic_sensors(self, sensors: FunctionsList = None, path_folder: str = None, **kwargs):
        """
        Assign the magic sensors to the GEIM model either from a FunctionsList object or by loading from a folder.

        Parameters
        ----------
        sensors : FunctionsList, optional
            An instance of `FunctionsList` containing the sensor functions.
        path_folder : str, optional
            The path to the folder containing the sensor functions.
        **kwargs : dict
            Additional keyword arguments to pass to the loading function.

        Returns
        -------
        None

        """

        if sensors is not None:
            self.sensors = OnlineSensors(library=sensors, gdim=self.gdim, grid=self.grid)
        elif path_folder is not None:
            _filename = os.path.join(path_folder, f'ms_{self.varname}')
            self.sensors = OnlineSensors(library=ImportFunctionsList(_filename, **kwargs), gdim=self.gdim, grid=self.grid)
        else:
            raise ValueError("Either 'sensors' or 'path_folder' must be provided.")
        
    def compute_B_matrix(self):
        r"""

        Compute the matrix B used in the GEIM: this matrix is the evaluation of the magic functions at the magic sensors.

        .. math::
            B_{ij} = v_i(q_j) \qquad i,j = 1, \ldots, M

        where :math:`\{q_j\}_{j=1}^M` are the magic functions and :math:`\{v_i\}_{i=1}^M` are the magic sensors.
    
        """

        M = len(self.sensors._library)

        assert M == len(self._basis), f"Number of magic sensors {M} must be equal to the number of magic functions {len(self._basis)}."

        self.matrix_B = self.sensors.action(self._basis)
        
    def set_tikhonov_matrices(self, beta_coeffs: np.ndarray = None, train_snaps: FunctionsList | np.ndarray = None):
        r"""
        This method is used to compute the matrices and vectors needed for the Tikhonov regularization, following `Introini et al. (2023) <https://doi.org/10.1016/j.cma.2022.115773>`_, from the training reduced coefficients.
        Given the training reduced coefficients :math:`\{\beta_m(\boldsymbol{\mu}_i)\}_{m=1}^{M}, i=1,\ldots,N_{train}` it computes the following:

        .. math::
            \langle \boldsymbol{\beta} \rangle = \sum_{i=1}^{N_{train}} \boldsymbol{\beta}(\boldsymbol{\mu}_i) \in\mathbb{R}^{M}

        .. math::
            \sigma_{\beta_j}^2 = \frac{1}{N_{train}-1}\sum_{i=1}^{N_{train}} (\beta_j(\boldsymbol{\mu}_i) - \langle \beta_j \rangle)^2 \qquad j=1,\ldots,M

        from which the Tikhonov matrix is defined:

        .. math::
            \mathbb{T} = \mathrm{diag}\left(\frac{1}{|\sigma_{\beta_1}|}, \ldots, \frac{1}{|\sigma_{\beta_M}|}\right) \in\mathbb{R}^{M\times M}

        The training coefficients can be provided as input or they can be computed from the training snapshots using the `_reduce` method.

        Parameters
        ----------
        beta_coeffs : np.ndarray, optional
            The training reduced coefficients, shaped :math:`(M, N_{train})`. If not provided, they will be computed from the training snapshots.
        train_snaps : FunctionsList or np.ndarray, optional
            The training snapshots to compute the reduced coefficients from. Required if `beta_coeffs` is not provided.

        """

        if beta_coeffs is not None:
            self.train_beta_coeffs = beta_coeffs
        elif train_snaps is not None:
            self.train_beta_coeffs = self._reduce(train_snaps)
        else:
            raise ValueError("Either 'beta_coeffs' or 'train_snaps' must be provided.")

        self.tikhonov = {
            'beta_mean': np.mean(self.train_beta_coeffs, axis=1), # shape (M,)
            'beta_std':  np.std(self.train_beta_coeffs, axis=1)
        }

        # Compute Tikhonov matrix
        self.tikhonov['T'] = np.diag(1.0 / (self.tikhonov['beta_std'] + 1e-16)) # shape (M, M)

    def get_measurements(self, snaps: FunctionsList | np.ndarray, M: int = None,
                         noise_std: float = 0.0):
        r"""
        This method extracts the measures from the input functions at the magic sensors locations.

        Parameters
        ----------
        snaps : FunctionsList or np.ndarray
            Function object to extract the measures from.
        M : int, optional (Default = None)
            Number of sensors to use for the extraction. If `None`, all available magic sensors are used.
        noise_std : float, optional (Default = 0.0)
            Standard deviation of the Gaussian noise to be added to the measurements. Default is 0.
            
        Returns
        -------
        measures : np.ndarray
            Measures :math:`\{y_m\}_{m=1}^M`, shaped :math:`(M,N_s)`.
        """

        if isinstance(snaps, FunctionsList):
            snaps = snaps.return_matrix()
        elif isinstance(snaps, np.ndarray):
            if snaps.ndim == 1:
                snaps = np.atleast_2d(snaps).T
        else:
            raise TypeError("Input must be a FunctionsList or a numpy ndarray.")

        assert snaps.shape[0] == self._basis.fun_shape, "Input shape must match the magic functions shape."

        measures = self.sensors.action(snaps, M=M) # shape (M, Ns)

        if noise_std > 0.0:
            noise = np.random.normal(0, noise_std, measures.shape)
            measures += noise

        return measures

    def _reconstruct(self, coeffs: np.ndarray):
        r"""
        This method reconstructs the state from the reduced coefficients :math:`\{\beta_m\}_{m=1}^M` using the magic functions :math:`\{q_m\}_{m=1}^M`.

        .. math::
            u(\cdot;\,\boldsymbol{\mu}) = \sum_{m=1}^M \beta_m(\boldsymbol{\mu}) q_m(\cdot)

        Parameters
        ----------
        coeffs : np.ndarray
            Reduced coefficients :math:`\{\beta_m\}_{m=1}^M`, shaped :math:`(M,N_s)`.

        Returns
        -------
        estimation : FunctionsList
            An instance of `FunctionsList` containing the estimated state.
        """

        assert coeffs.shape[0] <= len(self.basis), "The number of coefficients must be less than or equal to the number of POD modes."
        coeffs = np.atleast_2d(coeffs)

        estimation = FunctionsList(self.basis.fun_shape)

        for nn in range(coeffs.shape[1]):
            estimation.append(
                self.basis.lin_combine(coeffs[:, nn])
            )

        return estimation
    
    def _solve_geim_linear_system(self, measures: np.ndarray):
        r"""
        Computes the reduced coefficients :math:`\{\beta_m\}_{m=1}^{M}}` with as many magic functions/points (synthetic) as the input measures :math:`M`
        
        .. math::
            y_m = u(\vec{x}_m) \qquad m = 1, \dots, M_{max}

        from the following linear system

        .. math::
            \mathbb{B}\boldsymbol{\beta} = \mathbf{y}
        
        Parameters
        ----------
        measures : np.ndarray
            Measurements vector :math:`\{y_m\}_{m=1}^{M_{max}}` of the function `u` at the sensors locations.

        Returns
        ----------
        beta_coeff : np.ndarray
            Array of coefficients for the interpolant :math:`\{\beta_m\}_{m=1}^{M_{max}}`
        """

        assert measures.shape[0] <= len(self._basis), "The number of measures must be equal to the number of magic functions"

        if self.matrix_B is None:
            self.compute_B_matrix()

        _m = measures.shape[0]
        beta_coeff = scipy.linalg.solve(self.matrix_B[:_m, :_m], measures[:_m], lower=True)

        return beta_coeff
    
    def _solve_trgeim_linear_system(self, measures: np.ndarray, regularization_params: dict):
        r"""
        Computes the reduced coefficients :math:`\{\beta_m\}_{m=1}^{M}}` with as many magic functions/points (synthetic) as the input measures :math:`M`
        
        .. math::
            y_m = u(\vec{x}_m) \qquad m = 1, \dots, M_{max}

        from the following Tikhonov-regularized linear system

        .. math::
            (\mathbb{B}^T\mathbb{B} + \lambda \mathbb{T}^T\mathbb{T})\boldsymbol{\beta} = \mathbb{B}^T\mathbf{y}+\lambda \mathbb{T}^T\mathbb{T}\langle \boldsymbol{\beta}\rangle
        where :math:`\mathbb{T}` is the Tikhonov matrix, :math:`\langle \boldsymbol{\beta}\rangle` is the mean of the training reduced coefficients, and :math:`\lambda` is the regularization parameter.

        Parameters
        ----------
        measures : np.ndarray
            Measurements vector :math:`\{y_m\}_{m=1}^{M_{max}}` of the function `u` at the sensors locations.
        regularization_params : dict
            Dictionary containing the regularization parameters. It must contain the following keys:
            - 'type': str, the type of regularization.
            - 'lambda': float, the regularization parameter.

        Returns
        ----------
        beta_coeff : np.ndarray
            Array of coefficients for the interpolant :math:`\{\beta_m\}_{m=1}^{M_{max}}`
        """

        assert measures.shape[0] <= len(self._basis), "The number of measures must be equal to the number of magic functions"

        if self.matrix_B is None:
            self.compute_B_matrix()
        if self.tikhonov is None:
            raise ValueError("Tikhonov matrices not set. Please call 'set_tikhonov_matrices' method first.")

        # Extract quantities for the linear system
        _m = measures.shape[0]
        lambda_reg = regularization_params['lambda']
        _B = self.matrix_B[:_m, :_m]
        _T = self.tikhonov['T'][:_m, :_m]
        _beta_mean = self.tikhonov['beta_mean'][:_m]


        # Solve the linear system
        A = _B.T @ _B + lambda_reg * (_T.T @ _T)
        rhs = _B.T @ measures[:_m] + lambda_reg * (_T.T @ _T @ _beta_mean[:, np.newaxis])

        beta_coeff = scipy.linalg.solve(A, rhs)

        return beta_coeff
    
    def estimate(self, measures: np.ndarray,
                 regularization_params: dict = None):
        r"""
        Estimate the state using the GEIM given the measures at the sensor locations.

        Parameters
        ----------
        measures : np.ndarray
            Measures :math:`\{y_m\}_{m=1}^{M_{max}}`, shaped :math:`(M_{max},N_s)`.
        regularization_params : dict, optional (Default = None)
            Dictionary containing the regularization parameters. If `None`, no regularization is applied.
            At the moment, the only supported regularization is Tikhonov from `Introini et al. (2023) <https://doi.org/10.1016/j.cma.2022.115773>`_.
            
        Returns
        -------
        estimation : FunctionsList
            An instance of `FunctionsList` containing the estimated state.
        """

        assert measures.shape[0] <= len(self.sensors), f"The number of measures {measures.shape[0]} must be less than or equal to the number of magic sensors {len(self.sensors)}."

        if self.matrix_B is None:
            self.compute_B_matrix()

        if regularization_params is not None:
            if regularization_params['type'] == 'tikhonov':
                beta_coeff = self._solve_trgeim_linear_system(measures, regularization_params)
            else:
                raise ValueError(f"Regularization type {regularization_params['type']} not recognized: available types: 'tikhonov'.")
        else:
            beta_coeff = self._solve_geim_linear_system(measures)
        estimation = self._reconstruct(beta_coeff)

        return estimation
        
    def _reduce(self, snaps: FunctionsList | np.ndarray, M: int = None):
        r"""
        This method can be used to generate the reduced coefficients based on the magic functions and points, by extracting the measures from the input functions and solve the associated GEIM linear system.

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
            M = len(self._basis)
        else:
            assert M <= len(self._basis), "M cannot be larger than the number of magic functions available"

        if isinstance(snaps, FunctionsList):
            snaps = snaps.return_matrix()
        elif isinstance(snaps, np.ndarray):
            if snaps.ndim == 1:
                snaps = np.atleast_2d(snaps).T # shape (Nh, 1)
        else:
            raise TypeError("Input must be a FunctionsList or a numpy ndarray.")

        assert snaps.shape[0] == self._basis.fun_shape, "Input shape must match the SVD modes shape."

        if self.matrix_B is None:
            self.compute_B_matrix()

        measures = self.get_measurements(snaps, M)
        return self._solve_geim_linear_system(measures)

    def compute_errors(self, snaps: FunctionsList | np.ndarray, Mmax: int = None,
                       noise_std: float = 0.0, regularization_params: dict = None,
                       verbose: bool = False):
        r"""
        Computes the errors between the original snapshots and the reconstructed ones.

        Parameters
        ----------
        snaps : FunctionsList or np.ndarray
            Original snapshots to compare with.
        Mmax : int, optional
            Maximum number of sensors and magic functions to use for the reconstruction. If None, all reduced basis is used.
        noise_std : float, optional (Default = 0.0)
            Standard deviation of the Gaussian noise to be added to the measurements. Default is 0.
        regularization_params : dict, optional (Default = None)
            Dictionary containing the regularization parameters. If `None`, no regularization is applied.
            At the moment, the only supported regularization is Tikhonov from `Introini et al. (2023) <https://doi.org/10.1016/j.cma.2022.115773>`_.
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
            assert snaps.fun_shape == self._basis.fun_shape, "The shape of the snapshots must match the shape of the magic functions."

            # Convert FunctionsList to numpy array for processing
            snaps = snaps.return_matrix()

        elif isinstance(snaps, np.ndarray):
            Ns = snaps.shape[1]
            assert snaps.shape[0] == self._basis.fun_shape, "The shape of the snapshots must match the shape of the magic functions."

        else:
            raise TypeError("Input must be a FunctionsList or a numpy ndarray.")

        if Mmax is None:
            Mmax = len(self._basis)
        else:
            assert Mmax <= len(self._basis), f"Mmax={Mmax} must be less than or equal to the number of magic functions, {len(self._basis)}."

        if self.matrix_B is None:
            self.compute_B_matrix()

        abs_err = np.zeros((Ns, Mmax))
        rel_err = np.zeros((Ns, Mmax))

        # Variables to store computational time
        computational_time = dict()
        computational_time['Measures']        = np.zeros((Ns, Mmax))
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

            # Extract measures
            timer.start()
            _measures = self.get_measurements(snaps, M=mm+1, noise_std=noise_std) # shape (M, Ns)
            computational_time['Measures'][:, mm] = timer.stop()

            timer.start()
            reconstructed_snaps = self.estimate(_measures, regularization_params=regularization_params)
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