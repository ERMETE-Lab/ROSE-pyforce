# Online Phase: Parameterised Background Data-Weak (PBDW) Method - including regularized version
# Author: Stefano Riva, PhD Student, NRG, Politecnico di Milano
# Latest Code Update: 07 October 2025
# Latest Doc  Update: 07 October 2025

import warnings
import numpy as np
import pyvista as pv
from collections import namedtuple
import os
import scipy

from ..tools.functions_list import FunctionsList
from ..tools.backends import IntegralCalculator, LoopProgress, Timer
from ..tools.write_read import ImportFunctionsList
from .online_base import OnlineDDROM, OnlineSensors

class PBDW(OnlineDDROM):
    r"""
    A class to estimate the state using the PBDW.

    This class implements the parameterized background data-weak (PBDW) method to estimate the state using measurements.

    Parameters
    ----------
    grid : pyvista.UnstructuredGrid
        The computational grid. It is used to define the spatial domain of the snapshots.
    gdim : int, optional (Default = 3)
        The geometric dimension of the grid. It can be either 2 or 3.
    varname : str, optional
        The name of the variable to be used for the POD. Default is 'u'.

    """

    def __init__(self, grid: pv.UnstructuredGrid, gdim: int = 3, varname: str = 'u'):

        super().__init__(grid, gdim=gdim, varname=varname)

        self.matrix_K = None
        self.matrix_A = None
        self.matrix_Z = None

    def set_basis(self, basis: FunctionsList = None, path_folder: str = None, **kwargs):
        """
        Assign the basis functions of the reduced space :math:`Z_N` either from a FunctionsList object or by loading from a folder.

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
        
        self.N = len(self._basis)
        
    def set_basis_sensors(self, sensors: FunctionsList = None, path_folder: str = None, **kwargs):
        """
        Assign the basis sensors to the PBDW model either from a FunctionsList object or by loading from a folder.
        At the moment, they are the Riesz representation in :math:`L^2` of the functionals.

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
        
    def compute_matrices(self):
        r"""

        Compute the matrices K, A and Z used in the PBDW: given :math:`\{g_j\}_{j=1}^M` the basis sensors and :math:`\{\zeta_i\}_{i=1}^N` the basis functions

        .. math::
            \mathbb{A}_{ij} = (g_i, g_j)_{L^2} \in\mathbb{R}^{M\times M}

        .. math::
            \mathbb{K}_{ji} = (\zeta_i, g_j)_{L^2} \in\mathbb{R}^{M\times N}

        .. math::
            \mathbb{Z}_{ij} = (\zeta_i, \zeta_j)_{L^2} \in\mathbb{R}^{N\times N}

        where :math:`(\cdot, \cdot)_{L^2}` is the :math:`L^2` inner product.

        """

        M = len(self.sensors._library)

        assert M >= self.N, f"Number of sensors {M} must be larger than or equal to the number of basis functions {self.N}."

        self.matrix_A = self.sensors.action(self.sensors._library) # shape (M, M)
        self.matrix_K = self.sensors.action(self._basis)  # shape (M, N)
        
        self.matrix_Z = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(i, self.N):
                if i >= j:
                    self.matrix_Z[i, j] = self.calculator.L2_inner_product(self._basis(i), self._basis(j))
                else:
                    self.matrix_Z[j, i] = self.matrix_Z[i, j]

    def get_measurements(self, snaps: FunctionsList | np.ndarray, M: int = None,
                         noise_std: float = 0.0):
        r"""
        This method extracts the measures from the input functions at the sensor locations.

        Parameters
        ----------
        snaps : FunctionsList or np.ndarray
            Function object to extract the measures from.
        M : int, optional (Default = None)
            Number of sensors to use for the extraction. If `None`, all available sensors are used.
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

    def _reconstruct(self, alpha_coeffs: np.ndarray, theta_coeffs: np.ndarray):
        r"""
        This method reconstructs the state from the reduced coefficients :math:`\{\alpha_n\}_{n=1}^N` and :math:`\{\theta_m\}_{m=1}^M`
        
        .. math::
            u(\cdot;\,\boldsymbol{\mu}) = \sum_{n=1}^N \alpha_n(\boldsymbol{\mu}) \zeta_n(\cdot)+\sum_{m=1}^M \theta_m(\boldsymbol{\mu}) g_m(\cdot)

        given the basis functions :math:`\{\zeta_n\}_{n=1}^N` and the basis sensors :math:`\{g_m\}_{m=1}^M`.

        Parameters
        ----------
        alpha_coeffs : np.ndarray
            Array of coefficients for the basis functions :math:`\{\alpha_n\}_{n=1}^N`, shaped :math:`(N,N_s)`.
        theta_coeffs : np.ndarray
            Array of coefficients for the basis sensors :math:`\{\theta_m\}_{m=1}^M`, shaped :math:`(M,N_s)`.
        
        Returns
        -------
        estimation : FunctionsList
            An instance of `FunctionsList` containing the estimated state.
        """

        assert alpha_coeffs.shape[0] <= self.N, "The number of coefficients must be less than or equal to the number of basis functions."
        assert theta_coeffs.shape[0] <= len(self.sensors._library), "The number of coefficients must be less than or equal to the number of sensors."
        alpha_coeffs = np.atleast_2d(alpha_coeffs)
        theta_coeffs = np.atleast_2d(theta_coeffs)

        assert alpha_coeffs.shape[1] == theta_coeffs.shape[1], "The number of columns of alpha_coeffs and theta_coeffs must be equal."

        estimation = FunctionsList(self.basis.fun_shape)

        for mu in range(alpha_coeffs.shape[1]):
            estimation.append(
                self.basis.lin_combine(alpha_coeffs[:, mu]) + self.sensors._library.lin_combine(theta_coeffs[:, mu])
            )

        return estimation
    
    def _solve_pbdw_linear_system(self, measures: np.ndarray, xi: float = 0.0):
        r"""
        Computes the reduced coefficients :math:`\{\alpha_n\}_{n=1}^{N}}` and :math:`\{\theta_m\}_{m=1}^{M}}` with as many basis sensors as the input measures :math:`M`. The following linear system is solved

        .. math::
            \left[ 
                \begin{array}{ccc}
                    \xi \cdot M \cdot \mathbb{I} + \mathbb{A} & & \mathbb{K}  \\  & & \\
                    \mathbb{K}^T & & 0
                \end{array}
                \right] \cdot
                \left[ 
                \begin{array}{c}
                    \boldsymbol{\theta} \\ \\ \boldsymbol{\alpha}
                \end{array}
                \right]   =
                \left[ 
                \begin{array}{c}
                    \mathbf{y} \\ \\ \mathbf{0}
                \end{array}
            \right]

        given :math:`\mathbf{y}\in\mathbb{R}^M` as the measurements vector.

        Parameters
        ----------
        measures : np.ndarray
            Measurements vector :math:`\{y_m\}_{m=1}^{M_{max}}` of the function `u` at the sensors locations.
        xi : float, optional (Default = 0.0)
            Regularization parameter for the regularization. Default is 0.0

        Returns
        ----------
        beta_coeff : np.ndarray
            Array of coefficients for the interpolant :math:`\{\beta_m\}_{m=1}^{M_{max}}`
        """

        assert measures.shape[0] >= len(self._basis), "The number of measures must be larger than or equal to the number of basis functions"
        assert measures.shape[0] <= len(self.sensors._library), "The number of measures must be less than or equal to the number of sensors"
        
        if self.matrix_A is None or self.matrix_K is None or self.matrix_Z is None:
            self.compute_matrices()

        _m = measures.shape[0]
        _A = self.matrix_A[:_m, :_m] + xi * _m * np.eye(_m)
        _K = self.matrix_K[:_m]
        _Z = self.matrix_Z

        # Assemble the full linear system
        sys_matrix = np.block([
            [_A, _K],
            [_K.T, np.zeros((_Z.shape[0], _Z.shape[0]))]
        ])
        
        # Right-hand side
        rhs = np.concatenate([measures, np.zeros((_Z.shape[0], measures.shape[1]))], axis=0)

        # Solve the linear system
        _coeffs = scipy.linalg.solve(sys_matrix, rhs)
        theta_coeff = _coeffs[:_m]
        alpha_coeff = _coeffs[_m:]

        return alpha_coeff, theta_coeff
    
    def estimate(self, measures: np.ndarray, xi: float = 0.0):
        r"""
        Estimate the state using the PBDW given the measures at the sensor locations.

        Parameters
        ----------
        measures : np.ndarray
            Measures :math:`\{y_m\}_{m=1}^{M}`, shaped :math:`(M, N_s)`.
        xi : float, optional (Default = 0.0)
            Regularization parameter for the regularization. Default is 0.0
                
        Returns
        -------
        estimation : FunctionsList
            An instance of `FunctionsList` containing the estimated state.
        """

        if self.matrix_A is None or self.matrix_K is None or self.matrix_Z is None:
            self.compute_matrices()

        alpha_coeffs, theta_coeffs = self._solve_pbdw_linear_system(measures, xi=xi)

        estimation = self._reconstruct(alpha_coeffs, theta_coeffs)

        return estimation
        
    def _reduce(self):
        r"""
        Not implemented for PBDW.

        """
        warnings.warn("The '_reduce' method is not implemented for PBDW.")
        pass

    def compute_errors(self, snaps: FunctionsList | np.ndarray, Mmax: int = None,
                       noise_std: float = 0.0, xi: float = 0.0,
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
        xi : float, optional (Default = 0.0)
            Regularization parameter for the regularization. Default is 0.0
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
            Mmax = len(self.sensors)
        else:
            assert Mmax <= len(self.sensors), f"Mmax={Mmax} must be less than or equal to the number of sensors, {len(self.sensors)}."

        if self.matrix_A is None or self.matrix_K is None or self.matrix_Z is None:
            self.compute_matrices()

        abs_err = np.zeros((Ns, Mmax - self.N + 1))
        rel_err = np.zeros((Ns, Mmax - self.N + 1))

        # Variables to store computational time
        computational_time = dict()
        computational_time['Measures']        = np.zeros((Ns, Mmax - self.N + 1))
        computational_time['StateEstimation'] = np.zeros((Ns, Mmax - self.N + 1))
        computational_time['Errors']          = np.zeros((Ns, Mmax - self.N + 1))

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
            progressBar = LoopProgress(msg = f"Computing errors - {self.varname}", final = Mmax - self.N + 1)

        for mm in range(self.N, Mmax+1):

            if verbose: 
                progressBar.update(1, percentage = False)

            # Extract measures
            timer.start()
            _measures = self.get_measurements(snaps, M=mm, noise_std=noise_std) # shape (M, Ns)
            computational_time['Measures'][:, mm - self.N] = timer.stop()

            timer.start()
            reconstructed_snaps = self.estimate(_measures, xi=xi)
            computational_time['StateEstimation'][:, mm - self.N] = timer.stop()

            for mu_i in range(Ns):
                timer.start()
                _resid = snaps[:, mu_i] - reconstructed_snaps(mu_i)
                abs_err[mu_i, mm - self.N] = self.calculator.L2_norm(_resid)
                rel_err[mu_i, mm - self.N] = abs_err[mu_i, mm - self.N] / _snap_norm[mu_i]
                computational_time['Errors'][mu_i, mm - self.N] += timer.stop()

        Results = namedtuple('Results', ['mean_abs_err', 'mean_rel_err', 'computational_time'])
        _res = Results(mean_abs_err = abs_err.mean(axis = 0), mean_rel_err = rel_err.mean(axis = 0), computational_time = computational_time)

        return _res