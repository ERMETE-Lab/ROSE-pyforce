# Online Phase: Failing sensors
# Author: Stefano Riva, PhD Student, NRG, Politecnico di Milano
# Latest Code Update: 24 October 2025
# Latest Doc  Update: 24 October 2025

import numpy as np
import pyvista as pv
from collections import namedtuple
import os
import scipy

from ..tools.functions_list import FunctionsList
from ..tools.backends import IntegralCalculator, LoopProgress, Timer
from ..tools.write_read import ImportFunctionsList
from .online_base import OnlineDDROM, OnlineSensors
from .geim import GEIM
from .pbdw import PBDW

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class FailingGEIM(GEIM):
    """
    Class for the online phase of GEIM with failing sensors.
    Inherits from GEIM class.

    Parameters
    ----------
    grid : pv.UnstructuredGrid
        The computational grid.
    gdim : int, optional
        Geometrical dimension of the problem (default is 3).
    varname : str, optional
        Name of the variable in the grid (default is 'u').
    """

    def __init__(self, grid: pv.UnstructuredGrid, gdim: int = 3, varname: str = 'u'):
        super().__init__(grid, gdim=gdim, varname=varname)
    

    def get_measurements(self, snaps: FunctionsList | np.ndarray, M: int = None,
                         noise_std: float = 0.0,
                         drift_dict: dict = None) -> np.ndarray:
        r"""
        This method extracts the measures from the input functions at the magic sensors locations. It allows also to add Gaussian noise to the measurements.

        Moreover, the drift failure of the sensor can be simulated by providing a drift_dict with the following keys:
        
        - 'kappa' : float
            Shift from the average value of the sensor.
        - 'rho' : float, optional 
            High frequency oscillation amplitude.
        - 'idx_failed' : list of int, optional
            List of indices of the failed sensors.
        - 'mu_failure' : int, optional (Default = 0)
            Starting index of the failure in the snapshots.

        Parameters
        ----------
        snaps : FunctionsList or np.ndarray
            Function object to extract the measures from.
        M : int, optional (Default = None)
            Number of sensors to use for the extraction. If `None`, all available magic sensors are used.
        noise_std : float, optional (Default = 0.0)
            Standard deviation of the Gaussian noise to be added to the measurements. Default is 0.
        drift_dict : dict, optional (Default = None)
            Dictionary containing the parameters for the drift failure simulation.
            
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

        # Handle sensor drift failure
        if drift_dict is not None:
            assert 'kappa' in drift_dict, "'drift_dict' must contain the key 'kappa'."
            kappa = drift_dict['kappa']

            assert 'rho' in drift_dict, "'drift_dict' must contain the key 'rho'."
            rho = drift_dict['rho']

            assert 'idx_failed' in drift_dict, "'drift_dict' must contain the key 'idx_failed'."
            idx_failed = drift_dict['idx_failed']
            assert min(idx_failed) >= 0 and max(idx_failed) < measures.shape[0], f"'idx_failed' must contain indices between 0 and {measures.shape[0]-1}."

            if 'mu_failure' in drift_dict:
                mu_failure = drift_dict['mu_failure']
            else:
                mu_failure = 0

            drift = np.random.normal(kappa, rho, size=measures[idx_failed, mu_failure:].shape)

            # Apply the drift to the failed sensors
            measures[idx_failed, mu_failure:] += drift

        return measures

    def _solve_hardfailure_trgeim_linear_system(self, measures: np.ndarray, 
                                                regularization_params: dict,
                                                hard_failure_dict: dict):
        r"""
        Computes the reduced coefficients :math:`\{\beta_m\}_{m=1}^{M}}` with as many magic functions/points (synthetic) as the input measures :math:`M`
        
        .. math::
            y_m = u(\vec{x}_m) \qquad m = 1, \dots, M_{max}

        from the following Tikhonov-regularized linear system

        .. math::
            (\mathbb{B}^T\mathbb{B} + \lambda \mathbb{T}^T\mathbb{T})\boldsymbol{\beta} = \mathbb{B}^T\mathbf{y}+\lambda \mathbb{T}^T\mathbb{T}\langle \boldsymbol{\beta}\rangle
        where :math:`\mathbb{T}` is the Tikhonov matrix, :math:`\langle \boldsymbol{\beta}\rangle` is the mean of the training reduced coefficients, and :math:`\lambda` is the regularization parameter.

        This method is able to handle hard failures by removing the failed sensors from the linear system after the failure occurs.

        Parameters
        ----------
        measures : np.ndarray
            Measurements vector :math:`\{y_m\}_{m=1}^{M_{max}}` of the function `u` at the sensors locations.
        regularization_params : dict
            Dictionary containing the regularization parameters. It must contain the following keys:
            - 'type': str, the type of regularization.
            - 'lambda': float, the regularization parameter.
        hard_failure_dict : dict
            Dictionary containing the parameters for the hard failure simulation. It must contain the following keys:
            - 'idx_failed': list of int, list of indices of the failed sensors.
            - 'mu_failure': int, starting index of the failure in the snapshots.

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

        # Assemble the linear system
        A = _B.T @ _B + lambda_reg * (_T.T @ _T)
        rhs = _B.T @ measures[:_m] + lambda_reg * (_T.T @ _T @ _beta_mean[:, np.newaxis])

        beta_coeff = np.zeros((_m, measures.shape[1]))

        for mu in range(measures.shape[1]):

            if mu < hard_failure_dict['mu_failure']:
                beta_coeff[:, mu] = scipy.linalg.solve(A, rhs[:, mu])
            else:
                _rhs = np.delete(rhs[:, mu], hard_failure_dict['idx_failed'], axis=0)
                _A = np.delete(
                                np.delete(A, hard_failure_dict['idx_failed'], axis=1),
                                hard_failure_dict['idx_failed'], axis=0
                              )
                _beta_sol = scipy.linalg.solve(_A, _rhs)

                idx_sol = 0
                for i in range(_m):
                    if i in hard_failure_dict['idx_failed']:
                        beta_coeff[i, mu] = 0.0
                    else:
                        beta_coeff[i, mu] = _beta_sol[idx_sol]
                        idx_sol += 1

        return beta_coeff
    
    def estimate(self,  measures: np.ndarray,
                        regularization_params: dict = None,
                        hard_failure_dict: dict = None
                        ):
        r"""
        Estimate the state using the GEIM given the measures at the sensor locations.

        Parameters
        ----------
        measures : np.ndarray
            Measures :math:`\{y_m\}_{m=1}^{M_{max}}`, shaped :math:`(M_{max},N_s)`.
        regularization_params : dict, optional (Default = None)
            Dictionary containing the regularization parameters. If `None`, no regularization is applied.
            At the moment, the only supported regularization is Tikhonov from `Introini et al. (2023) <https://doi.org/10.1016/j.cma.2022.115773>`_.
        hard_failure_dict : dict, optional (Default = None)
            Dictionary containing the parameters for the hard failure simulation.
            If provided, the method will handle hard failures by removing the corresponding sensors from the estimation process.

        Returns
        -------
        estimation : FunctionsList
            An instance of `FunctionsList` containing the estimated state.
        """

        assert measures.shape[0] <= len(self.sensors), f"The number of measures {measures.shape[0]} must be less than or equal to the number of magic sensors {len(self.sensors)}."

        if self.matrix_B is None:
            self.compute_B_matrix()

        if regularization_params is not None and hard_failure_dict is None:
            if regularization_params['type'] == 'tikhonov':
                beta_coeff = self._solve_trgeim_linear_system(measures, regularization_params)
            else:
                raise ValueError(f"Regularization type {regularization_params['type']} not recognized: available types: 'tikhonov'.")
        
        elif hard_failure_dict is not None:
            
            assert regularization_params is not None, "Regularization parameters must be provided for hard failure handling."
            assert regularization_params['type'] == 'tikhonov', "Only Tikhonov regularization is supported for hard failure handling."
            
            beta_coeff = self._solve_hardfailure_trgeim_linear_system(measures, regularization_params, hard_failure_dict)
        
        else:
            beta_coeff = self._solve_geim_linear_system(measures)
        
        estimation = self._reconstruct(beta_coeff)

        return estimation

class FailingPBDW(PBDW):
    """
    Class for the online phase of PBDW with failing sensors.
    Inherits from PBDW class.

    Parameters
    ----------
    grid : pv.UnstructuredGrid
        The computational grid.
    gdim : int, optional
        Geometrical dimension of the problem (default is 3).
    varname : str, optional
        Name of the variable in the grid (default is 'u').
    """

    def __init__(self, grid: pv.UnstructuredGrid, gdim: int = 3, varname: str = 'u'):
        super().__init__(grid, gdim=gdim, varname=varname)
    
    def get_measurements(self, snaps: FunctionsList | np.ndarray, M: int = None,
                         noise_std: float = 0.0, drift_dict: dict = None):
        r"""
        This method extracts the measures from the input functions at the magic sensors locations. It allows also to add Gaussian noise to the measurements.

        Moreover, the drift failure of the sensor can be simulated by providing a drift_dict with the following keys:
        
        - 'kappa' : float
            Shift from the average value of the sensor.
        - 'rho' : float, optional 
            High frequency oscillation amplitude.
        - 'idx_failed' : list of int, optional
            List of indices of the failed sensors.
        - 'mu_failure' : int, optional (Default = 0)
            Starting index of the failure in the snapshots.

        Parameters
        ----------
        snaps : FunctionsList or np.ndarray
            Function object to extract the measures from.
        M : int, optional (Default = None)
            Number of sensors to use for the extraction. If `None`, all available magic sensors are used.
        noise_std : float, optional (Default = 0.0)
            Standard deviation of the Gaussian noise to be added to the measurements. Default is 0.
        drift_dict : dict, optional (Default = None)
            Dictionary containing the parameters for the drift failure simulation.
            
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

        # Handle sensor drift failure
        if drift_dict is not None:
            assert 'kappa' in drift_dict, "'drift_dict' must contain the key 'kappa'."
            kappa = drift_dict['kappa']

            assert 'rho' in drift_dict, "'drift_dict' must contain the key 'rho'."
            rho = drift_dict['rho']

            assert 'idx_failed' in drift_dict, "'drift_dict' must contain the key 'idx_failed'."
            idx_failed = drift_dict['idx_failed']
            assert min(idx_failed) >= 0 and max(idx_failed) < measures.shape[0], f"'idx_failed' must contain indices between 0 and {measures.shape[0]-1}."

            if 'mu_failure' in drift_dict:
                mu_failure = drift_dict['mu_failure']
            else:
                mu_failure = 0

            drift = np.random.normal(kappa, rho, size=measures[idx_failed, mu_failure:].shape)

            # Apply the drift to the failed sensors
            measures[idx_failed, mu_failure:] += drift

        return measures
    
    def _solve_hardfailure_pbdw_linear_system(self, measures: np.ndarray, xi: float = 0.0,
                                                hard_failure_dict: dict = None):
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

        This method is able to handle hard failures by removing the failed sensors from the linear system after the failure occurs.

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
        _coeffs = np.zeros((_A.shape[0] + _Z.shape[0], measures.shape[1]))

        for mu in range(measures.shape[1]):

            if mu < hard_failure_dict['mu_failure']:
                _coeffs[:, mu] = scipy.linalg.solve(sys_matrix, rhs[:, mu])
            else:
                _rhs = np.delete(rhs[:, mu], hard_failure_dict['idx_failed'], axis=0)
                _sys_matrix = np.delete(
                                        np.delete(sys_matrix, hard_failure_dict['idx_failed'], axis=1),
                                        hard_failure_dict['idx_failed'], axis=0
                                        )
                _coeff_sol = scipy.linalg.solve(_sys_matrix, _rhs)

                idx_sol = 0
                for i in range(_A.shape[0] + _Z.shape[0]):
                    if i in hard_failure_dict['idx_failed']:
                        _coeffs[i, mu] = 0.0
                    else:
                        _coeffs[i, mu] = _coeff_sol[idx_sol]
                        idx_sol += 1

        theta_coeff = _coeffs[:_m]
        alpha_coeff = _coeffs[_m:]

        return alpha_coeff, theta_coeff
    
    def estimate(self,  measures: np.ndarray, xi: float = 0.0,
                        hard_failure_dict: dict = None):
        r"""
        Estimate the state using the PBDW given the measures at the sensor locations.

        Parameters
        ----------
        measures : np.ndarray
            Measures :math:`\{y_m\}_{m=1}^{M}`, shaped :math:`(M, N_s)`.
        xi : float, optional (Default = 0.0)
            Regularization parameter for the regularization. Default is 0.0
        hard_failure_dict : dict, optional (Default = None)
            Dictionary containing the parameters for the hard failure simulation.
            If provided, the method will handle hard failures by removing the corresponding sensors from the estimation process.-

        Returns
        -------
        estimation : FunctionsList
            An instance of `FunctionsList` containing the estimated state.
        """

        if self.matrix_A is None or self.matrix_K is None or self.matrix_Z is None:
            self.compute_matrices()

        if hard_failure_dict is not None:
            alpha_coeffs, theta_coeffs = self._solve_hardfailure_pbdw_linear_system(measures, xi=xi, hard_failure_dict=hard_failure_dict)
        else:
            alpha_coeffs, theta_coeffs = self._solve_pbdw_linear_system(measures, xi=xi)

        estimation = self._reconstruct(alpha_coeffs, theta_coeffs)

        return estimation
