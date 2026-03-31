# Indirect Reconstruction algorithm tools
# Author: Stefano Riva, PhD Student, NRG, Politecnico di Milano
# Latest Code Update: 08 October 2025
# Latest Doc  Update: 08 October 2025

import numpy as np
from scipy.optimize import brute, shgo, differential_evolution
from collections import namedtuple
import pyvista as pv

from ..tools.backends import IntegralCalculator, LoopProgress, Timer
from ..tools.functions_list import FunctionsList
from .online_base import OnlineDDROM, SurrogateModelWrapper, OnlineSensors

# Define Objective Function
def objective(mu: np.ndarray, matrix_B: np.ndarray, target_measure: np.ndarray, maps: SurrogateModelWrapper):
    r"""
    This function evaluates the standard loss function :math:`\mathcal{L}_{PE}` to be minimized during the Parameter Estimation phase
    
    .. math::
      \mathcal{L}_{PE}(\boldsymbol{\mu}) = \|B\boldsymbol{\beta}(\boldsymbol{\mu}) - \mathbf{y}\|_2^2 \qquad\qquad
      \text{ given }\mathcal{F}_m(\boldsymbol{\mu}) = \beta_m(\boldsymbol{\mu})
    
    given the maps :math:`\mathcal{F}_m:\boldsymbol{\mu} \rightarrow \beta_m` (:math:`m = 1, \dots, M`).

    Parameters
    ----------
    mu : np.ndarray
        The input parameter vector :math:`\boldsymbol{\mu} \in \mathbb{R}^{p}`.
    matrix_B : np.ndarray
        The (G)EIM matrix :math:`B \in \mathbb{R}^{M \times M}`.
    target_measure : np.ndarray
        The target measurement vector :math:`\mathbf{y} \in \mathbb{R}^{M}`.
    maps : SurrogateModelWrapper
        The surrogate model wrapper containing the maps :math:`\mathcal{F}_m` from parameters to (G)EIM coefficients.
    """

    M = len(target_measure)
    assert matrix_B.shape == (M, M), f"Matrix B shape {matrix_B.shape} must be (M, M) with M = {M}."

    beta = maps.predict(mu)[:M]  # shape (M,)
    return (np.linalg.norm( np.dot(matrix_B, beta) - target_measure))**2

# Parameter Estimation using scipy optimization tools
class ParameterEstimation():
    def __init__(self, grid: pv.UnstructuredGrid, maps: SurrogateModelWrapper, bnds: list, gdim: int = 3):
        r"""
        A class to perform Parameter Estimation (PE) exploiting GEIM magic functions and sensors and numerical optimisation starting from a set of measurements :math:`\mathbf{y}\in\mathbb{R}^M`
        
        .. math::
            \hat{\boldsymbol{\mu}} = \text{arg}\,\min\limits_{\boldsymbol{\mu}\in\mathcal{D}} \mathcal{L}_{PE}(\boldsymbol{\mu})

        Parameters
        ----------
        grid : pv.UnstructuredGrid
            The computational grid where the solution is defined.
        maps : SurrogateModelWrapper
            A surrogate model wrapper containing the maps :math:`\mathcal{F}_m` from parameters to (G)EIM coefficients.
        bnds : list
            A list of tuples defining the bounds for each parameter in the optimisation problem.
        gdim : int, optional (Default = 3)
            The geometric dimension of the problem (2D or 3D).

        """

        self.grid = grid
        self.gdim = gdim

        # Store surrogate model -> parameters to (G)EIM coefficients
        self.maps = maps

        # Store bounds
        self.p = len(bnds) # Number of parameters
        self.bnds = bnds

        # Initialize (G)EIM matrix
        self.matrix_B = None

    def set_magic_functions(self, basis: FunctionsList = None, path_folder: str = None, **kwargs):
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
        self.Mmax = M

        assert M == len(self._basis), f"Number of magic sensors {M} must be equal to the number of magic functions {len(self._basis)}."

        self.matrix_B = self.sensors.action(self._basis)

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
    
    def optimise(self, measure: np.ndarray, use_brute = True, grid_elem = 10):
        """
        This auxiliary function performs the optimisation process given an input collection of measurements.
        
        Two options are available:
        
            - **brute force method** for finding the first guess of the parameter estimation + Least squares
            - **differential evolution method** for finding the first guess of the parameter estimation + Least squares
            
        Parameters
        ----------
        measure : np.ndarray
            Measurement vector of `M` elements, array-like.
        use_brute : bool, optional (Default = True)
            brute force method for finding the first guess of the parameter estimation
        grid_elem : int, optional (Default = 10)
            Number of elements in the grid for the brute force method
        
        Returns
        ----------
        solution : np.ndarray
            Solution of the optimisation (after least squares)
        guess : np.ndarray
            Solution of the optimisation (before least squares)
        """
        M = len(measure)
        assert(M <= self.Mmax)

        if use_brute:
            # defining bounds for optimisation
            opt_bnds = list()
            for param in range(self.p):
                delta_mu = (self.bnds[param][1] - self.bnds[param][0]) / grid_elem
                opt_bnds.append(slice(self.bnds[param][0], self.bnds[param][1], delta_mu))

            sol = brute(objective, opt_bnds, args=(self.matrix_B[:M, :M], measure, self.maps))
        else:
            sol = differential_evolution(objective, self.bnds, args=(self.matrix_B[:M, :M], measure, self.maps)).x

        return sol

    def compute_errors(self, test_param: np.ndarray, test_snaps: FunctionsList | np.ndarray, M: int, 
                        noise_std = 0.0, use_brute = True, grid_elem = 10,
                        verbose = False):
        r"""
        The absolute and relative error of the PE phase, using different measurements coming from the test set, are computed.
        
        Parameters
        ----------
        test_param : np.ndarray
            The test parameter set :math:`\{\boldsymbol{\mu}^{(n)}\}_{n=1}^{N_{test}}`, shaped :math:`(N_{test}, p)`.
        test_snaps : FunctionsList or np.ndarray
            The test snapshots set :math:`\{u^{(n)}\}_{n=1}^{N_{test}}`.
        M : int
            Number of sensors to use for the generation of the measures.
        noise_std : float, optional (Default = 0.0)
            Standard deviation of the Gaussian noise to be added to the measurements. 
        use_brute : bool, optional (Default = True)
            If `True`, the brute force method is used for the initial guess in the optimisation. If `False`, differential evolution is used.
        grid_elem : int, optional (Default = 10)
            Number of elements in the grid for the brute force method.
        verbose : bool, optional (Default = False)
            If `True`, progress information is printed to the console.
        
        Returns
        -------
        mean_abs_err : np.ndarray
            Average absolute error of the parameter estimation
        mean_rel_err : np.ndarray
            Average relative error of the parameter estimation
        computational_time : dict
            Dictionary with the CPU time of the most relevant operations during the online phase.
        mu_PE : list
            List containing the estimated parameters after least squares at varying number of measurements
        mu_PE_guess : list
            List containing the estimated parameters before least squares at varying number of measurements
        
        """

        Ns = len(test_snaps)

        if verbose:
            bar = LoopProgress(msg = f"Computing PE errors", final = Ns)

        mu_PE       = np.zeros((Ns, M, self.p))

        # Variables to store computational time
        computational_time = dict()
        computational_time['Measures']     = np.zeros((Ns, M))
        computational_time['Optimisation'] = np.zeros((Ns, M))

        timer = Timer()

        # Loop over test parameters
        for idx_mu in range(Ns):
            for mm in range(M):
                
                # Get measurements
                timer.start()
                measure = self.get_measurements(
                    test_snaps(idx_mu), M=mm+1, noise_std=noise_std
                )
                computational_time['Measures'][idx_mu, mm] = timer.stop()

                # Perform optimisation
                timer.start()
                mu_PE[idx_mu, mm, :] = self.optimise(
                    measure=measure, use_brute=use_brute, grid_elem=grid_elem
                )
                computational_time['Optimisation'][idx_mu, mm] = timer.stop()

            if verbose:
                bar.update(1)

        # Compute errors
        abs_err = np.abs(mu_PE - test_param[:, np.newaxis, :]).mean(axis=0) # shape (M, p)
        rel_err = (np.abs(mu_PE - test_param[:, np.newaxis, :]) / (np.abs(test_param[:, np.newaxis, :]) + 1e-12)).mean(axis=0) # shape (M, p)

        # Store results
        Results = namedtuple('Results', ['mean_abs_err', 'mean_rel_err', 'computational_time', 'mu_PE'])

        return Results(mean_abs_err=abs_err, mean_rel_err=rel_err, computational_time=computational_time, mu_PE=mu_PE)