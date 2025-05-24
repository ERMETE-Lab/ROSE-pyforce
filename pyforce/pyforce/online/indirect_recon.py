# Indirect Reconstruction algorithm tools
# Author: Stefano Riva, PhD Student, NRG, Politecnico di Milano
# Latest Code Update: 16 September 2024
# Latest Doc  Update: 16 September 2024

import numpy as np
from scipy.optimize import brute, shgo, differential_evolution, minimize, least_squares, Bounds
from collections import namedtuple

from dolfinx.fem import FunctionSpace, Function
from pyforce.tools.backends import norms, LoopProgress
from pyforce.tools.functions_list import FunctionsList
from pyforce.tools.timer import Timer

# Define Objective Function
def objective(mu: np.ndarray, measure: np.ndarray, B: np.ndarray, maps: list):
    r"""
    This function evaluates the standard loss function :math:`\mathcal{L}_{PE}` to be minimized during the Parameter Estimation phase
    
    .. math::
      \mathcal{L}_{PE}(\boldsymbol{\mu}) = \|B\boldsymbol{\beta}(\boldsymbol{\mu}) - \mathbf{y}\|_2^2 \qquad\qquad
      \text{ given }\mathcal{F}_m(\boldsymbol{\mu}) = \beta_m(\boldsymbol{\mu})
    
    given the maps :math:`\mathcal{F}_m:\boldsymbol{\mu} \rightarrow \beta_m` (:math:`m = 1, \dots, M`).

    Parameters
    ----------
    mu : np.ndarray
        Input parameter, array-like.
    measure : np.ndarray
        Input measurements of `M` elements, array-like.
    B : np.ndarray
        Lower-triangular matrix :math:`\mathbb{B}` of dimension :math:`M\times M`.
    maps : list
        List containing the map :math:`\mathcal{F}_m:\boldsymbol{\mu} \rightarrow \beta_m`.
    """
    M = len(measure)
    beta = np.zeros((M,))
    for mm in range(M):
        beta[mm] = maps[mm](mu)
    
    return (np.linalg.norm( np.dot(B, beta) - measure))**2

# Parameter estimation using optimisation tools
class PE():
    r"""
    A class to perform Parameter Estimation (PE) exploiting GEIM and numerical optimisation starting from a set of measurements :math:`\mathbf{y}\in\mathbb{R}^M`
    
    .. math::
        \hat{\boldsymbol{\mu}} = \text{arg}\,\min\limits_{\boldsymbol{\mu}\in\mathcal{D}} \mathcal{L}_{PE}(\boldsymbol{\mu}) 
    
    Parameters
    ----------
    B : np.ndarray
        Lower-triangular matrix :math:`\mathbb{B}` of dimension :math:`M\times M`, arising from GEIM magic function and sensors.
    coeff_maps : list
        List containing the map :math:`\mathcal{F}_m:\boldsymbol{\mu} \rightarrow \beta_m`.
    bnds : list
        List of tuples, each element contains the mininum and maximum of the components of the parameter.
    """
    def __init__(self, B: np.ndarray, coeff_maps: list, bnds: list) -> None:
        self.B = B
        self.maps = coeff_maps

        # Asserting the coefficients maps and the matrix are compatible
        assert(len(coeff_maps) == self.B.shape[0])
        self.Mmax = len(coeff_maps)

        self.p = len(bnds)
        self.bnds = bnds
    
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

            guess = brute(objective, opt_bnds, args=(measure, self.B[:M, :M], self.maps))
        else:
            guess = differential_evolution(objective, self.bnds, args=(measure, self.B[:M, :M], self.maps)).x
        
        lower_bounds = [bound[0] for bound in self.bnds]  
        upper_bounds = [bound[1] for bound in self.bnds]  
        bounds = Bounds(lower_bounds, upper_bounds)
        
        # Add a check for the guess and clip it to the bounds if necessary
        if np.any(guess < lower_bounds) or np.any(guess > upper_bounds):
            guess = np.clip(guess, lower_bounds, upper_bounds)
        
        # Perform least squares optimisation to refine the guess
        sol = least_squares(objective, x0 = guess, bounds=bounds, args=(measure, self.B[:M, :M], self.maps))        

        if sol.status < 1:
            print('Optimisation failed: error ' +str(sol.status))

        return sol.x, guess

    def synt_test_error(self, test_param: np.ndarray, test_snaps: FunctionsList, GEIM_msen: FunctionsList, Mmax: int, 
                        noise_value = None, use_brute = True, grid_elem = 10,
                        verbose = False):
        r"""
        The absolute and relative error of the PE phase, using different measurements coming from the test set, are computed.
        
        Parameters
        ----------
        test_param : np.ndarray
            `np.ndarray` with shape $(N_s, p)$ given $N_s$ the number of samples and $p$ the dimension of the parameter vector
        test_snaps : FunctionsList
            List of functions belonging to the test set, used to generate the measurements
        GEIM_msen : FunctionsList
            List of sensors to mimic the measurement process
        Mmax : int
            Maximum number of sensor to use
        noise_value : float, optional (Default = None)
            Standard deviation of the noise, modelled as a normal :math:`\mathcal{N}(0, \sigma^2)`.
        use_brute : bool, optional (Default = True)
            brute force method for finding the first guess of the parameter estimation
        grid_elem : int, optional (Default = 10)
            Number of elements in the grid for the brute force method
        verbose : bool, optional (Default = False)
            If true, printing is produced

        Returns
        ----------
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
        normss = norms(test_snaps.fun_space)
        if verbose:
            bar = LoopProgress(msg = 'Solving Parameter Estimation ', final = Ns)
        mu_PE = []
        mu_PE_guess = []

        # Variables to store the computational times
        computational_time = dict()
        computational_time['Measure']      = np.zeros((Ns, Mmax))
        computational_time['Optimisation'] = np.zeros((Ns, Mmax))

        timing = Timer()

        for idx_mu in range(Ns):
            
            y_clean = np.zeros((Mmax,))
            for mm in range(Mmax):
                timing.start()  
                y_clean[mm] = normss.L2innerProd(test_snaps(idx_mu), GEIM_msen(mm))
                computational_time['Measure'][idx_mu, mm] = timing.stop()
                
            # Adding random noise (synthetic)
            timing.start()
            if noise_value is not None:
                measure = y_clean + np.random.normal(0, noise_value, len(y_clean))
            else:
                measure = y_clean
            computational_time['Measure'][idx_mu, :] = timing.stop()
                
            mu_estimated = np.zeros((Mmax, self.p))
            mu_guess     = np.zeros((Mmax, self.p))

            for mm in range(Mmax):
                timing.start()
                mu_estimated[mm, :], mu_guess[mm, :] = self.optimise(measure[:mm+1], use_brute = use_brute, grid_elem=grid_elem)
                computational_time['Optimisation'][idx_mu, mm] = timing.stop()

            mu_PE.append(mu_estimated)
            mu_PE_guess.append(mu_guess)
            if verbose:
                bar.update(1, percentage=False)

        abs_errPE = []
        rel_errPE = []

        for idx_mu in range(len(test_snaps)):
            abs_errPE.append(abs(mu_PE[idx_mu] - test_param[idx_mu]))
            rel_errPE.append(abs_errPE[idx_mu] / test_param[idx_mu])

        mean_abs_err = sum(abs_errPE) / len(test_snaps)
        mean_rel_err = sum(rel_errPE) / len(test_snaps)

        Results = namedtuple('Results', ['mean_abs_err', 'mean_rel_err', 'computational_time', 'mu_PE', 'mu_PE_guess'])
        synt_res = Results(mean_abs_err = mean_abs_err, mean_rel_err = mean_rel_err, computational_time = computational_time,
                           mu_PE = mu_PE, mu_PE_guess = mu_PE_guess)

        return synt_res