# Synthetic Online Phase: Proper Orthogonal Decomposition with Interpolation
# Author: Stefano Riva, PhD Student, NRG, Politecnico di Milano
# Latest Code Update: 06 December 2023
# Latest Doc  Update: 06 December 2023

import numpy as np
import scipy.linalg as la

from dolfinx.fem import FunctionSpace, Function
from pyforce.tools.backends import norms, LoopProgress
from pyforce.tools.functions_list import FunctionsList
from pyforce.tools.timer import Timer

class PODI():
    r"""
    A class to perform the online phase of the POD with Inteprolation (PODI) algorithm, in which the modal coefficients are found from suitable maps :math:`\mathcal{F}_n:\boldsymbol{\mu} \rightarrow \alpha_n` (:math:`n = 1, \dots, N`).

    Parameters
    ----------
    modes : FunctionsList
        List of POD modes computed during the offline phase.
    maps : list
        List of maps for the POD modal coefficients, they must be callable.
    name : str
        Name of the snapshots (e.g., temperature T)
    """
    def __init__(self, modes: FunctionsList, maps: list, name: str) -> None:
        
        self.V = modes.fun_space
        self.PODmodes = FunctionsList(self.V)
        self.PODmodes._list = modes.copy()

        # Store the coefficients maps of the POD basis
        self.maps = maps
        
        # Store the variable name
        self.name = name

        # Defining the norm class to make scalar products and norms
        self.norm = norms(self.V)

    def synt_test_error(self, test_snap: FunctionsList, mu_estimated: np.ndarray, maxBasis: int, 
                              alpha_coeffs : np.ndarray = None, verbose = False):
        r"""
        The maximum absolute :math:`E_N` and relative :math:`\varepsilon_N` error on the test set is computed, by projecting it onto the reduced space in :math:`L^2` with the coefficients estimated through callable maps or given as input.

        .. math::
            E_N = \max\limits_{\boldsymbol{\mu}\in\Xi_{\text{test}}} \left\| u(\mathbf{x};\,\boldsymbol{\mu}) -  \sum_{n=1}^N \alpha_n(\boldsymbol{\mu})\cdot \psi_n(\mathbf{x})\right\|_{L^2}
        .. math::
            \varepsilon_N = \max\limits_{\boldsymbol{\mu}\in\Xi_{\text{test}}} \frac{\left\| u(\mathbf{x};\,\boldsymbol{\mu}) -  \sum_{n=1}^N \alpha_n(\boldsymbol{\mu})\cdot \psi_n(\mathbf{x})\right\|_{L^2}}{\left\| u(\mathbf{x};\,\boldsymbol{\mu})\right\|_{L^2}}
    
        The coefficients of the POD basis are obtained by interpolating using the maps.
        
        Parameters
        ----------
        test_snap : FunctionsList
            List of snapshots onto which the test error of the POD basis is performed.
        mu_estimated : np.ndarray
            Arrays with the estimated parameters from the Parameter Estimation phase, it must have dimension `[Ns, p]` in which `Ns` is the number of test snapshots and `p` the number of parameters.
        maxBasis : int
            Integer input indicating the maximum number of modes to use.
        alpha_coeff : np.ndarray (optional, Default: None)
            Matrix with the estimated coefficients :math:`\alpha_n`, they will be used if the input `alpha_coeffs` is not `None`.
        
        verbose : boolean, optional (Default = False) 
            If `True`, print of the progress is enabled.

        Returns
        -------
        ave_absErr : np.ndarray
            Average absolute errors as a function of the dimension of the reduced space.
        ave_relErr : np.ndarray
            Average absolute errors as a function of the dimension of the reduced space.
        computational_time : dict
            Dictionary with the CPU time of the most relevant operations during the online phase.
        """
        
        Ns_test = len(test_snap)
        
        if mu_estimated is not None:
            assert (mu_estimated.shape[0] == Ns_test)
            n_feature = mu_estimated.shape[1]
        
        absErr = np.zeros((Ns_test, maxBasis))
        relErr = np.zeros_like(absErr)

        if verbose:
            progressBar = LoopProgress(msg = "Computing POD test error (interpolation) - " + self.name, final = Ns_test)
        
        # Variables to store the computational times
        computational_time = dict()
        computational_time['CoeffEstimation'] = np.zeros((Ns_test, maxBasis))
        computational_time['LinearSystem']    = np.zeros((Ns_test, maxBasis))
        computational_time['Errors']          = np.zeros((Ns_test, maxBasis))
         
        timing = Timer() 
        
        resid = Function(self.V).copy()
        for mu in range(Ns_test):
            
            timing.start()
            norma_snap = self.norm.L2norm(test_snap(mu))
            computational_time['Errors'][mu, :] = timing.stop()
            
            # Coefficient Estimation
            coeff = np.zeros((maxBasis,))
            for nn in range(maxBasis):
                
                timing.start()
                if alpha_coeffs is not None:
                    coeff[nn] = alpha_coeffs[mu, nn]
                else:
                    coeff[nn] = self.maps[nn](mu_estimated[mu].reshape(-1, n_feature))
                computational_time['CoeffEstimation'][mu, nn] = timing.stop()
                
            # for nn in range(maxBasis):
                # building residual field and computing the errors
                timing.start()
                resid.x.array[:] = test_snap(mu) - self.PODmodes.lin_combine(coeff[:nn+1])
                absErr[mu, nn] = self.norm.L2norm(resid)
                relErr[mu, nn] = absErr[mu, nn] / norma_snap
                computational_time['Errors'][mu, nn] += timing.stop()

            if verbose:
                progressBar.update(1, percentage = False)

        return absErr.mean(axis = 0), relErr.mean(axis = 0), computational_time
    
    def reconstruct(self, snap: np.ndarray, mu_estimated: np.ndarray, maxBasis: int):
        r"""
        After the coefficients of the POD basis are obtained by interpolating using the maps, the `snap` is approximated using linear combination of the POD modes.
        
        Parameters
        ----------
        snap : Function as np.ndarray
            Snap to reconstruct, if a function is provided, the variable is reshaped.
        mu_estimated : np.ndarray
            Arrays with the estimated parameters from the Parameter Estimation phase, it must have dimension `[1, p]` in which `p` the number of parameters.
        maxBasis : int
            Integer input indicating the maximum number of modes to use.
        
        Returns
        -------
        reconstruction : np.ndarray
            Reconstructed field using `maxBasis` POD modes.
        resid : np.ndarray
            Residual field using `maxBasis` POD modes.
        """
        
        n_feature = mu_estimated.shape[1]
    
        # Variables to store the computational times
        computational_time = dict()
        timing = Timer() 
        
        # Estimate the coefficients
        timing.start()
        coeff = np.zeros((maxBasis,))
        for nn in range(maxBasis):
            coeff[nn] = self.maps[nn](mu_estimated.reshape(-1, n_feature))
        computational_time['CoeffEstimation'] = timing.stop()
            
        if isinstance(snap, Function):
            snap = snap.x.array[:]
        
        # Compute the interpolant and residual
        timing.start()
        recon = self.PODmodes.lin_combine(coeff)
        computational_time['Reconstruction'] = timing.stop()
        
        resid = np.abs(snap - recon)

        return recon, resid, computational_time