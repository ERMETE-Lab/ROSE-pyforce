# Synthetic Online Phase: Generalised Empirical Interpolation Method
# Author: Stefano Riva, PhD Student, NRG, Politecnico di Milano
# Latest Code Update: 25 June 2023
# Latest Doc  Update: 18 July 2023

import numpy as np
import scipy.linalg as la

from dolfinx.fem import FunctionSpace, Function
from pyforce.tools.backends import norms, LoopProgress
from pyforce.tools.functions_list import FunctionsList
from pyforce.tools.timer import Timer

class POD_project():
    r"""
    A class to perform the online phase of the POD with projection from true field, in which the modal coefficients are found from the projection of the snapshot onto the reduced space.
    
    Parameters
    ----------
    modes : FunctionsList
        List of POD modes computed during the offline phase.
    name : str
        Name of the snapshots (e.g., temperature T)
    """
    def __init__(self, modes: FunctionsList, name: str) -> None:
        
        self.V = modes.fun_space
        self.PODmodes = FunctionsList(self.V)
        self.PODmodes._list = modes.copy()
        
        # Store the variable name
        self.name = name

        # Defining the norm class to make scalar products and norms
        self.norm = norms(self.V)
    
    def synt_test_error(self, test_snap: FunctionsList, maxBasis: int, 
                              verbose = False):
        r"""
        The maximum absolute :math:`E_N` and relative :math:`\varepsilon_N` error on the test set is computed, by projecting it onto the reduced space in :math:`L^2`

        .. math::
            E_N = \max\limits_{\boldsymbol{\mu}\in\Xi_{\text{test}}} \left\| u(\mathbf{x};\,\boldsymbol{\mu}) -  \sum_{n=1}^N \alpha_n(\boldsymbol{\mu})\cdot \psi_n(\mathbf{x})\right\|_{L^2}
        .. math::
            \varepsilon_N = \max\limits_{\boldsymbol{\mu}\in\Xi_{\text{test}}} \frac{\left\| u(\mathbf{x};\,\boldsymbol{\mu}) -  \sum_{n=1}^N \alpha_n(\boldsymbol{\mu})\cdot \psi_n(\mathbf{x})\right\|_{L^2}}{\left\| u(\mathbf{x};\,\boldsymbol{\mu})\right\|_{L^2}}
    
        The coefficients of the POD basis are obtained by interpolating using the maps.
        
        Parameters
        ----------
        test_snap : FunctionsList
            List of snapshots onto which the test error of the POD basis is performed.
        maxBasis : int
            Integer input indicating the maximum number of modes to use.
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
        
        absErr = np.zeros((Ns_test, maxBasis))
        relErr = np.zeros_like(absErr)

        if verbose:
            progressBar = LoopProgress(msg = "Computing POD test error (projection) - " + self.name, final = Ns_test)

        # Variables to store the computational times
        computational_time = dict()
        computational_time['CoeffEstimation'] = np.zeros((Ns_test, maxBasis))
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
                coeff[nn] = self.norm.L2innerProd(self.PODmodes(nn), test_snap(mu))
                computational_time['CoeffEstimation'][mu, nn] = timing.stop()
                
                # building residual field
                timing.start()
                resid.x.array[:] = test_snap(mu) - self.PODmodes.lin_combine(coeff[:nn+1])
                absErr[mu, nn] = self.norm.L2norm(resid)
                relErr[mu, nn] = absErr[mu, nn] / norma_snap
                computational_time['Errors'][mu, nn] += timing.stop()

            if verbose:
                progressBar.update(1, percentage = False)

        return absErr.mean(axis = 0), relErr.mean(axis = 0), computational_time
        
    def project(self, snap: Function, maxBasis: int):
        r"""
        Project `snap` onto the reduced space of dimension `maxBasis`, by computing the modal coefficients :math:`\{\alpha_i\}`
        
        .. math::
            \alpha_i = (u, \psi_i)_{L^2} = \int_\Omega u\cdot \psi_i\,d\Omega
        
        Parameters
        ----------
        snap : Function
            Snap to project.
        maxBasis : int
            Integer input indicating the maximum number of modes to use.
        
        Returns
        -------
        coeff : np.ndarray
            Modal coefficient obtain by projection with POD modes.
        """
    
        coeff = np.zeros((maxBasis,))
        for nn in range(maxBasis):
          coeff[nn] = self.norm.L2innerProd(snap, self.PODmodes(nn)) 
            
        return coeff
        
    def reconstruct(self, snap: np.ndarray, maxBasis: int):
        r"""
        The coefficients of the POD basis are obtained by projection, the `snap` is approximated using linear combination of the POD modes.
        
        Parameters
        ----------
        snap : Function as np.ndarray
            Snap to reconstruct, if a function is provided, the variable is reshaped.
        maxBasis : int
            Integer input indicating the maximum number of modes to use.
        
        Returns
        -------
        reconstruction : np.ndarray
            Reconstructed field using `maxBasis` POD modes.
        resid : np.ndarray
            Residual field using `maxBasis` POD modes.
        computational_time : dict
            Dictionary with the CPU time of the most relevant operations during the online phase.
        """
        
        # Variables to store the computational times
        computational_time = dict()
        timing = Timer() 
        
        if isinstance(snap, Function):
            snap = snap.x.array[:]
            
        # Estimate the coefficients
        timing.start()
        coeff = self.project(snap, maxBasis)
        computational_time['CoeffEstimation'] = timing.stop()
            
        
        # Compute the interpolant and residual
        timing.start()
        recon = self.PODmodes.lin_combine(coeff)
        computational_time['Reconstruction'] = timing.stop()
        
        resid = np.abs(snap - recon)

        return recon, resid, computational_time