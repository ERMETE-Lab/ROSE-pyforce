# Online Phase: Generalised Empirical Interpolation Method
# Author: Stefano Riva, PhD Student, NRG, Politecnico di Milano
# Latest Code Update: 16 September 2024
# Latest Doc  Update: 16 September 2024

import numpy as np
import scipy.linalg as la
from collections import namedtuple

from dolfinx.fem import FunctionSpace, Function
from pyforce.tools.backends import norms, LoopProgress
from pyforce.tools.functions_list import FunctionsList
from pyforce.tools.timer import Timer

# GEIM online (synthetic and real measures)
class GEIM():
    r"""
    This class can be used to perform the online phase of the GEIM algorihtm for synthetic and real measures :math:`\mathbf{y}\in\mathbb{R}^M` either obtained as evaluations of the magic sensors on the snapshot :math:`u(\mathbf{x};\,\boldsymbol{\mu})` as
    
    .. math::
        y_m = v_m(u)+\varepsilon_m \qquad \qquad m = 1, \dots, M

    given :math:`\varepsilon_m` random noise (either present or not), or by real experimental data on the physical system.

    Parameters
    ----------
    magic_fun : FunctionsList
        List of magic functions computed during the offline phase.
    magic_sen : FunctionsList
        List of magic sensors computed during the offline phase.
    name : str
        Name of the snapshots (e.g., temperature T)
    """
    def __init__(self, magic_fun: FunctionsList, magic_sen: FunctionsList, name: str) -> None:
        
        self.mf = magic_fun
        self.ms = magic_sen
        
        self.V = magic_fun.fun_space
        
        self.name = name

        # Defining the norm class to make scalar products and norms
        self.norms = norms(self.V)

        # Computing the matrix B
        assert (len(self.ms) == len(self.mf)), "Number of magic sensors must be equal to number of magic functions"
        self.Mmax = len(self.ms)
        self.B = np.zeros((self.Mmax, self.Mmax))
    
        for nn in range(self.Mmax):
            for mm in range(self.Mmax):
                self.B[nn, mm] = self.norms.L2innerProd(self.ms(nn), self.mf(mm))

    def synt_test_error(self, snaps: FunctionsList, M = None, noise_value = None, 
                        verbose = False) -> namedtuple:
        r"""
        The absolute and relative error on the test set is computed, by solving the GEIM linear system
        
        .. math::
            \mathbb{B}\boldsymbol{\beta} = \mathbf{y}
        
        Parameters
        ----------
        snaps : FunctionsList
            List of functions belonging to the test set to reconstruct
        M : int, optional (default = None)
            Maximum number of magic functions to use (if None is set to the number of magic functions/sensors)
        noise_value : float, optional (default = None)
            Standard deviation of the noise, modelled as a normal :math:`\mathcal{N}(0, \sigma^2)`
        verbose : boolean, optional (default = False)
            If true, output is printed.

        Returns
        ----------
        mean_abs_err : np.ndarray
            Average absolute error measured in :math:`L^2`.
        mean_rel_err : np.ndarray
            Average relative error measured in :math:`L^2`.
        computational_time : dict
            Dictionary with the CPU time of the most relevant operations during the online phase.
            
        """
        # Check on the input M, maximum number of sensors to use
        if M is None:
            M = self.Mmax
        elif M > self.Mmax:
            print('The maximum number of measures must not be higher than '+str(self.Mmax)+' --> set equal to '+str(self.Mmax))
            M = self.Mmax

        # Computing the test error on a given set of snapshots
        Ns = len(snaps)
        abs_err = np.zeros((Ns, M))
        rel_err = np.zeros_like(abs_err)

        if verbose:
            progressBar = LoopProgress(msg = "Computing GEIM test error (synthetic) - " + self.name, final = Ns )
    
        # Variables to store the computational times
        computational_time = dict()
        computational_time['Measure']      = np.zeros((Ns, M))
        computational_time['LinearSystem'] = np.zeros((Ns, M))
        computational_time['Errors']       = np.zeros((Ns, M))
         
        timing = Timer() 
        
        resid = Function(snaps.fun_space).copy()
        for mu in range(Ns):
            
            timing.start()
            norma_snap = self.norms.L2norm(snaps(mu))
            computational_time['Errors'][mu, :] = timing.stop()

            for mm in range(M):

                # Generating the rhs
                timing.start()
                if mm == 0:
                    y_clean = np.array([self.norms.L2innerProd(snaps(mu), self.ms(mm))])
                else:
                    y_clean = np.hstack([y_clean, np.array([self.norms.L2innerProd(snaps(mu), self.ms(mm))])])
                
                # Adding random noise (synthetic)
                if noise_value is not None:
                    y = y_clean + np.random.normal(0, noise_value, len(y_clean))
                else:
                    y = y_clean
                computational_time['Measure'][mu, mm] = timing.stop()

                # Solving the linear system
                timing.start()
                coeff = la.solve(self.B[:mm+1, :mm+1], y, lower = True)
                computational_time['LinearSystem'][mu, mm] = timing.stop()

                # Compute the error
                timing.start()
                resid.x.array[:] = snaps(mu) - self.mf.lin_combine(coeff)
                abs_err[mu, mm] = self.norms.L2norm(resid)
                rel_err[mu, mm] = abs_err[mu, mm] / norma_snap
                computational_time['Errors'][mu, mm] += timing.stop()

                
            if verbose:
                progressBar.update(1, percentage = False)

        Results = namedtuple('Results', ['mean_abs_err', 'mean_rel_err', 'computational_time'])
        synt_res = Results(mean_abs_err = abs_err.mean(axis = 0), mean_rel_err = rel_err.mean(axis = 0), computational_time = computational_time)

        return synt_res
    
    def reconstruct(self, snap: np.ndarray, M, noise_value = None):
        r"""
        The interpolant for `snap` :math:`u` input is computed, by solving the GEIM linear system
        
        .. math::
            \mathbb{B}\boldsymbol{\beta} = \mathbf{y}
        
        then the inteprolant and residual are computed and returned
        
        .. math::
            \mathcal{I}_M[u] = \sum_{m=1}^M \beta_m[u] \cdot q_m\qquad\qquad
            r_M = \left| u - \mathcal{I}_M[u]\right|
        
        Parameters
        ----------
        snap : Function as np.ndarray
            Snap to reconstruct, if a function is provided, the variable is reshaped.
        M : int
            Number of sensor to use
        noise_value : float, optional (default = None)
            Standard deviation of the noise, modelled as a normal :math:`\mathcal{N}(0, \sigma^2)`
        
        Returns
        ----------
        interp : np.ndarray
            Interpolant Field :math:`\mathcal{I}_M[u]` of GEIM
        resid : np.ndarray 
            Residual Field :math:`r_M[u]`
        computational_time : dict
            Dictionary with the CPU time of the most relevant operations during the online phase.
            
        """
        if M > self.Mmax:
            print('The maximum number of measures must not be higher than '+str(self.Mmax)+' --> set equal to '+str(self.Mmax))
            M = self.Mmax
        
        if isinstance(snap, Function):
            snap = snap.x.array[:]
        
        # Variables to store the computational times
        computational_time = dict()
        timing = Timer() 
        
        # Compute measurement vector
        timing.start()
        y_clean = self.compute_measure(snap, M)
        
        # Adding random noise (synthetic)
        if noise_value is not None:
            y = y_clean + np.random.normal(0, noise_value, len(y_clean))
        else:
            y = y_clean
        computational_time['Measure'] = timing.stop()
        
        # Solving the linear system
        timing.start()
        coeff = la.solve(self.B[:M, :M], y, lower = True)
        computational_time['LinearSystem'] = timing.stop()

        # Compute the interpolant and residual
        timing.start()
        interp = self.mf.lin_combine(coeff)
        computational_time['Reconstruction'] = timing.stop()
        
        resid = np.abs(snap - interp)
        
        return interp, resid, computational_time
    

    def compute_measure(self, snap: Function, M = None) -> np.ndarray:
        r"""
        Computes the measurement vector :math:`\mathbf{y}\in\mathbb{R}^M` from the `snap` :math:`u` input, using the magic sensors stored.
        
        .. math::
            y_m = v_m(u) \qquad \qquad m = 1, \dots, M
        
        If the dimension :math:`M` is not given, the whole set of magic sensors is used.
        
        Parameters
        ----------
        snap : Function
            Function from which measurements are to be extracted
        M : int, optional (default = None)
            Maximum number of sensor to use (if None is set to the number of magic functions/sensors)

        Returns
        ----------
        measure : np.ndarray
            Measurement vector :math:\mathbf{y}\in\mathbb{R}^M`
        """
        # Check on the input M, maximum number of sensors to use
        if M is None:
            M = self.Mmax
        elif M > self.Mmax:
            print('The maximum number of measures must not be higher than '+str(self.Mmax)+' --> set equal to '+str(self.Mmax))
            M = self.Mmax

        measure = np.zeros((M,))
        for mm in range(M):
            measure[mm] = self.norms.L2innerProd(snap, self.ms(mm))

        return measure
    
    def real_reconstruct(self, measure: np.ndarray):
        r"""
        The interpolant given the `measure` vector :math:`\mathbf{y}` input is computed, by solving the GEIM linear system
        
        .. math::
            \mathbb{B}\boldsymbol{\beta} = \mathbf{y}
        
        then the interpolant is computed and returned
        
        .. math::
            \mathcal{I}_M(\mathbf{x}) = \sum_{m=1}^M \beta_m[u] \cdot q_m(\mathbf{x})
        
        Parameters
        ----------
        measure : np.ndarray
            Measurement vector, shaped as :math:`M \times N_s`, given :math:`M` the number of sensors used and :math:`N_s` the number of parametric realisation.
        
        Returns
        ----------
        interp : np.ndarray
            Interpolant Field :math:`\mathcal{I}_M` of GEIM
        computational_time : dict
            Dictionary with the CPU time of the most relevant operations during the online phase.
            
        """
        
        M, Ns = measure.shape
        
        if M > self.Mmax:
            print('The maximum number of measures must not be higher than '+str(self.Mmax)+' --> set equal to '+str(self.Mmax))
            M = self.Mmax
        
        # Variables to store the computational times
        computational_time = dict()
        timing = Timer() 
        
        interps = FunctionsList(self.V)
        
        for mu in range(Ns):
            
            y = measure[:, mu]
            
            # Solving the linear system
            timing.start()
            coeff = la.solve(self.B[:M, :M], y, lower = True)
            computational_time['LinearSystem'] = timing.stop()

            # Compute the interpolant and residual
            timing.start()
            interps.append(self.mf.lin_combine(coeff))
            
            computational_time['Reconstruction'] = timing.stop()
        
        return interps, computational_time