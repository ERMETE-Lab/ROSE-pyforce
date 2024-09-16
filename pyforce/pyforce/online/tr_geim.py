# Online Phase: Tikhonov-Regularisation Generalised Empirical Interpolation Method
# Author: Stefano Riva, PhD Student, NRG, Politecnico di Milano
# Latest Code Update: 16 September 2024
# Latest Doc  Update: 16 September 2024

import numpy as np
import numpy.linalg as la
from collections import namedtuple

from dolfinx.fem import FunctionSpace, Function
from pyforce.tools.backends import norms, LoopProgress
from pyforce.tools.functions_list import FunctionsList
from pyforce.tools.timer import Timer

from typing import Tuple

# TR-GEIM online (synthetic and real measures)
class TRGEIM():
    r"""
    This class can be used to perform the online phase of the TR-GEIM algorihtm for synthetic and real measures :math:`\mathbf{y}\in\mathbb{R}^M` obtained as evaluations of the magic sensors on the snapshot :math:`u(\mathbf{x};\,\boldsymbol{\mu})` as
    
    .. math::
        y_m = v_m(u)+\varepsilon_m \qquad \qquad m = 1, \dots, M

    given :math:`\varepsilon_m` random noise, or by real experimental data on the physical system.

    Parameters
    ----------
    magic_fun : FunctionsList
        List of magic functions computed during the offline phase.
    magic_sen : FunctionsList
        List of magic sensors computed during the offline phase.
    mean_beta : np.ndarray
        Mean values :math:`\langle\beta_m\rangle` of the training reduced coefficients
    std_beta : np.ndarray
        Standard deviations :math:`\sigma_{\beta_m}` of the training reduced coefficients
    name : str
        Name of the snapshots (e.g., temperature T)
    """
    def __init__(self, magic_fun: FunctionsList, magic_sen: FunctionsList, mean_beta: np.ndarray, std_beta: np.ndarray, name: str) -> None:
        
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

        # Compute matrix T for the regularisation
        assert (mean_beta.shape[0] == self.Mmax), "The size of the mean values of beta must be equal to the number of magic sensors/functions"
        assert (mean_beta.shape == std_beta.shape), "The size of the mean values of beta must be equal to the size of the standard deviations"

        self.mean_beta = mean_beta
        self.T = np.diag(1 / std_beta)

    def synt_test_error(self, snaps: FunctionsList, noise_value: float, reg_param: float, 
                        M = None, verbose = False) -> namedtuple:
        r"""
        The absolute and relative error on the test set is computed using the reconstruction obtained by solving the TR-GEIM linear system
        
        .. math::
            \left(\mathbb{B}^T\mathbb{B}+\lambda \mathbb{T}^T\mathbb{T}\right)\boldsymbol{\beta} = \mathbb{B}^T\mathbf{y}+\lambda \mathbb{T}^T\mathbb{T} \langle{\boldsymbol{\beta}}\rangle
        
        Parameters
        ----------
        snaps : FunctionsList
            List of functions belonging to the test set to reconstruct
        M : int, optional (default = None)
            Maximum number of magic functions to use (if None is set to the number of magic functions/sensors)
        noise_value : float
            Standard deviation of the noise, modelled as a normal :math:`\mathcal{N}(0, \sigma^2)`
        reg_param : float
            Regularising parameter :math:`\lambda`
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
            progressBar = LoopProgress(msg = "Computing TR-GEIM test error (synthetic) - " + self.name, final = Ns )
    
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
                
                # Adding synthetic noise
                y = y_clean + np.random.normal(0, noise_value, len(y_clean))
                computational_time['Measure'][mu, mm] = timing.stop()
                
                # Creating matrix and vector
                timing.start()
                sys_matrix = (self.B[:mm+1, :mm+1].T @ self.B[:mm+1, :mm+1]) + reg_param * (self.T[:mm+1, :mm+1].T @ self.T[:mm+1, :mm+1])
                rhs = np.dot(self.B[:mm+1, :mm+1].T, y) + reg_param * np.dot((self.T[:mm+1, :mm+1].T @ self.T[:mm+1, :mm+1]), self.mean_beta[:mm+1])

                # Solving the linear system
                coeff = la.solve(sys_matrix, rhs)
                computational_time['LinearSystem'][mu, mm] = timing.stop()

                # Compute errors
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

    def compute_measure(self, snap: Function, noise_value: float, M = None) -> np.ndarray:
        r"""
        Computes the measurement vector from the `snap` input, using the magic sensors stored to which synthetic random noise is added.
        If the dimension `M` is not given, the whole set of magic sensors is used.
        
        .. math::
            y_m = v_m(u) +\varepsilon_m \qquad \qquad m = 1, \dots, M
        
        If the dimension :math:`M` is not given, the whole set of magic sensors is used.
        
        Parameters
        ----------
        snap : Function
            Function from which measurements are to be extracted
        noise_value : float
            Standard deviation of the noise, modelled as a normal :math:`\mathcal{N}(0, \sigma^2)`
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

        # Adding synthetic noise
        measure += np.random.normal(0, noise_value, len(measure))

        return measure
    
    def reconstruct(self, snap: np.ndarray, M: int, noise_value: float, reg_param: float) -> Tuple[Function, Function]:
        r"""
        The interpolant for `snap` :math:`u` input is computed, by solving the TR-GEIM linear system
        
         .. math::
            \left(\mathbb{B}^T\mathbb{B}+\lambda \mathbb{T}^T\mathbb{T}\right)\boldsymbol{\beta} = \mathbb{B}^T\mathbf{y}+\lambda \mathbb{T}^T\mathbb{T} \langle{\boldsymbol{\beta}}\rangle
       
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
        noise_value : float
            Standard deviation of the noise, modelled as a normal :math:`\mathcal{N}(0, \sigma^2)`
        reg_param : float
            Regularising parameter :math:`\lambda`
            
        Returns
        ----------
        interp : Function
            Interpolant Field :math:`\mathcal{I}_M[u]` of TR-GEIM
        resid : Function 
            Residual Field :math:`r_M[u]`
        computational_time : dict
            Dictionary with the CPU time of the most relevant operations during the online phase.
        coeff : np.ndarray 
            Coefficients of the GEIM expansion :math:`\boldsymbol{\beta}(\boldsymbol{\mu})`
            
        """
        if M > self.Mmax:
            print('The maximum number of measures must not be higher than '+str(self.Mmax)+' --> set equal to '+str(self.Mmax))
            M = self.Mmax
        
        if isinstance(snap, Function):
            snap = snap.x.array[:]
        
        # Variables to store the computational times
        computational_time = dict()
        timing = Timer() 
        
        # Compute measurement vector with noise
        timing.start()
        y = self.compute_measure(snap, noise_value, M)
        computational_time['Measure'] = timing.stop()
        
        # Creating matrix and vector
        timing.start()
        sys_matrix = (self.B[:M, :M].T @ self.B[:M, :M]) + reg_param * (self.T[:M, :M].T @ self.T[:M, :M])
        rhs = np.dot(self.B[:M, :M].T, y) + reg_param * np.dot((self.T[:M, :M].T @ self.T[:M, :M]), self.mean_beta[:M])

        # Solving the linear system
        coeff = la.solve(sys_matrix, rhs)
        computational_time['LinearSystem'] = timing.stop()
        
        # Compute the interpolant and residual
        timing.start()
        interp = self.mf.lin_combine(coeff)
        computational_time['Reconstruction'] = timing.stop()
        
        resid = np.abs(snap - interp)
        
        return interp, resid, computational_time, coeff
    
    def hyperparameter_tuning(self, snaps: FunctionsList, noise_value: float, 
                              lambda_lim = [10, 50], n_samples: int = 20, M: int = None, verbose = False):
        r"""
        The regularising parameter :math:`\lambda` of the TR-GEIM linear system is calibrated as the one minimising the absolute error in :math:`L^2`, namely
        
        .. math::
            \hat{\lambda} = \text{arg}\,\min\limits_{\lambda\in\mathbb{R}^+} E_{M, \lambda}
        
        given :math:`E_{M, \lambda}` the average absolute error in :math:`L^2` with :math:`M` sensors. The optimization problem is solved sampling :math:`\lambda` in a specific range.
        
        Parameters
        ----------
        snaps : FunctionsList
            List of functions belonging to the validation set to reconstruct
        noise_value : float
            Standard deviation of the noise, modelled as a normal :math:`\mathcal{N}(0, \sigma^2)`
        lambda_lim : tuple of floats, optional (Default = [10, 50])
            Lower and upper bound for :math:`\lambda^*` entering in the regularisation parameter of TR-GEIM as :math:`\lambda = \lambda^* \cdot \sigma^2`, given :math:`\sigma^2` as the variance of the noise.
        n_samples: int, optional (Default = 20)
            Number of samples for the hyperparameter :math:`\lambda` to optimize.
        M : int, optional (Default = None)
            Number of sensor to use (if None is set to the number of magic functions/sensors)
        verbose : boolean, optional (default = False)
            If true, output is printed.

        
        Returns
        ----------
        lambda_opt : float
            Optimal value of :math:`\hat{\lambda}`
        lambda_star_samples : np.ndarray
            Sample values of :math:`\lambda^\star` entering the regularising parameter as :math:`\lambda=\lambda^\star \cdot \sigma^2`
        ave_abs_err : np.ndarray
            Average absolure reconstruction error associated to each output value of :math:`\lambda^\star`
        
        """
        # Check on the input M, maximum number of sensors to use
        if M is None:
            M = self.Mmax
        elif M > self.Mmax:
            print('The maximum number of measures must not be higher than '+str(self.Mmax)+' --> set equal to '+str(self.Mmax))
            M = self.Mmax

        # Defining the lambda^* parameters for the optimization problem
        lambda_star_samples = np.linspace(lambda_lim[0], lambda_lim[1], n_samples)

        # Defining structures to store the absolute errors
        Ns = len(snaps)
        abs_err = np.zeros((Ns, n_samples))
    
        resid = Function(snaps.fun_space).copy()
        
        if verbose:
            progressBar = LoopProgress(msg = "TR-GEIM HyperParameter Tuning for sigma = {:.2f}".format(noise_value)+' - '+ self.name, final = n_samples )
            
            
        clean_measures = np.zeros((Ns, M))
        for mu in range(Ns):
            clean_measures[mu, :] = self.compute_measure(snaps(mu), M)
                
        # Solving the optimization problem using brute force
        for ii, lambda_star in enumerate(lambda_star_samples):
            
            reg_param = lambda_star * noise_value ** 2
            
            for mu in range(Ns):
                
                # Adding synthetic noise
                y = clean_measures[mu, :] + np.random.normal(0, noise_value, len(clean_measures[mu, :]))

                # Creating matrix and vector
                sys_matrix = (self.B[:M, :M].T @ self.B[:M, :M]) + reg_param * (self.T[:M, :M].T @ self.T[:M, :M])
                rhs = np.dot(self.B[:M, :M].T, y) + reg_param * np.dot((self.T[:M, :M].T @ self.T[:M, :M]), self.mean_beta[:M])

                # Solving the linear system
                coeff = la.solve(sys_matrix, rhs)
                
                # Computing absolute error
                resid.x.array[:] = snaps(mu) - self.mf.lin_combine(coeff)
                abs_err[mu, ii] = self.norms.L2norm(resid)
            
            if verbose:
                progressBar.update(1, percentage = False)
                
        lambda_opt = lambda_star_samples[np.argmin(abs_err.mean(axis=0))]
        
        return lambda_opt, lambda_star_samples, abs_err.mean(axis = 0)
    
    def real_reconstruct(self, measure: np.ndarray, reg_param: float):
        r"""
        The interpolant given the `measure` vector :math:`\mathbf{y}` input is computed, by solving the GEIM linear system
        
        .. math::
            \left(\mathbb{B}^T\mathbb{B}+\lambda \mathbb{T}^T\mathbb{T}\right)\boldsymbol{\beta} = \mathbb{B}^T\mathbf{y}+\lambda \mathbb{T}^T\mathbb{T} \langle{\boldsymbol{\beta}}\rangle
       
        then the interpolant is computed and returned
        
        .. math::
            \mathcal{I}_M(\mathbf{x}) = \sum_{m=1}^M \beta_m[u] \cdot q_m(\mathbf{x})
        
        Parameters
        ----------
        measure : np.ndarray
            Measurement vector, shaped as :math:`M \times N_s`, given :math:`M` the number of sensors used and :math:`N_s` the number of parametric realisation.
        reg_param : float
            Regularising parameter :math:`\lambda`
        
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
            sys_matrix = (self.B[:M, :M].T @ self.B[:M, :M]) + reg_param * (self.T[:M, :M].T @ self.T[:M, :M])
            rhs = np.dot(self.B[:M, :M].T, y) + reg_param * np.dot((self.T[:M, :M].T @ self.T[:M, :M]), self.mean_beta[:M])

            coeff = la.solve(sys_matrix, rhs)
            computational_time['LinearSystem'] = timing.stop()

            # Compute the interpolant and residual
            timing.start()
            interps.append(self.mf.lin_combine(coeff))
            
            computational_time['Reconstruction'] = timing.stop()
        
        return interps, computational_time