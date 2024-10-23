# Synthetic Online Phase: Parameterised-Background Data-Weak formulation
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

# PBDW online (synthetic and real measures)
class PBDW():
    r"""
    A class implementing the online phase of the Parameterised-Background Data-Weak (PBDW) formulation, given a list of sensors' Riesz representation :math:`\{g_m\}_{m=1}^M` for the update space and basis functions :math:`\{\zeta_n\}_{n=1}^N` for the reduced one.
    In particular, the following matrices are defined :math:`(n,n' = 1,\dots,N)` and :math:`(m,m' = 1,\dots,M)` 
    
    .. math::
        \mathbb{A}_{mm'}=\left(g_m,\,g_{m'}\right)_{\mathcal{U}}  \qquad \qquad 
        \mathbb{K}_{mn}=\left(g_m,\,\zeta_{n}\right)_{\mathcal{U}}
      
    given :math:`\mathcal{U}` as the functional space.
    
    Parameters
    ----------
    basis_functions : FunctionsList
        List of functions spanning the reduced space
    basis_sensors : FunctionsList
        List of sensors representation spanning the update space
    name : str
        Name of the snapshots (e.g., temperature T)
    is_H1: boolean, optional (Default= False)
        Boolean indicating whether to use scalar product in :math:`\mathcal{H}^1` or :math:`L^2`.

    """

    def __init__(self, basis_functions: FunctionsList, basis_sensors: FunctionsList, name: str, is_H1: bool = False) -> None:

        # store basis function and basis sensors
        self.basis_functions = FunctionsList(basis_functions.fun_space)
        self.basis_sensors = FunctionsList(basis_sensors.fun_space)
        
        self.basis_functions._list = basis_functions.copy()
        self.basis_sensors._list   = basis_sensors.copy()
        
        self.V = basis_functions.fun_space
        
        # Defining the norm class to make scalar products and norms
        self.is_H1 = is_H1
        self.norm = norms(self.V, is_H1=is_H1)
        self.name = name

        N = len(basis_functions)
        M = len(basis_sensors)

        # A_{ii,jj} = (basis_sensors[ii], basis_sensors[jj])
        self.A = np.zeros((M,M))
        for ii in range(M):
            for jj in range(M):
                if jj>=ii:
                    if self.is_H1:
                        self.A[ii,jj] = self.norm.H1innerProd(basis_sensors(ii), basis_sensors(jj), semi = False)
                    else:
                        self.A[ii, jj] = self.norm.L2innerProd(basis_sensors(ii), basis_sensors(jj))
                else:
                    self.A[ii,jj] = self.A[jj, ii]
        
        # K_{ii,jj} = (basis_sensors[ii], basis_functions[jj])
        self.K = np.zeros((M, N))
        for ii in range(M):
            for jj in range(N):
                if self.is_H1:
                    self.K[ii,jj] = self.norm.H1innerProd(basis_sensors(ii), basis_functions(jj), semi = False)
                else:
                    self.K[ii, jj] = self.norm.L2innerProd(basis_sensors(ii), basis_functions(jj))

        self.Nmax = N
        self.Mmax = M

    def synt_test_error(self, test_snap: FunctionsList, N : int = None, M : int = None, 
                        noise_value : float = None, reg_param : float = 0.,
                        verbose : bool = False) -> namedtuple:
        r"""
        The absolute and relative error on the test set is computed, by solving the PBDW system
        
        .. math::
            \left[ 
                \begin{array}{ccc}
                    \xi \cdot M \cdot \mathbb{I} + \mathbb{A} & & \mathbb{K}  \\  & & \\
                    \mathbb{K}^T & & 0
                \end{array}
                \right] \cdot
                \left[ 
                \begin{array}{c}
                    \boldsymbol{\alpha} \\ \\ \boldsymbol{\theta}
                \end{array}
                \right]   =
                \left[ 
                \begin{array}{c}
                    \mathbf{y} \\ \\ \mathbf{0}
                \end{array}
            \right]

        given :math:`\mathbf{y}\in\mathbb{R}^M`. Then, the full can state for the snapshot :math:`u` can be written as
        
        .. math::
            u(\mathbf{x};\boldsymbol{\mu}) \simeq z_N(\mathbf{x};\boldsymbol{\mu})+\eta_M(\mathbf{x};\boldsymbol{\mu})
                                        = \sum_{n=1}^N\alpha_n(\boldsymbol{\mu})\cdot \zeta_n(\mathbf{x})+
                                          \sum_{m=1}^M \theta_m(\boldsymbol{\mu})\cdot g_m(\mathbf{x})

        Parameters
        ----------
        test_snap : FunctionsList
            List of functions belonging to the test set to reconstruct with PBDW
        N : int, optional (default = None)
            Maximum number of basis functions :math:`\zeta_n` to use (if None is set to the number of basis functions)
        M : int, optional (default = None)
            Maximum number of basis sensors :math:`g_m` to use (if None is set to the number of basis sensors)
        noise_value : float, optional (default = None)
            Standard deviation of the noise, modelled as a normal :math:`\mathcal{N}(0, \sigma^2)`
        reg_param : float, optional (default = 0.)
            Hyperparameter :math:`\xi` weighting the importance of the model with respect to the measurements.
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

        if M is None:
            M = self.Mmax
        elif M > self.Mmax:
            print('The maximum number of measures must not be higher than '+str(self.Mmax)+' --> set equal to '+str(self.Mmax))
            M = self.Mmax

        if N is None:
            N = self.Nmax
        elif N > self.Nmax:
            print('The maximum number of basis functions must not be higher than '+str(self.Nmax)+' --> set equal to '+str(self.Nmax))
            N = self.Nmax

        Ns_test = len(test_snap)
        abs_err = np.zeros((Ns_test, M - N + 1))
        rel_err = np.zeros_like(abs_err)

        if verbose:
            progressBar = LoopProgress(msg = "Computing PBDW test error (synthetic) with N = "+str(N)+" - " + self.name, final = Ns_test )

        # Variables to store the computational times
        computational_time = dict()
        computational_time['Measure']      = np.zeros((Ns_test, M))
        computational_time['LinearSystem'] = np.zeros((Ns_test, M - N + 1))
        computational_time['Errors']       = np.zeros((Ns_test, M - N + 1))
         
        timing = Timer()    
        
        resid = Function(self.V).copy()
        for mu in range(Ns_test):
            
            # Compute snapshot norm
            timing.start()
            norma_snap = self.norm.L2norm(test_snap(mu))
            computational_time['Errors'][mu, :] = timing.stop()

            # Compute measures
            y_clean = np.zeros((M,))
            for mm in range(M):
                timing.start()
                if self.is_H1:
                    y_clean[mm] = self.norm.H1innerProd(test_snap(mu), self.basis_sensors(mm), semi=False)
                else:
                    y_clean[mm] = self.norm.L2innerProd(test_snap(mu), self.basis_sensors(mm))
                computational_time['Measure'][mu, mm] = timing.stop()
            
            timing.start()
            if noise_value is not None:
                y = y_clean + np.random.normal(scale = noise_value, size = (M,))
            else:
                y = y_clean
            computational_time['Measure'][mu, :] += timing.stop()

            for mm in range(N, M+1):

                # Assembling and solving the linear system
                timing.start()
                rhs = np.hstack([y[:mm], np.zeros((N,))]).flatten()
                sys_matr1 = np.hstack([reg_param * (mm * np.eye(mm, mm)) + self.A[:mm, :mm], self.K[:mm, :N]])
                sys_matr2 = np.hstack([self.K[:mm, :N].T, np.zeros((N, N))])
                sys_matr = np.vstack([sys_matr1, sys_matr2])

                coeff = la.solve(sys_matr, rhs)
                computational_time['LinearSystem'][mu, mm - N] = timing.stop()

                # Compute the error
                timing.start()
                resid.x.array[:] = test_snap(mu) - (self.basis_sensors.lin_combine(coeff[:mm]) + self.basis_functions.lin_combine(coeff[mm:]))
                abs_err[mu,mm - N] = self.norm.L2norm(resid)
                rel_err[mu,mm - N] = abs_err[mu,mm - N] / norma_snap
                computational_time['Errors'][mu, mm - N] += timing.stop()

            if verbose:
                progressBar.update(1, percentage = False)

        Results = namedtuple('Results', ['mean_abs_err', 'mean_rel_err', 'computational_time'])
        synt_res = Results(mean_abs_err = abs_err.mean(axis = 0), mean_rel_err = rel_err.mean(axis = 0), computational_time = computational_time)

        return synt_res
        
    def reconstruct(self, snap: np.ndarray, N : int, M : int, 
                          noise_value : float = None, reg_param : float = 0.):
        r"""
        The PBDW system is solved
        
        .. math::
            \left[ 
                \begin{array}{ccc}
                    \xi \cdot M \cdot \mathbb{I} + \mathbb{A} & & \mathbb{K}  \\  & & \\
                    \mathbb{K}^T & & 0
                \end{array}
                \right] \cdot
                \left[ 
                \begin{array}{c}
                    \boldsymbol{\alpha} \\ \\ \boldsymbol{\theta}
                \end{array}
                \right]   =
                \left[ 
                \begin{array}{c}
                    \mathbf{y} \\ \\ \mathbf{0}
                \end{array}
            \right]

        given :math:`\mathbf{y}\in\mathbb{R}^M`. Then, the full can state for the snapshot :math:`u` can be written as
        
        .. math::
            u(\mathbf{x};\boldsymbol{\mu}) \simeq z_N(\mathbf{x};\boldsymbol{\mu})+\eta_M(\mathbf{x};\boldsymbol{\mu})
                                        = \sum_{n=1}^N\alpha_n(\boldsymbol{\mu})\cdot \zeta_n(\mathbf{x})+
                                          \sum_{m=1}^M \theta_m(\boldsymbol{\mu})\cdot g_m(\mathbf{x})
 
        Parameters
        ----------
        snap : Function as np.ndarray
            Snap to reconstruct, if a function is provided, the variable is reshaped.
        N : int, optional (default = None)
            Maximum number of basis functions :math:`\zeta_n` to use (if None is set to the number of basis functions)
        M : int, optional (default = None)
            Maximum number of basis sensors :math:`g_m` to use (if None is set to the number of basis sensors)
        noise_value : float, optional (default = None)
            Standard deviation of the noise, modelled as a normal :math:`\mathcal{N}(0, \sigma^2)`
        reg_param : float, optional (default = 0.)
            Hyperparameter :math:`\xi` weighting the importance of the model with respect to the measurements.

        Returns
        ----------
        recon : Function
            Function containing the reconstruction using :math:`M` sensors
        resid : Function
            Function containing the residual field (absolute difference between interpolant and true field) using :math:`M` sensors
        computational_time : dict
            Dictionary with the CPU time of the most relevant operations during the online phase.
       """

        assert(M >= N)
        
        if isinstance(snap, Function):
            snap = snap.x.array[:]
        
        # Variables to store the computational times
        computational_time = dict()
        timing = Timer() 
        
        # Compute measurement vector
        timing.start()
        y_clean = np.zeros((M,))
        for mm in range(M):
            if self.is_H1:
                y_clean[mm] = self.norm.H1innerProd(snap, self.basis_sensors(mm), semi=False)
            else:
                y_clean[mm] = self.norm.L2innerProd(snap, self.basis_sensors(mm))
    
        if noise_value is not None:
            y = y_clean + np.random.normal(scale = noise_value, size = (M,))
        else:
            y = y_clean
        computational_time['Measure'] = timing.stop()

        # Assembling and solving the linear system
        timing.start()
        rhs = np.hstack([y[:M], np.zeros((N,))]).flatten()
        sys_matr1 = np.hstack([reg_param * (M * np.eye(M, M)) + self.A[:M, :M], self.K[:M, :N]])
        sys_matr2 = np.hstack([self.K[:M, :N].T, np.zeros((N, N))])
        sys_matr = np.vstack([sys_matr1, sys_matr2])

        coeff = la.solve(sys_matr, rhs)
        computational_time['LinearSystem'] = timing.stop()

        recon = self.basis_sensors.lin_combine(coeff[:M]) + self.basis_functions.lin_combine(coeff[M:])
        resid = np.abs(snap - recon)
        
        return recon, resid, computational_time

    def hyperparameter_tuning(self, snaps: FunctionsList, noise_value: float, 
                              xi_lim = [-4, 5], n_samples : int = 10, num_rep_exp: int = 10,
                              N: int = None, M: int = None, verbose : bool = False):
        r"""
        The hyperparameter :math:`\xi` of the PBDW statement is calibrated as the one minimising the absolute error in :math:`L^2`, namely
        
        .. math::
            \hat{\xi} = \text{arg}\,\min\limits_{\xi\in\mathbb{R}^+} E_{M, \xi}
        
        given :math:`E_{M, \xi}` the average absolute error in :math:`L^2` with :math:`M` sensors. The optimization problem is solved sampling :math:`\xi` in a specific range :math:`\Xi \subset\mathbb{R}^+`.
        
        Parameters
        ----------
        snaps : FunctionsList
            List of functions belonging to the validation set to reconstruct
        noise_value : float
            Standard deviation of the noise, modelled as a normal :math:`\mathcal{N}(0, \sigma^2)`
        xi_lim : tuple of floats, optional (Default = [0, 6])
            Lower and upper bound (the input is the exponent for :math:`10^x`) of :math:`\xi`, regularisation parameter of PBDW.
        num_rep_exp : int, optional (Default = 10)
            Number of repeated numerical experiment to ensure statistical robustness
        N : int, optional (Default = None)
            Number of basis function to use (if None is set to the number of basis functions)
        M : int, optional (Default = None)
            Number of sensor to use (if None is set to the number of basis sensors)
        verbose : boolean, optional (default = False)
            If true, output is printed.
        
        Returns
        ----------
        xi_opt : float
            Optimal value of :math:`\hat{\xi}` per each numerical experiment
        xi_samples : np.ndarray
            Sample values of :math:`\xi`
        ave_abs_err : np.ndarray
            Average (also averaged wrt to the num. experiments) absolure reconstruction error associated to each output value of :math:`\xi_{sample}`
        
        """
        if M is None:
            M = self.Mmax
        elif M > self.Mmax:
            print('The maximum number of measures must not be higher than '+str(self.Mmax)+' --> set equal to '+str(self.Mmax))
            M = self.Mmax

        if N is None:
            N = self.Nmax
        elif N > self.Nmax:
            print('The maximum number of basis functions must not be higher than '+str(self.Nmax)+' --> set equal to '+str(self.Nmax))
            N = self.Nmax
            
        # Defining the lambda^* parameters for the optimization problem
        xi_samples = np.logspace(xi_lim[0], xi_lim[1], n_samples, base = 10)

        # Defining structures to store the absolute errors
        Ns = len(snaps)
    
        resid = Function(snaps.fun_space).copy()
        
        if verbose:
            progressBar = LoopProgress(msg = "PBDW HyperParameter Tuning for sigma = {:.2e}".format(noise_value)+' - '+ self.name, final = num_rep_exp )
            
            
        clean_measures = np.zeros((Ns, M))
        snap_norm = np.zeros((Ns,))
        
        for mu in range(Ns):
            # snap_norm[mu] = self.norm.L2norm(snaps(mu))
            snap_norm[mu] = self.norm.integral(np.abs(snaps(mu)))
            for mm in range(M):
                if self.is_H1:
                    clean_measures[mu, mm] = self.norm.H1innerProd(snaps(mu), self.basis_sensors(mm), semi=False)
                else:
                    clean_measures[mu, mm] = self.norm.L2innerProd(snaps(mu), self.basis_sensors(mm))
        
       # The numerical experiment is repeated to ensure statistical robustness 
        abs_errs = list()
        rel_errs = list()
        xi_opts  = list()
        
        for exp in range(num_rep_exp):
            abs_errs.append(np.zeros((Ns, n_samples)))
            rel_errs.append(np.zeros((Ns, n_samples)))
            
            # Solving the optimization problem using brute force
            for ii, xi in enumerate(xi_samples):
                for mu in range(Ns):
                    # Adding synthetic noise
                    y = clean_measures[mu, :] + np.random.normal(0, noise_value, len(clean_measures[mu, :]))

                    # Creating matrix and vector
                    rhs = np.hstack([y[:mm], np.zeros((N,))]).flatten()
                    sys_matr1 = np.hstack([xi * (mm * np.eye(mm, mm)) + self.A[:mm, :mm], self.K[:mm, :N]])
                    sys_matr2 = np.hstack([self.K[:mm, :N].T, np.zeros((N, N))])
                    sys_matr = np.vstack([sys_matr1, sys_matr2])

                    # Solving the linear system
                    coeff = la.solve(sys_matr, rhs)
                    
                    # Computing absolute error
                    resid.x.array[:] = np.abs(snaps(mu) - (self.basis_sensors.lin_combine(coeff[:mm]) + self.basis_functions.lin_combine(coeff[mm:])))
                    abs_errs[exp][mu, ii] = self.norm.average(resid)
                    rel_errs[exp][mu, ii] = abs_errs[exp][mu, ii] / snap_norm[mu]
                
            if verbose:
                progressBar.update(1, percentage = False)
                    
            xi_opts.append(xi_samples[np.argmin(abs_errs[exp].mean(axis=0))])
        
        return xi_opts, xi_samples, abs_errs, rel_errs
    
    def compute_measure(self, snap: Function, noise_value: float, M = None) -> np.ndarray:
        r"""
        Computes the measurement vector from the `snap` input, using the basis sensors stored to which synthetic random noise is added.
        If the dimension `M` is not given, the whole set of basis sensors is used.
        
        .. math::
            y_m = v_m(u) +\varepsilon_m \qquad \qquad m = 1, \dots, M
        
        If the dimension :math:`M` is not given, the whole set of basis sensors is used.
        
        Parameters
        ----------
        snap : Function
            Function from which measurements are to be extracted
        noise_value : float
            Standard deviation of the noise, modelled as a normal :math:`\mathcal{N}(0, \sigma^2)`
        M : int, optional (default = None)
            Maximum number of sensor to use (if None is set to the number of basis sensors)

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

        y_clean = np.zeros((M,))
        for mm in range(M):
            if self.is_H1:
                y_clean[mm] = self.norm.H1innerProd(snap, self.basis_sensors(mm), semi=False)
            else:
                y_clean[mm] = self.norm.L2innerProd(snap, self.basis_sensors(mm))
    
        if noise_value is not None:
            measure = y_clean + np.random.normal(scale = noise_value, size = (M,))
        else:
            measure = y_clean

        return measure
    
    def real_reconstruct(self, measure: np.ndarray, 
                         N : int = None, reg_param : float = 0.):
        r"""
        The state estimation given the `measure` vector :math:`\mathbf{y}` input is computed, by solving the PBDW linear system
        
        .. math::
            \left[ 
                \begin{array}{ccc}
                    \xi \cdot M \cdot \mathbb{I} + \mathbb{A} & & \mathbb{K}  \\  & & \\
                    \mathbb{K}^T & & 0
                \end{array}
                \right] \cdot
                \left[ 
                \begin{array}{c}
                    \boldsymbol{\alpha} \\ \\ \boldsymbol{\theta}
                \end{array}
                \right]   =
                \left[ 
                \begin{array}{c}
                    \mathbf{y} \\ \\ \mathbf{0}
                \end{array}
            \right]

        given :math:`\mathbf{y}\in\mathbb{R}^M`. Then, the full can state for the snapshot :math:`u` can be written as
        
        .. math::
            u(\mathbf{x};\boldsymbol{\mu}) \simeq z_N(\mathbf{x};\boldsymbol{\mu})+\eta_M(\mathbf{x};\boldsymbol{\mu})
                                        = \sum_{n=1}^N\alpha_n(\boldsymbol{\mu})\cdot \zeta_n(\mathbf{x})+
                                          \sum_{m=1}^M \theta_m(\boldsymbol{\mu})\cdot g_m(\mathbf{x})
 
        Parameters
        ----------
        measure : np.ndarray
            Measurement vector, shaped as :math:`M \times N_s`, given :math:`M` the number of sensors used and :math:`N_s` the number of parametric realisation.
        N : int, optional (default = None)
            Maximum number of basis functions :math:`\zeta_n` to use (if None is set to the number of basis functions)
        reg_param : float, optional (default = 0.)
            Hyperparameter :math:`\xi` weighting the importance of the model with respect to the measurements.
        
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
            
        if N is None:
            N = self.Nmax
        elif N > self.Nmax:
            print('The maximum number of basis functions must not be higher than '+str(self.Nmax)+' --> set equal to '+str(self.Nmax))
            N = self.Nmax
        
        assert M >= self.Nmax
        
        # Variables to store the computational times
        computational_time = dict()
        timing = Timer() 
        
        interps = FunctionsList(self.V)
        
        for mu in range(Ns):
            
            y = measure[:, mu]
            
            # Solving the linear system
            timing.start()
            rhs = np.hstack([y, np.zeros((N,))]).flatten()
            sys_matr1 = np.hstack([reg_param * (M * np.eye(M, M)) + self.A[:M, :M], self.K[:M, :N]])
            sys_matr2 = np.hstack([self.K[:M, :N].T, np.zeros((N, N))])
            sys_matr = np.vstack([sys_matr1, sys_matr2])

            coeff = la.solve(sys_matr, rhs)
            computational_time['LinearSystem'] = timing.stop()

            # Compute the interpolant and residual
            timing.start()
            interps.append(self.basis_sensors.lin_combine(coeff[:M]) + self.basis_functions.lin_combine(coeff[M:]))
            
            computational_time['Reconstruction'] = timing.stop()
        
        return interps, computational_time