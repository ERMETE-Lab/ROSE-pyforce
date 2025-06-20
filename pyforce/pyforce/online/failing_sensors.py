# Synthetic Online Phase: Failing Sensors with drifted measures
# Author: Stefano Riva, PhD Student, NRG, Politecnico di Milano
# Latest Code Update: 7 June 2025
# Latest Doc  Update: 7 June 2025

import numpy as np
import numpy.linalg as la

from dolfinx.fem import Function
from pyforce.tools.backends import norms, LoopProgress
from pyforce.tools.functions_list import FunctionsList

def remove_lin_combine(fun_list: FunctionsList, vec: np.ndarray, M: int, sensI_drifted: int):

    r"""
    This auxiliary function is used to perform a linear combination of basis functions without `sensI_drifted` (i.e., :math:`j`).

    .. math::
        u = \sum_{i=1,i\neq j}^{M} \alpha_i \cdot \psi_i

    Parameters
    ----------
    fun_list : FunctionsList
        List of basis functions
    vec : np.ndarray
        Iterable containing the coefficients :math:`\boldsymbol{\alpha}\in\mathbb{R}^{M-1}` of the linear combination.
    M : int
        Maximum number of basis functions without removal.
    sensI_drifted : int
        Index of the drifted sensor and hence drifted coefficient :math:`\alpha_j` .
    
    Returns
    -------
    combination : np.ndarray
        `np.ndarray` object storing the result of the linear combination
    """
        

    kk = 0
    for ii in range(M):
        combination = np.zeros(fun_list.fun_shape,)
        for ii in range(len(vec)):
            if ii != sensI_drifted:
                combination += vec[kk] * fun_list(ii)
                kk += 1
        return combination


def compute_measure(snaps: FunctionsList, sensors: FunctionsList, noise_value: float = None, M = None) -> np.ndarray:
        r"""
        Computes the measurement matrix from the `snaps` input, using `sensors` given as input to which synthetic random noise is added.
        If the dimension `M` is not given, the whole set of magic sensors is used.
        
        .. math::
            y_m = v_m(u) +\epsilon_m \qquad \qquad m = 1, \dots, M
        
        If the dimension :math:`M` is not given, the whole set of magic sensors is used.
        
        Parameters
        ----------
        snap : FunctionsList
            FunctionsList from which measurements are to be extracted
        sensors : FunctionsList
            FunctionsList containing the sensors
        noise_value : float, optional (Default = None)
            Standard deviation of the noise, modelled as a normal :math:`\mathcal{N}(0, \sigma^2)`
        M : int, optional (default = None)
            Maximum number of sensor to use (if None is set to the number of magic functions/sensors)

        Returns
        ----------
        measure : np.ndarray
            Measurement vector :math:`\mathbf{y}\in\mathbb{R}^{M\times N_s}`
        """
        # Check on the input M, maximum number of sensors to use
        if M is None:
            M = len(sensors)
        elif M > len(sensors):
            print('The maximum number of measures must not be higher than '+str(len(sensors))+' --> set equal to '+str(len(sensors)))
            M = len(sensors)

        Ns = len(snaps)
        norm_ = norms(snaps.fun_space)
    
        measure = np.zeros((M, Ns))
        for mm in range(M):
            for mu in range(len(snaps)):
                measure[mm, mu] = norm_.L2innerProd(snaps(mu), sensors(mm))

        # Adding synthetic noise
        if noise_value is not None:
            measure += np.random.normal(0, noise_value, measure.shape)

        return measure

# TR-GEIM online with drifted synthetic measures
class TRGEIM():
    r"""
    This class can be used to perform the online phase of the TR-GEIM algorihtm for synthetic measures :math:`\mathbf{y}\in\mathbb{R}^M` obtained as evaluations of the magic sensors on the snapshot :math:`u(\mathbf{x};\,\boldsymbol{\mu})` as
    
    .. math::
        y_m = v_m(u(\boldsymbol{\mu})) + \epsilon_m + \delta_{m}  \qquad \qquad m = 1, \dots, M

    in which :math:`\epsilon_m\sim \mathcal{N}(0, \sigma^2)` is random noise and :math:`\delta_m \sim \mathcal{N}(\kappa, \rho^2)` acting on some measurements. This term is referred to as drift.
    
    Two approaches are implemented:
    
    - **Unregularised case**: how the drifted measurement affect the reconstruction?
    - **Remove measure case**: the failed measure (once at a time) is removed as well as the correspondent magic function


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

    def compute_measure(self, snap: Function, noise_value: float = None, M = None) -> np.ndarray:
        r"""
        Computes the measurement vector from the `snap` input, using the magic sensors stored to which synthetic random noise is added.
        If the dimension `M` is not given, the whole set of magic sensors is used.
        
        .. math::
            y_m = v_m(u) +\epsilon_m \qquad \qquad m = 1, \dots, M
        
        If the dimension :math:`M` is not given, the whole set of magic sensors is used.
        
        Parameters
        ----------
        snap : Function
            Function from which measurements are to be extracted
        noise_value : float, optional (Default = None)
            Standard deviation of the noise, modelled as a normal :math:`\mathcal{N}(0, \sigma^2)`
        M : int, optional (default = None)
            Maximum number of sensor to use (if None is set to the number of magic functions/sensors)

        Returns
        ----------
        measure : np.ndarray
            Measurement vector :math:`\mathbf{y}\in\mathbb{R}^M`
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
        if noise_value is not None:
            measure += np.random.normal(0, noise_value, len(measure))

        return measure
    
    def drift_test_err(self, snaps: FunctionsList, M: int, noise_value: float, reg_param: float,
                             kappa: float, rho: float, idx_failed: list[int],
                             num_rep_exp: int = 30, mu_failure: int = 0, verbose = False):
        r"""
        The TR-GEIM algorithm is used to reconstruct the `snaps` (`FunctionsList`), by solving the TR-GEIM linear system
        
         .. math::
            \left(\mathbb{B}^T\mathbb{B}+\lambda \mathbb{T}^T\mathbb{T}\right)\boldsymbol{\beta} = \mathbb{B}^T\mathbf{y}+\lambda \mathbb{T}^T\mathbb{T} \langle{\boldsymbol{\beta}}\rangle
       
        then the inteprolant and residual are computed and returned per each element of the list.
        
        Parameters
        ----------
        snaps : FunctionsList
            Function to reconstruction
        M : int
            Number of sensor to use
        noise_value : float
            Standard deviation of the noise, modelled as a normal :math:`\mathcal{N}(0, \sigma^2)`
        reg_param : float
            Regularising parameter :math:`\lambda`
        kappa : float
            Mean value :math:`\kappa` of the drift
        rho : float
            Standard deviation  :math:`\rho` of the drift
        idx_failed : list[int]
            List of integers with the failed sensors
        num_rep_exp : int, optional (default = 30)
            Number of repeated experiments.
        mu_failure : int, optional (default = 0)
            Index from which failure starts, typically time.
        verbose : boolean, optional (default = False)
            If true, output is printed.
            
        Returns
        ----------
        errors : np.ndarray
            Errors per each element in `snaps` (first column: absolute, second column: relative), measure in :math:`||\cdot ||_{L^2}` and averaged for the numerical experiments.
        interps : FunctionsList
            Interpolant Field :math:`\mathcal{I}_M[u]` of TR-GEIM.
        resids : FunctionsList 
            Residual Field :math:`r_M[u]`,
            
        """
        
        if M > self.Mmax:
            print('The maximum number of measures must not be higher than '+str(self.Mmax)+' --> set equal to '+str(self.Mmax))
            M = self.Mmax
        
        assert(max(idx_failed) < M)
        
        Ns = len(snaps)
        
        # Computing clean measures and norms of the snapshots
        clean_measures = np.zeros((Ns, M))
        snap_norms = np.zeros((Ns,))
        for mu in range(Ns):
            clean_measures[mu] = self.compute_measure(snaps(mu), M = M)
            snap_norms[mu] = self.norms.L2norm(snaps(mu))
        
        errors  = np.zeros((num_rep_exp, Ns, 2))
        
        interps = FunctionsList(self.V)
        resids  = FunctionsList(self.V)
        
        if verbose:
            progressBar = LoopProgress(msg = "Computing TR-GEIM drifted (synthetic) - " + self.name, final = num_rep_exp )
    
        # Assembling matrix for TR-GEIM
        sys_matrix = (self.B[:M, :M].T @ self.B[:M, :M]) + reg_param * (self.T[:M, :M].T @ self.T[:M, :M])
    
        for kk in range(num_rep_exp):
            
            for mu in range(Ns):
            
                # Compute measurement vector with noise
                # y = self.compute_measure(snaps(mu), noise_value, M)
                y = clean_measures[mu].reshape(M,) + np.random.normal(0, noise_value, M).reshape(M,)
                
                # Adding drift
                if mu >= mu_failure:
                    y[idx_failed] += np.random.normal(kappa, rho, len(idx_failed))
            
                # Creating vector
                rhs = np.dot(self.B[:M, :M].T, y) + reg_param * np.dot((self.T[:M, :M].T @ self.T[:M, :M]), self.mean_beta[:M])

                # Solving the linear system
                coeff = la.solve(sys_matrix, rhs)
                
                # Compute the interpolant and residual
                interpolant = self.mf.lin_combine(coeff)
                residual = np.abs(snaps(mu) - interpolant)
                
                errors[kk, mu, 0] = self.norms.L2norm(residual)
                errors[kk, mu, 1] = errors[kk, mu, 0] / snap_norms[mu]

                # Storing interpolant and residual
                if kk + 1 == num_rep_exp:
                    interps.append(interpolant)
                    resids.append(residual)
                
            if verbose:
                progressBar.update(1, percentage = False)
                
        return errors, interps, resids
    
    def pure_remove_test_err(self, snaps: FunctionsList, M: int, noise_value: float, reg_param: float,
                                   idx_failed: list[int], mu_failure: int = 0, verbose = False):
        r"""
        The TR-GEIM algorithm is used to reconstruct the `snaps` (`FunctionsList`), by solving the modified TR-GEIM linear system: in particular, `idx_failed` measure is removed thus from :math:`\mathbf{y}` the `idx_failed` row is deleted, from :math:`\mathbb{T}` and :math:`\mathbb{B}` their `idx_failed` row and col are deleted.
        
        The interpolant is then defined as the sum over the obtained coefficients from the modified TR-GEIM linear system, without `idx_failed`.
        
        Parameters
        ----------
        snaps : FunctionsList
            Function to reconstruction
        M : int
            Number of sensor to use
        noise_value : float
            Standard deviation of the noise, modelled as a normal :math:`\mathcal{N}(0, \sigma^2)`
        reg_param : float
            Regularising parameter :math:`\lambda`
        idx_failed : list
            List of integers with the failed sensor
        mu_failure : int, optional (default = 0)
            Index from which failure starts, typically time.
        verbose : boolean, optional (default = False)
            If true, output is printed.
            
        Returns
        ----------
        errors : np.ndarray
            Errors per each element in `snaps` (first column: absolute, second column: relative), measure in :math:`||\cdot ||_{L^2}`.
        interps : FunctionsList
            Interpolant Field :math:`\mathcal{I}_M[u]` of TR-GEIM.
        resids : FunctionsList 
            Residual Field :math:`r_M[u]`,
            
        """
        
        if M > self.Mmax:
            print('The maximum number of measures must not be higher than '+str(self.Mmax)+' --> set equal to '+str(self.Mmax))
            M = self.Mmax
        
        assert(max(idx_failed) < M)
        
        Ns = len(snaps)
        
        # Computing clean measures and norms of the snapshots
        clean_measures = np.zeros((Ns, M))
        snap_norms = np.zeros((Ns,))
        for mu in range(Ns):
            clean_measures[mu] = self.compute_measure(snaps(mu), M = M)
            snap_norms[mu] = self.norms.L2norm(snaps(mu))
        
        errors  = np.zeros((Ns, 2))
        
        interps = FunctionsList(self.V)
        resids  = FunctionsList(self.V)
        
        if verbose:
            progressBar = LoopProgress(msg = "Computing TR-GEIM remove (synthetic) - " + self.name, final = Ns )
    
        # Assembling matrix for TR-GEIM
        sys_matrix = (self.B[:M, :M].T @ self.B[:M, :M]) + reg_param * (self.T[:M, :M].T @ self.T[:M, :M])
    
        for mu in range(Ns):
        
            # Compute measurement vector with noise
            y = clean_measures[mu].reshape(M,) + np.random.normal(0, noise_value, M).reshape(M,)
            rhs = np.dot(self.B[:M, :M].T, y) + reg_param * np.dot((self.T[:M, :M].T @ self.T[:M, :M]), self.mean_beta[:M])

            # If failure the measurement and the matrix are modified
            if mu >= mu_failure:
                _rhs = np.delete(rhs, idx_failed)
                _sys_matrix = np.delete(np.delete(sys_matrix, idx_failed, axis=1), idx_failed, axis=0)
            else:
                _rhs = rhs
                _sys_matrix = sys_matrix

            # Solving the linear system
            coeff = la.solve(_sys_matrix, _rhs)
            
            # Compute the interpolant and residual
            if mu >= mu_failure:
                interpolant = remove_lin_combine(self.mf, coeff, M, sensI_drifted=idx_failed)
            else:
                interpolant = self.mf.lin_combine(coeff)
            residual = np.abs(snaps(mu) - interpolant)
            
            errors[mu, 0] = self.norms.L2norm(residual)
            errors[mu, 1] = errors[mu, 0] / snap_norms[mu]

            # Storing interpolant and residual
            interps.append(interpolant)
            resids.append(residual)
            
            if verbose:
                progressBar.update(1, percentage = False)
                
        return errors, interps, resids
    
    def gpr_measure_test_err(self, snaps: FunctionsList, M: int, noise_value: float, reg_param: float,
                                   ext_sens: FunctionsList, surrogate_model: list, idx_failed: list[int], mu_failure: int = 0, verbose = False):
        r"""
        The TR-GEIM algorithm is used to reconstruct the `snaps` (`FunctionsList`), by solving the TR-GEIM linear system with `idx_failed` measure.
        In order to retrieve information on the "failed measure", a surrogate model (e.g., GPR) has been trained to learn the map from non-failed external measures and the one related to `idx_failed`.
        
        The interpolant is then defined in the standard way.
        
        Parameters
        ----------
        snaps : FunctionsList
            Functions to reconstruction
        M : int
            Number of sensor to use
        noise_value : float
            Standard deviation of the noise, modelled as a normal :math:`\mathcal{N}(0, \sigma^2)`
        reg_param : float
            Regularising parameter :math:`\lambda`
        ext_sens : FunctionsList
            Basis sensors adopted to compute the external measures, input of `surrogate_model`
        surrogate_model : list
            List of all the trained surrogate models
        idx_failed : list[int]
            List of integers with the failed sensors
        mu_failure : int, optional (default = 0)
            Index from which failure starts, typically time.
        verbose : boolean, optional (default = False)
            If true, output is printed.
            
        Returns
        ----------
        errors : np.ndarray
            Errors per each element in `snaps` (first column: absolute, second column: relative), measure in :math:`||\cdot ||_{L^2}`.
        interps : FunctionsList
            Interpolant Field :math:`\mathcal{I}_M[u]` of TR-GEIM.
        resids : FunctionsList 
            Residual Field :math:`r_M[u]`,
            
        """
        
        if M > self.Mmax:
            print('The maximum number of measures must not be higher than '+str(self.Mmax)+' --> set equal to '+str(self.Mmax))
            M = self.Mmax
        
        assert(max(idx_failed) < M)
        
        Ns = len(snaps)
        
        # Computing clean measures and norms of the snapshots
        clean_measures = np.zeros((Ns, M))
        snap_norms = np.zeros((Ns,))
        for mu in range(Ns):
            clean_measures[mu] = self.compute_measure(snaps(mu), M = M)
            snap_norms[mu] = self.norms.L2norm(snaps(mu))
        
        ext_measures = compute_measure(snaps, ext_sens, noise_value=noise_value).T
        
        errors  = np.zeros((Ns, 2))
        
        interps = FunctionsList(self.V)
        resids  = FunctionsList(self.V)
        
        if verbose:
            progressBar = LoopProgress(msg = "Computing TR-GEIM ml-remove (synthetic) - " + self.name, final = Ns )
    
        # Assembling matrix for TR-GEIM
        sys_matrix = (self.B[:M, :M].T @ self.B[:M, :M]) + reg_param * (self.T[:M, :M].T @ self.T[:M, :M])
    
        for mu in range(Ns):
        
            # Compute measurement vector with noise
            y = clean_measures[mu].reshape(M,) + np.random.normal(0, noise_value, M).reshape(M,)

            # If failure the measurement and the matrix are modified
            if mu >= mu_failure:
                for ii, _idx in enumerate(idx_failed):
                    y[_idx] = surrogate_model[ii].predict(ext_measures[mu].reshape(1,-1))[0].flatten()
            rhs = np.dot(self.B[:M, :M].T, y) + reg_param * np.dot((self.T[:M, :M].T @ self.T[:M, :M]), self.mean_beta[:M])
            
            # Solving the linear system
            coeff = la.solve(sys_matrix, rhs)
            
            # Compute the interpolant and residual
            interpolant = self.mf.lin_combine(coeff)
            residual = np.abs(snaps(mu) - interpolant)
            
            errors[mu, 0] = self.norms.L2norm(residual)
            errors[mu, 1] = errors[mu, 0] / snap_norms[mu]

            # Storing interpolant and residual
            interps.append(interpolant)
            resids.append(residual)
            
            if verbose:
                progressBar.update(1, percentage = False)
                
        return errors, interps, resids
    
class PBDW():
    r"""
    This class can be used to perform the online phase of the PBDW formulation for synthetic measures :math:`\mathbf{y}\in\mathbb{R}^M` obtained as evaluations of the magic sensors on the snapshot :math:`u(\mathbf{x};\,\boldsymbol{\mu})` as
    
    .. math::
        y_m = v_m(u(\boldsymbol{\mu})) + \epsilon_m + \delta_{m}  \qquad \qquad m = 1, \dots, M

    in which :math:`\epsilon_,\sim \mathcal{N}(0, \sigma^2)` is random noise and :math:`\delta_m \sim \mathcal{N}(\kappa, \rho^2)` acting on some measurements. This term is referred to as drift.

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
        self.norms = norms(self.V, is_H1=is_H1)
        self.name = name

        N = len(basis_functions)
        M = len(basis_sensors)

        # A_{ii,jj} = (basis_sensors[ii], basis_sensors[jj])
        self.A = np.zeros((M,M))
        for ii in range(M):
            for jj in range(M):
                if jj>=ii:
                    if self.is_H1:
                        self.A[ii,jj] = self.norms.H1innerProd(basis_sensors(ii), basis_sensors(jj), semi = False)
                    else:
                        self.A[ii, jj] = self.norms.L2innerProd(basis_sensors(ii), basis_sensors(jj))
                else:
                    self.A[ii,jj] = self.A[jj, ii]
        
        # K_{ii,jj} = (basis_sensors[ii], basis_functions[jj])
        self.K = np.zeros((M, N))
        for ii in range(M):
            for jj in range(N):
                if self.is_H1:
                    self.K[ii,jj] = self.norms.H1innerProd(basis_sensors(ii), basis_functions(jj), semi = False)
                else:
                    self.K[ii, jj] = self.norms.L2innerProd(basis_sensors(ii), basis_functions(jj))

        self.Nmax = N
        self.Mmax = M

    def drift_test_err(self, snaps: FunctionsList, N: int, M: int,
                             noise_value : float, reg_param : np.ndarray,
                             kappa: float, rho: float, idx_failed: list[int],
                             num_rep_exp: int = 30, mu_failure: int = 0, verbose = False):
        r"""
        The PBDW online algorithm is used to reconstruct the `snaps` (`FunctionsList`), by solving the PBDW linear system
        
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
            
        then the inteprolant and residual are computed and returned per each element of the list.
        
        Parameters
        ----------
        snaps : FunctionsList
            Function to reconstruction
        N : int
            Dimension of the reduced space
        M : int
            Number of sensor to use
        noise_value : float
            Standard deviation of the noise, modelled as a normal :math:`\mathcal{N}(0, \sigma^2)`
        reg_param : np.ndarray
            Regularising parameter :math:`\lambda`
        kappa : float
            Mean value :math:`\kappa` of the drift
        rho : float
            Standard deviation  :math:`\rho` of the drift
        idx_failed : list[int]
            List of integers with the failed sensors
        num_rep_exp : int, optional (default = 30)
            Number of repeated experiments.
        mu_failure : int, optional (default = 0)
            Index from which failure starts, typically time.
        verbose : boolean, optional (default = False)
            If true, output is printed.
            
        Returns
        ----------
        errors : np.ndarray
            Errors per each element in `snaps` (first column: absolute, second column: relative), measure in :math:`||\cdot ||_{L^2}` and averaged for the numerical experiments.
        interps : FunctionsList
            Interpolant Field :math:`\mathcal{I}_M[u]` of TR-GEIM.
        resids : FunctionsList 
            Residual Field :math:`r_M[u]`, 
        """

        if M > self.Mmax:
            print('The maximum number of measures must not be higher than '+str(self.Mmax)+' --> set equal to '+str(self.Mmax))
            M = self.Mmax

        if N > self.Nmax:
            print('The maximum number of basis functions must not be higher than '+str(self.Nmax)+' --> set equal to '+str(self.Nmax))
            N = self.Nmax

        assert(max(idx_failed) < M)
        
        Ns = len(snaps)
        
        # Computing clean measures and norms of the snapshots
        clean_measures = np.zeros((Ns, M))
        snap_norms = np.zeros((Ns,))
        for mu in range(Ns):
            for mm in range(M):
                if self.is_H1:
                    clean_measures[mu, mm] = self.norms.H1innerProd(snaps(mu), self.basis_sensors(mm), semi=False)
                else:
                    clean_measures[mu, mm] = self.norms.L2innerProd(snaps(mu), self.basis_sensors(mm))
            snap_norms[mu] = self.norms.L2norm(snaps(mu))
        
        errors  = np.zeros((num_rep_exp, Ns, 2))
        
        interps = FunctionsList(self.V)
        resids  = FunctionsList(self.V)
        
        if verbose:
            progressBar = LoopProgress(msg = "Computing PBDW drifted (synthetic) - " + self.name, final = num_rep_exp )
    
        # Assembling PBDW matrix
        assert(len(reg_param) == M)
        sys_matr1 = np.hstack([reg_param * (M * np.eye(M, M)) + self.A[:M, :M], self.K[:M, :N]])
        sys_matr2 = np.hstack([self.K[:M, :N].T, np.zeros((N, N))])
        sys_matr = np.vstack([sys_matr1, sys_matr2])
    
        for kk in range(num_rep_exp):
            for mu in range(Ns):

                # Compute measurement vector with noise
                y = clean_measures[mu].reshape(M,) + np.random.normal(0, noise_value, M).reshape(M,)
            
                # Adding drift
                if mu >= mu_failure:
                    y[idx_failed] += np.random.normal(kappa, rho, len(idx_failed))
                    
                # Creating rhs
                rhs = np.hstack([y, np.zeros((N,))]).flatten()
                
                # Solving the linear system
                coeff = la.solve(sys_matr, rhs)

                # Compute the interpolant and residual
                interpolant = self.basis_sensors.lin_combine(coeff[:M]) + self.basis_functions.lin_combine(coeff[M:])
                residual = np.abs(snaps(mu) - interpolant)
                
                errors[kk, mu, 0] = self.norms.L2norm(residual)
                errors[kk, mu, 1] = errors[kk, mu, 0] / snap_norms[mu]
                
                # Storing interpolant and residual
                if kk + 1 == num_rep_exp:
                    interps.append(interpolant)
                    resids.append(residual)
                
            if verbose:
                progressBar.update(1, percentage = False)
                
        return errors, interps, resids

    def pure_remove_test_err(self, snaps: FunctionsList, N: int, M: int,
                             noise_value : float, reg_param : np.ndarray,
                             idx_failed: list[int], mu_failure: int = 0, verbose = False):
        r"""
        The PBDW online algorithm is used to reconstruct the `snaps` (`FunctionsList`), by solving the modified PBDW linear system: in particular, `idx_failed` measure is removed thus from :math:`\mathbf{y}` the `idx_failed` row is deleted, from :math:`\mathbb{A}` its `idx_failed` row and col are deleted and from :math:`\mathbb{K}` its `idx_failed` row  is deleted.
        
        The interpolant is then defined as the sum over the obtained coefficients from the modified PBDW linear system, without `idx_failed`.
        
        Parameters
        ----------
        snaps : FunctionsList
            Function to reconstruction
        N : int
            Dimension of the reduced space
        M : int
            Number of sensor to use
        noise_value : float
            Standard deviation of the noise, modelled as a normal :math:`\mathcal{N}(0, \sigma^2)`
        reg_param : np.ndarray
            Regularising parameter :math:`\lambda`
        idx_failed : list[int]
            List of integers with the failed sensors
        mu_failure : int, optional (default = 0)
            Index from which failure starts, typically time.
        verbose : boolean, optional (default = False)
            If true, output is printed.
            
        Returns
        ----------
        errors : np.ndarray
            Errors per each element in `snaps` (first column: absolute, second column: relative), measure in :math:`||\cdot ||_{L^2}`.
        interps : FunctionsList
            Interpolant Field :math:`\mathcal{I}_M[u]` of TR-GEIM.
        resids : FunctionsList 
            Residual Field :math:`r_M[u]`, 
        """

        if M > self.Mmax:
            print('The maximum number of measures must not be higher than '+str(self.Mmax)+' --> set equal to '+str(self.Mmax))
            M = self.Mmax

        if N > self.Nmax:
            print('The maximum number of basis functions must not be higher than '+str(self.Nmax)+' --> set equal to '+str(self.Nmax))
            N = self.Nmax

        assert(max(idx_failed) < M)
        
        Ns = len(snaps)
        
        # Computing clean measures and norms of the snapshots
        clean_measures = np.zeros((Ns, M))
        snap_norms = np.zeros((Ns,))
        for mu in range(Ns):
            for mm in range(M):
                if self.is_H1:
                    clean_measures[mu, mm] = self.norms.H1innerProd(snaps(mu), self.basis_sensors(mm), semi=False)
                else:
                    clean_measures[mu, mm] = self.norms.L2innerProd(snaps(mu), self.basis_sensors(mm))
            snap_norms[mu] = self.norms.L2norm(snaps(mu))
        
        errors  = np.zeros((Ns, 2))
        
        interps = FunctionsList(self.V)
        resids  = FunctionsList(self.V)
        
        if verbose:
            progressBar = LoopProgress(msg = "Computing PBDW remove (synthetic) - " + self.name, final = Ns )
    
        # Assembling PBDW matrix
        assert(len(reg_param) == M)
        sys_matr1 = np.hstack([reg_param * (M * np.eye(M, M)) + self.A[:M, :M], self.K[:M, :N]])
        sys_matr2 = np.hstack([self.K[:M, :N].T, np.zeros((N, N))])
        sys_matr = np.vstack([sys_matr1, sys_matr2])
    
        for mu in range(Ns):

            # Compute measurement vector with noise
            y = clean_measures[mu].reshape(M,) + np.random.normal(0, noise_value, M).reshape(M,)
        
            # Creating rhs
            rhs = np.hstack([y, np.zeros((N,))]).flatten()
            
            # If failure the measurement and the matrix are modified
            if mu >= mu_failure:
                _rhs = np.delete(rhs, idx_failed)
                _sys_matrix = np.delete(np.delete(sys_matr, idx_failed, axis=1), idx_failed, axis=0)
            else:
                _rhs = rhs
                _sys_matrix = sys_matr
            
            # Solving the linear system
            coeff = la.solve(_sys_matrix, _rhs)

            # Compute the interpolant and residual
            if mu >= mu_failure:
                interpolant = remove_lin_combine(self.basis_sensors, coeff[:M-1], M, sensI_drifted=idx_failed) + self.basis_functions.lin_combine(coeff[M-1:])
            else:
                interpolant = self.basis_sensors.lin_combine(coeff[:M]) + self.basis_functions.lin_combine(coeff[M:])
            residual = np.abs(snaps(mu) - interpolant)
            
            errors[mu, 0] = self.norms.L2norm(residual)
            errors[mu, 1] = errors[mu, 0] / snap_norms[mu]
            
            # Storing interpolant and residual
            interps.append(interpolant)
            resids.append(residual)
                
            if verbose:
                progressBar.update(1, percentage = False)
                
        return errors, interps, resids

    def gpr_measure_test_err(self, snaps: FunctionsList, N: int, M: int,
                             noise_value : float, reg_param : np.ndarray,
                             ext_sens: FunctionsList, surrogate_model: list, idx_failed: list[int], mu_failure: int = 0, verbose = False):
        r"""
        The PBDW online algorithm is used to reconstruct the `snaps` (`FunctionsList`), by solving the PBDW linear systemwith `idx_failed` measure.
        In order to retrieve information on the "failed measure", a surrogate model (e.g., GPR) has been trained to learn the map from non-failed external measures and the one related to `idx_failed`.
        
        The interpolant is then defined in the standard way.
        
        Parameters
        ----------
        snaps : FunctionsList
            Function to reconstruction
        N : int
            Dimension of the reduced space
        M : int
            Number of sensor to use
        noise_value : float
            Standard deviation of the noise, modelled as a normal :math:`\mathcal{N}(0, \sigma^2)`
        reg_param : np.ndarray
            Regularising parameter :math:`\lambda`
        ext_sens : FunctionsList
            Basis sensors adopted to compute the external measures, input of `surrogate_model`
        surrogate_model : list
            List of all the trained surrogate models
        idx_failed : list[int]
            List of integers with the failed sensors
        mu_failure : int, optional (default = 0)
            Index from which failure starts, typically time.
        verbose : boolean, optional (default = False)
            If true, output is printed.
            
        Returns
        ----------
        errors : np.ndarray
            Errors per each element in `snaps` (first column: absolute, second column: relative), measure in :math:`||\cdot ||_{L^2}`.
        interps : FunctionsList
            Interpolant Field :math:`\mathcal{I}_M[u]` of TR-GEIM.
        resids : FunctionsList 
            Residual Field :math:`r_M[u]`, 
        """

        if M > self.Mmax:
            print('The maximum number of measures must not be higher than '+str(self.Mmax)+' --> set equal to '+str(self.Mmax))
            M = self.Mmax

        if N > self.Nmax:
            print('The maximum number of basis functions must not be higher than '+str(self.Nmax)+' --> set equal to '+str(self.Nmax))
            N = self.Nmax

        assert(max(idx_failed) < M)
        
        Ns = len(snaps)
        
        # Computing clean measures and norms of the snapshots
        clean_measures = np.zeros((Ns, M))
        snap_norms = np.zeros((Ns,))
        for mu in range(Ns):
            for mm in range(M):
                if self.is_H1:
                    clean_measures[mu, mm] = self.norms.H1innerProd(snaps(mu), self.basis_sensors(mm), semi=False)
                else:
                    clean_measures[mu, mm] = self.norms.L2innerProd(snaps(mu), self.basis_sensors(mm))
            snap_norms[mu] = self.norms.L2norm(snaps(mu))
        
        ext_measures = compute_measure(snaps, ext_sens, noise_value=noise_value).T
        
        errors  = np.zeros((Ns, 2))
        
        interps = FunctionsList(self.V)
        resids  = FunctionsList(self.V)
        
        if verbose:
            progressBar = LoopProgress(msg = "Computing PBDW remove (synthetic) - " + self.name, final = Ns )
    
        # Assembling PBDW matrix
        sys_matr1 = np.hstack([reg_param * (M * np.eye(M, M)) + self.A[:M, :M], self.K[:M, :N]])
        sys_matr2 = np.hstack([self.K[:M, :N].T, np.zeros((N, N))])
        sys_matr = np.vstack([sys_matr1, sys_matr2])
    
        for mu in range(Ns):

            # Compute measurement vector with noise
            y = clean_measures[mu].reshape(M,) + np.random.normal(0, noise_value, M).reshape(M,)
            if mu >= mu_failure:    
                for ii, _idx in enumerate(idx_failed):
                    y[_idx] = surrogate_model[ii].predict(ext_measures[mu].reshape(1,-1))[0].flatten()
                    
            # Creating rhs
            rhs = np.hstack([y, np.zeros((N,))]).flatten()
            
            # Solving the linear system
            coeff = la.solve(sys_matr, rhs)

            # Compute the interpolant and residual
            interpolant = self.basis_sensors.lin_combine(coeff[:M]) + self.basis_functions.lin_combine(coeff[M:])
            residual = np.abs(snaps(mu) - interpolant)
            
            errors[mu, 0] = self.norms.L2norm(residual)
            errors[mu, 1] = errors[mu, 0] / snap_norms[mu]
            
            # Storing interpolant and residual
            interps.append(interpolant)
            resids.append(residual)
                
            if verbose:
                progressBar.update(1, percentage = False)
                
        return errors, interps, resids