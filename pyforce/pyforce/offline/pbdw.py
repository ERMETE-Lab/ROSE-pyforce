# Offline Phase: Parameterised-Background Data-Weak
# Author: Stefano Riva, PhD Student, NRG, Politecnico di Milano
# Latest Code Update: 12 October 2023
# Latest Doc  Update: 18 July 2023

import numpy as np
import scipy.linalg as la

from pyforce.tools.backends import norms
from pyforce.tools.functions_list import FunctionsList
from .sensors import gaussian_sensors
    
# PBDW: computing inf-sup
class PBDW():
    r"""
    A class implementing the *a priori* error analysis of the Parameterised-Background Data-Weak (PBDW) formulation, given a list of sensors' Riesz representation :math:`\{g_m\}_{m=1}^M` for the update space and basis functions :math:`\{\zeta_n\}_{n=1}^N` for the reduced one.
    In particular, the following matrices are defined :math:`(n,n' = 1,\dots,N)` and :math:`(m,m' = 1,\dots,M)` 
    
    .. math::
        \mathbb{A}_{mm'}=\left(g_m,\,g_{m'}\right)_{\mathcal{U}}
    .. math::
        \mathbb{K}_{mn}=\left(g_m,\,\zeta_{n}\right)_{\mathcal{U}}
    .. math::
        \mathbb{Z}_{nn'}=\left(\zeta_{n},\,\zeta_{n'}\right)_{\mathcal{U}}
      
    given :math:`\mathcal{U}` as the functional space.
    
    
    Parameters
    ----------
    basis_functions : FunctionsList
        List of functions spanning the reduced space
    basis_sensors : FunctionsList
        List of sensors representation spanning the update space
    name : str
        Name of the snapshots (e.g., temperature T)

    """
    def __init__(self, basis_functions: FunctionsList, basis_sensors: FunctionsList, name: str) -> None:
    
        # store basis function and basis sensors
        self.basis_functions = FunctionsList(basis_functions.fun_space)
        self.basis_sensors = FunctionsList(basis_sensors.fun_space)
        
        self.basis_functions._list = basis_functions.copy()
        self.basis_sensors._list   = basis_sensors.copy()
        
        self.V = basis_functions.fun_space
        
        # Defining the norm class to make scalar products and norms
        self.norms = norms(self.V)
        self.name = name

        N = len(basis_functions)
        M = len(basis_sensors)

        # A_{ii,jj} = (basis_sensors[ii], basis_sensors[jj])_L2
        self.A = np.zeros((M,M))
        for ii in range(M):
            for jj in range(M):
                if jj>=ii:
                    self.A[ii, jj] = self.norms.L2innerProd(basis_sensors(ii), basis_sensors(jj))
                else:
                    self.A[ii,jj] = self.A[jj, ii]
        
        # K_{ii,jj} = (basis_sensors[ii], basis_functions[jj])_L2
        self.K = np.zeros((M, N))
        for ii in range(M):
            for jj in range(N):
                self.K[ii, jj] = self.norms.L2innerProd(basis_sensors(ii), basis_functions(jj))

        # Z_{ii, jj} = (basis_functions[ii], basis_functions[jj])_L2
        self.Z = np.zeros((N,N))
        for ii in range(N):
            for jj in range(N):
                if jj>=ii:
                    self.Z[ii, jj] = self.norms.L2innerProd(basis_functions(ii), basis_functions(jj))
                else:
                    self.Z[ii,jj] = self.Z[jj, ii]

        self.Nmax = N
        self.Mmax = M

    def compute_infsup(self, N: int, M: int):
        r"""
        Compute the inf-sup constant :math:`\beta_{N,M}` for the couple basis functions (dimension :math:`N`) - basis sensors (dimension :math:`M`). It's the square root of the minimum eigenvalue of the following eigenvalue problem
        
        .. math::
            \mathbb{K}^T\mathbb{A}^{-1}\mathbb{K}\mathbf{z}_k = \lambda_k \mathbb{Z}\mathbf{z}_k \qquad\Longrightarrow\qquad
            \beta_{N,M} = \min\limits_{k=1,\dots,N} \sqrt{\lambda_k}

        Parameters
        ----------
        N : int
            Dimension of the reduced space to use
        M : int
            Dimension of the update space to use

        Returns
        ----------
        inf_sup : np.ndarray
            Inf-sup constant :math:`\{\beta_{N,m}\}_{m=1}^M` (fixed :math:`N`), measuring how good the update space spanned by basis sensors is.
        """

        assert (N <= self.Nmax)
        assert (M <= self.Mmax)

        if N > M:
            print('The number of basis functions is higher than the basis sensors: the inf-sup is identically null!')
            print('Lower the dimension of the reduced space, the PBDW may be unstable')

        else:
            inf_sup = np.zeros((M,))
            
            for m in range(M):
                schurComplement = np.matmul(self.K[:m+1, :N+1].T, np.matmul(np.linalg.inv(self.A[:m+1, :m+1]), self.K[:m+1, :N+1]))

                beta_squared, _ = la.eigh(schurComplement, self.Z[:N+1, :N+1])

                inf_sup[m] =  np.sqrt(min(abs(beta_squared)))

        return inf_sup