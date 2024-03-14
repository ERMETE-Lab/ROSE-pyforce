# Offline Phase: Weak-Greedy Algorithm
# Author: Stefano Riva, PhD Student, NRG, Politecnico di Milano
# Latest Code Update: 05 March 2024
# Latest Doc  Update: 05 March 2024

import numpy as np
import warnings

from dolfinx.fem import (Function, FunctionSpace)

from pyforce.tools.backends import norms, LoopProgress
from pyforce.tools.functions_list import FunctionsList

# WeakGreedy offline + online (projection)
class WeakGreedy():
  def __init__(self, V: FunctionSpace, name: str) -> None:
    r"""
    A class to perform the WeakGreedy algorithm on a list of snapshots :math:`u(\mathbf{x};\,\boldsymbol{\mu})` dependent on some parameter :math:`\boldsymbol{\mu}\in\mathcal{P}\subset\mathbb{R}^p`.

    Parameters
    ----------
    V: FunctionSpace
        Functional space of the snapshots
    name : str
        Name of the field.
    """
    self.V = V
    self.norm = norms(self.V)
    self.name = name

  def GrahmSchmidt(self, fun: Function):
    r"""
    Perform a step of the Gram-Schmidt process on basis functions :math:`\{\psi_k\}_{k=1}^r` adding `fun` :math:`=f` to enforce the orthonormality in :math:`L^2`

    .. math::
      \psi_{r+1} = f - \sum_{k=1}^r \frac{(f, \psi_k)_{L^2}}{(\psi_k, \psi_k)_{L^2}}\psi_k

    Parameters
    ----------
    fun : Function
      Function to add to the WeakGreedy basis.

    Returns
    -------
    normalised_fun : Function
      Orthonormalised function :math:`\psi_{r+1}` with respect to the WeakGreedy basis :math:`\{\psi_k\}_{k=1}^r`.
    """
    ii = len(self.basis)
    
    # Defining the summation term
    _rescaling = list()
    for jj in range(ii+1):
        if jj < ii:
          _rescaling.append( self.norm.L2innerProd(fun, self.basis(jj)) / self.norm.L2innerProd(self.basis(jj), self.basis(jj)) * self.basis(jj) )
    
    # Computing the normalised function
    normalised_fun = fun - sum(_rescaling)
    return normalised_fun / self.norm.L2norm(normalised_fun)
  
    # Deprecated?  
    # for jj in range(ii+1):
    #     if jj < ii:
    #         fun.vector.axpy(- self.norm.L2innerProd(fun, self.basis(jj)) / self.norm.L2innerProd(self.basis(jj), self.basis(jj)),
    #                                                 self.basis.map(jj).vector)
            
    # return fun.x.array[:]  / self.norm.L2norm(fun)
    
  def projection(self, u: Function, maxBasis: int):
    r"""
    The reduced coefficients :math:`\{\alpha_k\}_{k=1}^N` of `u` using `N` basis functions :math:`\{\psi_k\}_{k=1}^N` are computed using projection in :math:`L_2`, i.e.

    .. math::
      \alpha_k(\boldsymbol{\mu}) = (u(\cdot;\,\boldsymbol{\mu}), \,\psi_k)_{L^2}\qquad k = 1, \dots, N
    
    Parameters
    ----------
    u : Function 
      Function object to project onto the reduced space of dimension `N`.
    maxBasis : int
      Dimension of the reduced space, modes to be used.
    
    Returns
    -------
    coeff : np.ndarray
      Reduced coefficients of `u`, :math:`\{\alpha_k\}_{k=1}^N`.
    """
    
    # Compute the coefficients of the reduced space
    coeff = np.zeros((maxBasis,))
    for ii in range(maxBasis):
      coeff[ii] = self.norm.L2innerProd(u, self.basis(ii)) 

    return coeff

  def compute_basis(self, train_snap: FunctionsList, N: int, verbose = False):
    r"""
    Computes the WeakGreedy (WG) basis functions (orthonormalised using Grahm-Schmidt), as the set of snapshots that minimises the reconstruction error.

    Parameters
    ----------
    train_snap : FunctionsList
        List of snapshots onto which the WG algorithm is performed.
    N : int
        Integer input indicating the maximum number of modes to define

    Returns
    ----------
    maxAbsErr
        Maximum absolute error measured in :math:`L^2`
    maxRelErr
        Maximum relative error measured in :math:`L^2`
    alpha_coeff
        Matrix of the reduced coefficients, obtained by the :math:`L^2` projection
    """

    # Number of snaphots
    self.Ns   = len(train_snap)

    self.basis = FunctionsList(self.V)
    snapNormList = []
    maxNorm = 0.

    # find first generating function
    for mu in range(len(train_snap)):
        tmpNorm = self.norm.L2norm(train_snap(mu))
        snapNormList.append(tmpNorm)
        if maxNorm < tmpNorm:
            maxNorm = tmpNorm
            generatingIdx = mu
    self.basis.append(train_snap(generatingIdx) / maxNorm)
    # Deprecated?
    # tmp = Function(self.V).copy()
    # tmp.x.array[:] = train_snap(generatingIdx) / maxNorm
    
    # Create the variables to store the data
    alpha_coeff = np.zeros((len(train_snap), N))
    maxAbsErr = np.zeros((N,))
    maxRelErr = np.zeros_like(maxAbsErr)

    # Deprecated?
    resid = Function(self.V).copy()
    
    for nn in range(1, N+1):
        max_abs_err = 0.
        max_rel_err = 0.

        for mu in range(len(train_snap)):
          
            # Define the projected coefficient
            coeff = self.projection(train_snap(mu), nn)
            alpha_coeff[mu, :nn] = coeff

            # Compute residual field
            resid = train_snap(mu) - self.basis.lin_combine(coeff)
            # Deprecated?
            # resid.x.array[:] = train_snap(mu) - self.basis.lin_combine(coeff)
            abs_err = self.norm.L2norm(resid)
            rel_err = abs_err / snapNormList[mu]

            if max_abs_err < abs_err:
                max_abs_err = abs_err
                generatingIdx = mu

            if max_rel_err < rel_err:
                max_rel_err = rel_err
            

        maxAbsErr[nn-1] = max_abs_err
        maxRelErr[nn-1] = max_rel_err

        # Print output
        if verbose:
            print(f'  Iteration {nn+0:03} | Abs Err: {max_abs_err:.2e} | Rel Err: {max_rel_err:.2e}', end = "\r")

        # Generate the next basis function (with orthonormalisation)
        if nn < N + 1:
            # Deprecated?
            # tmp = Function(self.V).copy()
            # tmp.x.array[:] = train_snap(generatingIdx)
            # self.GrahmSchmidt(tmp)
            # self.basis.append(tmp)
            self.basis.append( self.GrahmSchmidt(train_snap(generatingIdx)) )

    return maxAbsErr, maxRelErr, alpha_coeff
        

  def test_error(self, test_snap: FunctionsList, maxBasis: int, verbose = False):
    r"""
    The maximum absolute :math:`E_N` and relative :math:`\varepsilon_N` error on the test set is computed, by projecting it onto the reduced space in :math:`L^2`-sense

    .. math::
      E_N = \max\limits_{\boldsymbol{\mu}\in\Xi_{\text{test}}} \left\| u(\mathbf{x};\,\boldsymbol{\mu}) -  \sum_{n=1}^N \alpha_n(\boldsymbol{\mu})\cdot \psi_n(\mathbf{x})\right\|_{L^2}
    .. math::
      \varepsilon_N = \max\limits_{\boldsymbol{\mu}\in\Xi_{\text{test}}} \frac{\left\| u(\mathbf{x};\,\boldsymbol{\mu}) -  \sum_{n=1}^N \alpha_n(\boldsymbol{\mu})\cdot \psi_n(\mathbf{x})\right\|_{L^2}}{\left\| u(\mathbf{x};\,\boldsymbol{\mu})\right\|_{L^2}}
    
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
    meanAbsErr : np.ndarray
      Maximum absolute errors as a function of the dimension of the reduced space.
    maxRelErr : np.ndarray
      Maximum absolute errors as a function of the dimension of the reduced space.
    coeff_matrix : np.ndarray 
      Matrix of the modal coefficients, obtained by projection in :math:`L^2`.
    """
    
    Ns_test = len(test_snap)
    absErr = np.zeros((Ns_test, maxBasis))
    relErr = np.zeros_like(absErr)
    coeff_matrix = np.zeros_like(absErr)

    if verbose:
        progressBar = LoopProgress(msg = "Computing WeakGreedy test error (projection) - " + self.name, final = Ns_test)

    resid = Function(self.V).copy()
    for mu in range(Ns_test):

        coeff = self.projection(test_snap(mu), maxBasis)

        for M in range(maxBasis):
            # building residual field
            resid.x.array[:] = test_snap(mu) - self.basis.lin_combine(coeff[:M+1])
            absErr[mu,M] = self.norm.L2norm(resid)
            relErr[mu,M] = absErr[mu,M] / self.norm.L2norm(test_snap(mu))

        coeff_matrix[mu, :] = coeff[:]
        if verbose:
            progressBar.update(1, percentage = False)

    meanAbsErr = absErr.mean(axis = 0)
    meanRelErr = relErr.mean(axis = 0)
    return meanAbsErr, meanRelErr, coeff_matrix