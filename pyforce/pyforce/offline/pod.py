# Offline Phase: Proper Orthogonal Decomposition
# Author: Stefano Riva, PhD Student, NRG, Politecnico di Milano
# Latest Code Update: 09 September 2024
# Latest Doc  Update: 24 May 2024

import numpy as np
import scipy
import warnings

from dolfinx.fem import (Function, FunctionSpace)
from sklearn.utils.extmath import randomized_svd

from pyforce.tools.backends import norms, LoopProgress
from pyforce.tools.functions_list import *

class POD():
  r"""
    A class to perform the POD on a list of snapshots :math:`u(\mathbf{x};\,\boldsymbol{\mu})` dependent on some parameter :math:`\boldsymbol{\mu}\in\mathcal{P}\subset\mathbb{R}^p`. 
    This class is used for `FunctionsList`, the POD modes are obtained from the eigendecomposition of the correlation matrix :math:`C\in\mathbb{R}^{N_s\times N_s}`
    
    .. math::
        C_{ij} = \left(u(\cdot;\,\boldsymbol{\mu}_i),\,u(\cdot;\,\boldsymbol{\mu}_j)\right)_{L^2}\qquad i,j = 1, \dots, N_s

    .. math::
        C \boldsymbol{\eta_n} = \lambda_n \boldsymbol{\eta_n}\qquad\qquad\qquad n = 1, \dots, N_s
    
    The eigenvalues :math:`\lambda_n` and eigenvectors :math:`\boldsymbol{\eta_n}` are immediately computed.

    Parameters
    ----------
    train_snap : FunctionsList
      List of snapshots onto which the POD is performed.
    name : str
      Name of the field.
    verbose : boolean, optional (Default = False) 
      If `True`, print of the progress is enabled.

    """
  def __init__(self, train_snap: FunctionsList, name: str, 
               svd_acceleration_rank: int = None,
               use_scipy=False, verbose = False) -> None:

    self.Ns   = len(train_snap)
    self.V = train_snap.fun_space
    self.norm = norms(self.V)
    self.name = name

    # Generate the correlation matrix using inner product in L^2
    if svd_acceleration_rank is not None:
      _svd_u, _svd_s, _ = randomized_svd(train_snap.return_matrix(), n_components=svd_acceleration_rank, n_iter='auto')

      # Compute the residual energy and check if the rank is too low or too high
      residual_energy = np.sum(_svd_s[:-1]**2) / np.sum(_svd_s**2)
      if residual_energy <= 0.99:
          print("Warning: The residual energy of the SVD is {} <= 0.9 for rank {}. This may indicate that the rank is too low.".format(residual_energy, svd_acceleration_rank))
          
      _svd_v = _svd_u.T @ train_snap.return_matrix() # shape (svd_acceleration_rank, Ns)

      # Compute the matrix for L2 inner product
      _P_matrix = np.zeros((svd_acceleration_rank, svd_acceleration_rank))

      if verbose:
          progressBar = LoopProgress(msg = "Computing " + self.name + ' correlation matrix', final = svd_acceleration_rank)

      for ii in range(svd_acceleration_rank):
        for jj in range(svd_acceleration_rank):
          if jj >= ii:
            _P_matrix[ii, jj] = self.norm.L2innerProd(_svd_u[:, ii], _svd_u[:, jj])
          else:
            _P_matrix[ii, jj] = _P_matrix[jj, ii]
        progressBar.update(1, percentage = False)

      # Compute the correlation matrix using the SVD
      corrMatrix = _svd_v.T @ _P_matrix @ _svd_v
      assert corrMatrix.shape == (self.Ns, self.Ns), "The correlation matrix has the wrong shape: {}".format(corrMatrix.shape)
    
    else:

      if verbose:
          progressBar = LoopProgress(msg = "Computing " + self.name + ' correlation matrix', final = self.Ns)

      corrMatrix = np.zeros((self.Ns, self.Ns))
      for ii in range(self.Ns):
          for jj in range(self.Ns):
              if jj>=ii:
                  corrMatrix[ii,jj] = self.norm.L2innerProd(train_snap(ii), train_snap(jj))
              else:
                  corrMatrix[ii,jj] = corrMatrix[jj,ii]

          if verbose:
              progressBar.update(1, percentage = False)

    # Solving the eigenvalue problem and sorting the eigenvalue/eigenvector pairs
    if use_scipy:
      eigenvalues, eigenvectors = scipy.linalg.eigh(corrMatrix, subset_by_value=[0,np.inf])
    else:    
      eigenvalues, eigenvectors = np.linalg.eigh(corrMatrix) 
    sorted_indexes = np.argsort( eigenvalues * (-1.) )
    eigenvalues = eigenvalues[sorted_indexes]
    eigenvectors = eigenvectors[:,sorted_indexes]

    # Store the eigenvalue/eigenvector pairs
    self.eigenvalues  = eigenvalues
    self.eigenvectors = eigenvectors

  def GramSchmidt(self, fun: Function) -> np.ndarray:
    r"""
    Perform a step of the Gram-Schmidt process on POD modes :math:`\{\psi_k\}_{k=1}^r` adding `fun` :math:`=f` to enforce the orthonormality in :math:`L^2`

    .. math::
      \psi_{r+1} = f - \sum_{k=1}^r \frac{(f, \psi_k)_{L^2}}{(\psi_k, \psi_k)_{L^2}}\psi_k

    Parameters
    ----------
    fun : Function
      Function to add to the POD basis.

    Returns
    -------
    normalised_fun : Function
      Orthonormalised function :math:`\psi_{r+1}` with respect to the POD basis :math:`\{\psi_k\}_{k=1}^r`.
    """
    ii = len(self.PODmodes)
    
    # Defining the summation term
    _rescaling = list()
    for jj in range(ii+1):
        if jj < ii:
          _rescaling.append( self.norm.L2innerProd(fun, self.PODmodes(jj)) / self.norm.L2innerProd(self.PODmodes(jj), self.PODmodes(jj)) * self.PODmodes(jj) )
    
    # Computing the normalised function
    normalised_fun = fun - sum(_rescaling)
    return normalised_fun / self.norm.L2norm(normalised_fun)
  
    # Deprecated?  
    # for jj in range(ii+1):
    #     if jj < ii:
    #         fun.vector.axpy(- self.norm.L2innerProd(fun, self.PODmodes(jj)) / self.norm.L2innerProd(self.PODmodes(jj), self.PODmodes(jj)),
    #                                                 self.PODmodes.map(jj).vector)
            
    # return fun.x.array[:]  / self.norm.L2norm(fun)
  
  def mode(self, train_snap: FunctionsList, r: int) -> Function:
    r"""
    Computes the `r`-th POD mode, according to the following formula
    
    .. math::
      \psi_{r} (\mathbf{x})= \frac{1}{\lambda_r}\sum_{i=1}^{N_s} \eta_{r, i}\,u(\mathbf{x};\,\boldsymbol{\mu}_i)

    Parameters
    ----------
    train_snap : FunctionsList
      List of snapshots onto which the POD is performed.
    r : int
      Integer input indicating the mode to define.

    """
    return train_snap.lin_combine(self.eigenvectors[:,r] / np.sqrt(self.eigenvalues[r]))

  def compute_basis(self, train_snap: FunctionsList, maxBasis: int, normalise = False) -> None:
    r"""
    Computes the POD modes.
    
    To enforce the orthonormality in :math:`L^2`, the Gram-Schmidt procedure can be used, if the number of modes to be used is high the numerical error in the eigendecomposition may be too large and the orthonormality is lost.

    Parameters
    ----------
    train_snap : FunctionsList
      List of snapshots onto which the POD is performed.
    maxBasis : int
      Integer input indicating the number of modes to define.
    normalise : boolean, optional (Default = False)
      If True, the Gram-Schmidt procedure is used to normalise the POD modes.

    """
    self.PODmodes = FunctionsList(self.V)

    for rankII in range(maxBasis):
      if normalise:
        self.PODmodes.append(self.GramSchmidt(self.mode(train_snap, rankII)))
      else:
       self.PODmodes.append(self.mode(train_snap, rankII))
       
  def projection(self, u: Function, N: int) -> np.ndarray:
    r"""
    The reduced coefficients :math:`\{\alpha_k\}_{k=1}^N` of `u` using `N` modes :math:`\{\psi_k\}_{k=1}^N` are computed using projection in :math:`L_2`, i.e.

    .. math::
      \alpha_k(\boldsymbol{\mu}) = (u(\cdot;\,\boldsymbol{\mu}), \,\psi_k)_{L^2}\qquad k = 1, \dots, N
    
    Parameters
    ----------
    u : Function 
      Function object to project onto the reduced space of dimension `N`.
    N : int
      Dimension of the reduced space, modes to be used.
    
    Returns
    -------
    coeff : np.ndarray
      Modal POD coefficients of `u`, :math:`\{\alpha_k\}_{k=1}^N`.
    """

    # The coefficients are computed using projection in L^2
    coeff = np.zeros((N,))
    for ii in range(N):
      coeff[ii] = self.norm.L2innerProd(u, self.PODmodes(ii)) 

    return coeff

  def train_error(self, train_snap: FunctionsList, maxBasis: int, verbose : bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    The maximum absolute :math:`E_N` and relative :math:`\varepsilon_N` error on the train set is computed, by projecting it onto the reduced space in :math:`L^2`-sense

    .. math::
      E_N = \max\limits_{\boldsymbol{\mu}\in\Xi_{\text{train}}} \left\| u(\mathbf{x};\,\boldsymbol{\mu}) -  \sum_{n=1}^N \alpha_n(\boldsymbol{\mu})\cdot \psi_n(\mathbf{x})\right\|_{L^2}
    .. math::
      \varepsilon_N = \max\limits_{\boldsymbol{\mu}\in\Xi_{\text{train}}} \frac{\left\| u(\mathbf{x};\,\boldsymbol{\mu}) -  \sum_{n=1}^N \alpha_n(\boldsymbol{\mu})\cdot \psi_n(\mathbf{x})\right\|_{L^2}}{\left\| u(\mathbf{x};\,\boldsymbol{\mu})\right\|_{L^2}}
    
    Parameters
    ----------
    train_snap : FunctionsList
      List of snapshots onto which the train error of the POD basis is performed.
    maxBasis : int
      Integer input indicating the maximum number of modes to use.
    verbose : boolean, optional (Default = False) 
      If `True`, print of the progress is enabled.

    Returns
    -------
    maxAbsErr : np.ndarray
      Maximum absolute errors as a function of the dimension of the reduced space.
    maxRelErr : np.ndarray
      Maximum absolute errors as a function of the dimension of the reduced space.
    coeff_matrix : np.ndarray 
      Matrix of the modal coefficients, obtained by projection in :math:`L^2`.
    """
    
    absErr = np.zeros((self.Ns, maxBasis))
    relErr = np.zeros_like(absErr)
    coeff_matrix = np.zeros_like(absErr)

    if verbose:
        progressBar = LoopProgress(msg = "Computing train error " + self.name, final = self.Ns)

    resid = Function(self.V).copy()
    for mu in range(self.Ns):
        
        # Projecting the snapshots onto the reduced space
        coeff = self.projection(train_snap.map(mu), maxBasis)

        for M in range(maxBasis):
          
            # building residual field
            resid.x.array[:] = train_snap(mu) - self.PODmodes.lin_combine(coeff[:M+1])
            absErr[mu,M] = self.norm.L2norm(resid)
            relErr[mu,M] = absErr[mu,M] / self.norm.L2norm(train_snap(mu))

        coeff_matrix[mu, :] = coeff[:]
        if verbose:
            progressBar.update(1, percentage = False)

    return absErr.max(axis = 0), relErr.max(axis = 0), coeff_matrix

  def test_error(self, test_snap: FunctionsList, maxBasis: int, verbose : bool = False):
    r"""
    The average absolute :math:`E_N` and relative :math:`\varepsilon_N` error on the test set is computed, by projecting it onto the reduced space in :math:`L^2`-sense

    .. math::
      E_N = \left\langle \left\| u(\mathbf{x};\,\boldsymbol{\mu}) -  \sum_{n=1}^N \alpha_n(\boldsymbol{\mu})\cdot \psi_n(\mathbf{x})\right\|_{L^2} \right\rangle_{\boldsymbol{\mu}\in\Xi_{\text{test}}}
    .. math::
      \varepsilon_N =\left\langle \frac{\left\| u(\mathbf{x};\,\boldsymbol{\mu}) -  \sum_{n=1}^N \alpha_n(\boldsymbol{\mu})\cdot \psi_n(\mathbf{x})\right\|_{L^2}}{\left\| u(\mathbf{x};\,\boldsymbol{\mu})\right\|_{L^2}} \right\rangle_{\boldsymbol{\mu}\in\Xi_{\text{test}}}
    
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
        progressBar = LoopProgress(msg = "Computing POD test error (projection) - " + self.name, final = Ns_test)

    resid = Function(self.V).copy()
    for mu in range(Ns_test):
      
        # Projecting the snapshots onto the reduced space
        coeff = self.projection(test_snap.map(mu), maxBasis)

        for M in range(maxBasis):
            # building residual field
            resid.x.array[:] = test_snap(mu) - self.PODmodes.lin_combine(coeff[:M+1])
            absErr[mu,M] = self.norm.L2norm(resid)
            relErr[mu,M] = absErr[mu,M] / self.norm.L2norm(test_snap(mu))

        coeff_matrix[mu, :] = coeff[:]
        if verbose:
           progressBar.update(1, percentage = False)

    meanAbsErr = absErr.mean(axis = 0)
    meanRelErr = relErr.mean(axis = 0)
    return meanAbsErr, meanRelErr, coeff_matrix
  

# discete POD - the SVD of the full snapshot matrix is used
class DiscretePOD():
  r"""
    A class to perform the POD on a list of snapshots :math:`u(\mathbf{x};\,\boldsymbol{\mu})` dependent on some parameter :math:`\boldsymbol{\mu}`. 
    This class is used for several kind of inputs (`FunctionsList`, vectors, images, matrices...).

    The snapshots are represented by a matrix :math:`\mathbb{S}\in\mathbb{R}^{\mathcal{N}_h\times N_s}`, such that

    .. math::
        \mathbb{S}_{ij} = u(\mathbf{x}_i;\,\boldsymbol{\mu}_j)
    
    in which the dependence on :math:`\mathbf{x}_i` can be a true spatial dependence or the dofs of a matrix/image.

    The basis functions are computed using the `svd`, i.e.
    
    .. math::
        \mathbb{U}, \Sigma, \mathbb{V}^\dagger = \text{svd}(\mathbb{S})

    The basis functions are orthogonal in :math:`l_2` sense, hence the matrix containing the modes is orthogonal.

    Parameters
    ----------
    train_snap : FunctionsMatrix or FunctionsList
      List of snapshots onto which the POD is performed.
    name : str
      Name of the field.
    Nmax : int, optional (default=None)
      If `None` the full matrices are stored, else only the first `Nmax`.
    random : bool, optional (default = False)
      If True and if `Nmax` is provided, the randomised SVD is used.
    """
  def __init__(self, train_snap: FunctionsList, name: str, Nmax = None, random = False) -> None:

    self.Ns = len(train_snap)
    self.Nh = len(train_snap(0))
    self.name = name
    
    self.modes = FunctionsList(dofs = train_snap.fun_shape)

    # Performing SVD of the snapshot matrix
    if random and Nmax is not None:
      U, Sigma, V_T  = randomized_svd(train_snap.return_matrix(), n_components=Nmax, n_iter='auto')
    else:
      U, Sigma, V_T = np.linalg.svd(train_snap.return_matrix(), full_matrices=False)
    if sum(Sigma < 0) > 0: 
        warnings.warn("Check singular values: some of them are negative!")   

    if Nmax is None:
      self.Nmax = len(Sigma)
    else:
      self.Nmax = Nmax

    self.sing_vals = Sigma[:self.Nmax]
    self.Vh_train = V_T[:self.Nmax, :] # shape (Nmax, Ns)

    # Computing POD basis
    self.U = U
    for rankII in range(self.Nmax):
        self.modes.append(U[:, rankII]) # shape (Nh, Nmax)
        
       
  def projection(self, snap: np.ndarray, N: int = None):
    r"""
    The reduced coefficients :math:`\mathbf{V}^\dagger_\star\in\mathbb{R}^{N\times N_\star}` of `snap`:math:`=\mathbb{S}_\star\in\mathbb{R}^{\mathcal{N}_h\times N_\star}` using `N` modes :math:`\mathbb{U}\in\mathbb{R}^{\mathcal{N}_h\times N}` are computed using projection in :math:`l_2`, i.e.

    .. math::
      \mathbf{V}^\dagger_\star = \Sigma^{-1}\mathbb{U}^T\mathbb{S}_\star
    
    Parameters
    ----------
    snap : np.ndarray 
      Matrix object to project onto the reduced space of dimension `N`. Must be :math:`(\mathcal{N}_h, N_\star)`.
    N : int, optional (default = None)
      Dimension of the reduced space, modes to be used. If `None` all the modes are used.
    
    Returns
    -------
    coeff : np.ndarray
      Modal POD coefficients of `snap`, :math:`\mathbf{V}^\dagger_\star`.
    """

    if N is None:
      N = self.Nmax
    
    if isinstance(snap, FunctionsList):
      _snap = snap.return_matrix()
    else:
      _snap = snap
    
    Vh_star = np.dot(np.linalg.inv(np.diag(self.sing_vals[:N])), 
                     np.dot(self.modes.return_matrix().T[:N], _snap))
    
    return Vh_star
  
  def reconstruct(self, Vh_star: np.ndarray):
    r"""
    The reduced coefficients :math:`\mathbf{V}^\dagger_\star\in\mathbb{R}^{N\times N_\star}` are used to decode into the Full Order space :math:`\mathbb{R}^{\mathcal{N}_h}` using `N` modes :math:`\mathbb{U}\in\mathbb{R}^{\mathcal{N}_h\times N}`.

    .. math::
      \mathbb{S}_\star = \mathbb{U}\Sigma\mathbf{V}^\dagger_\star
    
    Parameters
    ----------
    Vh_star : np.ndarray 
      Matrix object containing the POD coefficients. Must be :math:`(N, N_\star)`.
    
    Returns
    -------
    snaps : np.ndarray
      Reconstructed field returned as an element of :math:`\mathbf{R}^{\mathcal{N_h}\times N_\star}`.
    """
    
    N = len(Vh_star)
    assert( N <= self.Nmax )
    
    return np.dot(self.modes.return_matrix()[:, :N] * self.sing_vals[:N], Vh_star)

  def train_error(self, train_snap: FunctionsList, maxBasis: int, verbose = False):
    r"""
    The maximum absolute :math:`E_N` and relative :math:`\varepsilon_N` error on the train set is computed, by projecting it onto the reduced space in :math:`l^2`-sense

    .. math::
      E_N = \max\limits_{\boldsymbol{\mu}\in\Xi_{\text{train}}} \left\| u(\mathbf{x};\,\boldsymbol{\mu}) -  \sum_{n=1}^N \alpha_n(\boldsymbol{\mu})\cdot \psi_n(\mathbf{x})\right\|_{2}
    .. math::
      \varepsilon_N = \max\limits_{\boldsymbol{\mu}\in\Xi_{\text{train}}} \frac{\left\| u(\mathbf{x};\,\boldsymbol{\mu}) -  \sum_{n=1}^N \alpha_n(\boldsymbol{\mu})\cdot \psi_n(\mathbf{x})\right\|_{2}}{\left\| u(\mathbf{x};\,\boldsymbol{\mu})\right\|_{2}}
    
    The POD coefficients used are the ones obtained by the SVD during the initialisation.

    Parameters
    ----------
    train_snap : FunctionsMatrix or FunctionsList
      List of snapshots to compute errors
    maxBasis : int
      Integer input indicating the maximum number of modes to use.
    verbose : boolean, optional (Default = False) 
      If `True`, print of the progress is enabled.

    Returns
    -------
    maxAbsErr : np.ndarray
      Maximum absolute errors as a function of the dimension of the reduced space.
    maxRelErr : np.ndarray
      Maximum absolute errors as a function of the dimension of the reduced space.
      
    """
    
    assert(maxBasis <= self.Nmax)
    
    absErr = np.zeros((self.Ns, maxBasis))
    relErr = np.zeros_like(absErr)

    if verbose:
        progressBar = LoopProgress(msg="Computing train error - "+self.name, final = self.Ns)
      
    for mu in range(self.Ns):
      norm = np.linalg.norm(train_snap(mu))
      for rank in range(maxBasis):
        recon = self.reconstruct(self.Vh_train[:rank+1, mu])
        absErr[mu, rank] = np.linalg.norm(recon - train_snap(mu))
        relErr[mu, rank] = absErr[mu, rank] / norm
      
      if verbose:
        progressBar.update(1, percentage=False)
        
    return absErr.max(axis = 0), relErr.max(axis = 0)

  def test_error(self, test_snap: FunctionsList, maxBasis: int, verbose = False):
    r"""
    The maximum absolute :math:`E_N` and relative :math:`\varepsilon_N` error on the train set is computed, by projecting it onto the reduced space in :math:`l^2`-sense

    .. math::
      E_N = \max\limits_{\boldsymbol{\mu}\in\Xi_{\text{train}}} \left\| u(\mathbf{x};\,\boldsymbol{\mu}) -  \sum_{n=1}^N \alpha_n(\boldsymbol{\mu})\cdot \psi_n(\mathbf{x})\right\|_{2}
    .. math::
      \varepsilon_N = \max\limits_{\boldsymbol{\mu}\in\Xi_{\text{train}}} \frac{\left\| u(\mathbf{x};\,\boldsymbol{\mu}) -  \sum_{n=1}^N \alpha_n(\boldsymbol{\mu})\cdot \psi_n(\mathbf{x})\right\|_{2}}{\left\| u(\mathbf{x};\,\boldsymbol{\mu})\right\|_{2}}
    
    The POD coefficients used are the ones obtained by projection of `test_snap`.

    Parameters
    ----------
    test_snap : FunctionsMatrix or FunctionsList
      List of snapshots to project and compute errors
    maxBasis : int
      Integer input indicating the maximum number of modes to use.
    verbose : boolean, optional (Default = False) 
      If `True`, print of the progress is enabled.

    Returns
    -------
    maxAbsErr : np.ndarray
      Maximum absolute errors as a function of the dimension of the reduced space.
    maxRelErr : np.ndarray
      Maximum absolute errors as a function of the dimension of the reduced space.
      
    """
    
    assert(maxBasis <= self.Nmax)
    
    absErr = np.zeros((self.Ns, maxBasis))
    relErr = np.zeros_like(absErr)

    if verbose:
        progressBar = LoopProgress(msg="Computing train error - "+self.name, final = self.Ns)
      
    for mu in range(self.Ns):
      norm = np.linalg.norm(test_snap(mu))
      for rank in range(maxBasis):

        # Obtain the POD coefficient by projection
        vh = self.projection(test_snap(mu), rank+1)

        # Reconstruct the field
        recon = self.reconstruct(vh)

        # Compute the error
        absErr[mu, rank] = np.linalg.norm(recon - test_snap(mu))
        relErr[mu, rank] = absErr[mu, rank] / norm
      
      if verbose:
        progressBar.update(1, percentage=False)
        
    return absErr.max(axis = 0), relErr.max(axis = 0)