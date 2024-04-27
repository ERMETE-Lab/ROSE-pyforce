# Offline Phase: sensors positioning
# Author: Stefano Riva, PhD Student, NRG, Politecnico di Milano
# Latest Code Update: 25 April 2024
# Latest Doc  Update: 25 April 2024

import numpy as np
import scipy
from dolfinx import fem
import warnings

import ufl
from dolfinx.fem import (Function, FunctionSpace, assemble_scalar, form)
from ufl import dx, grad, inner, dot
from petsc4py import PETSc

from pyforce.tools.backends import norms, LoopProgress
from pyforce.tools.functions_list import FunctionsList

# Class to define gaussian sensors with a Riesz representation in L2
class gaussian_sensors():
  r"""
    A class to define normalised gaussian sensors in terms of functional to mimic measurements of scalar fields.
    The measurement procedure is described through a linear functional with gaussian kernel, i.e.

    .. math::
        v_k = v(u;\,\mathbf{x}_k,s) = \int_\Omega u(\mathbf{x})\cdot g(\mathbf{x};\,\mathbf{x}_k,s)\,d\Omega 
        \qquad
        \text{ given } g(\mathbf{x};\,\mathbf{x}_k,s) = 
        \frac{e^{-\frac{\|{\mathbf{x}-\mathbf{x}_k}\|_2^2}{2s^2}}}{\displaystyle\int_\Omega e^{-\frac{\|{\mathbf{x}-\mathbf{x}_k}\|_2^2}{2s^2}}\,d\Omega}
        
    such that :math:`v(1;\,\mathbf{x}_k,s) = 1`.

    Parameters
    ----------
    domain : dolfinx.mesh
        Mesh onto which the sensors are defined.
    V : FunctionSpace
        Functional space onto which the kernel of the sensors are interpolated.
    s : float
        Standard deviation of the gaussian kernel.
    assemble_riesz : boolean, optional (Default: False)
        Logic variable indicating whether or not to assemble the variational forms for the Riesz representations.

  """
  def __init__(self, domain, V: FunctionSpace, s: float, assemble_riesz: bool = False) -> None:
    self.domain = domain
    self.V = V
    self.s = s # standard deviation of the guassian

    self.norms = norms(self.V)
    self.g = Function(self.V).copy()

    if assemble_riesz:
        self.trial = ufl.TrialFunction(self.V)
        self.test  = ufl.TestFunction(self.V)

        self.riesz_lhs = ( dot(grad(self.trial), grad(self.test)) + dot(self.trial, self.test) ) * dx
        self.riesz_rhs = dot(self.g, self.test) * dx

        self.riesz_bilin = form(self.riesz_lhs)
        self.riesz_lin   = form(self.riesz_rhs)

        self.riesz_A = fem.petsc.assemble_matrix(self.riesz_bilin)
        self.riesz_A.assemble()
        self.riesz_b = fem.petsc.create_vector(self.riesz_lin)

        self.solver = PETSc.KSP().create(self.domain.comm)
        self.solver.setOperators(self.riesz_A)
        self.solver.setType(PETSc.KSP.Type.PREONLY)
        self.solver.getPC().setType(PETSc.PC.Type.LU)

  def define(self, x_m: np.ndarray) -> Function:
    r"""
    Given a position :math:`x_m` defines the kernel function of the sensor as a Gaussian with given point spread `s`
    
    .. math::
        g(\mathbf{x};\,\mathbf{x}_m,s) = 
        \frac{e^{-\frac{\|{\mathbf{x}-\mathbf{x}_m}\|_2^2}{2s^2}}}{\displaystyle\int_\Omega e^{-\frac{\|{\mathbf{x}-\mathbf{x}_m}\|_2^2}{2s^2}}\,d\Omega}
    
    This kernel function is the Riesz representation :math:`q_m` in :math:`L^2` of the functional, i.e.
    
    .. math::
        (q_m, \varphi)_{L^2} =
        \int_\Omega q_m\cdot \varphi \,d\Omega = v_m(\varphi)\qquad \forall \varphi\in L^2

    """
    Gaussian = self.g.copy()
    # normGaussian = self.g.copy()
    
    Gaussian.interpolate(    lambda x:  np.exp( - ((x[0]-x_m[0])**2+(x[1]-x_m[1])**2+(x[2]-x_m[2])**2) / 2 / self.s**2)  )
    # normGaussian.interpolate(lambda x: (np.exp( - ((x[0]-x_m[0])**2+(x[1]-x_m[1])**2+(x[2]-x_m[2])**2) / 2 / self.s**2) ) 
    #                                     / self.norms.integral(Gaussian) )
    
    normGaussian = Gaussian.x.array[:] / self.norms.integral(Gaussian)
    
    return normGaussian

  def define_riesz(self, x_m: np.ndarray) -> Function:
    r"""
    Given a position :math:`x_m` defines the kernel function :math:`q_m` of the sensor as the Riesz representation in :math:`H^1` of the functional, i.e.
    
    .. math::
        (q_m, \varphi)_{H^1} =
        \int_\Omega q_m\cdot \varphi \,d\Omega + \int_\Omega \nabla q_m\cdot \nabla \varphi \,d\Omega
        = v_m(\varphi)\qquad \forall \varphi\in H^1

    """
    self.g.x.array[:] = self.define(x_m)

    with self.riesz_b.localForm() as loc_b:
        loc_b.set(0)
    fem.petsc.assemble_vector(self.riesz_b, self.riesz_lin)

    representation = self.g.copy()
    self.solver(self.riesz_b, representation.vector)
    representation.x.scatter_forward()
    return representation

  def create(self, xm : list = None, sampleEvery: int = 10, is_H1 : bool = False, verbose = False) -> FunctionsList:
    r"""
    This function creates the list of sensors (using Riesz representation either in :math:`L^2` or in :math:`H^1`), either by sampling from the mesh or from a prescribed list of positions.

    Parameters
    ----------
    xm : list, optional (Default: None)
        Possible List of positions for the sensors inside the domain.
    sampleEvery : int, optional (Default: 10)
        Integers indicating sampling rate to be used on the mesh, if `x_m` is `None`.
    is_H1 : bool, optional (Default: False)
        If `True`, the Riesz representation in :math:`H^1` is used.
    verbose : boolean, optional (Default = False)
        If `True`, printing is enabled.

    Returns
    -------
    sens_list : FunctionsList
        List of kernel functions using a Riesz representation in :math:`L^2`
    """

    sens_list = FunctionsList(self.V)
    
    if xm is None:
        geom_dofs = self.domain.geometry.x
        mSample = np.arange(0, len(geom_dofs), sampleEvery) # the cells on the boundary are avoided
        self.xm_list = []
        if verbose:
            progressBar = LoopProgress(msg = "Generating sensors (sampled every "+str(sampleEvery)+" cells)", final = len(mSample))
        for m in mSample:
            x_m = geom_dofs[m]
            self.xm_list.append(x_m)
    else:
        if verbose:
            progressBar = LoopProgress(msg = "Generating sensors (coinstrained cells)", final = len(xm))
        self.xm_list = xm.copy()
    
    for m in range(len(self.xm_list)):
        if is_H1:
            sens_list.append(self.define_riesz(self.xm_list[m]))
        else:
            sens_list.append(self.define(self.xm_list[m]))
        if verbose:
            progressBar.update(1, percentage=False)
        
    return sens_list

  def action_single(self, fun: Function, sensor: Function):
    r"""
    Given an input function `fun` :math:`=f` and a sensors :math:`v_m`, the action of the sensor is applied to the function, as an inner product in :math:`L^2`:

    .. math::
        y_m = v_m\left(f\right)

    Parameters
    ----------
    fun : Function
        Function onto which the action is applied.
    sensor : Function
        Riesz representation of the sensor
    
    Returns
    -------
    measure : float
        Scalar :math:`y_m` with the measure of the function with respect to the sensor :math:`v_m`.
    """
        
    measure = self.norms.L2innerProd(fun, sensor)

    return measure

  def action(self, fun: Function, sens: FunctionsList):
    r"""
    Given an input function `fun` :math:`=f` and a list of sensors :math:`\{v_m\}_{m=1}^M`, the action of the sensor is applied to the function, as an inner product in :math:`L^2`:

    .. math::
        y_m = v_m\left(f\right)\qquad m = 1, \dots, M

    Parameters
    ----------
    fun : Function
        Function onto which the action is applied.
    sens : FunctionsList
        List of available sensors
    
    Returns
    -------
    measure : np.ndarray
        Vector :math:`\mathbf{y}\in\mathbb{R}^M`, whose dimension is equal to the number of input sensors.
    """
    measure = np.zeros((len(sens),))
    for sensI in range(len(sens)):
      measure[sensI] = self.norms.L2innerProd(fun, sens(sensI))

    return measure
  
# SGREEDY algorithm for sensor selection (both representation in L2 and H1)
class SGREEDY(): # to be extended when inf-sup > tol !!!
  r"""
    A class to perform the SGREEDY algorithm, given a list of basis functions :math:`\{\phi_n\}_{n=1}^N`.

    Parameters
    ----------
    domain : dolfinx.mesh
        Mesh for the sensor placement.
    basis : FunctionsList
        List of basis functions :math:`\{\phi_n\}_{n=1}^N`, previously generated.
    V : FunctionSpace
        Functional space of the functions.
    name : str
        Name of the snapshots (e.g., temperature T)
    s : float
        Standard deviation of the gaussian kernel for the sensors

  """
  def __init__(self, domain, basis: FunctionsList, V: FunctionSpace, name: str, s: float) -> None:

    self.basis = basis
    self.V = V
    self.name = name
    self.domain = domain

    # Generate sensor library
    self.sens_class = gaussian_sensors(domain, self.V, s, assemble_riesz = True)

  def generate(self, N: int, Mmax: int, tol: float = 0.2,
               xm : list = None, sampleEvery : int = 10, is_H1 : bool = False, verbose = False):
    r"""
    Selection of sensors position with a Riesz representation :math:`\{g_m\}_{m=1}^M` either in :math:`L^2` or :math:`H^1`.
    The positions of the sensors are either freely selected on the mesh or given as input.

    Parameters
    ----------
    N : int
        Dimension of the reduced space.
    Mmax : int
        Maximum number of sensors to select.
    tol : float, optional (Default=0.2)
        Tolerance to exit the stability loop
    xm : list, optional (Default=None)
        If not `None`, list of available positions for the sensors.
    sampleEvery : int, optional (Default = 10)
        Sampling points on the mesh.
    is_H1 : bool, optional (Default: False)
        If `True`, the Riesz representation in :math:`H^1` is used.
    verbose : boolean, optional (Default = False)
        If `True`, printing is enabled.

    """

    self.norm = norms(self.V, is_H1 = is_H1)
    sens_lib = self.sens_class.create(xm = xm, sampleEvery=sampleEvery, is_H1=is_H1, verbose=verbose)

    inf_sup_list = []

    self.xm_sens = []
    self.basis_sens = FunctionsList(self.V)

    # Define first point
    if is_H1:
        measure = np.zeros((len(sens_lib), ))
        for jj in range(len(sens_lib)):
            measure[jj] = self.norm.H1innerProd(np.abs(self.basis(0)), sens_lib(jj), semi = False)
        sensIDX = np.argmax(measure)
    else:
        sensIDX = np.argmax( self.sens_class.action(np.abs(self.basis(0)), sens_lib) )
    self.basis_sens.append(sens_lib(sensIDX))
    self.xm_sens.append(self.sens_class.xm_list[sensIDX])
    
    # Is this necessary?
    sens_lib.delete(sensIDX)
    self.sens_class.xm_list.pop(sensIDX)
    
    m = 1

    resid = Function(self.V).copy()
    while m < Mmax:
        n = min(np.array([N, m]))

        matr_A = np.zeros((m, m))
        matr_K = np.zeros((m, n))
        matr_Z = np.zeros((n, n))

        for ii in range(m):
            for jj in range(m):
                if jj>=ii:
                    if is_H1:
                        matr_A[ii, jj] = self.norm.H1innerProd(self.basis_sens(ii), self.basis_sens(jj), semi = False)
                    else:
                        matr_A[ii, jj] = self.norm.L2innerProd(self.basis_sens(ii), self.basis_sens(jj))
                else:
                    matr_A[ii, jj] = matr_A[jj, ii]
                    
            for kk in range(n):
                if is_H1:
                    matr_K[ii, kk] = self.norm.H1innerProd(self.basis_sens(ii), self.basis(kk), semi = False)
                else:
                    matr_K[ii, kk] = self.norm.L2innerProd(self.basis_sens(ii), self.basis(kk))

        for ii in range(n):
            for jj in range(n):
                if jj>=ii:
                    if is_H1:
                        matr_Z[ii, jj] = self.norm.H1innerProd(self.basis(ii), self.basis(jj), semi = False)
                    else:
                        matr_Z[ii, jj] = self.norm.L2innerProd(self.basis(ii), self.basis(jj))
                else:
                    matr_Z[ii, jj] = matr_Z[jj, ii]
        
        schurComplement = np.matmul(matr_K.T, np.matmul(np.linalg.inv(matr_A), matr_K))
        
        # Solve the eig problem for beta - schurComplement is an Hermitian matrix
        beta_squared, eigenVec_beta = scipy.linalg.eigh(schurComplement, matr_Z)

        inf_sup_list.append( np.sqrt(min(beta_squared)) )
        idx_min_eig = np.argmin(beta_squared)

        # print output
        if verbose:
            print(f'm = {m+0:02}, n = {n+0:02} | beta_n,m = {inf_sup_list[m-1]:.6f}', end = "\r")

        # Compute the least stable mode
        w_inf = self.basis.lin_combine(eigenVec_beta[:, idx_min_eig])

        # Compute projection 
        coeff = np.zeros((len(self.basis_sens),))
        for jj in range(len(self.basis_sens)):
            if is_H1:        
                coeff[jj] = self.norm.H1innerProd(w_inf, self.basis_sens(jj), semi = False)
            else:
                coeff[jj] = self.norm.L2innerProd(w_inf, self.basis_sens(jj))
        # coeff = self.sens_class.action(w_inf, self.basis_sens)

        # Identify the least approximated functional    
        resid.x.array[:] = self.basis_sens.lin_combine(coeff) - w_inf
        if is_H1:
            measure = np.zeros((len(sens_lib),))
            for jj in range(len(sens_lib)):
                measure[jj] = abs(self.norm.H1innerProd(resid, sens_lib(jj), semi = False))
        else:
            measure = abs(self.sens_class.action(resid, sens_lib))
        sensIDX = np.argmax(measure)
        self.xm_sens.append(self.sens_class.xm_list.pop(sensIDX))
        self.basis_sens.append(sens_lib._list.pop(sensIDX))

        m += 1
        
        if inf_sup_list[-1] > tol and m >= N:
            break
    
    if m < Mmax:
        if verbose:
            print(' ')
            print('Starting approximation loop', end = "\r")
        self.approx_loop(Mmax, is_H1=is_H1)
        m = Mmax

    # Last step
    n = min(np.array([N, m]))

    matr_A = np.zeros((m, m))
    matr_K = np.zeros((m, n))
    matr_Z = np.zeros((n, n))

    for ii in range(m):
        for jj in range(m):
            if is_H1:
                matr_A[ii, jj] = self.norm.H1innerProd(self.basis_sens(ii), self.basis_sens(jj), semi = False)
            else:
                matr_A[ii, jj] = self.norm.L2innerProd(self.basis_sens(ii), self.basis_sens(jj))
        for kk in range(n):
            if is_H1:
                matr_K[ii, kk] = self.norm.H1innerProd(self.basis_sens(ii), self.basis(kk), semi = False)
            else:
                matr_K[ii, kk] = self.norm.L2innerProd(self.basis_sens(ii), self.basis(kk))

    for ii in range(n):
        for jj in range(n):
            if is_H1:
                matr_Z[ii, jj] = self.norm.H1innerProd(self.basis(ii), self.basis(jj), semi = False)
            else:
                matr_Z[ii, jj] = self.norm.L2innerProd(self.basis(ii), self.basis(jj))
        
    schurComplement = np.matmul(matr_K.T, np.matmul(np.linalg.inv(matr_A), matr_K))
    
    # Solve the eig problem for beta
    beta_squared, eigenVec_beta = scipy.linalg.eigh(schurComplement, matr_Z)

    inf_sup_list.append( np.sqrt(min(beta_squared)) )
    if verbose:
        print(f'm = {m+0:02}, n = {n+0:02} | beta_n,m = {inf_sup_list[-1]:.6f}')

  def approx_loop(self, Mmax: int, is_H1 : bool = True):
    r"""
    Approximation loop for the selection of sensors position with a Riesz representation :math:`\{g_m\}_{m=1}^M` either in :math:`L^2` or :math:`H^1`.
    At each step `m`, the next position is selected by the following
    
    .. math::
        \mathbf{x}_{m+1} = \text{arg }\max\limits_{\mathbf{x}\in \Omega^\star}\left(\min\limits_{i=1, \dots, m} \| \mathbf{x}-\mathbf{x}_i\|_2\right)
    
    given $\Omega^\star\subset\Omega$ a subset of the whole domain, in which sensors are allowed to be placed. Once the position is known, the functional and its Riesz representation are straightforwardly defined.
    
    Parameters
    ----------
    Mmax : int
        Maximum number of sensors allows.
    is_H1 : bool, optional (Default: False)
        If `True`, the Riesz representation in :math:`H^1` is used.

    """
    m = len(self.xm_sens)
    xdofs = np.array(self.sens_class.xm_list).reshape(-1,3)
    xm_sens_np = np.array(self.xm_sens)
    
    while m <= Mmax:
        min_xdofs = np.zeros((len(xdofs),))
        for ii in range(len(xdofs)):
            x = xdofs[ii]
            min_xdofs[ii] = np.min(np.linalg.norm(xm_sens_np - x, axis=1))
        
        idx_max = np.argmax(min_xdofs)
        xm_sens_np = np.vstack([xm_sens_np, xdofs[idx_max]])
        
        # append to the xm sensor list
        self.xm_sens.append(xm_sens_np[-1])
        
        # Generate sensor
        if is_H1:
            self.basis_sens.append(self.sens_class.define_riesz(xdofs[idx_max]))
        else:
            self.basis_sens.append(self.sens_class.define(xdofs[idx_max]))
        
        m += 1