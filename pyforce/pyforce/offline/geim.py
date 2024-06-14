# Offline Phase: Generalised Empirical Interpolation Method
# Author: Stefano Riva, PhD Student, NRG, Politecnico di Milano
# Latest Code Update: 05 March 2024
# Latest Doc  Update: 05 March 2024

import numpy as np
import scipy

from dolfinx.fem import (Function, FunctionSpace)

from pyforce.tools.backends import norms, LoopProgress
from pyforce.tools.functions_list import FunctionsList
from .sensors import gaussian_sensors
    
# GEIM: offline
class GEIM():
    """
    This class is used to perform the offline phase of the Generalised Empirical Interpolation Method (GEIM) to a scalar field.
    Given a list of training snapshots, this class generates the magic functions and sensors through a greedy algorithm.

    Parameters
    ----------
    domain : dolfinx.mesh
        Mesh of the problem onto which the sensors are generated.
    V : FunctionSpace
        Functional space of the snapshots.
    name : str
        Name of the snapshots (e.g., temperature T)
    s : float
        Standard deviation of the gaussian kernel for the sensors

    """
    def __init__(self, domain, V: FunctionSpace, name: str, s=1e-2) -> None:
        self.V = V
        self.norm = norms(self.V)
        self.name = name
        
        # Generate a sensor library class - assuming gaussian kernel and scalar fields
        self.sens_class = gaussian_sensors(domain, self.V, s)

    def offline(self, train_snap: FunctionsList, Mmax: int, 
                xm = None, sampleEvery = 5, verbose = False):
        r"""
        The greedy algorithm chooses the magic functions and magic sensors by minimising the reconstruction error.
        
        Parameters
        ----------
        train_snap : FunctionsList
            List of snapshots serving as training set.
        Mmax : int
            Integer input indicating the maximum number of functions and sensors to define
        xm : list, optional (default = None)
            User-defined available positions for the sensors, if `None` the positions are taken from the mesh elements.
        sampleEvery : int, optional (default = 5)
            If `xm` is not `None`, sampling rate for the selection of points from the mesh
        verbose : boolean, optional (Default = False) 
            If `True`, print of the progress is enabled.
            
        Returns
        ----------
        maxAbsErr : np.ndarray
            Maximum absolute error measured in :math:`L^2`
        maxRelErr : np.ndarray
            Maximum relative error measured in :math:`L^2`
        beta_coeff : np.ndarray
            Matrix of the reduced coefficients :math:`\{ \beta_m \}_{m=1}^M`, obtained by greedy procedure
        """
        self.Ns = len(train_snap)

        # Generate sensor library
        sensList = self.sens_class.create(xm = xm, sampleEvery=sampleEvery, verbose=verbose)
            
        maxNorm = 0.
        snapNormList = []

        ## find first generating function
        for mu in range(len(train_snap)):
            tmpNorm = self.norm.L2norm(train_snap(mu))
            snapNormList.append(tmpNorm)
            if maxNorm < tmpNorm:
                maxNorm = tmpNorm
                generatingIdx = mu

        ## find first magic sensor
        measure = self.sens_class.action(train_snap(generatingIdx), sensList)
        sensIDX = np.argmax( abs(measure) )
        maxMeasure = measure[sensIDX]

        self.magic_sens = FunctionsList(self.V)
        self.magic_sens.append(sensList(sensIDX))
        self.msens_xm_list = list()
        self.msens_xm_list.append(self.sens_class.xm_list[sensIDX])

        ## find first magic function
        self.magic_fun = FunctionsList(self.V)
        self.magic_fun.append(train_snap(generatingIdx) / maxMeasure)
        # Deprecated?
        # tmp = Function(self.V).copy()
        # tmp.x.array[:] = train_snap(generatingIdx) / maxMeasure
        # self.magic_fun.append(tmp)

        ##### GEIM Offline: main loop
        beta_coeff = np.zeros((len(train_snap), Mmax))
        rhs_values = np.zeros((len(train_snap), Mmax))
        
        maxAbsErr = np.zeros((Mmax,))
        maxRelErr = np.zeros_like(maxAbsErr)

        iter = 0

        # Deprecated?
        # resid = Function(self.V)
        # tmp2 = Function(self.V)

        B = np.zeros((Mmax, Mmax))
        
        while iter < Mmax:

            max_abs_err = 0.
            
            # Generate matrix B
            for ii in range(iter+1):
                B[iter, ii] = self.sens_class.action_single(self.magic_fun(ii), self.magic_sens(iter))

            # Look for the worst approximated function
            for mu in range(len(train_snap)):
                # Define the rhs - the measurement vector
                rhs_values[mu, iter] = self.sens_class.action_single(train_snap(mu), self.magic_sens(iter))
                
                # solve the linear system to find the coefficients
                beta_coeff[mu, :iter+1] = scipy.linalg.solve(B[:iter+1, :iter+1], 
                                                             rhs_values[mu, :iter+1], lower = True)

                # Compute residual field
                resid = train_snap(mu) - self.magic_fun.lin_combine(beta_coeff[mu, :iter+1])
                # Deprecated ?
                # resid.x.array[:] = train_snap(mu) - self.magic_fun.lin_combine(beta_coeff[mu, :iter+1])
                
                # Compute absolute error
                abs_err = self.norm.L2norm(resid)
                if max_abs_err < abs_err:
                    max_abs_err = abs_err
                    generatingIdx = mu

            maxAbsErr[iter] = max_abs_err
            maxRelErr[iter] = max_abs_err / snapNormList[generatingIdx]

            # Generate the magic sensor and function
            if iter < Mmax - 1:
                
                # Look for the maximising sensor  
                resid = train_snap(generatingIdx) - self.magic_fun.lin_combine(beta_coeff[generatingIdx, :iter+1])   
                measure = self.sens_class.action(resid, sensList)
                                            
                sensIDX = np.argmax(abs(measure))
                maxMeasure = measure[sensIDX]
                self.magic_sens.append(sensList(sensIDX))
                self.msens_xm_list.append(self.sens_class.xm_list[sensIDX])

                # Generate the magic function
                tmp2 = ( train_snap(generatingIdx) - self.magic_fun.lin_combine(beta_coeff[generatingIdx, :iter+1]) ) / maxMeasure
                self.magic_fun.append(tmp2)
                
                # Deprecated?
                # tmp2.x.array[:] = train_snap(generatingIdx)
                # interpolant = self.magic_fun.lin_combine(beta_coeff[generatingIdx, :iter+1], use_numpy = False)
                # tmp2.vector.axpy(-1., interpolant.vector)
                # tmp2.x.array[:] /= maxMeasure
                # self.magic_fun.append(tmp2.copy())
            
            ## Update and print output
            iter += 1
            if verbose:
                print(f'  Iteration {iter+0:03} | Abs Err: {maxAbsErr[iter-1]:.2e} | Rel Err: {maxRelErr[iter-1]:.2e}', end="\r")

        # Storing B
        self.B = B

        return maxAbsErr, maxRelErr, beta_coeff

    def reconstruct(self, snap: Function, Mmax: int):
        r"""
        Computes the reduced coefficients :math:`\{\beta_m\}_{m=1}^{M_{max}}` with 'Mmax' magic functions/sensors (synthetic) and returns the vector measurement from the snapshots :math:`u`
        
        .. math::
            y_m = v_m(u) \qquad m = 1, \dots, M_{max}
        
        Parameters
        ----------
        snap : Function
            Function from which the measuremets are computed.
        Mmax: int
            Maximum number of sensors to use

        Returns
        ----------
        beta_coeff : np.ndarray
            Array of coefficients for the interpolant :math:`\{\beta_m\}_{m=1}^{M_{max}}`
        Measure : np.ndarray
            Array with the evaluation of the `snap` :math:`u` at the sensors locations

        """

        # Computing the measurement vector
        measure = self.sens_class.action(snap, self.magic_sens)
        
        # Solve the linear system
        beta_coeff = scipy.linalg.solve(self.B[:Mmax, :Mmax], measure[:Mmax], lower = True)

        return beta_coeff, measure

    def test_error(self, test_snap: FunctionsList, Mmax: int = None, verbose = False):
        r"""
        The absolute and relative error on the test set is computed, by solving the GEIM system
        
        .. math::
            \mathbb{B}\boldsymbol{\beta} = \mathbf{y}
        
        Parameters
        ----------
        test_snap : FunctionsList
            List of functions belonging to the test set to reconstruct with GEIM
        Mmax : int, optional (default = None)
            Maximum number of magic functions to use (if None is set to the number of magic functions/sensors)
        verbose : boolean, optional (default = False)
            If true, output is printed.
            
        Returns
        ----------
        meanAbsErr : np.ndarray
            Average absolute error measured in :math:`L^2`
        meanRelErr : np.ndarray
            Average relative error measured in :math:`L^2`
        coeff_matrix
            Matrix of the reduced coefficients, obtained by the solving the GEIM linear system
        """

        # Check on the input M, maximum number of sensors to use
        if Mmax is None:
            Mmax = len(self.magic_fun)
        elif Mmax > len(self.magic_fun):
            print('The maximum number of measures must not be higher than '+str(len(self.magic_fun))+' --> set equal to '+str(len(self.magic_fun)))
            Mmax = len(self.magic_fun)

        Ns_test = len(test_snap)
        absErr = np.zeros((Ns_test, Mmax))
        relErr = np.zeros_like(absErr)
        coeff_matrix = np.zeros_like(absErr)

        if verbose:
            progressBar = LoopProgress(msg = "Computing GEIM test error (synthetic) - " + self.name, final = Ns_test)

        resid = Function(self.V).copy()
        for mu in range(Ns_test):

            snap_norm = self.norm.L2norm(test_snap(mu))

            for M in range(Mmax):
                coeff, _ = self.reconstruct(test_snap(mu), M+1)
                resid.x.array[:] = test_snap(mu) - self.magic_fun.lin_combine(coeff)
                absErr[mu,M] = self.norm.L2norm(resid)
                relErr[mu,M] = absErr[mu,M] / snap_norm

            coeff_matrix[mu, :] = coeff[:]
            if verbose:
                progressBar.update(1, percentage = False)

        return absErr.mean(axis = 0), relErr.mean(axis = 0), coeff_matrix

    def assemble_penalisation_matrix(self, train_snap: FunctionsList, Mmax: int, verbose: bool = False):
        r"""Construct penalisation matrix for the TR-GEIM
        
        """
        
        

def computeLebesgue(magic_fun: FunctionsList, magic_sens: FunctionsList):
    r"""
    The Lebesgue constant :math:`\Lambda_M` is computed from the magic functions and sensors, to measure the good properties of the interpolation procedure.
    This function follows the implementation reported in `https://www.sciencedirect.com/science/article/pii/S0045782515000389?via%3Dihub`.
    
    Parameters
    ----------
    magic_fun : FunctionsList
        List of the magic functions
    magic_sens : FunctionsList
        List of the magic sensors

    Returns
    ----------
    Lebesgue : np.ndarray
        Array containing the Lebesgue constant at different number of sensors
    """
    assert (len(magic_fun) == len(magic_sens))
    M = len(magic_fun)

    # Define variables and class for norm computing
    orth_magic_fun  = FunctionsList(magic_fun.fun_space)
    orth_magic_sens = FunctionsList(magic_fun.fun_space)
    
    norm = norms(magic_fun.fun_space)

    # Generate a set of orthonormal basis functions using Grahm-Schmidt procedure
    for ii in range(M):
        
        orth_fun  = Function(magic_fun.fun_space)
        orth_sens = Function(magic_fun.fun_space)
        
        orth_fun.x.array[:] = magic_fun(ii)
        orth_sens.x.array[:] = magic_sens(ii)
        
        for jj in range(ii+1):
            if jj < ii:
                orth_fun.vector.axpy(- norm.L2innerProd(orth_fun, orth_magic_fun(jj)) / norm.L2innerProd(orth_magic_fun(jj), orth_magic_fun(jj)),
                                                    orth_magic_fun.map(jj).vector)
                orth_sens.vector.axpy(- norm.L2innerProd(orth_sens, orth_magic_sens(jj)) / norm.L2innerProd(orth_magic_sens(jj), orth_magic_sens(jj)),
                                                    orth_magic_sens.map(jj).vector)

        orth_fun.x.array[:]   /= norm.L2norm(orth_fun)
        orth_sens.x.array[:]  /= norm.L2norm(orth_sens)

        orth_magic_fun.append(orth_fun)
        orth_magic_sens.append(orth_sens)

    # Compute the lebesgue constant
    Lebesgue = np.zeros((M,))
    for mm in range(M):
        A = np.zeros((mm+1, mm+1))

        for ii in range(mm+1):
            for jj in range(mm+1):
                A[ii, jj] = norm.L2innerProd(orth_magic_fun(ii), orth_magic_sens(jj))

        eigenval = np.linalg.eigvals(A.T @ A)
        Lebesgue[mm] = 1. / np.sqrt(min(eigenval))
    return Lebesgue