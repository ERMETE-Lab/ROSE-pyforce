# Offline Phase: Empirical Interpolation Method
# Author: Stefano Riva, PhD Student, NRG, Politecnico di Milano
# Latest Code Update: 14 June 2024
# Latest Doc  Update: 14 June 2024

import numpy as np
from scipy import linalg

from dolfinx.fem import (Function, FunctionSpace)

from pyforce.tools.backends import norms, LoopProgress
from pyforce.tools.functions_list import FunctionsList
    
# EIM: offline
class EIM():
    r"""
    This class is used to perform the offline phase of the Empirical Interpolation Method (EIM) to a scalar field.
    Given a list of training snapshots, this class generates the magic functions and points through a greedy algorithm.

    Parameters
    ----------
    mesh : np.ndarray
        Mesh Points shaped as :math:`\mathcal{N}_h\times gdim`, given :math:`gdim` be the number of independent coordinates (1,2 or 3).
    name : str
        Name of the snapshots (e.g., temperature T)
    
    """
    def __init__(self, mesh: np.ndarray, name: str) -> None:
        
        self.mesh = mesh
        self.name = name
        
        self.Nh = mesh.shape[0]
        self.gdim = mesh.shape[1]
        
    def offline(self, train_snap: FunctionsList, Mmax: int, _xm = None, verbose = False):
        r"""
        The greedy algorithm chooses the magic functions and magic points by minimising the reconstruction error.
        
        Parameters
        ----------
        train_snap : FunctionsList
            List of snapshots serving as training set.
        Mmax : int
            Integer input indicating the maximum number of functions and sensors to define
        _xm : list, optional (default = None)
            User-defined available indices positions for the points (sensors), if `None` the positions are all mesh elements.
        verbose : boolean, optional (Default = False) 
            If `True`, print of the progress is enabled.
            
        Returns
        ----------
        maxAbsErr : np.ndarray
            Maximum absolute error measured in :math:`L^\infty` (max norm)
        maxRelErr : np.ndarray
            Maximum relative error measured in :math:`L^\infty` (max norm)
        beta_coeff : np.ndarray
            Matrix of the reduced coefficients :math:`\{ \beta_m \}_{m=1}^M`, obtained by greedy procedure
        """
        self.Ns = len(train_snap)

        # Generate sensor library
        if _xm is None:
            xm = np.arange(0, self.mesh.shape[0], 1, dtype=int)
        else:
            assert max(_xm) < self.mesh.shape[0], "Indices are out of range"
            assert min(_xm) > 0, "Indices are out of range"
            assert len(_xm) < self.mesh.shape[0], "List of positions not compatible with the mesh points"
            
            xm = np.asarray(_xm, dtype=int)
            
        beta_coeff = np.zeros((len(train_snap), Mmax))
        
        self.magic_fun = FunctionsList(dofs = train_snap.fun_shape)
        self.generating_fun = list()
        self.magic_points = {'idx': list(), 'points': list()}
        
        maxAbsErr = list()
        maxRelErr = list()
        
        snap_matrix = train_snap.return_matrix()
        
        # Generating the first magic function and associated point
        mm = 0
        self.generating_fun.append( int(np.argmax(np.max(abs(snap_matrix[xm]), axis=0))) )
        self.magic_points['idx'].append( xm[np.argmax(abs( snap_matrix[xm, self.generating_fun[mm]]))] )
        self.magic_points['points'].append( self.mesh[self.magic_points['idx'][mm]] )
        
        self.magic_fun.append( snap_matrix[:, self.generating_fun[mm]] / (snap_matrix[self.magic_points['idx'][mm], self.generating_fun[mm]])  )
        
        # Generate the first interpolant
        self.matrix_B = np.zeros((Mmax, Mmax))
        self.matrix_B[mm, mm] = self.magic_fun(mm)[self.magic_points['idx'][mm]]
        
        beta_coeff[:,0] = snap_matrix[self.magic_points['idx'][mm]]
        interpolant = self.magic_fun.return_matrix() @ beta_coeff[:,:mm+1].T
        
        assert interpolant.shape == snap_matrix.shape, "Interpolant shape mismatch with snap_matrix"
        
        for mm in range(1, Mmax):
            residual_matrix = snap_matrix - interpolant
            
            # Find the next maximum error
            self.generating_fun.append( int(np.argmax(np.max(abs(residual_matrix[xm]), axis=0))) )
            maxAbsErr.append( np.max(abs(residual_matrix[xm, self.generating_fun[mm]])) )
            maxRelErr.append( maxAbsErr[mm-1] / np.max(snap_matrix[:, self.generating_fun[mm]]) )
            
            # Find the next magic point
            self.magic_points['idx'].append( xm[np.argmax(abs( residual_matrix[xm, self.generating_fun[mm]]))] )
            self.magic_points['points'].append( self.mesh[self.magic_points['idx'][mm]] )
        
            # Generate the next magic function
            self.magic_fun.append( residual_matrix[:, self.generating_fun[mm]] / residual_matrix[self.magic_points['idx'][mm], self.generating_fun[mm]] )
            
            # Build matrix B
            self.matrix_B[:mm+1, :mm+1] = self.magic_fun.return_matrix()[self.magic_points['idx']]
            
            # Create interpolants    
            for muI in range(self.Ns):
                rhs = snap_matrix[self.magic_points['idx'], muI]
                beta_coeff[muI, :mm+1] = linalg.solve(self.matrix_B[:mm+1, :mm+1], rhs, lower = True)
                
            interpolant = self.magic_fun.return_matrix() @ beta_coeff[:,:mm+1].T
            
            assert interpolant.shape == snap_matrix.shape, "Interpolant shape mismatch with snap_matrix"
        
        residual_matrix = snap_matrix - interpolant
            
        # Find the next maximum error
        last_gen_fun = int(np.argmax(np.max(abs(residual_matrix), axis=0)))
        
        maxAbsErr.append( np.max(residual_matrix[:, last_gen_fun]) )
        maxRelErr.append( maxAbsErr[mm-1] / np.max(snap_matrix[:, last_gen_fun]) )
        
        return maxAbsErr, maxRelErr, beta_coeff

    def reconstruct(self, snap: np.ndarray, Mmax: int):
        r"""
        Computes the reduced coefficients :math:`\{\beta_m\}_{m=1}^{M_{max}}` with 'Mmax' magic functions/points (synthetic) and returns the vector measurement from the snapshots :math:`u`
        
        .. math::
            y_m = u(\vec{x}_m) \qquad m = 1, \dots, M_{max}
        
        Parameters
        ----------
        snap : np.ndarray
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
        measure = snap[self.magic_points['idx']]
        
        # Solve the linear system
        beta_coeff = linalg.solve(self.matrix_B[:Mmax, :Mmax], measure[:Mmax], lower = True)

        return beta_coeff, measure

    def test_error(self, test_snap: FunctionsList, Mmax: int = None):
        r"""
        The absolute and relative error on the test set is computed, by solving the EIM system
        
        .. math::
            \mathbb{B}\boldsymbol{\beta} = \mathbf{y}
        
        Parameters
        ----------
        test_snap : FunctionsList
            List of functions belonging to the test set to reconstruct with EIM
        Mmax : int, optional (default = None)
            Maximum number of magic functions to use (if None is set to the number of magic functions/points)
            
        Returns
        ----------
        meanAbsErr : np.ndarray
            Average absolute error measured in :math:`l^2`
        meanRelErr : np.ndarray
            Average relative error measured in :math:`l^2`
        beta_coeffs
            Matrix of the reduced coefficients, obtained by the solving the EIM linear system
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

        beta_coeffs = np.zeros((Mmax, Ns_test))
        
        for mu in range(Ns_test):
            beta_coeffs[:, mu], _ = self.reconstruct(test_snap(mu), Mmax)
            
        snaps_norm = np.linalg.norm(test_snap.return_matrix(), axis=0)
        
        assert len(snaps_norm) == Ns_test

        for mm in range(Mmax):
            absErr[:,mm]  = np.linalg.norm(self.magic_fun.return_matrix()[:, :mm+1] @ beta_coeffs[:mm+1] - test_snap.return_matrix(), axis = 0)
            relErr[:,mm]  = absErr[:,mm] / snaps_norm

        return absErr.mean(axis = 0), relErr.mean(axis = 0), beta_coeffs