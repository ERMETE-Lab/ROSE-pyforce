# Offline Phase: Empirical Interpolation Method (EIM) - including discrete empirical interpolation method (DEIM)
# Author: Stefano Riva, NRG, Politecnico di Milano
# Latest Code Update: 07 October 2025
# Latest Doc  Update: 07 October 2025

import numpy as np
import scipy
from collections import namedtuple
import os
import warnings

import scipy.linalg

from ..tools.functions_list import FunctionsList
from ..tools.backends import IntegralCalculator, LoopProgress, Timer
from .offline_base import OfflineDDROM

class EIM(OfflineDDROM):
    r"""
    A class implementing the Empirical Interpolation Method (EIM) for dimensionality reduction and sensor placement.

    Parameters
    ----------
    grid: pv.UnstructuredGrid
        The computational grid.
    gdim: int, optional (default=3)
        The geometric dimension of the problem. It can be either 2 or 3.
    varname: str, optional (default='u')
        The name of the variable to analyse. Default is 'u'.
    """

    def fit(self, train_snaps: FunctionsList, Mmax: int, _xm_idx : list = None, verbose= False):
        r"""
        The greedy algorithm chooses the magic functions and magic points by minimising the reconstruction error.
        
        Parameters
        ----------
        train_snaps : FunctionsList
            List of snapshots serving as training set.
        Mmax : int
            Integer input indicating the maximum number of functions and sensors to define
        _xm_idx : list, optional (default = None)
            User-defined available indices for the points (sensors), if `None` the indices are all mesh elements.
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

        # Select if the user provided cell field data or point field data
        if train_snaps.fun_shape == self.grid.n_cells:
            self.field_type = 'cell'
            self.Nh = self.grid.n_cells
            nodes = self.grid.cell_centers().points
        elif train_snaps.fun_shape == self.grid.n_points:
            self.field_type = 'point'
            self.Nh = self.grid.n_points
            nodes = self.grid.points
        else:
            if train_snaps.fun_shape * self.gdim == self.grid.n_cells or train_snaps.fun_shape * self.gdim == self.grid.n_points:
                raise ValueError(f"The shape of the input functions ({train_snaps.fun_shape}) seems to be a vector field, not supported: only scalar fields are allowed.")
            
            raise ValueError(f"The shape of the input functions ({train_snaps.fun_shape}) does not match the number of cells ({self.grid.n_cells}) or points ({self.grid.n_points}) of the grid.")
        

        Ns_train = len(train_snaps)

        # Generate sensor library
        if _xm_idx is None:
            xm = np.arange(0, self.Nh, 1, dtype=int)
        else:
            xm = np.asarray(_xm_idx, dtype=int)

        # Initialize arrays
        beta_coeff = np.zeros((Mmax, len(train_snaps)))

        self.magic_functions = FunctionsList(dofs = train_snaps.fun_shape)
        self.magic_points = {
            'idx': list(), 
            'points': list()
        }    
        self.generating_fun = list()

        # Initialize errors
        maxAbsErr = np.zeros(Mmax)

        # Convert snaps to matrix
        snaps_matrix = train_snaps.return_matrix()

        # Generating the first magic function and associated point from the function maximizing the L_inf norm
        mm = 0

        self.generating_fun.append(
            np.abs(snaps_matrix[xm]).max(axis=0).argmax()
        )
        self.magic_points['idx'].append(
            xm[np.abs(snaps_matrix[xm, self.generating_fun[mm]]).argmax()]
        )
        self.magic_points['points'].append(
            nodes[self.magic_points['idx'][mm], :]
        )
        self.magic_functions.append(
            snaps_matrix[:, self.generating_fun[mm]] / snaps_matrix[self.magic_points['idx'][mm], self.generating_fun[mm]]
        )

        # Generate the first interpolant 
        self.matrix_B = np.zeros((Mmax, Mmax))
        self.matrix_B[mm, mm] = self.magic_functions[mm][self.magic_points['idx'][mm]]

        beta_coeff[mm] = snaps_matrix[self.magic_points['idx'][mm]]
        interpolants = self._reconstruct_from_coeffs(beta_coeff[mm].reshape(1, -1))

        # Main Loop
        for mm in range(1, Mmax):
            residuals_matrix = snaps_matrix - interpolants.return_matrix()

            # Find the next point with maximum residual
            self.generating_fun.append(
                np.abs(residuals_matrix[xm]).max(axis=0).argmax()
            )

            # Assign maximum errors
            maxAbsErr[mm-1] = np.abs(residuals_matrix[xm, self.generating_fun[mm]]).max()

            if verbose:
                print(f'EIM Iteration {(mm+1)+0:03} | Abs Err (Linfty): {maxAbsErr[mm-1]:.2e}', end="\r")

            # Find the next magic point
            self.magic_points['idx'].append(
                xm[np.abs(residuals_matrix[xm, self.generating_fun[mm]]).argmax()]
            )
            self.magic_points['points'].append(
                nodes[self.magic_points['idx'][mm], :]
            )

            # Generate the next magic function
            self.magic_functions.append(
                residuals_matrix[:, self.generating_fun[mm]] / residuals_matrix[self.magic_points['idx'][mm], self.generating_fun[mm]]
            )

            # Update matrix B
            self.matrix_B[:mm+1, :mm+1] = self.magic_functions.return_matrix()[self.magic_points['idx']]

            # Solve for the reduced coefficients
            for mu_i in range(Ns_train):
                beta_coeff[:mm+1, mu_i] = self._solve_eim_linear_system(snaps_matrix[self.magic_points['idx'], mu_i])

            # Update the interpolant
            interpolants = self._reconstruct_from_coeffs(beta_coeff[:mm+1])

        # Final error assignment
        residuals_matrix = snaps_matrix - interpolants.return_matrix()
        self.generating_fun.append(
            np.abs(residuals_matrix[xm]).max(axis=0).argmax()
        )
        maxAbsErr[mm] = np.abs(residuals_matrix[xm, self.generating_fun[-1]]).max()

        if verbose:
            print(f'  Iteration {(mm+1)+0:03} | Abs Err (Linfty): {maxAbsErr[mm-1]:.2e} - EIM done')

        return maxAbsErr, beta_coeff
    
    def compute_lebesgue_constant(self):
        r"""
        This method computes the Lebesgue constant associated with the selected magic points.

        The procedure exploits the Lagrange basis functions, depicted in `Tiglio and Villanueva (2021) <https://iopscience.iop.org/article/10.1088/1361-6382/abf894>`_.

        Returns
        -------
        Lambda : float
            The Lebesgue constant.
        """

        if not hasattr(self, 'magic_functions'):
            raise ValueError("The model has not been fitted yet. Please call the 'fit' method before computing the Lebesgue constant.")

        Lambda = list()

        for mm in range(1, len(self.magic_functions)+1):
            _mf = self.magic_functions.return_matrix()[:,:mm]
            _matrix_B = self.matrix_B[:mm, :mm]
            lagrange_fun = scipy.linalg.solve(_matrix_B, _mf.T, lower=True).T

            # sum of absolute values at each point
            leb_at_x = np.sum(np.abs(lagrange_fun), axis=1)

            # max over domain
            Lambda.append(np.max(leb_at_x))

        return Lambda

    def _reconstruct_from_coeffs(self, coeffs: np.ndarray):
        r"""
        This method reconstructs the function `u` from the reduced coefficients :math:`\{\beta_m\}_{m=1}^M` using the magic functions :math:`\{q_m\}_{m=1}^M`.

        .. math::
            u(\cdot;\,\boldsymbol{\mu}) = \sum_{k=1}^N \beta_k(\boldsymbol{\mu}) q_k(\cdot)


        Parameters
        ----------
        coeffs : np.ndarray
            Reduced coefficients :math:`\{\beta_m\}_{m=1}^M`, shaped :math:`(M,N_s)`.

        Returns
        -------
        u : FunctionsList
            Reconstructed functions.
        """

        assert coeffs.shape[0] <= len(self.magic_functions), "The number of coefficients must be lower to the number of magic functions"
        coeffs = np.atleast_2d(coeffs)

        reconstructed_snaps = FunctionsList(dofs = self.Nh)

        for nn in range(coeffs.shape[1]):
            reconstructed_snaps.append(
                self.magic_functions.lin_combine(coeffs[:, nn])
            )

        return reconstructed_snaps
    
    def _solve_eim_linear_system(self, measures: np.ndarray):
        r"""
        Computes the reduced coefficients :math:`\{\beta_m\}_{m=1}^{M}}` with as many magic functions/points (synthetic) as the input measures :math:`M`.
        
        .. math::
            y_m = u(\vec{x}_m) \qquad m = 1, \dots, M_{max}
        
        Parameters
        ----------
        measures : np.ndarray
            Measurements vector :math:`\{y_m\}_{m=1}^{M_{max}}` of the function `u` at the sensors locations.

        Returns
        ----------
        beta_coeff : np.ndarray
            Array of coefficients for the interpolant :math:`\{\beta_m\}_{m=1}^{M_{max}}`
        """

        assert measures.shape[0] <= len(self.magic_functions), "The number of measures must be equal to the number of magic functions"

        _m = measures.shape[0]
        beta_coeff = scipy.linalg.solve(self.matrix_B[:_m, :_m], measures[:_m], lower = True)

        return beta_coeff
    
    def reconstruct(self, measures: np.ndarray):
        r"""
        This method reconstructs the state by obtaining the reduced coefficients :math:`\{\beta_m\}_{m=1}^M` from the input measures :math:`\{y_m\}_{m=1}^M` by solving the EIM linear system

        .. math::
            \mathbb{B} \boldsymbol{\beta} = \mathbf{y}

        where
        - :math:`\mathbb{B}` is the matrix of magic functions
        - :math:`\boldsymbol{\beta}` are the reduced coefficients
        - :math:`\mathbf{y}` are the input measures

        Parameters
        ----------
        measures : np.ndarray
            Input measures :math:`\{y_m\}_{m=1}^M`, shaped :math:`(M,N_s)`.

        Returns
        -------
        u : FunctionsList
            Reconstructed functions.
        """
        beta_coeff = self._solve_eim_linear_system(measures)
        return self._reconstruct_from_coeffs(beta_coeff)

    def reduce(self, snaps: FunctionsList | np.ndarray, M: int = None):
        r"""
        This method can be used to generate the reduced coefficients based on the magic functions and points, by extracting the measures from the input functions and solve the associated EIM linear system.

        Parameters
        ----------
        snaps : FunctionsList or np.ndarray
            Function object to project onto the reduced space of dimension `M`.
        M : int, optional (Default = None)
            Number of magic functions/points to use for the projection. If `None`, all available magic functions/points are used.

        Returns
        -------
        beta_coeff : np.ndarray
            Reduced coefficients :math:`\{\beta_m\}_{m=1}^M`, shaped :math:`(M,N_s)`.

        """

        if M is None:
            M = len(self.magic_functions)
        else:
            assert M <= len(self.magic_functions), "M cannot be larger than the number of magic functions available"

        if isinstance(snaps, FunctionsList):
            snaps = snaps.return_matrix()
        elif isinstance(snaps, np.ndarray):
            if snaps.ndim == 1:
                snaps = np.atleast_2d(snaps).T # shape (Nh, 1)
        else:
            raise TypeError("Input must be a FunctionsList or a numpy ndarray.")

        assert snaps.shape[0] == self.magic_functions.fun_shape, "Input shape must match the SVD modes shape."

        measures = snaps[self.magic_points['idx'][:M]]

        return self._solve_eim_linear_system(measures)
    
    def _get_measures(self, snaps: FunctionsList | np.ndarray, M: int = None):
        r"""
        This method extracts the measures from the input functions at the magic points locations.

        Parameters
        ----------
        snaps : FunctionsList or np.ndarray
            Function object to extract the measures from.
        M : int, optional (Default = None)
            Number of magic points to use for the extraction. If `None`, all available magic points

        Returns
        -------
        measures : np.ndarray
            Measures :math:`\{y_m\}_{m=1}^M`, shaped :math:`(M,N_s)`.
        """

        if isinstance(snaps, FunctionsList):
            snaps = snaps.return_matrix()
        elif isinstance(snaps, np.ndarray):
            if snaps.ndim == 1:
                snaps = np.atleast_2d(snaps).T
        else:
            raise TypeError("Input must be a FunctionsList or a numpy ndarray.")

        assert snaps.shape[0] == self.magic_functions.fun_shape, "Input shape must match the SVD modes shape."

        measures = snaps[self.magic_points['idx'][:M]]

        return measures

    def compute_errors(self, snaps: FunctionsList | np.ndarray, Mmax: int = None, verbose: bool = False):
        r"""
        Computes the errors between the original snapshots and the reconstructed ones.

        Parameters
        ----------
        snaps : FunctionsList or np.ndarray
            Original snapshots to compare with.
        Mmax : int, optional
            Maximum number of sensors and magic functions to use for the reconstruction. If None, all reduced basis is used.
        verbose : bool, optional
            If True, print progress messages. Default is False.

        Returns
        ----------
        mean_abs_err : np.ndarray
            Average absolute error measured in :math:`L^2`.
        mean_rel_err : np.ndarray
            Average relative error measured in :math:`L^2`.
        computational_time : dict
            Dictionary with the CPU time of the most relevant operations during the online phase.

        """

        if isinstance(snaps, FunctionsList):
            Ns = len(snaps)
            assert snaps.fun_shape == self.magic_functions.fun_shape, "The shape of the snapshots must match the shape of the magic functions."

            # Convert FunctionsList to numpy array for processing
            snaps = snaps.return_matrix()

        elif isinstance(snaps, np.ndarray):
            Ns = snaps.shape[1]
            assert snaps.shape[0] == self.magic_functions.fun_shape, "The shape of the snapshots must match the shape of the magic functions."

        else:
            raise TypeError("Input must be a FunctionsList or a numpy ndarray.")

        if Mmax is None:
            Mmax = len(self.magic_functions)
        else:
            assert Mmax <= len(self.magic_functions), f"Mmax={Mmax} must be less than or equal to the number of magic functions, {len(self.magic_functions)}."

        abs_err = np.zeros((Ns, Mmax))
        rel_err = np.zeros((Ns, Mmax))

        # Variables to store computational time
        computational_time = dict()
        computational_time['Measurements']    = np.zeros((Ns, Mmax))
        computational_time['StateEstimation'] = np.zeros((Ns, Mmax))
        computational_time['Errors']          = np.zeros((Ns, Mmax))

        timer = Timer()

        if verbose:
            print(f"Computing L2 norm of snapshot", end='\r')

        _snap_norm = list()
        for mu_i in range(Ns):

            timer.start()
            _snap_norm.append(
                self.calculator.L2_norm(snaps[:, mu_i])
            )
            
            computational_time['Errors'][mu_i, :] = timer.stop()
        
        if verbose: 
            progressBar = LoopProgress(msg = f"Computing errors - {self.varname}", final = Mmax)

        for mm in range(Mmax):

            if verbose: 
                progressBar.update(1, percentage = False)

            timer.start()
            _measures = self._get_measures(snaps, M=mm+1) # shape (M, Ns)
            computational_time['Measurements'][:, mm] = timer.stop()

            timer.start()
            reconstructed_snaps = self.reconstruct(_measures)
            computational_time['StateEstimation'][:, mm] = timer.stop()

            for mu_i in range(Ns):
                timer.start()
                _resid = snaps[:, mu_i] - reconstructed_snaps(mu_i)
                abs_err[mu_i, mm] = self.calculator.L2_norm(_resid)
                rel_err[mu_i, mm] = abs_err[mu_i, mm] / _snap_norm[mu_i]
                computational_time['Errors'][mu_i, mm] += timer.stop()

        Results = namedtuple('Results', ['mean_abs_err', 'mean_rel_err', 'computational_time'])
        _res = Results(mean_abs_err = abs_err.mean(axis = 0), mean_rel_err = rel_err.mean(axis = 0), computational_time = computational_time)

        return _res


    def save(self, path_folder: str, **kwargs):
        r"""
        Save the magic functions and points to a specified path.

        Parameters
        ----------
        path_folder : str
            The folder path where the model will be saved.
        **kwargs : dict
            Additional keyword arguments for saving options.
        """

        os.makedirs(path_folder, exist_ok=True)

        # Save the magic functions
        self.magic_functions.store( f'mf_{self.varname}',
                                    filename=os.path.join(path_folder, f'mf_{self.varname}'),
                                    **kwargs)
        
        np.save(os.path.join(path_folder, f'magic_points_{self.varname}.npy'), self.magic_points)

def deim(basis_functions: FunctionsList, 
         Mmax: int,
         varname: str = 'u',
         _xm_idx : list = None,
         path_folder: str = None):
    
    r"""
    This function implements the Discrete Empirical Interpolation Method (DEIM) for selecting magic points from a given set of basis functions.

    Parameters
    ----------
    basis_functions : FunctionsList
        The list of basis functions from which to select magic points.
    Mmax : int
        The maximum number of magic points to select.
    varname : str, optional
        The name of the variable for which the magic points are being selected. Default is 'u'.
    _xm_idx : list, optional
        A list of indices to consider for magic point selection. If None, all indices are considered.
    path_folder : str, optional
        The folder path where the selected magic points will be saved. If None, the points are simply returned.

    Returns
    -------
    magic_pt : np.ndarray
        An array of selected magic point indices.
    P : np.ndarray
        The DEIM selection matrix (observation operator).
    """

    modes = basis_functions.return_matrix()

    # Define available indices
    if _xm_idx is None:
        xm = np.arange(modes.shape[0], dtype=int)
    else:
        xm = _xm_idx

    # Initialize DEIM arrays
    magic_pt = np.zeros(Mmax, dtype=int)
    P = np.zeros((Mmax, modes.shape[0]))

    # First DEIM point
    p = np.argmax(np.abs(modes[xm, 0]))
    P[0, xm[p]] = 1
    magic_pt[0] = xm[p]

    # DEIM loop
    for j in range(1, Mmax):

        print(f'DEIM iteration {j+1}/{Mmax}', end='\r')

        # Avoid warnings for numerical issues
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            # Solve linear system
            PtU = (P[:j] @ modes[:, :j]).reshape(j,j)
            y = modes[magic_pt[:j], j]
            c = np.linalg.solve(PtU, y)

            # Compute the residual
            r = modes[xm, j] - modes[xm, :j] @ c

            # Select the next DEIM point
            p = np.argmax(np.abs(r))
            P[j, xm[p]] = 1
            magic_pt[j] = xm[p]

    if path_folder is not None:
        os.makedirs(path_folder, exist_ok=True)
        np.save(os.path.join(path_folder, f'magic_points_deim_{varname}.npy'), magic_pt)

    return magic_pt, P