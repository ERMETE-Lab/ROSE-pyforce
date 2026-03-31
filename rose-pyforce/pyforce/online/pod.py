# Online Phase: Proper Orthogonal Decomposition
# Author: Stefano Riva, PhD Student, NRG, Politecnico di Milano
# Latest Code Update: 07 October 2025
# Latest Doc  Update: 07 October 2025

import numpy as np
import pyvista as pv
from collections import namedtuple
import os

from ..tools.functions_list import FunctionsList
from ..tools.backends import IntegralCalculator, LoopProgress, Timer
from ..tools.write_read import ImportFunctionsList
from .online_base import OnlineDDROM, SurrogateModelWrapper

class POD(OnlineDDROM):
    r"""
    A class to estimate the state using the POD modes.

    This class implements the pure projection method to estimate the state using the POD modes and allows for the integration with surrogate models for the reduced dynamics (e.g., interpolation, machine learning, etc.), adopting the scheme of the POD with Interpolation (Demo et al, 2019 - https://doi.org/10.48550/arXiv.1905.05982).
    
    Parameters
    ----------
    grid : pyvista.UnstructuredGrid
        The grid on which the POD is performed. It is used to define the spatial domain of the snapshots.
    gdim : int, optional (Default = 3)
        The geometric dimension of the grid. It can be either 2 or 3.
    varname : str, optional
        The name of the variable to be used for the POD. Default is 'u'.

    """

    def set_basis(self, basis: FunctionsList = None, path_folder: str = None, **kwargs):
        """
        Assign the basis functions to the POD model either from a FunctionsList object or by loading from a folder.

        Parameters
        ----------
        basis : FunctionsList, optional
            An instance of `FunctionsList` containing the basis functions.
        path_folder : str, optional
            The path to the folder containing the basis functions.
        **kwargs : dict
            Additional keyword arguments to pass to the loading function.

        Returns
        -------
        None

        """
        if basis is not None:
            self._basis = basis
        elif path_folder is not None:
            _filename = os.path.join(path_folder, f'PODmode_{self.varname}')
            self._basis = ImportFunctionsList(_filename, **kwargs)
        else:
            raise ValueError("Either 'basis' or 'path_folder' must be provided.")

    def _reconstruct(self, coeffs: np.ndarray):
        r"""
        This method reconstructs the state from the reduced coefficients :math:`\{\alpha_k\}_{k=1}^N` using the POD modes :math:`\{\psi_k\}_{k=1}^N`.

        .. math::
            u(\cdot;\,\boldsymbol{\mu}) = \sum_{k=1}^N \alpha_k(\boldsymbol{\mu}) \psi_k(\cdot)


        Parameters
        ----------
        coeffs : np.ndarray
            Reduced coefficients :math:`\{\alpha_k\}_{k=1}^N`, shaped :math:`(N,N_s)`.

        Returns
        -------
        estimation : FunctionsList
            An instance of `FunctionsList` containing the estimated state.
        """

        assert coeffs.shape[0] <= len(self.basis), "The number of coefficients must be less than or equal to the number of POD modes."
        coeffs = np.atleast_2d(coeffs)

        estimation = FunctionsList(self.basis.fun_shape)

        for nn in range(coeffs.shape[1]):
            estimation.append(
                self.basis.lin_combine(coeffs[:, nn])
            )

        return estimation
    
    def estimate(self, coeff_model: SurrogateModelWrapper, input_vector: np.ndarray):
        r"""
        This method can be adopted to estimate the state using a surrogate model :math:`\{\mathcal{F}_k\}_{k=1}^N` (i.e., `coeff_model`) for the reduced coefficients :math:`\{\alpha_k\}_{k=1}^N`, given the input vector :math:`\mathbf{z}`, such that

        .. math::
            \alpha_k(\mathbf{z}) = \mathcal{F}_k(\mathbf{z})

        Parameters
        ----------
        coeff_model : SurrogateModelWrapper
            The surrogate model to use for estimating the coefficients.
        input_vector : np.ndarray
            The input vector for which to estimate the state.

        Returns
        -------
        estimation : FunctionsList
            An instance of `FunctionsList` containing the estimated state.

        """

        coeffs = coeff_model.predict(input_vector)

        return self._reconstruct(coeffs)

    def _project(self, u: np.ndarray, N: int):
        r"""
        Projects the function `u` onto the first `N` POD modes to obtain the reduced coefficients :math:`\{\alpha_k\}_{k=1}^N`.

        The projection is done using the inner product in :math:`L_2`, i.e.

        .. math::
            \alpha_k(\boldsymbol{\mu}) = (u(\cdot;\,\boldsymbol{\mu}), \,\psi_k)_{L^2}\qquad k = 1, \dots, N
        
        Parameters
        ----------
        u : np.ndarray
            Function object to project onto the reduced space of dimension `N`.
        N : int
            Dimension of the reduced space, modes to be used.
        
        Returns
        -------
        coeffs : np.ndarray
            Modal POD coefficients of `u`, :math:`\{\alpha_k\}_{k=1}^N`.
        
        """

        coeffs = np.zeros(N)

        for nn in range(N):
            coeffs[nn] = self.calculator.L2_inner_product(u, self.basis[nn])

        return coeffs

    def _reduce(self, snaps: FunctionsList | np.ndarray, N: int = None):
        r"""
        The reduced coefficients :math:`\{\alpha_k\}_{k=1}^N` of the snapshots using `N` modes :math:`\{\psi_k\}_{k=1}^N` are computed using projection in :math:`L_2`.
        
        Parameters
        ----------
        snaps : FunctionsList or np.ndarray
            Function object to project onto the reduced space of dimension `N`.
        N : int
            Dimension of the reduced space, modes to be used.
        
        Returns
        -------
        coeff : np.ndarray
            Modal POD coefficients of `u`, :math:`\{\alpha_k\}_{k=1}^N`.
        """

        if N is None:
            N = len(self.basis)
        else:
            assert N <= len(self.basis), "N must be less than or equal to the number of POD modes."

        if isinstance(snaps, FunctionsList):
            snaps = snaps.return_matrix()
        elif isinstance(snaps, np.ndarray):
            if snaps.ndim == 1:
                snaps = np.atleast_2d(snaps).T # shape (Nh, 1)
        else:
            raise TypeError("Input must be a FunctionsList or a numpy ndarray.")
        
        assert snaps.shape[0] == self.basis.fun_shape, "Input shape must match the POD modes shape."

        coeffs = np.zeros((N, snaps.shape[1]))

        for nn in range(coeffs.shape[1]):
            coeffs[:, nn] = self._project(snaps[:, nn], N)

        return coeffs
    

    def compute_errors(self, snaps: FunctionsList | np.ndarray, 
                       coeff_model: SurrogateModelWrapper, input_vector: np.ndarray,
                       verbose: bool = False):
        r"""
        Computes the errors between the original snapshots and the reconstructed ones.

        Parameters
        ----------
        snaps : FunctionsList or np.ndarray
            Original snapshots to compare with.
        coeff_model : SurrogateModelWrapper
            The surrogate model to use for estimating the coefficients.
        input_vector : np.ndarray
            The input vector for which to estimate the state.
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
            assert snaps.fun_shape == self.basis.fun_shape, "The shape of the snapshots must match the shape of the POD modes."

            # Convert FunctionsList to numpy array for processing
            snaps = snaps.return_matrix()

        elif isinstance(snaps, np.ndarray):
            Ns = snaps.shape[1]
            assert snaps.shape[0] == self.basis.fun_shape, "The shape of the snapshots must match the shape of the POD modes."

        else:
            raise TypeError("Input must be a FunctionsList or a numpy ndarray.")
        
        Nmax = len(coeff_model.models)

        abs_err = np.zeros((Ns, Nmax))
        rel_err = np.zeros((Ns, Nmax))

        # Variables to store computational time
        computational_time = dict()
        computational_time['StateEstimation'] = np.zeros((Ns, Nmax))
        computational_time['Errors']          = np.zeros((Ns, Nmax))

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
            progressBar = LoopProgress(msg = f"Computing errors (POD-I) - {self.varname}", final = Nmax)

        estimated_coeffs = coeff_model.predict(input_vector)

        for nn in range(Nmax):

            if verbose: 
                progressBar.update(1, percentage = False)

            timer.start()
            reconstructed_snaps = self._reconstruct(estimated_coeffs[:nn+1, :]) # shape (fun_shape, Ns)
            computational_time['StateEstimation'][:, nn] = timer.stop()

            for mu_i in range(Ns):
                timer.start()
                _resid = snaps[:, mu_i] - reconstructed_snaps(mu_i)
                abs_err[mu_i, nn] = self.calculator.L2_norm(_resid)
                rel_err[mu_i, nn] = abs_err[mu_i, nn] / _snap_norm[mu_i]
                computational_time['Errors'][mu_i, nn] += timer.stop()

        Results = namedtuple('Results', ['mean_abs_err', 'mean_rel_err', 'computational_time'])
        _res = Results(mean_abs_err = abs_err.mean(axis = 0), mean_rel_err = rel_err.mean(axis = 0), computational_time = computational_time)

        return _res