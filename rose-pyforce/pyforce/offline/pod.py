# Offline Phase: Singular Value Decomposition and Proper Orthogonal Decomposition
# Author: Stefano Riva, PhD Student, NRG, Politecnico di Milano
# Latest Code Update: 07 October 2025
# Latest Doc  Update: 07 October 2025

import numpy as np
import scipy
import pyvista as pv
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd
from collections import namedtuple
import os

from ..tools.functions_list import FunctionsList
from ..tools.backends import IntegralCalculator, LoopProgress, Timer
from .offline_base import OfflineDDROM
import warnings

class rSVD(OfflineDDROM):
    r"""
    A class to perform randomized Singular Value Decomposition (rSVD) on a list of snapshots :math:`u(\mathbf{x};\,\boldsymbol{\mu})` dependent on some parameter :math:`\boldsymbol{\mu}\in\mathcal{P}\subset\mathbb{R}^p`.

    Parameters
    ----------
    grid : pyvista.UnstructuredGrid
        The grid on which the rSVD is performed. It is used to define the spatial domain of the snapshots.
    gdim : int, optional (Default = 3)
        The geometric dimension of the grid. It can be either 2 or 3.
    varname : str, optional (default='u')
        The name of the variable to be used for the rSVD. Default is 'u'.

    """

    def fit(self, train_snaps: FunctionsList, rank: int, verbose: bool=False, **kwargs):
        r"""
        This method is used to perform the randomized SVD on the training snapshots.

        Parameters
        ----------
        train_snaps : FunctionsList
            The training snapshots used to compute the rSVD modes.
        rank : int
            The rank for the truncated SVD.
        verbose : bool, optional
            If True, print progress messages. Default is False.
        kwargs : dict, optional
            Additional keyword arguments for the SVD solver.
        """

        Ns = len(train_snaps)

        if verbose:
            print('Computing ' + self.varname + ' SVD', end='\r')

        _time = Timer()
        _time.start()
        
        # Ignore runtime warnings from sklearn
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            # Compute the randomized SVD
            _U, _S, _ = randomized_svd(train_snaps.return_matrix(), n_components=rank, **kwargs)

        self.singular_values = _S
        self.svd_modes = FunctionsList(train_snaps.fun_shape)
        self.svd_modes.build_from_matrix(_U)

        elapsed_time = _time.stop()
        if verbose:
            print(f"SVD of {self.varname} snapshots calculated in {elapsed_time:.6f} seconds (cpu).")

    def plot_sing_vals(self):
        """
        Simple method to plot the eigenvalues of the POD modes: in linear scale, residual and cumulative energy.

        Returns 
        -------
        fig : matplotlib.figure.Figure
            The figure containing the plots of the eigenvalues, residual energy, and cumulative energy.

        """

        fontsize = 17

        fig, axs = plt.subplots(1, 3, figsize=(16, 4), sharex=True)

        _Nplot = np.arange(1, len(self.singular_values)+1)

        axs[0].plot(_Nplot, self.singular_values, 'r', label='Singular Values')
        axs[0].set_ylabel(r'Singular Values: $\sigma_r$', fontsize=fontsize)

        axs[1].semilogy(_Nplot[:-1], 1 - np.cumsum(self.singular_values[:-1]**2) / np.sum(self.singular_values**2), 'b', label='Cumulative residual energy')
        axs[1].set_ylabel(r'Residual energy: $1 - \frac{\sum_{k=1}^r \sigma_k^2}{\sum_{n=1}^{r_{max}} \sigma_n^2}$', fontsize=fontsize)

        axs[2].plot(_Nplot, np.cumsum(self.singular_values**2) / np.sum(self.singular_values**2), 'g', label='Cumulative energy')
        axs[2].set_ylabel(r'Cumulative energy: $\frac{\sum_{k=1}^r \sigma_k^2}{\sum_{n=1}^{N_s} \sigma_n^2}$', fontsize=fontsize)

        for ax in axs:
            ax.set_xlabel(r'Rank $r$', fontsize=fontsize)
            ax.grid(which='both', linestyle='--', linewidth=0.5)
            ax.tick_params(axis='both', which='major', labelsize=fontsize-2)

        fig.suptitle(f'Singular Values - {self.varname}', y=1.02, fontsize=fontsize+3)
        plt.tight_layout()

        return fig
    
    def _project(self, u: np.ndarray, N: int):
        r"""
        Projects the function `u` onto the first `N` SVD modes to obtain the reduced coefficients :math:`\{\alpha_k\}_{k=1}^N`.

        The projection is done using the inner product in :math:`l_2` (euclidean product).

        Parameters
        ----------
        u : np.ndarray
            Function object to project onto the reduced space of dimension `N`.
        N : int
            Dimension of the reduced space, modes to be used.
        
        Returns
        -------
        coeffs : np.ndarray
            Modal SVD coefficients of `u`, :math:`\{\alpha_k\}_{k=1}^N`.
        
        """

        coeffs = self.svd_modes.return_matrix()[:, :N].T @ u

        return coeffs

    def reduce(self, snaps: FunctionsList | np.ndarray, N: int = None):
        r"""
        The reduced coefficients :math:`\{\alpha_k\}_{k=1}^N` of the snapshots using `N` modes :math:`\{\psi_k\}_{k=1}^N` are computed using projection in :math:`l_2`.
        
        Parameters
        ----------
        snaps : FunctionsList or np.ndarray
            Function object to project onto the reduced space of dimension `N`.
        N : int
            Dimension of the reduced space, modes to be used.
        
        Returns
        -------
        coeff : np.ndarray
            Modal SVD coefficients of `u`, :math:`\{\alpha_k\}_{k=1}^N`.
        """

        if N is None:
            N = len(self.svd_modes)
        else:
            assert N <= len(self.svd_modes), "N must be less than or equal to the number of SVD modes."

        if isinstance(snaps, FunctionsList):
            snaps = snaps.return_matrix()
        elif isinstance(snaps, np.ndarray):
            if snaps.ndim == 1:
                snaps = np.atleast_2d(snaps).T # shape (Nh, 1)
        else:
            raise TypeError("Input must be a FunctionsList or a numpy ndarray.")

        assert snaps.shape[0] == self.svd_modes.fun_shape, "Input shape must match the SVD modes shape."

        coeffs = np.zeros((N, snaps.shape[1]))

        for nn in range(coeffs.shape[1]):
            coeffs[:, nn] = self._project(snaps[:, nn], N)

        return coeffs

    def reconstruct(self, coeffs: np.ndarray):
        r"""
        This method reconstructs the function `u` from the reduced coefficients :math:`\{\alpha_k\}_{k=1}^N` using the POD modes :math:`\{\psi_k\}_{k=1}^N`.

        .. math::
            u(\cdot;\,\boldsymbol{\mu}) = \sum_{k=1}^N \alpha_k(\boldsymbol{\mu}) \psi_k(\cdot)


        Parameters
        ----------
        coeffs : np.ndarray
            Reduced coefficients :math:`\{\alpha_k\}_{k=1}^N`, shaped :math:`(N,N_s)`.

        Returns
        -------
        u : FunctionsList
            Reconstructed functions.
        """

        assert coeffs.shape[0] <= len(self.svd_modes), "The number of coefficients must be less than or equal to the number of POD modes."
        coeffs = np.atleast_2d(coeffs)

        reconstructed_snaps = FunctionsList(self.svd_modes.fun_shape)

        for nn in range(coeffs.shape[1]):
            reconstructed_snaps.append(
                self.svd_modes.lin_combine(coeffs[:, nn])
            )

        return reconstructed_snaps

    def compute_errors(self, snaps: FunctionsList | np.ndarray, Nmax: int = None, verbose: bool = False):
        r"""
        Computes the errors between the original snapshots and the reconstructed ones.

        Parameters
        ----------
        snaps : FunctionsList or np.ndarray
            Original snapshots to compare with.
        Nmax : int, optional
            Maximum number of modes to use for the reconstruction. If None, all modes are used. 
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
            assert snaps.fun_shape == self.svd_modes.fun_shape, "The shape of the snapshots must match the shape of the SVD modes."

            # Convert FunctionsList to numpy array for processing
            snaps = snaps.return_matrix()

        elif isinstance(snaps, np.ndarray):
            Ns = snaps.shape[1]
            assert snaps.shape[0] == self.svd_modes.fun_shape, "The shape of the snapshots must match the shape of the SVD modes."

        else:
            raise TypeError("Input must be a FunctionsList or a numpy ndarray.")
        
        if Nmax is None:
            Nmax = len(self.svd_modes)
        else:
            assert Nmax <= len(self.svd_modes), f"Nmax={Nmax} must be less than or equal to the number of SVD modes, {len(self.svd_modes)}."

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
            progressBar = LoopProgress(msg = f"Computing errors (SVD-projection) - {self.varname}", final = Nmax)

        for nn in range(Nmax):

            if verbose: 
                progressBar.update(1, percentage = False)

            timer.start()
            _coeffs = self.reduce(snaps, nn+1) # shape (N, Ns)
            reconstructed_snaps = self.reconstruct(_coeffs)
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

    def save(self, path_folder: str, **kwargs):
        r"""
        Save the SVD modes and the singular values to a specified path.

        Parameters
        ----------
        path_folder : str
            The folder path where the model will be saved.
        **kwargs : dict
            Additional keyword arguments for saving options.
        """
        
        os.makedirs(path_folder, exist_ok=True)

        # Save the SVD modes
        self.svd_modes.store(f'SVDmode_{self.varname}', 
                             filename=os.path.join(path_folder, f'SVDmode_{self.varname}'),
                             **kwargs)
        
        np.save(os.path.join(path_folder, f'sing_vals_{self.varname}.npy'), self.singular_values)

class POD(OfflineDDROM):
    r"""
    A class to perform the POD on a list of snapshots :math:`u(\mathbf{x};\,\boldsymbol{\mu})` dependent on some parameter :math:`\boldsymbol{\mu}\in\mathcal{P}\subset\mathbb{R}^p`. 
    
    Parameters
    ----------
    grid : pyvista.UnstructuredGrid
        The grid on which the POD is performed. It is used to define the spatial domain of the snapshots.
    gdim : int, optional (Default = 3)
        The geometric dimension of the grid. It can be either 2 or 3.
    varname : str, optional
        The name of the variable to be used for the POD. Default is 'u'.

    """
    
    def fit(self, train_snaps: FunctionsList, verbose: bool=False):
        r"""
        This method is used to obtain the eigendecomposition of the correlation matrix :math:`C\in\mathbb{R}^{N_s\times N_s}` from a list of snapshots as `FunctionsList`.
        
        .. math::
            C_{ij} = \left(u(\cdot;\,\boldsymbol{\mu}_i),\,u(\cdot;\,\boldsymbol{\mu}_j)\right)_{L^2}\qquad i,j = 1, \dots, N_s

        .. math::
            C \boldsymbol{\eta_n} = \lambda_n \boldsymbol{\eta_n}\qquad\qquad\qquad n = 1, \dots, N_s
        
        The eigenvalues :math:`\lambda_n` and eigenvectors :math:`\boldsymbol{\eta_n}` are then computed.
        
        Parameters
        ----------
        train_snaps : FunctionsList
            The training snapshots used to compute the POD modes.
        verbose : bool, optional
            If True, print progress messages. Default is False.
        """

        Ns = len(train_snaps)

        # Calculate the correlation matrix
        if verbose: 
            progressBar = LoopProgress(msg = "Computing " + self.varname + ' correlation matrix', final = Ns)

        _time = Timer()
        _time.start()
        
        corr_matr = np.zeros((Ns, Ns))

        for ii in range(Ns):
            for jj in range(ii, Ns):
                corr_matr[ii, jj] = self.calculator.L2_inner_product(train_snaps(ii), train_snaps(jj))
                corr_matr[jj, ii] = corr_matr[ii, jj]
                
            if verbose: 
                progressBar.update(1, percentage = False)

        eigenvalues, eigenvectors = scipy.linalg.eigh(corr_matr, subset_by_value=[0,np.inf])

        # Sort eigenvalues and eigenvectors in descending order
        sorted_indexes = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[sorted_indexes]
        self.eigenvectors = eigenvectors[:, sorted_indexes]

        elapsed_time = _time.stop()
        if verbose:
            print(f"Eigenvalues calculated in {elapsed_time:.6f} seconds.")

    def plot_eigenvalues(self):
        """
        Simple method to plot the eigenvalues of the POD modes: in linear scale, residual and cumulative energy.

        Returns 
        -------
        fig : matplotlib.figure.Figure
            The figure containing the plots of the eigenvalues, residual energy, and cumulative energy.

        """

        fig, axs = plt.subplots(1, 3, figsize=(16, 4), sharex=True)

        _Nplot = np.arange(1, len(self.eigenvalues)+1)

        axs[0].plot(_Nplot, self.eigenvalues, 'r', label='Eigenvalues')
        axs[0].set_ylabel(r'Eigenvalues $\lambda_r$')

        axs[1].semilogy(_Nplot, 1 - np.cumsum(self.eigenvalues) / np.sum(self.eigenvalues), 'b', label='Cumulative residual energy')
        axs[1].set_ylabel(r'Residual energy $1 - \frac{\sum_{k=1}^r \lambda_k}{\sum_{n=1}^{N_s} \lambda_n}$')

        axs[2].plot(_Nplot, np.cumsum(self.eigenvalues) / np.sum(self.eigenvalues), 'g', label='Cumulative energy')
        axs[2].set_ylabel(r'Cumulative energy $\frac{\sum_{k=1}^r \lambda_k}{\sum_{n=1}^{N_s} \lambda_n}$')

        for ax in axs:
            ax.set_xlabel(r'Rank $r$')
            ax.grid(which='both', linestyle='--', linewidth=0.5)

        plt.tight_layout()

        return fig
    
    def gram_schmidt(self, fun: np.ndarray):
        r"""
        Perform a step of the Gram-Schmidt process on POD modes :math:`\{\psi_k\}_{k=1}^r` adding `fun` :math:`=f` to enforce the orthonormality in :math:`L^2`

        .. math::
            \psi_{r+1} = f - \sum_{k=1}^r \frac{(f, \psi_k)_{L^2}}{(\psi_k, \psi_k)_{L^2}}\psi_k

        Parameters
        ----------
        fun : np.ndarray
            Function to add to the POD basis.

        Returns
        -------
        normalised_fun : np.ndarray
            Orthonormalised function :math:`\psi_{r+1}` with respect to the POD basis :math:`\{\psi_k\}_{k=1}^r`.
        """

        ii = len(self.pod_modes)

        # Defining the summation term
        _rescaling = list()
        for jj in range(ii+1):
            if jj < ii:
                _rescaling.append( 
                    self.calculator.L2_inner_product(fun, self.pod_modes(jj)) / 
                    self.calculator.L2_inner_product(self.pod_modes(jj), self.pod_modes(jj)) *
                    self.pod_modes(jj)
                )

        # Computing the orthonormalised function
        normalised_fun = fun - sum(_rescaling)
        return normalised_fun / self.calculator.L2_norm(normalised_fun)

    def compute_basis(self, train_snaps: FunctionsList, rank: int, normalise: bool = False):
        r"""
        Computes the POD modes.
        
        To enforce the orthonormality in :math:`L^2`, the Gram-Schmidt procedure can be used, if the number of modes to be used is high the numerical error in the eigendecomposition may be too large and the orthonormality is lost.

        Parameters
        ----------
        train_snap : FunctionsList
            List of snapshots onto which the POD is performed.
        rank : int
            Integer input indicating the number of modes to define.
        normalise : boolean, optional (Default = False)
            If True, the Gram-Schmidt procedure is used to normalise the POD modes.

        """

        self.pod_modes = FunctionsList(train_snaps.fun_shape)

        for rr in range(rank):

            _mode = train_snaps.lin_combine(self.eigenvectors[:,rr] / np.sqrt(self.eigenvalues[rr]))

            if normalise:
                # Perform Gram-Schmidt process to ensure orthonormality
                self.pod_modes.append(
                    self.gram_schmidt(_mode)
                    )
            else:
                self.pod_modes.append(
                    _mode
                    )
    
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
            coeffs[nn] = self.calculator.L2_inner_product(u, self.pod_modes[nn])

        return coeffs

    def reduce(self, snaps: FunctionsList | np.ndarray, N: int = None):
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
            N = len(self.pod_modes)
        else:
            assert N <= len(self.pod_modes), "N must be less than or equal to the number of POD modes."

        if isinstance(snaps, FunctionsList):
            snaps = snaps.return_matrix()
        elif isinstance(snaps, np.ndarray):
            if snaps.ndim == 1:
                snaps = np.atleast_2d(snaps).T # shape (Nh, 1)
        else:
            raise TypeError("Input must be a FunctionsList or a numpy ndarray.")
        
        assert snaps.shape[0] == self.pod_modes.fun_shape, "Input shape must match the POD modes shape."

        coeffs = np.zeros((N, snaps.shape[1]))

        for nn in range(coeffs.shape[1]):
            coeffs[:, nn] = self._project(snaps[:, nn], N)

        return coeffs

    def reconstruct(self, coeffs: np.ndarray):
        r"""
        This method reconstructs the function `u` from the reduced coefficients :math:`\{\alpha_k\}_{k=1}^N` using the POD modes :math:`\{\psi_k\}_{k=1}^N`.

        .. math::
            u(\cdot;\,\boldsymbol{\mu}) = \sum_{k=1}^N \alpha_k(\boldsymbol{\mu}) \psi_k(\cdot)


        Parameters
        ----------
        coeffs : np.ndarray
            Reduced coefficients :math:`\{\alpha_k\}_{k=1}^N`, shaped :math:`(N,N_s)`.

        Returns
        -------
        u : FunctionsList
            Reconstructed functions.
        """

        assert coeffs.shape[0] <= len(self.pod_modes), "The number of coefficients must be less than or equal to the number of POD modes."
        coeffs = np.atleast_2d(coeffs)

        reconstructed_snaps = FunctionsList(self.pod_modes.fun_shape)

        for nn in range(coeffs.shape[1]):
            reconstructed_snaps.append(
                self.pod_modes.lin_combine(coeffs[:, nn])
            )

        return reconstructed_snaps
    
    def compute_errors(self, snaps: FunctionsList | np.ndarray, Nmax: int = None, verbose: bool = False):
        r"""
        Computes the errors between the original snapshots and the reconstructed ones.

        Parameters
        ----------
        snaps : FunctionsList or np.ndarray
            Original snapshots to compare with.
        Nmax : int, optional
            Maximum number of modes to use for the reconstruction. If None, all modes are used. 
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
            assert snaps.fun_shape == self.pod_modes.fun_shape, "The shape of the snapshots must match the shape of the POD modes."

            # Convert FunctionsList to numpy array for processing
            snaps = snaps.return_matrix()

        elif isinstance(snaps, np.ndarray):
            Ns = snaps.shape[1]
            assert snaps.shape[0] == self.pod_modes.fun_shape, "The shape of the snapshots must match the shape of the POD modes."

        else:
            raise TypeError("Input must be a FunctionsList or a numpy ndarray.")
        
        if Nmax is None:
            Nmax = len(self.pod_modes)
        else:
            assert Nmax <= len(self.pod_modes), f"Nmax={Nmax} must be less than or equal to the number of POD modes, {len(self.pod_modes)}."

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
            progressBar = LoopProgress(msg = f"Computing errors (POD-projection) - {self.varname}", final = Nmax)

        for nn in range(Nmax):

            if verbose: 
                progressBar.update(1, percentage = False)

            timer.start()
            _coeffs = self.reduce(snaps, nn+1) # shape (N, Ns)
            reconstructed_snaps = self.reconstruct(_coeffs)
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
    
    def save(self, path_folder: str, **kwargs):
        r"""
        Save the POD modes and the eigenvalues to a specified path.

        Parameters
        ----------
        path_folder : str
            The folder path where the model will be saved.
        **kwargs : dict
            Additional keyword arguments for saving options.
        """
        
        os.makedirs(path_folder, exist_ok=True)

        # Save the POD modes
        self.pod_modes.store(f'PODmode_{self.varname}', 
                             filename=os.path.join(path_folder, f'PODmode_{self.varname}'),
                             **kwargs)

        np.save(os.path.join(path_folder, f'eigenvalues_{self.varname}.npy'), self.eigenvalues)

class HierarchicalSVD(rSVD):
    r"""
    A class to perform hierarchical Singular Value Decomposition (hSVD) from `Iwen and Ong (2016) <https://epubs.siam.org/doi/10.1137/140971500>`_. This class inherits from the `rSVD` class and extends its functionality to perform hSVD and update the SVD modes accordingly. This method is particularly useful for large datasets where a hierarchical approach can improve computational efficiency, especially when dealing with parametric problems.

    Parameters
    ----------
    grid : pyvista.UnstructuredGrid
        The grid on which the hSVD is performed. It is used to define the spatial domain of the snapshots.
    gdim : int, optional (Default = 3)
        The geometric dimension of the grid. It can be either 2 or 3.
    varname : str, optional (default='u')
        The name of the variable to be used for the hSVD. Default is 'u'.

    """

    def __init__(self, grid, gdim=3, varname='u', **kwargs):
        super().__init__(grid, gdim, varname, **kwargs)

        self.svd_modes = None
        self.singular_values = None

    def update(self, train_snaps: FunctionsList = None, new_modes: FunctionsList = None, new_sing_vals: np.ndarray = None, rank: int = None,
               **kwargs):
        r"""
        This method updates the SVD modes and singular values using the hierarchical SVD approach, either using new training snapshots or directly providing new modes and singular values.

        Parameters
        ----------
        train_snaps : FunctionsList, optional
            New training snapshots used to compute the new SVD modes and singular values.
        new_modes : FunctionsList, optional
            New SVD modes to be used for the update.
        new_sing_vals : np.ndarray, optional
            New singular values to be used for the update.

        """

        if rank is not None:
            _rank = rank
        else:
            _rank = len(self.singular_values) if self.singular_values is not None else len(train_snaps)

        if train_snaps is not None:
            # Ignore runtime warnings from sklearn
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                # Perform rSVD on the new training snapshots
                _U_new, _S_new, _ = randomized_svd(train_snaps.return_matrix(), n_components=_rank, **kwargs)

            new_modes = FunctionsList(train_snaps.fun_shape)
            new_modes.build_from_matrix(_U_new)
            new_sing_vals = _S_new
        else:
            assert new_modes is not None and new_sing_vals is not None, "Either train_snaps or both new_modes and new_sing_vals must be provided."

        if self.svd_modes is None:
            # First update, simply assign the new modes and singular values
            self.svd_modes = new_modes
            self.singular_values = new_sing_vals
        else:
            
            # Ignore runtime warnings from sklearn
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)

                _A = np.hstack([self.svd_modes.return_matrix() @ np.diag(self.singular_values),
                            new_modes.return_matrix() @ np.diag(new_sing_vals)])
            
                _U_updated, _S_updated, _ = randomized_svd(_A, n_components=_rank, **kwargs)

            # Update the SVD modes and singular values
            self.svd_modes = FunctionsList(self.svd_modes.fun_shape)
            self.svd_modes.build_from_matrix(_U_updated)
            self.singular_values = _S_updated

class IncrementalSVD(rSVD):
    r"""
    A class to perform incremental Singular Value Decomposition (iSVD) from `Brand (2002) <https://link.springer.com/chapter/10.1007/3-540-47969-4_47>`_. This class inherits from the `rSVD` class and extends its functionality to perform iSVD and update the SVD modes accordingly. This method is particularly useful for streaming data or when new snapshots become available over time.

    Parameters
    ----------
    grid : pyvista.UnstructuredGrid
        The grid on which the iSVD is performed. It is used to define the spatial domain of the snapshots.  
    gdim : int, optional (Default = 3)
        The geometric dimension of the grid. It can be either 2 or 3.
    varname : str, optional (default='u')
        The name of the variable to be used for the iSVD. Default is 'u'.

    """

    def __init__(self, grid, gdim=3, varname='u', **kwargs):
        super().__init__(grid, gdim, varname, **kwargs)

        self.svd_modes = None
        self.singular_values = None

    
    def fit(self, train_snaps: FunctionsList, rank: int, verbose: bool=False, **kwargs):
        r"""
        This method is used to perform the randomized SVD on the training snapshots.

        Parameters
        ----------
        train_snaps : FunctionsList
            The training snapshots used to compute the rSVD modes.
        rank : int
            The rank for the truncated SVD.
        verbose : bool, optional
            If True, print progress messages. Default is False.
        kwargs : dict, optional
            Additional keyword arguments for the SVD solver.
        """

        self.Ns = len(train_snaps)
        self.Nh = train_snaps.fun_shape
        self.rank = rank

        if verbose:
            print('Computing ' + self.varname + ' SVD', end='\r')

        _time = Timer()
        _time.start()

        # Ignore runtime warnings from sklearn
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            _U, _S, _Vh = randomized_svd(train_snaps.return_matrix(), n_components=rank, **kwargs)

        self.singular_values = _S
        self.svd_modes = FunctionsList(train_snaps.fun_shape)
        self.svd_modes.build_from_matrix(_U)
        self.Vh = _Vh

        elapsed_time = _time.stop()
        if verbose:
            print(f"SVD of {self.varname} snapshots calculated in {elapsed_time:.6f} seconds (cpu).")

    def update(self, new_snap: FunctionsList | np.ndarray):
        r"""
        This method updates the SVD modes and singular values using the incremental SVD approach with a new snapshot.
        """

        if isinstance(new_snap, FunctionsList):
            new_snap = new_snap.return_matrix()
        elif isinstance(new_snap, np.ndarray):
            if new_snap.ndim == 1:
                new_snap = np.atleast_2d(new_snap).T # shape (Nh, 1)
        else:
            raise TypeError("Input must be a FunctionsList or a numpy ndarray.")
        
        # Step 1: eigen-decom (projection onto the existing SVD modes)
        L, num_new_snaps = self._eigen_decomposition(new_snap)

        # Step 2: non-orthogonal components over the existing SVD modes with QR
        J, K = self._qr_decomposition(L, new_snap)

        # Step 3: new SVD
        Uplus, Splus, Vhplus = self._compute_new_svd(L, K)

        # Step 4: update the SVD
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            self.svd_modes = FunctionsList(snap_matrix = np.hstack([self.svd_modes.return_matrix(), J]) @ Uplus)
        self.singular_values = Splus
        
        tmp_V = np.vstack([ np.hstack([self.Vh.T, np.zeros((self.Vh.shape[1], J.shape[1]))]), 
                            np.hstack([np.zeros((J.shape[1], self.Vh.shape[0])), np.eye(J.shape[1])])])
        self.Vh = (tmp_V @ Vhplus.T).T

        # Update number of snapshots and check SVD dimensions
        self.Ns += num_new_snaps
        self._check_svd()

    def _eigen_decomposition(self, new_data: np.ndarray):
        """
        This method performs the eigen-decomposition step of the incremental SVD algorithm.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            L = self.svd_modes.return_matrix().T @ new_data
        num_new_snaps = new_data.shape[1]
        
        assert L.shape[0] == self.rank
        assert L.shape[1] == num_new_snaps

        return L, num_new_snaps

    def _qr_decomposition(self, L: np.ndarray, new_data: np.ndarray):
        """
        This method performs the QR decomposition step of the incremental SVD algorithm.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            H = new_data - self.svd_modes.return_matrix() @ L

        J, K = np.linalg.qr(H)

        return J, K
    
    def _compute_new_svd(self, L: np.ndarray, K: np.ndarray):
        """
        This method computes the new SVD after the eigen-decomposition and QR decomposition steps.
        """
        Q = np.vstack([np.hstack([np.diag(self.singular_values), L]),
                        np.hstack([np.zeros((K.shape[0], self.singular_values.shape[0])), K])])

        # Ignore runtime warnings from sklearn
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            Uplus, Splus, Vhplus = randomized_svd(Q, n_components=self.rank)

        return Uplus, Splus, Vhplus

    def _check_svd(self):
        assert self.svd_modes.return_matrix().shape[0] == self.Nh
        assert self.svd_modes.return_matrix().shape[1] == self.rank
        assert self.singular_values.shape[0] == self.rank
        assert self.Vh.shape[0] == self.rank
        assert self.Vh.shape[1] == self.Ns

    