# List of functions for storage
# Author: Stefano Riva, PhD Student, NRG, Politecnico di Milano
# Latest Code Update: 07 October 2025
# Latest Doc  Update: 07 October 2025

import numpy as np
from sklearn.model_selection import train_test_split as sk_split
import h5py
from .plotting import plot_function as pf
import pyvista as pv
from IPython.display import clear_output as clc
import os

# Class to define a list of function, useful to collect snapshots and basis functions
class FunctionsList():
    r"""
    A class wrapping a list of functions. They are stored as a `list` of `np.ndarray`.

    Parameters
    ----------
    dofs : int
        Degrees of freedom of the functions :math:`\mathcal{N}_h`.

    Attributes
    ----------
    _list : list
        List containing the functions as `np.ndarray`.
    fun_shape : int
        Number of degrees of freedom of the functions :math:`\mathcal{N}_h`.
    """
    def __init__(self, dofs: int = None, snap_matrix: np.ndarray = None) -> None:
        
        self._list = list()

        # If dofs is not given, it must be inferred from the snap_matrix
        if dofs is None and snap_matrix is None:
            raise ValueError("Either dofs or snap_matrix must be provided.")
        elif snap_matrix is not None:
            self.fun_shape = snap_matrix.shape[0]
            self.build_from_matrix(snap_matrix)
        else:
            self.fun_shape = dofs

    def __call__(self, idx) -> np.ndarray:
        """
        Defining the class as callable, returns the idx-th element of the list
        """
        return self._list[idx]
    
    def __getitem__(self, idx):
        """
        Defining the class as indexable.
        If idx is an integer, return the corresponding element.
        If idx is a slice, return a new FunctionsList with the sliced elements.
        """
        if isinstance(idx, slice):
            new_list = FunctionsList(dofs=self.fun_shape)
            new_list._list = self._list[idx]
            return new_list
        else:
            return self._list[idx]
    
    def __len__(self) -> int:
        """
        Defining the length of the class as the length of the stored list
        """
        return len(self._list)
    
    def __iter__(self):
        """
        Defining the iterator of the class as the iterator of the stored list
        """
        return iter(self._list)

    def append(self, dofs_fun: np.ndarray) -> None:
        """
        Extend the current list by adding a new function.
        The snapshot is stored in a list as a numpy array, to avoid problems when the number of elements becomes large.

        Parameters
        ----------
        dofs_fun : np.ndarray
            Functions to be appended.

        """
        assert dofs_fun.shape[0] == self.fun_shape, "The input function dofs_fun has "+str(dofs_fun.shape[0])+", instead of "+str(self.fun_shape)
        self._list.append(dofs_fun)
    
    def build_from_matrix(self, matrix: np.ndarray) -> None:
        r"""
        Build the list from a matrix :math:`S\in\mathbb{R}^{\mathcal{N}_h\times N_s}`.
        The matrix is transposed to match the shape of the functions.

        Parameters
        ----------
        matrix : np.ndarray
            Matrix containing the functions as columns.
        """
        assert matrix.shape[0] == self.fun_shape, "The input matrix has "+str(matrix.shape[0])+", instead of "+str(self.fun_shape)
        self._list = [matrix[:, ii] for ii in range(matrix.shape[1])]

    def delete(self, idx: int) -> None:
        r"""
        Delete a single element in position `idx`.
        
        Parameters
        ----------
        idx : int
            Integers indicating the position inside the `_list` to delete.
        """
        del self._list[idx]

    def copy(self):
        r"""
        Defining the copy of the `_list` of elements
        """
        
        return self._list.copy()

    def clear(self) -> None:
        """Clear the storage."""
        self._list = list()

    def shape(self) -> tuple:
        """
        Returns the shape of the list as a tuple `(dofs, number of functions)`.
        """
        return (self.fun_shape, len(self._list))

    def sort(self, order: list) -> None:
        """
        Sort the list according to the given order (list of indices).

        Parameters
        ----------
        order : list
            List of indices for the sorting.
        """
        assert len(self._list) == len(order), (
            f"The order vector ({len(order)}) must have the same length as the list ({len(self._list)})"
        )
        self._list = [self._list[i] for i in order]
        
    def return_matrix(self) -> np.ndarray:
        r"""
        Returns the list of arrays as a matrix :math:`S\in\mathbb{R}^{\mathcal{N}_h\times N_s}`.
        """
        
        return np.asarray(self._list).T

    def lin_combine(self, vec: np.ndarray) -> np.ndarray:
        r"""
        Linearly combine functions (`a_i = vec[i]`) in the list :math:`\{\phi_i\}_{i=0}^N`:

        .. math::
            \sum_{i=0}^N a_i\cdot \phi_i
            
        given `N = len(vec)` and :math:`\mathbf{a}\in\mathbb{R}^N`.

        Parameters
        ----------
        vec : np.ndarray
            Iterable containing the coefficients of the linear combination.
        
        Returns
        -------
        combination : np.ndarray or Function
            Function object storing the result of the linear combination
        """

        combination = np.zeros(self.fun_shape,)
        for ii in range(len(vec)):
            combination += vec[ii] * self._list[ii]
        return combination
        
    def min(self, axis: int = None) -> np.ndarray:
        """
        Returns the minimum of the functions in the list along the specified axis.

        Parameters
        ----------
        axis : int, optional (Default=None)
            Axis along which to compute the minimum. If None, computes the minimum over the entire array.
        
        Returns
        -------
        min_values : np.ndarray
            Minimum values of the functions in the list.
        """
        return np.min(self.return_matrix(), axis=axis)
    
    def max(self, axis: int = None) -> np.ndarray:
        """
        Returns the maximum of the functions in the list along the specified axis.

        Parameters
        ----------
        axis : int, optional (Default=None)
            Axis along which to compute the maximum. If None, computes the maximum over the entire array.
        
        Returns
        -------
        max_values : np.ndarray
            Maximum values of the functions in the list.
        """
        return np.max(self.return_matrix(), axis=axis)
    
    def mean(self, axis: int = None) -> np.ndarray:
        """
        Returns the mean of the functions in the list along the specified axis.

        Parameters
        ----------
        axis : int, optional (Default=None)
            Axis along which to compute the mean. If None, computes the mean over the entire array.
        
        Returns
        -------
        mean_values : np.ndarray
            Mean values of the functions in the list.
        """
        return np.mean(self.return_matrix(), axis=axis)
    
    def std(self, axis: int = None) -> np.ndarray:
        """
        Returns the standard deviation of the functions in the list along the specified axis.

        Parameters
        ----------
        axis : int, optional (Default=None)
            Axis along which to compute the standard deviation. If None, computes the standard deviation over the entire array.
        
        Returns
        -------
        std_values : np.ndarray
            Standard deviation values of the functions in the list.
        """
        return np.std(self.return_matrix(), axis=axis)
    
    def store(self, var_name: str, filename: str, order: list = None,
              format: str = 'h5', compression: bool = True) -> None:
        """
        Store the functions in the list to a file.

        Parameters
        ----------
        var_name : str
            Name of the variable.
        filename : str
            Name of the file to save.
        order : list, optional (Default=None)
            List of integers containing the ordered indices.
        format : str, optional (Default='h5')
            Format of the file to save. It can be either 'h5' or 'npz'.
        compression : bool, optional (Default=True)
            If `True`, the data is compressed when saved in h5/npz format.
        """
        
        _data_to_store = self.return_matrix()

        if order is not None:
            _data_to_store = _data_to_store[:, order]

        fmt = format.lower()
        if fmt == 'h5':
            filepath = filename + '.h5'
        elif fmt == 'npz':
            filepath = filename + '.npz'
        else:
            raise ValueError(f"Unsupported format {fmt}. Use 'h5' or 'npz'.")

        # Check if file exists and delete it
        if os.path.exists(filepath):
            os.remove(filepath)

        # Saving procedure in the proper format
        if fmt == 'h5':
            mode = 'w'
            comp = 'gzip' if compression else None
            with h5py.File(filepath, mode) as f:
                f.create_dataset(name=var_name, data=_data_to_store, compression=comp)
        elif fmt == 'npz':
            if compression:
                np.savez_compressed(filepath, **{var_name: _data_to_store})
            else:
                np.savez(filepath, **{var_name: _data_to_store})
                
    def plot(self, grid: pv.UnstructuredGrid, idx_to_plot: int,
                  varname: str = 'u',
                  clim: tuple = None,
                  cmap: str = 'jet',
                  resolution: int = [1080, 720],
                  title: str = None,
                  **kwargs) -> pv.Plotter:
        """
        Plot a function from the list.

        Parameters
        ----------
        grid : pyvista.UnstructuredGrid
            The PyVista grid on which the function will be plotted.
        idx_to_plot : int
            Index of the function to plot.
        varname : str
            The name to assign to the data in the PyVista grid (default is 'u').
        clim : tuple, optional
            The color limits for the plot. If None, the limits will be automatically determined from the data.
        cmap : str, optional
            The colormap to use for the plot (default is 'jet'). Other options include 'viridis', 'plasma', 'inferno', etc.
        resolution : list, optional
            The resolution of the plot in pixels (default is [1080, 720]).
        zoom : float, optional
            The zoom level for the plot (default is 1.0).   
        title : str, optional
            The title of the plot. If None, no title will be displayed.
        **kwargs : dict, optional
            Additional keyword arguments passed to the PyVista plotting functions.

        """
        
        if self.fun_shape == grid.n_points or self.fun_shape == grid.n_cells:
            snap = self._list[idx_to_plot]
        elif self.fun_shape == grid.n_points * 2 or self.fun_shape == grid.n_cells * 2:
            snap = self._list[idx_to_plot].reshape(-1, 2)
        elif self.fun_shape == grid.n_points * 3 or self.fun_shape == grid.n_cells * 3:
            snap = self._list[idx_to_plot].reshape(-1, 3)

        pl = pf(grid, snap, filename=None, varname=varname,
           clim=clim, cmap=cmap, resolution=resolution, title=title, **kwargs)
        
        pl.show(jupyter_backend='static')
        del pl
        grid.clear_cell_data()
        grid.clear_point_data()
    
    def plot_sequence(self, grid: pv.UnstructuredGrid,
                      sampling: int = 1,
                      varname: str = 'u',
                      clim: tuple = None,
                      cmap: str = 'jet',
                      resolution: int = [1080, 720],
                      title: str = None,
                      title_size: int = 20,
                      view: str = 'xy',
                      **kwargs) -> pv.Plotter:
        
        """
        Plot a sequence of functions from the list with a specified sampling rate.  
        
        Parameters  
        ----------
        grid : pyvista.UnstructuredGrid
            The PyVista grid on which the functions will be plotted.    
        sampling : int, optional (Default = 1)
            The sampling rate for the functions to be plotted. If `sampling=1`, all functions are plotted; if `sampling=2`, every second function is plotted, and so on.
        varname : str, optional (Default = 'u')
            The name to assign to the data in the PyVista grid.
        clim : tuple, optional
            The color limits for the plot. If None, the limits will be automatically determined from the data.
        cmap : str, optional (Default = 'jet')
            The colormap to use for the plot. Other options include 'viridis', 'plasma', 'inferno', etc.
        resolution : list, optional (Default = [1080, 720])
            The resolution of the plot in pixels.
        title : str, optional
            The title of the plot. If None, no title will be displayed.
        title_size : int, optional (Default = 20)
            The font size of the title.
        view : str, optional (Default = 'xy')
            The view direction for the plot. Options include 'xy', 'xz', 'yz'.
        **kwargs : dict, optional
            Additional keyword arguments passed to the PyVista plotting functions.
    
        Returns 
        ------- 
        pv.Plotter
            The PyVista plotter object used for the visualization.
        """

        pl = pv.Plotter(window_size=resolution)

        for idx in range(sampling, len(self._list), sampling):

            if self.fun_shape == grid.n_points or self.fun_shape == grid.n_cells:
                snap = self._list[idx]
            elif self.fun_shape == grid.n_points * 2 or self.fun_shape == grid.n_cells * 2:
                snap = self._list[idx].reshape(-1, 2)
            elif self.fun_shape == grid.n_points * 3 or self.fun_shape == grid.n_cells * 3:
                snap = self._list[idx].reshape(-1, 3)
                
            grid[varname] = snap
            grid.set_active_scalars(varname)

            pl.add_mesh(grid, scalars=varname, cmap=cmap, clim=clim, show_edges=False, **kwargs)

            if view == 'xy':
                pl.view_xy()
            elif view == 'xz':
                pl.view_xz()
            elif view == 'yz':
                pl.view_yz()

            if title is not None:
                pl.add_title(f"{title}{idx}", font_size=title_size, color='k')

            pl.set_background('white')
            pl.show(jupyter_backend='static', auto_close=True)
            clc(wait=True)

            grid.clear_data()
            
def train_test_split(params: list, fun_list: FunctionsList, test_size: float = 0.33, random_state: int = 42):
    """
    This function can be used to create a train and test using `sklearn` capabilities.

    Parameters
    ----------
    params : list
        List of parameters to be split.
    fun_list: FunctionsList
        Object containing the functions as a list of arrays to convert.
    test_size : float
        DimensionFunctional space of the functions (the dofs should be compliant).
    random_state : int, optional (Default = 42)
        Random seed for the splitting algorithm.
        
    Returns
    -------
    train_params : list
        List of the train parameters.
    test_params : list
        List of the test parameters.
    train_fun : list
        List of the train functions.
    test_fun : list
        List of the test functions.
    """
    
    assert len(fun_list) == len(params), "Snapshots and parameters must have the same length."
    
    res = sk_split(params, fun_list._list, test_size=test_size, random_state=random_state)
    
    # Store results - X
    train_params = res[0]
    test_params  = res[1]
    
    # Store results - Y
    train_fun = FunctionsList(dofs = fun_list.fun_shape)
    test_fun  = FunctionsList(dofs = fun_list.fun_shape)
    
    for train in res[2]:
        train_fun.append(train)    
    for test in res[3]:
        test_fun.append(test)
        
    return train_params, test_params, train_fun, test_fun