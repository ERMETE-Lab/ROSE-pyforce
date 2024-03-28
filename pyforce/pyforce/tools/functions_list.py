# List of functions for storage
# Author: Stefano Riva, PhD Student, NRG, Politecnico di Milano
# Latest Code Update: 02 August 2023
# Latest Doc  Update: 02 August 2023

import numpy as np
import warnings
from sklearn.model_selection import train_test_split as sk_split

from dolfinx.fem import (Function, FunctionSpace)
from petsc4py import PETSc

# Class to define a list of function, useful to collect snapshots and basis functions
class FunctionsList():
    r"""
    A class wrapping a list of functions. They are stored as a `list` of `np.ndarray`.

    Parameters
    ----------
    function_space : FunctionSpace
        Functional Space onto which the Function are defined.

    """
    def __init__(self, function_space: FunctionSpace) -> None:
        self.fun_space = function_space
        self._list = list()

        tmp_fun = Function(self.fun_space).copy()
        self.fun_shape = tmp_fun.x.array.shape

    def __call__(self, idx) -> np.ndarray:
        """
        Defining the class as callable, returns the idx-th element of the list
        """
        return self._list[idx]
    
    def __len__(self) -> int:
        """
        Defining the length of the class as the length of the stored list
        """
        return len(self._list)

    def append(self, dofs_fun: np.ndarray) -> None:
        """
        Extend the current list by adding a new function.
        The dolfinx.fem.Function element is stored in a list as a numpy array, to avoid problems when the number of elements becomes large.
        The input can be either a `np.ndarray` or a `dolfinx.fem.Function`, in the latter case it is mapped to `np.ndarray`.

        Parameters
        ----------
        dofs_fun : np.ndarray
            Functions to be appended.

        """

        if isinstance(dofs_fun, Function):
            assert(dofs_fun.x.array.shape == self.fun_shape)
            self._list.append(dofs_fun.x.array[:])
        else:
            assert(dofs_fun.shape == self.fun_shape)
            self._list.append(dofs_fun)

    def delete(self, idx: int) -> None:
        """
        Delete a single element in position `idx`.
        
        Parameters
        ----------
        idx : int
            Integers indicating the position inside the `_list` to delete.
        """
        del self._list[idx]

    def copy(self):
        """
        Defining the copy of the `_list` of elements
        """
        
        return self._list.copy()

    def clear(self) -> None:
        """Clear the storage."""
        self._list = list()

    def sort(self, order: list) -> None:
       """
        Sorting the list according to order - iterable of indices.


        Parameters
        ----------
        order : list
            List of indices for the sorting.
       
       """

       tmp = self._list.copy()
       self.clear()

       assert len(tmp) == len(order)
       for ii in range(len(order)):
          self.append(tmp[order[ii]])

    def map(self, idx: int) -> Function:
        """ 
        Mapping the element in position `idx` into a `dolfinx.fem.Function`.
        
        Parameters
        ----------
        idx : int
            Integers indicating the position inside the `_list`.
        
        Returns
        -------
        eval_fun : Function
            Evaluated dofs into a Function
        """

        eval_fun = Function(self.fun_space).copy()
        with eval_fun.vector.localForm() as loc:
            loc.set(0.0)
        eval_fun.x.array[:] = self._list[idx]
        eval_fun.x.scatter_forward()
        return eval_fun

    def lin_combine(self, vec: np.ndarray, use_numpy=True) -> Function:
        r"""
        Linearly combine functions (`a_i = vec[i]`) in the list :math:`\{\phi_i\}_{i=0}^N`:

        .. math::
            \sum_{i=0}^N a_i\cdot \phi_i
            
        given `N = len(vec)` and :math:`\mathbf{a}\in\mathbb{R}^N`.

        Parameters
        ----------
        vec : np.ndarray
            Iterable containing the coefficients of the linear combination.
        use_numpy : boolean, optional (Default=True)
            If `True` the functions are treated as `np.ndarray`, otherwise the formulation in `dolfinx` is used.
        
        Returns
        -------
        combination : Function
            Function object storing the result of the linear combination
        """
        
        if use_numpy:
            combination = np.zeros(self.fun_shape,)
            for ii in range(len(vec)):
                combination += vec[ii] * self._list[ii]
            return combination
        else:
            combination = Function(self.fun_space).copy()
            with combination.vector.localForm() as loc:
                loc.set(0.0)
            for ii in range(len(vec)):
                combination.vector.axpy(vec[ii], self.map(ii).vector)
            combination.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            return combination
    
# Class to define a list of function, useful to collect snapshots and basis functions
class FunctionsMatrix():
    r"""
    A class wrapping a list of functions without relying on *dolfinx* capabilities. They are stored as a `list` of `np.ndarray`.

    Parameters
    ----------
    dofs : np.ndarray
        Degrees of freedom of the functions :math:`\mathcal{N}_h`.

    """
    def __init__(self, dofs: float) -> None:
        self.dofs = dofs
        self._list = list()

    def __call__(self, idx) -> np.ndarray:
        """
        Defining the class as callable, returns the idx-th element of the list
        """
        return self._list[idx]
    
    def __len__(self) -> int:
        """
        Defining the length of the class as the length of the stored list
        """
        return len(self._list)
    
    def return_matrix(self) -> np.ndarray:
        r"""
        Returns the list of arrays as a matrix :math:`S\in\mathbb{R}^{\mathcal{N}_h\times N_s}`.
        """
        
        return np.asarray(self._list).T

    def append(self, dofs_fun: np.ndarray) -> None:
        """
        Extend the current list by adding a new function. The input must be a `np.ndarray` object.

        Parameters
        ----------
        dofs_fun : np.ndarray
            Functions to be appended.

        """

        if isinstance(dofs_fun, Function):
            assert(dofs_fun.x.array.shape == self.dofs)
            self._list.append(dofs_fun.x.array[:])
        else:
            assert(len(dofs_fun) == self.dofs)
            self._list.append(dofs_fun)

    def delete(self, idx: int) -> None:
        """
        Delete a single element in position `idx`.
        
        Parameters
        ----------
        idx : int
            Integers indicating the position inside the `_list` to delete.
        """
        del self._list[idx]

    def copy(self):
        """
        Defining the copy of the `_list` of elements
        """
        
        return self._list.copy()

    def clear(self) -> None:
        """Clear the storage."""
        self._list = list()

    def sort(self, order: list) -> None:
       """
        Sorting the list according to order - iterable of indices.


        Parameters
        ----------
        order : list
            List of indices for the sorting.
       
       """

       tmp = self._list.copy()
       self.clear()

       assert len(tmp) == len(order)
       for ii in range(len(order)):
          self.append(tmp[order[ii]])

    def lin_combine(self, vec: np.ndarray) -> Function:
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
        combination : Function
            Function object storing the result of the linear combination
        """
        
        combination = np.zeros((self.dofs,))

        for ii in range(len(vec)):
            combination += vec[ii] * self._list[ii]
        
        return combination
    
    
def fun_list_2_fun_matrix(fun_list: FunctionsList):
    """
    This function can be used to convert a `FunctionsList` object to a `FunctionsMatrix` one.

    Parameters
    ----------
    fun_list : FunctionsList
        Object with the list of functions to convert.
        
    Returns
    -------
    fun_matrix: FunctionsMatrix
        Object containing the functions list converted to a list of arrays, not relying on *dolfinx*.
    """
    fun_matrix = FunctionsMatrix(fun_list.fun_shape[0])
    
    for ii in range(len(fun_list)):
        fun_matrix.append(fun_list(ii))
        
    return fun_matrix

def fun_matrix_2_fun_list(fun_matrix: FunctionsMatrix, V: FunctionSpace):
    """
    This function can be used to convert a `FunctionsMatrix` object to a `FunctionsList` one.

    Parameters
    ----------
    fun_matrix: FunctionsMatrix
        Object containing the functions as a list of arrays to convert.
    V : FunctionSpace
        Functional space of the functions (the dofs should be compliant).
        
    Returns
    -------
    fun_list : FunctionsList
        Object with the list of functions converted.
    """
    fun_list = FunctionsList(V)
    
    assert(fun_list.fun_shape == fun_matrix.dofs)
    
    for ii in range(len(fun_matrix)):
        fun_list.append(fun_matrix(ii))
        
    return fun_list

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
    
    assert len(fun_list) == len(params),"Snapshots and parameters must have the same length."
    
    res = sk_split(params, fun_list._list, test_size=test_size, random_state=random_state)
    
    # Store results - X
    train_params = res[0]
    test_params  = res[1]
    
    # Store results - Y
    train_fun = FunctionsList(fun_list.fun_space)
    for train in res[2]:
        train_fun.append(train)
        
    test_fun  = FunctionsList(fun_list.fun_space)
    for test in res[3]:
        test_fun.append(test)
        
    return train_params, test_params, train_fun, test_fun