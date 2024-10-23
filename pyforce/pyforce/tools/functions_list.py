# List of functions for storage
# Author: Stefano Riva, PhD Student, NRG, Politecnico di Milano
# Latest Code Update: 24 May 2024
# Latest Doc  Update: 24 May 2024

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
    function_space : FunctionSpace, optional
        Functional Space onto which the Function are defined. If not provided, `dofs` must be specified.
    dofs : int, optional
        Degrees of freedom of the functions :math:`\mathcal{N}_h`. Required if `function_space` is not provided.
    """
    def __init__(self, function_space : FunctionSpace = None, dofs: int = None) -> None:
        self.fun_space = function_space
        self._list = list()

        if self.fun_space:
            tmp_fun = Function(self.fun_space).copy()
            self.fun_shape = tmp_fun.x.array.shape[0]
        elif dofs is not None:
            self.fun_shape = dofs
        else:
            raise ValueError("Either `function_space` or `dofs` must be provided.")

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
            assert dofs_fun.x.array.shape[0] == self.fun_shape, "The input function dofs_fun has "+str(dofs_fun.x.array.shape[0])+", instead of "+str(self.fun_shape)
            self._list.append(dofs_fun.x.array[:])
        else:
            assert dofs_fun.shape[0] == self.fun_shape, "The input function dofs_fun has "+str(dofs_fun.shape[0])+", instead of "+str(self.fun_shape)
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

       assert len(tmp) == len(order), "The order vector ("+str(len(order))+") must have the same length of the list ("+str(len(tmp))+")"
       for ii in range(len(order)):
          self.append(tmp[order[ii]])
    
    def return_matrix(self) -> np.ndarray:
        r"""
        Returns the list of arrays as a matrix :math:`S\in\mathbb{R}^{\mathcal{N}_h\times N_s}`.
        """
        
        return np.asarray(self._list).T

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
        if not self.fun_space:
            raise ValueError("Function space not defined.")
            
        eval_fun = Function(self.fun_space).copy()
        with eval_fun.vector.localForm() as loc:
            loc.set(0.0)
        eval_fun.x.array[:] = self._list[idx]
        eval_fun.x.scatter_forward()
        return eval_fun

    def lin_combine(self, vec: np.ndarray, use_numpy=True) -> np.ndarray:
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
        combination : np.ndarray or Function
            Function object storing the result of the linear combination
        """
        if use_numpy or not self.fun_space:
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
    if fun_list is not None:
        train_fun = FunctionsList(function_space = fun_list.fun_space)
        test_fun  = FunctionsList(function_space = fun_list.fun_space)
    else:
        train_fun = FunctionsList(dofs=fun_list.fun_shape)
        test_fun  = FunctionsList(dofs=fun_list.fun_shape)
    
    for train in res[2]:
        train_fun.append(train)    
    for test in res[3]:
        test_fun.append(test)
        
    return train_params, test_params, train_fun, test_fun