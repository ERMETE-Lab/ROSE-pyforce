import numpy as np
import pytest
from pyforce.tools.functions_list import FunctionsList, train_test_split
from dolfinx.fem import FunctionSpace, Function
from dolfinx import mesh
from mpi4py import MPI

@pytest.fixture
def function_space():
    domain = mesh.create_interval(MPI.COMM_WORLD, 10, [0,1])  # Define or mock a mesh object
    return FunctionSpace(domain, ("Lagrange", 1))

@pytest.fixture
def example_functions(function_space: FunctionSpace):
    # Create example functions for testing
    fun1 = Function(function_space)
    fun1.x.array[:] = np.linspace(0, 1, function_space.tabulate_dof_coordinates().shape[0])  # Mock function values
    fun2 = Function(function_space)
    fun2.x.array[:] = np.linspace(0, 2, function_space.tabulate_dof_coordinates().shape[0])  # Mock function values
    return [fun1, fun2]

def test_functions_list_initialization(function_space: FunctionSpace):
    # Test initialization of FunctionsList
    fun_list = FunctionsList(function_space)
    assert len(fun_list) == 0

def test_functions_list_append(function_space: FunctionSpace, example_functions: list[Function]):
    # Test appending functions to FunctionsList
    fun_list = FunctionsList(function_space)
    fun_list.append(example_functions[0])
    fun_list.append(example_functions[1])
    assert len(fun_list) == 2

def test_functions_list_call(function_space: Function, example_functions: list[Function]):
    # Test calling FunctionsList to retrieve a function
    fun_list = FunctionsList(function_space)
    fun_list.append(example_functions[0])
    retrieved_fun = fun_list(0)
    assert np.array_equal(retrieved_fun, example_functions[0].x.array[:])

# def test_functions_matrix_return_matrix(example_functions):
#     # Test returning matrix from FunctionsMatrix
#     fun_matrix = FunctionsMatrix(3)
#     fun_matrix.append(example_functions[0].x.array)
#     fun_matrix.append(example_functions[1].x.array)
#     matrix = fun_matrix.return_matrix()
#     assert np.array_equal(matrix, np.array([[1, 4], [2, 5], [3, 6]]))

# def test_train_test_split(example_functions):
#     # Test train test split function
#     params = [1, 2]  # Mock parameters
#     train_params, test_params, train_fun, test_fun = train_test_split(params, example_functions)
#     assert len(train_params) == len(train_fun) == 1
#     assert len(test_params) == len(test_fun) == 1

# Additional tests can be added to cover more scenarios and edge cases
