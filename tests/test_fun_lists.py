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

def test_functions_list_call(function_space: FunctionSpace, example_functions: list[Function]):
    # Test calling FunctionsList to retrieve a function
    fun_list = FunctionsList(function_space)
    fun_list.append(example_functions[0])
    retrieved_fun = fun_list(0)
    assert np.array_equal(retrieved_fun, example_functions[0].x.array[:])

def test_train_test_split(function_space: Function, example_functions: list[Function]):
    
    # Test train test split function
    fun_list = FunctionsList(function_space) # Mock FunctionsList
    fun_list.append(example_functions[0])
    fun_list.append(example_functions[1])
    params = [1, 2]  # Mock parameters
    train_params, test_params, train_fun, test_fun = train_test_split(params, fun_list)
    assert len(train_params) == len(train_fun) == 1
    assert len(test_params) == len(test_fun) == 1