from dolfinx import mesh
from dolfinx.fem import FunctionSpace, Function
import numpy as np
from mpi4py import MPI
from pyforce.tools.backends import norms

def test_integral():
    domain = mesh.create_unit_interval(MPI.COMM_WORLD, 100)
    V = FunctionSpace(domain, ("Lagrange", 1))
    norm_calculator = norms(V)
    
    u = Function(V)
    u.interpolate(lambda x: x[0]**2)
    
    integral_value = norm_calculator.integral(u)
    expected_value = 1/3  # Integral of x^2 from 0 to 1 is 1/3
    assert np.isclose(integral_value, expected_value, atol=1e-3)

def test_average():
    domain = mesh.create_interval(MPI.COMM_WORLD, 50, [0,2])
    V = FunctionSpace(domain, ("Lagrange", 1))
    norm_calculator = norms(V)
    
    u = Function(V)
    u.interpolate(lambda x: x[0]**2)
    
    average_value = norm_calculator.average(u)
    expected_value = 4/3  # Average of x^2 over [0, 1] is 1/4
    print(average_value)
    assert np.isclose(average_value, expected_value, atol=1e-3)

def test_L2norm():
    domain = mesh.create_unit_interval(MPI.COMM_WORLD, 50)
    V = FunctionSpace(domain, ("Lagrange", 1))
    norm_calculator = norms(V)
    
    u = Function(V)
    u.interpolate(lambda x: x[0])
    
    L2norm_value = norm_calculator.L2norm(u)
    expected_value = np.sqrt(1/3)  # L2 norm of x over [0, 1] is sqrt(1/3)
    assert np.isclose(L2norm_value, expected_value, atol=1e-3)

def test_H1norm():
    domain = mesh.create_unit_interval(MPI.COMM_WORLD, 50)
    V = FunctionSpace(domain, ("Lagrange", 1))
    norm_calculator = norms(V, is_H1=True)
    
    u = Function(V)
    u.interpolate(lambda x: x[0])
    
    H1_seminorm_value = norm_calculator.H1norm(u, semi=True)
    expected_seminorm = np.sqrt(1)  # H1 seminorm of x over [0, 1] is sqrt(1)
    assert np.isclose(H1_seminorm_value, expected_seminorm, atol=1e-3)
    
    H1_fullnorm_value = norm_calculator.H1norm(u, semi=False)
    expected_fullnorm = np.sqrt(1 + 1/3)  # H1 norm of x over [0, 1] is sqrt(1 + 1/3)
    assert np.isclose(H1_fullnorm_value, expected_fullnorm, atol=1e-3)

def test_Linftynorm():
    domain = mesh.create_unit_interval(MPI.COMM_WORLD, 50)
    V = FunctionSpace(domain, ("Lagrange", 1))
    norm_calculator = norms(V)
    
    u = Function(V)
    u.interpolate(lambda x: x[0]**2)
    
    Linftynorm_value = norm_calculator.Linftynorm(u)
    expected_value = 1  # L-infinity norm of x^2 over [0, 1] is 1
    assert np.isclose(Linftynorm_value, expected_value, atol=1e-3)

def test_L2innerProd():
    domain = mesh.create_unit_interval(MPI.COMM_WORLD, 50)
    V = FunctionSpace(domain, ("Lagrange", 1))
    norm_calculator = norms(V)
    
    u = Function(V)
    v = Function(V)
    u.interpolate(lambda x: x[0])
    v.interpolate(lambda x: 2*x[0])
    
    inner_prod_value = norm_calculator.L2innerProd(u, v)
    expected_value = 2 * 1/3  # L2 inner product of x and 2x over [0, 1] is 2/3
    assert np.isclose(inner_prod_value, expected_value, atol=1e-3)

def test_H1innerProd():
    domain = mesh.create_unit_interval(MPI.COMM_WORLD, 50)
    V = FunctionSpace(domain, ("Lagrange", 1))
    norm_calculator = norms(V, is_H1=True)
    
    u = Function(V)
    v = Function(V)
    u.interpolate(lambda x: x[0])
    v.interpolate(lambda x: 2*x[0])
    
    inner_prod_semi_value = norm_calculator.H1innerProd(u, v, semi=True)
    expected_semi_value = 2  # H1 semi-inner product of x and 2x over [0, 1] is 2
    assert np.isclose(inner_prod_semi_value, expected_semi_value, atol=1e-3)
    
    inner_prod_full_value = norm_calculator.H1innerProd(u, v, semi=False)
    expected_full_value = 2 + 2 * 1/3  # H1 full inner product of x and 2x over [0, 1] is 2 + 2/3
    assert np.isclose(inner_prod_full_value, expected_full_value, atol=1e-3)