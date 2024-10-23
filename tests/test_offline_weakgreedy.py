import pytest
import numpy as np
from dolfinx.fem import Function, FunctionSpace
from dolfinx.mesh import create_unit_square
from mpi4py import MPI
from pyforce.tools.backends import norms, LoopProgress
from pyforce.tools.functions_list import FunctionsList
from pyforce.offline.weak_greedy import WeakGreedy as WG 

@pytest.fixture
def function_space():
    domain = create_unit_square(MPI.COMM_WORLD, 10, 10)
    return FunctionSpace(domain, ("Lagrange", 1))

@pytest.fixture
def snaps(function_space):
    snap = FunctionsList(function_space)
    
    psi = np.linspace(0.25, 1, 20)
    for pp in psi:
        f = Function(function_space)
        f.interpolate(lambda x: np.sin(np.pi * pp * x[0]) + np.cos(np.pi * pp * x[1]))
        snap.append(f)
    return snap

@pytest.fixture
def setup_wg(function_space):
    
    name = "u"
    wg_instance = WG(function_space, name)

    return wg_instance

# Test WeakGreedy initialization
def test_weak_greedy_init(setup_wg):
    
    assert isinstance(setup_wg.V, FunctionSpace)
    assert setup_wg.name == "u"

# Test basis computation
def test_compute_basis(setup_wg, snaps):
    
    N = 5  # Set maximum number of basis modes to compute
    
    # Call the WeakGreedy basis computation
    maxAbsErr, maxRelErr, alpha_coeff = setup_wg.compute_basis(snaps, N=N)
    
    # Test if returned errors and coefficients have the expected shape
    assert maxAbsErr.shape == (N,), f"Expected maxAbsErr shape ({N},), got {maxAbsErr.shape}"
    assert maxRelErr.shape == (N,), f"Expected maxRelErr shape ({N},), got {maxRelErr.shape}"
    assert alpha_coeff.shape == (len(snaps), N), f"Expected alpha_coeff shape ({len(snaps)}, {N}), got {alpha_coeff.shape}"

# Test Gram-Schmidt orthonormalization
def test_gram_schmidt(setup_wg, snaps):
    
    N = 2  # Set maximum number of basis modes to compute
    
    # Call the WeakGreedy basis computation
    _, _, _ = setup_wg.compute_basis(snaps, N=N)
    
    # Take the 17th snapshot (not selected by the greedy procedure) and normalize it using Gram-Schmidt
    fun = snaps(17)
    normalized_fun = setup_wg.GrahmSchmidt(fun)
    
    # Test if the normalized function has L2 norm close to 1
    norm_value = setup_wg.norm.L2norm(normalized_fun)
    assert np.isclose(norm_value, 1.0, rtol=1e-3), f"Expected L2 norm 1.0, got {norm_value}"

    for ii in range(N):
        dot_prod = setup_wg.norm.L2innerProd(normalized_fun, setup_wg.basis(ii))
        assert np.isclose(dot_prod, 0.0, rtol=1e-3), f"Expected inner product 0.0, got {dot_prod}"