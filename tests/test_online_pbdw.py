import pytest
import numpy as np
from dolfinx.fem import FunctionSpace, Function
from dolfinx.mesh import create_interval
from mpi4py import MPI

from pyforce.offline.pod import POD as offlinePOD
from pyforce.tools.functions_list import FunctionsList
from pyforce.offline.sensors import SGREEDY
from pyforce.online.pbdw import PBDW

@pytest.fixture
def domain():
    return create_interval(MPI.COMM_WORLD, 200, [-1, 1])

@pytest.fixture
def V(domain):
    return FunctionSpace(domain, ("Lagrange", 1))

@pytest.fixture
def train_snap(V):
    # Create a mock training snapshot
    _train_snap = FunctionsList(V)
    mu_train = np.linspace(0.25, 2, 100)

    x = V.tabulate_dof_coordinates()[:,0]
    for ii in range(len(mu_train)):
        _train_snap.append((1-x) * np.cos(np.pi * mu_train[ii]**2 * (1+x)) * np.exp(-mu_train[ii] * (1+x)))

    return _train_snap

@pytest.fixture
def test_snap(V):
    # Create a mock training snapshot
    _test_snap = FunctionsList(V)
    mu_test = np.linspace(0.27, 1.9, 100)

    x = V.tabulate_dof_coordinates()[:,0]
    for ii in range(len(mu_test)):
        _test_snap.append((1-x) * np.cos(np.pi * mu_test[ii]**2 * (1+x)) * np.exp(-mu_test[ii] * (1+x)))

    return _test_snap

@pytest.fixture
def pod_modes(train_snap):
    offlinePOD_instance = offlinePOD(train_snap, 'u')
    offlinePOD_instance.compute_basis(train_snap, maxBasis = 20)

    pod_modes = FunctionsList(offlinePOD_instance.PODmodes.fun_space)
    for mode_i in range(5):
        pod_modes.append(offlinePOD_instance.PODmodes(mode_i))

    return pod_modes

@pytest.fixture
def sgreedy_basis(domain, pod_modes, V):
    s = 0.1  # Standard deviation of the Gaussian kernel
    sgreedy_instance = SGREEDY(domain, pod_modes, V, name="Test Snapshot", s=s)

    N = 5
    M = 10

    sgreedy_instance.generate(N=N, Mmax=M, xm=None, is_H1=True)

    return sgreedy_instance.basis_sens

@pytest.fixture
def pbdw_instance(pod_modes, sgreedy_basis):
    return PBDW(pod_modes, sgreedy_basis, name='u', is_H1=True)

def test_initialization(pbdw_instance):
    """Test the initialization of the PBDW class."""
    assert isinstance(pbdw_instance.basis_functions, FunctionsList)
    assert isinstance(pbdw_instance.basis_sensors, FunctionsList)
    assert pbdw_instance.norm is not None
    assert pbdw_instance.Nmax == len(pbdw_instance.basis_functions)
    assert pbdw_instance.Mmax == len(pbdw_instance.basis_sensors)

    assert pbdw_instance.A.shape == (pbdw_instance.Mmax, pbdw_instance.Mmax)
    assert pbdw_instance.K.shape == (pbdw_instance.Mmax, pbdw_instance.Nmax)
    
def test_pbdw_synt_test_error(pbdw_instance, test_snap):
    
    # Test synthetic error computation
    N = 3
    M = 5
    noise_value = 0.01
    reg_param = 0.1
    verbose = False

    result = pbdw_instance.synt_test_error(test_snap, N=N, M=M, noise_value=noise_value, reg_param=reg_param, verbose=verbose)

    assert isinstance(result.mean_abs_err, np.ndarray), "The mean_abs_err should be an ndarray"
    assert isinstance(result.mean_rel_err, np.ndarray), "The mean_rel_err should be an ndarray"
    assert isinstance(result.computational_time, dict), "The computational_time should be a dict"
    assert result.mean_abs_err.size > 0, "The result for mean_abs_err should not be empty"
    assert result.mean_rel_err.size > 0, "The result for mean_rel_err should not be empty"


def test_pbdw_reconstruct(pbdw_instance, test_snap):
    # Test reconstruction functionality
    N = 3
    M = 5
    noise_value = 0.01
    reg_param = 0.1

    snap = test_snap(0)  # Taking the first test snapshot
    recon, resid, computational_time = pbdw_instance.reconstruct(snap, N=N, M=M, noise_value=noise_value, reg_param=reg_param)

    assert recon is not None, "The reconstruction should not be None"
    assert resid is not None, "The residual should not be None"
    assert isinstance(computational_time, dict), "The computational_time should be a dict"
    assert recon.size == snap.size, "The reconstructed field should have the same size as the snapshot"
    assert resid.size == snap.size, "The residual field should have the same size as the snapshot"

def test_pbdw_compute_measure(pbdw_instance, V):
    # Test measurement computation
    noise_value = 0.01
    M = 5

    # Create a simple function for testing
    snap = Function(V)
    snap.interpolate(lambda x: (1 - x[0]) * np.cos(np.pi * x[0]**2))

    measure = pbdw_instance.compute_measure(snap, noise_value=noise_value, M=M)

    assert isinstance(measure, np.ndarray), "The measurement should be an ndarray"
    assert measure.size == M, f"The measurement vector should have size {M}, but got {measure.size}"
    assert np.all(np.isfinite(measure)), "The measurement vector should not contain infinite or NaN values"


def test_real_reconstruct(pbdw_instance):
    
    # Test data
    M = 8
    Ns = 4
    measure = np.random.rand(M, Ns)
    N = 4
    reg_param = 0.1

    # Call the method
    interp, computational_time = pbdw_instance.real_reconstruct(measure, N, reg_param)

    # Assertions
    assert isinstance(interp, FunctionsList), "The output should be a FunctionsList"
    assert isinstance(computational_time, dict), "Computational time should be returned as a dictionary"
    assert 'LinearSystem' in computational_time, "The key 'LinearSystem' should be in the computational time dict"
    assert 'Reconstruction' in computational_time, "The key 'Reconstruction' should be in the computational time dict"
