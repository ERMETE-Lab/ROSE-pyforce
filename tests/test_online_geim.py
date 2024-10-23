import pytest
import numpy as np
from dolfinx.fem import FunctionSpace, Function
from dolfinx.mesh import create_interval
from mpi4py import MPI

from scipy.interpolate import interp1d

from pyforce.offline.geim import GEIM as offlineGEIM
from pyforce.online.geim import GEIM
from pyforce.tools.functions_list import FunctionsList

@pytest.fixture
def domain():
    return create_interval(MPI.COMM_WORLD, 200, [-1, 1])

@pytest.fixture
def V(domain):
    return FunctionSpace(domain, ("Lagrange", 1))

@pytest.fixture
def mu_train():
    return np.linspace(0.25, 2, 100).reshape(-1,1)

@pytest.fixture
def mu_test():
    return np.linspace(0.27, 1.9, 100).reshape(-1,1)

@pytest.fixture
def train_snap(V, mu_train):
    # Create a mock training snapshot
    _train_snap = FunctionsList(V)

    x = V.tabulate_dof_coordinates()[:,0]
    for ii in range(len(mu_train)):
        _train_snap.append((1-x) * np.cos(np.pi * mu_train[ii]**2 * (1+x)) * np.exp(-mu_train[ii] * (1+x)))

    return _train_snap

@pytest.fixture
def test_snap(V, mu_test):
    # Create a mock training snapshot
    _test_snap = FunctionsList(V)
    
    x = V.tabulate_dof_coordinates()[:,0]
    for ii in range(len(mu_test)):
        _test_snap.append((1-x) * np.cos(np.pi * mu_test[ii]**2 * (1+x)) * np.exp(-mu_test[ii] * (1+x)))

    return _test_snap

@pytest.fixture
def magic_fun_sens(domain, V, train_snap):

    name = "u"
    s = 1e-2
    geim_instance = offlineGEIM(domain, V, name, s)

    Mmax = 20
    geim_instance.offline(train_snap, Mmax)

    return geim_instance.magic_fun, geim_instance.magic_sens

@pytest.fixture
def setup_geim(magic_fun_sens):
    
    name = "u"
    geim_instance = GEIM(*magic_fun_sens, name)

    return geim_instance


def test_initialization(setup_geim, magic_fun_sens):
    geim_instance = setup_geim

    assert geim_instance.name == "u"
    assert geim_instance.Mmax == len(magic_fun_sens[1])
    assert geim_instance.B.shape == (geim_instance.Mmax, geim_instance.Mmax)

    # Asserting that the matrix B is lower triangular with the absolute value of all elements <= 1 
    for mm in range(0, 5):
        for nn in range(5):
            action_value = geim_instance.B[mm, nn]
            assert abs(action_value) <= 1.0 or np.isclose(abs(action_value), 1.0, rtol=0.01)  # 1% relative tolerance

            if mm == nn:
                diag_value = geim_instance.B[mm, nn]
                assert np.isclose(abs(diag_value), 1.0, rtol=0.01)  # 1% relative tolerance


@pytest.mark.parametrize("noise_value, M", [
    (None, 15), # No noise, with 15 sensors
    (None, None), # No noise, with max sensors
    (0.1, 15),  # Noisy data, with 15 sensors
    (0.1, None),  # Noisy data, with max sensors
])
def test_synt_test_error(setup_geim, test_snap, noise_value, M):

    geim_instance = setup_geim

    result = geim_instance.synt_test_error(test_snap, M=M, noise_value=noise_value)

    if M is None:
        mmax_check = geim_instance.Mmax
    else:
        mmax_check = M

    assert result.mean_abs_err.shape == (mmax_check,)
    assert result.mean_rel_err.shape == (mmax_check,)
    assert isinstance(result.computational_time, dict)

@pytest.mark.parametrize("use_fun, noise_value", [
    (False, None), # Input is a numpy.array without noise
    (True, 0.1), # Input is a dolfinx function with noise
    (False, 0.1), # Input is a numpy.array with noise
    (True, None), # Input is a dolfinx function without noise
])
def test_reconstruct(test_snap, setup_geim, use_fun, noise_value):

    if use_fun:
        snap = test_snap.map(0)
    else:
        snap = test_snap(0)

    M = 3
    interp, resid, computational_time = setup_geim.reconstruct(snap=snap, M=M, 
                                                                noise_value=noise_value)

    assert np.linalg.norm(interp) > 0 # Should return a non-zero function

    assert interp is not None
    assert resid is not None

@pytest.mark.parametrize("use_fun", [
    (False), # Input is a numpy.array
    (True), # Input is a dolfinx function
])
def test_compute_measure(test_snap, setup_geim, use_fun):

    if use_fun:
        snap = test_snap.map(0)
    else:
        snap = test_snap(0)

    measures = setup_geim.compute_measure(snap=snap, M=3)
    
    assert measures.shape == (3,)  # Should return a measurement vector of length M
    assert np.linalg.norm(measures) > 0 # Should return a non-zero vector

def test_real_reconstruct(setup_geim):
    measure = np.random.rand(3, 5)  # Dummy measure for 3 sensors and 5 realizations
    interps, computational_time = setup_geim.real_reconstruct(measure=measure)

    assert len(interps) == 5  # Should return interpolants for each realization