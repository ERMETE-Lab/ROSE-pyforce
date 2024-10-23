import pytest
import numpy as np
from dolfinx.fem import FunctionSpace, Function
from dolfinx.mesh import create_interval
from mpi4py import MPI

from scipy.interpolate import interp1d

from pyforce.offline.geim import GEIM as offlineGEIM
from pyforce.online.tr_geim import TRGEIM
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
    beta_coeffs = geim_instance.offline(train_snap, Mmax)[2]

    return geim_instance.magic_fun, geim_instance.magic_sens, beta_coeffs

@pytest.fixture
def setup_trgeim(magic_fun_sens):
    
    mf = magic_fun_sens[0]
    ms = magic_fun_sens[1]
    beta_coeffs = magic_fun_sens[2]

    mean_beta = np.mean(beta_coeffs, axis=0)
    std_beta  = np.std(beta_coeffs, axis=0)

    assert len(mean_beta) == len(mf)
    assert len(std_beta) == len(mf)

    name = "u"
    trgeim_instance = TRGEIM(mf, ms, mean_beta, std_beta, name)

    return trgeim_instance

def test_initialization(setup_trgeim, magic_fun_sens):


    # Test if TRGEIM is initialized properly
    assert setup_trgeim.name == "u"
    assert setup_trgeim.Mmax == len(magic_fun_sens[1])
    assert setup_trgeim.B.shape == (setup_trgeim.Mmax, setup_trgeim.Mmax)


    # Asserting that the matrix B is lower triangular with the absolute value of all elements <= 1 
    for mm in range(0, 5):
        for nn in range(5):
            action_value = setup_trgeim.B[mm, nn]
            assert abs(action_value) <= 1.0 or np.isclose(abs(action_value), 1.0, rtol=0.01)  # 1% relative tolerance

            if mm == nn:
                diag_value = setup_trgeim.B[mm, nn]
                assert np.isclose(abs(diag_value), 1.0, rtol=0.01)  # 1% relative tolerance


    assert setup_trgeim.mean_beta.shape == (setup_trgeim.Mmax,)
    assert setup_trgeim.T.shape == (setup_trgeim.Mmax, setup_trgeim.Mmax)


@pytest.mark.parametrize("M", [
    (15), # With 15 sensors
    (None), # With max sensors
])
def test_synt_test_error(setup_trgeim, test_snap, M):

    trgeim_instance = setup_trgeim

    noise_value = 0.1
    result = trgeim_instance.synt_test_error(test_snap, M=M, 
                                             noise_value=noise_value,
                                             reg_param = noise_value**2)

    if M is None:
        mmax_check = trgeim_instance.Mmax
    else:
        mmax_check = M

    assert result.mean_abs_err.shape == (mmax_check,)
    assert result.mean_rel_err.shape == (mmax_check,)
    assert isinstance(result.computational_time, dict)

@pytest.mark.parametrize("use_fun", [
    (False), # Input is a numpy.array
    (True), # Input is a dolfinx function
])
def test_reconstruct(test_snap, setup_trgeim, use_fun):

    if use_fun:
        snap = test_snap.map(0)
    else:
        snap = test_snap(0)

    M = 3
    noise_value = 0.1
    reg_param = noise_value**2
    interp, resid, computational_time, coeff = setup_trgeim.reconstruct(snap=snap, M=M, noise_value=noise_value, reg_param=reg_param)

    assert np.linalg.norm(interp) > 0 # Should return a non-zero function

    assert interp is not None
    assert resid is not None
    assert coeff.shape == (M,)  # Number of sensors used

@pytest.mark.parametrize("use_fun", [
    (False), # Input is a numpy.array
    (True), # Input is a dolfinx function
])
def test_compute_measure(test_snap, setup_trgeim, use_fun):

    if use_fun:
        snap = test_snap.map(0)
    else:
        snap = test_snap(0)

    noise_value = 0.1
    measures = setup_trgeim.compute_measure(snap=snap, noise_value=noise_value, M=3)
    
    assert measures.shape == (3,)  # Should return a measurement vector of length M
    assert np.linalg.norm(measures) > 0 # Should return a non-zero vector

def test_real_reconstruct(setup_trgeim):
    measure = np.random.rand(3, 5)  # Dummy measure for 3 sensors and 5 realizations
    reg_param = 0.5
    interps, computational_time = setup_trgeim.real_reconstruct(measure=measure, reg_param=reg_param)

    assert len(interps) == 5  # Should return interpolants for each realization