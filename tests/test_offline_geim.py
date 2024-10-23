import pytest
import numpy as np
from dolfinx.fem import FunctionSpace, Function
from dolfinx.mesh import create_interval
from mpi4py import MPI

from pyforce.tools.functions_list import FunctionsList
from pyforce.offline.geim import GEIM, computeLebesgue

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
def setup_geim(domain, V):
    
    name = "Temperature"
    s = 1e-2
    geim_instance = GEIM(domain, V, name, s)

    return geim_instance

def test_geim_initialization(setup_geim):
    geim_instance = setup_geim
    assert geim_instance.name == "Temperature"
    assert isinstance(geim_instance.V, FunctionSpace)

def test_offline(setup_geim, train_snap):
    Mmax = 5
    maxAbsErr, maxRelErr, beta_coeff = setup_geim.offline(train_snap, Mmax)

    assert maxAbsErr.shape[0] == Mmax
    assert maxRelErr.shape[0] == Mmax
    assert beta_coeff.shape == (len(train_snap), Mmax)

    assert isinstance(setup_geim.B, np.ndarray)

    # Asserting that the matrix B is lower triangular with the absolute value of all elements <= 1 
    assert np.allclose(setup_geim.sens_class.action_single( setup_geim.magic_fun(0), 
                                                            setup_geim.magic_sens(0)), 
                        1.0, rtol=1e-2)
    for mm in range(1, Mmax):
        assert np.allclose(setup_geim.sens_class.action_single( setup_geim.magic_fun(mm), 
                                                                setup_geim.magic_sens(0)), 
                            0.0, rtol=1e-2)
        
        for nn in range(Mmax):
            action_value = setup_geim.sens_class.action_single( setup_geim.magic_fun(mm), 
                                                                setup_geim.magic_sens(nn))
            assert abs(action_value) <= 1.0 or np.isclose(abs(action_value), 1.0, rtol=0.01)  # 1% relative tolerance

def test_reconstruct(setup_geim, train_snap, test_snap):
    Mmax = 5
    setup_geim.offline(train_snap, Mmax)

    # Test reconstruction
    snap_to_reconstruct = test_snap(0)  # Use the first snapshot for reconstruction
    beta_coeff, measure = setup_geim.reconstruct(snap_to_reconstruct, Mmax)

    assert beta_coeff.shape[0] == Mmax
    assert measure.shape[0] == Mmax

def test_test_error(setup_geim, train_snap, test_snap):
    Mmax = 5
    setup_geim.offline(train_snap, Mmax)

    meanAbsErr, meanRelErr, coeff_matrix = setup_geim.test_error(test_snap, Mmax)

    assert meanAbsErr.shape[0] == Mmax
    assert meanRelErr.shape[0] == Mmax
    assert coeff_matrix.shape == (len(test_snap), Mmax)

def test_computeLebesgue(setup_geim, train_snap):

    Mmax = 5
    setup_geim.offline(train_snap, Mmax)

    # Create mock magic functions and sensors
    size = 5  # Number of magic functions/sensors
    
    # Call the function
    leb_constant = computeLebesgue(setup_geim.magic_fun, setup_geim.magic_sens)

    # Check that the output is an array of the correct shape
    assert leb_constant.shape == (size,)
    # Check that the Lebesgue constant values are non-negative (you might want to adjust this based on your knowledge of the expected output)
    assert np.all(leb_constant >= 0)