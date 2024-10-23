import pytest
import numpy as np
from dolfinx.fem import FunctionSpace, Function
from dolfinx.mesh import create_interval
from mpi4py import MPI

from pyforce.offline.pod import POD as offlinePOD
from pyforce.tools.functions_list import FunctionsList
from pyforce.offline.sensors import SGREEDY
from pyforce.offline.pbdw import PBDW

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
    return PBDW(pod_modes, sgreedy_basis, is_H1 = True)

def test_initialization(pbdw_instance):
    """Test the initialization of the PBDW class."""
    assert isinstance(pbdw_instance.basis_functions, FunctionsList)
    assert isinstance(pbdw_instance.basis_sensors, FunctionsList)
    assert pbdw_instance.norm is not None
    assert pbdw_instance.Nmax == len(pbdw_instance.basis_functions)
    assert pbdw_instance.Mmax == len(pbdw_instance.basis_sensors)

    assert pbdw_instance.A.shape == (pbdw_instance.Mmax, pbdw_instance.Mmax)
    assert pbdw_instance.K.shape == (pbdw_instance.Mmax, pbdw_instance.Nmax)
    assert pbdw_instance.Z.shape == (pbdw_instance.Nmax, pbdw_instance.Nmax)

def test_compute_infsup(pbdw_instance):
    """Test the compute_infsup method."""
    N = 5
    M = 10
    inf_sup = pbdw_instance.compute_infsup(N, M)

    assert len(inf_sup) == M
    assert np.all(inf_sup >= 0)  # Check that inf-sup values are non-negative
    assert np.all(inf_sup[:N-1] < 1e-7)  # Check that inf-sup values are non-negative