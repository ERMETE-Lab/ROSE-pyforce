import pytest
import numpy as np
from dolfinx.fem import FunctionSpace, Function
from dolfinx.mesh import create_interval
from mpi4py import MPI

from pyforce.offline.pod import POD as offlinePOD
from pyforce.tools.functions_list import FunctionsList
from pyforce.offline.sensors import SGREEDY

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

    return offlinePOD_instance.PODmodes

@pytest.fixture
def sgreedy_instance(domain, pod_modes, V):
    s = 0.1  # Standard deviation of the Gaussian kernel
    return SGREEDY(domain, pod_modes, V, name="Test Snapshot", s=s)

def test_initialization(sgreedy_instance):
    assert isinstance(sgreedy_instance.basis, FunctionsList)
    assert isinstance(sgreedy_instance.V, FunctionSpace)
    assert sgreedy_instance.name == "Test Snapshot"
    assert sgreedy_instance.sens_class is not None

@pytest.mark.parametrize("_xm, is_H1", [
    ('free', False), # All positions are available with L2 representation
    ('free', True),  # All positions are available with H1 representation
    ('random', False), # Selected positions are available with L2 representation
    ('random', True),  # Selected positions are available with H1 representation
])
def test_generate(sgreedy_instance, _xm, is_H1):

    if _xm == 'free':
        xm = None
    else:
        choices = np.random.choice(sgreedy_instance.domain.geometry.x.shape[0], 50, replace=False)
        xm = [sgreedy_instance.V.tabulate_dof_coordinates()[choice] for choice in choices]
              
    N    = 5   # Dimension of the reduced space
    Mmax = 10  # Maximum number of sensors to select
    sgreedy_instance.generate(N=N, Mmax=Mmax, xm=xm, is_H1=is_H1)

    assert len(sgreedy_instance.xm_sens) == Mmax + 1
    assert len(sgreedy_instance.basis_sens) == Mmax +1
