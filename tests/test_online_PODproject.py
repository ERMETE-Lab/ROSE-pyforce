import pytest
import numpy as np
from dolfinx.fem import FunctionSpace, Function
from dolfinx.mesh import create_interval
from mpi4py import MPI

from pyforce.offline.pod import POD as offlinePOD
from pyforce.online.pod_projection import PODproject 
from pyforce.tools.functions_list import FunctionsList

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
def setup_podproject(pod_modes):
    
    name = "u"
    podproj_instance = PODproject(pod_modes, name)

    return podproj_instance

def test_synt_test_error(setup_podproject, test_snap):
    
    maxBasis = 15
    result = setup_podproject.synt_test_error(test_snap, maxBasis)

    # Check results
    assert result.mean_abs_err.shape == (maxBasis,)
    assert result.mean_rel_err.shape == (maxBasis,)
    assert result.mean_rel_err[-1] <= 1e-4 # accetable value for this problem
    assert isinstance(result.computational_time, dict)