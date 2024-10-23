import pytest
import numpy as np
from dolfinx.fem import FunctionSpace, Function
from dolfinx.mesh import create_interval
from mpi4py import MPI

from scipy.interpolate import interp1d

from pyforce.offline.pod import POD as offlinePOD
from pyforce.online.pod_interpolation import PODI
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
def pod_modes_maps(train_snap, mu_train):
    offlinePOD_instance = offlinePOD(train_snap, 'u')

    maxBasis = 20
    offlinePOD_instance.compute_basis(train_snap, maxBasis = maxBasis)
    train_coeffs = offlinePOD_instance.train_error(train_snap, maxBasis=maxBasis)[2]


    assert len(mu_train) == len(train_coeffs[:, 0])

    maps = list()
    for nn in range(maxBasis):
        coeff_map = interp1d(mu_train.flatten(), train_coeffs[:, nn], kind='linear', fill_value='extrapolate')
        maps.append(coeff_map)

    return offlinePOD_instance.PODmodes, maps

@pytest.fixture
def setup_podproject(pod_modes_maps):
    
    name = "u"
    podi_instance = PODI(*pod_modes_maps, name)

    return podi_instance

@pytest.mark.parametrize("mu_is_none", [
    (False), # Use maps inside  the class
    (True),  # Use maps outside the class
])
def test_synt_test_error(setup_podproject, test_snap, mu_test, mu_is_none):
    
    maxBasis = 15

    if mu_is_none:
        mu_estim = None
        alpha_coeffs = np.zeros((len(test_snap), maxBasis))
        for nn in range(maxBasis):
            alpha_coeffs[:, nn] = setup_podproject.maps[nn](mu_test.flatten())
    else:
        mu_estim = mu_test
        alpha_coeffs = None

    result = setup_podproject.synt_test_error(test_snap, mu_estim, maxBasis, 
                                              alpha_coeffs=alpha_coeffs)

    # Check results
    assert result.mean_abs_err.shape == (maxBasis,)
    assert result.mean_rel_err.shape == (maxBasis,)
    assert result.mean_rel_err[-1] <= 1e-3 # accetable value for this problem
    assert isinstance(result.computational_time, dict)