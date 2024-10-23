import pytest
import numpy as np
from dolfinx.fem import Function, FunctionSpace
from dolfinx.mesh import create_unit_square
from mpi4py import MPI
from pyforce.tools.backends import norms, LoopProgress
from pyforce.tools.functions_list import FunctionsList
from pyforce.offline.pod import POD, DiscretePOD

@pytest.fixture
def function_space():
    domain = create_unit_square(MPI.COMM_WORLD, 10, 10)
    return FunctionSpace(domain, ("Lagrange", 1))

@pytest.fixture
def create_snapshots(function_space):
    snap = FunctionsList(function_space)
    
    psi = np.linspace(0.25, 1, 20)
    for pp in psi:
        f = Function(function_space)
        f.interpolate(lambda x: np.sin(np.pi * pp * x[0]) + np.cos(np.pi * pp * x[1]))
        snap.append(f)
    return snap

@pytest.fixture
def setup_pod(create_snapshots):
    snap = create_snapshots
    pod = POD(train_snap=snap, name="u")
    return pod

@pytest.fixture
def setup_discrete_default_pod(create_snapshots):
    snap = create_snapshots
    pod = DiscretePOD(train_snap=snap, name="u")
    return pod

def test_pod_initialization(setup_pod, create_snapshots):
    
    pod = setup_pod
    assert pod.Ns == 20
    assert pod.name == "u"
    assert pod.V == create_snapshots.fun_space
    assert len(pod.eigenvalues) == 20
    assert len(pod.eigenvectors) == 20

def test_pod_compute_basis(setup_pod, create_snapshots):
    snap = create_snapshots

    setup_pod.compute_basis(train_snap=snap, maxBasis=3)
    assert len(setup_pod.PODmodes) == 3

def test_pod_projection(setup_pod, create_snapshots):
    snap = create_snapshots
    pod = setup_pod

    pod.compute_basis(train_snap=snap, maxBasis=3)
    test_func = Function(snap.fun_space)
    test_func.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    coeffs = pod.projection(test_func, 3)
    
    assert len(coeffs) == 3

def test_pod_train_error(setup_pod, create_snapshots):
    snap = create_snapshots
    pod = setup_pod

    pod.compute_basis(train_snap=snap, maxBasis=3)
    abs_err, rel_err, coeff_matrix = pod.train_error(train_snap=snap, maxBasis=3)
    
    assert len(abs_err) == 3
    assert len(rel_err) == 3
    assert coeff_matrix.shape == (20, 3)

def test_pod_test_error(setup_pod, create_snapshots):
    snap = create_snapshots
    pod = setup_pod

    pod.compute_basis(train_snap=snap, maxBasis=3)
    abs_err, rel_err, coeff_matrix = pod.test_error(test_snap=snap, maxBasis=3)
    assert len(abs_err) == 3
    assert len(rel_err) == 3
    assert coeff_matrix.shape == (20, 3)

@pytest.mark.parametrize("Nmax, random", [
    (15, False), # With 15 basis without randomised svd
    (None, False), # With max basis without randomised svd
    (15, True), # With 15 basis with randomised svd
    (None, True), # With max basis with randomised svd
])
def test_discretepod_initialization(create_snapshots, Nmax, random):
    
    discrete_pod = DiscretePOD(create_snapshots, 'u', Nmax=Nmax, random = random)

    assert discrete_pod.Ns == len(create_snapshots), "Number of snapshots (Ns) mismatch"
    assert discrete_pod.Nh == len(create_snapshots(0)), "Number of degrees of freedom (Nh) mismatch"
    assert discrete_pod.name == "u", "Name attribute mismatch"

    if Nmax is None:
        Nmax = len(create_snapshots)
    assert len(discrete_pod.modes) == discrete_pod.Nmax, "Number of POD modes does not match Nmax"

def test_pod_projection(setup_discrete_default_pod, create_snapshots):
    """Test the projection of a snapshot onto POD modes."""
    pod = setup_discrete_default_pod
    snapshots = create_snapshots

    projected_coeff = pod.projection(snapshots, N=5)

    assert projected_coeff.shape[0] == 5, "Projection did not return correct number of coefficients"
    assert projected_coeff.ndim == 2, "Projection result should be 2D (N modes, N_snapshots)"

def test_pod_reconstruction(setup_discrete_default_pod, create_snapshots):
    """Test the reconstruction of a snapshot from POD coefficients."""
    pod = setup_discrete_default_pod
    snapshots = create_snapshots

    # Test for first snapshot with first 5 modes
    coeff = pod.Vh_train[:5, 0]
    recon_snap = pod.reconstruct(coeff)

    assert recon_snap.shape[0] == pod.Nh, "Reconstructed snapshot has incorrect dimensions"
    assert isinstance(recon_snap, np.ndarray), "Reconstructed snapshot should be a numpy array"

def test_pod_train_error(setup_discrete_default_pod, create_snapshots):
    """Test the training error computation."""
    pod = setup_discrete_default_pod
    max_basis = 5
    max_abs_err, max_rel_err = pod.train_error(create_snapshots, maxBasis=max_basis, verbose=False)

    assert len(max_abs_err) == max_basis, "Max absolute error array size mismatch"
    assert len(max_rel_err) == max_basis, "Max relative error array size mismatch"
    assert np.all(max_abs_err >= 0), "Max absolute errors should be non-negative"
    assert np.all(max_rel_err >= 0), "Max relative errors should be non-negative"