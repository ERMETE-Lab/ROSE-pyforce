import pytest
import numpy as np
from dolfinx.fem import Function, FunctionSpace
from dolfinx.mesh import create_unit_square
from mpi4py import MPI
from pyforce.tools.backends import norms, LoopProgress
from pyforce.tools.functions_list import FunctionsList
from pyforce.offline.pod import POD  # Assuming the POD class is in a file named pod_module.py

@pytest.fixture
def create_function_space():
    domain = create_unit_square(MPI.COMM_WORLD, 10, 10)
    return FunctionSpace(domain, ("Lagrange", 1))

@pytest.fixture
def create_snapshots(create_function_space):
    V = create_function_space
    snap = FunctionsList(V)
    
    psi = np.linspace(0.25, 1, 20)
    for pp in psi:
        f = Function(V)
        f.interpolate(lambda x: np.sin(np.pi * pp * x[0]) + np.cos(np.pi * pp * x[1]))
        snap.append(f)
    return snap

def test_pod_initialization(create_snapshots):
    snap = create_snapshots
    pod = POD(train_snap=snap, name="test")
    assert pod.Ns == 20
    assert pod.name == "test"
    assert pod.V == snap.fun_space
    assert len(pod.eigenvalues) == 20
    assert len(pod.eigenvectors) == 20

def test_pod_compute_basis(create_snapshots):
    snap = create_snapshots
    pod = POD(train_snap=snap, name="test")
    pod.compute_basis(train_snap=snap, maxBasis=3)
    assert len(pod.PODmodes) == 3

def test_pod_projection(create_snapshots):
    snap = create_snapshots
    pod = POD(train_snap=snap, name="test")
    pod.compute_basis(train_snap=snap, maxBasis=3)
    test_func = Function(snap.fun_space)
    test_func.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    coeffs = pod.projection(test_func, 3)
    assert len(coeffs) == 3

def test_pod_train_error(create_snapshots):
    snap = create_snapshots
    pod = POD(train_snap=snap, name="test")
    pod.compute_basis(train_snap=snap, maxBasis=3)
    abs_err, rel_err, coeff_matrix = pod.train_error(train_snap=snap, maxBasis=3)
    assert len(abs_err) == 3
    assert len(rel_err) == 3
    assert coeff_matrix.shape == (20, 3)

def test_pod_test_error(create_snapshots):
    snap = create_snapshots
    pod = POD(train_snap=snap, name="test")
    pod.compute_basis(train_snap=snap, maxBasis=3)
    abs_err, rel_err, coeff_matrix = pod.test_error(test_snap=snap, maxBasis=3)
    assert len(abs_err) == 3
    assert len(rel_err) == 3
    assert coeff_matrix.shape == (20, 3)
