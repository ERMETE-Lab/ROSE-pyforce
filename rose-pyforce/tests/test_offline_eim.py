import numpy as np
import pytest
import os

from pyforce.offline.eim import EIM, deim
from pyforce.offline.pod import rSVD
from pyforce.tools.functions_list import FunctionsList
from pyforce.tools.write_read import ReadFromOF

from pathlib import Path
from pyvista import examples

@pytest.fixture
def grid():
    """
    Prepare a reader using a downloaded OpenFOAM example.
    The pyvista cavity case is a safe and widely used test.
    """
    # PyVista returns ".../cavity/cavity.foam"
    foam_file = Path(examples.download_cavity(load=False))
    case_dir = foam_file.parent   # folder containing system/, constant/, 0/, etc.
    return ReadFromOF(str(case_dir)).mesh()

@pytest.fixture
def sample_data(grid):
    """Create a simple FunctionsList from cavity grid """

    _grid = grid
    nodes = _grid.points
    nodes = _grid.points

    func = lambda x, mu: np.sin(2*mu* x[:, 0]) * np.cos(4*mu*x[:, 1])
    fun_list = FunctionsList(dofs = nodes.shape[0])
    mu_params = np.linspace(0.1, 40, 50)
    for mu in mu_params:
        fun_list.append(func(nodes, mu))

    return fun_list

# ==========================================
# Tests for EIM Class
# ==========================================

def test_eim_fit_properties(grid, sample_data):
    """Test the greedy fitting procedure and attribute population."""
    model = EIM(grid=grid, gdim=3)
    
    Mmax = 20
    maxAbsErr, beta_coeff = model.fit(sample_data, Mmax=Mmax, verbose=False)

    # 1. Check Output Shapes
    assert len(maxAbsErr) == Mmax
    assert beta_coeff.shape == (Mmax, len(sample_data))

    # 2. Check Magic Point/Function Storage
    assert len(model.magic_functions) == Mmax
    assert len(model.magic_points['idx']) == Mmax
    assert len(model.magic_points['points']) == Mmax
    
    # 3. Check Interpolation Matrix (B) structure
    # EIM Matrix B should be Lower Triangular with unity diagonal if normalized correctly
    # (Though your implementation stores actual values, checking shape is crucial)
    assert model.matrix_B.shape == (Mmax, Mmax)
    
    # 4. Check monotonicity of error (Greedy algo should generally reduce error)
    # Note: On noisy random data this isn't strictly guaranteed at every single step, 
    # but for smooth parametric data it usually is.
    assert maxAbsErr[-1] < maxAbsErr[0]

def test_eim_reconstruction_accuracy(grid, sample_data):
    """Test that the EIM interpolant accurately reconstructs the training data."""
    model = EIM(grid=grid, gdim=3)
    Mmax = 10
    model.fit(sample_data, Mmax=Mmax)

    # 1. Compute Measures for all snapshots
    measures = model._get_measures(sample_data)
    
    # 2. Reconstruct from Measurements
    rec_snap = model.reconstruct(measures)

    # 3. Compare (Error should be small for M=15 on this dataset)
    # Relative L2 error
    err = np.linalg.norm(sample_data.return_matrix() - rec_snap.return_matrix(), axis=0) / np.linalg.norm(sample_data.return_matrix(), axis=0)
    mean_err = np.mean(err)
    assert mean_err < 0.1  # Threshold depends on data complexity

def test_eim_lebesgue(grid, sample_data):
    """Test the computation of the Lebesgue constant (stability metric)."""
    model = EIM(grid=grid, gdim=3)
    model.fit(sample_data, Mmax=10)

    lambdas = model.compute_lebesgue_constant()
    
    assert isinstance(lambdas, list)
    assert len(lambdas) == 10
    # Lebesgue constant is always >= 1
    assert all(l >= 1.0 for l in lambdas)

def test_eim_compute_errors(grid, sample_data):
    """Test the internal error computation loop."""
    model = EIM(grid=grid, gdim=3)
    model.fit(sample_data, Mmax=8)

    # Compute errors on the training set
    res = model.compute_errors(sample_data, Mmax=5)

    assert hasattr(res, 'mean_abs_err')
    assert hasattr(res, 'mean_rel_err')
    assert len(res.mean_abs_err) == 5
    # Error should decrease as we add more magic points
    assert res.mean_rel_err[-1] < res.mean_rel_err[0]

def test_eim_save(tmp_path, grid, sample_data):
    """Test saving magic functions and points."""
    model = EIM(grid=grid, gdim=3, varname="pressure")
    model.fit(sample_data, Mmax=5)

    out_dir = tmp_path / "eim_model"
    model.save(str(out_dir))

    # Check Magic Points file
    mp_file = out_dir / "magic_points_pressure.npy"
    assert mp_file.exists()

    # Check Magic Functions (FunctionsList storage)
    mf_files = list(out_dir.glob("mf_pressure*"))
    assert len(mf_files) > 0

# ==========================================
# Tests for DEIM Function
# ==========================================

@pytest.fixture
def mock_basis(grid, sample_data):
    """
    Helper fixture to create orthonormal basis functions (POD modes)
    required for testing the DEIM function.
    """
    _svd = rSVD(grid=grid, gdim=3)
    _svd.fit(sample_data, rank=10)
    return _svd.svd_modes

def test_deim_function(mock_basis):
    """
    Test the standalone DEIM function. 
    Note: DEIM expects orthonormal basis functions (like POD modes), not raw snapshots.
    """
    Mmax = 10
    magic_indices, P_matrix = deim(
        basis_functions=mock_basis, 
        Mmax=Mmax, 
        varname='u'
    )

    # 1. Check Shapes
    assert len(magic_indices) == Mmax
    assert P_matrix.shape == (Mmax, mock_basis.fun_shape)

    # 2. Check P matrix properties
    # P is a selection matrix: rows should be one-hot vectors
    assert np.all(np.sum(P_matrix, axis=1) == 1)
    
    # Check that the 1s are at the magic indices locations
    for i, idx in enumerate(magic_indices):
        assert P_matrix[i, idx] == 1.0

def test_deim_save(tmp_path, mock_basis):
    """Test DEIM saving functionality."""
    Mmax = 10
    out_dir = tmp_path / "deim_out"
    
    deim(
        basis_functions=mock_basis, 
        Mmax=Mmax, 
        varname='velocity',
        path_folder=str(out_dir)
    )
    
    file_path = out_dir / "magic_points_deim_velocity.npy"
    assert file_path.exists()
    
    # Verify content
    loaded_pts = np.load(file_path)
    assert len(loaded_pts) == Mmax