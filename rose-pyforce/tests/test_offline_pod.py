import numpy as np
import pytest
import os

from pyforce.offline.pod import rSVD, POD, HierarchicalSVD, IncrementalSVD
from pyforce.tools.functions_list import FunctionsList
from pyforce.tools.write_read import ReadFromOF

from sklearn.utils.extmath import randomized_svd

import warnings
def pytest_configure():
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        module="sklearn.utils.extmath"
    )

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

@pytest.fixture
def split_data(grid):
    """
    Returns a dictionary with 'initial' and 'new' FunctionsList 
    to simulate streaming/batch data.
    """
    nodes = grid.points
    dofs = nodes.shape[0]
    
    # Synthetic parameterized function
    # u(x, mu) = sin(mu * r) + ...
    def generate_batch(mu_range):
        f_list = FunctionsList(dofs=dofs)
        for mu in mu_range:
            r = np.linalg.norm(nodes[:, :2], axis=1)
            val = np.sin(mu * r) + 0.1 * nodes[:, 0] * mu
            f_list.append(val)
        return f_list

    # Create two distinct batches of data
    data_1 = generate_batch(np.linspace(0.1, 2.0, 20))
    data_2 = generate_batch(np.linspace(2.1, 4.0, 20))

    return {"batch1": data_1, "batch2": data_2}

# ==========================================
# Tests for Randomized SVD (rSVD)
# ==========================================

def test_rsvd_init(grid):
    model = rSVD(grid=grid, gdim=3, varname="u")
    assert hasattr(model, "svd_modes") is False  # not yet fitted
    assert hasattr(model, "singular_values") is False

def test_rsvd_fit(grid, sample_data):
    model = rSVD(grid=grid, gdim=3, varname="u")

    rank = 3
    model.fit(sample_data, rank=rank, random_state=32)

    # Check shapes
    assert len(model.singular_values) == rank
    assert len(model.svd_modes) == rank

    # Ensure modes have right spatial dimension
    assert model.svd_modes.fun_shape == sample_data.fun_shape

def test_rsvd_reduce(grid, sample_data):
    model = rSVD(grid=grid, gdim=3)
    model.fit(sample_data, rank=6, random_state=32)

    N = 5
    coeffs = model.reduce(sample_data, N=N)

    assert coeffs.shape == (N, len(sample_data))
    assert np.isfinite(coeffs).all()

def test_rsvd_reconstruct(grid, sample_data):
    model = rSVD(grid=grid, gdim=3)
    model.fit(sample_data, rank=6, random_state=32)

    N = 5
    coeffs = model.reduce(sample_data, N=N)
    snaps_rec = model.reconstruct(coeffs)

    assert isinstance(snaps_rec, FunctionsList)
    assert len(snaps_rec) == len(sample_data)

    # Basic sanity: reconstructed has same spatial shape
    assert snaps_rec.fun_shape == sample_data.fun_shape

def test_rsvd_compute_errors(grid, sample_data):
    model = rSVD(grid=grid, gdim=3)
    model.fit(sample_data, rank=6, random_state=32)

    # Use fewer modes for speed
    result = model.compute_errors(sample_data, Nmax=5, verbose=False)

    assert hasattr(result, "mean_abs_err")
    assert hasattr(result, "mean_rel_err")
    assert hasattr(result, "computational_time")

    assert len(result.mean_abs_err) == 5
    assert len(result.mean_rel_err) == 5

def test_rsvd_save(tmp_path, grid, sample_data):
    model = rSVD(grid=grid, gdim=3, varname="u")
    model.fit(sample_data, rank=6, random_state=32)

    out_dir = tmp_path / "model"
    model.save(str(out_dir))

    # Check files exist
    svd_modes_file = out_dir / "SVDmode_u.npy"  # FunctionsList.store creates several formats depending on kwargs
    sing_vals_file = out_dir / "sing_vals_u.npy"

    assert sing_vals_file.exists()

    # For FunctionsList.store, allow any extension:
    mode_files = list(out_dir.glob("SVDmode_u*"))
    assert len(mode_files) >= 1

# ==================================================
# Tests for Proper Orthogonal Decomposition (POD)
# ==================================================

def test_pod_init(grid):
    """Test initialization state"""
    model = POD(grid=grid, gdim=3, varname="u")
    assert hasattr(model, "eigenvalues") is False
    assert hasattr(model, "pod_modes") is False

def test_pod_fit_and_basis(grid, sample_data):
    """Test the split workflow: fit() -> compute_basis()"""
    model = POD(grid=grid, gdim=3, varname="u")
    
    # 1. Fit (Eigenvalue problem)
    model.fit(sample_data, verbose=True)
    assert hasattr(model, "eigenvalues")
    assert hasattr(model, "eigenvectors")
    # Check that eigenvalues are sorted descending
    assert np.all(np.diff(model.eigenvalues) <= 0)

    # 2. Compute Basis
    rank = 5
    model.compute_basis(sample_data, rank=rank, normalise=False)
    
    assert hasattr(model, "pod_modes")
    assert len(model.pod_modes) == rank
    assert model.pod_modes.fun_shape == sample_data.fun_shape

def test_pod_gram_schmidt(grid, sample_data):
    """Test if Gram-Schmidt normalization enforces orthogonality"""
    model = POD(grid=grid, gdim=3)
    model.fit(sample_data)
    
    rank = 3
    # Enable normalization
    model.compute_basis(sample_data, rank=rank, normalise=True)
    
    # Check Orthogonality: (psi_0, psi_1)_L2 should be ~ 0
    psi_0 = model.pod_modes(0)
    psi_1 = model.pod_modes(1)
    
    inner_prod = model.calculator.L2_inner_product(psi_0, psi_1)
    norm_0 = model.calculator.L2_norm(psi_0)
    
    # Assertions with tolerance
    assert np.isclose(inner_prod, 0.0, atol=1e-6)
    assert np.isclose(norm_0, 1.0, atol=1e-6)

def test_pod_reduce_reconstruct(grid, sample_data):
    """Test projection and reconstruction"""
    model = POD(grid=grid, gdim=3)
    model.fit(sample_data)
    model.compute_basis(sample_data, rank=5)

    N = 4
    coeffs = model.reduce(sample_data, N=N)

    assert coeffs.shape == (N, len(sample_data))
    
    rec_snaps = model.reconstruct(coeffs)
    assert isinstance(rec_snaps, FunctionsList)
    assert rec_snaps.fun_shape == sample_data.fun_shape
    assert len(rec_snaps) == len(sample_data)

def test_pod_compute_errors(grid, sample_data):
    """Test error computation structure"""
    model = POD(grid=grid, gdim=3)
    model.fit(sample_data)
    model.compute_basis(sample_data, rank=5)

    result = model.compute_errors(sample_data, Nmax=4, verbose=False)

    assert hasattr(result, "mean_abs_err")
    assert hasattr(result, "mean_rel_err")
    assert len(result.mean_abs_err) == 4
    
    # Error should generally decrease or stay flat as modes increase
    assert result.mean_rel_err[-1] < result.mean_rel_err[0]

def test_pod_save(tmp_path, grid, sample_data):
    """Test saving functionality"""
    model = POD(grid=grid, gdim=3, varname="velocity")
    model.fit(sample_data)
    model.compute_basis(sample_data, rank=4)

    out_dir = tmp_path / "pod_model"
    model.save(str(out_dir))

    # Check for eigenvalues file
    eig_file = out_dir / "eigenvalues_velocity.npy"
    assert eig_file.exists()

    # Check for modes file (FunctionsList.store behavior)
    # Usually creates .npy or .npz depending on implementation
    mode_files = list(out_dir.glob("PODmode_velocity*"))
    assert len(mode_files) >= 1

# ==========================================
# Tests for HierarchicalSVD (hSVD)
# ==========================================
def test_hsvd_init(grid):
    model = HierarchicalSVD(grid=grid, gdim=3)
    assert model.svd_modes is None
    assert model.singular_values is None

def test_hsvd_first_update(grid, split_data):
    """Test that the first update acts like a standard fit"""
    model = HierarchicalSVD(grid=grid, gdim=3)
    
    rank = 5
    # First update: acts as initialization
    model.update(train_snaps=split_data["batch1"], rank=rank, random_state=42)

    assert model.svd_modes is not None
    assert len(model.svd_modes) == rank
    assert len(model.singular_values) == rank
    assert model.svd_modes.fun_shape == split_data["batch1"].fun_shape

def test_hsvd_hierarchical_update(grid, split_data):
    """Test updating an existing model with new snapshots"""
    model = HierarchicalSVD(grid=grid, gdim=3)
    rank = 5
    
    # 1. Initial Fit
    model.update(train_snaps=split_data["batch1"], rank=rank, random_state=42)
    s_vals_initial = model.singular_values.copy()
    
    # 2. Hierarchical Update
    # This should merge the subspaces of batch1 and batch2
    model.update(train_snaps=split_data["batch2"], rank=rank, random_state=42)

    assert len(model.svd_modes) == rank
    
    # The singular values should change after seeing new data
    assert not np.allclose(model.singular_values, s_vals_initial)
    
    # Verify reconstruction works on new data
    coeffs = model.reduce(split_data["batch2"], N=rank)
    rec = model.reconstruct(coeffs)
    
    # Check error is reasonable (sanity check)
    err = model.compute_errors(split_data["batch2"], Nmax=rank)
    assert np.mean(err.mean_rel_err) < 1.0

def test_hsvd_explicit_update(grid, split_data):
    """Test updating using explicit modes/values instead of snapshots"""
    model = HierarchicalSVD(grid=grid, gdim=3)
    rank = 4
    
    # 1. Init with batch 1
    model.update(train_snaps=split_data["batch1"], rank=rank)
    
    # 2. Manually compute SVD for batch 2
    # We simulate another process sending us modes instead of raw data
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        U2, S2, _ = randomized_svd(split_data["batch2"].return_matrix(), n_components=rank)

    modes2 = FunctionsList(split_data["batch2"].fun_shape)
    modes2.build_from_matrix(U2)
    
    # 3. Update using modes/vals
    model.update(new_modes=modes2, new_sing_vals=S2, rank=rank)
    
    assert len(model.svd_modes) == rank
    assert model.singular_values is not None

# ==========================================
# Tests for IncrementalSVD (iSVD)
# ==========================================

def test_isvd_fit(grid, split_data):
    """Test the overridden fit method which initializes Vh"""
    model = IncrementalSVD(grid=grid, gdim=3)
    rank = 5
    
    model.fit(split_data["batch1"], rank=rank, random_state=42)
    
    assert hasattr(model, "Vh")
    assert model.Vh.shape == (rank, len(split_data["batch1"]))
    assert model.Ns == len(split_data["batch1"])

def test_isvd_update_single(grid, split_data):
    """Test updating with a single snapshot (streaming)"""
    model = IncrementalSVD(grid=grid, gdim=3)
    rank = 5
    
    # Fit on batch 1
    model.fit(split_data["batch1"], rank=rank, random_state=42)
    initial_Ns = model.Ns
    
    # Update with ONE snapshot from batch 2
    # Extract one column/snapshot
    new_snap = split_data["batch2"].return_matrix()[:, 0] 
    
    model.update(new_snap)
    
    assert model.Ns == initial_Ns + 1
    # Vh grows horizontally
    assert model.Vh.shape[1] == model.Ns
    # Modes rank stays constant (Brand's algorithm keeps rank fixed usually)
    assert len(model.svd_modes) == rank 

def test_isvd_update_batch(grid, split_data):
    """Test updating with a functions list (mini-batch)"""
    model = IncrementalSVD(grid=grid, gdim=3)
    rank = 5
    
    model.fit(split_data["batch1"], rank=rank, random_state=42)
    initial_Ns = model.Ns
    
    # Update with WHOLE batch 2
    model.update(split_data["batch2"])
    
    expected_Ns = initial_Ns + len(split_data["batch2"])
    assert model.Ns == expected_Ns
    assert model.Vh.shape[1] == expected_Ns
    
    # Check internal consistency
    model._check_svd()

def test_isvd_reconstruction_quality(grid, split_data):
    """
    Ensure the incremental update actually 'learns' the new data.
    Relative error on batch 2 should be low after updating with batch 2.
    """
    model = IncrementalSVD(grid=grid, gdim=3)
    rank = 3
    
    # 1. Fit on Batch 1
    model.fit(split_data["batch1"], rank=rank)
    
    # 2. Calculate error on Batch 2 BEFORE update (should be high-ish)
    res_before = model.compute_errors(split_data["batch2"], Nmax=rank)
    err_before = np.mean(res_before.mean_rel_err)
    
    # 3. Update with Batch 2
    model.update(split_data["batch2"])
    
    # 4. Calculate error on Batch 2 AFTER update (should improve)
    res_after = model.compute_errors(split_data["batch2"], Nmax=rank)
    err_after = np.mean(res_after.mean_rel_err)
    
    # The representation of batch2 should be better after the model has seen it
    assert err_after < err_before