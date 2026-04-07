import numpy as np
import pytest
import os
from pathlib import Path
import pyvista as pv

from pyforce.offline.eim import EIM as OfflineEIM
from pyforce.online.eim import EIM as OnlineEIM
from pyforce.tools.functions_list import FunctionsList

@pytest.fixture
def grid(nx = 10, ny = 10):
    """
    Generate a refined, structured 2D grid.
    Increased resolution (50x50) ensures that high-frequency 
    synthetic functions are captured without aliasing.
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    z = np.zeros(1)
    
    # pv.RectilinearGrid or pv.ImageData provides a clean, structured mesh
    grid = pv.RectilinearGrid(x, y, z)
    
    # Optional: Ensure it's treated as an UnstructuredGrid if the ROM requires it
    return grid.cast_to_unstructured_grid()

@pytest.fixture
def sample_data(grid):
    nodes = grid.points
    func = lambda x, mu: np.sin(mu * x[:, 0]) * np.cos(mu * x[:, 1])
    fun_list = FunctionsList(dofs=nodes.shape[0])
    mu_params = np.linspace(0.1, 10, 15)
    for mu in mu_params:
        fun_list.append(func(nodes, mu))
    return fun_list

@pytest.fixture
def trained_offline_eim(grid, sample_data):
    """Produces a fitted offline EIM model to provide basis and magic points."""
    model = OfflineEIM(grid=grid, gdim=2, varname="u")
    # Fit with small Mmax for speed
    model.fit(sample_data, Mmax=5)
    return model

# -----------------------
# Online EIM Tests
# -----------------------

def test_online_eim_init(grid):
    model = OnlineEIM(grid=grid, gdim=2, varname='u')
    assert model.matrix_B is None
    assert model.tikhonov is None

def test_online_eim_set_basis_and_points(grid, trained_offline_eim):
    model = OnlineEIM(grid=grid, gdim=2, varname='u')
    
    # 1. Set Magic Functions
    model.set_basis(basis=trained_offline_eim.magic_functions)
    # 2. Set Magic Points
    model.set_magic_points(magic_points=trained_offline_eim.magic_points)
    
    assert len(model.basis) == 5
    assert len(model.magic_points['idx']) == 5

def test_online_eim_loading_from_disk(tmp_path, grid, trained_offline_eim):
    # Save offline components (Manually simulating EIM.save behavior)
    out_dir = tmp_path / "eim_basis"
    os.makedirs(out_dir, exist_ok=True)
    
    trained_offline_eim.magic_functions.store(f'mf_u', filename=os.path.join(out_dir, 'mf_u'))
    np.save(os.path.join(out_dir, 'magic_points_u.npy'), trained_offline_eim.magic_points)
    
    model = OnlineEIM(grid=grid, gdim=2, varname='u')
    model.set_basis(path_folder=str(out_dir))
    model.set_magic_points(path_folder=str(out_dir))
    
    assert len(model.basis) == 5
    assert len(model.magic_points['idx']) == 5

def test_online_eim_matrix_B(grid, trained_offline_eim):
    model = OnlineEIM(grid=grid, gdim=2, varname='u')
    model.set_basis(basis=trained_offline_eim.magic_functions)
    model.set_magic_points(magic_points=trained_offline_eim.magic_points)
    
    model.compute_B_matrix()
    
    # Check shape and diagonal
    assert model.matrix_B.shape == (5, 5)
    assert np.allclose(np.diag(model.matrix_B), 1.0)
    
    # Check lower triangularity (standard EIM property)
    upper_tri = model.matrix_B[np.triu_indices(5, k=1)]
    assert np.allclose(upper_tri, 0.0, atol=1e-7)

def test_online_eim_estimate_standard(grid, sample_data, trained_offline_eim):
    model = OnlineEIM(grid=grid, gdim=2, varname='u')
    model.set_basis(basis=trained_offline_eim.magic_functions)
    model.set_magic_points(magic_points=trained_offline_eim.magic_points)
    
    # Extract measures from a test snapshot at magic points
    test_snap = sample_data[0]
    measures = model.get_measurements(test_snap) 
    
    # Estimate
    estimation = model.estimate(measures)
    
    assert isinstance(estimation, FunctionsList)
    assert estimation.fun_shape == grid.n_points
    assert len(estimation) == 1

def test_online_eim_tikhonov_regularization(grid, sample_data, trained_offline_eim):
    model = OnlineEIM(grid=grid, gdim=2, varname='u')
    model.set_basis(basis=trained_offline_eim.magic_functions)
    model.set_magic_points(magic_points=trained_offline_eim.magic_points)
    
    # 1. Prepare Tikhonov matrices
    model.set_tikhonov_matrices(train_snaps=sample_data)
    
    assert 'T' in model.tikhonov
    assert model.tikhonov['T'].shape == (5, 5)

    # 2. Estimate with regularization
    measures = model.get_measurements(sample_data[2], noise_std=0.01)
    reg_params = {'type': 'tikhonov', 'lambda': 0.1}
    
    est_reg = model.estimate(measures, regularization_params=reg_params)
    assert len(est_reg) == 1

def test_online_eim_compute_errors(grid, sample_data, trained_offline_eim):
    model = OnlineEIM(grid=grid, gdim=2, varname='u')
    model.set_basis(basis=trained_offline_eim.magic_functions)
    model.set_magic_points(magic_points=trained_offline_eim.magic_points)
    
    res = model.compute_errors(
        snaps=sample_data, 
        Mmax=3, 
        noise_std=0.0, 
        verbose=False
    )
    
    assert len(res.mean_abs_err) == 3
    assert 'Measures' in res.computational_time
    assert 'StateEstimation' in res.computational_time

def test_online_eim_noise_impact(grid, sample_data, trained_offline_eim):
    model = OnlineEIM(grid=grid, gdim=2, varname='u')
    model.set_basis(basis=trained_offline_eim.magic_functions)
    model.set_magic_sensors = None # EIM uses points, just ensuring clean state
    model.set_magic_points(magic_points=trained_offline_eim.magic_points)
    
    res_no_noise = model.compute_errors(sample_data, Mmax=5, noise_std=0.0)
    res_high_noise = model.compute_errors(sample_data, Mmax=5, noise_std=0.5)
    
    # Noise should degrade reconstruction quality
    assert res_high_noise.mean_rel_err[-1] > res_no_noise.mean_rel_err[-1]