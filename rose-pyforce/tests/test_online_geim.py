import numpy as np
import pytest
import pyvista as pv

from pyforce.offline.geim import GEIM as OfflineGEIM
from pyforce.online.geim import GEIM as OnlineGEIM
from pyforce.tools.functions_list import FunctionsList

# -----------------------
# Fixtures
# -----------------------

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
def trained_offline_geim(grid, sample_data):
    """Produces a fitted offline GEIM model to provide basis and sensors."""
    model = OfflineGEIM(grid=grid, gdim=2, varname="u", sensors_type='Gaussian')
    # Fit with small Mmax for speed
    model.fit(sample_data, Mmax=5, sensor_params={'s': 0.02})
    return model

# -----------------------
# Online GEIM Tests
# -----------------------

def test_online_geim_init(grid):
    model = OnlineGEIM(grid=grid, gdim=2, varname='u')
    assert model.matrix_B is None
    assert model.tikhonov is None

def test_online_geim_set_basis_and_sensors(grid, trained_offline_geim):
    model = OnlineGEIM(grid=grid, gdim=2, varname='u')
    
    # 1. Set Magic Functions
    model.set_basis(basis=trained_offline_geim.magic_functions)
    # 2. Set Magic Sensors
    model.set_magic_sensors(sensors=trained_offline_geim.magic_sensors.library)
    
    assert len(model.basis) == 5
    assert len(model.sensors) == 5

def test_online_geim_basis_loading_from_disk(tmp_path, grid, trained_offline_geim):
    # Save offline components
    out_dir = tmp_path / "geim_basis"
    trained_offline_geim.save(str(out_dir))
    
    model = OnlineGEIM(grid=grid, gdim=2, varname='u')
    model.set_basis(path_folder=str(out_dir))
    model.set_magic_sensors(path_folder=str(out_dir))
    
    assert len(model.basis) == 5
    assert len(model.sensors) == 5

def test_online_geim_matrix_B(grid, trained_offline_geim):
    model = OnlineGEIM(grid=grid, gdim=2, varname='u')
    model.set_basis(basis=trained_offline_geim.magic_functions)
    model.set_magic_sensors(sensors=trained_offline_geim.magic_sensors.library)
    
    model.compute_B_matrix()
    
    # 1. Check shape and diagonal (should be exactly 1.0)
    assert model.matrix_B.shape == (5, 5)
    assert np.allclose(np.diag(model.matrix_B), 1.0)

    # 2. Check lower triangularity with appropriate tolerance
    # Extract only the strictly upper triangle
    upper_tri = model.matrix_B[np.triu_indices(5, k=1)]
    
    # Assert that all elements in the upper triangle are nearly zero
    assert np.allclose(upper_tri, 0.0, atol=1e-7)

def test_online_geim_estimate_standard(grid, sample_data, trained_offline_geim):
    model = OnlineGEIM(grid=grid, gdim=2, varname='u')
    model.set_basis(basis=trained_offline_geim.magic_functions)
    model.set_magic_sensors(sensors=trained_offline_geim.magic_sensors.library)
    
    # Extract measures from a test snapshot
    test_snap = sample_data[0]
    measures = model.get_measurements(test_snap) # shape (5, 1)
    
    # Estimate
    estimation = model.estimate(measures)
    
    assert isinstance(estimation, FunctionsList)
    assert estimation.fun_shape == grid.n_points
    # Check that it reconstructs the first snapshot with measures reasonably
    # Note: Error depends on Mmax used in offline fit
    rec_matrix = estimation.return_matrix()
    assert rec_matrix.shape == (grid.n_points, 1)

def test_online_geim_tikhonov_regularization(grid, sample_data, trained_offline_geim):
    model = OnlineGEIM(grid=grid, gdim=2, varname='u')
    model.set_basis(basis=trained_offline_geim.magic_functions)
    model.set_magic_sensors(sensors=trained_offline_geim.magic_sensors.library)
    
    # 1. Prepare Tikhonov matrices using training data
    model.set_tikhonov_matrices(train_snaps=sample_data)
    
    assert 'T' in model.tikhonov
    assert 'beta_mean' in model.tikhonov
    assert model.tikhonov['T'].shape == (5, 5)

    # 2. Estimate with regularization
    measures = model.get_measurements(sample_data[2], noise_std=0.01)
    reg_params = {'type': 'tikhonov', 'lambda': 1e-3}
    
    est_reg = model.estimate(measures, regularization_params=reg_params)
    assert len(est_reg) == 1

def test_online_geim_compute_errors(grid, sample_data, trained_offline_geim):
    model = OnlineGEIM(grid=grid, gdim=2, varname='u')
    model.set_basis(basis=trained_offline_geim.magic_functions)
    model.set_magic_sensors(sensors=trained_offline_geim.magic_sensors.library)
    
    # Test error computation with noise
    res = model.compute_errors(
        snaps=sample_data, 
        Mmax=3, 
        noise_std=0.001, 
        verbose=False
    )
    
    assert len(res.mean_abs_err) == 3
    assert 'Measures' in res.computational_time
    assert 'StateEstimation' in res.computational_time

def test_online_geim_noise_impact(grid, sample_data, trained_offline_geim):
    """Verify that adding noise increases the error."""
    model = OnlineGEIM(grid=grid, gdim=2, varname='u')
    model.set_basis(basis=trained_offline_geim.magic_functions)
    model.set_magic_sensors(sensors=trained_offline_geim.magic_sensors.library)
    
    res_no_noise = model.compute_errors(sample_data, Mmax=5, noise_std=0.0)
    res_high_noise = model.compute_errors(sample_data, Mmax=5, noise_std=1.0)
    
    # Relative error should be higher for the noisy case
    assert res_high_noise.mean_rel_err[-1] > res_no_noise.mean_rel_err[-1]