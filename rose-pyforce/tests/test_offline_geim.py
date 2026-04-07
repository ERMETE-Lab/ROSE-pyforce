import numpy as np
import pytest
import os
import matplotlib.pyplot as plt
import pyvista as pv

from pyforce.tools.write_read import ReadFromOF
from pyforce.offline.geim import GEIM
from pyforce.tools.functions_list import FunctionsList

from pathlib import Path
from pyvista import examples

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
def sensor_params():
    """
    Generic parameters for sensor libraries. 
    Adjust these keys if your actual GaussianSensorLibrary requires specific arguments.
    """
    return {'s': 0.01}

# -----------------------
# Tests for GEIM Class
# -----------------------

def test_geim_initialization(grid):
    """Test that GEIM initializes correctly and validates sensor types."""
    # Valid initialization
    model = GEIM(grid=grid, gdim=2, varname='temp', sensors_type='Gaussian')
    assert model.sensors_type == 'Gaussian'
    assert model.varname == 'temp'

    # Invalid sensor type should raise AssertionError
    with pytest.raises(AssertionError, match="sensors_type must be one of"):
        GEIM(grid=grid, gdim=2, sensors_type='Polynomial')

def test_geim_invalid_field_shape(grid, sensor_params):
    """Test that GEIM raises ValueError when provided with vector fields."""
    model = GEIM(grid=grid, gdim=2)
    
    # Create a vector field (shape = n_points * gdim)
    vector_data = FunctionsList(dofs=grid.n_points * 2)
    vector_data.append(np.ones(grid.n_points * 2))  # Just a dummy vector field
    
    with pytest.raises(ValueError, match="vector fields"):
        model.fit(vector_data, Mmax=2, sensor_params=sensor_params)

def test_geim_fit_properties(grid, sample_data, sensor_params):
    """Test the GEIM greedy fitting procedure and attribute population."""
    model = GEIM(grid=grid, gdim=2)
    Mmax = 5
    
    maxAbsErr, maxRelErr, beta_coeff = model.fit(
        train_snaps=sample_data, 
        Mmax=Mmax, 
        sensor_params=sensor_params, 
        verbose=False
    )

    # 1. Check Output Shapes
    assert len(maxAbsErr) == Mmax
    assert len(maxRelErr) == Mmax
    assert beta_coeff.shape == (Mmax, len(sample_data))

    # 2. Check Magic Functions and Sensors Storage
    assert len(model.magic_functions) == Mmax
    assert len(model.magic_sensors.library) == Mmax
    
    # 3. Matrix B shape
    assert model.matrix_B.shape == (Mmax, Mmax)

def test_geim_reconstruction(grid, sample_data, sensor_params):
    """Test the reduction and reconstruction workflow."""
    model = GEIM(grid=grid, gdim=2)
    Mmax = 6
    model.fit(sample_data, Mmax=Mmax, sensor_params=sensor_params)

    # 1. Extract measures
    measures = model._get_measures(sample_data, M=4)
    assert measures.shape == (4, len(sample_data))

    # 2. Reconstruct from measures
    reconstructed = model.reconstruct(measures)
    assert reconstructed.fun_shape == grid.n_points
    assert len(reconstructed) == len(sample_data)

def test_geim_reduce(grid, sample_data, sensor_params):
    """Test the projection of snapshots onto the reduced space."""
    model = GEIM(grid=grid, gdim=2)
    model.fit(sample_data, Mmax=5, sensor_params=sensor_params)

    # Reduce a single numpy array
    single_snap = sample_data[0]
    beta = model.reduce(single_snap, M=3)
    
    assert beta.shape == (3, 1)

def test_geim_compute_errors(grid, sample_data, sensor_params):
    """Test the GEIM internal error computation over various truncation sizes."""
    model = GEIM(grid=grid, gdim=2)
    model.fit(sample_data, Mmax=5, sensor_params=sensor_params)

    res = model.compute_errors(sample_data, Mmax=4, verbose=False)

    assert hasattr(res, 'mean_abs_err')
    assert hasattr(res, 'computational_time')
    assert res.mean_abs_err.shape == (4,)
    
    # Errors should decrease as M increases
    assert res.mean_abs_err[-1] < res.mean_abs_err[0]

def test_geim_lebesgue_constant(grid, sample_data, sensor_params):
    """Test the Lebesgue constant computation using Gram-Schmidt orthogonalization."""
    model = GEIM(grid=grid, gdim=2)
    
    # Should raise error if called before fitting
    with pytest.raises(ValueError, match="model has not been fitted"):
        model.compute_lebesgue_constant()

    model.fit(sample_data, Mmax=4, sensor_params=sensor_params)
    lambdas = model.compute_lebesgue_constant(verbose=False)

    assert isinstance(lambdas, np.ndarray)
    assert len(lambdas) == 4
    # The Lebesgue constant is always >= 1 mathematically
    assert np.all(lambdas >= 1.0)

def test_geim_save(tmp_path, grid, sample_data, sensor_params):
    """Test saving the magic functions and magic sensors to disk."""
    model = GEIM(grid=grid, gdim=2, varname='velocity')
    model.fit(sample_data, Mmax=3, sensor_params=sensor_params)

    out_dir = tmp_path / "geim_output"
    model.save(str(out_dir), format='npz')

    # Both magic functions and magic sensors should be saved
    mf_file = out_dir / "mf_velocity.npz"
    ms_file = out_dir / "ms_velocity.npz"
    
    assert mf_file.exists()
    assert ms_file.exists()