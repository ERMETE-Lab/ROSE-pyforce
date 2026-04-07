import numpy as np
import pytest
import os
import matplotlib.pyplot as plt
import pyvista as pv

from pyforce.offline.sgreedy import SGREEDY, compute_inf_sup
from pyforce.tools.functions_list import FunctionsList
from pyforce.offline.sensors import GaussianSensorLibrary

# -----------------------
# Fixtures
# -----------------------

@pytest.fixture
def grid():
    """Create a simple 2D plane grid (25 points) for fast testing."""
    return pv.Plane(i_resolution=5, j_resolution=5)

@pytest.fixture
def basis_functions(grid):
    """Create a FunctionsList with dummy reduced basis functions (scalar fields)."""
    fl = FunctionsList(dofs=grid.n_points)
    nodes = grid.points
    
    # 3 dummy basis functions
    fl.append(np.ones(grid.n_points))
    fl.append(np.sin(nodes[:, 0]))
    fl.append(np.cos(nodes[:, 1]))
    return fl

@pytest.fixture
def sensor_params():
    """Parameters for Gaussian sensor creation."""
    return {'s': 0.5}

# -----------------------
# Tests for SGREEDY Class
# -----------------------

def test_sgreedy_initialization(grid):
    """Test valid and invalid initialization of SGREEDY."""
    # Valid initialization
    model = SGREEDY(grid=grid, gdim=2, varname='pressure', sensors_type='Exponential')
    assert model.sensors_type == 'Exponential'
    assert model.varname == 'pressure'

    # Invalid sensor type should raise AssertionError
    with pytest.raises(AssertionError, match="sensors_type must be one of"):
        SGREEDY(grid=grid, gdim=2, sensors_type='Polynomial')

def test_sgreedy_fit_exceptions(grid, basis_functions, sensor_params):
    """Test that fit properly catches invalid parameters and shapes."""
    model = SGREEDY(grid=grid, gdim=2)
    
    # Mmax < Nmax
    with pytest.raises(AssertionError, match="Mmax must be greater or equal to Nmax"):
        model.fit(basis_functions, Mmax=1, Nmax=2, sensor_params=sensor_params)

    # tol <= 0
    with pytest.raises(AssertionError, match="tol must be greater than 0"):
        model.fit(basis_functions, Mmax=4, tol=-0.1, sensor_params=sensor_params)

    # Vector fields 
    vector_basis = FunctionsList(dofs=grid.n_points * 2)
    vector_basis.append(np.ones(grid.n_points * 2))
    with pytest.raises(ValueError, match="vector-valued"):
        model.fit(vector_basis, Mmax=3, sensor_params=sensor_params)

def test_sgreedy_fit_stability_loop(grid, basis_functions, sensor_params):
    """
    Test the fit method ensuring the stability loop runs. 
    A high tolerance ensures the loop doesn't break early.
    """
    model = SGREEDY(grid=grid, gdim=2, sensors_type='Gaussian')
    Mmax = 3
    Nmax = 2
    
    # High tol guarantees we stay in the stability loop to place all Mmax sensors
    model.fit(basis_functions, Mmax=Mmax, Nmax=Nmax, sensor_params=sensor_params, tol=10.0, verbose=False)
    
    assert len(model.sensors) == Mmax
    assert len(model.sensor_centers) == Mmax
    # Verify the centers are actually coordinates (arrays of length 3)
    assert len(model.sensor_centers[0]) == 3

def test_sgreedy_fit_approximation_loop(grid, basis_functions, sensor_params):
    """
    Test the fit method hitting the approximation loop.
    A very low tolerance forces an early exit from the stability loop.
    """
    model = SGREEDY(grid=grid, gdim=2, sensors_type='Gaussian')
    Mmax = 5
    Nmax = 2
    
    # Low tol ensures stability loop breaks early, pushing the rest into approximation loop
    model.fit(basis_functions, Mmax=Mmax, Nmax=Nmax, sensor_params=sensor_params, tol=0.0001, verbose=False)
    
    # The total number of sensors placed should still equal Mmax
    assert len(model.sensors) == Mmax
    assert len(model.sensor_centers) == Mmax

def test_sgreedy_save(tmp_path, grid, basis_functions, sensor_params):
    """Test saving the selected sensors to disk."""
    model = SGREEDY(grid=grid, gdim=2, varname='temp', sensors_type='Gaussian')
    model.fit(basis_functions, Mmax=5, sensor_params=sensor_params, verbose=False)

    out_dir = tmp_path / "sgreedy_output"
    model.save(str(out_dir), format='npz')

    saved_file = out_dir / "sgreedy_sens_temp.npz"
    assert saved_file.exists()

# -----------------------
# Tests for compute_inf_sup
# -----------------------

def test_compute_inf_sup(grid, basis_functions, sensor_params):
    """Test the standalone compute_inf_sup function."""
    from pyforce.tools.backends import IntegralCalculator
    
    # Manually setup a small sensor library and calculator
    calc = IntegralCalculator(grid, gdim=2)
    sens_lib = GaussianSensorLibrary(grid, gdim=2, use_centroids=False)
    sens_lib.create_library(**sensor_params)
    
    # We need a subset of sensors to act as the "chosen" ones
    chosen_sensors = GaussianSensorLibrary(grid, gdim=2, use_centroids=False)
    chosen_sensors.add_sensor(sens_lib.library[0])
    chosen_sensors.add_sensor(sens_lib.library[-1])
    
    # Test 1: Return only the inf-sup constant
    beta = compute_inf_sup(
        sensors=chosen_sensors, 
        basis_functions=basis_functions, 
        calculator=calc, 
        N=2, 
        return_eigenvector=False
    )
    
    assert isinstance(beta, float)
    assert beta >= 0.0  # Inf-sup constant is defined as a non-negative sqrt
    
    # Test 2: Return constant and eigenvector
    beta_vec, eigvec = compute_inf_sup(
        sensors=chosen_sensors, 
        basis_functions=basis_functions, 
        calculator=calc, 
        N=2, 
        return_eigenvector=True
    )
    
    assert np.isclose(beta, beta_vec)
    assert isinstance(eigvec, np.ndarray)
    assert len(eigvec) == 2 # Because N=2