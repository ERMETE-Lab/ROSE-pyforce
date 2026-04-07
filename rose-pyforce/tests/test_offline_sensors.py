import numpy as np
import pytest
import pyvista as pv

from pyforce.offline.sensors import GaussianSensorLibrary, ExponentialSensorLibrary, IndicatorFunctionSensorLibrary
from pyforce.tools.functions_list import FunctionsList

# -----------------------
# Fixtures
# -----------------------

@pytest.fixture
def grid():
    """Create a simple 2D plane grid (9 points, 4 cells) for fast testing."""
    return pv.Plane(i_resolution=2, j_resolution=2)

@pytest.fixture
def dofs(grid):
    """Use points as DOFs for these tests."""
    return grid.n_points

@pytest.fixture
def sample_funcs(dofs):
    """Create a FunctionsList with two mock functions."""
    fl = FunctionsList(dofs=dofs)
    fl.append(np.ones(dofs))
    fl.append(np.arange(dofs, dtype=float))
    return fl

@pytest.fixture
def sample_ndarray(dofs):
    """Create a mock 2D numpy array representing 3 snapshots."""
    return np.random.rand(dofs, 3)

# -----------------------
# Tests for SensorLibraryBase Methods
# (Tested using GaussianSensorLibrary as the concrete implementation)
# -----------------------

def test_sensor_library_uninitialized(grid, sample_funcs):
    """Test behavior when the library is not yet created."""
    lib = GaussianSensorLibrary(grid, gdim=2)
    
    assert len(lib) == 0
    
    with pytest.raises(ValueError, match="Sensor library is not created"):
        lib._action_single(sample_funcs[0], 0)
        
    with pytest.raises(ValueError, match="Sensor library is not created"):
        lib.action(sample_funcs)

def test_add_and_set_library(grid, dofs):
    """Test manual population of the sensor library."""
    lib = GaussianSensorLibrary(grid, use_centroids=False, gdim=2)
    
    # Test add_sensor
    lib.add_sensor(np.ones(dofs))
    assert len(lib) == 1
    assert lib.library.fun_shape == dofs

    # Test set_library
    fl = FunctionsList(dofs=dofs)
    fl.append(np.zeros(dofs))
    fl.append(np.ones(dofs))
    
    lib.set_library(fl)
    assert len(lib) == 2
    assert np.allclose(lib.library[0], np.zeros(dofs))

def test_action_and_call(grid, dofs, sample_funcs, sample_ndarray):
    """Test the action/evaluation of sensors on functions."""
    lib = GaussianSensorLibrary(grid, use_centroids=False, gdim=2)
    
    # Manually add 2 sensors (kernels)
    lib.add_sensor(np.ones(dofs) * 0.5)
    lib.add_sensor(np.ones(dofs) * 2.0)
    
    # 1. Action on FunctionsList -> shape (Nsensors, Ns)
    meas_fl = lib.action(sample_funcs)
    assert meas_fl.shape == (2, len(sample_funcs))
    
    # 2. Action on 2D ndarray -> shape (Nsensors, Ns)
    meas_nd = lib.action(sample_ndarray)
    assert meas_nd.shape == (2, sample_ndarray.shape[1])
    
    # 3. Action on 1D ndarray -> shape (Nsensors, 1)
    meas_1d = lib.action(sample_ndarray[:, 0])
    assert meas_1d.shape == (2, 1)

    # 4. Action with truncation (M)
    meas_trunc = lib.action(sample_funcs, M=1)
    assert meas_trunc.shape == (1, len(sample_funcs))

    # 5. Check __call__ routing
    assert np.allclose(lib(sample_funcs), meas_fl)

def test_invalid_input_type(grid):
    """Test that __call__ rejects unsupported input types."""
    lib = GaussianSensorLibrary(grid, use_centroids=False, gdim=2)
    lib.add_sensor(np.ones(grid.n_points))
    
    with pytest.raises(TypeError, match="Input must be a FunctionsList or a numpy array"):
        lib([1, 2, 3]) # Passing a standard list instead of ndarray/FunctionsList

# -----------------------
# Tests for Specific Sensor Libraries
# -----------------------

def test_gaussian_sensor_library(grid):
    """Test the creation and mathematical definition of Gaussian sensors."""
    lib = GaussianSensorLibrary(grid, use_centroids=False, gdim=2)
    
    # Create library with variance s=0.5
    lib.create_library(s=0.5)
    
    # Should create a sensor for every node since xm_list was not provided
    assert len(lib) == grid.n_points
    
    # Test Normalization: The L1 norm of the defined kernel should be 1.0
    kernel = lib.library(0)
    l1_norm = lib.calculator.L1_norm(kernel)
    assert np.isclose(l1_norm, 1.0)

def test_exponential_sensor_library(grid):
    """Test the creation and mathematical definition of Exponential sensors."""
    lib = ExponentialSensorLibrary(grid, use_centroids=False, gdim=2)
    
    # Create library with variance s=0.5
    lib.create_library(s=0.5)
    
    assert len(lib) == grid.n_points
    
    # Test Normalization: The L1 norm of the defined kernel should be 1.0
    kernel = lib.library(0)
    l1_norm = lib.calculator.L1_norm(kernel)
    assert np.isclose(l1_norm, 1.0)

def test_indicator_function_sensor_library(grid):
    """Test the creation and geometric definition of Indicator sensors."""
    lib = IndicatorFunctionSensorLibrary(grid, use_centroids=False, gdim=2)
    
    # Create library with radius r=0.2
    # The grid is a 1x1 plane centered at origin by default in PyVista
    lib.create_library(r=0.2)
    
    assert len(lib) == grid.n_points
    
    # Indicator function values should strictly be 1.0 or 0.0
    kernel = lib.library(0)
    unique_vals = set(np.unique(kernel))
    assert unique_vals.issubset({0.0, 1.0})

def test_sensor_subset_creation(grid):
    """Test creating a library using a specific subset of xm_list."""
    lib = GaussianSensorLibrary(grid, use_centroids=False, gdim=2)
    
    # Pass only the first two points of the grid
    subset_xm = grid.points[:2, :]
    lib.create_library(s=0.5, xm_list=subset_xm)
    
    # Library should only contain 2 sensors now
    assert len(lib) == 2