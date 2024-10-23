import pytest
import numpy as np
from dolfinx.fem import Function, FunctionSpace
from mpi4py import MPI
import ufl

from toy_data import toy_domain

from pyforce.offline.sensors import GaussianSensors  # Replace with the actual import
from pyforce.tools.backends import norms

@pytest.fixture
def domain():
    # Create domain from toy_domain
    return toy_domain(Nh = 40, gdim=2)[0]

@pytest.fixture
def fun_space():
    # Create domain from toy_domain
    return toy_domain(Nh = 40, gdim=2)[1]

# Test for initialization
def test_gaussian_sensors_initialization(domain, fun_space):
    gaussian_sensors = GaussianSensors(domain, fun_space, s=0.1)
    assert gaussian_sensors.s == 0.1 
    assert isinstance(gaussian_sensors.g, Function)

# Define the Gaussian at a sample point
def test_gaussian_sensors_define(domain, fun_space):

    gaussian_sensors = GaussianSensors(domain, fun_space, s=0.1)
    x_m = np.array([0.5, 0.5, 0.0])
    gaussian_function = gaussian_sensors.define(x_m)

    # Check that the function was defined correctly
    assert isinstance(gaussian_function, np.ndarray)
    assert gaussian_function.shape[0] == fun_space.dofmap.index_map.size_local

    # Check that the Gaussian integrates to 1
    _norm = norms(fun_space)
    assert np.isclose(_norm.integral(gaussian_function), 1.0, atol=1e-2), "Gaussian should integrate to 1"

# Define the Riesz representation of the sensor at a sample point
def test_gaussian_sensors_define_riesz(domain, fun_space):
    gaussian_sensors = GaussianSensors(domain, fun_space, s=0.1, assemble_riesz=True)
    x_m = np.array([0.5, 0.5, 0.0])
    riesz_rep = gaussian_sensors.define_riesz(x_m)

    # Check that the Riesz representation is a Function and of the correct space
    assert isinstance(riesz_rep, Function)
    assert riesz_rep.x.array.shape[0] == fun_space.dofmap.index_map.size_local

@pytest.mark.parametrize("xm", [
    (None), # Free selection
    ([np.array([0.2, 0.3, 0.0]), np.array([0.6, 0.7, 0.0])]), # Fixed position
])
def test_gaussian_sensors_create_default(domain, fun_space, xm):
    
    gaussian_sensors = GaussianSensors(domain, fun_space, s=0.1)
    sensor_list = gaussian_sensors.create(xm = xm)

    # Check that a FunctionsList is returned
    assert len(sensor_list) > 0
    assert sensor_list.fun_shape == fun_space.dofmap.index_map.size_local

def test_gaussian_sensors_action_single(domain, fun_space):
    # Test the action_single method with a sample function and sensor
    gaussian_sensors = GaussianSensors(domain, fun_space, s=0.1)

    fun = Function(fun_space)
    fun.interpolate(lambda x: x[0] + x[1])

    x_m = np.array([0.5, 0.5, 0.0])
    sensor = gaussian_sensors.define(x_m)

    # Perform the sensor action
    result = gaussian_sensors.action_single(fun, sensor)

    # Check the result is a float and has a reasonable value
    assert isinstance(result, float)
    assert np.isclose(result, 1., atol=1e-3)  # Expected integral value is 1.

def test_gaussian_sensors_action(domain, fun_space):
    # Test the action method with a sample function and sample list of sensors
    gaussian_sensors = GaussianSensors(domain, fun_space, s=0.1)

    fun = Function(fun_space)
    fun.interpolate(lambda x: x[0] + x[1])

    x_m = [np.array([0.5, 0.5, 0.0]), np.array([0.25, 0.25, 0.0])]
    sensor = gaussian_sensors.create(x_m)

    # Perform the sensor action
    results = gaussian_sensors.action(fun, sensor)

    # Check the result is a float and has a reasonable value
    assert isinstance(results, np.ndarray)
    assert np.isclose(results[0], 1, atol=1e-3)  # Expected integral value is 1.
    assert np.isclose(results[1], 0.5, atol=1e-2)  # Expected integral value is 0.5.