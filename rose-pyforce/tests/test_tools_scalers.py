import numpy as np
import pytest

from pyforce.tools.scalers import StandardScaler, MinMaxScaler
from pyforce.tools.functions_list import FunctionsList

@pytest.fixture
def sample_data():
    """Create a simple FunctionsList with known values."""
    X = FunctionsList(dofs = 3)

    random_matrix = np.random.rand(3, 5)
    for i in range(random_matrix.shape[1]):
        X.append(random_matrix[:, i])
    return X


# ---------------------------------------------------------------------
# --- StandardScaler tests --------------------------------------------
# ---------------------------------------------------------------------
def test_standard_scaler_basic_fit(sample_data):
    scaler = StandardScaler()
    scaler.fit(sample_data)

    # Expected means and stds (feature-wise)
    mat = sample_data.return_matrix()
    assert np.allclose(scaler._mean, mat.mean(axis=1))
    assert np.allclose(scaler._std, mat.std(axis=1))

# ---------------------------------------------------------------------
# --- StandardScaler tests --------------------------------------------
# ---------------------------------------------------------------------
def test_standard_scaler_basic_fit(sample_data):
    scaler = StandardScaler()
    scaler.fit(sample_data)

    # Expected means and stds (feature-wise)
    mat = sample_data.return_matrix()
    assert np.allclose(scaler._mean, mat.mean(axis=1))
    assert np.allclose(scaler._std, mat.std(axis=1))


def test_standard_scaler_transform_inverse(sample_data):
    scaler = StandardScaler()
    scaler.fit(sample_data)

    X_t = scaler.transform(sample_data)
    X_rt = scaler.inverse_transform(X_t)

    # Check reconstruction
    for i in range(len(sample_data)):
        original = sample_data(i)
        reconstructed = X_rt(i)
        assert np.allclose(original, reconstructed, atol=1e-10)


def test_standard_scaler_zero_std_raises():
    X = FunctionsList(dofs=3)
    X.append(np.array([1, 2, 3]))
    X.append(np.array([1, 2, 3]))

    scaler = StandardScaler()

    with pytest.raises(ValueError):
        scaler.fit(X)

def test_standard_scaler_wrt_ic(sample_data):
    scaler = StandardScaler(wrt_ic=True)
    scaler.fit(sample_data)

    # Mean should be equal to mean of first snapshot
    expected_mean = np.ones(3) * sample_data(0).mean()
    assert np.allclose(scaler._mean, expected_mean)


def test_standard_scaler_global_scale(sample_data):
    scaler = StandardScaler(global_scale=True)
    scaler.fit(sample_data)

    mat = sample_data.return_matrix()
    expected_mean = np.ones(3) * mat.mean()
    assert np.allclose(scaler._mean, expected_mean)


# ---------------------------------------------------------------------
# --- MinMaxScaler tests ----------------------------------------------
# ---------------------------------------------------------------------
def test_minmax_scaler_fit(sample_data):
    scaler = MinMaxScaler()
    scaler.fit(sample_data)

    mat = sample_data.return_matrix()
    assert np.allclose(scaler._min, mat.min(axis=1))
    assert np.allclose(scaler._max, mat.max(axis=1))


def test_minmax_scaler_transform_inverse(sample_data):
    scaler = MinMaxScaler()
    scaler.fit(sample_data)

    X_t = scaler.transform(sample_data)
    X_rt = scaler.inverse_transform(X_t)

    for i in range(len(sample_data)):
        assert np.allclose(sample_data(i), X_rt(i), atol=1e-10)


def test_minmax_scaler_wrt_ic(sample_data):
    scaler = MinMaxScaler(wrt_ic=True)
    scaler.fit(sample_data)

    v0 = sample_data(0)
    assert np.allclose(scaler._min, np.ones(3) * v0.min())
    assert np.allclose(scaler._max, np.ones(3) * v0.max())


def test_minmax_scaler_global(sample_data):
    scaler = MinMaxScaler(global_scale=True)
    scaler.fit(sample_data)

    mat = sample_data.return_matrix()
    assert np.allclose(scaler._min, np.ones(3) * mat.min())
    assert np.allclose(scaler._max, np.ones(3) * mat.max())
