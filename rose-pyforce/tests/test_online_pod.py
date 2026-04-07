import numpy as np
import pytest
import os
from pathlib import Path
from pyvista import examples

from pyforce.offline.pod import POD as OfflinePOD
from pyforce.online.pod import POD as OnlinePOD
from pyforce.online.online_base import SurrogateModelWrapper
from pyforce.tools.functions_list import FunctionsList
from pyforce.tools.write_read import ReadFromOF

# ==========================================
# Mock Surrogate Model for Testing
# ==========================================

class MockSurrogate(SurrogateModelWrapper):
    """
    A simple mock that returns pre-defined coefficients 
    to test the Online POD's estimate() and compute_errors().
    """
    def __init__(self, coeffs):
        self.coeffs = coeffs

    def predict(self, input_vector):
        # In a real scenario, input_vector would be parameters mu
        # Here we just return the stored coefficients
        return self.coeffs

# ==========================================
# Fixtures
# ==========================================

@pytest.fixture
def grid():
    foam_file = Path(examples.download_cavity(load=False))
    case_dir = foam_file.parent
    return ReadFromOF(str(case_dir)).mesh()

@pytest.fixture
def sample_data(grid):
    nodes = grid.points
    func = lambda x, mu: np.sin(mu * x[:, 0]) * np.cos(mu * x[:, 1])
    fun_list = FunctionsList(dofs=nodes.shape[0])
    mu_params = np.linspace(0.1, 10, 20)
    for mu in mu_params:
        fun_list.append(func(nodes, mu))
    return fun_list

@pytest.fixture
def trained_offline_pod(grid, sample_data):
    """Provides a POD model that has already computed a basis."""
    model = OfflinePOD(grid=grid, gdim=3, varname="u")
    model.fit(sample_data)
    model.compute_basis(sample_data, rank=10)
    return model

# ==========================================
# Online POD Tests
# ==========================================

def test_online_pod_init(grid):
    model = OnlinePOD(grid=grid, gdim=3, varname="u")
    assert not hasattr(model, "_basis")

def test_online_pod_set_basis_direct(grid, trained_offline_pod):
    online_model = OnlinePOD(grid=grid, gdim=3, varname="u")
    
    # Set basis directly from the offline result
    online_model.set_basis(basis=trained_offline_pod.pod_modes)
    
    assert len(online_model.basis) == 10
    assert online_model.basis.fun_shape == trained_offline_pod.pod_modes.fun_shape

def test_online_pod_set_basis_load(tmp_path, grid, trained_offline_pod):
    # 1. Save offline basis
    save_path = tmp_path / "modes_folder"
    trained_offline_pod.save(str(save_path))
    
    # 2. Load into online model
    online_model = OnlinePOD(grid=grid, gdim=3, varname="u")
    online_model.set_basis(path_folder=str(save_path))
    
    assert len(online_model.basis) == 10

def test_online_pod_estimate(grid, sample_data, trained_offline_pod):
    online_model = OnlinePOD(grid=grid, gdim=3, varname="u")
    online_model.set_basis(basis=trained_offline_pod.pod_modes)
    
    # Generate some coefficients (e.g., from projection)
    rank = 5
    true_coeffs = trained_offline_pod.reduce(sample_data, N=rank)
    
    # Use mock surrogate to "predict" these coefficients
    mock_surrogate = MockSurrogate(true_coeffs)
    
    # Estimate
    estimation = online_model.estimate(mock_surrogate, input_vector=np.array([1, 2, 3]))
    
    assert isinstance(estimation, FunctionsList)
    assert len(estimation) == len(sample_data)
    assert estimation.fun_shape == sample_data.fun_shape

def test_online_pod_compute_errors(grid, sample_data, trained_offline_pod):
    online_model = OnlinePOD(grid=grid, gdim=3, varname="u")
    online_model.set_basis(basis=trained_offline_pod.pod_modes)
    
    rank = 8
    # In a real case, we'd use a real regressor. 
    # Here we project to get "perfect" coefficients to check error logic.
    coeffs = trained_offline_pod.reduce(sample_data, N=rank)
    mock_surrogate = MockSurrogate(coeffs)
    
    # We pass dummy input_vector because our mock ignores it
    results = online_model.compute_errors(
        snaps=sample_data, 
        coeff_model=mock_surrogate, 
        input_vector=np.zeros(len(sample_data)),
        verbose=False
    )
    
    assert len(results.mean_abs_err) == rank
    assert len(results.mean_rel_err) == rank
    assert "StateEstimation" in results.computational_time
    # Relative error should be very low since we used exact projection coefficients
    assert results.mean_rel_err[-1] < 1e-3

def test_online_pod_internal_reduce_reconstruct(grid, sample_data, trained_offline_pod):
    """Tests the protected methods _reduce and _reconstruct"""
    online_model = OnlinePOD(grid=grid, gdim=3, varname="u")
    online_model.set_basis(basis=trained_offline_pod.pod_modes)
    
    rank = 4
    coeffs = online_model._reduce(sample_data, N=rank)
    assert coeffs.shape == (rank, len(sample_data))
    
    rec = online_model._reconstruct(coeffs)
    assert len(rec) == len(sample_data)
    assert rec.fun_shape == sample_data.fun_shape