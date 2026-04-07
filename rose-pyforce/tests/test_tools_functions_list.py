import numpy as np
import pytest
import os
import h5py
import pyvista as pv
from unittest.mock import patch

from pyforce.tools.functions_list import FunctionsList, train_test_split

# Assuming the module is named storage.py, adjust the import as needed
# from pyforce.tools.storage import FunctionsList, train_test_split
# For the sake of this script, assuming FunctionsList and train_test_split are in the namespace

# -----------------------
# Fixtures
# -----------------------

@pytest.fixture
def dofs():
    return 10

@pytest.fixture
def num_snaps():
    return 5

@pytest.fixture
def snap_matrix(dofs, num_snaps):
    """Creates a mock snapshot matrix of shape (dofs, num_snaps)"""
    np.random.seed(42)
    return np.random.rand(dofs, num_snaps)

@pytest.fixture
def fun_list(dofs, snap_matrix):
    """Creates an empty FunctionsList and populates it with a matrix."""
    fl = FunctionsList(dofs=dofs)
    fl.build_from_matrix(snap_matrix)
    return fl

# -----------------------
# Tests for FunctionsList
# -----------------------

def test_initialization_with_dofs(dofs):
    """Test standard initialization with degrees of freedom."""
    fl = FunctionsList(dofs=dofs)
    assert fl.fun_shape == dofs
    assert len(fl) == 0

def test_initialization_with_matrix(snap_matrix):
    """Test initialization directly from a snapshot matrix."""
    fl = FunctionsList(snap_matrix=snap_matrix)
    assert fl.fun_shape == snap_matrix.shape[0]
    assert len(fl) == snap_matrix.shape[1]

def test_initialization_error():
    """Test that ValueError is raised when neither dofs nor snap_matrix is provided."""
    with pytest.raises(ValueError):
        FunctionsList()

def test_append(dofs):
    """Test appending a new function to the list."""
    fl = FunctionsList(dofs=dofs)
    new_fun = np.ones(dofs)
    fl.append(new_fun)
    assert len(fl) == 1
    assert np.allclose(fl[0], new_fun)

def test_append_shape_mismatch(dofs):
    """Test that appending a function with wrong shape raises AssertionError."""
    fl = FunctionsList(dofs=dofs)
    wrong_fun = np.ones(dofs + 5)
    with pytest.raises(AssertionError):
        fl.append(wrong_fun)

def test_magic_methods(fun_list, num_snaps):
    """Test __len__, __call__, __getitem__ (index), and __iter__."""
    assert len(fun_list) == num_snaps
    
    # __call__ and __getitem__ should return the same array
    assert np.allclose(fun_list(2), fun_list[2])
    
    # Test iterator
    count = 0
    for f in fun_list:
        assert f.shape[0] == fun_list.fun_shape
        count += 1
    assert count == num_snaps

def test_getitem_slice(fun_list):
    """Test slicing a FunctionsList returns a new FunctionsList."""
    sliced_fl = fun_list[1:4]
    assert isinstance(sliced_fl, FunctionsList)
    assert len(sliced_fl) == 3
    assert sliced_fl.fun_shape == fun_list.fun_shape

def test_delete_and_clear(fun_list, num_snaps):
    """Test deleting single elements and clearing the list."""
    fun_list.delete(0)
    assert len(fun_list) == num_snaps - 1
    
    fun_list.clear()
    assert len(fun_list) == 0
    assert isinstance(fun_list._list, list)

def test_copy(fun_list):
    """Test that copy returns an independent list of arrays."""
    list_copy = fun_list.copy()
    assert isinstance(list_copy, list)
    assert len(list_copy) == len(fun_list)
    assert list_copy is not fun_list._list

def test_shape_method(fun_list, dofs, num_snaps):
    """Test the shape method."""
    assert fun_list.shape() == (dofs, num_snaps)

def test_sort(fun_list, num_snaps):
    """Test reordering the list."""
    original_first = fun_list[0].copy()
    original_last = fun_list[-1].copy()
    
    # Reverse the list using sort
    order = list(range(num_snaps))[::-1]
    fun_list.sort(order)
    
    assert np.allclose(fun_list[0], original_last)
    assert np.allclose(fun_list[-1], original_first)

def test_sort_mismatch(fun_list):
    """Test sorting with incorrect order length raises AssertionError."""
    with pytest.raises(AssertionError):
        fun_list.sort([0, 1])

def test_return_matrix(fun_list, snap_matrix):
    """Test reconstructing the matrix from the list of arrays."""
    mat = fun_list.return_matrix()
    assert mat.shape == snap_matrix.shape
    assert np.allclose(mat, snap_matrix)

def test_lin_combine(fun_list, num_snaps, dofs):
    """Test linear combination of stored functions."""
    coeffs = np.ones(num_snaps)
    comb = fun_list.lin_combine(coeffs)
    
    expected = np.sum(fun_list.return_matrix(), axis=1)
    
    assert comb.shape == (dofs,)
    assert np.allclose(comb, expected)

def test_aggregations(fun_list):
    """Test min, max, mean, std methods."""
    mat = fun_list.return_matrix()
    
    assert np.allclose(fun_list.min(axis=1), np.min(mat, axis=1))
    assert np.allclose(fun_list.max(axis=0), np.max(mat, axis=0))
    assert np.allclose(fun_list.mean(), np.mean(mat))
    assert np.allclose(fun_list.std(), np.std(mat))

# -----------------------
# Tests for Storage (I/O)
# -----------------------

def test_store_h5(fun_list, tmp_path):
    """Test saving data to HDF5 format."""
    filepath = tmp_path / "test_data"
    
    fun_list.store(var_name="snapshots", filename=str(filepath), format='h5')
    full_path = str(filepath) + ".h5"
    
    assert os.path.exists(full_path)
    
    with h5py.File(full_path, 'r') as f:
        assert "snapshots" in f
        loaded_data = f["snapshots"][:]
        assert np.allclose(loaded_data, fun_list.return_matrix())

def test_store_npz(fun_list, tmp_path):
    """Test saving data to NPZ format."""
    filepath = tmp_path / "test_data"
    
    fun_list.store(var_name="snapshots", filename=str(filepath), format='npz')
    full_path = str(filepath) + ".npz"
    
    assert os.path.exists(full_path)
    
    loaded = np.load(full_path)
    assert "snapshots" in loaded
    assert np.allclose(loaded["snapshots"], fun_list.return_matrix())

def test_store_invalid_format(fun_list, tmp_path):
    """Test unsupported format raises ValueError."""
    filepath = tmp_path / "test_data"
    with pytest.raises(ValueError):
        fun_list.store(var_name="snaps", filename=str(filepath), format='csv')

# -----------------------
# Tests for train_test_split
# -----------------------

def test_train_test_split(fun_list, num_snaps):
    """Test splitting a parameter list and a FunctionsList."""
    params = [f"param_{i}" for i in range(num_snaps)]
    
    tr_p, te_p, tr_f, te_f = train_test_split(
        params=params, 
        fun_list=fun_list, 
        test_size=0.4, 
        random_state=42
    )
    
    # 5 snaps, 40% test size -> 3 train, 2 test
    assert len(tr_p) == 3
    assert len(te_p) == 2
    
    assert len(tr_f) == 3
    assert len(te_f) == 2
    
    assert isinstance(tr_f, FunctionsList)
    assert isinstance(te_f, FunctionsList)
    assert tr_f.fun_shape == fun_list.fun_shape

def test_train_test_split_mismatch(fun_list, num_snaps):
    """Test that dimension mismatch between params and fun_list raises AssertionError."""
    # Create params list shorter than the number of snapshots
    params_short = [f"param_{i}" for i in range(num_snaps - 1)]
    
    with pytest.raises(AssertionError):
        train_test_split(params=params_short, fun_list=fun_list)