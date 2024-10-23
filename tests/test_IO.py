import pytest
import os
import numpy as np
from pathlib import Path

from pyvista import examples

import ufl
from dolfinx.fem import FunctionSpace, Function
from mpi4py import MPI
from dolfinx.mesh import create_unit_square, Mesh
from pyforce.tools.functions_list import FunctionsList
from pyforce.tools.write_read import StoreFunctionsList, ImportH5, ReadFromOF

def test_store_functions_list():
    domain = create_unit_square(MPI.COMM_WORLD, 10, 10)
    V = FunctionSpace(domain, ("Lagrange", 1))
    snap = FunctionsList(V)
    for _ in range(5):
        f = Function(V)
        f.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
        snap.append(f.vector.getArray())
    
    path_test = "testIO/test"

    StoreFunctionsList(domain, snap, "u", path_test)

    assert os.path.exists(path_test+".xdmf")
    assert os.path.exists(path_test+".h5")
    
def test_import_h5():
    domain = create_unit_square(MPI.COMM_WORLD, 10, 10)
    V = FunctionSpace(domain, ("Lagrange", 1))
    snap = FunctionsList(V)
    for _ in range(5):
        f = Function(V)
        f.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
        snap.append(f.vector.getArray())
    
    filename = Path("./test")
    StoreFunctionsList(domain, snap, "test_var", str(filename))
    
    snap_loaded, _ = ImportH5(V, str(filename), "test_var")
    
    assert len(snap) == len(snap_loaded)
    
    os.remove(str(filename)+".xdmf")
    os.remove(str(filename)+".h5")

## add OpenFOAM import test using pyvista examples
@pytest.fixture
def setup_of_reader():
    filename = examples.download_cavity(load=False)[:-9]
    return ReadFromOF(filename)

def test_readOF_initialization(setup_of_reader):
    
    assert (setup_of_reader.mode == 'pyvista') or (setup_of_reader.mode == 'fluidfoam')

@pytest.mark.parametrize("var_name, is_vector", [
    ('U', True), ('p', False)
])
def test_reader_of_import_field(setup_of_reader, var_name, is_vector):
    
    snaps = setup_of_reader.import_field(var_name, vector=is_vector, verbose=False)

    assert len(snaps[0]) == len(snaps[1])
    assert np.allclose(snaps[1], setup_of_reader.reader.time_values[1:], rtol = 1e-2)

def test_reader_of_create_mesh(setup_of_reader):
    assert isinstance(setup_of_reader.create_mesh(), Mesh)

@pytest.mark.parametrize("var_name, is_vector", [
    ('U', True), ('p', False)
])
def test_reader_of_foam_to_dolfinx(setup_of_reader, var_name, is_vector):
    
    snaps = setup_of_reader.import_field(var_name, vector=is_vector, verbose=False)
    domain = setup_of_reader.create_mesh()

    if is_vector:
        V = FunctionSpace(domain, ufl.VectorElement("Lagrange", domain.ufl_cell(), 1))
    else:
        V = FunctionSpace(domain, ("Lagrange", 1))

    snap_dolfinx = setup_of_reader.foam_to_dolfinx(V, snaps[0])

    assert len(snap_dolfinx) == len(snaps[0])
    assert isinstance(snap_dolfinx, FunctionsList)