import pytest
import os
import numpy as np
from pathlib import Path

from dolfinx.fem import FunctionSpace, Function
from mpi4py import MPI
from dolfinx.mesh import create_unit_square
from pyforce.tools.functions_list import FunctionsList
from pyforce.tools.write_read import StoreFunctionsList, ImportH5, import_OF

def test_store_functions_list():
    domain = create_unit_square(MPI.COMM_WORLD, 10, 10)
    V = FunctionSpace(domain, ("Lagrange", 1))
    snap = FunctionsList(V)
    for _ in range(5):
        f = Function(V)
        f.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
        snap.append(f.vector.getArray())
    
    path_test = "./testIO/test"

    StoreFunctionsList(domain, snap, "u", path_test)

    assert os.path.exists(path_test+".xdmf")
    assert os.path.exists(path_test+".h5")
    
    os.remove(path_test+".xdmf")
    os.remove(path_test+".h5")

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