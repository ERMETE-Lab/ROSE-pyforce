import pytest
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
    
    filename = Path("./test")

    StoreFunctionsList(domain, snap, "test_var", str(filename))
    
    assert (filename.with_suffix(".xdmf")).exists()

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