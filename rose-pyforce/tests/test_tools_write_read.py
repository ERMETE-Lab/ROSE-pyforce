import pytest
import numpy as np
from pathlib import Path
from pyvista import examples

from pyforce.tools.write_read import ReadFromOF
from pyforce.tools.functions_list import FunctionsList
import pyvista as pv

@pytest.fixture
def setup_of_reader(tmp_path):
    """
    Prepare a reader using a downloaded OpenFOAM example.
    The pyvista cavity case is a safe and widely used test.
    """
    # PyVista returns ".../cavity/cavity.foam"
    foam_file = Path(examples.download_cavity(load=False))
    case_dir = foam_file.parent   # folder containing system/, constant/, 0/, etc.
    return ReadFromOF(str(case_dir))

# -------------------------------------------------------------
# Initialization tests
# -------------------------------------------------------------

def test_initialization_creates_or_detects_foam(tmp_path):
    case_dir = tmp_path / "case"
    case_dir.mkdir()

    # No foam file initially → ReadFromOF must create one
    reader = ReadFromOF(str(case_dir))
    foam_files = list(case_dir.glob("*.foam"))
    assert len(foam_files) == 1
    assert foam_files[0].name == "foam.foam"

    # Now add another foam file → reader should still load without error
    custom = case_dir / "myfile.foam"
    custom.write_text("")

    reader2 = ReadFromOF(str(case_dir))
    foam_files2 = list(case_dir.glob("*.foam"))
    assert len(foam_files2) == 2
    # Internal PyVista reader should be initialized
    assert hasattr(reader2, "reader")


# -------------------------------------------------------------
# Mesh tests
# -------------------------------------------------------------

def test_mesh_returns_unstructured_grid(setup_of_reader):
    mesh = setup_of_reader.mesh()
    assert isinstance(mesh, pv.UnstructuredGrid)
    assert mesh.n_cells > 0


def test_save_mesh(tmp_path, setup_of_reader):
    out = tmp_path / "mesh_output"
    setup_of_reader.save_mesh(str(out))
    assert (tmp_path / "mesh_output.vtk").exists()


# -------------------------------------------------------------
# Field import tests
# -------------------------------------------------------------

def test_import_field_scalar_pressure(setup_of_reader):
    """
    Import scalar field 'p' using pyvista.
    """
    snaps, times = setup_of_reader.import_field("p", import_mode='pyvista', verbose=False)

    # snaps must be a FunctionsList
    assert isinstance(snaps, FunctionsList)
    assert len(snaps) == len(times)

    # DoFs match mesh cells
    mesh = setup_of_reader.mesh()
    assert snaps.fun_shape == mesh.n_cells

    # Snapshot content is flat array of correct size
    snap0 = snaps[0]
    assert isinstance(snap0, np.ndarray)
    assert snap0.shape == (mesh.n_cells,)


def test_import_field_vector_velocity(setup_of_reader):
    """
    Import vector field U.
    Should flatten to (n_cells * 3,)
    """
    snaps, times = setup_of_reader.import_field("U", import_mode='pyvista', verbose=False)
    mesh = setup_of_reader.mesh()

    expected_dofs = mesh.n_cells * 3
    assert snaps.fun_shape == expected_dofs
    assert len(snaps) == len(times)

    snap0 = snaps[0]
    assert snap0.size == expected_dofs


def test_import_field_point_data(setup_of_reader):
    """
    Test extract_cell_data=False → use point_data
    """
    snaps, times = setup_of_reader.import_field(
        "p",
        import_mode='pyvista',
        extract_cell_data=False,
        verbose=False,
    )

    mesh = setup_of_reader.reader.read()['internalMesh']
    expected_dofs = mesh.n_points

    assert snaps.fun_shape == expected_dofs
    assert len(snaps) == len(times)
    assert snaps[0].shape == (expected_dofs,)


# -------------------------------------------------------------
# FunctionsList behavior tests
# -------------------------------------------------------------

def test_functionslist_basic_operations():
    fl = FunctionsList(dofs=3)
    fl.append(np.array([1, 2, 3]))
    fl.append(np.array([4, 5, 6]))

    assert len(fl) == 2
    assert np.array_equal(fl[0], np.array([1, 2, 3]))
    assert fl.shape() == (3, 2)

    M = fl.return_matrix()
    assert M.shape == (3, 2)
    assert np.array_equal(M[:, 0], np.array([1, 2, 3]))


def test_functionslist_linear_combination():
    fl = FunctionsList(dofs=3)
    fl.append(np.array([1, 0, 0]))
    fl.append(np.array([0, 2, 0]))
    fl.append(np.array([0, 0, 3]))

    vec = np.array([1, 1, 1])
    out = fl.lin_combine(vec)
    assert np.array_equal(out, np.array([1, 2, 3]))