import pytest
from dolfinx.fem import Function, FunctionSpace
import pyvista
import ufl

from toy_data import toy_domain

from pyforce.tools.plotting import grids

@pytest.mark.parametrize("varname, log_plot, mag_plot", [
    ("u", False, False),  # Scalar field test case
    ("v", True, False),
    ("w", False, True),  # Vector field test cases
])
def test_grids(varname, log_plot, mag_plot):

    Nh = 40
    domain = toy_domain(Nh=Nh, gdim=2)[0]

    # Create function space based on varname
    if varname == "w":
        V = FunctionSpace(domain, ufl.VectorElement("Lagrange", domain.ufl_cell(), 1))
    else:
        V = FunctionSpace(domain, ("Lagrange", 1))
        
    # Create a simple field (scalar or vector)
    if V.num_sub_spaces > 0:
        u = Function(V)
        u.sub(0).interpolate(lambda x: x[0] ** 2)  # x-component field
        u.sub(1).interpolate(lambda x: x[1] ** 3)  # x-component field
    else:
        u = Function(V)
        u.interpolate(lambda x: x[0] ** 2 + x[1] ** 2)  # Scalar field

    # Call the grids function
    grid, values = grids(u, varname, log_plot, mag_plot)

    # Check the output grid and values
    assert isinstance(grid, pyvista.UnstructuredGrid)
    assert grid.n_points == V.tabulate_dof_coordinates().shape[0]

    # Adjust expected data length based on scalar/vector field
    expected_data_length = len(u.x.array)
    if V.num_sub_spaces > 0:
        expected_data_length = len(u.x.array) // V.num_sub_spaces

    assert len(grid.point_data[varname]) == expected_data_length
    assert len(values) == expected_data_length