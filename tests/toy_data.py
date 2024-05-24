import pytest
import numpy as np
from dolfinx import mesh
from dolfinx.fem import FunctionSpace, Function, dirichletbc, locate_dofs_geometrical
from mpi4py import MPI
import ufl
from petsc4py import PETSc
from dolfinx import fem

from pyforce.tools.functions_list import FunctionsList
from pyforce.tools.backends import LoopProgress

def toy_domain(Nh: int = 40, gdim: int = 1):
    
    if gdim == 1:
        domain = mesh.create_unit_interval(MPI.COMM_WORLD, Nh)
    else:
        domain = mesh.create_unit_square(MPI.COMM_WORLD, Nh, Nh)
    V = FunctionSpace(domain, ("Lagrange", 1))

    return domain, V

def left_boundary(x):
    return np.isclose(x[0], 0)
def right_boundary(x):
    return np.isclose(x[0], 1)
def top_boundary(x):
    return np.isclose(x[1], 0)
def bot_boundary(x):
    return np.isclose(x[1], 1)

def toy_dataset(domain: mesh.Mesh, V: FunctionSpace, Ns: list = [10,10]):
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    metadata = {"quadrature_degree": 4}
    dx = ufl.Measure('dx', domain=domain, metadata=metadata)
    
    k = .2
    
    lhs = k * ufl.dot(ufl.grad(u), ufl.grad(v)) * dx
    
    q3 = Function(V)
    rhs = ufl.dot(q3, v) * dx
    
    bilin_form = fem.form(lhs)
    lin_form   = fem.form(rhs)
    
    # Boundary Conditions
    bcs = [dirichletbc(0., locate_dofs_geometrical(V, left_boundary), V),
           dirichletbc(0., locate_dofs_geometrical(V, right_boundary), V)]
    if domain.topology.dim > 1:
        bcs.append(dirichletbc(0., locate_dofs_geometrical(V, top_boundary), V))
        bcs.append(dirichletbc(0., locate_dofs_geometrical(V, bot_boundary), V))
    
    matrix = fem.petsc.assemble_matrix(bilin_form, bcs)
    matrix.assemble()
    vec_b  = fem.petsc.create_vector(lin_form)
    
    # Solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(matrix)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)  
    
    mu = [  np.linspace(1, 3, Ns[0]),
            np.linspace(0.2, 0.5, Ns[1])]
    
    gdim = domain.topology.dim
    
    snapshots = FunctionsList(V)
    params = list()
    
    bar = LoopProgress('Creating Dataset', final = np.prod(Ns))
    
    for kk in range(Ns[0]):
        for jj in range(Ns[1]):
            q3.interpolate(lambda x: np.sin(mu[0][kk] * np.pi * x[0]) * np.exp( - sum([(x[ii])**2 for ii in range(gdim)])/mu[1][jj]))
            
            with vec_b.localForm() as loc_b:
                loc_b.set(0)
            fem.petsc.assemble_vector(vec_b, lin_form)
            fem.petsc.apply_lifting(vec_b, [bilin_form], [bcs])
            vec_b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            fem.petsc.set_bc(vec_b, bcs)
            
            # Solve linear problem
            solution = Function(V).copy()
            solver.solve(vec_b, solution.vector)
            solution.x.scatter_forward()  
                    
            snapshots.append(solution)   
            params.append(np.array([mu[0][kk], mu[1][jj]]))     
            bar.update(1)
            
    return np.asarray(params), snapshots