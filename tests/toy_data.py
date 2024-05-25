import pytest
import numpy as np
from dolfinx import mesh
from dolfinx.fem import FunctionSpace, Function, dirichletbc, locate_dofs_geometrical, locate_dofs_topological
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

def toy_neutronic_dataset(domain: mesh.Mesh, Ns: list = [10,10]):
    
    gdim = domain.topology.dim
    
    # Functional Space
    G = 2
    P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
    V = FunctionSpace(domain, ufl.MixedElement([P1]*G))

    # Trial and test functions
    (phi_1, phi_2) = ufl.TrialFunctions(V)
    (varphi_1, varphi_2) = ufl.TestFunctions(V)

    metadata = {"quadrature_degree": 4}
    dx = ufl.Measure('dx', domain=domain, metadata=metadata)

    # Boundaries
    bound_marker  = 10
    fdim = gdim - 1
    boundaries = [ 
                  (bound_marker,  lambda x: np.isclose(x[0], 0)), 
                  (bound_marker,  lambda x: np.isclose(x[0], 1))
                  ]
    if gdim > 1:
        boundaries.extend([ 
                           (bound_marker,  lambda x: np.isclose(x[1], 0)), 
                           (bound_marker,  lambda x: np.isclose(x[1], 1))
                           ])

    # Find boundaries face tags
    facet_indices, facet_markers = [], []
    for (marker, locator) in boundaries:
        facets = mesh.locate_entities(domain, fdim, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full(len(facets), marker))
    facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
    facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
    sorted_facets = np.argsort(facet_indices)
    ft = mesh.meshtags(domain, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

    # Boundary Conditions
    zero = Function(V)
    zero.x.set(0.)
    bcs = list()
    for g in range(G):
        bcs.append( dirichletbc(zero.sub(g), 
                                locate_dofs_topological((V.sub(g), V.sub(g).collapse()[0]), 
                                                        fdim, ft.find(bound_marker)), V.sub(g)) )
        
    # Nuclear Data - 2 groups
    # nusigma_f1_value = 0.015986 / k_eff # Fission
    # nusigma_f2_value = 0.349442 / k_eff
    sigma_a1     = 0.030855 # Absorption
    sigma_a2     = 0.235927
    sigma_s_1to2 = 0.055029 # Scattering Fast    -> Thermal
    sigma_s_2to1 = 0.000000 # Scattering Thermal -> Fast
    D_1          = 0.544130 # Fast    Diffusion Coeff
    D_2          = 0.117483 # Thermal Diffusion Coeff

    # Variational formulations
    left_side   = ufl.dot(D_1 * ufl.grad(phi_1), ufl.grad(varphi_1)) * dx + ufl.dot( (sigma_a1 + sigma_s_1to2) * phi_1, varphi_1) * dx 
    left_side  -= ufl.dot(sigma_s_2to1 * phi_2, varphi_1) * dx 
    left_side  += ufl.dot(D_2 * ufl.grad(phi_2), ufl.grad(varphi_2)) * dx + ufl.dot( (sigma_a2 + sigma_s_2to1) * phi_2, varphi_2) * dx 
    left_side  -= ufl.dot(sigma_s_1to2 * phi_1, varphi_2) * dx 

    source_term = Function(V)
    right_side   = ufl.dot(source_term[0], varphi_1) * dx 
    right_side  += ufl.dot(source_term[1], varphi_2) * dx 

    # Assembling matrices
    bilin_form = fem.form(left_side)
    lin_form   = fem.form(right_side)

    matrix = fem.petsc.assemble_matrix(bilin_form, bcs)
    matrix.assemble()
    vec_b  = fem.petsc.create_vector(lin_form)

    # Solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(matrix)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)  
    
    
    mu = [  np.linspace(0.2, 0.5, Ns[0]),
            np.linspace(0.2, 0.5, Ns[1])]
    
    snapshots = FunctionsList(V)
    params = list()
    
    bar = LoopProgress('Creating Dataset', final = np.prod(Ns))
    
    for kk in range(Ns[0]):
        for jj in range(Ns[1]):
            
            source_term.sub(0).interpolate(lambda x: np.exp( - sum([(x[ii])**2 for ii in range(gdim)]) / mu[0][kk]))
            source_term.sub(1).interpolate(lambda x: np.exp( - sum([(x[ii]-0.25)**2 for ii in range(gdim)] / mu[1][kk])))
                        
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
            
            # Append parametric solution
            snapshots.append(solution)   
            params.append(np.array([mu[0][kk], mu[1][jj]]))     
            bar.update(1)
            
    return np.asarray(params), snapshots