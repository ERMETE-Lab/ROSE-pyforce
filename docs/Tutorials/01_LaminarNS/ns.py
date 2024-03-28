import dolfinx
import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem import (Function, FunctionSpace, assemble_scalar, form, 
                        locate_dofs_topological, dirichletbc, petsc)
from ufl import dX, grad, inner, dot, nabla_grad, div
from petsc4py import PETSc


class tentative_velocity():
    def __init__(self, domain: dolfinx.mesh.Mesh, ct: dolfinx.cpp.mesh.MeshTags_int32, ft: dolfinx.cpp.mesh.MeshTags_int32, inlet,
                       physical_param: dict, bound_markers: dict, time_adv = 'EI'):

        self.domain = domain
        self.ct = ct
        self.ft = ft
        self.fdim = domain.geometry.dim - 1
        self.bound_markers = bound_markers

        metadata = {"quadrature_degree": 4}
        self.dx = ufl.Measure('dx', domain=self.domain, subdomain_data=self.ct, metadata=metadata)
        self.ds = ufl.Measure('ds', domain=self.domain, subdomain_data=self.ft, metadata=metadata)

        # Parameters
        self.nu = fem.Constant(self.domain, PETSc.ScalarType(physical_param['nu']))
        self.dt = fem.Constant(self.domain, PETSc.ScalarType(physical_param['dt']))
        self.t = 0
        
        # Functional Spaces
        self.v_cg2 = ufl.VectorElement("Lagrange", self.domain.ufl_cell(), 2)
        self.V = FunctionSpace(self.domain, self.v_cg2)
        self.Q = FunctionSpace(self.domain, ("Lagrange", 1))
        
        # Trial and test functions
        self.u_ = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)

        # Boundary conditions
        fdim = self.domain.topology.dim - 1

        ## Inlet
        self.u_in = Function(self.V)
        self.inlet_velocity = inlet(self.t, physical_param['T'])
        self.u_in.interpolate(self.inlet_velocity)
        self.bcu_inflow = dirichletbc(self.u_in, locate_dofs_topological(self.V, fdim, self.ft.find(bound_markers['inlet'])))
        
        ## Walls
        self.u_nonslip = np.array((0,) * self.domain.geometry.dim, dtype=PETSc.ScalarType)
        self.bcu_walls = dirichletbc(self.u_nonslip, locate_dofs_topological(self.V, fdim, self.ft.find(bound_markers['walls'])), self.V)
        
        ## Obstacle
        self.bcu_obs = dirichletbc(self.u_nonslip, locate_dofs_topological(self.V, fdim, self.ft.find(bound_markers['obstacle'])), self.V)
        
        self.bcu = [self.bcu_inflow, self.bcu_walls, self.bcu_obs]
        
        # Define functions
        self.uOld    = Function(self.V) # solution at t_n-1
        if time_adv != 'EI':
            self.u_n    = Function(self.V) # solution at t_n-2
            
        self.time_adv = time_adv
        self.u_tilde = Function(self.V)
        self.pOld    = Function(self.Q)
        self.pOld.name = "p"
        
    def assembleForm(self, direct=False):
        
        if self.time_adv == 'CN':
            # Time derivative
            self.left_side  = inner(self.u_ / self.dt, self.v) * self.dx
            
            # Advection with Adams-Bashforth approximation
            self.bs = 3 / 2 * self.uOld - 1 / 2 * self.u_n
            self.left_side += inner(dot(self.bs, nabla_grad(0.5 * self.u_)), self.v) * self.dx
            
            # Viscosity
            self.left_side += 0.5 * inner(self.nu * grad(self.u_), grad(self.v)) * self.dx
            
            # Known term: time derivative and pressure term
            self.right_side  = (inner(self.uOld / self.dt - grad(self.pOld), self.v)) * self.dx
            self.right_side -= inner(dot(self.bs, nabla_grad(0.5 * self.uOld)), self.v) * self.dx
            self.right_side -= 0.5 * inner(self.nu * grad(self.uOld), grad(self.v)) * self.dx
            
        elif self.time_adv == 'BDF2':
            # Time derivative
            self.left_side  = inner(3 / 2 * self.u_ / self.dt, self.v) * self.dx
            
            # Advection with Adams-Bashforth approximation
            self.bs = 2 * self.uOld - self.u_n
            self.left_side += inner(dot(self.bs, nabla_grad(self.u_)), self.v) * self.dx
            
            # Viscosity
            self.left_side += inner(self.nu * grad(self.u_), grad(self.v)) * self.dx
            
            # Known term: time derivative and pressure term
            self.right_side = (inner(2 * self.uOld / self.dt - 0.5 * self.u_n / self.dt - grad(self.pOld), self.v)) * self.dx
            
        else:
            # Time derivative
            self.left_side  = inner(self.u_ / self.dt, self.v) * self.dx 
        
            # Advection with linear approximation
            self.left_side += inner(dot(self.uOld, nabla_grad(self.u_)), self.v) * self.dx
            
            # Viscosity
            self.left_side += inner(self.nu * grad(self.u_), grad(self.v)) * self.dx
            # self.left_side = (inner(self.u_, self.v) + 
            #                 self.dt * inner(dot(self.uOld, nabla_grad(self.u_)), self.v) + 
            #                 self.dt * inner(self.nu * grad(self.u_), grad(self.v))
            #                 ) * self.dx
            
            # Known term: time derivative and pressure term
            self.right_side = (inner(self.uOld / self.dt - grad(self.pOld), self.v)) * self.dx
        
        self.a = form(self.left_side)
        self.L = form(self.right_side)
        
        self.A = petsc.create_matrix(self.a)
        self.b = petsc.create_vector(self.L)

        self.solver = PETSc.KSP().create(self.domain.comm)
        self.solver.setOperators(self.A)

        if direct:
            self.solver.setType(PETSc.KSP.Type.PREONLY)
            self.solver.getPC().setType(PETSc.PC.Type.LU)  
        else: 
            self.solver.setType(PETSc.KSP.Type.BCGS)
            self.solver.getPC().setType(PETSc.PC.Type.JACOBI)
            # self.solver.setType(PETSc.KSP.Type.GMRES)
            # self.solver.getPC().setType(PETSc.PC.Type.ILU)
            
    def advance(self, t):
        self.t = t
        self.inlet_velocity.t = t
        self.u_in.interpolate(self.inlet_velocity)
        
        # Assemble matrix
        self.A.zeroEntries()
        fem.petsc.assemble_matrix(self.A, self.a, self.bcu)
        self.A.assemble()
        
        # Assemble rhs vector
        with self.b.localForm() as loc:
            loc.set(0.)
        fem.petsc.assemble_vector(self.b, self.L)
        fem.petsc.apply_lifting(self.b, [self.a], [self.bcu])
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(self.b, self.bcu)
        
        self.solver.solve(self.b, self.u_tilde.vector)
        self.u_tilde.x.scatter_forward()
        
        
class pressure_projection():
    def __init__(self, domain: dolfinx.mesh.Mesh, ct: dolfinx.cpp.mesh.MeshTags_int32, ft: dolfinx.cpp.mesh.MeshTags_int32,
                       physical_param: dict, bound_markers: dict, time_adv = 'EI'):

        self.domain = domain
        self.ct = ct
        self.ft = ft
        self.fdim = domain.geometry.dim - 1
        self.bound_markers = bound_markers

        metadata = {"quadrature_degree": 4}
        self.dx = ufl.Measure('dx', domain=self.domain, subdomain_data=self.ct, metadata=metadata)
        self.ds = ufl.Measure('ds', domain=self.domain, subdomain_data=self.ft, metadata=metadata)

        # Parameters
        self.dt = fem.Constant(self.domain, PETSc.ScalarType(physical_param['dt']))
        
        # Functional Spaces
        self.v_cg2 = ufl.VectorElement("Lagrange", self.domain.ufl_cell(), 2)
        self.V = FunctionSpace(self.domain, self.v_cg2)
        self.Q = FunctionSpace(self.domain, ("Lagrange", 1))
        
        # Trial and test functions
        self.p_ = ufl.TrialFunction(self.Q)
        self.q = ufl.TestFunction(self.Q)

        # Boundary conditions
        fdim = self.domain.topology.dim - 1

        ## Inlet
        self.bcp = [dirichletbc(PETSc.ScalarType(0), locate_dofs_topological(self.Q, fdim, self.ft.find(bound_markers['outlet'])), self.Q)]
        
        # Define functions
        self.u_tilde = Function(self.V)
        self.phi = Function(self.Q)
        if time_adv == 'BDF2':
            self.alpha = 3 / 2
        else: 
            self.alpha = 1.
        
    def assembleForm(self, direct=False):
        # Laplacian term
        self.left_side  = inner(grad(self.p_), grad(self.q)) * self.dx
        
        # Divergence of the tentative velocity
        self.right_side = - self.alpha / self.dt * inner(div(self.u_tilde), self.q) * self.dx
        
        self.a = form(self.left_side)
        self.L = form(self.right_side)
        
        self.A = fem.petsc.assemble_matrix(self.a, bcs = self.bcp)
        self.A.assemble()
        self.b = fem.petsc.create_vector(self.L)

        self.solver = PETSc.KSP().create(self.domain.comm)
        self.solver.setOperators(self.A)

        if direct:
            self.solver.setType(PETSc.KSP.Type.PREONLY)
            self.solver.getPC().setType(PETSc.PC.Type.LU)  
        else: 
            self.solver.setType(PETSc.KSP.Type.MINRES)
            self.solver.getPC().setType(PETSc.PC.Type.HYPRE)
            self.solver.getPC().setHYPREType("boomeramg")
            
    def advance(self, u_tilde: dolfinx.fem.Function):
        
        assert(len(u_tilde.x.array[:]) == len(self.u_tilde.x.array))
        self.u_tilde.x.array[:] = u_tilde.x.array
        
        # Assemble rhs vector
        with self.b.localForm() as loc:
            loc.set(0)
        fem.petsc.assemble_vector(self.b, self.L)
        fem.petsc.apply_lifting(self.b, [self.a], [self.bcp])
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(self.b, self.bcp)

        self.solver.solve(self.b, self.phi.vector)
        self.phi.x.scatter_forward()
        
class update_velocity():
    def __init__(self, domain: dolfinx.mesh.Mesh, ct: dolfinx.cpp.mesh.MeshTags_int32, ft: dolfinx.cpp.mesh.MeshTags_int32, physical_param: dict,
                 time_adv='EI'):

        self.domain = domain
        self.ct = ct
        self.ft = ft
        self.fdim = domain.geometry.dim - 1

        metadata = {"quadrature_degree": 4}
        self.dx = ufl.Measure('dx', domain=self.domain, subdomain_data=self.ct, metadata=metadata)
        self.ds = ufl.Measure('ds', domain=self.domain, subdomain_data=self.ft, metadata=metadata)

        # Parameters
        self.dt = fem.Constant(self.domain, PETSc.ScalarType(physical_param['dt']))
        
        # Functional Spaces
        self.v_cg2 = ufl.VectorElement("Lagrange", self.domain.ufl_cell(), 2)
        self.V = FunctionSpace(self.domain, self.v_cg2)
        self.Q = FunctionSpace(self.domain, ("Lagrange", 1))
        
        # Trial and test functions
        self.u_ = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)

        # Define functions
        self.u_tilde = Function(self.V)
        self.phi = Function(self.Q)
        self.u_new = Function(self.V)
        self.u_new.name = "U"
        
        if time_adv == 'BDF2':
            self.alpha = 3 / 2
        else: 
            self.alpha = 1.
        
        
    def assembleForm(self, direct=False):
        
        # Velocity update lhs
        self.left_side  = inner(self.u_, self.v) * self.dx
        
        # Update term
        self.right_side = inner(self.u_tilde - self.dt / self.alpha * grad(self.phi), self.v) * self.dx
        
        self.a = form(self.left_side)
        self.L = form(self.right_side)
        
        self.A = fem.petsc.assemble_matrix(self.a)
        self.A.assemble()
        self.b = fem.petsc.create_vector(self.L)

        self.solver = PETSc.KSP().create(self.domain.comm)
        self.solver.setOperators(self.A)
        
        if direct:
            self.solver.setType(PETSc.KSP.Type.PREONLY)
            self.solver.getPC().setType(PETSc.PC.Type.LU)  
        else: 
            self.solver.setType(PETSc.KSP.Type.CG)
            self.solver.getPC().setType(PETSc.PC.Type.SOR)
            
    def advance(self, u_tilde: dolfinx.fem.Function, phi):
        
        assert(len(u_tilde.x.array[:]) == len(self.u_tilde.x.array))
        self.u_tilde.x.array[:] = u_tilde.x.array
        assert(len(phi.x.array[:]) == len(self.phi.x.array))
        self.phi.x.array[:] = phi.x.array
        
        # Assemble rhs vector
        with self.b.localForm() as loc:
            loc.set(0)
        fem.petsc.assemble_vector(self.b, self.L)
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        
        self.solver.solve(self.b, self.u_new.vector)
        self.u_new.x.scatter_forward()
        
        
class drag_lift():
    def __init__(self, domain: dolfinx.mesh.Mesh, ft: dolfinx.cpp.mesh.MeshTags_int32, physical_param: dict, obstacle_mark: int,
                 points = [[0.15, 0.2, 0], [0.25, 0.2, 0]]):
        self.normal = -ufl.FacetNormal(domain)
        self.dObs = ufl.Measure("ds", domain=domain, subdomain_data=ft, subdomain_id=obstacle_mark)

        self.nu     = physical_param['nu']
        self.rhoLU2 = physical_param['rhoLU2']

        # Functional Spaces
        self.v_cg2 = ufl.VectorElement("Lagrange", domain.ufl_cell(), 2)
        self.V = FunctionSpace(domain, self.v_cg2)
        self.Q = FunctionSpace(domain, ("Lagrange", 1))
        
        # Define functions
        self.u = Function(self.V)
        self.u_t = inner(ufl.as_vector((self.normal[1], -self.normal[0])), self.u)
        self.p = Function(self.Q)

        self.drag = form( 2 / self.rhoLU2 * (self.nu * inner(grad(self.u_t), self.normal) * self.normal[1] - self.p * self.normal[0]) * self.dObs)
        self.lift = form(-2 / self.rhoLU2 * (self.nu * inner(grad(self.u_t), self.normal) * self.normal[0] + self.p * self.normal[1]) * self.dObs)

        if domain.comm.rank == 0:
            self.C_D = []
            self.C_L = []
            self.t_u = []
            self.t_p = []
    
        self.points = np.array(points)
        
        self.tree = dolfinx.geometry.BoundingBoxTree(domain, domain.geometry.dim)
        self.cell_candidates = dolfinx.geometry.compute_collisions(self.tree, self.points)
        # self.tree = dolfinx.geometry.bb_tree(domain, domain.geometry.dim)
        # self.cell_candidates = dolfinx.geometry.compute_collisions_points(self.tree, self.points)
        
        self.colliding_cells = dolfinx.geometry.compute_colliding_cells(domain, self.cell_candidates, self.points)
        
        self.front_cells = self.colliding_cells.links(0)
        self.back_cells  = self.colliding_cells.links(1)
        if domain.comm.rank == 0:
            self.p_diff = []
            
        self.domain = domain
        
    def compute(self, t: float, dt: float, u_new: dolfinx.fem.Function, p_new: dolfinx.fem.Function):
        
        # Update velocity and pressure
        self.u.x.array[:] = u_new.x.array
        self.p.x.array[:] = p_new.x.array
        
        drag_coeff = self.domain.comm.gather(assemble_scalar(self.drag), root=0)
        lift_coeff = self.domain.comm.gather(assemble_scalar(self.lift), root=0)
        
        p_front = None
        if len(self.front_cells) > 0:
            p_front = self.p.eval(self.points[0], self.front_cells[:1])
        p_front = self.domain.comm.gather(p_front, root=0)
        
        p_back = None
        if len(self.back_cells) > 0:
            p_back = self.p.eval(self.points[1], self.back_cells[:1])
        p_back = self.domain.comm.gather(p_back, root=0)
    
    
        if self.domain.comm.rank == 0:
            self.t_u.append(t)
            self.t_p.append(t - dt / 2)
            
            self.C_D.append( sum(drag_coeff) )
            self.C_L.append( sum(lift_coeff) )
            
            # Choose first pressure that is found from the different processors
            for pressure in p_front:
                if pressure is not None:
                    p_diff_t = pressure[0]
                    break
            for pressure in p_back:
                if pressure is not None:
                    p_diff_t -= pressure[0]
                    break
            self.p_diff.append(p_diff_t)