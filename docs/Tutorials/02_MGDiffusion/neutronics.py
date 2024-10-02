import dolfinx
import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem import (Function, FunctionSpace, assemble_scalar, form, 
                        locate_dofs_topological, dirichletbc)
from ufl import grad, inner, dot
from petsc4py import PETSc
import gmsh
from mpi4py import MPI
from dolfinx.io.gmshio import read_from_msh, model_to_mesh

def create_anl11a2_mesh(use_msh : bool = False, save_mesh : bool = False):
    gdim = 2

    model_rank = 0
    mesh_comm = MPI.COMM_WORLD

    # If the snapshots are generated from a different computer than the one used for the offline/online simulation, there may be issues in the number of dofs
    if use_msh:
        domain, ct, ft = read_from_msh("ANL11A2_octave.msh", comm = mesh_comm, rank = model_rank, gdim = gdim)
    else:
        # Initialize the gmsh module
        gmsh.initialize()

        # Load the .geo file
        gmsh.merge('ANL11A2_octave.geo')
        gmsh.model.geo.synchronize()

        # Set algorithm (adaptive = 1, Frontal-Delaunay = 6)
        gmsh.option.setNumber("Mesh.Algorithm", 6)

        gmsh.model.mesh.generate(gdim)
        gmsh.model.mesh.optimize("Netgen")
        
         # Save the mesh
        if save_mesh:
            gmsh.write("cyl_dfg2D.msh")
        
        # Domain
        domain, ct, ft = model_to_mesh(gmsh.model, comm = mesh_comm, rank = model_rank, gdim = gdim )
        gmsh.finalize()

    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)

    return domain, ct, ft
    

class steady_neutron_diff():
    r""""
    This class implements the solution of the steady multigroup neutron diffusion equation through the :math:`k`-eigenvalue method.
    
    
    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        Mesh imported.
    ct : dolfinx.cpp.mesh.MeshTags_int32
        Cell tags of the regions in the domain.
    ft : dolfinx.cpp.mesh.MeshTags_int32
        Face tags of the boundaries.
    physical_param : dict
        Dictionary with the main physical parameters at reference condition.
    regions_markers : dict
        List of the region markers inside the domain.
    void_bound_marker : int
        Integer describing the index of the void boundary.
    albedo : np.ndarray, optional (Default = None)
        If not `None`, it, denoted as :math:`\gamma`, must be an array of :math:`G`, as the energy groups, to impose the following BC :math:`-D_g\nabla \phi_g\cdot \mathbf{n} = \gamma \phi_g`.
    coupling : str, optional (Default = None)
        If not `None`, it indicates the type of coupling with the temperature: *log* and *sqrt* are available.
            
    """
    def __init__(self, domain: dolfinx.mesh.Mesh, ct: dolfinx.cpp.mesh.MeshTags_int32, ft: dolfinx.cpp.mesh.MeshTags_int32, 
                       physical_param: dict, regions_markers: list, void_bound_marker: int, albedo: np.ndarray = None, coupling: str = None):

        self.domain = domain
        self.ft = ft
        self.ct = ct
        self.fdim = domain.geometry.dim - 1
        self.void_bound_marker = void_bound_marker

        metadata = {"quadrature_degree": 4}
        self.dx = ufl.Measure('dx', domain=self.domain, subdomain_data=self.ct, metadata=metadata)
        self.ds = ufl.Measure('ds', domain=self.domain, subdomain_data=self.ft, metadata=metadata)

        # Defining the number of energy groups
        self.G = physical_param['Energy Groups']

        # Flux Functional Space
        self.P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
        spaces = list()
        for g in range(self.G):
            spaces.append(self.P1)
        self.V = FunctionSpace(self.domain, ufl.MixedElement(spaces))
        
        # Material Properties Functional Space
        self.Qn = FunctionSpace(domain, ("DG", 0))
        if coupling is not None:
            self.T = Function(self.Qn)
            self.Tref = physical_param['Tref']
            self.T.x.set(self.Tref)
            
        self.coupling = coupling
        self.phys_param = physical_param
        self.regions = regions_markers        

        # Neutronics Parameters
        self.Dg = list()
        self.xs_ag = list()
        self.nu_xs_fg = list()
        self.chi_g = list()
        self.B2z_g = list()
        self.xs_sg_to_gp = list()
        for g in range(self.G):
            self.Dg.append(Function(self.Qn))
            self.xs_ag.append(Function(self.Qn))
            self.nu_xs_fg.append(Function(self.Qn))
            self.chi_g.append(Function(self.Qn))
            self.B2z_g.append(Function(self.Qn))
            
            self.xs_sg_to_gp.append(list())
            for gp in range(self.G):
                if gp != g:
                    self.xs_sg_to_gp[g].append(Function(self.Qn))
                else:
                    self.xs_sg_to_gp[g].append(None)
        
        # Trial and test functions
        self.phi = ufl.TrialFunctions(self.V)
        self.varphi = ufl.TestFunctions(self.V)

        # Boundary conditions - if albedo is None then homogeneous Dirichlet BC is imposed
        self.albedo = albedo
        if self.albedo is None:
            self.zero = Function(self.V)
            self.zero.x.set(0.)
            self.bcs = list()
            for g in range(self.G):
                self.bcs.append( dirichletbc(self.zero.sub(g), 
                                             locate_dofs_topological((self.V.sub(g), self.V.sub(g).collapse()[0]), 
                                                                     self.fdim, self.ft.find(self.void_bound_marker)), self.V.sub(g)) )
            
        # Old function for the inverse power method
        self.old = Function(self.V)
        self.new = Function(self.V)
        self.k = Function(self.Qn)

        self.normalised_fluxes = list()
        for g in range(self.G):
            self.normalised_fluxes.append(Function(self.V.sub(g).collapse()[0]))
            
    def assembleForm(self, direct: bool = True):
        r"""
        This function assembles the variational forms and the correspondent linear algebra structures.
        
        Parameters
        ----------
        direct : bool, optional (Default=True)
            If `True`, a direct linear solver is used (Gauss Elimination), otherwise an iterative one is adopted (preconditioned GMRES with ILU).
            
        """
        # Left Hand Side - Diffusion with Neumann Boundary Conditions - if albedo is not None
        self.left_side = inner(self.Dg[0] * grad(self.phi[0]), grad(self.varphi[0])) * self.dx
        for g in range(1, self.G):
            self.left_side += inner(self.Dg[g] * grad(self.phi[g]), grad(self.varphi[g])) * self.dx
        if self.albedo is not None:
            assert len(self.albedo) == self.G
            for g in range(self.G):
                self.left_side  += inner(self.albedo[g] * self.phi[g], self.varphi[g]) * self.ds(self.void_bound_marker)
        
        # Left Hand Side - absorption and leakage
        for g in range(self.G):
            self.left_side += inner( (self.xs_ag[g] + self.Dg[g] * self.B2z_g[g]) * self.phi[g], self.varphi[g]) * self.dx 
        
        # Left Hand Side - Scattering
        for g in range(self.G):
            for gp in range(self.G):
                if gp != g:
                    self.left_side += inner(self.xs_sg_to_gp[g][gp] * self.phi[g],  self.varphi[g]) * self.dx 
                    self.left_side -= inner(self.xs_sg_to_gp[gp][g] * self.phi[gp], self.varphi[g]) * self.dx

        # Right Hand Side - Fission
        self.right_side   = 1. / self.k * self.chi_g[0] * inner(self.nu_xs_fg[0] * self.old.sub(0), self.varphi[0]) * self.dx 
        self.right_side2  =               self.chi_g[0] * inner(self.nu_xs_fg[0] * self.old.sub(0), self.varphi[0]) * self.dx 
        
        for g in range(self.G):
            for gp in range(1, self.G):
                self.right_side  += 1. / self.k * self.chi_g[g] * inner(self.nu_xs_fg[gp] * self.old.sub(gp), self.varphi[g]) * self.dx 
                self.right_side2 +=               self.chi_g[g] * inner(self.nu_xs_fg[gp] * self.old.sub(gp), self.varphi[g]) * self.dx 
        
        self.a  = form(self.left_side)
        self.b  = form(self.right_side)
        self.b2 = form(self.right_side2)

        self.A   = fem.petsc.create_matrix(self.a)
        self.rhs = fem.petsc.create_vector(self.b)
        self.B   = fem.petsc.create_vector(self.b2)

        self.solver = PETSc.KSP().create(self.domain.comm)
        self.solver.setOperators(self.A)

        if direct:
            self.solver.setType(PETSc.KSP.Type.PREONLY)
            self.solver.getPC().setType(PETSc.PC.Type.LU)  
        else: 
            self.solver.setType(PETSc.KSP.Type.GMRES)
            self.solver.getPC().setType(PETSc.PC.Type.ILU)   

    def update_xs(self):
        """
        This function updates the cross sections according to the coupling scheme chosen.
        """
        if self.coupling is None: # no thermal feedback
            for idx, regionI in enumerate(self.regions):
                cells = self.ct.find(regionI)
                for g in range(self.G):
                    self.Dg[g].x.array[cells]       = self.phys_param['D'][g][idx]
                    self.xs_ag[g].x.array[cells]    = self.phys_param['xs_a'][g][idx]
                    self.nu_xs_fg[g].x.array[cells] = self.phys_param['nu_xs_f'][g][idx]
                    self.chi_g[g].x.array[cells]    = self.phys_param['chi'][g][idx]
                    self.B2z_g[g].x.array[cells]    = self.phys_param['B2z'][g][idx]
                    for gp in range(self.G):
                        if g != gp:
                            self.xs_sg_to_gp[g][gp].x.array[cells] = self.phys_param['xs_s'][g][gp][idx]
                            
        elif self.coupling == 'log': # log law
            for idx, regionI in enumerate(self.regions):
                cells = self.ct.find(regionI)
                for g in range(self.G):
                    self.Dg[g].x.array[cells]       = self.phys_param['D'][g][0][idx]       + self.phys_param['D'][g][1][idx]       * np.log(self.T.x.array[cells] / self.Tref)
                    self.xs_ag[g].x.array[cells]    = self.phys_param['xs_a'][g][0][idx]    + self.phys_param['xs_a'][g][1][idx]    * np.log(self.T.x.array[cells] / self.Tref)
                    self.nu_xs_fg[g].x.array[cells] = self.phys_param['nu_xs_f'][g][0][idx] + self.phys_param['nu_xs_f'][g][1][idx] * np.log(self.T.x.array[cells] / self.Tref)
                    self.chi_g[g].x.array[cells]    = self.phys_param['chi'][g][0][idx]     + self.phys_param['chi'][g][1][idx]     * np.log(self.T.x.array[cells] / self.Tref)
                    self.B2z_g[g].x.array[cells]    = self.phys_param['B2z'][g][idx]
                    for gp in range(self.G):
                        if g != gp:
                            self.xs_sg_to_gp[g][gp].x.array[cells] = self.phys_param['xs_s'][g][gp][0][idx] + self.phys_param['xs_s'][g][gp][1][idx] * np.log(self.T.x.array[cells] / self.Tref)
                            
        elif self.coupling == 'sqrt': # sqrt law
            for idx, regionI in enumerate(self.regions):
                cells = self.ct.find(regionI)
                for g in range(self.G):
                    self.Dg[g].x.array[cells]       = self.phys_param['D'][g][0][idx]       * ( 1 + self.phys_param['D'][g][1][idx]       * ((np.sqrt(self.T.x.array[cells]) - np.sqrt(self.Tref))) )
                    self.xs_ag[g].x.array[cells]    = self.phys_param['xs_a'][g][0][idx]    * ( 1 + self.phys_param['xs_a'][g][1][idx]    * ((np.sqrt(self.T.x.array[cells]) - np.sqrt(self.Tref))) )
                    self.nu_xs_fg[g].x.array[cells] = self.phys_param['nu_xs_f'][g][0][idx] * ( 1 + self.phys_param['nu_xs_f'][g][1][idx] * ((np.sqrt(self.T.x.array[cells]) - np.sqrt(self.Tref))) )
                    self.chi_g[g].x.array[cells]    = self.phys_param['chi'][g][0][idx]     * ( 1 + self.phys_param['chi'][g][1][idx]     * ((np.sqrt(self.T.x.array[cells]) - np.sqrt(self.Tref))) )
                    self.B2z_g[g].x.array[cells]    = self.phys_param['B2z'][g][idx]
                    for gp in range(self.G):
                        if g != gp:
                            self.xs_sg_to_gp[g][gp].x.array[cells] = self.phys_param['xs_s'][g][gp][0][idx]  * (1 + self.phys_param['xs_s'][g][gp][1][idx] * ((np.sqrt(self.T.x.array[cells]) - np.sqrt(self.Tref))) )
        
    def solve(self, temperature: dolfinx.fem.Function = None, power: float = 1, nu: float = 2.41, Ef: float = 1.6e-19 * 200e6, 
              tol: float = 1e-10, maxIter: int = 200, LL: int = 50, verbose: bool = False):

        r"""
        This function implements the :math:`k`-eigenvalue method: starting from an initial non-trivial guess of eigenvalue-eigenvector pair :math:`\left(k_{\text{eff}}^{(0)},\,\boldsymbol{\phi}^{(0)}\right)`, at each iteration :math:`n\geq 1` the following linear system is solved

        .. math::
            \mathbb{A}\boldsymbol{\phi}^{(n)} = \frac{1}{k_{\text{eff}}^{(n-1)}} \mathbb{B}\boldsymbol{\phi}^{(n-1)}

        The convergence check is made on the relative difference for the eigenvalue as
        
        .. math::
            \frac{|k_{\text{eff}}^{(n)} - k_{\text{eff}}^{(n-1)}|}{k_{\text{eff}}^{(n)}} < \epsilon

        
        Parameters
        ----------
        temperature : dolfinx.fem.Function, optional (Default = None)
            Temperature field as a `dolfinx.fem.Function`.
        power : float, optional (Default = 1)
            Power level :math:`P` used to normalise the flux as :math:`P = \displaystyle\int_V E_f\cdot \sum_{g=1}^G \Sigma_{f,g}\phi_g \, dV`.
        nu : float, optional (Default = 2.41)
            Number of neutrons emitted by the fission event.
        Ef : float, optional (Default = `1.6e-19 * 200e6`)
            Energy released by the fission event.
        tol : float, optional (Default = 1e-10)
            Tolerance of the inverse power method, :math:`\epsilon`.
        maxIter : int, optional (Default = 200)
            Maximum iteration for the inverse power method.
        LL : int, optional (Default = 50)
            Frequency for printing the output.
        verbose : bool, optional (Default = False)
            If `True`, the output is printed every `LL` iterations.
        
        Returns
        -------
        normalised_fluxes : list[dolfinx.fem.Function]
            List of `dolfinx.fem.Function` with the solution of the inverse power method.
        k_eff_ : float
            Converged eigenvalue.
        """
        
        if self.coupling is not None:
            assert(temperature is not None)
            # Updating temperature
            if len(self.T.x.array[:]) == len(temperature.x.array[:]):
                self.T.x.array[:] = temperature.x.array[:]
            else:
                self.T.interpolate(fem.Expression( temperature, self.Qn.element.interpolation_points() ))

        # Update XS
        self.update_xs()

        # Assembling LHS matrix
        self.A.zeroEntries()
        if self.albedo is None:
            fem.petsc.assemble_matrix(self.A, self.a, self.bcs)
        else:
            fem.petsc.assemble_matrix(self.A, self.a)
        self.A.assemble()  

        # Setting initial guess
        self.old.x.set(1.)
        self.k.x.set(1.02)
        k_eff_list = []

        error = 1.
        ii = 0
        
        while error > tol:
            
            # Assembling RHS and applying Dirichlet BC with lifting, if albedo is None
            with self.rhs.localForm() as loc_b:
                loc_b.set(0)
            fem.petsc.assemble_vector(self.rhs, self.b)
            if self.albedo is None:
                fem.petsc.apply_lifting(self.rhs, [self.a], [self.bcs])
                self.rhs.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
                fem.petsc.set_bc(self.rhs, self.bcs)

            # Solve linear problem
            self.solver.solve(self.rhs, self.new.vector)
            self.new.x.scatter_forward()  

            # Computing LHS as vector using new solution
            tmpA = fem.petsc.assemble_vector(form(ufl.replace(self.left_side, 
                                                              {self.phi[g]: self.new.sub(g) for g in range(self.G)})))
            if self.albedo is None:
                fem.petsc.apply_lifting(tmpA, [self.a], [self.bcs])
                tmpA.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
                fem.petsc.set_bc(tmpA, self.bcs)
            Aphi = tmpA[:]

            # Updating old solution
            self.old.x.array[:] = self.new.x.array[:]

            # Computing RHS as vector using new solution
            with self.B.localForm() as loc_b:
                loc_b.set(0)
            fem.petsc.assemble_vector(self.B, self.b2)
            if self.albedo is None:
                fem.petsc.apply_lifting(self.B, [self.a], [self.bcs])
                self.B.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
                fem.petsc.set_bc(self.B, self.bcs)
            Bphi = self.B.array[:]
            
            k_eff_list.append( np.dot(Bphi, Bphi) / np.dot(Aphi, Bphi) )
            self.k.x.set(k_eff_list[ii])
            
            if ii > 0:
                error = abs(k_eff_list[ii] - k_eff_list[ii-1]) / k_eff_list[ii]

                if ii % LL == 0 and verbose == True:
                    print(f'    Iter {ii+0:03} | k_eff: {k_eff_list[ii]:.6f} | Rel Error: {error :.3e}')

                if error <= tol and verbose == True:
                    print(f'    Neutronics converged with {ii+0:03} iter | k_eff: {k_eff_list[ii]:.8f} | rho: {(1-1./k_eff_list[ii])*1e5:.2f} pcm | Rel Error: {error :.3e}')
            
            ii += 1

            if ii > maxIter:
                print('Max iteration reached! Exiting loop')
                error = 0

        # Normalising Flux according to reactor power
        power_form = Ef / nu / k_eff_list[-1] * self.nu_xs_fg[0] * self.new.sub(0)
        for g in range(1, self.G):
            power_form += Ef / nu / k_eff_list[-1] * self.nu_xs_fg[g] * self.new.sub(g)
        normalisation = power / assemble_scalar(form( power_form * self.dx) ) 

        res_norm_flux = list()
        for g in range(self.G):
            res_norm_flux.append(np.zeros_like((self.normalised_fluxes[g].x.array,)))
            self.normalised_fluxes[g].interpolate(fem.Expression (normalisation * self.new.sub(g), 
                                                             self.V.sub(g).collapse()[0].element.interpolation_points()) )
            res_norm_flux[g][:] = self.normalised_fluxes[g].x.array[:]
                                                  
        return res_norm_flux, k_eff_list[-1]
    

class transient_neutron_diff():
    r""""
    This class implements the solution of the transient multigroup neutron diffusion equation coupled with the precursors equations with a semi-implicit first-order advancement scheme.
    
    
    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        Mesh imported.
    ct : dolfinx.cpp.mesh.MeshTags_int32
        Cell tags of the regions in the domain.
    ft : dolfinx.cpp.mesh.MeshTags_int32
        Face tags of the boundaries.
    physical_param : dict
        Dictionary with the main physical parameters at reference condition.
    regions_markers : dict
        List of the region markers inside the domain.
    void_bound_marker : int
        Integer describing the index of the void boundary.
    albedo : np.ndarray, optional (Default = None)
        If not `None`, it, denoted as :math:`\gamma`, must be an array of :math:`G`, as the energy groups, to impose the following BC :math:`-D_g\nabla \phi_g\cdot \mathbf{n} = \gamma \phi_g`.
    coupling : str, optional (Default = None)
        If not `None`, it indicates the type of coupling with the temperature: *log* and *sqrt* are available.
            
    """
    def __init__(self, domain: dolfinx.mesh.Mesh, ct: dolfinx.cpp.mesh.MeshTags_int32, ft: dolfinx.cpp.mesh.MeshTags_int32, 
                       physical_param: dict, regions_markers: list, void_bound_marker: int,
                       albedo: np.ndarray = None, coupling: str = None):

        self.domain = domain
        self.ct = ct
        self.ft = ft
        self.fdim = domain.geometry.dim - 1
        self.void_bound_marker = void_bound_marker
        
        metadata = {"quadrature_degree": 4}
        self.dx = ufl.Measure('dx', domain=self.domain, subdomain_data=self.ct, metadata=metadata)
        self.ds = ufl.Measure('ds', domain=self.domain, subdomain_data=self.ft, metadata=metadata)

        # Defining the number of energy groups
        self.G = physical_param['Energy Groups']

        # Defining the number of precursors groups
        assert(len(physical_param['beta_l']) == len(physical_param['beta_l']))
        self.prec_groups = len(physical_param['beta_l'])

        self.P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
        self.P0 = ufl.FiniteElement("DG", self.domain.ufl_cell(), 0)
        spaces = list()
        for g in range(self.G):
            spaces.append(self.P1)
        for ll in range(self.prec_groups):
            spaces.append(self.P0)
        self.V = FunctionSpace(self.domain, ufl.MixedElement(spaces) )
        self.Q = FunctionSpace(self.domain, self.P1)
        
        # Material Properties Functional Space
        self.Qn = FunctionSpace(domain, ("DG", 0))
        if coupling is not None:
            self.T = Function(self.Qn)
            self.Tref = physical_param['Tref']
            self.T.x.set(self.Tref)
            
        self.coupling = coupling
        self.phys_param = physical_param
        self.regions = regions_markers        
            
        # Neutronics Parameters
        self.Dg = list()
        self.xs_ag = list()
        self.nu_xs_fg = list()
        self.chi_g = list()
        self.B2z_g = list()
        self.xs_sg_to_gp = list()
        for g in range(self.G):
            self.Dg.append(Function(self.Qn))
            self.xs_ag.append(Function(self.Qn))
            self.nu_xs_fg.append(Function(self.Qn))
            self.chi_g.append(Function(self.Qn))
            self.B2z_g.append(Function(self.Qn))
            
            self.xs_sg_to_gp.append(list())
            for gp in range(self.G):
                if gp != g:
                    self.xs_sg_to_gp[g].append(Function(self.Qn))
                else:
                    self.xs_sg_to_gp[g].append(None)
        
        # Transient parameters
        self.recip_v = list()
        for g in range(self.G):
            self.recip_v.append(fem.Constant(self.domain, PETSc.ScalarType(1./physical_param['v'][g])))

        self.beta_l   = list()
        self.lambda_l = list()

        for ll in range(self.prec_groups):
            self.beta_l.append(Function(self.Qn))
            self.lambda_l.append(Function(self.Qn))
            
            for idx, regionI in enumerate(self.regions):
                cells = self.ct.find(regionI)
                self.beta_l[ll].x.array[cells]   = self.phys_param['beta_l'][ll][idx]
                self.lambda_l[ll].x.array[cells] = self.phys_param['lambda_p_l'][ll][idx]
                 
        self.beta = Function(self.Qn)
        for idx, regionI in enumerate(self.regions):
            cells = self.ct.find(regionI)
            self.beta.x.array[cells] = np.sum(self.phys_param['beta_l'], axis = 0)[idx] 
            
        # Trial and test functions
        self.trial = ufl.TrialFunctions(self.V)
        self.test = ufl.TestFunctions(self.V)

        # Boundary conditions - if albedo is None then homogeneous Dirichlet BC is imposed
        self.albedo = albedo
        if self.albedo is None:
            self.zero = Function(self.V)
            self.zero.x.set(0.)
            self.bcs = list()
            for g in range(self.G):
                self.bcs.append( dirichletbc(self.zero.sub(g), 
                                             locate_dofs_topological((self.V.sub(g), self.V.sub(g).collapse()[0]), 
                                                                     self.fdim, self.ft.find(self.void_bound_marker)), self.V.sub(g)) )
        self.dt = Function(self.Qn)
        
        # Old function
        self.old = Function(self.V)
        self.new = Function(self.V)
        self.k = Function(self.Qn)

    def assembleForm(self, phi_ss: list[dolfinx.fem.Function], dt: float, 
                     nu: float = 2.41, Ef: float = 1.6e-19 * 200e6, direct = True):
        r"""
        This function assembles the variational forms and the correspondent linear algebra structures.
        
        Parameters
        ----------
        phi_ss : list[dolfinx.fem.Function
            List of the steady state group fluxes.
        dt : float
            Time step size.
        nu : float, optional (Default = 2.41)
            Number of neutrons emitted by the fission event.
        Ef : float, optional (Default = `1.6e-19 * 200e6`)
            Energy released by the fission event.
        direct : bool, optional (Default=True)
            If `True`, a direct linear solver is used (Gauss Elimination), otherwise an iterative one is adopted (preconditioned GMRES with ILU).
            
        """
    
        self.dt.x.set(dt)
        
        # Left Hand Side - Flux - Time derivative
        self.left_side = inner(self.recip_v[0] / self.dt * self.trial[0], self.test[0]) * self.dx
        for g in range(1, self.G):
            self.left_side += inner(self.recip_v[g] / self.dt * self.trial[g], self.test[g]) * self.dx
        
        # Left Hand Side - Flux - Diffusion with Neumann Boundary Conditions - if albedo is not None
        for g in range(self.G):
            self.left_side += inner(self.Dg[g] * grad(self.trial[g]), grad(self.test[g])) * self.dx
        if self.albedo is not None:
            assert len(self.albedo) == self.G
            for g in range(self.G):
                self.left_side  += inner(self.albedo[g] * self.trial[g], self.test[g]) * self.ds(self.void_bound_marker)
        
        # Left Hand Side - Flux - absorption and leakage
        for g in range(self.G):
            self.left_side += inner( (self.xs_ag[g] + self.Dg[g] * self.B2z_g[g]) * self.trial[g], self.test[g]) * self.dx 
        
        # Left Hand Side - Flux - Scattering
        for g in range(self.G):
            for gp in range(self.G):
                if gp != g:
                    self.left_side += inner(self.xs_sg_to_gp[g][gp] * self.trial[g],  self.test[g]) * self.dx 
                    self.left_side -= inner(self.xs_sg_to_gp[gp][g] * self.trial[gp], self.test[g]) * self.dx

        # Left Hand Side - Flux - Fission Prompt and Delayed
        for g in range(self.G):
            for gp in range(self.G):
                self.left_side -= (1-self.beta) * self.chi_g[g] * inner(self.nu_xs_fg[gp] * self.trial[gp], self.test[g]) * self.dx 
            for ll in range(self.prec_groups):
                self.left_side -= self.chi_g[g] * dot(self.lambda_l[ll] * self.trial[self.G+ll], self.test[g]) * self.dx

        # Right Hand Side - Flux - Time derivative at previous time step
        self.right_side = inner(self.recip_v[0] / self.dt * self.old.sub(0), self.test[0]) * self.dx
        for g in range(1, self.G):
            self.right_side += inner(self.recip_v[g] / self.dt * self.old.sub(g), self.test[g]) * self.dx
            
        # Left Hand Side - Precursors
        for ll in range(self.prec_groups):
            self.left_side += inner(1. / self.dt * self.trial[self.G+ll],      self.test[self.G+ll]) * self.dx # Time Derivative
            self.left_side += inner(self.lambda_l[ll] * self.trial[self.G+ll], self.test[self.G+ll]) * self.dx # Decay
            for gp in range(self.G):
                self.left_side -= inner(self.beta_l[ll] * self.nu_xs_fg[gp] * self.trial[gp], self.test[self.G+ll]) * self.dx # Fission Product
        
        # Right Hand Side - Precursors
        for ll in range(self.prec_groups):
            self.right_side += inner(1. / self.dt * self.old.sub(self.G+ll), self.test[self.G+ll]) * self.dx # Time Derivative
        
        self.bilinear = fem.form(self.left_side)
        self.linear   = fem.form(self.right_side)

        self.A = fem.petsc.create_matrix(self.bilinear)
        self.b = fem.petsc.create_vector(self.linear)

        self.solver = PETSc.KSP().create(self.domain.comm)
        self.solver.setOperators(self.A)
        
        if direct:
            self.solver.setType(PETSc.KSP.Type.PREONLY)
            self.solver.getPC().setType(PETSc.PC.Type.LU)  
        else: 
            self.solver.setType(PETSc.KSP.Type.GMRES)
            self.solver.getPC().setType(PETSc.PC.Type.ILU)   

        # Initialise flux and precursors
        for g in range(self.G):
            self.old.sub(g).interpolate(fem.Expression(phi_ss[g], self.V.sub(g).element.interpolation_points()))
 
        self.update_xs()
        
        for ll in range(self.prec_groups): 
            power_form = self.nu_xs_fg[0] * self.old[0]
            for g in range(1, self.G):
                power_form += self.nu_xs_fg[g] * self.old[g]
            self.old.sub(self.G+ll).interpolate(fem.Expression(
                                                ufl.conditional(ufl.gt(self.lambda_l[ll], 0.), 
                                                                self.beta_l[ll] * power_form / self.lambda_l[ll], 0.), 
                                           self.V.sub(self.G+ll).collapse()[0].element.interpolation_points()))

        # Initialise power form
        self.power_form = form( Ef / nu * power_form * self.dx )

    def update_xs(self):
        """
        This function updates the cross sections according to the coupling scheme chosen.
        """
        
        if self.coupling is None: # no thermal feedback
            for idx, regionI in enumerate(self.regions):
                cells = self.ct.find(regionI)
                for g in range(self.G):
                    self.Dg[g].x.array[cells]       = self.phys_param['D'][g][idx]
                    self.xs_ag[g].x.array[cells]    = self.phys_param['xs_a'][g][idx]
                    self.nu_xs_fg[g].x.array[cells] = self.phys_param['nu_xs_f'][g][idx]
                    self.chi_g[g].x.array[cells]    = self.phys_param['chi'][g][idx]
                    self.B2z_g[g].x.array[cells]    = self.phys_param['B2z'][g][idx]
                    for gp in range(self.G):
                        if g != gp:
                            self.xs_sg_to_gp[g][gp].x.array[cells] = self.phys_param['xs_s'][g][gp][idx]
        
        elif self.coupling == 'log': # log law
            for idx, regionI in enumerate(self.regions):
                cells = self.ct.find(regionI)
                for g in range(self.G):
                    self.Dg[g].x.array[cells]       = self.phys_param['D'][g][0][idx]       + self.phys_param['D'][g][1][idx]       * np.log(self.T.x.array[cells] / self.Tref)
                    # self.xs_ag[g].x.array[cells]    = self.phys_param['xs_a'][g][0][idx]    + self.phys_param['xs_a'][g][1][idx]    * np.log(self.T.x.array[cells] / self.Tref)
                    self.nu_xs_fg[g].x.array[cells] = self.phys_param['nu_xs_f'][g][0][idx] + self.phys_param['nu_xs_f'][g][1][idx] * np.log(self.T.x.array[cells] / self.Tref)
                    self.chi_g[g].x.array[cells]    = self.phys_param['chi'][g][0][idx]     + self.phys_param['chi'][g][1][idx]     * np.log(self.T.x.array[cells] / self.Tref)
                    self.B2z_g[g].x.array[cells]    = self.phys_param['B2z'][g][idx]
                    for gp in range(self.G):
                        if g != gp:
                            self.xs_sg_to_gp[g][gp].x.array[cells] = self.phys_param['xs_s'][g][gp][0][idx] + self.phys_param['xs_s'][g][gp][1][idx] * np.log(self.T.x.array[cells] / self.Tref)
                            
        elif self.coupling == 'sqrt': # sqrt law
            for idx, regionI in enumerate(self.regions):
                cells = self.ct.find(regionI)
                for g in range(self.G):
                    self.Dg[g].x.array[cells]       = self.phys_param['D'][g][0][idx]       * ( 1 + self.phys_param['D'][g][1][idx]       * ((np.sqrt(self.T.x.array[cells]) - np.sqrt(self.Tref))) )
                    # self.xs_ag[g].x.array[cells]    = self.phys_param['xs_a'][g][0][idx]    * ( 1 + self.phys_param['xs_a'][g][1][idx]    * ((np.sqrt(self.T.x.array[cells]) - np.sqrt(self.Tref))) )
                    self.nu_xs_fg[g].x.array[cells] = self.phys_param['nu_xs_f'][g][0][idx] * ( 1 + self.phys_param['nu_xs_f'][g][1][idx] * ((np.sqrt(self.T.x.array[cells]) - np.sqrt(self.Tref))) )
                    self.chi_g[g].x.array[cells]    = self.phys_param['chi'][g][0][idx]     * ( 1 + self.phys_param['chi'][g][1][idx]     * ((np.sqrt(self.T.x.array[cells]) - np.sqrt(self.Tref))) )
                    self.B2z_g[g].x.array[cells]    = self.phys_param['B2z'][g][idx]
                    for gp in range(self.G):
                        if g != gp:
                            self.xs_sg_to_gp[g][gp].x.array[cells] = self.phys_param['xs_s'][g][gp][0][idx]  * (1 + self.phys_param['xs_s'][g][gp][1][idx] * ((np.sqrt(self.T.x.array[cells]) - np.sqrt(self.Tref))) )
            
    def advance(self, t: float, sigma_ag_transient_value: list, 
                      temperature: dolfinx.fem.Function = None, return_prec: bool = False):
        r"""
        This function implements a single step forward in time.

        Parameters
        ----------
        t : float
            Time to advance to, say :math:`t_{n+1}`.
        sigma_ag_transient_value : list
            List of callable function dependent on time used to impose variation of the reactivity.
        temperature : dolfinx.fem.Function, optional (Default = None)
            Temperature field as a `dolfinx.fem.Function`.
        return_prec : bool, optional (Default = False)
            If `True`, the precursors are returned in the solution.
            
        Returns
        -------
        power : float
            Power at time :math:`t_{n+1}`, defined as :math:`P(t) = \displaystyle\int_V E_f\cdot \sum_{g=1}^G \Sigma_{f,g}\phi_g \, dV`.
        solution : list[dolfinx.fem.Function]
            List of `dolfinx.fem.Function` with the output of this time step.
        """
        if self.coupling is not None:
            assert(temperature is not None)
            # Updating temperature
            if len(self.T.x.array[:]) == len(temperature.x.array[:]):
                self.T.x.array[:] = temperature.x.array[:]
            else:
                self.T.interpolate(fem.Expression( temperature, self.Qn.element.interpolation_points() ))
            
        # assembling LHS matrix
        self.A.zeroEntries()
        for g in range(self.G):
            for idx, regionI in enumerate(self.regions):
                cells = self.ct.find(regionI)
                if self.coupling is None: # no thermal feedback
                    self.xs_ag[g].x.array[cells] = sigma_ag_transient_value[g](t)[idx]
                else:
                    if self.coupling == 'log':
                        self.xs_ag[g].x.array[cells] = sigma_ag_transient_value[g](t)[idx] + self.phys_param['xs_a'][g][1][idx]    * np.log(self.T.x.array[cells] / self.Tref)
                    elif self.coupling == 'sqrt':
                        self.xs_ag[g].x.array[cells] = sigma_ag_transient_value[g](t)[idx] * ( 1 + self.phys_param['xs_a'][g][1][idx]    * ((np.sqrt(self.T.x.array[cells]) - np.sqrt(self.Tref))) )
                    self.update_xs()
                
        if self.albedo is None:
            fem.petsc.assemble_matrix(self.A, self.bilinear, self.bcs)
        else:
            fem.petsc.assemble_matrix(self.A, self.bilinear)
        self.A.assemble()  

        # Update the rhs and apply lifting
        with self.b.localForm() as loc_b:
            loc_b.set(0)
        fem.petsc.assemble_vector(self.b, self.linear)
        if self.albedo is None:
            fem.petsc.apply_lifting(self.b, [self.bilinear], [self.bcs])
            self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            fem.petsc.set_bc(self.b, self.bcs)
        
        self.solver.solve(self.b, self.new.vector)
        self.new.x.scatter_forward()

        # Update old
        self.old.x.array[:] = self.new.x.array[:]
        
        # Compute power
        power = assemble_scalar(self.power_form)
        
        if return_prec:        
            return power, [self.old.sub(ii).collapse() for ii in range(self.G + self.prec_groups)]
        else:
            return power, [self.old.sub(gg).collapse() for gg in range(self.G)]