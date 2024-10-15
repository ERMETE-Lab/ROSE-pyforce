# I/O tools
# Author: Stefano Riva, PhD Student, NRG, Politecnico di Milano
# Latest Code Update: 15 October 2024
# Latest Doc  Update: 15 October 2024

# from pyforce.tools.functions_list import FunctionsList
from .functions_list import FunctionsList
from .backends import LoopProgress
from dolfinx.io import XDMFFile, gmshio
from mpi4py import MPI
import h5py
from dolfinx.fem import Function, FunctionSpace
import os
import numpy as np
import fluidfoam as of
import pandas as pd
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator

def StoreFunctionsList(domain, snap: FunctionsList, var_name: str, filename: str, order = None):
    """
    This function can be used to save in `xdmf/h5` files scalar or vector list of functions.

    Parameters
    ----------
    domain : dolfinx.mesh
        Mesh containing the geometry.
    snap : FunctionsList
        List of functions to save.
    varname : str
        Name of the variable.
    filename : str
        Name of the file to save as xdmf/h5 file.
    order : optional (Default = `None`)
        It must be passed as a list of integers containing the ordered indeces.
    """
    xdmf = XDMFFile(domain.comm, filename+".xdmf", "w")
    xdmf.write_mesh(domain)

    fun_to_store = Function(snap.fun_space)

    for ii in range(len(snap)):

        with fun_to_store.vector.localForm() as loc:
            loc.set(0.0)
        fun_to_store.x.array[:] = snap(ii)
        fun_to_store.name = var_name
        if order is None:
            xdmf.write_function(fun_to_store, ii * 1.)
        else:
            xdmf.write_function(fun_to_store, order[ii])
    xdmf.close()

def ImportH5(V: FunctionSpace, filename: str, var_name: str, verbose = False):
    """
    This function can be used to load from xdmf/h5 files scalar or vector list of functions.

    Parameters
    ----------
    V : FunctionSpace
        Functional space in which the functions live.
    varname : str
        Name of the variable.
    filename : str
        Name of the file to read as xdmf/h5 file.
    verbose: boolean, (Default = False) 
        If `True`, printing is enabled
        
    Returns
    -------
    snap : FunctionsList
        Imported list of functions
    params_np : list
        Imported list of parameters
    """
    filename = filename+'.h5'
    data = h5py.File(filename, 'r')

    dataset_list = list(data['Function/'+str(var_name)].keys())
    snap = FunctionsList(V)
    param = []

    if V.num_sub_spaces == 0:
        vector_field = False
        bar_msg = 'Importing scalar field - '+var_name
    else:
        vector_field = True
        gdim = V.num_sub_spaces
        bar_msg = 'Importing vectir field - '+var_name

    if verbose:
        bar = LoopProgress(bar_msg, len(dataset_list))

    for ii in range(len(dataset_list)):
        if vector_field == False:
            fun = data['Function/'+str(var_name)+'/'+str(dataset_list[ii])][:].flatten()
        else:
            fun = data['Function/'+str(var_name)+'/'+str(dataset_list[ii])][:][:, :gdim].flatten()

        snap.append(fun)
        param.append(dataset_list[ii])
        if verbose:
            bar.update(1)

    # Extracting keys
    params_np = []
    for mu in param:

        tmp = str(mu[0])
        if len(mu) > 1:
            for ii in range(1, len(mu)):
                if str(mu[ii]) == '_':
                    tmp += '.'
                else:
                    tmp += str(mu[ii]) 

        params_np.append(float(tmp))

    # Sorting snapshots according to the params_np
    indices = sorted(range(len(params_np)), key=lambda index: params_np[index])
    snap.sort(indices)
    params_np.sort()

    return snap, params_np


## Routine for importing from OpenFOAM using fluidfoam
class ImportOF():
    r"""
    A class used to import data from OpenFOAM and interpolate them onto *dolfinx* mesh.

    The fluidfoam library (see https://github.com/fluiddyn/fluidfoam) is exploited for the import process
        - Supported OpenFoam Versions : 2.4.0, 4.1 to 9 (org), v1712plus to v2212plus (com)
        
    Parameters
    ----------
    path : str
        Path of the OpenFOAM case directory.
    extract_dofs : boolean, optional (Default=False)
        If true, the dofs :math:`x,\,y,\,z` of the OpenFOAM mesh are imported.
        
    """
    def __init__(self, path: str, extract_dofs = False) -> None:
        
        self.path = path
        
        if extract_dofs:
            x, y, z = of.readmesh(path, verbose=False)
            self.of_dofs = np.array([x, y, z])
        else:
            self.of_dofs = None

    def import_field(self,  var_name: str, vector=False, verbose=True):
        """
        Importing all time instances (**skipping zero folder**) from OpenFOAM directory.

        Parameters
        ----------
        var_name : str
            Name of the field to import.
        vector : boolean, (Default: False)
            Labelling if the field is a vector.
        verbose: boolean, (Default = True) 
            If `True`, printing is enabled
            
        Returns
        -------
        field : list
            Imported list of functions (each element is a `numpy.ndarray`), sorted in time.
        time_instants : list
            Sorted list of time.
        """
        
        if verbose:
            print('Importing '+var_name)
        
        field = list()
        time_instants = list()
        
        for file in os.listdir(self.path):
            
            if not ((file == '0.orig') or (file == '0')):
                d = os.path.join(self.path, file)
                if os.path.isdir(d):
                        if os.path.exists(d+'/'+var_name):
                            if vector:
                                field.append( of.readvector(self.path, file, var_name, verbose=False).T )
                            else:
                                field.append( of.readscalar(self.path, file, var_name, verbose=False).reshape(-1,1) )
                            time_instants.append(float(file))
            
        # Sorting snapshots according to the params_np
        indices = sorted(range(len(time_instants)), key=lambda index: time_instants[index])
        
        tmp = field.copy()
        field.clear()

        assert len(tmp) == len(indices)
        for ii in range(len(indices)):
            field.append(tmp[indices[ii]])
        time_instants.sort()    
        
        return field, time_instants
    
    def foam_to_dolfinx(self, V: FunctionSpace, snaps: list, variables: list, 
                        cut_value = 0., verbose=None):
        """
        Converting the OpenFOAM data imported into `FunctionsList` based on *dolfinx* structures. The snapshots are projected onto a suitable functional space.

        Parameters
        ----------
        V : FunctionSpace
            Functional Space onto which data are projected.
        snaps : list
            List of `numpy.ndarray` of size `(ndofs, ndim)` with the OpenFOAM snapshots.
        variables : list
            Lisr of spatial variables, either 2D (e.g., `['x', 'y']`, `['x', 'z']`, `['y', 'z']`) or 3D (e.g., `['x', 'y', 'z']`).
        cut_value : float, (Default = 0.)
            For 2D meshes, one dimension is cut and this is the associated value.
        verbose: str, (Default = None) 
            If not `None`, printing is enabled with input message.
            
        Returns
        -------
        snap_dolfinx : FunctionsList
            Project list of functions in *dolfinx*.
            
        """
        
        
        variables.sort()
        gdim = len(variables)
        assert(gdim == 2 or gdim == 3)
        
        if self.of_dofs is None:
            x, y, z = of.readmesh(self.path, verbose=False)
            self.of_dofs = np.array([x, y, z])
        
        snap_dolfinx = FunctionsList(V)
        fun_ = Function(V).copy()
        
        if verbose is not None:
            progressBar = LoopProgress(msg = verbose, final = len(snaps))
        
        if V.num_sub_spaces > 0:
            for snap in snaps:
                interp_ = [NearestNDInterpolator(self.of_dofs.T, snap[:, ii].flatten()) for ii in range(3)]
                # interp_ = [LinearNDInterpolator(self.of_dofs.T, snap[:, ii].flatten()) for ii in range(3)]
                    
                if gdim == 2:
                    
                    if ((variables[0]=='x') and (variables[1]=='y')):
                        fun_.sub(0).interpolate(lambda xx: interp_[0](xx[0], xx[1], cut_value))
                        fun_.sub(1).interpolate(lambda xx: interp_[1](xx[0], xx[1], cut_value))
                        
                    elif ((variables[0]=='x') and (variables[1]=='z')):
                        fun_.sub(0).interpolate(lambda xx: interp_[0](xx[0], cut_value, xx[1]))
                        fun_.sub(1).interpolate(lambda xx: interp_[2](xx[0], cut_value, xx[1]))
                        
                    elif ((variables[0]=='y') and (variables[1]=='z')):
                        fun_.sub(0).interpolate(lambda xx: interp_[1](cut_value, xx[0], xx[1]))
                        fun_.sub(1).interpolate(lambda xx: interp_[2](cut_value, xx[0], xx[1]))
                        
                elif gdim == 3:
                    for ii in range(gdim):
                        fun_.sub(ii).interpolate(lambda xx: interp_[ii](xx[0], xx[1], xx[2]))
                        
                snap_dolfinx.append(fun_.copy())
                if verbose is not None:
                    progressBar.update(1, percentage=False)
                
        else:
            for snap in snaps:
                interp = NearestNDInterpolator(self.of_dofs.T, snap.flatten())
                # interp = LinearNDInterpolator(self.of_dofs.T, snap.flatten())
                    
                if gdim == 2:
                    
                    if ((variables[0]=='x') and (variables[1]=='y')):
                        fun_.interpolate(lambda xx: interp(xx[0], xx[1], cut_value))
                        
                    elif ((variables[0]=='x') and (variables[1]=='z')):
                        fun_.interpolate(lambda xx: interp(xx[0], cut_value, xx[1]))
                        
                    elif ((variables[0]=='y') and (variables[1]=='z')):
                        fun_.interpolate(lambda xx: interp(cut_value, xx[0], xx[1]))
                        
                elif gdim == 3:
                    fun_.interpolate(lambda xx: interp(xx[0], xx[1], xx[2]))
                    
                snap_dolfinx.append(fun_.copy())
                if verbose is not None:
                    progressBar.update(1, percentage=False)
            
        return snap_dolfinx

# ## Routine for importing from OpenFOAM using pyvista and fluidfoam --- to complete
# class ImportOF():
#     r"""
#     A class used to import data from OpenFOAM and interpolate them onto *dolfinx* mesh.

#     The fluidfoam library (see https://github.com/fluiddyn/fluidfoam) is exploited for the import process
#         - Supported OpenFoam Versions : 2.4.0, 4.1 to 9 (org), v1712plus to v2212plus (com)
        
#     Parameters
#     ----------
#     path : str
#         Path of the OpenFOAM case directory.
#     extract_dofs : boolean, optional (Default=False)
#         If true, the dofs :math:`x,\,y,\,z` of the OpenFOAM mesh are imported.
        
#     """
#     def __init__(self, path: str, mode: str = 'pyvista', extract_dofs: bool = False) -> None:
        
#         self.path = path
        
#         if mode != 'fluidfoam' and mode != 'pyvista':
            
#             warnings.warn('The mode selected is not supported. The default mode is set to `pyvista`.')
#             mode = 'pyvista'
#         self.mode = mode
            
#         if mode == 'fluidfoam':
#             if extract_dofs:
#                 x, y, z = of.readmesh(path, verbose=False)
#                 self.of_dofs = np.array([x, y, z])
#             else:
#                 self.of_dofs = None
#         else:
#             if not os.path.exists(path+'/foam.foam'):
#                 with open(path+'/foam.foam', 'w') as file:
#                     file.write('')
#             self.reader = pv.POpenFOAMReader(path+'/foam.foam')

#     def import_field(self, var_name: str, vector: bool =False, verbose: bool =True):
#         """
#         Importing all time instances (**skipping zero folder**) from OpenFOAM directory.

#         Parameters
#         ----------
#         var_name : str
#             Name of the field to import.
#         vector : boolean, (Default: False)
#             Labelling if the field is a vector.
#         verbose: boolean, (Default = True) 
#             If `True`, printing is enabled
            
#         Returns
#         -------
#         field : list
#             Imported list of functions (each element is a `numpy.ndarray`), sorted in time.
#         time_instants : list
#             Sorted list of time.
#         """
        
#         field = list()
#         time_instants = list()
        
#         if self.mode == 'fluidfoam':
#             if verbose:
#                 print('Importing '+var_name+' using fluidfoam')
            
#             for file in os.listdir(self.path):
                
#                 if not ((file == '0.orig') or (file == '0')):
#                     d = os.path.join(self.path, file)
#                     if os.path.isdir(d):
#                             if os.path.exists(d+'/'+var_name):
#                                 if vector:
#                                     field.append( of.readvector(self.path, file, var_name, verbose=False).T )
#                                 else:
#                                     field.append( of.readscalar(self.path, file, var_name, verbose=False).reshape(-1,1) )
#                                 time_instants.append(float(file))
                
#             # Sorting snapshots according to the params_np
#             indices = sorted(range(len(time_instants)), key=lambda index: time_instants[index])
            
#             tmp = field.copy()
#             field.clear()

#             assert len(tmp) == len(indices)
#             for ii in range(len(indices)):
#                 field.append(tmp[indices[ii]])
#             time_instants.sort()    
            
#         else: 
#             print('Importing '+var_name+' using pyvista')
#             for idx_t in range(1, len(self.reader.time_values)):
#                 self.reader.set_active_time_value(self.reader.time_values[idx_t])  
                
#                 field.append(self.reader.read()['internalMesh'].point_data[var_name])
#                 time_instants.append(self.reader.time_values[idx_t])
                
#         return field, time_instants
    
#     def create_mesh(self):
        
#             # Creating the mesh from the OpenFOAM points - works only with blockMeshDict (hexahedrons only)!
            
#             ## Nodes of the mesh - (Nh, gdim)
#             self.nodes = self.reader.read()['internalMesh'].points
            
#             ## Connectivity of the mesh (topology) - The second dimension indicates the type of cell used
#             args_conn = np.argsort(self.reader.read()['internalMesh'].cells_dict[12], axis=1)
#             connectivity = np.array([self.reader.read()['internalMesh'].cells_dict[12][i, arg] for i, arg in enumerate(args_conn)])

#             # # Use advanced indexing to reorder the connectivity in one step
#             # rows = np.arange(self.reader.read()['internalMesh'].cells_dict[10].shape[0])[:, None]
#             # connectivity = self.reader.read()['internalMesh'].cells_dict[10][rows, args_conn]

#             ## Define mesh element
#             shape = "hexahedron"
#             # shape = 'tetrahedron'
#             degree = 1
#             cell = ufl.Cell(shape, geometric_dimension=3)
#             mesh_ufl = ufl.Mesh(ufl.VectorElement("Lagrange", cell, degree))

#             cells = np.array(connectivity, dtype=np.int32)
            
#             return create_mesh(MPI.COMM_WORLD, cells, self.nodes, mesh_ufl)
    
#     def foam_to_dolfinx(self, V: FunctionSpace, snaps: list, variables: list = ['x', 'y', 'z'], 
#                         cut_value: float  = 0., verbose: bool = None):
#         """
#         Converting the OpenFOAM data imported into `FunctionsList` based on *dolfinx* structures. The snapshots are projected onto a suitable functional space.

#         Parameters
#         ----------
#         V : FunctionSpace
#             Functional Space onto which data are projected.
#         snaps : list
#             List of `numpy.ndarray` of size `(ndofs, ndim)` with the OpenFOAM snapshots.
#         variables : list, (Default = `['x', 'y', 'z']`)
#             List of spatial variables, either 2D (e.g., `['x', 'y']`, `['x', 'z']`, `['y', 'z']`) or 3D (e.g., `['x', 'y', 'z']`).
#         cut_value : float, (Default = 0.)
#             For 2D meshes, one dimension is cut and this is the associated value.
#         verbose: str, (Default = None) 
#             If not `None`, printing is enabled with input message.
            
#         Returns
#         -------
#         snap_dolfinx : FunctionsList
#             Project list of functions in *dolfinx*.
            
#         """
        
#         snap_dolfinx = FunctionsList(V)
#         if verbose is not None:
#             progressBar = LoopProgress(msg = verbose, final = len(snaps))
            
#         if self.mode == 'fluidfoam':
            
#             if self.of_dofs is None:
#                 x, y, z = of.readmesh(self.path, verbose=False)
#                 self.of_dofs = np.array([x, y, z])
            
#             variables.sort()
#             gdim = len(variables)
#             assert(gdim == 2 or gdim == 3)
#             fun_ = Function(V).copy()
            
#             if V.num_sub_spaces > 0:
#                 for snap in snaps:
#                     interp_ = [NearestNDInterpolator(self.of_dofs.T, snap[:, ii].flatten()) for ii in range(3)]
#                     # interp_ = [LinearNDInterpolator(self.of_dofs.T, snap[:, ii].flatten()) for ii in range(3)]
                        
#                     if gdim == 2:
                        
#                         if ((variables[0]=='x') and (variables[1]=='y')):
#                             fun_.sub(0).interpolate(lambda xx: interp_[0](xx[0], xx[1], cut_value))
#                             fun_.sub(1).interpolate(lambda xx: interp_[1](xx[0], xx[1], cut_value))
                            
#                         elif ((variables[0]=='x') and (variables[1]=='z')):
#                             fun_.sub(0).interpolate(lambda xx: interp_[0](xx[0], cut_value, xx[1]))
#                             fun_.sub(1).interpolate(lambda xx: interp_[2](xx[0], cut_value, xx[1]))
                            
#                         elif ((variables[0]=='y') and (variables[1]=='z')):
#                             fun_.sub(0).interpolate(lambda xx: interp_[1](cut_value, xx[0], xx[1]))
#                             fun_.sub(1).interpolate(lambda xx: interp_[2](cut_value, xx[0], xx[1]))
                            
#                     elif gdim == 3:
#                         for ii in range(gdim):
#                             fun_.sub(ii).interpolate(lambda xx: interp_[ii](xx[0], xx[1], xx[2]))
                            
#                     snap_dolfinx.append(fun_.copy())
#                     if verbose is not None:
#                         progressBar.update(1, percentage=False)
                    
#             else:
#                 for snap in snaps:
#                     interp = NearestNDInterpolator(self.of_dofs.T, snap.flatten())
#                     # interp = LinearNDInterpolator(self.of_dofs.T, snap.flatten())
                        
#                     if gdim == 2:
                        
#                         if ((variables[0]=='x') and (variables[1]=='y')):
#                             fun_.interpolate(lambda xx: interp(xx[0], xx[1], cut_value))
                            
#                         elif ((variables[0]=='x') and (variables[1]=='z')):
#                             fun_.interpolate(lambda xx: interp(xx[0], cut_value, xx[1]))
                            
#                         elif ((variables[0]=='y') and (variables[1]=='z')):
#                             fun_.interpolate(lambda xx: interp(cut_value, xx[0], xx[1]))
                            
#                     elif gdim == 3:
#                         fun_.interpolate(lambda xx: interp(xx[0], xx[1], xx[2]))
                        
#                     snap_dolfinx.append(fun_.copy())
#                     if verbose is not None:
#                         progressBar.update(1, percentage=False)
                
#         else:
            
#             x_V = V.tabulate_dof_coordinates().T
#             for snap in snaps: 
                
#                 if V.num_sub_spaces > 0:
#                     interp_ = [NearestNDInterpolator(self.nodes, snap[:, ii].flatten()) for ii in range(3)]
#                     u_array = np.array([interp_[ii](*x_V) for ii in range(3)]).T.flatten()

#                 else:
#                     interp_ = NearestNDInterpolator(self.nodes, snap)
#                     u_array = interp_(*x_V)
                    
#                 snap_dolfinx.append(u_array)
                
#                 if verbose is not None:
#                     progressBar.update(1, percentage=False)
                
#         return snap_dolfinx