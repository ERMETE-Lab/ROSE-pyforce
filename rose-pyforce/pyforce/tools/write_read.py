# I/O tools
# Authors: Stefano Riva, Yantao Luo, NRG, Politecnico di Milano
# Latest Code Update: 20 January 2026
# Latest Doc  Update: 20 January 2026

import warnings
from .functions_list import FunctionsList
from .backends import LoopProgress

import h5py

import os
import glob
import numpy as np
import fluidfoam as of
import pyvista as pv

## Routine for importing from OpenFOAM using pyvista and fluidfoam - single region
class ReadFromOF():
    r"""
    A class used to import data from OpenFOAM - either decomposed or not, single region.

    Either the fluidfoam library (see https://github.com/fluiddyn/fluidfoam) or pyvista are exploited for the import process
        - Supported OpenFoam Versions : 2.4.0, 4.1 to 9 (org), v1712plus to v2212plus (com) - to check
        
    Parameters
    ----------
    path : str
        Path of the OpenFOAM case directory.
    skip_zero_time : bool, optional
        If `True`, the zero time folder is skipped. Default is `False`.
    decomposed_case : bool, optional
        If `True`, the case is considered as decomposed. Default is `False`.
        
    """
    def __init__(self, path: str,
                 skip_zero_time: bool = False,
                 decomposed_case: bool = False
                 ) -> None:

        self.path = path

        # Check if any file with .foam extension exists in the directory
        foam_files = glob.glob(os.path.join(path, '*.foam'))
        
        if not foam_files:
            # If no .foam file exists, create foam.foam
            foam_file_path = os.path.join(path, 'foam.foam')
            with open(foam_file_path, 'w') as file:
                file.write('')
        else:
            # Use the first found .foam file
            foam_file_path = foam_files[0]
        
        self.reader = pv.POpenFOAMReader(foam_file_path)
        self.reader.skip_zero_time = skip_zero_time

        # Set case type - decomposed or reconstructed
        if decomposed_case:
            self.reader.reader.SetCaseType(0)  # Decomposed case
            self.reader.reader.UpdateInformation() # Update information after changing case type
            print('Case Type '+ self.reader.case_type)
            self.decomposed_case = True
        else:
            self.decomposed_case = False

    def mesh(self):
        """
        Returns the mesh of the OpenFOAM case.
        
        Returns
        -------
        mesh : pyvista.UnstructuredGrid
            The mesh of the OpenFOAM case.
        """
        grid = self.reader.read()['internalMesh']
        grid.clear_data() # remove all the data
        return grid

    def save_mesh(self, filename: str):
        """
        Saves the mesh of the OpenFOAM case to a vtk-file.
        
        Parameters
        ----------
        filename : str
            Name of the file to save the mesh.
        """
        mesh = self.mesh()
        mesh.save(filename + '.vtk')


    # Deprecated version
    # def import_field(self, var_name: str, 
    #                  use_fluidfoam: bool = False, 
    #                  extract_cell_data: bool = True,
    #                  verbose: bool = True):
    #     r"""
    #     Importing all time instances (**skipping zero folder**) from OpenFOAM directory.

    #     Parameters
    #     ----------
    #     var_name : str
    #         Name of the field to import.
    #     use_fluidfoam : boolean, (Default: False)
    #         If `True`, the fluidfoam library is used for reading the data (only cell data supported); otherwise, pyvista is used.
    #     vector : boolean, (Default: False)
    #         Labelling if the field is a vector (needed only is `use_fluidfoam==True`).
    #     extract_cell_data : boolean, (Default: True)
    #         If `True`, the cell data from centroids is extracted; otherwise, point data is extracted.
    #     verbose: boolean, (Default = True) 
    #         If `True`, printing is enabled
            
    #     Returns
    #     -------
    #     field : list
    #         Imported list of functions (each element is a `numpy.ndarray`), sorted in time.
    #     time_instants : list
    #         Sorted list of time.
    #     """
        
    #     field = list()
    #     time_instants = list()
        
    #     if use_fluidfoam:
    #         assert self.decomposed_case==False, "Fluidfoam reader does not support decomposed cases."

    #     if use_fluidfoam and extract_cell_data:

    #         file_list = sorted(os.listdir(self.path))
            
    #         for jj, file in enumerate(file_list):

    #             if verbose:
    #                 print('Importing '+var_name+f' using fluidfoam - {(jj+1)/len(file_list)*100:.2f}%', end="\r")
            
    #             if not ((file == '0.orig') or (file == '0') or (file == '0.ss')):
    #                 d = os.path.join(self.path, file)
    #                 if os.path.isdir(d):
    #                         try: # scalar field
    #                             field.append( of.readscalar(self.path, file, var_name, verbose=False).reshape(-1,1) )
    #                         except ValueError:
    #                             try: # vector field
    #                                 field.append(of.readvector(self.path, file, var_name, verbose=False).T)
    #                             except ValueError: # tensor field
    #                                 try: # tensor field
    #                                     field.append(of.readtensor(self.path, file, var_name, verbose=False).T)
    #                                 except ValueError: # symmetric tensor field
    #                                     field.append(of.readsymmtensor(self.path, file, var_name, verbose=False).T)
    #                         time_instants.append(float(file))
            
    #     else: 
    #         for idx_t in range(len(self.reader.time_values)):
    #             if verbose:
    #                 print('Importing '+var_name+f' using pyvista - {(idx_t+1)/len(self.reader.time_values)*100:.2f}%', end="\r")

    #             self.reader.set_active_time_value(self.reader.time_values[idx_t])

    #             # Extract data
    #             if extract_cell_data: # centroids data
    #                 field.append(self.reader.read()['internalMesh'].cell_data[var_name])
    #             else: # vertices data
    #                 field.append(self.reader.read()['internalMesh'].point_data[var_name])

    #             # Append time instant
    #             time_instants.append(self.reader.time_values[idx_t])
                
    #     # Convert list to FunctionsList
    #     snaps = FunctionsList(dofs=field[0].flatten().shape[0])
    #     for f in field:
    #         snaps.append(f.flatten())

    #     return snaps, time_instants
    
    def import_field(self, var_name: str, 
                     use_fluidfoam: bool = False, 
                     extract_cell_data: bool = True,
                     verbose: bool = True):
        r"""
        Importing all time instances (**skipping zero folder**) from OpenFOAM directory.

        Parameters
        ----------
        var_name : str
            Name of the field to import.
        use_fluidfoam : boolean, (Default: False)
            If `True`, the fluidfoam library is used for reading the data (only cell data supported); otherwise, pyvista is used.
        vector : boolean, (Default: False)
            Labelling if the field is a vector (needed only is `use_fluidfoam==True`).
        extract_cell_data : boolean, (Default: True)
            If `True`, the cell data from centroids is extracted; otherwise, point data is extracted.
        verbose: boolean, (Default = True) 
            If `True`, printing is enabled
            
        Returns
        -------
        field : FunctionsList
            Imported list of functions (each element is a `numpy.ndarray`), sorted in time.
        time_instants : np.ndarray
            Sorted list of time.
        """
        
        if use_fluidfoam:
            assert self.decomposed_case==False, "Fluidfoam reader does not support decomposed cases."

        if use_fluidfoam and extract_cell_data:
            field, time_instants = self._import_with_fluidfoam(self.path, var_name, verbose)
        else:
            field, time_instants = self._import_with_pyvista(var_name, extract_cell_data, verbose)

        # Convert list to FunctionsList
        snaps = FunctionsList(dofs=field[0].flatten().shape[0])
        for f in field:
            snaps.append(f.flatten())

        return snaps, time_instants
    
    def _import_with_fluidfoam(self, var_name: str, verbose: bool = True, file_list: list[str] = None):
        """
        Importing time instances from OpenFOAM directory using fluidfoam.
        
        Parameters
        ----------
        var_name : str
            Name of the field to import.
        verbose: boolean, (Default = True) 
            If `True`, printing is enabled
        file_list : list[str], optional
            List of folders to read. If `None`, all folders in the case directory are read.

        Returns
        -------
        field : list
            Imported list of functions (each element is a `numpy.ndarray`), sorted in time.
        time_instants : np.ndarray
            Sorted list of time.
        """

        field = list()
        time_instants = list()

        if file_list is None:
            file_list = sorted(os.listdir(self.path))
        if verbose:
            bar = LoopProgress(msg=f'Importing {var_name} using fluidfoam', final = len(file_list))
        
        for file in file_list:
        
            if not ((file == '0.orig') or (file == '0') or (file == '0.ss')):
                d = os.path.join(self.path, file)
                if os.path.isdir(d):
                        try: # scalar field
                            field.append( of.readscalar(self.path, file, var_name, verbose=False).reshape(-1,1) )
                        except ValueError:
                            try: # vector field
                                field.append(of.readvector(self.path, file, var_name, verbose=False).T)
                            except ValueError: # tensor field
                                try: # tensor field
                                    field.append(of.readtensor(self.path, file, var_name, verbose=False).T)
                                except ValueError: # symmetric tensor field
                                    field.append(of.readsymmtensor(self.path, file, var_name, verbose=False).T)
                        time_instants.append(float(file))

            if verbose:
                bar.update(1)

        return field, np.asarray(time_instants)
    
    def _import_with_pyvista(self, var_name: str, extract_cell_data: bool = True, verbose: bool = True, time_instants: list[float] = None):
        """
        Importing time instances from OpenFOAM directory using pyvista.
        
        Parameters  
        ----------
        var_name : str
            Name of the field to import.
        extract_cell_data : boolean, (Default: True)
            If `True`, the cell data from centroids is extracted; otherwise, point data is extracted.
        verbose: boolean, (Default = True) 
            If `True`, printing is enabled
        time_instants : list[float], optional
            List of time instants to read. If `None`, all time instants are read.
        
        Returns
        -------
        field : list
            Imported list of functions (each element is a `numpy.ndarray`), sorted in time.
        time_instants : np.ndarray
            Sorted list of time.
        """

        if time_instants is None:
            time_instants = self.reader.time_values
            
        field = list()

        if verbose:
            bar = LoopProgress(msg=f'Importing {var_name} using pyvista', final = len(time_instants))


        for idx_t, t in enumerate(time_instants):
            if verbose:
                bar.update(1)

            # Set active time
            self.reader.set_active_time_value(t)
            grid = self.reader.read()
            mesh = grid['internalMesh']

            # Extract data
            if extract_cell_data: # centroids data
                available = mesh.cell_data.keys()
                if var_name not in available:
                    raise KeyError(f"Field '{var_name}' not found at time {t}. Available fields: {list(available)}")
                    
                field.append(mesh.cell_data[var_name])
            else: # vertices data
                available = mesh.point_data.keys()
                if var_name not in available:
                    raise KeyError(f"Field '{var_name}' not found at time {t}. Available fields: {list(available)}")
                field.append(mesh.point_data[var_name])

            # Extract data
            # if extract_cell_data: # centroids data
            #     field.append(self.reader.read()['internalMesh'].cell_data[var_name])
            # else: # vertices data
            #     field.append(self.reader.read()['internalMesh'].point_data[var_name])

        return field, np.asarray(time_instants)
    
def ImportFunctionsList(filename: str, format: str = 'h5', return_var_name: bool = False):
    """
    This function can be used to load from `xdmf/h5` files scalar or vector list of functions.

    Parameters
    ----------
    filename : str
        Name of the file to read as xdmf/h5 file.
    format : str, optional (Default = 'h5')
        Format of the file to read. It can be either 'h5', 'npy', or 'npz'.
    return_var_name : bool, optional (Default = False)
        If `True`, the variable name is returned along with the FunctionsList.
    
    Returns
    -------
    snap : FunctionsList
        Imported list of functions.
    """
    
    fmt = format.lower()
    
    if fmt == 'h5':
        with h5py.File(filename + '.h5', 'r') as f:
            var_name = list(f.keys())[0]
            data = f[var_name][:]
    elif fmt == 'npz':
        data = np.load(filename + '.npz')
        var_name = list(data.keys())[0]
        data = data[var_name]
    else:
        raise ValueError(f"Unsupported format {fmt}. Use 'h5' or 'npz'.")
    
    # Create FunctionsList from the data
    snap = FunctionsList(dofs=data.shape[0])
    snap.build_from_matrix(data)

    if return_var_name:
        return snap, var_name
    return snap

def convert_cell_data_to_point_data(grid: pv.UnstructuredGrid, snaps: FunctionsList):
    """
    Convert cell data to point data for a given pyvista UnstructuredGrid and corresponding FunctionsList.

    Parameters
    ----------
    grid : pyvista.UnstructuredGrid
        The input mesh with cell data.
    snaps : FunctionsList
        The FunctionsList containing cell data.

    Returns
    -------
    new_snaps : FunctionsList
        The FunctionsList containing point data.
    """
    # Create a copy of the grid to avoid modifying the original
    grid_copy = grid.copy()
    grid_copy.clear_data()  # Clear existing data

    _snap_Nh = snaps.shape()[0]
    if _snap_Nh != grid.n_cells:
        vec_dim = int(_snap_Nh / grid.n_cells)
        if _snap_Nh != grid.n_cells * vec_dim:
            raise ValueError(f"Number of cell data points ({_snap_Nh}) does not match number of cells in the grid ({grid.n_cells}).")
    else:
        vec_dim = 1
        

    new_snaps = FunctionsList(dofs=grid.n_points * vec_dim)

    for _snap in snaps:
        # Reshape the snapshot to match cell data dimensions
        if vec_dim > 1:
            cell_data = _snap.reshape((grid.n_cells, vec_dim))
        else:
            cell_data = _snap

        # Assign cell data to the grid copy
        grid_copy.cell_data['u'] = cell_data

        # Convert cell data to point data
        point_data = grid_copy.cell_data_to_point_data()['u']

        # Append the point data to the new FunctionsList
        new_snaps.append(point_data.flatten())

    return new_snaps

def convert_point_data_to_cell_data(grid: pv.UnstructuredGrid, snaps: FunctionsList):
    """
    Convert point data to cell data for a given pyvista UnstructuredGrid and corresponding FunctionsList.

    Parameters
    ----------
    grid : pyvista.UnstructuredGrid
        The input mesh with point data.
    snaps : FunctionsList
        The FunctionsList containing point data.

    Returns
    -------
    new_snaps : FunctionsList
        The FunctionsList containing cell data.
    """
    # Create a copy of the grid to avoid modifying the original
    grid_copy = grid.copy()
    grid_copy.clear_data()  # Clear existing data

    _snap_Nh = snaps.shape()[0]
    if _snap_Nh != grid.n_points:
        vec_dim = int(_snap_Nh / grid.n_points)
        if _snap_Nh != grid.n_points * vec_dim:
            raise ValueError(f"Number of point data points ({_snap_Nh}) does not match number of points in the grid ({grid.n_points}).")
    else:
        vec_dim = 1
        

    new_snaps = FunctionsList(dofs=grid.n_cells * vec_dim)

    for _snap in snaps:
        # Reshape the snapshot to match point data dimensions
        if vec_dim > 1:
            point_data = _snap.reshape((grid.n_points, vec_dim))
        else:
            point_data = _snap

        # Assign point data to the grid copy
        grid_copy.point_data['u'] = point_data

        # Convert point data to cell data
        cell_data = grid_copy.point_data_to_cell_data()['u']

        # Append the cell data to the new FunctionsList
        new_snaps.append(cell_data.flatten())

    return new_snaps

## Routine for importing from OpenFOAM using pyvista - multi region
class ReadFromOFMultiRegion():
    r"""
    A class used to import data from OpenFOAM - either decomposed or not, multi region.

    The import process is based on pyvista.
        
    Parameters
    ----------
    path : str
        Path of the OpenFOAM case directory.
    skip_zero_time : bool, optional
        If `True`, the zero time folder is skipped. Default is `False`.
    decomposed_case : bool, optional
        If `True`, the case is considered as decomposed. Default is `False`.
        
    """
    def __init__(self, path: str,
                 skip_zero_time: bool = False,
                 decomposed_case: bool = False
                 ) -> None:

        self.path = path

        # Check if any file with .foam extension exists in the directory
        foam_files = glob.glob(os.path.join(path, '*.foam'))
        
        if not foam_files:
            # If no .foam file exists, create foam.foam
            foam_file_path = os.path.join(path, 'foam.foam')
            with open(foam_file_path, 'w') as file:
                file.write('')
        else:
            # Use the first found .foam file
            foam_file_path = foam_files[0]
        
        self.reader = pv.POpenFOAMReader(foam_file_path)
        self.reader.skip_zero_time = skip_zero_time

        # Set case type - decomposed or reconstructed
        if decomposed_case:
            self.reader.reader.SetCaseType(0)  # Decomposed case
            self.reader.reader.UpdateInformation() # Update information after changing case type
            print('Case Type '+ self.reader.case_type)
            self.decomposed_case = True
        else:
            self.decomposed_case = False

        # Read available regions
        self.regions = list(self.reader.read().keys())
        
        # Remove defaultRegion if present
        if 'defaultRegion' in self.regions:
            self.regions.remove('defaultRegion')

    def _region_mesh(self, region: str):
        """
        Returns the mesh of the specified region of the OpenFOAM case.
        
        Parameters
        ----------
        region : str
            Name of the region to extract the mesh from.
        
        Returns
        -------
        mesh : pyvista.UnstructuredGrid
            The mesh of the specified region of the OpenFOAM case.
        """
        grid = self.reader.read()[region]['internalMesh']
        grid.clear_data() # remove all the data
        return grid

    def mesh(self, regions: list[str] = None):
        """
        Returns the combines mesh of regions of the OpenFOAM case.

        Parameters
        ----------
        regions : str, optional
            If specified, only the mesh of the given region is returned. If `None`, the combined mesh of all regions is returned.

        Returns
        -------
        mesh : pyvista.UnstructuredGrid
            The mesh of the OpenFOAM case.

        """

        if regions is not None:
            _regions = regions
        else:
            _regions = self.regions

        blocks_to_merge = []
        for region in _regions:
            _mesh = self._region_mesh(region)
            _mesh.clear_data() # remove all the data
            blocks_to_merge.append(_mesh)

        combined_mesh = pv.MultiBlock(blocks_to_merge).combine(merge_points=True)
        
        # Equivalent deprecated code:
        # combined_mesh = blocks_to_merge[0]
        # for block in blocks_to_merge[1:]:
        #     combined_mesh += block

        return combined_mesh
    
    def save_mesh(self, filename: str, region: str = None):
        """
        Saves the mesh of the OpenFOAM case to a vtk-file.
        
        Parameters
        ----------
        filename : str
            Name of the file to save the mesh.
        region : str, optional
            If specified, only the mesh of the given region is saved. If `None`, the combined mesh of all regions is saved.
        """
        if region is None:
            mesh = self.mesh()
        else:
            mesh = self._region_mesh(region)
        
        mesh.save(filename + '.vtk')

    def _get_valid_regions_for_field(self, var_name: str):
        r"""
        Get the list of regions where the specified field is available.
        
        Parameters
        ----------
        var_name : str
            Name of the field to check.
        
        Returns
        -------
        valid_regions : list
            List of regions where the specified field is available.
        """
        t0 = self.reader.time_values[0]
        self.reader.set_active_time_value(t0)

        valid_regions = []
        for region in self.regions:
            grid = self.reader.read()[region]['internalMesh']

            # Check if the field is available in either cell data or point data
            if (var_name in grid.cell_data.keys()) or (var_name in grid.point_data.keys()):
                valid_regions.append(region)

        return valid_regions

    def import_field(self, var_name: str,
                     time_instants: list[float] = None,
                     verbose: bool = True):
        r"""
        Importing all time instances (**skipping zero folder**) from OpenFOAM directory for all available regions, if not skip.

        *Only cell data extraction is supported for multi-region cases: to prevent data misalignment issues.*

        Parameters
        ----------
        var_name : str
            Name of the field to import.
        verbose: boolean, (Default = True) 
            If `True`, printing is enabled


        Returns
        -------
        snaps : FunctionsList
            Imported list of functions (each element is a `numpy.ndarray`), sorted in time.
        time_instants : np.ndarray
            Sorted list of time.
        """

        if time_instants is None:
            time_instants = self.reader.time_values

        # Concatenate all regional snapshots
        regional_snaps = []

        if verbose:
            bar = LoopProgress(msg=f'Importing {var_name} from all regions', final = len(self.regions))

        for region in self.regions:
            if verbose:
                bar.update(1)

            try:
                snaps_region, _ = self._import_region_field(region, var_name,
                                                    extract_cell_data=True,
                                                    time_instants=time_instants,
                                                    verbose=False)
            except KeyError:
                if verbose:
                    warnings.warn(f"\nSkipping region '{region}': field '{var_name}' not found.")
                    continue 
        
            regional_snaps.append(snaps_region)

        # Concatenate all regional snapshots
        concatenated_snaps = np.concatenate([snaps.return_matrix() for snaps in regional_snaps], axis=0)
        snaps = FunctionsList(snap_matrix=concatenated_snaps)
        return snaps, np.asarray(time_instants)

    def _import_region_field(self, region: str, var_name: str,
                            extract_cell_data: bool = True, time_instants: list[float] = None,
                            verbose: bool = True):
        r"""
        Importing all time instances (**skipping zero folder**) from OpenFOAM directory for a specific region.

        Parameters
        ----------
        region : str
            Name of the region to import the field from.
        var_name : str
            Name of the field to import.
        extract_cell_data : boolean, (Default: True)
            If `True`, the cell data from centroids is extracted; otherwise, point data is extracted.
        time_instants : list[float], optional
            List of time instants to read. If `None`, all time instants are read.
        verbose: boolean, (Default = True) 
            If `True`, printing is enabled  

        Returns
        -------
        field : FunctionsList
            Imported list of functions (each element is a `numpy.ndarray`), sorted in time.
        time_instants : np.ndarray
            Sorted list of time.
        """

        if time_instants is None:
            time_instants = self.reader.time_values

        field = list()

        if verbose:
            bar = LoopProgress(msg=f'Importing {var_name} from region {region} using pyvista', final = len(time_instants))

        for idx_t, t in enumerate(time_instants):
            if verbose:
                bar.update(1)

            # Set active time
            self.reader.set_active_time_value(t)
            grid = self.reader.read()[region]
            mesh = grid['internalMesh']

            # Extract data
            if extract_cell_data: # centroids data
                available = mesh.cell_data.keys()
                if var_name not in available:
                    raise KeyError(f"Field '{var_name}' not found at time {t} in region '{region}'. Available fields: {list(available)}")
                    
                field.append(mesh.cell_data[var_name])
            else: # vertices data
                available = mesh.point_data.keys()
                if var_name not in available:
                    raise KeyError(f"Field '{var_name}' not found at time {t} in region '{region}'. Available fields: {list(available)}")
                field.append(mesh.point_data[var_name])

        # Convert list to FunctionsList
        Nh = field[0].flatten().shape[0]
        snaps = FunctionsList(dofs=Nh)
        for f in field:
            snaps.append(f.flatten())

        return snaps, np.asarray(time_instants)        