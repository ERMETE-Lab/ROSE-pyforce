# I/O tools
# Authors: Stefano Riva, Yantao Luo, NRG, Politecnico di Milano
# Latest Code Update: 11 March 2026
# Latest Doc  Update: 11 March 2026

import warnings

from .functions_list import FunctionsList
from .backends import LoopProgress

import h5py

import os
import glob
import numpy as np
import fluidfoam as of
import pyvista as pv
import foamlib as fl

## Routine for importing from OpenFOAM using pyvista or fluidfoam or foamlib - single region
class ReadFromOF():
    r"""
    A class used to import data from OpenFOAM - either decomposed or not, single region.
    **Note**: "Decomposed" here refers exclusively to parallel MPI domain decomposition (i.e., data split into `processor*` directories), not multi-region setups.

    Either the fluidfoam library (see https://github.com/fluiddyn/fluidfoam) or the foamlib (see https://github.com/gerlero/foamlib) or pyvista are exploited for the import process
    
    Both org and com version of OpenFOAM are supported.
        
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
        Returns the mesh of the OpenFOAM case using pyvista capabilities.
        
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
    
    def import_field(self, var_name: str, 
                     import_mode: str = 'pyvista', 
                     extract_cell_data: bool = True,
                     verbose: bool = True,
                     target_times: list[str] = None,
                     ) -> tuple[FunctionsList, np.ndarray]:
        r"""
        Importing all time instances (**skipping zero folder**) from OpenFOAM directory.

        Parameters
        ----------
        var_name : str
            Name of the field to import.
        import_mode : str, (Default: 'pyvista')
            The mode of import, either 'pyvista' or 'fluidfoam' or 'foamlib'.
        extract_cell_data : boolean, (Default: True)
            If `True`, the cell data from centroids is extracted; otherwise, point data is extracted.
        verbose: boolean, (Default = True) 
            If `True`, printing is enabled
        target_times: list[str], optional
            List of time folders to import. If `None`, all time instances are imported.

        Returns
        -------
        field : FunctionsList
            Imported list of functions (each element is a `numpy.ndarray`), sorted in time.
        time_instants : np.ndarray
            Sorted list of time.
        """

        if import_mode == 'fluidfoam' and extract_cell_data:
            field, time_instants = self._import_with_fluidfoam(var_name, verbose, target_times=target_times)
        elif import_mode == 'foamlib' and extract_cell_data:
            field, time_instants = self._import_with_foamlib(var_name, verbose, target_times=target_times)
        elif import_mode == 'pyvista':
            field, time_instants = self._import_with_pyvista(var_name, extract_cell_data, verbose, target_times=target_times)
        else:
            raise ValueError(f"Unsupported import method '{import_mode}'. Use 'pyvista' or 'fluidfoam' or 'foamlib'.")

        # Convert list to FunctionsList
        snaps = FunctionsList(dofs=field[0].flatten().shape[0])
        for f in field:
            snaps.append(f.flatten())

        return snaps, time_instants
    
    def _get_time_directories(self):
        """"
        Get the list of time directories in the OpenFOAM case, sorted in time, and skipping zero time if needed.

        Returns
        -------
        file_list : list[str]
            List of time directories in the OpenFOAM case, sorted in time, and skipping zero time if needed.
        """

        file_list = [
            f for f in os.listdir(self.path) 
            if os.path.isdir(os.path.join(self.path, f)) and f.replace('.', '', 1).isdigit()
        ]

        # Sort numerically
        file_list.sort(key=float)

        # Skip zero time folder if needed
        if self.reader.skip_zero_time and file_list[0] in ['0']:
            file_list = file_list[1:]

        return file_list
        
    def _import_with_foamlib(self, var_name: str, verbose: bool = True, target_times: list[str] = None):
        """
        Importing time instances from OpenFOAM directory using foamlib.
        
        Parameters
        ----------
        var_name : str
            Name of the field to import.
        verbose: boolean, (Default = True) 
            If `True`, printing is enabled
        target_times : list[str], optional
            List of time folders to read. If `None`, all time instants are read.

        Returns
        -------
        field : list
            Imported list of functions (each element is a `numpy.ndarray`), sorted in time.
        time_instants : np.ndarray
            Sorted list of time.
        """

        field = list()
        time_instants = list()

        if target_times is None:
            target_times = self._get_time_directories()

        if verbose:
            bar = LoopProgress(msg=f'Importing {var_name} using foamlib', final = len(target_times))

        for folder in target_times:
        
            # Decomposed case
            if self.decomposed_case:
                n_processors = len(glob.glob(os.path.join(self.path, 'processor*')))

                _single_time_field = list()

                for ii in range(n_processors):

                    d = os.path.join(self.path, f'processor{ii}', folder)
                    if os.path.isdir(d):

                        _path_field = os.path.join(d, var_name)

                        _single_time_field.append(
                            fl.FoamFieldFile(_path_field).internal_field
                        )

                # Append time instant and field
                time_instants.append(float(folder))
                field.append( np.concatenate(_single_time_field, axis=0) )

            # Reconstructed case
            else:
                d = os.path.join(self.path, folder)
                if os.path.isdir(d):
                    
                    _path_field = os.path.join(d, var_name)

                    # Read field using foamlib
                    field.append(
                        fl.FoamFieldFile(_path_field).internal_field
                    )

                    # Append time instant
                    time_instants.append(float(folder))

            if verbose:
                bar.update(1)

        return field, np.asarray(time_instants)
    
    def _import_with_fluidfoam(self, var_name: str, verbose: bool = True, target_times: list[str] = None):
        """
        Importing time instances from OpenFOAM directory using fluidfoam.
        
        Parameters
        ----------
        var_name : str
            Name of the field to import.
        verbose: boolean, (Default = True) 
            If `True`, printing is enabled
        target_times : list[str], optional
            List of time folders to read. If `None`, all time instants are read.

        Returns
        -------
        field : list
            Imported list of functions (each element is a `numpy.ndarray`), sorted in time.
        time_instants : np.ndarray
            Sorted list of time.
        """

        field = list()
        time_instants = list()

        if target_times is None:
            target_times = self._get_time_directories()

        if verbose:
            bar = LoopProgress(msg=f'Importing {var_name} using fluidfoam', final = len(target_times))
        
        # Helper function to handle fluidfoam's shape outputs
        reshape_field = lambda f: f.reshape(-1, 1) if f.ndim < 2 else f.T

        for folder in target_times:
        
            # Decomposed case
            if self.decomposed_case:
                n_processors = len(glob.glob(os.path.join(self.path, 'processor*')))

                _single_time_field = list()

                for ii in range(n_processors):

                    d = os.path.join(self.path, f'processor{ii}')
                    _tmp = of.readfield(d, folder, var_name, verbose=False)
                    
                    # Append to the list
                    _single_time_field.append(reshape_field(_tmp))

                # Append time instant and field
                time_instants.append(float(folder))
                field.append( np.concatenate(_single_time_field, axis=0) )

            # Reconstructed case
            else:
                d = os.path.join(self.path, folder)
                if os.path.isdir(d):
                        
                    _tmp = of.readfield(self.path, folder, var_name, verbose=False)

                    # Append to the list
                    field.append(reshape_field(_tmp))

                    # Deprecated code - to be removed after testing
                    # try: # scalar field
                    #     field.append( of.readscalar(self.path, file, var_name, verbose=False).reshape(-1,1) )
                    # except ValueError:
                    #     try: # vector field
                    #         field.append(of.readvector(self.path, file, var_name, verbose=False).T)
                    #     except ValueError: # tensor field
                    #         try: # tensor field
                    #             field.append(of.readtensor(self.path, file, var_name, verbose=False).T)
                    #         except ValueError: # symmetric tensor field
                    #             field.append(of.readsymmtensor(self.path, file, var_name, verbose=False).T)

                    # Append time instant
                    time_instants.append(float(folder))

            if verbose:
                bar.update(1)

        return field, np.asarray(time_instants)
    
    def _import_with_pyvista(self, var_name: str, extract_cell_data: bool = True, verbose: bool = True, target_times: list[str] = None):
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
        target_times : list[str], optional
            List of time folders to read. If `None`, all time instants are read.
        
        Returns
        -------
        field : list
            Imported list of functions (each element is a `numpy.ndarray`), sorted in time.
        time_instants : np.ndarray
            Sorted list of time.
        """

        if target_times is None:
             target_times = self._get_time_directories()

        field = list()
        time_instants = list()

        if verbose:
            bar = LoopProgress(msg=f'Importing {var_name} using pyvista', final = len(target_times))

        for folder in target_times:
            if verbose:
                bar.update(1)

            t = float(folder)

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

            time_instants.append(t)

        return field, np.asarray(time_instants)

    def _read_stacked_processor_mesh_fluidfoam(self):
        """
        Reads the centroids of a decomposed case using fluidfoam, stacking the meshes of all processors.

        Returns
        -------
        mesh : np.ndarray
            The combined nodes/centroids of all processors.
        """

        assert self.decomposed_case, "This method is only applicable for decomposed cases."

        n_processors = len(glob.glob(os.path.join(self.path, 'processor*')))

        blocks_to_merge = []
        for ii in range(n_processors):
            d = os.path.join(self.path, f'processor{ii}')
            mesh = of.readmesh(d, verbose=False)
            blocks_to_merge.append(mesh)

        combined_mesh = np.concatenate(blocks_to_merge, axis=1)

        return combined_mesh.T
    
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
        
    Parameters
    ----------
    path : str
        Path of the OpenFOAM case directory.
    skip_zero_time : bool, optional
        If `True`, the zero time folder is skipped. Default is `False`.
    decomposed_case : bool, optional
        If `True`, the case is considered as decomposed. Default is `False` (pyvista only).
        
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

        Moreover, the cumulative number of cells for each region is stored in `self.ncells_cumulative` to facilitate the data extraction process.

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
        self.ncells_cumulative = []
        self.ncells_cumulative.append(0)
        for region in _regions:
            _mesh = self._region_mesh(region)
            _mesh.clear_data() # remove all the data
            blocks_to_merge.append(_mesh)

            # Update cumulative cell count
            self.ncells_cumulative.append(self.ncells_cumulative[-1] + _mesh.n_cells)

        combined_mesh = pv.MultiBlock(blocks_to_merge).combine(merge_points=True)
        
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
                     import_mode: str = 'pyvista',
                     verbose: bool = True):
        r"""
        Importing all time instances (**skipping zero folder**) from OpenFOAM directory for all available regions, if not skip.
        
        Two methods are available for the import process: `pyvista` and `fluidfoam`.The latter is typically faster.

        *Only cell data extraction is supported for multi-region cases: to prevent data misalignment issues.*

        Parameters
        ----------
        var_name : str
            Name of the field to import.
        import_mode : str, optional
            Method to use for importing the data. It can be either 'pyvista' or 'fluidfoam'. Default is 'pyvista'.
        verbose: boolean, (Default = True) 
            If `True`, printing is enabled

        Returns
        -------
        snaps : FunctionsList
            Imported list of functions (each element is a `numpy.ndarray`), sorted in time.
        time_instants : np.ndarray
            Sorted list of time.
        """

        # Concatenate all regional snapshots
        regional_snaps = []

        if verbose:
            bar = LoopProgress(msg=f'Importing {var_name} from all regions - {import_mode}', final = len(self.regions))

        for region in self.regions:
            if verbose:
                bar.update(1)

            if import_mode == 'pyvista':
                try:
                    snaps_region, time_instants = self._import_region_field_pyvista(region, var_name, extract_cell_data=True, verbose=False)
                except KeyError:
                    if verbose:
                        warnings.warn(f"\nSkipping region '{region}': field '{var_name}' not found.")
                        continue 

            elif import_mode == 'fluidfoam':

                try: 
                    snaps_region, time_instants = self._import_region_field_fluidfoam(region, var_name, verbose=False)

                except FileNotFoundError:
                    if verbose:
                        warnings.warn(f"\nSkipping region '{region}': field '{var_name}' not found.")
                        continue
                    
            elif import_mode == 'foamlib':

                try: 
                    snaps_region, time_instants = self._import_region_field_foamlib(region, var_name, verbose=False)

                except FileNotFoundError:
                    if verbose:
                        warnings.warn(f"\nSkipping region '{region}': field '{var_name}' not found.")
                        continue

            else:
                raise ValueError(f"Unsupported import method '{import_mode}'. Use 'pyvista' or 'fluidfoam or 'foamlib'.")
            
            regional_snaps.append(snaps_region)

        # Concatenate all regional snapshots
        concatenated_snaps = np.concatenate([snaps.return_matrix() for snaps in regional_snaps], axis=0)
        snaps = FunctionsList(snap_matrix=concatenated_snaps)
        return snaps, np.asarray(time_instants)
    
    def _import_region_field_foamlib(self, region: str, var_name: str,
                            file_list: list[float] = None, verbose: bool = True):
        
        field = list()
        time_instants = list()

        if file_list is None:

            file_list = [
                f for f in os.listdir(self.path) 
                if os.path.isdir(os.path.join(self.path, f)) and f.replace('.', '', 1).isdigit()
            ]

            # Sort numerically
            file_list.sort(key=float)

            # Skip zero time folder if needed
            if self.reader.skip_zero_time and file_list[0] in ['0']:
                file_list = file_list[1:]

        if verbose:
            bar = LoopProgress(msg=f'Importing {var_name} from region {region} using foamlib', final = len(file_list))

        for idx_t, file in enumerate(file_list):

            if not ((file == '0.orig') or (file == '0') or (file == '0.ss')):
                d = os.path.join(self.path, file)
                if os.path.isdir(d):

                    _path_field = os.path.join(d, region, var_name)

                    # Read field using foamlib
                    field.append(
                        fl.FoamFieldFile(_path_field).internal_field
                    )
                
                    # Append time instant
                    time_instants.append(float(file))

            if verbose:
                bar.update(1)

        # Convert list to FunctionsList
        Nh = field[0].flatten().shape[0]
        snaps = FunctionsList(dofs=Nh)
        for f in field:
            snaps.append(f.flatten())

        return snaps, np.asarray(time_instants)
    
    def _import_region_field_fluidfoam(self, region: str, var_name: str,
                            file_list: list[float] = None, verbose: bool = True):
        
        field = list()
        time_instants = list()

        if file_list is None:

            file_list = [
                f for f in os.listdir(self.path) 
                if os.path.isdir(os.path.join(self.path, f)) and f.replace('.', '', 1).isdigit()
            ]

            # Sort numerically
            file_list.sort(key=float)

            # Skip zero time folder if needed
            if self.reader.skip_zero_time and file_list[0] in ['0']:
                file_list = file_list[1:]

        if verbose:
            bar = LoopProgress(msg=f'Importing {var_name} from region {region} using fluidfoam', final = len(file_list))

        for idx_t, file in enumerate(file_list):

            if not ((file == '0.orig') or (file == '0') or (file == '0.ss')):
                d = os.path.join(self.path, file)
                if os.path.isdir(d):

                    _tmp = of.readfield(self.path, file, var_name, region=region, verbose=False)

                    # Scalar field
                    if (_tmp.ndim < 2):
                        _tmp = _tmp.reshape(-1, 1)
                    else: # Vector or tensor field
                        _tmp = _tmp.T

                    # Append to the list
                    field.append(_tmp)

                    # Deprecated code - to be removed after testing
                    # try: # scalar field
                    #     field.append( of.readscalar(self.path, t, var_name, region=region, verbose=False).reshape(-1,1) )
                    # except ValueError:
                    #     try: # vector field
                    #         field.append(of.readvector(self.path, t, var_name, region=region, verbose=False).T)
                    #     except ValueError: # tensor field
                    #         try: # tensor field
                    #             field.append(of.readtensor(self.path, t, var_name, region=region, verbose=False).T)
                    #         except ValueError: # symmetric tensor field
                    #             field.append(of.readsymmtensor(self.path, t, var_name, region=region, verbose=False).T)

                    # Append time instant
                    time_instants.append(float(file))

            if verbose:
                bar.update(1)

        # Convert list to FunctionsList
        Nh = field[0].flatten().shape[0]
        snaps = FunctionsList(dofs=Nh)
        for f in field:
            snaps.append(f.flatten())

        return snaps, np.asarray(time_instants)

    def _import_region_field_pyvista(self, region: str, var_name: str,
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