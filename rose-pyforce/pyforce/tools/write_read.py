# I/O tools
# Authors: Stefano Riva, Yantao Luo, NRG, Politecnico di Milano
# Latest Code Update: 24 March 2026
# Latest Doc  Update: 24 March 2026

from typing import Union, List

from .functions_list import FunctionsList
from .backends import LoopProgress
from scipy.spatial import cKDTree

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
                     target_times: list[str] = None,
                     extract_cell_data: bool = True,
                     verbose: bool = True,
                     ) -> tuple[FunctionsList, np.ndarray]:
        r"""
        Importing time instances from OpenFOAM directory, if not specified, for all time folders. 

        Three methods are available for the import process: `pyvista`, `fluidfoam`, and `foamlib`. The latter two are typically faster than pyvista, especially for large cases.

        If you want to import specific time instances, provide their folder names in the `target_times` list.

        If you want to import point data instead of cell data, set `extract_cell_data` to `False`, only for pyvista method.

        Parameters
        ----------
        var_name : str
            Name of the field to import.
        import_mode : str, (Default: 'pyvista')
            The mode of import, either 'pyvista' or 'fluidfoam' or 'foamlib'.
        target_times: list[str], optional
            List of time folders to import. If `None`, all time instances are imported.
        extract_cell_data : boolean, (Default: True)
            If `True`, the cell data from centroids is extracted; otherwise, point data is extracted.
        verbose: boolean, (Default = True) 
            If `True`, printing is enabled.

        Returns
        -------
        field : FunctionsList
            Imported list of functions (each element is a `numpy.ndarray`), sorted in time.
        time_instants : np.ndarray
            Sorted list of time.
        """

        if import_mode == 'fluidfoam' and extract_cell_data:
            field, time_instants = self._import_with_fluidfoam(var_name, target_times = target_times, verbose=verbose)
        elif import_mode == 'foamlib' and extract_cell_data:
            field, time_instants = self._import_with_foamlib(var_name, target_times = target_times, verbose=verbose)
        elif import_mode == 'pyvista':
            field, time_instants = self._import_with_pyvista(var_name, extract_cell_data=extract_cell_data, target_times = target_times, verbose=verbose)
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

        base_path = self.path if not self.decomposed_case else os.path.join(self.path, 'processor0')
        
        file_list = [
            f for f in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, f))
            and f.replace('.', '', 1).isdigit()
        ]

        # Sort numerically
        file_list.sort(key=float)

        # Skip zero time folder if needed
        if self.reader.skip_zero_time and file_list[0] in ['0']:
            file_list = file_list[1:]

        return file_list
        
    def _import_with_foamlib(self, var_name: str, 
                             target_times: list[str] = None,
                             verbose: bool = True) -> tuple[list, np.ndarray]:
        """
        Importing time instances from OpenFOAM directory using foamlib.
        
        Parameters
        ----------
        var_name : str
            Name of the field to import.
        target_times : list[str], optional
            List of time folders to read. If `None`, all time instants are read.
        verbose: boolean, (Default = True) 
            If `True`, printing is enabled

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
    
    def _import_with_fluidfoam(self, var_name: str, 
                               target_times: list[str] = None,
                               verbose: bool = True) -> tuple[list, np.ndarray]:
        """
        Importing time instances from OpenFOAM directory using fluidfoam.
        
        Parameters
        ----------
        var_name : str
            Name of the field to import.
        target_times : list[str], optional
            List of time folders to read. If `None`, all time instants are read.
        verbose: boolean, (Default = True) 
            If `True`, printing is enabled

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
                        
                _tmp = of.readfield(self.path, folder, var_name, verbose=False)

                # Append to the list
                field.append(reshape_field(_tmp))

                # Append time instant
                time_instants.append(float(folder))

            if verbose:
                bar.update(1)

        return field, np.asarray(time_instants)
    
    def _import_with_pyvista(self, var_name: str, 
                             extract_cell_data: bool = True, 
                             target_times: list[str] = None,
                             verbose: bool = True) -> tuple[list, np.ndarray]:
        """
        Importing time instances from OpenFOAM directory using pyvista.
        
        Parameters  
        ----------
        var_name : str
            Name of the field to import.
        extract_cell_data : boolean, (Default: True)
            If `True`, the cell data from centroids is extracted; otherwise, point data is extracted.
        target_times : list[str], optional
            List of time folders to read. If `None`, all time instants are read.
        verbose: boolean, (Default = True) 
            If `True`, printing is enabled
        
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
    **Note**: "Decomposed" here refers exclusively to parallel MPI domain decomposition (i.e., data split into `processor*` directories), not multi-region setups.
        
    Either the fluidfoam library (see https://github.com/fluiddyn/fluidfoam) or the foamlib (see https://github.com/gerlero/foamlib) or pyvista are exploited for the import process.

    Both org and com version of OpenFOAM are supported.

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
        
        # Container for zero field, which is used by function import_field
        self.zero_field_regions = None

    def _region_mesh(self, region: str):
        """
        Returns the mesh of the specified region of the OpenFOAM case.
        
        Parameters
        ----------
        region : str
            Name of the region to extract the mesh from.
        decomposed_mode : bool
            If True, the mesh is read from processor* folders. Default is False.
            
        Returns
        -------
        mesh : pyvista.UnstructuredGrid
            The mesh of the specified region of the OpenFOAM case.
        """

        grid = self.reader.read()[region]["internalMesh"]
        grid.clear_data()  # remove all the data
        return grid

    # def _region_mesh(self, region: str, decomposed_mode: bool = False):
    #     """
    #     Returns the mesh of the specified region of the OpenFOAM case.
        
    #     Parameters
    #     ----------
    #     region : str
    #         Name of the region to extract the mesh from.
    #     decomposed_mode : bool
    #         If True, the mesh is read from processor* folders. Default is False.
            
    #     Returns
    #     -------
    #     mesh : pyvista.UnstructuredGrid
    #         The mesh of the specified region of the OpenFOAM case.
    #     """
    #     if decomposed_mode:
    #         case_dir = Path(self.path)
        
    #         # number of processor folders
    #         n_processors = len(glob.glob(os.path.join(self.path, "processor*")))
        
    #         # check if vtk exists
    #         def vtk_exists():
    #             for ii in range(n_processors):
    #                 vtk_dir = os.path.join(self.path, f"processor{ii}", "VTK", region)
    #                 if not os.path.exists(vtk_dir):
    #                     return False
    #             return True
        
    #         # generate vtk if missing
    #         if not vtk_exists():
    #             subprocess.run(
    #                 [
    #                     "mpirun", "-np", str(n_processors), "foamToVTK",
    #                     "-parallel", "-latestTime", "-region", region, "-noPointValues"
    #                 ],
    #                 cwd=self.path,
    #                 check=True
    #             )
        
    #         # read meshes
    #         meshes = []
    #         for ii in range(n_processors):
    #             vtk_dir = Path(self.path) / f"processor{ii}" / "VTK" / region
    #             files = sorted(vtk_dir.glob(f"processor{ii}_*.vtk"))
    #             if not files:
    #                 raise FileNotFoundError(f"No VTK files found in {vtk_dir}")
    #             meshes.append(pv.read(files[-1]))
        
    #         # merge in processor order
    #         grid = meshes[0].copy()
    #         for m in meshes[1:]:
    #             grid = grid.merge(m, merge_points=False)
        
    #     else:
    #         grid = self.reader.read()[region]["internalMesh"]
        
    #     grid.clear_data()  # remove all the data
    #     return grid

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
    
    def save_mesh(self, filename: str, region: str | list[str] = None):
        """
        Saves the mesh (either all regions or a specific list) of the OpenFOAM case to a vtk-file.
        
        Parameters
        ----------
        filename : str
            Name of the file to save the mesh.
        regions : str | list[str], optional
            If specified, only the mesh of the given region is saved. If `None`, the combined mesh of all regions is saved.
        """
        if region is None:
            mesh = self.mesh()
        elif isinstance(region, str):
            mesh = self._region_mesh(region)
        elif isinstance(region, list):
            mesh = self.mesh(region)
        else:
            raise TypeError(f"Unsupported region type: {type(region)}")

        mesh.save(f"{filename}.vtk")

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
    
    def _get_time_directories(self):
        """"
        Get the list of time directories in the OpenFOAM case, sorted in time, and skipping zero time if needed.

        Returns
        -------
        file_list : list[str]
            List of time directories in the OpenFOAM case, sorted in time, and skipping zero time if needed.
        """
        base_path = self.path if not self.decomposed_case else os.path.join(self.path, 'processor0')
        
        file_list = [
            f for f in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, f))
            and f.replace('.', '', 1).isdigit()
        ]

        # Sort numerically
        file_list.sort(key=float)

        # Skip zero time folder if needed
        if self.reader.skip_zero_time and file_list[0] in ['0']:
            file_list = file_list[1:]

        return file_list

    def import_field(self, var_name: str,
                     import_mode: str = 'pyvista',
                     target_times: list[str] = None,
                     regions_to_import: str | list[str] = None,
                     verbose: bool = True, 
                     inner_verbose: bool = False
                     ) -> tuple[FunctionsList, np.ndarray]:
        r"""
        Importing all time instances (**skipping zero folder**) from OpenFOAM directory for all available regions, if not skip.
        
        Three methods are available for the import process: `pyvista`, `fluidfoam`, and `foamlib`. The latter two are typically faster than pyvista, especially for large cases.

        If you want to import specific time instances, provide their folder names in the `target_times` list.

        *Only cell data extraction is supported for multi-region cases: to prevent data misalignment issues.*

        **Note on parallel-decomposed cases**: For multi-region cases that are also parallel-decomposed, the import process will automatically handle the reconstruction of the field across all processors for each region, there might be visualisation issues for 'foamlib' and 'fluidfoam' if the decomposition is not uniform and more advanced techniques are used. If needed, exploit 'reconstructPar' from OpenFOAM.
        Decomposition Methods tested:
        * hierarchical
        * simple
        * ... (to be extended if needed)

        Parameters
        ----------
        var_name : str
            Name of the field to import.
        import_mode : str, optional
            Method to use for importing the data. It can be either 'pyvista' or 'fluidfoam'. Default is 'pyvista'.
        target_times: list[str], optional
            List of time folders to import. If `None`, all time instances are imported.
        regions_to_import : str | list[str], optional
            If specified, only the given region(s) are imported. If `None`, all valid regions for the specified field are imported.
        verbose: boolean, (Default = True) 
            If `True`, printing is enabled
        inner_verbose: boolean, (Default = False)
            If `True`, verbose mode is enabled for the inner import functions (e.g., `_import_region_field_pyvista`), otherwise only the outer loop progress is printed.

        Returns
        -------
        snaps : FunctionsList
            Imported list of functions (each element is a `numpy.ndarray`), sorted in time.
        time_instants : np.ndarray
            Sorted list of time.
        """

        # Determine regions to import
        if regions_to_import is None:
            regions_to_import = self._get_valid_regions_for_field(var_name)
        elif isinstance(regions_to_import, str):
            regions_to_import = [regions_to_import]
        elif isinstance(regions_to_import, list):
            pass
        else:
            raise TypeError(f"Unsupported region type: {type(regions_to_import)}")
        
        # Check that the specified regions are valid
        for region in regions_to_import:

            if region not in self._get_valid_regions_for_field(var_name):
                raise ValueError(f"Region '{region}' is not available in the case. Available regions: {self._get_valid_regions_for_field(var_name)}")

        # Concatenate all regional snapshots
        regional_snaps = []

        if verbose:
            bar = LoopProgress(msg=f'Importing {var_name} - {import_mode}', final = len(regions_to_import))

        for region in regions_to_import:

            if import_mode == 'pyvista':
                snaps_region, time_instants = self._import_region_field_pyvista(var_name, region, target_times = target_times, verbose=inner_verbose)
            elif import_mode == 'fluidfoam':
                snaps_region, time_instants = self._import_region_field_fluidfoam(var_name, region, target_times = target_times, verbose=inner_verbose)
            elif import_mode == 'foamlib':
                snaps_region, time_instants = self._import_region_field_foamlib(var_name, region, target_times = target_times, verbose=inner_verbose)
            else:
                raise ValueError(f"Unsupported import method '{import_mode}'. Use 'pyvista' or 'fluidfoam or 'foamlib'.")
            
            regional_snaps.append(snaps_region)

            if verbose:
                bar.update(1)

        # Concatenate all regional snapshots
        concatenated_snaps = np.concatenate([snaps.return_matrix() for snaps in regional_snaps], axis=0)
        snaps = FunctionsList(snap_matrix=concatenated_snaps)

        return snaps, np.asarray(time_instants)
    
    def _import_region_field_foamlib(self, var_name: str, region: str,
                                        target_times: list[str] = None, 
                                        verbose: bool = True) -> tuple[FunctionsList, np.ndarray]:
        
        """
        Import time instances from OpenFOAM directory for a specific region using foamlib.

        Parameters
        ----------
        var_name : str
            Name of the field to import.
        region : str
            Name of the region to import the field from.
        target_times : list[str], optional
            List of time folders to read. If `None`, all time instants are read.
        verbose: boolean, (Default = True) 
            If `True`, printing is enabled

        Returns
        -------
        snaps : FunctionsList
            Imported list of functions (each element is a `numpy.ndarray`), sorted in time.
        time_instants : np.ndarray
            Sorted list of time.
        """
        
        field = list()
        time_instants = list()

        if target_times is None:
                target_times = self._get_time_directories()

        if verbose:
            bar = LoopProgress(msg=f'Importing {var_name} from region {region} using foamlib', final = len(target_times))

        # Flag to check if we are dealing with a uniform field
        uniform_field_flag = False

        for folder in target_times:
            
            # Decomposed case
            if self.decomposed_case:
                n_processors = len(glob.glob(os.path.join(self.path, 'processor*')))

                _single_time_field = list()
                
                for ii in range(n_processors):

                    d = os.path.join(self.path, f'processor{ii}', folder, region)

                    _path_field = os.path.join(d, var_name)
                    _file = fl.FoamFieldFile(_path_field)

                    _tmp = _file.internal_field

                    if isinstance(_tmp, float): # Check for uniform scalar field, which would cause issues in the concatenation process
                        assert _file.class_ == 'volScalarField', "Uniform fields are currently only supported for scalar fields, to be extended to vector and tensor fields if needed."
                        _single_time_field.append(np.array([_tmp]))
                        uniform_field_flag = True
                    else:
                        
                        # Check for uniform vector field, which would cause issues in the concatenation process - to be extended to tensor fields if needed
                        if _file.class_ == 'volVectorField' and _tmp.ndim == 1:
                            _tmp = _tmp.reshape(-1, 3) # reshape to (n_cells, 3) for vector fields
                            uniform_field_flag = True

                        # Append to the list
                        _single_time_field.append(
                            _tmp
                        )

                # Concatenate fields on different processors for the same time instant
                field_to_append = np.concatenate(_single_time_field, axis=0)

            # Reconstructed case
            else:
                d = os.path.join(self.path, folder, region)

                _path_field = os.path.join(d, var_name)

                # Read field using foamlib
                _file = fl.FoamFieldFile(_path_field)
                _tmp = _file.internal_field

                # Append to the list
                if isinstance(_tmp, float):
                    field_to_append = np.array([_tmp])
                    uniform_field_flag = True
                else:
                    
                    # Check for uniform vector field, which would cause issues in the concatenation process - to be extended to tensor fields if needed
                    if _file.class_ == 'volVectorField' and _tmp.ndim == 1:
                        _tmp = _tmp.reshape(-1, 3) # reshape to (n_cells, 3) for vector fields
                        uniform_field_flag = True

                    field_to_append = _tmp

            # Check that we are not dealing with a uniform field (i.e., a single value as for initial conditions), which would cause issues in the concatenation process - scalar and vector fields only for now, to be extended to fields if needed
            if uniform_field_flag:

                # Get the number of cells in the region mesh
                n_region_cells = self._region_mesh(region).n_cells

                # Get the shape of the field to append (for vector and tensor fields)
                fun_shape = field_to_append.shape[1:] if field_to_append.ndim > 1 else ()

                # Assign the uniform value to all cells of the region
                field_to_append = np.ones((n_region_cells, *fun_shape)) * field_to_append[0] # assign the uniform value to all cells of the region

            # Append time instant
            time_instants.append(float(folder))

            # Append field
            field.append(field_to_append)

            if verbose:
                bar.update(1)

        # Convert list to FunctionsList
        Nh = field[0].flatten().shape[0]
        snaps = FunctionsList(dofs=Nh)
        for f in field:
            snaps.append(f.flatten())

        return snaps, np.asarray(time_instants)
    
    def _import_region_field_fluidfoam(self, var_name: str, region: str, 
                                        target_times: list[str] = None, 
                                        verbose: bool = True) -> tuple[list, np.ndarray]:
        
        """
        Import time instances from OpenFOAM directory for a specific region using fluidfoam.

        Parameters
        ----------
        var_name : str
            Name of the field to import.
        region : str
            Name of the region to import the field from.
        target_times : list[str], optional
            List of time folders to read. If `None`, all time instants are read.
        verbose: boolean, (Default = True) 
            If `True`, printing is enabled

        Returns
        -------
        snaps : FunctionsList
            Imported list of functions (each element is a `numpy.ndarray`), sorted in time.
        time_instants : np.ndarray
            Sorted list of time.
        """

        field = list()
        time_instants = list()

        if target_times is None:
            target_times = self._get_time_directories()

        if verbose:
            bar = LoopProgress(msg=f'Importing {var_name} from region {region} using fluidfoam', final = len(target_times))

        # Helper function to handle fluidfoam's shape outputs
        reshape_field = lambda f: f.reshape(-1, 1) if f.ndim < 2 else f.T

        # Flag to check if we are dealing with a uniform field
        uniform_field_flag = False

        for folder in target_times:

            # Decomposed case
            if self.decomposed_case:
                n_processors = len(glob.glob(os.path.join(self.path, 'processor*')))

                _single_time_field = list()

                for ii in range(n_processors):

                    d = os.path.join(self.path, f'processor{ii}')

                    _tmp = of.readfield(d, folder, var_name, region=region, verbose=False)

                    # Reshape the field
                    _tmp = reshape_field(_tmp)

                    if _tmp.size == 1: # Check for uniform scalar field, which would cause issues in the concatenation process
                        uniform_field_flag = True

                    # Append to the list
                    _single_time_field.append(_tmp)
                
                # Define field to append
                field_to_append = np.concatenate(_single_time_field, axis=0)
            
            # Reconstructed case
            else:
                d = os.path.join(self.path, folder)
                if os.path.isdir(d):
                    _tmp = of.readfield(self.path, folder, var_name, region=region, verbose=False)

                    # Reshape the field
                    field_to_append = reshape_field(_tmp)

                    # Check for uniform scalar field, which would cause issues in the concatenation process
                    if field_to_append.size == 1:
                        uniform_field_flag = True

            # Check that we are not dealing with a uniform field (i.e., a single value as for initial conditions), which would cause issues in the concatenation process - scalar and vector fields only for now, to be extended to fields if needed
            if uniform_field_flag:

                # Get the number of cells in the region mesh
                n_region_cells = self._region_mesh(region).n_cells

                # Get the shape of the field to append (for vector and tensor fields)
                fun_shape = field_to_append.shape[1:] if field_to_append.ndim > 1 else ()

                # Assign the uniform value to all cells of the region
                field_to_append = np.ones((n_region_cells, *fun_shape)) * field_to_append[0] # assign the uniform value to all cells of the region

            # Append time instant
            time_instants.append(float(folder))

            # Append field
            field.append(field_to_append)

            if verbose:
                bar.update(1)

        # Convert list to FunctionsList
        Nh = field[0].flatten().shape[0]
        snaps = FunctionsList(dofs=Nh)
        for f in field:
            snaps.append(f.flatten())

        return snaps, np.asarray(time_instants)

    def _import_region_field_pyvista(self, var_name: str, region: str,
                                     target_times: list[str] = None, 
                                        verbose: bool = True) -> tuple[FunctionsList, np.ndarray]:
        """
        Import time instances from OpenFOAM directory for a specific region using pyvista.

        Parameters
        ----------
        var_name : str
            Name of the field to import.
        region : str
            Name of the region to import the field from.
        target_times : list[str], optional
            List of time folders to read. If `None`, all time instants are read.
        verbose: boolean, (Default = True) 
            If `True`, printing is enabled  

        Returns 
        -------
        snaps : FunctionsList
            Imported list of functions (each element is a `numpy.ndarray`), sorted in time.
        time_instants : np.ndarray
            Sorted list of time.
        """

        if target_times is None:
             target_times = self._get_time_directories()

        field = list()
        time_instants = list()

        if verbose:
            bar = LoopProgress(msg=f'Importing {var_name} from region {region} using pyvista', final = len(target_times))

        for folder in target_times:

            t = float(folder)

            # Set active time
            self.reader.set_active_time_value(t)
            mesh = self.reader.read()[region]['internalMesh']

            # Extract data
            if var_name in mesh.cell_data.keys(): # centroids data
                field.append(mesh.cell_data[var_name])
            else:
                raise KeyError(f"Field '{var_name}' not found in region '{region}' at time {t}. Available cell data fields: {list(mesh.cell_data.keys())}")
            
            time_instants.append(t)

            if verbose:
                bar.update(1)

        # Convert list to FunctionsList
        Nh = field[0].flatten().shape[0]
        snaps = FunctionsList(dofs=Nh)
        for f in field:
            snaps.append(f.flatten())

        return snaps, np.asarray(time_instants)
        
def get_candidate_regions_points(all_grids: pv.UnstructuredGrid, 
                                 candidate: Union[pv.UnstructuredGrid, List[pv.UnstructuredGrid]],
                                 tolerance=1e-6):
    r"""
    Get target region points and corresponding indices within all regions.

    Parameters
    ----------
    all_grids : pv.UnstructuredGrid
        The grid containing all valid regions.
    candidate : pv.UnstructuredGrid | list of pv.UnstructuredGrid
        Target region grid(s). Can be a single grid or a list of grids for multiple regions.
    tolerance : float
        Distance tolerance used during KDTree matching. Default is 1e-6.

    Returns
    -------
    candidate_points : np.ndarray of shape (N, 3)
        Coordinates of the target points, used for training or post-processing.
    candidate_idx : np.ndarray shape (M,)
        Indices of target points within `all_grids`, used for training or post-processing.
        
    """
    
    if isinstance(candidate, pv.UnstructuredGrid):
        candidate_points = candidate.cell_centers().points
        
    elif isinstance(candidate, list):
        if not all(isinstance(r, pv.UnstructuredGrid) for r in candidate):
            raise TypeError("Candidate list must contain only pv.UnstructuredGrid")

        candidate_points = np.vstack([rm.cell_centers().points for rm in candidate])
        
    else:
        raise TypeError(f"Wrong type candidate: {type(candidate)}")
        
    # match points using KDTree
    all_points = all_grids.cell_centers().points
    tree = cKDTree(candidate_points)
    dist, _ = tree.query(all_points)

    candidate_idx = np.where(dist < tolerance)[0].tolist()
    candidate_idx = np.asarray(candidate_idx, dtype=int)

    return candidate_points, candidate_idx

def get_candidate_probes(all_grids: pv.UnstructuredGrid, 
                         candidate_grid: Union[pv.UnstructuredGrid, List[pv.UnstructuredGrid]],
                         candidate_points: np.ndarray | list[np.ndarray]):
    r"""
    Get target points and corresponding indices within all regions.
    Step 1: Map the input candidate point(s) to the nearest points in the specified `candidate_grid`.
    Step 2: Map the points obtained in Step 1 to the corresponding points in `all_grids`.
    This two-step mapping avoids assigning candidate points to incorrect regions due to differences in mesh discretisation.

    Parameters
    ----------
    all_grids : pv.UnstructuredGrid
        The grid containing all valid regions.
    candidate_grid: pv.UnstructuredGrid | list of pv.UnstructuredGrid
        Target region grid(s). Used to ensure the candidate point(s) are belonged to this(these) region(s).
    candidate_points : np.ndarray | list of np.ndarray, shape (N, 3,) | (3,) each
        Target point(s). Can be a single point or a list of points.

    Returns
    -------
    candidate_points : np.ndarray of shape (N, 3)
        Coordinates of the target points, used for training or post-processing.
    candidate_idx : np.ndarray shape (M,)
        Indices of target points within `all_grids`, used for training or post-processing.
        
    """
    
    # constrain candidate points to the specified region ("candidate_grid")
    if isinstance(candidate_grid, pv.UnstructuredGrid):
        candidate_region_points = candidate_grid.cell_centers().points
        
    elif isinstance(candidate_grid, list):
        if not all(isinstance(r, pv.UnstructuredGrid) for r in candidate_grid):
            raise TypeError("candidate_grid list must contain only pv.UnstructuredGrid")

        candidate_region_points = np.vstack([rm.cell_centers().points for rm in candidate_grid])
        
    else:
        raise TypeError(f"wrong type candidate_grid: {type(candidate_grid)}")

    tree_region = cKDTree(candidate_region_points)
    candidate_points = np.atleast_2d(candidate_points)
    
    # check if candidate points are outside the region    
    xmin, ymin, zmin = candidate_region_points.min(axis=0)
    xmax, ymax, zmax = candidate_region_points.max(axis=0)

    outside_mask = (
        (candidate_points[:,0] < xmin) | (candidate_points[:,0] > xmax) |
        (candidate_points[:,1] < ymin) | (candidate_points[:,1] > ymax) |
        (candidate_points[:,2] < zmin) | (candidate_points[:,2] > zmax)
    )
    
    if np.any(outside_mask):
        raise ValueError(f"some candidate(s) are outside the grid bounds: {candidate_points[outside_mask]}")
    
    # Step1, get indices and points in the specified region ("candidate_grid")
    dist, region_idx = tree_region.query(candidate_points)
    candidate_region_idx = np.asarray(region_idx, dtype=int)
    candidate_region_selected_points = candidate_region_points[candidate_region_idx]
    
    # Step2, get indices and points in all regions ("all_grids")
    all_points = all_grids.cell_centers().points
    tree = cKDTree(all_points)
    
    dist, idx = tree.query(candidate_region_selected_points)
    candidate_idx = np.asarray(idx, dtype=int)
    candidate_points = all_points[candidate_idx]

    return candidate_points, candidate_idx
    
def get_candidate_channel_all_points(all_grids: pv.UnstructuredGrid, 
                                    candidate_grid: Union[pv.UnstructuredGrid, List[pv.UnstructuredGrid]],
                                    candidate_xy: np.ndarray | list[np.ndarray], 
                                    tol=1e-6):
    """
    Get all points along a channel (fixed XY, any Z) and corresponding indices in all_grids.
    This returns all points with XY matching the candidate_xy within a tolerance.
        
    Parameters
    ----------
    all_grids : pv.UnstructuredGrid
        The global mesh.
    candidate_grid : pv.UnstructuredGrid | list of pv.UnstructuredGrid
        Target region grid(s). Used to ensure the candidate point(s) are belonged to this(these) region(s).
    candidate_xy : np.ndarray or list of np.ndarray, shape (2,) or (N,2)
        Target XY coordinates of points. Can be a single point or multiple points.
    tol : float
        Tolerance for matching XY coordinates. Notice: it should be tuned according to different geometric discretisation.
        
    Returns
    -------
    candidate_points : np.ndarray, shape (M,3)
        All points in the global mesh with matching XY.
    candidate_idx : np.ndarray, shape (M,)
        Indices of candidate_points in all_grids.
    """
    candidate_xy = np.atleast_2d(candidate_xy)
    
    # put all "candidate_grid" points in one container
    if isinstance(candidate_grid, pv.UnstructuredGrid):
        candidate_region_points = candidate_grid.cell_centers().points
    elif isinstance(candidate_grid, list):
        if not all(isinstance(r, pv.UnstructuredGrid) for r in candidate_grid):
            raise TypeError("candidate_grid list must contain only pv.UnstructuredGrid")
        candidate_region_points = np.vstack([rm.cell_centers().points for rm in candidate_grid])
    else:
        raise TypeError(f"wrong type candidate_grid: {type(candidate_grid)}")
    
    # get candidate points based on "candidate_grid"
    mask = np.zeros(len(candidate_region_points), dtype=bool)
    for xy in candidate_xy:
        mask |= (np.abs(candidate_region_points[:,0] - xy[0]) <= tol) & (np.abs(candidate_region_points[:,1] - xy[1]) <= tol)
    
    candidate_region_selected_points = candidate_region_points[mask]
    
    # map candidate points to all_grids
    all_points = all_grids.cell_centers().points
    tree_all = cKDTree(all_points)
    
    candidate_idx_list = [tree_all.query_ball_point(p, r=tol) for p in candidate_region_selected_points]
    candidate_idx = np.unique(np.concatenate(candidate_idx_list))
    candidate_points = all_points[candidate_idx]
    
    return candidate_points, candidate_idx