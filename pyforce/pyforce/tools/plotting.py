# Basic plotting tools using pyvista
# Author: Stefano Riva, PhD Student, NRG, Politecnico di Milano
# Latest Code Update: 16 September 2024
# Latest Doc  Update: 16 September 2024

import dolfinx.plot
from dolfinx.fem import Function
import numpy as np
import warnings
from matplotlib import cm
import pyvista as pv

def grids(fun: dolfinx.fem.Function, varname='u', log_plot: bool = False, 
          mag_plot: bool = True, **kwargs):
    """
    Creates a PyVista grid from a dolfinx.fem.Function and returns the warped or glyph representation.
    
    Parameters
    ----------
    fun : `dolfinx.fem.Function` 
        The function representing the field to be visualized.
    varname : str
        The name to assign to the data in the PyVista grid (default is 'u').
    log_plot : bool
        If True, apply a logarithmic plot to scalar data (default is False).
    mag_plot : bool
        If True, creates a vector warp of the grid. If False, uses glyphs (default is True).
    **kwargs
        Additional keyword arguments passed to the function.
    
    Returns
    ----------
    grid : `pyvista.UnstructuredGrid`
        The resulting PyVista grid, which can be visualized using PyVista plotting functions.
    
    """
    
    # Create a VTK mesh from the function's function space
    topology, cells, geometry = dolfinx.plot.create_vtk_mesh(fun.function_space)
    grid = pv.UnstructuredGrid(topology, cells, geometry)
    
    # Handle vector fields (multiple subspaces)
    if fun.function_space.num_sub_spaces > 0:
        # Initialize the values array with zeros and assign the real part of the function's array
        values = np.zeros((geometry.shape[0], 3))
        values[:, :len(fun)] = np.real(fun.x.array.reshape(geometry.shape[0], len(fun)))
        grid[varname] = values

        # Choose between warping by vector or using glyphs
        if mag_plot:
            warped = grid.warp_by_vector(varname, **kwargs)  # Apply `kwargs`
        else:
            warped = grid.glyph(varname, **kwargs)  # Apply `kwargs`
        
        return warped, values
    
    # Handle scalar fields (single subspace)
    else:
        if log_plot:
            values = np.log10(np.real(fun.x.array[:]))    
        else:
            values = np.real(fun.x.array[:])
            
        grid.point_data[varname] = values
        grid.set_active_scalars(varname)
        
        return grid, values.reshape(-1,1)

def plot_function(   fun: Function, filename: str = None, format: str = 'png', varname: str = 'u',
                    clim = None, colormap = cm.jet, resolution = [1080, 720],
                    zoom = 1., title = None, **kwargs):
    """
    Python function to plot a scalar field.

    Parameters
    ----------
    fun : Function 
        Field to plot.
    varname : str, optional (Default = 'u')
        Name of the variable.
    filename : str (Default = None)
        If `None`, the plot is shown; otherwise this is the name of the file to save.
    clim : optional (Default = None)
        Colorbar limit, if `None` the mininum and maximum of `fun` are computed.
    colormap : optional (Default = jet)
        Colormap for the plot
    resolution : list, optional (Default = [1080, 720])
        Resolution of the image
    zoom : float (Default = 1)
        Zoom of the plot.
    title : str (Default = None)
        If `None` no title is displayed.
    **kwargs : 
        Additional keyword arguments passed to the function.
    
    """

    plotter = pv.Plotter(off_screen=True, border=False, window_size=resolution)
    
    lab_fontsize = 20
    title_fontsize = 25
    zoom = zoom
    
    dict_cb = dict(title = varname, width = 0.75,
                    title_font_size=title_fontsize,
                    label_font_size=lab_fontsize,
                    color='k',
                    position_x=0.125, position_y=0.875,
                    shadow=True) 
    
    u_grid, values = grids(fun, varname, **kwargs)
    
    if clim is None:
        normed_values = np.linalg.norm(values, axis=1)
        clim = [min(normed_values) * 0.98, max(normed_values)* 1.02]
        
    plotter.add_mesh(u_grid, cmap = colormap, clim = clim, show_edges=False, scalar_bar_args=dict_cb)
    plotter.view_xy()
    
    if title is not None:
        plotter.add_title(title, font_size=25, color ='k')
    
    plotter.camera.zoom(zoom)
    plotter.set_background('white', top='white')

    ## Save figure
    if filename is not None:
        if format == 'pdf':
            plotter.save_graphic(filename+'.pdf')
        elif format == 'svg':
            plotter.save_graphic(filename+'.svg')
        elif format == 'png':
            plotter.screenshot(filename+'.png', transparent_background = True,  window_size=resolution)
        else:
            warnings.warn("Available output format are 'pdf', 'svg' and 'png'. Saved 'png' screenshot.")
            plotter.screenshot(filename+'.png', transparent_background = True,  window_size=resolution)
        plotter.close()
    else:
        return plotter