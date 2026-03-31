# Basic plotting tools using pyvista
# Author: Stefano Riva, PhD Student, NRG, Politecnico di Milano
# Latest Code Update: 07 October 2025
# Latest Doc  Update: 07 October 2025

import numpy as np
import pyvista as pv

def plot_function(grid: pv.UnstructuredGrid, 
                  snap: np.ndarray,
                  varname: str = 'u',
                  filename: str = None,
                  clim: tuple = None,
                  cmap: str = 'jet',
                  resolution: int = [1080, 720],
                  title: str = None,
                  **kwargs) -> pv.Plotter:
    
    """
    Plots a function on a PyVista grid and saves the plot to a file.
    
    Parameters  
    ----------
    grid : `pyvista.UnstructuredGrid`
        The PyVista grid on which the function will be plotted.
    snap : np.ndarray
        The function values to be plotted, should match the number of points in the grid.
    varname : str
        The name to assign to the data in the PyVista grid (default is 'u').
    filename : str, optional       
        The name of the file to save the plot. If None, the plot will not be saved.
    clim : tuple, optional
        The color limits for the plot. If None, the limits will be automatically determined from the data.
    cmap : str, optional
        The colormap to use for the plot (default is 'jet'). Other options include 'viridis', 'plasma', 'inferno', etc.
    resolution : list, optional
        The resolution of the plot in pixels (default is [1080, 720]).
    zoom : float, optional
        The zoom level for the plot (default is 1.0).   
    title : str, optional
        The title of the plot. If None, no title will be displayed.
    **kwargs : dict, optional
        Additional keyword arguments passed to the PyVista plotting functions.

    Returns
    -------
    pv.Plotter
        The PyVista plotter object used for the visualization.
    """

    # Create the plotter object
    pl = pv.Plotter(window_size=resolution)

    grid[varname] = snap
    grid.set_active_scalars(varname)

    pl.add_mesh(grid, scalars=varname, cmap=cmap, clim=clim, show_edges=False, **kwargs)
    pl.view_xy()

    if title is not None:
        pl.add_title(title, font_size=20, color='k')

    pl.set_background('white')

    ## Save the plot if a filename is provided
    if filename is not None:
        pl.screenshot(filename + '.png', window_size=resolution)
        pl.close()
    else:
        return pl