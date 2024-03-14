# Basic plotting tools using pyvista
# Author: Stefano Riva, PhD Student, NRG, Politecnico di Milano
# Latest Code Update: 31 August 2023
# Latest Doc  Update: 31 August 2023

import dolfinx.plot
from dolfinx.fem import Function
import numpy as np
import warnings
from matplotlib import cm
import pyvista as pv

# pv.set_jupyter_backend("trame") # deprecated

def get_scalar_grid(fun: Function, varname: str, real = True):
    """
    This function extracts the dofs of a scalar function for the plot using pyvista.

    Parameters
    ----------
    fun : Function 
        Function from which the dofs are extracted.
    varname : str
        Name of the variable.
    real : boolean, optional (Default = True) 
        Real dofs are considered, if `False` imaginary dofs are used.
    """
    topology, cells, geometry = dolfinx.plot.create_vtk_mesh(fun.function_space)
    u_grid = pv.UnstructuredGrid(topology, cells, geometry)

    if real:
        u_grid.point_data[varname] = fun.x.array[:].real
    else: 
        u_grid.point_data[varname] = fun.x.array[:].imag

    return u_grid

def PlotScalar(fun: Function, filename: str, format: str = 'png', varname: str = None,
               clim = None, colormap = cm.jet, resolution = [1080, 720]):
    """
    Python function to plot a scalar field.

    Parameters
    ----------
    fun : Function 
        Field to plot.
    varname : str
        Name of the variable.
    filename : str
        Name of the file to save.
    clim : optional (Default = None)
        Colorbar limit, if `None` the mininum and maximum of `fun` are computed.
    colormap : optional (Default = jet)
        Colormap for the plot
    resolution : list, optional (Default = [1080, 720])
        Resolution of the image

    """

    plotter = pv.Plotter(off_screen=True, border=False, window_size=resolution)
    lab_fontsize = 20
    title_fontsize = 25
    zoom = 1.
    
    if varname is None:
        varname = 'f'
    
    u_grid = get_scalar_grid(fun, varname)
    u_grid.set_active_scalars(varname)

    dict_cb = dict(title = varname, width = 0.75,
                    title_font_size=title_fontsize,
                    label_font_size=lab_fontsize,
                    color='k',
                    position_x=0.125, position_y=0.875,
                    shadow=True) 
    
    if clim is None:
        clim = [min(fun.x.array.real) * 0.975, max(fun.x.array.real)* 1.025]
        
    plotter.add_mesh(u_grid, cmap = colormap, clim = clim, show_edges=False, scalar_bar_args=dict_cb)
    plotter.view_xy()
    # plotter.add_title(varname, font_size=25, color ='k')
    plotter.camera.zoom(zoom)

    plotter.set_background('white', top='white')

    ## Save figure
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
    
def PlotVector(fun: dolfinx.fem.Function, filename: str, format: str = 'png', varname: str = None,
               clim = None, mag_plot = True, colormap = cm.jet, resolution= [1080, 720]):  
    """
    Python function to plot a scalar field.

    Parameters
    ----------
    fun : Function 
        Field to plot.
    varname : str
        Name of the variable.
    filename : str
        Name of the file to save.
    mag_plot : boolean, optional (Default = True)
    clim : optional (Default = None)
        Colorbar limit, if `None` the maximum of `fun` are computed (minimum is assumed 0). 
        Magnitude plot is performed, otherwise glyphs.
    colormap : optional (Default = jet)
        Colormap for the plot
    resolution : list, optional (Default = [1080, 720])
        Resolution of the image

    """
    
    plotter = pv.Plotter(off_screen=True,  border=False, window_size=resolution)
    lab_fontsize = 20
    title_fontsize = 25
    zoom = 1.1

    if varname is None:
        varname = 'u'
    
    topology, cells, geometry = dolfinx.plot.create_vtk_mesh(fun.function_space)
    grid = pv.UnstructuredGrid(topology, cells, geometry)

    values = np.zeros((geometry.shape[0], 3))
    values[:, :len(fun)] = np.real(fun.x.array.reshape(geometry.shape[0], len(fun)))
    grid[varname] = values

    if mag_plot:
        warped = grid.warp_by_vector(varname, factor=0.0) 
    else:
        warped = grid.glyph(varname, factor=0.15, tolerance=0.02)

    dict_cb = dict(title = varname, width = 0.75,
                    title_font_size=title_fontsize,
                    label_font_size=lab_fontsize,
                    color = 'k',
                    position_x=0.125, position_y=0.875,
                    shadow=True) 
    
    if clim is None:
        clim = [0., max(np.sqrt(values[:, 0]**2+values[:, 1]**2+values[:, 2]**2))]
    plotter.add_mesh(warped, clim = clim, cmap = colormap, show_edges=False, scalar_bar_args=dict_cb)
    plotter.view_xy()
    plotter.camera.zoom(zoom)

    plotter.set_background('white', top='white')

    ## Save figure
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
        
# # Additional function to be used as template
# def plotScalar_FOMvsRecon(V, true_fun, no_drift, drift, varname: str, filename:str, colormap = cm.jet, 
#                          algo='GEIM', resolution = [2560, 1600]):
   
#     plotter = pv.Plotter(shape=(1,3), off_screen=True, border=False, window_size=resolution)
#     lab_fontsize = 20
#     title_fontsize = 35
#     zoom = 1.2

#     # True field
#     plotter.subplot(0,0)
#     u_grid = get_scalar_grid(true_fun, varname, V)
#     u_grid.set_active_scalars(varname)
#     clim = [min(true_fun.real) * 0.9, max(true_fun.real)* 1.1]

#     dict_cb = dict(title = varname, width = 0.75,
#                     title_font_size=title_fontsize,
#                     label_font_size=lab_fontsize,
#                     color='k',
#                     position_x=0.125, position_y=0.875,
#                     shadow=True) 
    
#     plotter.add_mesh(u_grid, cmap = colormap, clim = clim, show_edges=False, scalar_bar_args=dict_cb)
#     plotter.view_xy()
#     # plotter.add_title(varname, font_size=25, color ='k')
#     plotter.camera.zoom(zoom)

#     # reconstruction - no drift
#     plotter.subplot(0,1)
#     u2_grid = get_scalar_grid(no_drift, algo+' (no drift) - '+varname, V)
#     u2_grid.set_active_scalars( algo+' (no drift) - '+varname)

#     dict_cb['title'] =  algo+' (no drift) - '+varname

#     plotter.add_mesh(u2_grid, cmap = colormap, clim = clim, show_edges=False, scalar_bar_args=dict_cb)
#     plotter.view_xy()
#     plotter.camera.zoom(zoom)
#     # plotter.add_title(varname, font_size=25, color ='k')

#     # reconstruction - drift
#     plotter.subplot(0,2)
#     u3_grid = get_scalar_grid(drift, algo+' (drift) - '+varname, V)
#     u3_grid.set_active_scalars(algo+' (drift) - '+varname)

#     dict_cb['title'] = algo+' (drift) - '+varname

#     plotter.add_mesh(u3_grid, cmap = colormap, clim = clim, show_edges=False, scalar_bar_args=dict_cb)
#     plotter.view_xy()
#     plotter.camera.zoom(zoom)
#     # plotter.add_title(varname, font_size=25, color ='k')
    
#     # plotter.set_background('white', top='white')
#     plotter.add_text(r'Drifted Sensor = '+str(sensI), color= 'k', position='lower_edge', font_size=25)

#     ## Save figure
#     # plotter.save_graphic(filename+'.pdf')
#     plotter.screenshot(filename+'.png', transparent_background = True,  window_size=resolution)