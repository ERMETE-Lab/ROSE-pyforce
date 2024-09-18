import vtk
import pyvista as pv
import dolfinx
from matplotlib import cm
import numpy as np

from pyforce.tools.functions_list import FunctionsList
from pyforce.tools.plotting import grids

def plot_FOM_vs_ROM( fom: FunctionsList, rom: dict, mu: int, title: str, varname: str,
                    clim = None, mag_plot = True, colormap = cm.jet, 
                    colormap_res = cm.plasma_r, clim_res = None,
                    filename = None, 
                    factor : float = 0.05, tolerance : float = 0.01): 
    
    keys = list(rom.keys())
    
    resolution = [1800, 400 * ( len(keys) + 1)]
    plotter = pv.Plotter(shape=(len(keys)+1, 2), off_screen=False, border=False, window_size=resolution)
    
    lab_fontsize = 25
    title_fontsize = 35
    zoom = 2.25
    
    dict_cb = dict(title = varname, width = 0.75, height = 0.2,
                    title_font_size=title_fontsize,
                    label_font_size=lab_fontsize,
                    n_labels=3,
                    color = 'k',
                    position_x=0.125, position_y=0.745,
                    shadow=False) 
    
    if fom.fun_space.num_sub_spaces > 0:
        vector_plot = True
    else:
        vector_plot = False
    
    ############################ FOMs ###################################

    plotter.subplot(0,0)
    
    if vector_plot:
        if mag_plot:
            warped_fom, values_fom = grids(fom.map(mu), mag_plot=mag_plot, varname=varname)
        else:
            warped_fom, values_fom = grids(fom.map(mu), mag_plot=True, varname=varname)
            glyph_fom, _ = grids(fom.map(mu), mag_plot=False, varname=varname, factor=factor, tolerance=tolerance)
        
        if clim is None:
            clim = [0., max(np.sqrt(values_fom[:, 0]**2+values_fom[:, 1]**2+values_fom[:, 2]**2))]
    else:
        warped_fom, _ = grids(fom.map(mu))
        if clim is None:
            clim = [min(fom(mu)), max(fom(mu))]
        
    dict_cb['title'] = 'FOM - $'+varname+'$'
    plotter.add_mesh(warped_fom, clim = clim, cmap = colormap, show_edges=False, scalar_bar_args=dict_cb)
    
    if vector_plot and mag_plot == False:
        plotter.add_mesh(glyph_fom, color='k', opacity=0.75)
                
    plotter.view_xy()
    plotter.camera.zoom(zoom)

    ############################ ROMs ###################################


    for key_i in range(len(keys)):
        plotter.subplot(1+key_i,0)
        
        if vector_plot:
            if mag_plot:
                warped_rom, _ = grids(rom[keys[key_i]].map(mu), mag_plot=mag_plot, varname=varname)
            else:
                warped_rom, _ = grids(rom[keys[key_i]].map(mu), mag_plot=True, varname=varname)
                glyph_rom,  _ = grids(rom[keys[key_i]].map(mu), mag_plot=False, varname=varname, factor=factor, tolerance=tolerance)
            
        else:
            warped_rom, _ = grids(rom[keys[key_i]].map(mu))
        
        dict_cb['title'] = keys[key_i]+' - $'+varname+'$'
        plotter.add_mesh(warped_rom, clim = clim, cmap = colormap, show_edges=False, scalar_bar_args=dict_cb)
        
        if vector_plot and mag_plot == False:
                plotter.add_mesh(glyph_rom, color='k', opacity=0.75)
                
        plotter.view_xy()
        plotter.camera.zoom(zoom)

    ############################ Residuals ###################################

    max_res = 0.
    for key_i in range(len(keys)):
        plotter.subplot(1+key_i,1)
        
        residual = dolfinx.fem.Function(rom[keys[key_i]].fun_space)
        residual.x.array[:] = np.abs(rom[keys[key_i]](mu) - fom(mu))

        max_res = max([max_res, max(residual.x.array[:])])
        
        if vector_plot:
            warped_res, _ = grids(residual, mag_plot=True, varname=varname)
        else:
            warped_res, _ = grids(residual)
        
        if clim_res is None:
            clim_res = [0, max_res]
        
        dict_cb['title'] = 'Residual '+keys[key_i]+' - $'+varname+'$'
        plotter.add_mesh(warped_res, clim = clim_res, cmap = colormap_res, show_edges=False, scalar_bar_args=dict_cb)
        plotter.view_xy()
        plotter.camera.zoom(zoom)


    plotter.set_background('white', top='white')
    plotter.subplot(0,1)
    plotter.add_text(str(title), color= 'k', position=[250, 200], font_size=30)
    
    if filename is None:
        plotter.show(jupyter_backend='static')
    else:
        ## Save figure
        plotter.screenshot(filename+'.png', transparent_background = True,  window_size=resolution)
        plotter.close()
        
def plot_modes( pod_modes: FunctionsList, varname: str, shape : list,
                mag_plot : bool = True, colormap = cm.jet,
                factor : float = 0.03, tolerance : float = 0.01):  
    
    nrows = shape[0]
    ncols = shape[1]
    
    resolution = [900 * ncols, 400 * nrows]
    plotter = pv.Plotter(shape=shape, off_screen=False, border=False, window_size=resolution)
    
    lab_fontsize = 25
    title_fontsize = 35
    zoom = 2.25
    
    dict_cb = dict(title = varname, width = 0.75, height = 0.2,
                    title_font_size=title_fontsize,
                    label_font_size=lab_fontsize,
                    n_labels=3,
                    color = 'k',
                    position_x=0.125, position_y=0.745,
                    shadow=False) 
    
    if pod_modes.fun_space.num_sub_spaces > 0:
        vector_plot = True
    else:
        vector_plot = False
    
    idx = 0
    for row in range(nrows):
        for col in range(ncols):
            if vector_plot:
                
                if mag_plot:
                    warped, values = grids(pod_modes.map(idx), mag_plot=mag_plot, varname=varname)
                else:
                    warped, values = grids(pod_modes.map(idx), mag_plot=True, varname=varname)
                    glyph, _ = grids(pod_modes.map(idx), mag_plot=False, varname=varname, factor=factor, tolerance=tolerance)
                clim = [0., max(np.sqrt(values[:, 0]**2+values[:, 1]**2+values[:, 2]**2))]
            else:
                warped, _ = grids(pod_modes.map(idx))
                clim = [min(pod_modes(idx)), max(pod_modes(idx))]
                
            plotter.subplot(row, col)
            dict_cb['title'] = '$'+varname+'$'+' - POD mode '+str(idx+1)
            plotter.add_mesh(warped, clim = clim, cmap = colormap, show_edges=False, scalar_bar_args=dict_cb)
            
            if vector_plot and mag_plot == False:
                plotter.add_mesh(glyph, color='k', opacity=0.75)
            
            plotter.view_xy()
            plotter.camera.zoom(zoom)
            
            idx += 1

    plotter.set_background('white', top='white')
    plotter.show(jupyter_backend='static')