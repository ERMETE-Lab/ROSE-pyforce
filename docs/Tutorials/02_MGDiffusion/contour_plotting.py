import vtk
import pyvista as pv
from matplotlib import cm
import numpy as np

import dolfinx
from dolfinx.fem import Function

from pyforce.offline.geim import GEIM
from pyforce.tools.functions_list import FunctionsList
from pyforce.tools.plotting import grids

def plot_FOM_vs_ROM( fom: FunctionsList, rom: dict, mu: int, title: str, varname: str,
                    clim = None, mag_plot = True, colormap = cm.jet, 
                    colormap_res = cm.plasma_r, clim_res = None,
                    filename = None, _resolution = [900, 400],
                    factor : float = 0.05, tolerance : float = 0.01,
                    zoom = 2.25, position_cb = [0.125, 0.745]): 
    
    keys = list(rom.keys())
    
    resolution = [_resolution[0] * 2, _resolution[1] * ( len(keys) + 1)]
    plotter = pv.Plotter(shape=(len(keys)+1, 2), off_screen=False, border=False, window_size=resolution)
    
    lab_fontsize = 25
    title_fontsize = 35
    
    dict_cb = dict(title = varname, width = 0.75, height = 0.2,
                    title_font_size=title_fontsize,
                    label_font_size=lab_fontsize,
                    n_labels=3,
                    color = 'k',
                    position_x=position_cb[0], position_y=position_cb[1],
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
        
def plot_modes( modes: FunctionsList, varname: str, shape : list,
                mag_plot : bool = True, colormap = cm.jet,
                factor : float = 0.03, tolerance : float = 0.01,
                subfig_size = [800, 750], zoom = 1.15,
                lab_fontsize = 25, title_fontsize = 35,
                title = 'Basis Function'):  
    
    nrows = shape[0]
    ncols = shape[1]
    
    resolution = [subfig_size[0] * ncols, subfig_size[1] * nrows]
    plotter = pv.Plotter(shape=shape, off_screen=False, border=False, window_size=resolution)
    
    dict_cb = dict(title = varname, width = 0.75, height = 0.2,
                    title_font_size=title_fontsize,
                    label_font_size=lab_fontsize,
                    n_labels=3,
                    color = 'k',
                    position_x=0.125, position_y=0.825,
                    shadow=False) 
    
    if modes.fun_space.num_sub_spaces > 0:
        vector_plot = True
    else:
        vector_plot = False
    
    idx = 0
    for row in range(nrows):
        for col in range(ncols):
            if vector_plot:
                
                if mag_plot:
                    warped, values = grids(modes.map(idx), mag_plot=mag_plot, varname=varname)
                else:
                    warped, values = grids(modes.map(idx), mag_plot=True, varname=varname)
                    glyph, _ = grids(modes.map(idx), mag_plot=False, varname=varname, factor=factor, tolerance=tolerance)
                clim = [0., max(np.sqrt(values[:, 0]**2+values[:, 1]**2+values[:, 2]**2))]
            else:
                warped, _ = grids(modes.map(idx))
                clim = [min(modes(idx)), max(modes(idx))]
                
            plotter.subplot(row, col)
            dict_cb['title'] = '$'+varname+'$'+' - '+title+' '+str(idx+1)
            plotter.add_mesh(warped, clim = clim, cmap = colormap, show_edges=False, scalar_bar_args=dict_cb)
            
            if vector_plot and mag_plot == False:
                plotter.add_mesh(glyph, color='k', opacity=0.75)
            
            plotter.view_xy()
            plotter.camera.zoom(zoom)
            
            idx += 1

    plotter.set_background('white', top='white')
    plotter.show(jupyter_backend='static')
    

def sens_pos(sens: FunctionsList, domain, M):
    x_dofs = domain.geometry.x
    
    x_sens = list()
    for mm in range(M):
        x_sens.append(x_dofs[np.argmax(sens(mm))])
    return np.asarray(x_sens)

def plot_geim_sensors(  geim: list[GEIM], m_sens_to_plot: int, algo_names: list, field_i: int,
                        cmap = None, resolution = [2800, 800]):

    plotter = pv.Plotter(shape=(1,len(algo_names)), off_screen=False, border=False, window_size=resolution)
    lab_fontsize = 25
    title_fontsize = 35
    zoom = 1.25

    if cmap is None:
        map1 = cm.hot
    else: 
        map1 = cmap
        
    dict_cb = dict(title = ' ', 
                   width = 0.76,
                    title_font_size=title_fontsize,
                    label_font_size=lab_fontsize,
                    color='k',
                    position_x=0.12, position_y=0.9,
                    shadow=True, 
                    ) 

    for jj in range(len(algo_names)):
        sens_to_plot = Function(geim[jj].magic_sens.fun_space)
        
        for mm in range(m_sens_to_plot):
            sens_to_plot.x.array[:] += geim[jj].magic_sens(mm) / max(geim[jj].magic_sens(mm))
        
        plotter.subplot(0,jj)
        clim = [0.,  max(sens_to_plot.x.array)]
        dict_cb['title'] = r'Sensors - '+algo_names[jj]
        plotter.add_mesh(grids(sens_to_plot)[0], cmap = map1, clim = clim, show_edges=False, scalar_bar_args=dict_cb)    
        
        sgreedy_sens = sens_pos(geim[jj].magic_sens, geim[jj].V.mesh, m_sens_to_plot)
        # plotter.add_points(geim_sens, render_points_as_spheres=True, point_size=20, color='black')
        plotter.add_point_labels(sgreedy_sens, np.arange(1, m_sens_to_plot+1,1),italic=False,
                                font_size=35,
                                point_color='white',
                                shape_color='white',
                                shape_opacity=0.85,
                                point_size=1,
                                render_points_as_spheres=True,
                                always_visible=True,
                                shadow=True)
        
        plotter.view_xy()
        plotter.camera.zoom(zoom)
        plotter.remove_scalar_bar()
        plotter.add_text(r'$s = {:.2f}'.format(geim[jj].sens_class.s)+'$',
                            color= 'k', position='upper_edge', font_size=25)
    
    plotter.add_text(r'Sensors for flux $\phi_'+str(field_i+1)+'$',
                        color= 'k', position='lower_edge', font_size=25)

    plotter.set_background('white', top='white')
    
    plotter.show(jupyter_backend='static')
    
def extract_cells(domain, points):

    bb_tree = dolfinx.geometry.BoundingBoxTree(domain, domain.topology.dim)
    cells = []
    points_on_proc = []
    cell_candidates = dolfinx.geometry.compute_collisions(bb_tree, points.T)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i))>0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
    xPlot = np.array(points_on_proc, dtype=np.float64)

    return xPlot, cells