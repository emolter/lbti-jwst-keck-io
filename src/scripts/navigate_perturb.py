#!/usr/bin/env python

'''
Using Mike's Jupiter dataset as a test, write
workflow for perturbing best solution and evaluating data minus model
in an annulus near the planet's edge. 
'''

from pylanetary.navigation import *
from pylanetary.utils import Body
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import numpy.ma as ma
import astropy.units as u
from astropy.io import fits
from astropy.time import Time
from photutils.aperture import EllipticalAnnulus
from photutils.aperture import ApertureStats
from image_registration.fft_tools.shift import shift2d
from scipy.stats.mstats import pearsonr


def correlate_ldmodel_on_grid(masked_data, ldmodel, dx_init, dy_init, step, max_dist):
    '''
    compute statistical metrics of goodness of fit of limb-darkened
    model to data over a grid
    
    Parameters
    ----------
    
    '''
    
    grid = np.arange(-max_dist, max_dist+step,step)
    std_surface = np.empty((grid.size, grid.size))
    pearson_surface = np.empty((grid.size, grid.size))
    
    for i in range(grid.size):
        q = grid[i]
        print(f'Computing goodness-of-fit on grid {i+1} of {grid.size}')
        for j in range(grid.size):
            w = grid[j]
            model = shift2d(ldmodel, dx_init + q, dy_init + w)
            std = (model/masked_data).std()
            pearson, _ = pearsonr(model, masked_data)
            std_surface[i,j] = std
            pearson_surface[i,j] = pearson
    
    return grid, std_surface, pearson_surface
    
    
def compute_xy_uncertainty_from_stdgrid(grid, data, sigma_level):
    '''
    Given a 2-D error space and a sigma contour,
    compute the approximate x, y error to report as the
    overall limb fitting uncertainty
    
    Parameters
    ----------
    '''
    min_idx = np.unravel_index(np.argmin(data), data.shape)
    xx, yy = np.meshgrid(grid, grid)
    inside = data < sigma_level
    inside_x = ma.array(xx, mask=~inside)
    inside_y = ma.array(yy, mask=~inside)
    
    xerrh = inside_x.max() - min_idx[1]
    xerrl = min_idx[1] - inside_x.min()
    yerrh = inside_y.max() - min_idx[0]
    yerrl = min_idx[0] - inside_y.min()
    
    return np.mean([xerrh, xerrl]), np.mean([yerrh, yerrl])

    
def mask_planet_limb(nav, dx, dy, radius_fraction = 0.07, diagnostic_plot = False):
    '''
    Make an EllipticalAnnulus object at a given x,y location
    that surrounds the limb of the planet out to +/- radius_fraction
    Turn that into a mask, and mask the data
    
    Parameters
    ----------
    
    Returns
    -------
    numpy.ma.MaskedArray
        the masked data
    '''
    
    # make an elliptical annulus around the planet's limb
    xcen, ycen = nav.data.shape[0]/2, nav.data.shape[1]/2
    a_out = (nav.req / nav.pixscale_km) * (1+radius_fraction)
    b_out = (nav.rpol / nav.pixscale_km) * (1+radius_fraction)
    a_in = (nav.req / nav.pixscale_km) * (1-radius_fraction)
    theta = np.deg2rad(nav.ephem['NPole_ang'])
    ann = EllipticalAnnulus((xcen + dx, ycen + dy), a_in, a_out, b_out, theta=theta)

    # make a mask out of this annulus
    ann_mask = ann.to_mask(method='center').to_image(nav.data.shape).astype(bool)
    ann_mask = ~ann_mask
    nonan_data = np.copy(nav.data)
    nonan_data[np.isnan(nonan_data)] = 0.0
    masked_data = ma.array(nonan_data, mask=ann_mask)
    
    # plot masked data to ensure working ok
    if diagnostic_plot:
        
        fig, ax = plt.subplots(1,1, figsize = (12, 12))
        im = ax.imshow(masked_data, origin = 'lower')
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Pixel Value')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical', label='flux')

        plt.show()
        plt.close()
    
    return masked_data


def plot_statistical_metric_grid(fig, ax, grid, data, label='', extremum = None, extremum_label = '', contour_level = None, contour_label = ''):
    '''
    
    Parameters
    ----------
    extremum : tuple or None, optional
    '''
    
    im = ax.contourf(grid, grid, data, levels=25)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('bottom', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='horizontal', label=label)
    legend_elements = []
    if extremum is not None:
        ax.scatter(extremum[0], extremum[1], color='red', marker='o')
        legend_elements.append(Line2D([0], [0], color='red', marker='o', label=extremum_label))
    if contour_level is not None:
        ax.contour(grid, grid, data, levels=[contour_level], colors=['red',], linestyles=[':',], linewidths=[3,])
        legend_elements.append(Line2D([0], [0], color='red', lw=3, label=contour_label, linestyle = ':'))
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 
    ax.set_xlabel('X Pixel')
    ax.legend(handles = legend_elements)
    
    

if __name__ == "__main__":
    
    # hst parameters
    flux = 1.3e4 # surface brightness in whatever units are in the fits file
    a = 0.9945 # minnaert limb-darkening law exponent
    fwhm = 2 # approximate FWHM of the point-spread function in pixels
    hdul = fits.open('hlsp_wfcj_hst_wfc3-uvis_jupiter-2017-pj07_f631n_v2_0711ut0947-nav.fits')
    data = hdul[1].data
    obs_time = hdul[0].header['DATE-OBS']+' '+hdul[0].header['TIME-OBS']
    rotation = float(hdul[0].header['ORIENTAT'])
    pixscale_arcsec = float(hdul[0].header['PIXSCAL'])
    
    # instantiate the nav object
    jup = Body('Jupiter', epoch=obs_time, location='@hst') #Keck
    jup.ephem['NPole_ang'] = jup.ephem['NPole_ang'] - rotation
    nav = Nav(data, jup, pixscale_arcsec)

    # start of function for Nav
    
    # try centering using convolution with limb-darkened disk
    ldmodel = nav.ldmodel(flux, a, beam=fwhm, law='minnaert')
    dx, dy, dxerr, dyerr = nav.colocate(mode='canny', 
            tb = flux, 
            a = a, 
            beam = fwhm, 
            sigma=5,
            low_thresh = 0.1,
            high_thresh = 0.5,
            diagnostic_plot=False,
            )
            
    print(f'Initial guess dx, dy = {dx}, {dy}')
    #model_shifted = shift2d(ldmodel, dx, dy)

    # mask limb of planet
    masked_data = mask_planet_limb(nav, dx, dy, radius_fraction = 0.05, diagnostic_plot = False)

    # iterate over a grid around the solution
    step = 0.5
    max_dist = 10
    grid, std_surface, pearson_surface = correlate_ldmodel_on_grid(masked_data, ldmodel, dx, dy, step, max_dist)

    minstd_idx = np.unravel_index(np.argmin(std_surface), std_surface.shape)
    maxpearson_idx = np.unravel_index(np.argmax(pearson_surface), pearson_surface.shape)
    
    diagnostic_plot = True
    if diagnostic_plot:
        
        fig, (ax0, ax1) = plt.subplots(1,2,figsize = (16,7))
    
        plot_statistical_metric_grid(fig, ax0, grid, std_surface, 
                    label='Standard Deviation', 
                    extremum = (grid[minstd_idx[1]], grid[minstd_idx[0]]), 
                    extremum_label = 'Minimum std', 
                    contour_level = np.min(std_surface)*2,
                    contour_label = 'Uncertainty Region')
                    
        plot_statistical_metric_grid(fig, ax1, grid, pearson_surface, 
                    label='Pearson R-value', 
                    extremum = (grid[maxpearson_idx[1]], grid[maxpearson_idx[0]]), 
                    extremum_label = 'Maximum Correlation', 
                    )
        ax0.set_ylabel('Y Pixel')
        plt.show()
        plt.close()

    sigma_level = np.nanmin(std_surface)*1.5
    dxerr, dyerr = compute_xy_uncertainty_from_stdgrid(grid, std_surface, sigma_level)

    print('suggested x,y shift is ', dx + grid[minstd_idx[0]], dy + grid[minstd_idx[1]])
    print('x,y uncertainty is ', dxerr, dyerr)

    # end of function for Nav

    model_shifted = shift2d(ldmodel, dx+ grid[minstd_idx[0]], dy + grid[minstd_idx[1]])
    #model_shifted = shift2d(ldmodel, dx, dy)
    fig, ax = plt.subplots(1,1, figsize = (12, 12))
    im = ax.imshow(nav.data - model_shifted, origin = 'lower')
    ax.set_title('Model / Data')
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Pixel Value')
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical', label='flux')
    
    plt.show()
    plt.close()


