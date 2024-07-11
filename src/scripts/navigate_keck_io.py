#!/usr/bin/env python

import paths
import copy
import glob, os
import poppy

from pylanetary.navigation import Nav
from pylanetary.utils import Body, convolve_with_beam
from navigate_perturb import *

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib import patches

from astropy.io import fits
import astropy.units as u
from astroquery.jplhorizons import Horizons
from datetime import datetime, timedelta
from astropy.modeling import models, fitting
from astropy.table import QTable
from astropy.io import ascii

from scipy.interpolate import RegularGridInterpolator
import scipy.optimize as op
from image_registration.fft_tools.shift import shift2d

from photutils.detection import StarFinder, IRAFStarFinder, DAOStarFinder
from photutils.psf import FittableImageModel, EPSFModel, PSFPhotometry, IterativePSFPhotometry, SourceGrouper
from photutils.background import LocalBackground



def rayleigh_criterion(wl, d):
    '''
    inputs in meters
    returns in arcsec
    '''
    return np.rad2deg(1.22 * wl / d) * 3600
    
    
def latlon_interpf(lats):
    '''
    2-d cubic, curvature-minimizing interpolation for lat-lon grids
    see: scipy.interpolate.Regular2DInterpolator
    will only be valid when inputting x,y points with same shape as input lats (obviously)
    
    Parameters
    ----------
    lats: 2-D numpy array of latitudes or longitudes
    
    Returns
    -------
    interpf: interpolation function
    '''
    lats[np.isnan(lats)] = -999
    x = np.arange(lats.shape[0])
    y = np.arange(lats.shape[1])
    interp = RegularGridInterpolator((x, y), lats)
    return interp
    
    
def latlon_uncertainty(loc, err, lat_interp, lon_interp, npoints = 100, diagnostic_plot=False):
    '''
    Given 
    
    Parameters
    ----------
    loc : (x, y)
    err : (dx, dy)
    lat_interp : interpolation function from latlon_interpf
    lon_interp : interpolation function from latlon_interpf
    
    Returns
    -------
    (dlon, dlat) : uncertainty in lon, lat [deg]
    '''
    
    # assume dx, dy to be semimajor and semiminor axis of ellipse
    a = max([err[0], err[1]])
    b = min([err[0], err[1]])
    e = np.sqrt(1 - (a / b)**2)
    t = np.linspace(0, 2*np.pi, npoints)
    dxx, dyy = a*np.cos(t), b*np.sin(t)
    
    lats_out = []
    lons_out = []
    # project all ellipse points
    for i in range(len(t)):
        loci = (loc[0] + dxx[i], loc[1] + dyy[i])
        lats_out.append(lat_interp(loci))
        lons_out.append(lon_interp(loci))
    
    # take max and min to reproject onto lat-lon grid
    dlat = (np.max(lats_out) - np.min(lats_out))/2
    dlon = (np.max(lons_out) - np.min(lons_out))/2
    
    if diagnostic_plot:
        fig, ax = plt.subplots(1,1, figsize = (6,6))
        ax.scatter(lons_out, lats_out, color = 'k', marker = 'o')
        ax.errorbar(lon_interp(loc), lat_interp(loc), xerr=dlon, yerr=dlat, color = 'red', marker = 'o')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.show()
    
    return (dlat, dlon)
    
    
def ldmodel_op_func(theta, nav, psf):
    '''
    simple optimization function for optimizing limb-darkened disk model
    
    Parameters
    ----------
    theta : array-like, required
        [flux, a, dx, dy]
    nav : pylanetary.navigation.Nav object, required
    rms : float, required
        per-pixel RMS noise in image
    psf : np.array, required
        point-spread function
    
    Returns
    -------
    float
        absolute value of difference between data and model
    '''
    print(theta)
    [flux, a, dx, dy] = theta
    nav.mu0[nav.mu0 >= 1.0] = 1.0
    nav.mu[nav.mu >= 1.0] = 1.0
    nav.mu0[nav.mu0 <= 0.0] = 0.0
    nav.mu[nav.mu <= 0.0] = 0.0

    model = nav.ldmodel(flux, a, beam=psf, law='minnaert') #, psf_mode='airy')
    model = shift2d(model, dx, dy)
    #data = np.copy(nav.data)
    
    return np.sum(np.abs(nav.data - model))
    
    
def optimize_ldmodel(theta0, nav, psf):
    '''
    optimize the limb-darkened disk model
    '''
    bounds = [(0, np.inf), (0.0, 1.0), (-np.inf, np.inf), (-np.inf, np.inf)]
    res = op.minimize(ldmodel_op_func, theta0, args=(nav, psf), bounds=bounds, options={"maxiter":50000})
    thetaf = res.x
    
    return thetaf
    
    
def make_pointsource_model(data, psf, fwhm, thresh = 3., n=5, diagnostic_plot=True):
    '''
    Use Astropy tools to find brightest n point sources in field
    and optimize their brightness and location
    
    data : np.array, required
        2-D data
    psf : np.array, required.
        point-spread function kernel
    thresh : float, optional, default 3.
        threshold for star finder scaled to RMS noise of image
    n : int, optional, default 5
        max number of stars
    diagnostic_plot : bool, optional, default True
        show image of the stars found
    '''
    
    # find sources in field
    thresh = thresh*np.std(data)
    # why does this step fuck up the data when there are negative values?
    #star_finder = StarFinder(thresh, psf, brightest=n)
    star_finder = IRAFStarFinder(thresh, fwhm, brightest=n, minsep_fwhm=1.)
    sources = star_finder(data)
    print(sources)
    
    # put them into a star model
    psf_model = EPSFModel(psf, normalize=True, norm_radius = 50)
    
    # do PSF photometry
    s = int(fwhm)
    fit_shape = (s*5+1, s*5+1)
    grouper = SourceGrouper(fwhm)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=star_finder,
                            aperture_radius=s*5, grouper=grouper)
    phot = psfphot(data,)# error=error)
    model = psfphot.make_model_image(data.shape, fit_shape)
    resid = psfphot.make_residual_image(data, fit_shape)
    
    '''
    for i, src in enumerate(sources):
        star = copy.deepcopy(psf_model)
        #star.x_0 = src['xcentroid'] - data.shape[0]/2
        #star.y_0 = src['ycentroid'] - data.shape[1]/2
        #star.flux = src['flux'] #fudging here because for some reason fluxes are not realistic
        star.x_0 = phot['x_fit'][i]
        star.y_0 = phot['y_fit'][i]
        star.flux = phot['flux_fit'][i]
        #star.flux.min = 0.0
        if i == 0:
            combined_model = star
        else:
            combined_model += star
    '''
    
    # plot the solution and difference
    if diagnostic_plot:
        sz = data.shape[0]/2
        xx, yy = np.meshgrid(np.arange(-sz,sz), np.arange(-sz, sz))
        fig, (ax0, ax1, ax2) = plt.subplots(1,3, figsize = (16,6))
        ax0.imshow(data, origin = 'lower')
        ax0.set_title('Data')
        #ax1.imshow(combined_model(xx, yy), origin = 'lower')
        ax1.imshow(model, origin = 'lower')
        ax1.set_title('Point source model')
        #ax2.imshow(data - combined_model(xx, yy), origin='lower')
        ax2.imshow(resid, origin='lower')
        ax2.set_title('Difference')
        plt.show()
    
    return model
    

def fit_single_star(data, epsf, x0, y0, show_plot = True, fit_shape = (21, 21)):
    
    init_params = QTable()
    init_params['x'] = [y0]
    init_params['y'] = [x0]
    finder = DAOStarFinder(6.0, 2.0)
    psfphot = PSFPhotometry(epsf, fit_shape, finder=finder, aperture_radius=4)
    phot = psfphot(data, init_params=init_params)
    resid = psfphot.make_residual_image(data, fit_shape)
    
    if show_plot:
        model = psfphot.make_model_image(data.shape, fit_shape)
        fig, (ax0, ax1, ax2) = plt.subplots(1,3, figsize = (16,6))
        ax0.imshow(data, origin = 'lower')
        ax0.set_title('Data')
        ax1.imshow(model, origin = 'lower')
        ax1.set_title('Point source model')
        ax2.imshow(resid, origin='lower')
        ax2.set_title('Difference')
        plt.show()
        
    return phot, resid
  

def get_brightest(diff, epsf, show_plot = False):
    '''
    '''
    # get location and flux of brightest point in frame
    x0, y0 = np.unravel_index(np.argmax(diff), diff.shape)
    pointsrc_fit_table, resid = fit_single_star(diff, epsf, x0, y0, show_plot = show_plot)
    x_fit, y_fit = pointsrc_fit_table['x_fit'][0], pointsrc_fit_table['y_fit'][0]
    err_x, err_y = pointsrc_fit_table['x_err'][0], pointsrc_fit_table['y_err'][0]
    flux_fit = pointsrc_fit_table['flux_fit'][0]
    flux_err = pointsrc_fit_table['flux_err'][0]
    
    # translate to lat-lon grid, including uncertainty
    loc = (y_fit, x_fit)
    loc_err = (err_y, err_x)
    lat_interp = latlon_interpf(nav.lat_g)
    lon_interp = latlon_interpf(nav.lon_w)
    lat_f = lat_interp(loc)
    lon_f = lon_interp(loc)
    # for testing screw around with being near limb
    lat_err, lon_err = latlon_uncertainty(loc, loc_err, lat_interp, lon_interp, diagnostic_plot=False)
    
    return lat_f, lon_f, lat_err, lon_err, flux_fit, flux_err, resid
    
    

def aberrated_keck(pixscale_arcsec, wl, sz = 50, coeffs_in = [0.0, 0.0]):
    '''
    Make an Airy disk with spherical aberration to simulate Keck observations
    
    Parameters
    ----------
    pixscale_arcsec : float, required
        pixel scale of detector in arcsec
    wl : float, required
        wavelength in microns
    sz : int, optional, default 50
        size of output PSF kernel in pixels
    coeffs_in : list, optional, default [0.0, 0.0]
        first coeff is defocus
        second coeff is spherical aberration
        both in units of n wavelengths
    
    References
    ----------
    Table 2.3 of the below has the polynomials in same order as expected by code
    but note that code ignores piston
    https://wp.optics.arizona.edu/visualopticslab/wp-content/uploads/sites/52/2021/10/Zernike-Fit.pdf
    '''
    
    fov_arcsec = sz*pixscale_arcsec
    wl_meters = wl * 1e-6
    coeffs = np.zeros((11,))
    coeffs[3] = coeffs_in[0] * wl_meters
    coeffs[-1] = coeffs_in[1] * wl_meters
    
    osys = poppy.OpticalSystem()
    hex_aperture = poppy.MultiHexagonAperture(side=0.9, rings=3, gap=0.003, center=False) #side, gap both in meters. data from Mast & Nelson 1988 https://adsabs.harvard.edu/full/1988ESOC...30..411M
    osys.add_pupil(hex_aperture)
    thinlens = poppy.ZernikeWFE(radius=5.0, coefficients=coeffs)
    osys.add_pupil(thinlens)
    osys.add_detector(pixelscale=pixscale_arcsec, fov_arcsec=fov_arcsec, oversample=1)
    psf_with_zernikewfe = osys.calc_psf(wavelength=wl_meters, display_intermediates=False)
    
    psf_with_zernikewfe[0].header
    return psf_with_zernikewfe[0].data
    
    
def psf_plus_bkgd(coeffs_in, data_cropped, pixscale_arcsec, wl, oversample=1):
    '''
    Fit a PSF and background given telescope diameter and spherical aberration coeff
    '''
    # PSF 
    psf = aberrated_keck(pixscale_arcsec/oversample, wl, sz = data_cropped.shape[0], coeffs_in = coeffs_in)
    epsf = EPSFModel(psf, oversample=oversample)
    
    # add a low-order background
    p_init = epsf + models.Polynomial2D(degree=1)
    p_init.flux_0.min = 0.0
    p_init[0].origin = (0, 0)
    fit_p = fitting.LevMarLSQFitter()
    
    # find best fit to data given that set of coeffs
    y, x = np.mgrid[:data_cropped.shape[0], :data_cropped.shape[1]]
    #fig, (ax0, ax1) = plt.subplots(1,2, figsize = (12,6))
    #ax0.imshow(data_cropped, origin = 'lower')
    #ax1.imshow(p_init(x,y), origin = 'lower')
    #plt.show()
    p = fit_p(p_init, x, y, data_cropped, maxiter=1000)
    
    return p(x,y)
    
    
def psf_fit_func(theta, data_cropped, pixscale_arcsec, wl, oversample=1):
    '''
    simple wrapper to PSF maker for optimizing over the two coeffs
    '''
    print(theta)
    psf = psf_plus_bkgd(theta, data_cropped, pixscale_arcsec, wl, oversample=1)
    
    #fig, (ax0, ax1) = plt.subplots(1,2, figsize = (12,6))
    #ax0.imshow(data_cropped, origin = 'lower')
    #ax1.imshow(data_cropped - psf, origin = 'lower')
    #plt.show()
    
    return np.sum(np.abs(data_cropped - psf))
    
    
def plot_residuals(data_cropped, psf_f, outfile = None):
    
    fig, (ax0, ax1, ax2) = plt.subplots(1,3, figsize=(15, 6))
    cim0 = ax0.imshow(data_cropped, origin = 'lower')
    ax0.set_title('Cropped Data')
    cim1 = ax1.imshow(psf_f, origin = 'lower')
    ax1.set_title('PSF with background')
    cim2 = ax2.imshow(data_cropped - psf_f, origin = 'lower')
    ax2.set_title('Residual')
    
    cims = [cim0, cim1, cim2]
    for i, ax in enumerate([ax0, ax1, ax2]):
        cim = cims[i]
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="7%", pad="2%")
        cb = fig.colorbar(cim, cax=cax, orientation="vertical")
        ax.set_xticks([])
        ax.set_yticks([])
    
    if outfile is not None:
        fig.savefig(outfile, dpi=300)
    plt.show()
    
    
def disk_xyerr(nav, flux, a, psf, radius_fraction=0.1, diagnostic_plot=True, step=0.5, max_dist=10):
    '''
    Determine the best-fit x,y position of Io's limb-darkened disk in the image
    with uncertainty
    
    Parameters
    ----------
    nav : pylanetary.navigation.Nav object, required
    flux : float, required
        brightness of Io in image
    a : float, required
        limb darkening parameter
    psf : np.array, required
        point-spread function kernel
    radius_fraction : float, optional, default 0.05
        fraction of disk radius to mask
    diagnostic_plot : bool, optional, default True
        show diagnostic plot
    step : float, optional, default 0.5
        step size in pixels for grid search
    max_dist : float, optional, default 10
        maximum distance from initial guess to search
    '''
    # try centering using convolution with limb-darkened disk
    ldmodel = nav.ldmodel(flux, a, beam=psf, law='minnaert')
    dx, dy, _, _ = nav.colocate(mode='disk', 
            tb = flux, 
            a = a, 
            beam = psf, 
            sigma=5,
            low_thresh = 0.1,
            high_thresh = 0.5,
            diagnostic_plot=diagnostic_plot,
            )
    print(f'Initial guess dx, dy = {dx}, {dy}')
    
    # mask limb of planet
    masked_data = mask_planet_limb(nav, dx, dy, radius_fraction = radius_fraction, diagnostic_plot = diagnostic_plot)

    # iterate over a grid around the solution
    grid, std_surface, pearson_surface = correlate_ldmodel_on_grid(masked_data, ldmodel, dx, dy, step, max_dist)
    minstd_idx = np.unravel_index(np.argmin(std_surface), std_surface.shape)
    maxpearson_idx = np.unravel_index(np.argmax(pearson_surface), pearson_surface.shape)
    
    sigma_level = np.nanmin(std_surface)*1.5
    dxerr, dyerr = compute_xy_uncertainty_from_stdgrid(grid, std_surface, sigma_level)

    print('suggested x,y shift is ', dx + grid[minstd_idx[0]], dy + grid[minstd_idx[1]])
    print('x,y uncertainty is ', dxerr, dyerr)
    
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
    
    return dx, dy, dxerr, dyerr
    

def parse_filter_name(fname):
    
    wl_dict = {'l':3.776,
               'lp': 3.776,
               'm': 4.670,
               'ms': 4.670,
               'pah': 3.2904,
               'h2o': 3.0629,
               'kcont': 2.2706,
               'br_alpha': 4.052,
               'br_alpha_cont': 3.987,
               'brcont_alpha': 3.987,}
    
    filt_name = fname.split('/')[-1]
    filt_name = filt_name.strip('.fits')
    filt_name = filt_name.strip('stacked_nophot_')
    filt_name = filt_name.strip('_GWsrmu')
    filt_name = filt_name.strip('Io')
    filt_name = filt_name.strip('_N')
    filt_name = filt_name.lower()
    if filt_name == 'l':
        filt_name = 'lp'
    elif filt_name == 'm':
        filt_name = 'ms'
    elif filt_name == 'brcont_alpha':
        filt_name = 'br_alpha_cont'
    
    return wl_dict[filt_name], filt_name

def subtract_background(data, sz = 4):
    
    sz = int(data.shape[0]/sz)
    noise_region = np.concatenate([
                    data[:sz, :sz],
                    data[:sz, -sz:],
                    data[-sz:, :sz],
                    data[-sz:, -sz:]
                    ])
    bkgd_mean = np.mean(noise_region)
    rms = np.std(noise_region) 
    per_beam_rms = rms * np.sqrt(beam_area)
    out = data - bkgd_mean
    
    return out, rms, per_beam_rms


def make_optimized_keck_psf(data, wl, pixscale_arcsec, oversample=1, show_plot = False):
    '''
    Put it all together, making assumptions about Keck and the data
    in order to optimize the PSF at multiple wavelengths
    
    "we model PSF as Airy disk plus first spherically-symmetric Zernicke aberration"
    "performing this experiment on the three brightest sources in the field gives agreement?"
    if they agree, then this is probably defensible
    '''
    crop = 20
    max_index = np.unravel_index(np.argmax(data), data.shape)
    data_cropped = data[max_index[0]-crop:max_index[0]+crop, max_index[1]-crop:max_index[1]+crop]
    
    # find optimal PSF, changing first two spherically symmetric aberrations
    theta0 = [0.1, 0.0]
    res = op.minimize(psf_fit_func, theta0, 
                args=(data_cropped, pixscale_arcsec, wl), 
                bounds = [(0.0, 0.2),(-0.2, 0.2)], 
                options={"maxiter":50000})
    thetaf = res.x
    
    if show_plot:
        psf_f = psf_plus_bkgd(thetaf, data_cropped, pixscale_arcsec, wl, oversample=1)
        plot_residuals(data_cropped, psf_f, outfile = 'diagnostic_plots/emakong_fitted_psf_keck.png')
    
    return thetaf
    
if __name__ == "__main__":
    
    # test_date = '2022sep11'
    # infile = paths.data / 'keck' / date / 'reduced/stacked_nophot_lp.fits'
    # outstem = paths.data / 'keck' / date / 'reduced/lp'

    pixscale_arcsec = 0.009971 #arcsec, keck
    req = 1821.3
    rpol = req
    d = 10.0 #meters
    show_reprojection = False
    redo_all = False
    
    dates = ['2022jul09', '2022jul26', '2022jul27', '2022aug15', '2022aug16', '2022sep06', '2022sep07', '2022sep11', '2022sep12']
    calibrated = [False, False, False, True, True, True, True, False, True]
    rotated_up = [False]*3 + [True]*6
    filters_to_exclude = ['h2o', 'kcont', 'pah']
    
    #dates = dates[4:]
    #calibrated = calibrated[4:]
    #rotated_up = rotated_up[4:]
    dates = [dates[2]]
    calibrated = [calibrated[2]]
    rotated_up = [rotated_up[2]]
    redo_all = True
    
    for i, date in enumerate(dates):
        fnames = glob.glob(str(paths.data / 'keck' / date / '*.fits'))
        print(f'starting {date}...')
        if calibrated[i]:
            flux = 1.0
            a = 0.702
        else:
            flux = 1.131e4
            a = 0.702

        for fname in fnames:
            wl, filt_name = parse_filter_name(fname)
            if filt_name in filters_to_exclude:
                continue
            
            print(f'Starting filter {filt_name} ({wl}) for {date}')
            
            # compute some basic wavelength-dependent parameters
            rayleigh = rayleigh_criterion(wl*1.0e-6, d) / pixscale_arcsec #pixels
            fwhm = rayleigh * (1.025 / 1.22) 
            beam_area = (np.pi / (4 * np.log(2))) * fwhm**2
            
            # load the data
            hdul = fits.open(fname)
            header = hdul[0].header
            data = hdul[0].data
            obs_time = header['DATE-OBS'] + ' ' + header['EXPSTART'][:-4]
            outstem = paths.data / 'keck' / date
            
            # imke doesn't believe in NaNs so this is needed
            data[data > 1e10] = 0.0
            
            # calculate RMS of data and subtract slight negative background
            data, rms, per_beam_rms = subtract_background(data)
            
            # make the PSF. takes time, so run it only once
            if (not os.path.exists(outstem / f'{filt_name}_psf_optimized.npy')) or (redo_all):
                thetaf = make_optimized_keck_psf(data, wl, pixscale_arcsec, oversample=1, show_plot = True)
                np.save(outstem / f'{filt_name}_psf_optimized.npy', thetaf)
            thetaf = np.load(outstem / f'{filt_name}_psf_optimized.npy')
            psf = aberrated_keck(pixscale_arcsec, wl, sz = 50, coeffs_in = [thetaf[0], thetaf[1]])
            epsf = EPSFModel(psf, oversample=1)
            
            # instantiate the nav object
            io = Body('Io', epoch=obs_time, location='568') #Maunakea keyword is 568
            if rotated_up[i]:
                io.ephem['NPole_ang'] = 0.0
            nav = Nav(data, io, pixscale_arcsec)
            
            # figure out limb darkening parameters using optimizer
            if (not os.path.exists(outstem / f'{filt_name}_ldmodel_optimized.npy')) or (redo_all):
                theta0 = [flux, a, 0.0, 0.0]
                thetaf = optimize_ldmodel(theta0, nav, psf)
                np.save(outstem / f'{filt_name}_ldmodel_optimized.npy', thetaf)
            ldmodel_thetaf = np.load(outstem / f'{filt_name}_ldmodel_optimized.npy')
            flux, a, dx, dy = ldmodel_thetaf
            
            # colocate
            # do not compute with error functions because it depends on too many assumptions
            # ie.e we are just assuming the disk is found perfectly
            #dx, dy, dxerr, dyerr = disk_xyerr(nav, flux,a, psf,  diagnostic_plot=True, radius_fraction=0.05,step=0.5, max_dist=10)
            dx0, dy0, _, _ = nav.colocate(
                mode='disk', 
                tb = flux, 
                a = a, 
                beam = psf,
                law='minnaert', 
                err=rms, 
                diagnostic_plot=False)
            nav.xy_shift_model(dx, dy)
            
            nav.mu0[nav.mu0 >= 1.0] = 1.0
            nav.mu[nav.mu >= 1.0] = 1.0
            nav.mu0[nav.mu0 <= 0.0] = 0.0
            nav.mu[nav.mu <= 0.0] = 0.0
            
            # subtract optimized limb-darkened disk model from data
            ldmodel = nav.ldmodel(flux, a, beam=psf, law='minnaert', ) #psf_mode='airy'
            diff = nav.data - ldmodel
            
            # extract the brightest point source brightness and location
            try:
                lat_f, lon_f, lat_err, lon_err, flux_fit, flux_err, resid = get_brightest(diff, epsf, show_plot = False)
            except ValueError:
                print('No point sources found')
                lat_f, lon_f, lat_err, lon_err, flux_fit, flux_err = -999, -999, 0.0, 0.0, 0.0, 0.0
            print(f'Lat/lon of brightest: {lat_f:.2f} +/- {lat_err:.2f}, {lon_f:.2f} +/- {lon_err:.2f}')
            print(f'Flux of brightest: {flux_fit:.2f} +/- {flux_err:.2f}')
            
            try:
                lat_f2, lon_f2, lat_err2, lon_err2, flux_fit2, flux_err2, resid2 = get_brightest(resid, epsf, show_plot = False)
            except ValueError:
                print('Only one point source found')
                lat_f2, lon_f2, lat_err2, lon_err2, flux_fit2, flux_err2 = -999, -999, 0.0, 0.0, 0.0, 0.0
            print(f'Lat/lon of second-brightest: {lat_f2:.2f} +/- {lat_err2:.2f}, {lon_f2:.2f} +/- {lon_err2:.2f}')
            print(f'Flux of second-brightest: {flux_fit2:.2f} +/- {flux_err2:.2f}')
            
            try:
                lat_f3, lon_f3, lat_err3, lon_err3, flux_fit3, flux_err3, resid3 = get_brightest(resid2, epsf, show_plot = False)
            except ValueError:
                print('Only two point sources found')
                lat_f3, lon_f3, lat_err3, lon_err3, flux_fit3, flux_err3 = -999, -999, 0.0, 0.0, 0.0, 0.0
            print(f'Lat/lon of third-brightest: {lat_f3:.2f} +/- {lat_err3:.2f}, {lon_f3:.2f} +/- {lon_err3:.2f}')
            print(f'Flux of third-brightest: {flux_fit3:.2f} +/- {flux_err3:.2f}')
            
            # plot the result
            if show_reprojection:
                nav.data = diff
                proj, mu_proj = nav.reproject()
                fig, ax = plt.subplots(1,1, figsize = (10,10))
                ax.imshow(proj, origin = 'lower', extent = [0, 360, -90, 90], cmap = 'Greys_r')
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                ax.set_title(f'{date} {filt_name}')
                ax.scatter(lon_f, lat_f, color = 'red', marker = 'o')
                ax.scatter(lon_f2, lat_f2, color = 'red', marker = 'o')
                ax.scatter(lon_f3, lat_f3, color = 'red', marker = 'o')
                plt.show()
            
            # save the results as an astropy QTable
            out_table = QTable()
            out_table['legend'] = ['date', 'filter', 'wavelength', 'flux', 'flux_err', 'lat', 'lat_err', 'lon', 'lon_err']
            out_table['source1'] = [date, filt_name, wl, flux_fit, flux_err, lat_f, lat_err, lon_f, lon_err]
            out_table['source2'] = [date, filt_name, wl, flux_fit2, flux_err2, lat_f2, lat_err2, lon_f2, lon_err2]
            out_table['source3'] = [date, filt_name, wl, flux_fit3, flux_err3, lat_f3, lat_err3, lon_f3, lon_err3]
            ascii.write(out_table, outstem / f'{filt_name}_photometry.txt', overwrite=True)
            
            
