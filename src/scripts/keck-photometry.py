#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import ascii
import paths
import glob
import os

def parse_filter_name(filt_name):
    
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
    
    filt_name = filt_name.lower()
    if filt_name == 'l':
        filt_name = 'lp'
    elif filt_name == 'm':
        filt_name = 'ms'
    elif filt_name == 'brcont_alpha':
        filt_name = 'br_alpha_cont'
    
    return wl_dict[filt_name], filt_name

def load_tables(dates):
    
    fluxes = []
    flux_errs = []
    wls = []
    all_dates = []
    lats = []
    lons = []
    for i, date in enumerate(dates):
        phot_tables = glob.glob(str(paths.data / 'keck' / date / '*_photometry.txt'))
        for tab in phot_tables:
            data = np.loadtxt(tab, skiprows=3, usecols=[1, 2, 3])
            wls += list(data[0].flatten())
            fluxes += list(data[1].flatten())
            flux_errs += list(data[2].flatten())
            lats += list(data[-4].flatten())
            lons += list(data[-2].flatten())
            all_dates += [date,]*len(data[0])
            
    return np.array(all_dates), np.array(wls), np.array(fluxes), np.array(flux_errs), np.array(lats), np.array(lons)
        
        
def load_lddisk_solutions(dates):
    
    fluxes = []
    wls = []
    avals = []
    all_dates = []
    for i, date in enumerate(dates):
        outstem = paths.data / 'keck' / date
        ldmodel_dsets = glob.glob(str(paths.data / 'keck' / date / '*_ldmodel_optimized.npy'))
        for dset in ldmodel_dsets:

            ldmodel_thetaf = np.load(dset)
            flux, a, dx, dy = ldmodel_thetaf
            filt_name = dset.split('/')[-1]
            filt_name = filt_name[:-22]
            wl, filt_name = parse_filter_name(filt_name)
            wls.append(wl)
            fluxes.append(flux)
            avals.append(a)
            all_dates.append(date)
    
    return np.array(all_dates), np.array(wls), np.array(fluxes), np.array(avals)
    

def determine_calibrated_fluxes(calibrated_dates):
    '''
    Parameters
    ----------
    calibrated_dates : list of str
    
    Returns
    -------
    wls : array of float
        Wavelengths of the calibrated fluxes
    fluxes_out : array of float
        Calibrated fluxes
    flux_errs_out : array of float
        Fractional uncertainties on the calibrated fluxes
    '''
    
    _, wls_cal, fluxes_cal, avals_cal = load_lddisk_solutions(calibrated_dates)
    wls = np.unique(wls_cal)
    fluxes_out = np.zeros(len(wls))
    flux_errs_out = np.zeros(len(wls))
    for i, wl in enumerate(wls):
        
        cal_truth = wls_cal == wl
        fluxes_cal_at_wl = fluxes_cal[cal_truth]
        avals_cal_at_wl = avals_cal[cal_truth]
        
        print(wl)
        print(fluxes_cal_at_wl)
        
        fluxes_out[i] = np.mean(fluxes_cal_at_wl)
        flux_errs_out[i] = np.std(fluxes_cal_at_wl)
    
    return wls, fluxes_out, flux_errs_out/fluxes_out


def determine_flux_rescale_factor(uncal_dates, cal_wls, cal_fluxes, cal_frac_err):
    '''
    Parameters
    ----------
    uncal_dates : list of str
    cal_wls : array of float
    cal_fluxes : array of float
    cal_frac_err : array of float
    
    Returns
    -------
    all_uncal_dates : list of str
    wls_uncal : array of float
    rescale_factors : array of float
        Calibrated / Uncalibrated, i.e. the factor by which to multiply the uncalibrated fluxes to get the calibrated fluxes
    rescale_factor_errs : array of float
    '''
    
    all_uncal_dates, wls_uncal, fluxes_uncal, _ = load_lddisk_solutions(uncal_dates)
    rescale_factors = np.zeros(len(all_uncal_dates))
    rescale_factor_errs = np.zeros(len(all_uncal_dates))
    for i, date in enumerate(all_uncal_dates):
        wl = wls_uncal[i]
        flux = fluxes_uncal[i]
        cal_truth = cal_wls == wl
        cal_flux = cal_fluxes[cal_truth]
        cal_err = cal_frac_err[cal_truth]
        rescale_factor = cal_flux/flux
        rescale_error = rescale_factor * cal_err
        rescale_factors[i] = rescale_factor
        rescale_factor_errs[i] = rescale_error
    
    return all_uncal_dates, wls_uncal, rescale_factors, rescale_factor_errs


def apply_bootstrap(all_dates, all_uncal_dates, wls, wls_uncal, fluxes, rescale_factors, flux_errs, rescale_factor_errs):
    
    needs_rescale = np.isin(all_dates, all_uncal_dates) * np.isin(wls, wls_uncal)
    
    print(np.mean(fluxes[needs_rescale])/np.mean(fluxes[~needs_rescale]))
    print(np.mean(rescale_factors**-1))
    
    fluxes_out = np.zeros(len(fluxes))
    flux_errs_out = np.zeros(len(fluxes))
    for i, flux in enumerate(fluxes):
        if needs_rescale[i]:
            date = all_dates[i]
            wl = wls[i]
            flux_err = flux_errs[i]
            truth = np.isin(all_uncal_dates, date) * np.isin(wls_uncal, wl)
            factor = float(rescale_factors[truth])
            factor_err = float(rescale_factor_errs[truth])
            

            flux_out = flux * factor
            fluxes_out[i] = flux_out
            
            fractional_error = np.sqrt((flux_err/flux)**2 + (factor_err/factor)**2)
            flux_errs_out[i] = flux_out * fractional_error
        else:
            fluxes_out[i] = flux
            flux_errs_out[i] = flux_errs[i]

    return fluxes_out, flux_errs_out


def photometry_table_for_feature(dates, wls, lats, lons, fluxes, flux_errs, extent):
    '''
    determine photometry in all available bands 
    for a given feature, as defined by lying within a bounding box
    
    Parameters
    ----------
    dates : list of str
    lats : array of float
    lons : array of float
    fluxes : array of float
    flux_errs : array of float
    extent : list of float
        [lon_min, lon_max, lat_min, lat_max]
        
    Returns
    -------
    phot_table : astropy.table.Table
    '''
    
    truth = (lats < extent[3]) * (lats > extent[2]) * (lons < extent[1]) * (lons > extent[0])
    dates_in = dates[truth]
    wls_in = wls[truth]
    fluxes_in = fluxes[truth]
    flux_errs_in = flux_errs[truth]
    
    phot_table = Table()
    phot_table['date'] = dates_in
    phot_table['wl'] = wls_in
    phot_table['flux'] = fluxes_in
    phot_table['flux_err'] = flux_errs_in
    
    return phot_table

dates = np.array(['2022jul09', '2022jul26', '2022jul27', '2022aug15', '2022aug16', '2022sep06', '2022sep07', '2022sep11', '2022sep12'])
calibrated = np.array([False, False, False, True, True, True, True, False, True])
calibrated_dates = dates[calibrated]
uncalibrated_dates = dates[~calibrated]

wls, cal_diskflux, cal_fluxerr = determine_calibrated_fluxes(calibrated_dates)
all_uncal_dates, wls_uncal, rescale_factors, rescale_factor_errs = determine_flux_rescale_factor(uncalibrated_dates, wls, cal_diskflux, cal_fluxerr)

#print(wls)
#print(cal_diskflux) #these are low c.f. KdK dissertation figure 4.1 by a factor of ~2
# checking each individual measurement, all five are low. Their standard deviation is only ~10%
# this might be due to nav.ldmodel using Tb, i.e. peak flux, rather than total flux?
#print(cal_fluxerr)
# ld parameters were reasonable: according to KdK dissertation p. 135, "The best correction is
#typically found for values of k between 0.65 and 0.75, in agreement with Laver and de Pater
# (2008) and de Pater et al. (2014b)."

# load photometry tables
all_dates, wls, fluxes, flux_errs, lats, lons = load_tables(dates)

flux_errs[flux_errs == 0.0] = np.nan
fluxes, flux_errs = apply_bootstrap(all_dates, all_uncal_dates, wls, wls_uncal, fluxes, rescale_factors, flux_errs, rescale_factor_errs)

lons[lons<-360] = np.nan
#plt.errorbar(lons[wls < 3.8], fluxes[wls < 3.8], yerr = flux_errs[wls < 3.8], fmt = 'o', color = 'blue', label = 'Lp')
#plt.errorbar(lons[wls == 4.67], fluxes[wls == 4.67], yerr = flux_errs[wls == 4.67], fmt = 'o', color = 'red', label = 'Ms')
#plt.show()


phot_table_emakong = photometry_table_for_feature(all_dates, wls, lats, lons, fluxes, flux_errs, extent = [120, 140, -15, 5])
phot_table_loki = photometry_table_for_feature(all_dates, wls, lats, lons, fluxes, flux_errs, extent = [300, 320, 5, 25])
phot_table_loki.pprint_all()

def plot_phot(ax, phot_table):
    
    lband = phot_table['wl'] == 3.776
    mband = phot_table['wl'] == 4.67
    phot_l = phot_table[lband]
    phot_m = phot_table[mband]
    
    ax.errorbar(phot_l['date'], phot_l['flux'], yerr = phot_l['flux_err'], fmt = 'o', color = 'blue', label = 'Lp')
    ax.errorbar(phot_m['date'], phot_m['flux'], yerr = phot_m['flux_err'], fmt = 'o', color = 'red', label = 'Ms')

fig, (ax0, ax1) = plt.subplots(1,2, figsize = (9,5))

plot_phot(ax0, phot_table_emakong)
plot_phot(ax1, phot_table_loki)
ax0.set_ylabel('Flux (GW $\mu$m$^{-1}$ sr$^{-1}$)')
ax0.set_title('Emakong')
ax1.set_title('Loki')
for ax in [ax0, ax1]:
    ax.set_xlabel('Date')
    ax.legend()


plt.show()

# issues
# bootstrapped dates are factor of 2-3 larger than calibrated dates
#     - summing vs averaging the three frames? but this should be taken out by the bootstrap
#     - could be real but I doubt it
#     - applying incorrect wavelength somehow?
#     - check retrieved disk fluxes vs published values from KdK
#     - 2.9571e-12 erg/s/cm2/um/pixel for a Keck pixel (0.01x0.01") and a sun distance of 1 AU, in Ms band... these are extremely annoying units
# M-band on 26 July yielding results we do not believe for Loki

# to do
# - re-do first principles calculation of the expected reflected sunlight component
# - check the limb darkened disk fits to ensure their residuals look close to zero
# - email Imke asking for help