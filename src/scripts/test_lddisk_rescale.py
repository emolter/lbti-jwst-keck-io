#!/usr/bin/env python3

'''
The conversion factor between calibrated and uncalibrated models
seems to be incorrect because all the calibrated measurements of Seth
are systematically lower than the uncalibrated measurements.

Here we are testing whether it's sufficient to just use the flux output
by nav.lddisk to make this conversion factor, or if that scales non-linearly
with the true total flux of the disk.

Idea is simply to make two limb-darkened disk models with different fluxes but same a, 
and see if the total flux of the disk scales linearly with the input flux value.
The reason this could be the case is that the input flux value is really a surface brightness

Result: it does scale linearly, so this is not the problem
'''

from pylanetary import navigation
from pylanetary.utils import Body
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import paths

def make_lddisk_model(flux, a = 0.65):
    
    hdul = fits.open(paths.data / 'keck' / '2022jul09' / 'IoLp_nophot.fits')
    header = hdul[0].header
    obs_time = header['DATE-OBS'] + ' ' + header['EXPSTART'][:-4]
    io = Body('Io', epoch=obs_time, location='568') #Maunakea keyword is 568
    
    nav = navigation.ModelBody(io, 0.009971, shape=(160, 160)) #pixel scale of keck
    disk = nav.ldmodel(flux, a, beam=(5), law='minnaert', mu0=None, psf_mode='gaussian')
    return disk

tbs = np.linspace(1, 10, 10)
total_fluxes = []
for tb in tbs:
    disk = make_lddisk_model(tb)
    total_fluxes.append(np.sum(disk))


plt.plot(tbs, total_fluxes)
plt.xlabel('Input flux')
plt.ylabel('Total flux of disk')
plt.show()
    