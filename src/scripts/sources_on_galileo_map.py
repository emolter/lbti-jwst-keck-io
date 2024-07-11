#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import QTable
from astropy.io import ascii
import paths
import glob
from PIL import Image

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def overlay_galileo_and_features(dates, colors, extent = [0, 360, -90, 90], outfile = None, title=''):
    
    fig, ax = plt.subplots(1,1, figsize = (12, 8))
    all_lons = []
    all_lats = []
    for i, date in enumerate(dates):
        phot_tables = glob.glob(str(paths.data / 'keck' / date / '*_photometry.txt'))
        lons = []
        lats = []
        fluxes = []
        for tab in phot_tables:
            data = np.loadtxt(tab, skiprows=3, usecols=[1, 2, 3])
            lons += list(data[-2])
            lats += list(data[-4])

        lons = np.array(lons).flatten()
        lats = np.array(lats).flatten()
        truth = (lats < extent[3]) * (lats > extent[2]) * (lons < extent[1]) * (lons > extent[0])
        lats = lats[truth]
        lons = lons[truth]
        ax.scatter(lons, lats, marker = 'o', facecolor='none', edgecolor=colors[i], s = 100, label = date, linewidths=3)
        all_lons += list(lons)
        all_lats += list(lats)
    
    print(f'Feature is at: ')
    print(f'{np.mean(all_lats):.2f} +/- {np.std(all_lats)} latitude')
    print(f'{np.mean(all_lons):.2f} +/- {np.std(all_lons)} longitude')
    
    im = Image.open(str(paths.data / 'galileo-mosaic.jpg'))
    im = np.array(im)
    halfway = im.shape[1]//2
    im = np.concatenate((im[:, halfway:], im[:, :halfway]), axis=1)
    
    ax.imshow(im, extent=[360, 0, -90, 90], alpha=1.0, cmap='gray')
    ax.set_xlabel('Longitude (W)')
    ax.set_ylabel('Latitude')
    ax.set_xlim(extent[1], extent[0])
    ax.set_ylim(extent[2], extent[3])
    ax.legend()
    ax.set_title(title)
    
    if outfile is not None:
        fig.savefig(outfile, dpi = 300)
    plt.show()
    plt.close()
    
    
dates = ['2022jul09', '2022jul26', '2022jul27', '2022aug15', '2022aug16', '2022sep06', '2022sep07', '2022sep11', '2022sep12']
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown', 'cyan']
outstem = '/Users/emolter/Desktop/sources_on_galileo_map'

overlay_galileo_and_features(dates, colors, extent = [0, 360, -90, 90], outfile = outstem+'.png', title='Brightest Keck Sources on Galileo SSI Map')
overlay_galileo_and_features(dates, colors, extent = [120, 140, -15, 5], outfile = outstem+'_emakong.png', title='Keck Sources near Seth Patera')
overlay_galileo_and_features(dates, colors, extent = [300, 320, 5, 25], outfile = outstem+'_loki.png', title='Keck Sources near Loki Patera')

