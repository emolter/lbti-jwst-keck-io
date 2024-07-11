#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import glob, os
import paths
from astropy.io import fits
from navigate_keck_io import parse_filter_name
from scipy.ndimage import center_of_mass

dates = ['2022jul09', '2022jul26', '2022jul27', '2022aug15', '2022aug16', '2022sep06', '2022sep07', '2022sep11', '2022sep12']
filters = ['lp', 'ms', 'pah', 'br_alpha', 'kcont', 'h2o', 'br_alpha_cont']
wls = np.array([3.776, 4.67, 3.2904, 4.052, 2.2706, 3.0629, 3.987])
crop_size = 120
fig, axes = plt.subplots(len(dates), len(wls), figsize = (7, 10))
for i, date in enumerate(dates):
    fnames = glob.glob(str(paths.data / 'keck' / date / '*.fits'))
    indices_used = []
    for fname in fnames:

        wl, filt_name = parse_filter_name(fname)
        idx = np.argwhere(wls == wl)[0][0]
        indices_used.append(idx)
        
        data = fits.getdata(fname)
        data[data > 1e10] = 0.0
        com = np.array(center_of_mass(data)).astype(int)
        data = data[com[0]-crop_size:com[0]+crop_size, com[1]-crop_size:com[1]+crop_size]
        axes[i,idx].imshow(data, cmap = 'gist_heat', origin = 'lower')
        axes[i,idx].set_title(filt_name, fontsize = 9)
        axes[i,idx].set_xticks([])
        axes[i,idx].set_yticks([])
        axes[i,idx].patch.set_edgecolor('white') 
        axes[i,idx].patch.set_linewidth(2)

    indices_unused = np.setdiff1d(np.arange(len(filters)), indices_used)
    for idx in indices_unused:
        axes[i,idx].axis('off')
    axes[i,0].set_ylabel(date)

plt.subplots_adjust(wspace=0.0, hspace=0.0)
fig.savefig('/Users/emolter/Desktop/io-agu/keck-data-gallery.png', dpi = 300, bbox_inches = 0.0, transparent=True)
plt.show()
        
        