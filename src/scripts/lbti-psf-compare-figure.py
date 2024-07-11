#!/usr/bin/env python

'''
Theoretical, Emakong, and Standard Star PSFs all overlain on same figure
'''

import numpy as np
import matplotlib.pyplot as plt
import paths

# load the important data from combined-disk-solution.ipynb
emak_cut = np.load(paths.data / 'lbti' / 'emakong_1d_cut.npy')
emak_cut_std = np.load(paths.data / 'lbti' / 'emakong_1d_cut_std.npy')
psf_cut = np.load(paths.data / 'lbti' / 'psf_1d_cut.npy')

# load the theoretical PSF cut from theoretical-interferogram.ipynb
theory_cut = np.load(paths.data / 'lbti' / 'theoretical_psf_1d_cut.npy')
theory_cut = theory_cut / np.max(theory_cut)
psf_cut = psf_cut / np.max(psf_cut)



fig, ax2 = plt.subplots(1,1, figsize=(6, 5))

ax2.plot(emak_cut, color = 'red', label = 'Seth Patera Feature')
ax2.plot(emak_cut - emak_cut_std, color = 'red', linestyle = ':', label=r'1$\sigma$ uncertainty')
ax2.plot(emak_cut + emak_cut_std, color = 'red', linestyle = ':')
ax2.plot(psf_cut, color = 'k', label = 'Standard Star PSF')
ax2.plot(theory_cut, color = 'blue', linestyle = '--', label = 'Theoretical Ideal PSF')
ax2.set_xlim([0,23])
ax2.set_xlabel('X pixel')
ax2.set_ylabel('Normalized Flux')
ax2.legend()

fig.savefig('diagnostic_plots/emakong_psf_cut_compare.png', dpi=300)
plt.show()