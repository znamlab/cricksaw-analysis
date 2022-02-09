"""Small script to threshold downsampled data and find cells in injection site"""
import numpy as np

from cricksaw_analysis.main import find_cell_in_injection_site

NAPARI = False
project = 'rabies_barcoding'

mouse = 'BRYC64.2i'
channel = 3
pixel_size_um = 50
data_size_um = [8, 2, 2]

out = find_cell_in_injection_site(project, mouse, channel, pixel_size_um,
                                  cell_scale_um=data_size_um, inj_min_width=100,
                                  inj_vol_percentile=99.99, cell_type='Cells',
                                  cell_kwargs=None, view_in_napari=NAPARI)
cell_in_inj, inj_site, cell_out_inj = out[:3]

voxel_vol = (pixel_size_um * 1e-3)**3
inj_vol = np.sum(inj_site) * voxel_vol

nc = len(cell_in_inj) + len(cell_out_inj)
print('Analysis for %s' % mouse)
print('There are %d cells in total.' % nc)
msg = 'There are %d cells in the injection site.' % len(cell_in_inj)
msg += "That's %.2f %%" % (len(cell_in_inj)/nc * 100)
print(msg)
print('The injection site is %.2f mm^3.' % inj_vol)
print('The density of cells is therefore %d cells/mm^3' % (len(cell_in_inj) / inj_vol))
print('If we assume that there are 100,000 cells/mm^3, we stained %.3f %% of the cells'
      % (len(cell_in_inj) / inj_vol / 100000))

print('Done')

if NAPARI:
    viewer = out[-1]