"""
Create a dorsal view projected across layers.

Particularly useful to find blood vessel patterns
"""

import matplotlib
import socket
if socket.gethostname() == 'C02Z85AULVDC':
    matplotlib.use('macosx')

from pathlib import Path
import bg_atlasapi as bga
import itk
import numpy as np
from napari.viewer import Viewer
from cricksaw_analysis import atlas_utils

DATA_FOLDER = '/Users/blota/Data/PZAH5.6b/brainreg_results/'
imgs_path = dict(red=Path(DATA_FOLDER) / 'downsampled_standard_3.tiff',
                 blue=None,
                 green=None)

PATH_TO_SAVE = '/Users/blota/Data/dorsal_view_brainreg_chan3/stack'
ATLAS_NAME = 'allen_mouse_10um'
ATLAS_ANNOTATION = None #'/Users/blota/Data/ARA_CCFv3/ARA_10_micron_mhd/atlas.mhd'
NAPARI = False

# do that first to crash early if there is a display issue
if NAPARI:
    viewer = Viewer()

PATH_TO_SAVE = Path(PATH_TO_SAVE)
# make sure it exists
PATH_TO_SAVE.mkdir(exist_ok=True)

print('Loading atlas')
bg_atlas = bga.bg_atlas.BrainGlobeAtlas(ATLAS_NAME)
# get registered data
print('Loading img data')
image_volumes = dict()
for col, pa in imgs_path.items():
    if pa is None:
        continue
    print('   loading %s channel' % col)
    image_volumes[col] = itk.array_from_image(itk.imread(str(pa)))

# find layers
ctx_df = atlas_utils.create_ctx_table(bg_atlas)
layers = ['1']
if ATLAS_ANNOTATION is None:
    atlas_annot = bg_atlas.annotation.copy()
else:
   atlas_annot = itk.array_from_image(itk.imread(ATLAS_ANNOTATION))
peeled_atlas = np.array(atlas_annot, copy=True)

atlas_dorsal_by_layer = dict()
data_dorsal_by_layer = dict()
bg_dorsal_by_layer = dict()
atlas_index = dict()

# we will have dorsal views, so size of shape[0] x shape[2]
x, y = np.meshgrid(*[np.arange(bg_atlas.shape[i]) for i in [0, 2]])
x = np.array(x, dtype=int).T
y = np.array(y, dtype=int).T
top_of_layer = atlas_utils.external_view(peeled_atlas, axis='dorsal',
                                         border_only=False,
                                         get_index=True, which='first')
for l in layers:
    print('Doing %s' % l, flush=True)
    atlas_index[l] = top_of_layer
    layer_index = ctx_df.loc[ctx_df.layer == l, 'id'].values
    # make a dorsal view of the atlas
    atlas_layer = np.zeros(x.shape, dtype=atlas_annot.dtype)
    atlas_layer[x, y] = atlas_annot[x, top_of_layer, y]
    atlas_dorsal_by_layer[l] = atlas_layer
    print('Peeling layer %s' % l, flush=True)
    layer_mask = np.isin(atlas_annot, layer_index)
    peeled_atlas[layer_mask] = 0
    print('Find surface of next layer', flush=True)
    bottom_of_layer = atlas_utils.external_view(peeled_atlas, axis='dorsal',
                                                border_only=False,
                                                get_index=True, which='first')

    thickness = bottom_of_layer - top_of_layer
    max_diff = np.max(thickness)

    # save a stack
    for color, img_volume in image_volumes.items():
        data_view = np.zeros(x.shape, dtype=img_volume.dtype)
        zero = img_volume.min()
        extent = img_volume.max() - zero
        for name, i in enumerate(range(-10, 30)):
            img = img_volume[x, top_of_layer + i, y]
            img = (img - zero) * (65536 / extent.max())
            img = np.array(img, dtype='uint16')
            itk.imwrite(itk.image_from_array(img),
                        PATH_TO_SAVE / ('dorsal_view_%02d.png' % name))
        # do the extended depth of focus in FIJI
