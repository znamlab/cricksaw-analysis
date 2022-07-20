"""
Create a dorsal view projected across layers.

Particularly useful to find blood vessel patterns
"""

from pathlib import Path
import flexiznam as fzm
import bg_space as bgs
import bg_atlasapi as bga

import numpy as np
from napari.viewer import Viewer
from brainglobe_napari_io.cellfinder import reader_xml
import tifffile

from cricksaw_analysis import atlas_utils


REGISTERED_DATA = '/Volumes/lab-znamenskiyp/home/shared/projects/hey2_3d' \
                  '-vision_foodres_20220101/PZAH5.6a/registered_stacks/010_micron' \
                  '/025_atlas/red/downsampled_standard.tiff'
REGISTERED_BACKGROUND_DATA = '/Volumes/lab-znamenskiyp/home/shared/projects/hey2_3d' \
                             '-vision_foodres_20220101/PZAH5.6a/registered_stacks' \
                             '/010_micron/025_atlas/blue/downsampled_standard.tiff'
ATLAS_NAME = 'allen_mouse_25um'
NAPARI = False

# do that first to crash early if there is a display issue
if NAPARI:
    viewer = Viewer()

print('Loading atlas')
bg_atlas = bga.bg_atlas.BrainGlobeAtlas(ATLAS_NAME)
# get registered data
print('Loading img data')
reg_data = tifffile.imread(REGISTERED_DATA)
if REGISTERED_BACKGROUND_DATA is not None:
    print('Loading background data')
    reg_backgrnd = tifffile.imread(REGISTERED_BACKGROUND_DATA)


# find layers
ctx_df = atlas_utils.create_ctx_table(bg_atlas)
layers = ['1', '2/3', '4', '5', '6a', '6b']
peeled_atlas = bg_atlas.annotation.copy()


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
    atlas_index[l] = top_of_layer
    layer_index = ctx_df.loc[ctx_df.layer == l, 'id'].values
    # make a dorsal view of the atlas
    atlas_layer = np.zeros(x.shape, dtype=bg_atlas.annotation.dtype)
    atlas_layer[x, y] = bg_atlas.annotation[x, top_of_layer, y]
    atlas_dorsal_by_layer[l] = atlas_layer
    print('Peeling layer %s' % l, flush=True)
    layer_mask = np.isin(bg_atlas.annotation, layer_index)
    peeled_atlas[layer_mask] = 0
    bottom_of_layer = atlas_utils.external_view(peeled_atlas, axis='dorsal',
                                                border_only=False,
                                                get_index=True, which='first')

    thickness = bottom_of_layer - top_of_layer
    max_diff = np.max(thickness)

    # make a max proj of data, iterate on all thickness value
    reg_data_view = np.zeros(x.shape, dtype=reg_data.dtype)
    if REGISTERED_BACKGROUND_DATA is not None:
        bg_data_view = np.zeros(x.shape, dtype=reg_backgrnd.dtype)
    for i in range(max_diff):
        mask = thickness >= i
        # for max proj
        reg_data_view[x[mask], y[mask]] = np.maximum(reg_data_view[x[mask], y[mask]],
                                                     reg_data[x[mask],
                                                              top_of_layer[mask] - i,
                                                              y[mask]])
        if REGISTERED_BACKGROUND_DATA is not None:
            bg_data_view[x[mask], y[mask]] = np.maximum(bg_data_view[x[mask], y[mask]],
                                                        reg_data[x[mask],
                                                                 top_of_layer[mask] - i,
                                                                 y[mask]])

    data_dorsal_by_layer[l] = reg_data_view
    if REGISTERED_BACKGROUND_DATA is not None:
        bg_dorsal_by_layer[l] = bg_data_view
    # new top and iterate
    top_of_layer = bottom_of_layer
atlas_index['wm'] = bottom_of_layer

if NAPARI:
    print('Adding to napari', flush=True)
    for l in layers[::-1]:
        viewer.add_image(data=bg_dorsal_by_layer[l], name='Background layer %s' % l,
                         colormap='green', contrast_limits=[0, 500], blending='opaque',
                         visible=False)
        viewer.add_image(data=data_dorsal_by_layer[l], name='Signal layer %s' % l,
                         colormap='gray', contrast_limits=[0, 5000], blending='additive')

        viewer.add_labels(data=atlas_dorsal_by_layer[l], name='atlas %s' % l,
                          opacity=0.1, visible=True if l == '1' else False)

print('done')