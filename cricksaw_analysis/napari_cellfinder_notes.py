from pathlib import Path
import flexiznam as fzm
import bg_space as bgs
import  bg_atlasapi as bga

import numpy as np
from napari.viewer import Viewer
from brainglobe_napari_io.cellfinder import reader_xml
import tifffile

from cricksaw_analysis import atlas_utils

NAPARI = True
project = 'rabies_barcoding'
mouse = 'BRYC64.2i'
channel = 3
pixel_size_um = 50
data_size_um = [8, 2, 2]
cell_type = 'Cells'

# find cell count
# processed = roots['processed'] / project / mouse
cf_folder = Path('D:/cellfinder_results/') / mouse / ('cellfinder_results_%03d' %
                                                      pixel_size_um)
cell_xml = cf_folder / 'points' / 'cell_classification.xml'
cell_classification = reader_xml.xml_reader(str(cell_xml))
cell_classification = {c[1]['name']: c for c in cell_classification}
cells = cell_classification[cell_type]

# get registered data
reg_folder = cf_folder /'registration'
reg_ch0 = tifffile.imread(reg_folder / 'downsampled_standard_channel_0.tiff')

allen25 = bga.bg_atlas.BrainGlobeAtlas("allen_mouse_25um")

# find the surface
def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

surface = first_nonzero(allen25.hemispheres, 0)

# find layers
ctx_df = atlas_utils.create_ctx_table(allen25)
layers = ['1', '2/3', '4', '5', '6a', '6b']
peeled_atlas = np.array(allen25.annotation)

# we will have dorsal views, so size of shape[0] x shape[2]
x, y = np.meshgrid(*[np.arange(allen25.shape[i]) for i in [0, 2]])
x = np.array(x, dtype=int).T
y = np.array(y, dtype=int).T
top_of_layer = atlas_utils.external_view(peeled_atlas, axis='dorsal',
                                         border_only=False,
                                         get_index=True, which='first')

atlas_dorsal_by_layer = dict()
data_dorsal_by_layer = dict()
atlas_index = dict()
for il, l in enumerate(layers):
    atlas_index[l] = top_of_layer
    layer_index = ctx_df.loc[ctx_df.layer == l, 'id'].values
    # make a dorsal view of the atlas
    atlas_layer = np.zeros(x.shape, dtype=allen25.annotation.dtype)
    atlas_layer[x, y] = allen25.annotation[x, top_of_layer, y]
    atlas_dorsal_by_layer[l] = atlas_layer
    bottom_of_layer = atlas_utils.external_view(peeled_atlas, axis='dorsal',
                                                border_only=False,
                                                get_index=True, which='first')
    print('Peeling layer %s' % l)
    layer_mask = np.isin(allen25.annotation, layer_index)
    peeled_atlas[layer_mask] = 0

    thickness = bottom_of_layer - top_of_layer
    max_diff = np.max(thickness)

    # make a max proj of data, iterate on all thickness value
    reg_data_view = np.zeros(x.shape, dtype=reg_ch0.dtype)
    for i in range(max_diff):
        mask = thickness >= i
        # for max proj
        reg_data_view[x[mask], y[mask]] = np.maximum(reg_data_view[x[mask], y[mask]],
                                                     reg_ch0[x[mask],
                                                             top_of_layer[mask] - i,
                                                             y[mask]]
                                                     )
    data_dorsal_by_layer[l] = reg_data_view
    # new top and iterate
    top_of_layer = bottom_of_layer
atlas_index['wm'] = bottom_of_layer

if NAPARI:
    viewer = Viewer()
    for l in layers:
        viewer.add_image(data=data_dorsal_by_layer[l], name=l)
        viewer.add_labels(data=atlas_dorsal_by_layer[l], name='atlas %s' % l)
