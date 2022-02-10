from pathlib import Path
import flexiznam as fzm
import bg_space as bgs
import bg_atlasapi as bga

import numpy as np
from napari.viewer import Viewer
from brainglobe_napari_io.cellfinder import reader_xml
import tifffile

from cricksaw_analysis import atlas_utils

NAPARI = True
ADD_CELLS = True
project = 'rabies_barcoding'
mouse = 'BRYC64.2i'
channel = 3
pixel_size_um = 50
data_size_um = [8, 2, 2]
cell_type = 'Cells'
roots = {k: Path(v) for k, v in fzm.PARAMETERS['data_root'].items()}

# do that first to crash early if there is a display issue
if NAPARI:
    viewer = Viewer()

# processed = roots['processed'] / project / mouse
processed = Path('/Users/blota/Data/') / mouse
cf_folder = processed / ('cellfinder_results_%03d' % pixel_size_um)

# get registered data
reg_folder = cf_folder / 'registration'
reg_ch0 = tifffile.imread(reg_folder / 'downsampled_standard_channel_0.tiff')
reg_ch1 = tifffile.imread(reg_folder / 'downsampled_standard.tiff')

allen25 = bga.bg_atlas.BrainGlobeAtlas("allen_mouse_25um")

# find layers
ctx_df = atlas_utils.create_ctx_table(allen25)
layers = ['1', '2/3', '4', '5', '6a', '6b']
peeled_atlas = allen25.annotation.copy()

# we will have dorsal views, so size of shape[0] x shape[2]
x, y = np.meshgrid(*[np.arange(allen25.shape[i]) for i in [0, 2]])
x = np.array(x, dtype=int).T
y = np.array(y, dtype=int).T
top_of_layer = atlas_utils.external_view(peeled_atlas, axis='dorsal',
                                         border_only=False,
                                         get_index=True, which='first')

if ADD_CELLS:
    # find cell count
    reg_atlas = tifffile.imread(reg_folder / 'registered_atlas.tiff')
    cell_xml = cf_folder / 'points' / 'cell_classification.xml'
    cell_classification = reader_xml.xml_reader(str(cell_xml))
    cell_classification = {c[1]['name']: c for c in cell_classification}
    cells = cell_classification[cell_type]
    cells_scaled = np.array(np.round(cells[0] * np.array(data_size_um) / 25))
    px_cells = np.array(cells_scaled, dtype=int)
    cell_atlas_id = reg_atlas[px_cells[:, 0], px_cells[:, 1], px_cells[:, 2]]
    cells_in_layer = dict()
    colors = np.array([[127, 201, 127, 200],
                       [190, 174, 212, 200],
                       [253, 192, 134, 200],
                       [255, 255, 153, 200],
                       [56, 108, 176, 200],
                       [240, 2, 127, 200]], dtype=float)
    cell_colors = {l: c for l, c in zip(layers, colors / 255)}
atlas_dorsal_by_layer = dict()
data_dorsal_by_layer = dict()
bg_dorsal_by_layer = dict()
atlas_index = dict()

for l in layers:
    atlas_index[l] = top_of_layer
    layer_index = ctx_df.loc[ctx_df.layer == l, 'id'].values
    if ADD_CELLS:
        cells_in_layer[l] = cells_scaled[np.isin(cell_atlas_id, layer_index), :]
    # make a dorsal view of the atlas
    atlas_layer = np.zeros(x.shape, dtype=allen25.annotation.dtype)
    atlas_layer[x, y] = allen25.annotation[x, top_of_layer, y]
    atlas_dorsal_by_layer[l] = atlas_layer
    print('Peeling layer %s' % l)
    layer_mask = np.isin(allen25.annotation, layer_index)
    peeled_atlas[layer_mask] = 0
    bottom_of_layer = atlas_utils.external_view(peeled_atlas, axis='dorsal',
                                                border_only=False,
                                                get_index=True, which='first')

    thickness = bottom_of_layer - top_of_layer
    max_diff = np.max(thickness)

    # make a max proj of data, iterate on all thickness value
    reg_data_view = np.zeros(x.shape, dtype=reg_ch0.dtype)
    bg_data_view = np.zeros(x.shape, dtype=reg_ch0.dtype)
    for i in range(max_diff):
        mask = thickness >= i
        # for max proj
        reg_data_view[x[mask], y[mask]] = np.maximum(reg_data_view[x[mask], y[mask]],
                                                     reg_ch0[x[mask],
                                                             top_of_layer[mask] - i,
                                                             y[mask]])
        bg_data_view[x[mask], y[mask]] = np.maximum(bg_data_view[x[mask], y[mask]],
                                                    reg_ch1[x[mask],
                                                            top_of_layer[mask] - i,
                                                            y[mask]])

    data_dorsal_by_layer[l] = reg_data_view
    bg_dorsal_by_layer[l] = bg_data_view
    # new top and iterate
    top_of_layer = bottom_of_layer
atlas_index['wm'] = bottom_of_layer

if NAPARI:
    for l in layers[::-1]:
        viewer.add_image(data=bg_dorsal_by_layer[l], name='Background layer %s' % l,
                         colormap='green', contrast_limits=[0, 500], blending='opaque')
        viewer.add_image(data=data_dorsal_by_layer[l], name='Signal layer %s' % l,
                         colormap='red', contrast_limits=[0, 2000], blending='additive')
        viewer.add_labels(data=atlas_dorsal_by_layer[l], name='atlas %s' % l,
                          opacity=0.1)
        if ADD_CELLS:
            viewer.add_points(data=cells_in_layer[l], name='Cells in %s' % l,
                              size=[5, 10, 10], symbol='disc', edge_width=0.5,
                              face_color=cell_colors[l])
