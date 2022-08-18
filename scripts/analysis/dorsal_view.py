"""
Create a dorsal view projected across layers.

Particularly useful to find blood vessel patterns
"""

import matplotlib
import socket
if socket.gethostname() == 'C02Z85AULVDC':
    matplotlib.use('macosx')

from pathlib import Path
import flexiznam as fzm
import bg_space as bgs
import bg_atlasapi as bga
import matplotlib.pyplot as plt
import itk
import numpy as np
from napari.viewer import Viewer
from brainglobe_napari_io.cellfinder import reader_xml
import tifffile

from cricksaw_analysis import atlas_utils

DATA_FOLDER = '/camp/lab/znamenskiyp/home/shared/projects/hey2_3d' \
              '-vision_foodres_20220101/PZAH5.6a/brainregister/sample_to_ccf'
#DATA_FOLDER = '/camp/lab/znamenskiyp/home/shared/projects/hey2_3d' \
#              '-vision_foodres_20220101/PZAH5.6a/registration_10/PZAH5.6as_inverse_reg__elastix_out_step01'
REGISTERED_DATA = Path(DATA_FOLDER) / 'CCF_ds_PZAH5_6a_220624_160546_10_10_ch03_chan_3_red.nrrd'
#REGISTERED_DATA = Path(DATA_FOLDER) / 'PZAH5.6as_inverse_reg__registration_step01.mhd'
REGISTERED_BACKGROUND_DATA = Path(DATA_FOLDER) / 'CCF_ds_PZAH5_6a_220624_160546_10_10_ch02_chan_2_green.nrrd'

PATH_TO_SAVE = '/camp/lab/znamenskiyp/home/shared/projects/hey2_3d' \
              '-vision_foodres_20220101/PZAH5.6a/dorsal_view/'
ATLAS_NAME = 'allen_mouse_10um'
ATLAS_ANNOTATION = '/camp/lab/znamenskiyp/home/shared/resources/cellfinder_resources/ARA_CCFv3/ARA_10_micron_mhd/atlas.mhd'
NAPARI = False

# do that first to crash early if there is a display issue
if NAPARI:
    viewer = Viewer()

print('Loading atlas', flush=True)
bg_atlas = bga.bg_atlas.BrainGlobeAtlas(ATLAS_NAME)
# get registered data
print('Loading img data', flush=True)
reg_data = itk.array_from_image(itk.imread(REGISTERED_DATA))
if REGISTERED_BACKGROUND_DATA is not None:
    print('Loading background data', flush=True)
    reg_backgrnd = itk.array_from_image(itk.imread(REGISTERED_BACKGROUND_DATA))


# find layers
ctx_df = atlas_utils.create_ctx_table(bg_atlas)
layers = ['1', '2/3', '4', '5', '6a', '6b']
if ATLAS_ANNOTATION is None:
    atlas_annot =  bg_atlas.annotation.copy()
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
    bottom_of_layer = atlas_utils.external_view(peeled_atlas, axis='dorsal',
                                                border_only=False,
                                                get_index=True, which='first')

    thickness = bottom_of_layer - top_of_layer
    # force to look at 200um
    # thickness = np.zeros_like(bottom_of_layer) + 20
    max_diff = np.max(thickness)
    print('    Thickness: avg {}, max {}'.format(np.mean(thickness[thickness!=0]),
                                                 np.max(thickness)))
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
                                                        reg_backgrnd[x[mask],
                                                                 top_of_layer[mask] - i,
                                                                 y[mask]])

    data_dorsal_by_layer[l] = reg_data_view
    if REGISTERED_BACKGROUND_DATA is not None:
        bg_dorsal_by_layer[l] = bg_data_view
    # new top and iterate
    # XX KEEP GOING FROM SURFACE
    #top_of_layer = bottom_of_layer
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

if PATH_TO_SAVE is not None:
    print('Saving in %s' % PATH_TO_SAVE, flush=True)
    save_root = Path(PATH_TO_SAVE)
    save_root.mkdir(exist_ok=True)
    assert save_root.is_dir()
    fig = plt.figure()
    fig.set_size_inches([6, 7])
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(top=0.95, right=0.99, bottom=0.05, left=0.07)
    midline = int(atlas_dorsal_by_layer[l].shape[1] / 2)
    for l in layers:
        ax.clear()
        ax.set_aspect('equal')
        atlas_utils.plot_borders_and_areas(ax, atlas_dorsal_by_layer[l],
                                           areas_to_plot=[])
        # do a version with label on right hemisphere
        right_hem = np.zeros_like(atlas_dorsal_by_layer[l])
        right_hem[:, midline:] = atlas_dorsal_by_layer[l][:,midline:]
        atlas_utils.plot_borders_and_areas(ax, right_hem,
                                           areas_to_plot=[],
                                           label_atlas=bg_atlas)
        top = np.quantile(bg_dorsal_by_layer[l], 0.99)
        img = ax.imshow(data_dorsal_by_layer[l], cmap='Greys_r', vmax=top)
        ax.set_title('Layer %s' % l)
        layer_name = l.replace('/', '_')
        fig.savefig(save_root / ('dorsal_view_layer_%s.png' % layer_name), dpi=600)
        fig.savefig(save_root / ('dorsal_view_layer_%s.svg' % layer_name), dpi=600)
        if REGISTERED_BACKGROUND_DATA is not None:
            img.remove()
            ax.set_title('Layer %s - background' % l)
            top = np.quantile(bg_dorsal_by_layer[l], 0.99)
            img = ax.imshow(bg_dorsal_by_layer[l], cmap='Greys_r', vmax=top)
            fig.savefig(save_root / ('dorsal_view_layer_background_%s.png' % layer_name),
                        dpi=600)
            fig.savefig(save_root / ('dorsal_view_layer_background_%s.svg' % layer_name),
                        dpi=600)
            img.remove()
            rgb = np.zeros(list(bg_dorsal_by_layer[l].shape) + [3],
                           dtype=np.uint8)
            top = np.quantile(bg_dorsal_by_layer[l], 0.99)
            bg = bg_dorsal_by_layer[l] / top * 255
            bg[bg > 255] = 255
            bg = np.array(bg, dtype=np.uint8)
            rgb[:, :, 0] = bg
            rgb[:, :, 2] = bg
            top = np.quantile(data_dorsal_by_layer[l], 0.99)
            dt = data_dorsal_by_layer[l]/ top * 255
            dt[dt > 255] = 255
            dt = np.array(dt, dtype=np.uint8)

            rgb[:, :, 1] = dt
            img = ax.imshow(rgb)
            ax.set_title('Layer %s - Magenta background, Green data' % l)
            fig.savefig(save_root / ('dorsal_view_layer_rgb_%s.png' % layer_name),
                        dpi=600)
            fig.savefig(save_root / ('dorsal_view_layer_rgb_%s.svg' % layer_name),
                        dpi=600)
else:
    print('Do not save', flush=True)

print('done')
