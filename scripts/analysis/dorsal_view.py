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

DATA_FOLDER = '/Users/blota/Data/PZAH5.6b/brainreg_results/'
imgs_path = dict(red=Path(DATA_FOLDER) / 'downsampled_standard_3.tiff',
                 blue=None,
                 green=Path(DATA_FOLDER) / 'downsampled_standard.tiff')

PATH_TO_SAVE = '/Users/blota/Data/dorsal_view_brainreg_chan3/'
ATLAS_NAME = 'allen_mouse_10um'
PEELED_SAVE_PATH = '/Users/blota/Data/ARA_CCFv3/ARA_10_micron_mhd/'
ATLAS_ANNOTATION = None # '/Users/blota/Data/ARA_CCFv3/ARA_10_micron_mhd/atlas.mhd'
NAPARI = False

# do that first to crash early if there is a display issue
if NAPARI:
    viewer = Viewer()

PATH_TO_SAVE = Path(PATH_TO_SAVE)
# make sure it exists
PATH_TO_SAVE.mkdir(exist_ok=True)

print('Loading atlas')
bg_atlas = bga.bg_atlas.BrainGlobeAtlas(ATLAS_NAME)

# find layers
ctx_df = atlas_utils.create_ctx_table(bg_atlas)
layers = ['surface', '1', '2/3', '4', '5', '6a', '6b']
if ATLAS_ANNOTATION is None:
    atlas_annot =  bg_atlas.annotation
else:
   atlas_annot = itk.array_from_image(itk.imread(ATLAS_ANNOTATION))

# get the index of the layers, including surface
peel_atlas_ids = []
for l in layers:
    if l == 'surface':
        peel_atlas_ids.append([])
        continue
    peel_atlas_ids.append(ctx_df.loc[ctx_df.layer == l, 'id'].values)

peeled_cortical_ids = None
if PEELED_SAVE_PATH is not None:
    target = Path(PEELED_SAVE_PATH) / ('peeled_index_%s.npz' % ATLAS_NAME)
    if target.is_file():
        reloaded = np.load(target)
        if all([l in reloaded for l in layers]):
            peeled_cortical_ids = [reloaded[l] for l in layers]

if peeled_cortical_ids is None:
    peeled_cortical_ids = atlas_utils.peel_atlas(atlas_annot, peel_atlas_ids,
                                                 axis='dorsal', get_index=True,
                                                 which='first', verbose=True)
    if PEELED_SAVE_PATH is not None:
        np.savez(target, **{l: p for l, p in zip(layers, peeled_cortical_ids)})

# get registered data
print('Loading img data')
image_volumes = dict()
for col, pa in imgs_path.items():
    if pa is None:
        continue
    print('   loading %s channel' % col)
    image_volumes[col] = itk.array_from_image(itk.imread(str(pa)))

atlas_dorsal_by_layer = dict()
data_dorsal_by_layer = {c:dict() for c in image_volumes}
atlas_index = dict()
# we will have dorsal views, so size of shape[0] x shape[2]
x, y = np.meshgrid(*[np.arange(s) for s in peeled_cortical_ids[0].shape])
x = np.array(x, dtype=int).T
y = np.array(y, dtype=int).T
top_of_layer = np.zeros_like(x)
for layer_index, l in enumerate(layers):
    print('Doing %s' % l, flush=True)
    # make a dorsal view of the atlas
    atlas_layer = np.zeros(x.shape, dtype=atlas_annot.dtype)
    atlas_layer[x, y] = atlas_annot[x, top_of_layer, y]
    atlas_dorsal_by_layer[l] = atlas_layer
    bottom_of_layer = peeled_cortical_ids[layer_index]
    thickness = bottom_of_layer - top_of_layer
    max_diff = np.max(thickness)

    for color, img_volume in image_volumes.items():
        data_view = np.zeros(x.shape, dtype=img_volume.dtype)
        zero = img_volume.min()
        extent = img_volume.max() - zero
        for i in range(max_diff):
            mask = thickness >= i
            # for max proj
            data_view[x[mask], y[mask]] = np.maximum(data_view[x[mask], y[mask]],
                                                     img_volume[x[mask],
                                                                top_of_layer[mask] + i,
                                                                y[mask]]
                                                     )
        data_dorsal_by_layer[color][l] = data_view
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
        img = ax.imshow(data_dorsal_by_layer[l], cmap='Greys_r')
        ax.set_title('Layer %s' % l)
        layer_name = l.replace('/', '_')
        fig.savefig(save_root / ('dorsal_view_layer_%s.png' % layer_name), dpi=600)
        fig.savefig(save_root / ('dorsal_view_layer_%s.svg' % layer_name), dpi=600)
        if imgs_path['blue'] is not None:
            img.remove()
            ax.set_title('Layer %s - background' % l)
            img = ax.imshow(bg_dorsal_by_layer[l], cmap='Greys_r')
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
