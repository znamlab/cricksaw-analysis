"""
Create a dorsal view projected across layers.

Particularly useful to find blood vessel patterns
"""

import socket

import matplotlib

if socket.gethostname() == "C02Z85AULVDC":
    matplotlib.use("macosx")

from pathlib import Path

import brainglobe_atlasapi as bga
import itk
import matplotlib.pyplot as plt
import numpy as np
from napari.viewer import Viewer

from cricksaw_analysis import atlas_utils

PROCESSED = Path("/camp/lab/znamenskiyp/home/shared/projects")
PROJECT = "rabies_barcoding"
MOUSE = "BRYC64.2h"
REGISTRATION = "brainreg"
ATLAS_SIZE = 10
PEELED_SAVE_PATH = None
PROJECTION = "max"
SAVE_SVG = False

if REGISTRATION == "brainreg":
    DATA_FOLDER = (
        PROCESSED / PROJECT / MOUSE / "brainreg_results"
    )  # / "from_downsampled"
    imgs_path = dict(
        red=Path(DATA_FOLDER) / "downsampled_standard.tiff",
        blue=None,
        green=Path(DATA_FOLDER) / "downsampled_standard_2.tiff",
    )
    ATLAS_ANNOTATION = None
elif REGISTRATION == "elastix":
    DATA_FOLDER = PROCESSED / PROJECT / MOUSE / "registration_10"
    imgs_path = dict(
        red=None,
        blue=None,
        green=Path(DATA_FOLDER)
        / ("%ss_inverse_reg__elastix_out_step01" % MOUSE)
        / ("%ss_inverse_reg__registration_step01.mhd" % MOUSE),
    )
    resource_folder = Path("/camp/lab/znamenskiyp/home/shared/resources")
    atlas_resource = resource_folder / "cellfinder_resources/ARA_CCFv3"
    ATLAS_ANNOTATION = atlas_resource / ("ARA_%d_micron_mhd/atlas.mhd" % ATLAS_SIZE)
else:
    raise IOError("Registration must be `brainreg` or `elastix`")


PATH_TO_SAVE = DATA_FOLDER / "dorsal_view"
ATLAS_NAME = "allen_mouse_%dum" % ATLAS_SIZE  # codespell:ignore dum
NAPARI = False

projs = dict(max=np.maximum, min=np.minimum, mean=np.add, sum=np.add)
projection_function = projs[PROJECTION.lower()]

if not PATH_TO_SAVE.is_dir():
    PATH_TO_SAVE.mkdir()

# do that first to crash early if there is a display issue
if NAPARI:
    viewer = Viewer()

PATH_TO_SAVE = Path(PATH_TO_SAVE)
# make sure it exists
PATH_TO_SAVE.mkdir(exist_ok=True)

print("Loading atlas")
bg_atlas = bga.bg_atlas.BrainGlobeAtlas(ATLAS_NAME)

# find layers
ctx_df = atlas_utils.create_ctx_table(bg_atlas)
layers = ["surface", "1", "2/3", "4", "5", "6a", "6b"]
if ATLAS_ANNOTATION is None:
    atlas_annot = bg_atlas.annotation
else:
    atlas_annot = itk.array_from_image(itk.imread(ATLAS_ANNOTATION))

# get the index of the layers, including surface
peel_atlas_ids: list[list[int]] = []
for layer in layers:
    if layer == "surface":
        peel_atlas_ids.append([])
        continue
    peel_atlas_ids.append(list(ctx_df.loc[ctx_df.layer == layer, "id"].values))

peeled_cortical_ids = None
if PEELED_SAVE_PATH is not None:
    target = Path(PEELED_SAVE_PATH) / ("peeled_index_%s.npz" % ATLAS_NAME)
    if target.is_file():
        reloaded = np.load(target)
        if all([layer in reloaded for layer in layers]):
            peeled_cortical_ids = [reloaded[layer] for layer in layers]

if peeled_cortical_ids is None:
    peeled_cortical_ids = atlas_utils.peel_atlas(
        atlas_annot,
        peel_atlas_ids,
        axis="dorsal",
        get_index=True,
        which="first",
        verbose=True,
    )
    if PEELED_SAVE_PATH is not None:
        np.savez(
            target, **{layer: peel for layer, peel in zip(layers, peeled_cortical_ids)}
        )

# get registered data
print("Loading img data")
image_volumes = dict()
for col, pa in imgs_path.items():
    if pa is None:
        continue
    print("   loading %s channel" % col)
    image_volumes[col] = itk.array_from_image(itk.imread(str(pa)))

# find layers
ctx_df = atlas_utils.create_ctx_table(bg_atlas)
layers = ["0", "1", "2/3", "4", "5", "6a", "6b"]
if ATLAS_ANNOTATION is None:
    atlas_annot = bg_atlas.annotation.copy()
else:
    atlas_annot = itk.array_from_image(itk.imread(ATLAS_ANNOTATION))
peeled_atlas = np.array(atlas_annot, copy=True)

atlas_dorsal_by_layer: dict[str, np.ndarray] = dict()
data_dorsal_by_layer: dict[str, dict[str, np.ndarray]] = {
    c: dict() for c in image_volumes
}
atlas_index = dict()
# we will have dorsal views, so size of shape[0] x shape[2]
x, y = np.meshgrid(*[np.arange(s) for s in peeled_cortical_ids[0].shape])
x = np.array(x, dtype=int).T
y = np.array(y, dtype=int).T
if layers[0] == "0":
    top_of_layer = np.zeros(x.shape, dtype=int)
else:
    top_of_layer = atlas_utils.external_view(
        peeled_atlas, axis="dorsal", border_only=False, get_index=True, which="first"
    )
for layer in layers:
    print("Doing %s" % layer, flush=True)
    atlas_index[layer] = top_of_layer
    # make a dorsal view of the atlas
    atlas_layer = np.zeros(x.shape, dtype=atlas_annot.dtype)
    atlas_layer[x, y] = atlas_annot[x, top_of_layer, y]
    atlas_dorsal_by_layer[layer] = atlas_layer

    if layer == "0":
        # We don't change peeled_atlas, we will just look at the surface of registration
        pass
    else:
        layer_index = ctx_df.loc[ctx_df.layer == layer, "id"].values
        print("Peeling layer %s" % layer, flush=True)
        layer_mask = np.isin(atlas_annot, layer_index)
        peeled_atlas[layer_mask] = 0

    print("Find surface of next layer", flush=True)
    bottom_of_layer = atlas_utils.external_view(
        peeled_atlas, axis="dorsal", border_only=False, get_index=True, which="first"
    )

    thickness = bottom_of_layer - top_of_layer
    max_diff = np.max(thickness)

    for color, img_volume in image_volumes.items():
        data_view = np.zeros(x.shape, dtype=img_volume.dtype)
        zero = img_volume.min()
        extent = img_volume.max() - zero
        for i in range(max_diff):
            mask = thickness >= i
            # for max proj
            data_view[x[mask], y[mask]] = projection_function(
                data_view[x[mask], y[mask]],
                img_volume[
                    x[mask],
                    np.clip(top_of_layer[mask] + i, 0, img_volume.shape[1] - 1),
                    y[mask],
                ],
            )
        if PROJECTION == "mean":
            data_view = np.array(data_view, dtype=float)
            data_view[thickness != 0] /= thickness[thickness != 0]
            data_view = np.array(data_view, dtype=img_volume.dtype)
        if layer not in data_dorsal_by_layer:
            data_dorsal_by_layer[layer] = dict()
        data_dorsal_by_layer[layer][color] = data_view
        # TODO: From here the script has not been adapted to 3 colors

    # new top and iterate
    top_of_layer = bottom_of_layer
    atlas_index["wm"] = bottom_of_layer

if NAPARI:
    print("Adding to napari", flush=True)
    raise NotImplementedError("Needs to be adapted to 3 colors")

if PATH_TO_SAVE is not None:
    print("Saving in %s" % PATH_TO_SAVE, flush=True)
    save_root = Path(PATH_TO_SAVE)
    save_root.mkdir(exist_ok=True)
    assert save_root.is_dir()
    fig = plt.figure()
    fig.set_size_inches([6, 7])
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(top=0.95, right=0.99, bottom=0.05, left=0.07)
    midline = int(atlas_dorsal_by_layer[layer].shape[1] / 2)
    for layer in layers:
        # decide which layer of the atlas we will plot
        atlas_layer = layer if layer != "0" else "1"
        for color in image_volumes.keys():
            data_this_color = data_dorsal_by_layer[layer][color]
            ax.clear()
            ax.set_aspect("equal")
            atlas_utils.plot_borders_and_areas(
                ax, atlas_dorsal_by_layer[atlas_layer], areas_to_plot=[]
            )
            # do a version with label on right hemisphere
            right_hem = np.zeros_like(atlas_dorsal_by_layer[atlas_layer])
            right_hem[:, midline:] = atlas_dorsal_by_layer[atlas_layer][:, midline:]
            atlas_utils.plot_borders_and_areas(
                ax, right_hem, areas_to_plot=[], label_atlas=bg_atlas
            )
            img = ax.imshow(data_this_color, cmap="Greys_r")
            ax.set_title("Layer %s" % layer)
            layer_name = layer.replace("/", "_")
            fig.savefig(
                save_root
                / (
                    "dorsal_view_%s_proj_layer_%s_%s.png"
                    % (PROJECTION, layer_name, color)
                ),
                dpi=600,
            )
            if SAVE_SVG:
                fig.savefig(
                    save_root
                    / (
                        "dorsal_view_%s_proj_layer_%s_%s.svg"
                        % (PROJECTION, layer_name, color)
                    ),
                    dpi=600,
                )

        if len(data_dorsal_by_layer[layer]) > 1:
            print("Making RGB image for layer %s" % layer)
            img.remove()
            rgb = np.zeros(
                list(atlas_dorsal_by_layer[layer].shape) + [3], dtype=np.uint8
            )
            colord = ["red", "green", "blue"]
            for color, data in data_dorsal_by_layer[layer].items():
                if color not in colord:
                    raise IOError("unknown color: %s" % color)
                colind = colord.index(color)
                top = np.quantile(data, 0.99)
                normed = np.array(data) / top * 255
                normed[normed > 255] = 255
                normed = np.array(normed, dtype=np.uint8)
                rgb[:, :, colind] = normed
            img = ax.imshow(rgb)
            ax.set_title("Layer %s - RGB " % layer)
            fig.savefig(
                save_root
                / ("dorsal_view_%s_proj_layer_%s_rgb.png" % (PROJECTION, layer_name)),
                dpi=600,
            )
            if SAVE_SVG:
                fig.savefig(
                    save_root
                    / (
                        "dorsal_view_%s_proj_layer_%s_rgb.svg"
                        % (PROJECTION, layer_name)
                    ),
                    dpi=600,
                )
else:
    print("Do not save", flush=True)

print("done")
