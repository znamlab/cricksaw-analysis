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
import numpy as np
import tifffile
from napari.viewer import Viewer

from cricksaw_analysis import atlas_utils

PROCESSED = Path("/camp/lab/znamenskiyp/home/shared/projects")
PROJECT = "hey2_3d-vision_foodres_20220101"
MOUSE = "PZAH10.2b"
REGISTRATION = "brainreg"
ATLAS_SIZE = 25

if REGISTRATION == "brainreg":
    DATA_FOLDER = PROCESSED / PROJECT / MOUSE / "brainreg_results" / "from_downsampled"
    imgs_path = dict(
        red=Path(DATA_FOLDER) / "red" / "downsampled_standard.tiff",
        blue=None,
        green=Path(DATA_FOLDER) / "green" / "downsampled_standard.tiff",
    )
    ATLAS_ANNOTATION = None  # will load from brainglobe
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
PATH_TO_SAVE /= "stacks"
ATLAS_NAME = "allen_mouse_%dum" % ATLAS_SIZE
NAPARI = False

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
layers = ["1"]
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
top_of_layer = atlas_utils.external_view(
    peeled_atlas, axis="dorsal", border_only=False, get_index=True, which="first"
)
for l in layers:
    print("Doing %s" % l, flush=True)
    atlas_index[l] = top_of_layer
    layer_index = ctx_df.loc[ctx_df.layer == l, "id"].values
    # make a dorsal view of the atlas
    atlas_layer = np.zeros(x.shape, dtype=atlas_annot.dtype)
    atlas_layer[x, y] = atlas_annot[x, top_of_layer, y]
    atlas_dorsal_by_layer[l] = atlas_layer
    print("Peeling layer %s" % l, flush=True)
    layer_mask = np.isin(atlas_annot, layer_index)
    peeled_atlas[layer_mask] = 0
    print("Find surface of next layer", flush=True)
    bottom_of_layer = atlas_utils.external_view(
        peeled_atlas, axis="dorsal", border_only=False, get_index=True, which="first"
    )

    thickness = bottom_of_layer - top_of_layer
    max_diff = np.max(thickness)

    # save a stack
    for color, img_volume in image_volumes.items():
        data_view = np.zeros(x.shape, dtype=img_volume.dtype)
        zero, maxval_brain = np.quantile(img_volume, [0.0001, 0.9999])
        extent = maxval_brain - zero
        stack = []
        for name, i in enumerate(range(-10, 30)):
            img = img_volume[x, top_of_layer + i, y]
            img = (img - zero) * (2**16 / extent.max())
            img = np.array(np.clip(img, 0, 2**16), dtype="uint16")
            itk.imwrite(
                itk.image_from_array(img),
                PATH_TO_SAVE
                / (
                    "dorsal_view_stack_around_layer_%s_%s_%02d.png"
                    % (l.replace("/", ""), color, name)
                ),
            )
            stack.append(img)
        stack = np.stack(stack)
        tifffile.imwrite(
            PATH_TO_SAVE / (f"dorsal_view_stack_around_layer_1_{color}.tif"), stack
        )

        # do the extended depth of focus in FIJI
