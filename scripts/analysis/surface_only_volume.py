"""
Create a version of the data registered to the atlas but with only few layers

"""

import socket
import time

import matplotlib

if socket.gethostname() == "C02Z85AULVDC":
    matplotlib.use("macosx")
from pathlib import Path

import brainglobe_atlasapi as bga
import flexiznam as fzm
import itk
import numpy as np
import SimpleITK as sitk
import yaml

from cricksaw_analysis import atlas_utils

PROCESSED = Path(fzm.PARAMETERS["data_root"]["processed"])
PROJECT = "hey2_3d-vision_foodres_20220101"
MOUSE = "PZAG3.4f"
REGISTRATION = "elastix"
ATLAS_SIZE = 10
OFFSET_FIRST = -200
OFFSET_LAST = 100
LEFT_HEM = False
VIS_ONLY = True
OFFSET_VIS_ANT = 100


if REGISTRATION == "brainreg":
    DATA_FOLDER = PROCESSED / PROJECT / MOUSE / "brainreg_results" / "from_downsampled"
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


if ATLAS_ANNOTATION is None:
    atlas_annot = bg_atlas.annotation
else:
    atlas_annot = itk.array_from_image(itk.imread(ATLAS_ANNOTATION))

# make a cortical atlas
print("Make cortex only atlas")
ctx_annot = np.array(atlas_annot)
is_ctx = np.isin(ctx_annot, ctx_df.loc[ctx_df.layer == "1", "id"].values)
ctx_annot[~is_ctx] = 0

print("Find starting layer")
start_ids = ctx_df.loc[ctx_df.layer == "1", "id"].values

# peel_atlas works on a list of list, here we want only one batch, so keep only the
# first element
peeled_first = atlas_utils.peel_atlas(
    ctx_annot, [start_ids], axis="dorsal", get_index=True, which="first", verbose=True
)[0]
peeled_last = atlas_utils.peel_atlas(
    ctx_annot, [start_ids], axis="dorsal", get_index=True, which="last", verbose=True
)[0]

# add offset if needed
peeled_first = np.clip(peeled_first + OFFSET_FIRST, 0, ctx_annot.shape[1] - 1)
peeled_last = np.clip(peeled_last + OFFSET_LAST, 0, ctx_annot.shape[1] - 1)

# get registered data
print("Loading img data")
image_volumes = dict()
for col, pa in imgs_path.items():
    if pa is None:
        continue
    print("   loading %s channel" % col)
    image_volumes[col] = itk.array_from_image(itk.imread(str(pa)))


x, y = np.meshgrid(*[np.arange(s) for s in peeled_first.shape])
x = np.array(x, dtype=int).T
y = np.array(y, dtype=int).T
thickness = peeled_last - peeled_first
max_diff = np.max(thickness)
surface_volumes = dict()
for color, img_volume in image_volumes.items():
    print("Creating volume in %s" % color)
    # create volume
    surface_data = np.zeros_like(img_volume)
    for i in range(max_diff):
        mask = thickness >= i
        # for max proj
        surface_data[x[mask], peeled_first[mask] + i, y[mask]] = img_volume[
            x[mask], peeled_first[mask] + i, y[mask]
        ]
    surface_volumes[color] = surface_data


# cut extra part
print("Cut extra part")
has_data = np.nonzero(surface_data)
data_min = [np.min(d) for d in has_data]
data_max = [np.max(d) for d in has_data]

if VIS_ONLY:
    # keep only visual area
    v = [a.startswith("VIS") and not a.startswith("VISC") for a in ctx_df.acronym]
    isvisual = np.isin(ctx_annot, ctx_df[v].id.values)
    isvisual = np.nonzero(isvisual)
    vis_min = [np.min(d) for d in isvisual]
    vis_max = [np.max(d) for d in isvisual]
    vis_min[1] = np.clip(vis_min[1] + OFFSET_FIRST, 0, surface_data.shape[1] - 1)
    vis_max[1] = np.clip(vis_max[1] + OFFSET_LAST, 0, surface_data.shape[1] - 1)
    data_min = [max(v, d) for v, d in zip(vis_min, data_min)]
    data_max = [min(v, d) for v, d in zip(vis_max, data_max)]

x_min, y_min, z_min = data_min
x_max, y_max, z_max = data_max

x_min -= OFFSET_VIS_ANT

if LEFT_HEM:
    midline = int(thickness.shape[1] / 2)
    z_min = midline


for color in surface_volumes:
    surface_volumes[color] = surface_volumes[color][
        x_min:x_max, y_min:y_max, z_min:z_max
    ]
bounding_dict = dict(
    first_axis=[int(x_min), int(x_max)],
    second_axis=[int(y_min), int(y_max)],
    third_axis=[int(z_min), int(z_max)],
)

# save output
print("Write outputs")
times = dict(start=time.time())
for color, data in surface_volumes.items():
    print(" ....  %s channel" % color)
    target = PATH_TO_SAVE / ("surface_registered_volume_%s.tiff" % color)
    mhd_image = sitk.GetImageFromArray(data)
    sitk.WriteImage(mhd_image, target)
    times[color] = time.time()
    print(target)
# Also save bounding box
target = PATH_TO_SAVE / "surface_registered_volume_bounding_box.yml"
with open(target, "w") as fhandle:
    yaml.dump(bounding_dict, fhandle)
