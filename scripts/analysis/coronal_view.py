"""Generate a coronal view of data"""
import matplotlib
matplotlib.use('macosx')
import os
import PIL
import numpy as np
import pandas as pd
import tifffile
from matplotlib import pyplot as plt
from pathlib import Path
# brainglobe parts
import bg_atlasapi as bga
import bg_space
from imlib.cells.utils import get_cell_location_array
# flexiznam utils
import flexiznam as flz
from flexiznam.schema import Dataset
# local imports
from cricksaw_analysis.io import load_cellfinder_results
from cricksaw_analysis import atlas_utils


project = 'rabies_barcoding'
plane_per_section = 5
mouse_filter = dict(Dilution='1/100', Zresolution=8)
areas = ['ENT', 'PIR', 'AON', 'COA']
atlas_name = 'allen_mouse_10um'

bg_atlas = bga.bg_atlas.BrainGlobeAtlas(atlas_name)
PROJECTION_FUNC = np.max  # must have a axis argument

raw = Path(flz.PARAMETERS['data_root']['raw'])

flm_sess = flz.get_flexilims_session(project_id=project)
mice = flz.get_entities(datatype='mouse', flexilims_session=flm_sess)
for filt, val in mouse_filter.items():
    mice = mice[mice[filt] == val].copy()

fig = plt.figure()

for _, mouse in mice.iterrows():
    # look if I have cellfinder data
    children = flz.get_children(mouse.id, flexilims_session=flm_sess,
                                children_datatype='dataset')
    cellfinder_data = children[(children.dataset_type == 'cellfinder') &
                               (children.atlas == atlas_name)]
    if len(cellfinder_data) > 1:
        raise IOError('Got %d cellfinder datasets' % len(cellfinder_data))
    cellfinder_data = Dataset.from_flexilims(data_series=cellfinder_data.iloc[0])

    cells = get_cell_location_array(str(cellfinder_data.path_full / 'points' /
                                        'cell_classification.xml'), cells_only=True)
    atlas_coord = pd.read_hdf(cellfinder_data.path_full / 'points' /
                              'atlas.points').values
    rd = np.array(np.round(atlas_coord), dtype=int)
    atlas_id = bg_atlas.annotation[rd[:, 0], rd[:, 1], rd[:, 2]]

    ap_axis = 2
    dv_axis = 1
    ml_axis = 0
    source_origin = ("Anterior", "Superior", "Right")
    target_origin = ("Posterior", "Superior", "Left")
    trans_atlas = tifffile.imread(cellfinder_data.path_full / 'registration' /
                                'registered_atlas.tiff')
    trans_atlas = bg_space.map_stack_to(source_origin, target_origin, trans_atlas,
                                        copy=False)

    for parent_area in areas:
        area_names = [parent_area] + bg_atlas.get_structure_descendants(parent_area)
        area_ids = [bg_atlas.structures.acronym_to_id_map[a] for a in area_names]
        ok_cells = np.isin(atlas_id, area_ids)
        # keep only the right hemisphere
        midline = bg_atlas.annotation.shape[-1] / 2
        right = atlas_coord[:, -1] > midline
        ok_cells = np.logical_and(ok_cells, right)
        ok_planes = np.round(cells[ok_cells, ap_axis])
        plane_count = pd.value_counts(ok_planes)
        best_plane = plane_count.sort_values().index[-1]
        # get the raw data for this plane
        raw_data_dir = raw / project / mouse.name / 'stitchedImages_100'
        assert raw_data_dir.is_dir()
        # find the total number of slice
        # nplanes = len([s for s in os.listdir(raw_data_dir / '3')
        #               if s.startswith('section')])

        best_slice = best_plane // plane_per_section
        best_slice_z = best_plane - best_slice * plane_per_section

        img_data = dict()
        for channel in [2, 3]:
            chan_dir = raw_data_dir / str(channel)
            assert chan_dir.is_dir()
            for zslice in range(5):
                tif = chan_dir / ('section_%03d_%02d.tif' % (best_slice + 1,
                                                             zslice + 1))
                tif = tifffile.imread(tif)
                if channel not in img_data:
                    img_data[channel] = np.zeros(list(tif.shape) + [5], dtype=tif.dtype)
                img_data[channel][:, :, zslice] = tif

        for channel in img_data:
            img_data[channel] = PROJECTION_FUNC(img_data[channel], axis=2)

        rgb = np.zeros(list(img_data[3].shape) + [3], dtype=np.uint8)
        contrast = np.percentile(img_data[3], [5, 99.9])
        red = np.array((img_data[3] - contrast[0]) / np.diff(contrast) * 255)
        red[red < 0] = 0
        red[red > 255] = 255
        red = np.array(red, dtype=np.uint8)
        contrast = np.percentile(img_data[2], [5, 99.99])
        green = np.array((img_data[2] - contrast[0]) / np.diff(contrast) * 255)
        green[green < 0] = 0
        green[green > 255] = 255
        green = np.array(green, dtype=np.uint8)
        rgb[:, :, 0] = red
        rgb[:, :, 1] = green
        rgb[:, :, 2] = green
        in_plane = np.abs(cells[:, ap_axis] - best_plane) < 5

        fig.clear()
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(rgb)
        ax.plot(cells[in_plane, ml_axis], cells[in_plane, dv_axis], 'o',
                alpha=0.2, mfc='none', mec='k')
        ax.plot(cells[np.logical_and(ok_cells, in_plane), ml_axis],
                cells[np.logical_and(ok_cells, in_plane), dv_axis], 'o', color='lime',
                alpha=0.2)
        ax.set_title(parent_area)

        label_img = np.asarray(
            PIL.Image.fromarray(
                trans_atlas[int(best_plane * mouse['Zresolution'] / 25),:,:]
            ).resize(rgb.shape[1::-1], resample=PIL.Image.NEAREST))

        o = atlas_utils.plot_borders_and_areas(ax, label_img, areas_to_plot=[],
                                               color_kwargs=dict(),
                                               border_dilatation=2, area_dilatation=0,
                                               contour_version=True, cont_kwargs=dict())
        ax.set_xlim(4200, 4800)
        ax.set_ylim(3070, 2700)

area_to_plot = dict(ENT=dict(xlim=[4000,5300], ylim=[3100,1500]))