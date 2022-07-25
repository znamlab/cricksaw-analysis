"""Generate a coronal view of data"""
import matplotlib
import socket
if socket.gethostname() == 'C02Z85AULVDC':
    matplotlib.use('macosx')
from pathlib import Path
import numpy as np
import pandas as pd
import PIL
import tifffile
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
# brainglobe parts
import bg_atlasapi as bga
import bg_space
from imlib.cells.utils import get_cell_location_array
# flexiznam utils
import flexiznam as flz
from flexiznam.schema import Dataset
# local imports
from cricksaw_analysis import atlas_utils
# ensure txt is exported as txt in svg files:
plt.rcParams['svg.fonttype'] = 'none'


project = 'rabies_barcoding'
plane_per_section = 5
pixel_size = 2
mouse_filter = dict(Dilution='1/100', Zresolution=8)
plot_prop = {'BRJN101.5c':
                 dict(ENT=dict(xlim=[4250, 5200], ylim=[3050, 1550], z=350),
                      PIR=dict(xlim=(3550, 4350), ylim=(2900, 2070), z=1180),
                      AON=dict(xlim=(2750, 4070), ylim=(2980, 1900), z=1240),
                      COA=dict(xlim=[3700, 4500], ylim=[3680, 3200], z=650),
                      ),
             'BRJN101.5d':
                 dict(ENT=dict(xlim=[3900, 5200], ylim=[3000, 1700], z=330),
                      PIR=dict(xlim=(3400, 4300), ylim=(3000, 2200), z=1130),
                      AON=dict(xlim=(2750, 3900), ylim=(2980, 2000), z=1200),
                      COA=dict(xlim=[3700, 4400], ylim=[3580, 3000], z=570),),
             }
areas = ['ENT', 'PIR', 'AON', 'COA']
atlas_size = 10

atlas_name = 'allen_mouse_%dum' % atlas_size

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
    print("Doing %s" % mouse.name, flush=True)
    if mouse.name not in plot_prop:
        print('%s not in plot_prop, skip' % mouse.name)
        continue
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
    trans_hem = tifffile.imread(cellfinder_data.path_full / 'registration' /
                                'registered_hemispheres.tiff')
    trans_hem = bg_space.map_stack_to(source_origin, target_origin, trans_hem,
                                      copy=False)
    trans_atlas[trans_hem == 1] = 0
    fig.clear()
    fig.suptitle('Mouse %s' % mouse.name)
    for iax, parent_area in enumerate(areas):
        print('    doing %s' % parent_area, flush=True)
        area_names = [parent_area] + bg_atlas.get_structure_descendants(parent_area)
        area_ids = [bg_atlas.structures.acronym_to_id_map[a] for a in area_names]
        ok_cells = np.isin(atlas_id, area_ids)
        # keep only the right hemisphere
        midline = bg_atlas.annotation.shape[-1] / 2
        right = atlas_coord[:, 2] < midline
        ok_cells = np.logical_and(ok_cells, right)
        if parent_area in plot_prop[mouse.name]:
            best_plane = plot_prop[mouse.name][parent_area]['z']
        else:
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
            for zslice in range(plane_per_section):
                tif = chan_dir / ('section_%03d_%02d.tif' % (best_slice + 1,
                                                             zslice + 1))
                tif = tifffile.imread(tif)
                if channel not in img_data:
                    img_data[channel] = np.zeros(list(tif.shape) + [5], dtype=tif.dtype)
                img_data[channel][:, :, zslice] = tif

        for channel in [2, 3]:
            img_data[channel] = PROJECTION_FUNC(img_data[channel], axis=2)
        rgb = np.zeros(list(img_data[3].shape) + [3], dtype=np.uint8)
        contrast = np.percentile(img_data[3], [0, 99.9])
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

        zatlas = int(best_plane * mouse['Zresolution'] / atlas_size)
        label_img = PIL.Image.fromarray(trans_atlas[zatlas, :, :])
        label_img = np.asarray(label_img.resize(rgb.shape[1::-1],
                                                resample=PIL.Image.NEAREST))
        mprop = plot_prop[mouse.name]
        if parent_area in mprop:
            prop = mprop[parent_area]
            # crop img_data
            rgb = rgb[prop['ylim'][1]:prop['ylim'][0],
                      prop['xlim'][0]:prop['xlim'][1],
                      :]
            label_img = label_img[prop['ylim'][1]:prop['ylim'][0],
                                  prop['xlim'][0]:prop['xlim'][1]]

        in_plane = np.abs(cells[:, ap_axis] - best_plane) < 5

        ax = fig.add_subplot(2, 2, iax + 1)
        ax.imshow(rgb)

        o = atlas_utils.plot_borders_and_areas(ax, label_img, areas_to_plot=[],
                                               color_kwargs=dict(),
                                               cont_kwargs=dict(),
                                               label_atlas=bg_atlas)

        # ax.plot(cells[in_plane, ml_axis], cells[in_plane, dv_axis], 'o',
        #         alpha=0.2, mfc='none', mec='k')
        # ax.plot(cells[np.logical_and(ok_cells, in_plane), ml_axis],
        #         cells[np.logical_and(ok_cells, in_plane), dv_axis], 'o', color='lime',
        #         alpha=0.1, ms=3)
        ax.set_title(parent_area)

        # add a scale bar
        scale_bar = AnchoredSizeBar(ax.get_xaxis_transform(),
                                    size=500/pixel_size,
                                    label=r'$500\mu m$',
                                    loc='lower right',
                                    color='white',
                                    frameon=False,
                                    size_vertical=0.01)
        ax.add_artist(scale_bar)
        ax.set_aspect('equal')
        ax.set_axis_off()

    fig.subplots_adjust(top=0.95, bottom=0, left=0, right=1, wspace=0, hspace=0.05)
    # save output
    ds = Dataset.from_origin(origin_id=mouse.id, flexilims_session=flm_sess,
                             dataset_type='microscopy', conflicts='append')
    ds.flexilims_session = flm_sess
    ds.genealogy = list(ds.genealogy)[:-1] + ['coronal_view_%s.svg' % ('_'.join(areas))]
    ds.extra_attributes['atlas'] = atlas_name
    ds.extra_attributes['num_slices'] = plane_per_section
    ds.extra_attributes['projection'] = repr(PROJECTION_FUNC)
    ds.extra_attributes['pixel_size'] = pixel_size
    ds.path = ds.path.parent / ds.genealogy[-1]
    fig.savefig(ds.path_full, dpi=1200)
    print(ds.extra_attributes)
    ds.update_flexilims(mode='overwrite')



