"""Script to look at the density of PhP.eB cre cells"""
# import matplotlib
# matplotlib.use('macosx')
import flexiznam as flz
from pathlib import Path
import numpy as np
import pandas as pd
import bg_atlasapi as bga
from matplotlib import pyplot as plt
from sklearn.neighbors import KDTree

from cricksaw_analysis import atlas_utils
from cricksaw_analysis.io import load_cellfinder_results

NAPARI = False
PLOT_DENSITY = False
PLOT_SUMMARY = True
project = 'rabies_barcoding'
virus = 'A87'
brain_pixel_size = np.array([8, 2, 2], dtype=float)
atlas_pixel_size = 25.
cell_type = 'Cells'
channel = 0
cortical_areas = 'ALL'


processed = Path(flz.PARAMETERS['data_root']['processed'])
raw = Path(flz.PARAMETERS['data_root']['raw'])

mouse_csv = processed / project / 'mice_list.csv'

mice_df = pd.read_csv(mouse_csv, skipinitialspace=True)
mice_df.set_index('Mouse', inplace=True, drop=False)

bg_atlas = bga.bg_atlas.BrainGlobeAtlas('allen_mouse_25um')
cdf = atlas_utils.create_ctx_table(bg_atlas)
fig = plt.figure(figsize=(10, 5))

if cortical_areas == 'ALL':
    cortical_areas = list(cdf.area_acronym.unique())


def count_cells_by_areas(atlas_id, atlas, cortex_df=cdf,
                         pixel_size=atlas_pixel_size):
    pixel_volume = (pixel_size / 1000)**3
    out = dict()
    for c, adf in cortex_df.groupby('area_acronym'):
        n_cells = np.isin(atlas_id, adf.id)
        area = np.isin(atlas, adf.id)
        out[c] = dict(count=np.sum(n_cells), size=np.sum(area),
                      volume=np.sum(area) * pixel_volume)
    return out


def calculate_neighbourhood(distance_radii, kdtree, cells, cdf=cdf):
    neighbours = dict()
    distance = dict()
    for layer in ['all', '1', '2/3', '4', '5', '6a', '6b']:
        v1_indices = cdf.loc[cdf.area_acronym == 'VISp', :]
        if layer == 'all':
            v1_indices = v1_indices.id
        else:
            v1_indices = v1_indices.loc[v1_indices.layer == layer, 'id']
        cells_in_layer = np.isin(atlas_id, v1_indices)
        vlc_um = cells[cells_in_layer] * atlas_pixel_size
        dist, neigh = ctx_tree.query(vlc_um, k=2, return_distance=True)
        # the closest neighbour is always yourself
        distance[layer] = dist[:, 1]

        neighbours[layer] = np.vstack([kdtree.query_radius(vlc_um, r, count_only=True)
                                       for r in distance_radii]) - 1
    return neighbours, distance


summary_density = dict()
for mouse, m_df in mice_df[mice_df['Virus Batch'] == 'A87'].iterrows():
    if mouse in []:
        continue
    mouse_cellfinder_folder = processed / project / mouse / 'cellfinder_results'
    if not mouse_cellfinder_folder.is_dir():
        print('No cellfinder folder for %s' % mouse)
        print(mouse_cellfinder_folder)
        continue
    print('Doing %s' % mouse)
    try:
        cells, downsampled_stacks, atlas = load_cellfinder_results(
                                            mouse_cellfinder_folder)
    except IOError or FileNotFoundError as err:
        print('Failed to load data: %s' % err)
        continue
    rd = np.array(np.round(cells.values), dtype=int)
    atlas_id = atlas[rd[:, 0], rd[:, 1], rd[:, 2]]
    if PLOT_SUMMARY:
        summary_density[mouse] = count_cells_by_areas(atlas_id, atlas)

    if PLOT_DENSITY:
        # keep only cortical cells
        ctx_indices = cdf.id
        v1_indices = cdf.loc[cdf.area_acronym == 'VISp', 'id']
        cell_in_v1 = np.isin(atlas_id, v1_indices)
        cell_in_ctx = np.isin(atlas_id, ctx_indices)
        ctx_um = cells[cell_in_ctx] * atlas_pixel_size
        ctx_tree = KDTree(ctx_um)

        distance_radii = np.arange(0, 1000, 10)
        volume = 4/3 * np.pi * (distance_radii/1e3)**3
        neighbours, distance = calculate_neighbourhood(distance_radii,
                                                       kdtree=ctx_tree,
                                                       cells=cells)
        fig.clear()
        ax0 = fig.add_subplot(2, 3, 1)
        ax0.set_title('Total: %d cells, Cortex %d cells, V1: %d cells'
                      % (len(cells), np.sum(cell_in_ctx), np.sum(cell_in_v1)))
        cutoff = 1000
        n, b, p = ax0.hist(distance['all'], bins=np.arange(0, cutoff, 5))
        ax0.set_xlim([0, b[:-1][n > 0][-1]])
        ax0.set_xlabel('Distance to first neighbour')
        ax0.set_ylabel('Number of cells')

        ax1 = fig.add_subplot(2, 3, 2)
        rad = 100
        radinx = distance_radii.searchsorted(rad)
        ax1.hist(neighbours['all'][radinx], bins=np.arange(-0.5, 100.5))
        ax1.set_xlim(-0.5, 100)
        ax1.set_xlabel('# of neighbours in %d um radius' % rad)
        ax1.set_ylabel('# of cells')

        ax2 = fig.add_subplot(2, 3, 3)
        ax2.clear()
        ax2.set_ylabel('Fraction of cells\nwith Ns neighbour')
        ax2.set_xlabel('Distance')
        for n in [1, 5, 10, 20, 50]:
            has_neigh = neighbours['all'] >= n
            frac = np.sum(has_neigh, axis=1) / len(has_neigh[0])
            ax2.plot(distance_radii, frac, label=str(n))
        ax2.set_xlim([0, distance_radii[frac.argmax()]])
        ax2.set_ylim([0, 1])
        ax2.legend(loc=0)

        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)
        ax3.set_xlabel('Distance')
        ax3.set_ylabel('Mean number of neighbours')
        ax4.set_xlabel('Distance')
        ax4.set_ylabel('Mean cell denstity (cells/mm$^3$)')

        cutoff = 500
        beg_density = np.searchsorted(distance_radii, 50)
        for layer, nei in neighbours.items():
            m = np.mean(nei, axis=1)
            std = np.std(nei, axis=1)
            ax3.plot(distance_radii, m, label=layer, lw=3 if layer == 'all' else 1,
                     zorder=3 if layer == 'all' else 1)
            if layer == 'all':
                dens = (nei[beg_density:, :].T / volume[beg_density:]).T
                m = np.mean(dens, axis=1)
                std = np.std(dens, axis=1)
                ax4.plot(distance_radii[beg_density:], m, label=layer, lw=3,
                         zorder=3)
                ax4.fill_between(distance_radii[beg_density:], m-std, m+std,
                                 color='grey', alpha=0.4)
                ax3.set_ylim([0, m[distance_radii.searchsorted(cutoff)]])
        ax3.set_xlim([0, cutoff])
        ax4.set_xlim([0, cutoff])
        ax3.legend(loc='upper center', bbox_to_anchor=(
            0.5, 1.05), ncol=4, fontsize=6)
        fig.subplots_adjust(wspace=0.4, hspace=0.4)
        fig.suptitle('%s - dilution %s' % (mouse, m_df.Dilution))

        fig.savefig(processed / project / mouse / ('%s_neighbour_by_distance.png' % mouse),
                    dpi=600)
        fig.savefig(processed / project / mouse / ('%s_neighbour_by_distance.svg' % mouse),
                    dpi=600)
        if NAPARI:
            from napari.viewer import Viewer
            viewer = Viewer()
            img = downsampled_stacks['downsampled_channel_%d' % channel]
            viewer.add_image(img, name='Channel %d' % channel)
            viewer.add_labels(atlas, name='Registered atlas', opacity=0.1)
            viewer.add_points(cells, name='Cells', face_color='blue', opacity=0.2,
                              size=5)
            viewer.add_points(cells[cell_in_v1], name='V1 Cells',
                              face_color='orange', opacity=0.5, size=5)

if PLOT_SUMMARY:
    # we got data for all mice
    # get area volumes
    area_prop = dict()
    px_volume = (atlas_pixel_size / 1000)**3
    for c, adf in cdf.groupby('area_acronym'):
        area_prop[c] = dict()
        area_prop[c]['n_pixels'] = np.sum(np.isin(bg_atlas.annotation, adf.id))
        area_prop[c]['volume'] = area_prop[c]['n_pixels'] * px_volume

    count_df = pd.DataFrame(summary_density)
    prop = pd.DataFrame(area_prop)
