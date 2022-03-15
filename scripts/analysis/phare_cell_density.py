"""Script to look at the density of PhP.eB cre cells"""
# import matplotlib
# matplotlib.use('macosx')
import flexiznam as flz
from pathlib import Path
import numpy as np
import pandas as pd
import bg_atlasapi as bga
import bg_space as bg
import tifffile
from brainglobe_napari_io.cellfinder import reader_xml
from matplotlib import pyplot as plt
from sklearn.neighbors import KDTree

from cricksaw_analysis import atlas_utils

NAPARI = False
project = 'rabies_barcoding'
virus = 'A87'
brain_pixel_size = np.array([8, 2, 2], dtype=float)
atlas_pixel_size = 25.
cell_type = 'Cells'
channel = 0


def load_cellfinder_results(cellfinder_folder):
    """Loads the registration results

    Will load the downsampled data, downsampled cell position and downsampled atlas
    registered to the data

    Args:
        cellfinder_folder (str): path to the cellfinder folder. Must contain a
        'registration' and a 'points' folders

    Returns:
        pts (pd.Dataframe): cells in downsampled coordinates
        downsampled_stacks (dict): downsampled stacks, one per channels
        atlas (np.array): atlas registered to brain

    """
    cellfinder_folder = Path(cellfinder_folder)
    if not cellfinder_folder.is_dir():
        raise IOError('%s is not a directory' % cellfinder_folder)
    reg_folder = cellfinder_folder / 'registration'
    if not reg_folder.is_dir():
        raise IOError('Registration folder not found')
    pts_folder = cellfinder_folder / 'points'
    if not pts_folder.is_dir():
        raise IOError('Points folder not found')

    downsampled_stacks = {}
    for fname in reg_folder.glob('downsampled*.tiff'):
        if 'standard' in fname.stem:
            continue
        downsampled_stacks[fname.stem] = tifffile.imread(fname)

    atlas = tifffile.imread(reg_folder / 'registered_atlas.tiff')

    pts = pd.read_hdf(pts_folder / 'downsampled.points')

    return pts, downsampled_stacks, atlas


processed = Path(flz.PARAMETERS['data_root']['processed'])
raw = Path(flz.PARAMETERS['data_root']['raw'])

mouse_csv = processed / project / 'mice_list.csv'

mice_df = pd.read_csv(mouse_csv)
mice_df.set_index('Mouse', inplace=True, drop=False)

atlas = bga.bg_atlas.BrainGlobeAtlas('allen_mouse_25um')
cdf = atlas_utils.create_ctx_table(atlas)
fig = plt.figure(figsize=(10, 5))


for mouse, m_df in mice_df[mice_df['Virus Batch'] == 'A87'].iterrows():
    if mouse in ['BRJN104.2h', 'BRJN104.2i']:
        continue
    mouse_cellfinder_folder = processed / project / mouse / 'cellfinder_results'
    if not mouse_cellfinder_folder.is_dir():
        continue
    print('Doing %s' % mouse)
    cells, downsampled_stacks, atlas = load_cellfinder_results(
        mouse_cellfinder_folder)
    rd = np.array(np.round(cells.values), dtype=int)
    atlas_id = atlas[rd[:, 0], rd[:, 1], rd[:, 2]]

    # keep only cortical cells
    ctx_indices = cdf.id
    v1_indices = cdf.loc[cdf.area_acronym == 'VISp', 'id']
    cell_in_v1 = np.isin(atlas_id, v1_indices)
    cell_in_ctx = np.isin(atlas_id, ctx_indices)
    ctx_um = cells[cell_in_ctx] * atlas_pixel_size
    ctx_tree = KDTree(ctx_um)

    distance_radii = np.arange(0, 1000, 10)
    volume = 4/3 * np.pi * (distance_radii/1e3)**3
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

        neighbours[layer] = np.vstack([ctx_tree.query_radius(vlc_um, r, count_only=True)
                                       for r in distance_radii]) - 1
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
    rad = 200
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
    for layer, nei in neighbours.items():
        m = np.mean(nei, axis=1)
        ax3.plot(distance_radii, m, label=layer, lw=3 if layer == 'all' else 1,
                 zorder=3 if layer == 'all' else 1)
        if layer == 'all':
            ax4.plot(distance_radii, m/volume, label=layer, lw=3 if layer == 'all' else 1,
                     zorder=3 if layer == 'all' else 1)
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
