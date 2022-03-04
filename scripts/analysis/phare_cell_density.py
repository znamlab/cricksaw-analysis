"""Script to look at the density of PhP.eB cre cells"""
import matplotlib
matplotlib.use('macosx')
import flexiznam as flz
from pathlib import Path
import numpy as np
import pandas as pd
import bg_atlasapi as bga
import tifffile
from brainglobe_napari_io.cellfinder import reader_xml
from matplotlib import pyplot as plt
from sklearn.neighbors import KDTree

from cricksaw_analysis import atlas_utils

project = 'rabies_barcoding'
virus = 'A87'
brain_pixel_size = np.array([8, 2, 2], dtype=float)
atlas_pixel_size = 25.
cell_type = 'Cells'

processed = Path(flz.PARAMETERS['data_root']['processed'])
raw = Path(flz.PARAMETERS['data_root']['raw'])

mouse_csv = processed / project / 'mice_list.csv'

mice_df = pd.read_csv(mouse_csv)
mice_df.set_index('Mouse', inplace=True, drop=False)

atlas = bga.bg_atlas.BrainGlobeAtlas('allen_mouse_25um')
cdf = atlas_utils.create_ctx_table(atlas)
fig = plt.figure()
for mouse, m_df in mice_df[mice_df['Virus Batch'] == 'A87'].iterrows():
    mouse_cellfinder_folder = processed / project / mouse / 'cellfinder_results'
    if not mouse_cellfinder_folder.is_dir():
        continue
    print('Doing %s' % mouse)
    cell_classification = mouse_cellfinder_folder / 'points' / 'cell_classification.xml'
    if not cell_classification.is_file():
        print('   no cell classification. Did you run cellfinder?')
        continue
    cell_classification = reader_xml.xml_reader(cell_classification)
    cell_classification = {c[1]['name']: c for c in cell_classification}
    cells = cell_classification[cell_type][0]

    px_cells = np.array(np.round(cells * brain_pixel_size / atlas_pixel_size),
                        dtype=int)
    reg_atlas = mouse_cellfinder_folder / 'registration' / 'registered_atlas.tiff'
    if not reg_atlas.is_file():
        print('  no registered atlas. Did you run cellfinder?')
        continue
    reg_atlas = tifffile.imread(reg_atlas)
    atlas_id = reg_atlas[px_cells[:, 0], px_cells[:, 1], px_cells[:, 2]]

    # keep only V1 cells
    v1_indices = cdf.loc[cdf.area_acronym == 'VISp', 'id']
    cell_in_v1 = np.isin(atlas_id, v1_indices)
    v1c = cells[cell_in_v1, :]
    v1c_um = v1c * brain_pixel_size
    v1_tree = KDTree(v1c_um)

    distance_radii = np.arange(0, 1000, 10)
    neighbours = dict()
    distance = dict()
    for layer in ['all', '1', '2/3', '4', '5', '6a', '6b']:
        v1_indices = cdf.loc[cdf.area_acronym == 'VISp', :]
        if layer == 'all':
            v1_indices = v1_indices.id
        else:
            v1_indices = v1_indices.loc[v1_indices.layer == layer, 'id']
        cells_in_layer = np.isin(atlas_id, v1_indices)
        vlc_um = cells[cells_in_layer, :] * brain_pixel_size
        dist, neigh = v1_tree.query(vlc_um, k=2, return_distance=True)
        # the closest neighbour is always yourself
        distance[layer] = dist[:, 1]

        neighbours[layer] = np.vstack([v1_tree.query_radius(vlc_um, r, count_only=True)
                                       for r in distance_radii]) - 1
    fig.clear()
    ax0 = fig.add_subplot(2, 2, 1)
    ax0.set_title('Total: %d cells, V1: %d cells' % (len(cells), len(v1c)))
    ax0.hist(distance['all'], bins=np.arange(0, 200, 5))
    ax0.set_xlabel('Distance to first neighbour')
    ax0.set_ylabel('Number of cells')

    ax1 = fig.add_subplot(2, 2, 2)
    rad = 100
    radinx = distance_radii.searchsorted(rad)
    ax1.hist(neighbours['all'][radinx], bins=np.arange(-0.5, 20.5))
    ax1.set_xlim(-0.5, 20)
    ax1.set_xlabel('# of neighbours in %d um radius' % rad)
    ax1.set_ylabel('# of cells')

    ax2 = fig.add_subplot(2, 2, 3)
    ax2.clear()
    ax2.set_ylabel('Fraction of cells with Ns neighbour')
    ax2.set_xlabel('Distance')
    for n in [1, 5, 10, 20]:
        has_neigh = neighbours['all'] >= n
        frac = np.sum(has_neigh, axis=1) / len(has_neigh[0])
        ax2.plot(distance_radii, frac, label=str(n))
    ax2.set_xlim([0, distance_radii[frac.argmax()]])
    ax2.set_ylim([0, 1])
    ax2.legend(loc=0)

    ax3 = fig.add_subplot(2, 2, 4)
    ax3.set_xlabel('Distance')
    ax3.set_ylabel('Average number of neighbours')
    for layer, nei in neighbours.items():
        m = np.mean(nei, axis=1)
        ax3.plot(distance_radii, m, label=layer)
        if layer == 'all':
            ax3.set_ylim([0, m[distance_radii.searchsorted(200)]])
    ax3.set_xlim([0, 200])
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4, fontsize=6)
    fig.subplots_adjust(wspace=0.4, hspace=0.4)
    fig.suptitle('%s - dilution %s' % (mouse, m_df.Dilution))

    fig.savefig(processed / project / mouse / ('%s_neighbour_by_distance.png' % mouse),
                dpi=600)
    fig.savefig(processed / project / mouse / ('%s_neighbour_by_distance.svg' % mouse),
                dpi=600)
