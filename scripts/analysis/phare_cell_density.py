"""Script to look at the density of PhP.eB cre cells"""
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
    print('There are %s cells, with %s in V1' % (len(cells), len(v1c)))

    v1c_um = v1c * brain_pixel_size
    v1_tree = KDTree(v1c_um)
    dist, neigh = v1_tree.query(v1c_um, k=2, return_distance=True)
    # the closest neighbour is always yourself
    dist = dist[:, 1]

    fig.clear()
    ax0 = fig.add_subplot(2, 2, 1)
    ax0.hist(dist, bins=np.arange(500))
    ax0.set_xlabel('Distance to first neighbour')
    ax0.set_ylabel('Number of cells')