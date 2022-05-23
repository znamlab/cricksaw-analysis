import numpy as np
import pandas as pd
import bg_atlasapi as bga


def create_ctx_table(atlas=None):
    """Create a dataframe with cortical area info

    Args:
        atlas: either the name of a valid BrainGlobe atlas or the BrainGlobeAtlas
               instance
    Returns:
        ctx_df: a pandas DataFrame with information about cortical area and layers
    """

    if atlas is None:
        atlas = bga.bg_atlas.BrainGlobeAtlas('allen_mouse_25um')
    if isinstance(atlas, str):
        atlas = bga.bg_atlas.BrainGlobeAtlas(atlas)

    atlas.structures['Isocortex']['id']
    ctx_acr = atlas.get_structure_descendants('Isocortex')
    ctx_leaves = atlas.structures.tree.leaves(atlas.structures['Isocortex']['id'])
    ctx_df = []
    for node in ctx_leaves:
        data = atlas.structures[node.identifier]
        # bga.structures element come with a mesh that kinda crash. Remove it
        valid_key = [k for k in data.keys() if 'mesh' not in k.lower()]
        data = {k: data[k] for k in valid_key}

        full_name = data['name']
        parent_id = data['structure_id_path'][-2]
        parent = atlas.structures[parent_id]
        if 'layer' in full_name.lower():
            layer = full_name[full_name.lower().find('layer') + 5:].strip()
            data = pd.Series(dict(data, cortical_area=parent['name'],
                                  cortical_area_id=parent_id, layer=layer,
                                  area_acronym=parent['acronym']))
        elif data['id'] in [810, 819]:
            # two annoying area have layer specification but "layer" is not written
            layer = full_name.lower().split(' ')[-1].strip()
            data = pd.Series(dict(data, cortical_area=parent['name'],
                                  cortical_area_id=parent_id, layer=layer,
                                  area_acronym=parent['acronym']))
        else:
            data = pd.Series(dict(data, cortical_area=parent['name'],
                                  cortical_area_id=parent_id, layer='nd',
                                  area_acronym=parent['acronym']))
        ctx_df.append(data)
    return pd.DataFrame(ctx_df)


def external_view(atlas, axis='dorsal', border_only=False, get_index=False,
                  which='first'):
    """Get a view from atlas from one side

    axis can be dorsal, ventral, left, right, front, back

    if get_index, doesn't return the view but the index of the pixels generating the view

    if which is first return the surface. If which is last, return the first index
    after the surface that is out of the brain. It is not equivalent to changing the
    axis if you have multiple pieces of brain overlapping"""

    # reorder atlas to put the relevant view as first axis
    if axis in ['front', 'back']:
        # nothing to do
        proj_atlas = atlas
    elif axis in ['dorsal', 'ventral', 'top', 'bottom']:
        proj_atlas = np.array(atlas.transpose([1, 0, 2]))
    elif axis in ['left', 'right']:
        proj_atlas = np.array(atlas.transpose([2, 0, 1]))
    else:
        raise IOError('Unknown view. Type better')

    if axis in ['ventral', 'left', 'bottom', 'back']:
        proj_atlas = proj_atlas[::-1, :, :]
    # make the atlas int8 to gaian space and still have 0, 1 and -1
    bin_atlas = np.array(proj_atlas != 0, dtype='int8')

    # now find the first value not outside brain (i.e. not 0)
    pad = np.zeros([1] + list(bin_atlas.shape[1:]), dtype='int8')
    is_in_brain = np.diff(np.vstack([pad, bin_atlas]), axis=0)
    if which == 'first':
        # argmax return the first occurence of max (which is 1 here)
        first_in_brain = np.argmax(is_in_brain, axis=0)
    elif which == 'last':
        # argmin return the first occurence of min (which is -1 here)
        first_in_brain = np.argmin(is_in_brain, axis=0)
    else:
        raise IOError('Unknown type. Which should be first or last')
    if get_index:
        return first_in_brain
    # get the first value in brain for each
    x, y = np.meshgrid(*[np.arange(s) for s in first_in_brain.shape])
    proj_atlas = proj_atlas[first_in_brain, x.T, y.T]

    if border_only:
        proj_atlas = get_border(proj_atlas)
    return proj_atlas


def get_border(label_image, threed=False):
    """Get the borders of a 2d image"""
    if threed:
        stacked = np.vstack([np.zeros([1, label_image.shape[1], label_image.shape[2]]), label_image])
        diff_updowm = np.diff(stacked, axis=0) != 0
        stacked = np.hstack([np.zeros([label_image.shape[0], 1, label_image.shape[2]]), label_image])
        diff_leftright = np.diff(stacked, axis=1) != 0
        stacked = np.dstack([np.zeros([label_image.shape[0], label_image.shape[1], 1]), label_image])
        diff_frontback = np.diff(stacked, axis=2) != 0
        diff = diff_leftright + diff_updowm + diff_frontback
        return diff

    diff_updowm = np.diff(np.vstack([np.zeros([1, label_image.shape[1]]), label_image]), axis=0) != 0
    diff_leftright = np.diff(np.hstack([np.zeros([label_image.shape[0], 1]), label_image]), axis=1) != 0
    diff = diff_leftright + diff_updowm
    return diff


def cell_density_by_areas(atlas_id, atlas, cortex_df, pixel_size, bg_atlas):
    """Count the number of cells across areas and find area volume in this brain

    Args:
        atlas_id: id of the atlas for each cell
        atlas: atlas volume (ID of area per pixel, usually the registered to brain). Is
               used to find the volume of each area in this specific brain
        cortex_df: output of atlas_utils.create_ctx_table
        pixel_size: size of the pixel in atlas
        bg_atlas: brainglobe atlas instance (to get annotations)

    Returns:
        dict of dict with one element per area. Contains info about number of cells,
        number of pixels in the area and volume of the area
    """
    pixel_volume = (pixel_size / 1000)**3
    out = dict()

    # first do the cortex
    for c, adf in cortex_df.groupby('area_acronym'):
        n_cells = np.isin(atlas_id, adf.id)
        area = np.isin(atlas, adf.id)
        area_d = dict(count=np.sum(n_cells), size=np.sum(area))
        area_d['volume'] = np.sum(area) * pixel_volume
        area_d['density'] = area_d['count'] / area_d['volume']
        out[c] = area_d
    # now do rest of the brain
    all_ids = np.unique(atlas_id)
    is_ctx = np.isin(all_ids, cortex_df.id.values)
    rest_of_brain = all_ids[np.logical_not(is_ctx)]
    for area_id in rest_of_brain:
        if area_id == 0:
            # skip out of brain
            continue
        area_name = bg_atlas.structures.data[area_id]['acronym']
        n_cells = np.isin(atlas_id, area_id)
        area = np.isin(atlas, area_id)
        area_d = dict(count=np.sum(n_cells), size=np.sum(area))
        area_d['volume'] = np.sum(area) * pixel_volume
        area_d['density'] = area_d['count'] / area_d['volume']
        out[area_name] = area_d
    return out
