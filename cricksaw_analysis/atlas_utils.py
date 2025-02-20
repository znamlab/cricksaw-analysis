import numpy as np
import pandas as pd
import bg_atlasapi as bga
from scipy.ndimage import binary_dilation


def plot_borders_and_areas(
    ax,
    label_img,
    areas_to_plot=None,
    color_kwargs=dict(),
    border_kwargs=dict(),
    label_kwargs=dict(),
    label_atlas=None,
):
    """Plot the atlas borders and highlight areas

    contour_version is very slow but vectorial

    Args:
        ax (matplotlib.Axes): matplotlib axis to draw the borders
        label_img (np.array): a 2d image of labels
        areas_to_plot (list, optional): a list (of list) of area ids to label. Each
            sublist will be grouped together (e.g. [10, [34,30]] will plot 2 contours,
            one for area 10 and one for the union of areas 34 and 30). If `label_atlas`
            is provided, `areas_to_plot` can also be a list of area acronyms
        color_kwargs (dict, optional): Keyword arguments for ax.contours of `areas_to_plot`
        border_kwargs (dict, optional): Keyword arguments for ax.contours of borders
        label_kwargs (dict, optional): Keyword arguments for ax.text of `areas_to_plot`
        label_atlas (Brainglobe atlas, optional): a brainglobe atlas. If provided, will add the
            name of each area in the center of that area (which might be at the midline
            for bilateral labels)
    Returns:
        contours: contours of the filled areas
    """
    if areas_to_plot is None:
        areas_to_plot = []

    kwargs = dict(vmin=0, vmax=len(areas_to_plot) + 1)
    kwargs.update(color_kwargs)
    cont_kwargs = dict(colors="Grey")
    cont_kwargs.update(border_kwargs)
    contours = []

    # first group subgroups of areas
    new_label = np.array(label_img)
    filled_areas = []

    def get_area_id_from_name_or_id(area):
        if isinstance(area, str):
            if label_atlas is None:
                raise IOError(
                    "`label_atlas` is required when providing acronyms "
                    + "as `areas_to_plot`."
                )
            atl_df = label_atlas.lookup_df
            area_id = atl_df.loc[atl_df.acronym == area, "id"]
            if area_id.shape[0] == 0:
                raise IOError(f"{area} is not a valid area name")
            area_id = area_id.iloc[0]
        else:
            area_id = area
        return area_id

    for area_subgroup in areas_to_plot:
        if (isinstance(area_subgroup, list) 
            or isinstance(area_subgroup, np.ndarray) 
            or isinstance(area_subgroup, tuple)
        ):
            ref_id = get_area_id_from_name_or_id(area_subgroup[0])
            filled_areas.append(ref_id)
            for area in area_subgroup:
                area_id = get_area_id_from_name_or_id(area)
                new_label[new_label == area_id] = ref_id
        else:
            filled_areas.append(get_area_id_from_name_or_id(area_subgroup))

    # create an image with continuous labelling of areas
    new_ids = np.unique(new_label)
    dtypes = ["uint8", "uint16", "uint32"]
    dtype_limit = [256, 65536, 4294967295]
    dtype = dtypes[np.searchsorted(dtype_limit, len(new_ids))]
    bin_image = np.zeros(new_label.shape, dtype=dtype)
    sorted_labels = np.sort(new_ids)
    for iar, area in enumerate(sorted_labels):
        bin_image[new_label == area] = iar + 1
    ax.contour(bin_image, levels=np.arange(0.5, len(new_ids) + 1, 0.5), **cont_kwargs)

    for area_id in filled_areas:
        # create an image with 1 in the area and 0 everywhere else

        bin_image *= 0
        bin_image[new_label == area_id] = 1
        i_area = filled_areas.index(area_id)
        bin_image[new_label == area] += i_area
        contours.append(
            ax.contourf(bin_image, levels=np.array([0.5, 1.5]) + i_area, **kwargs)
        )

    if label_atlas is not None:
        text_kw = dict(color="w", verticalalignment="center", horizontalalignment="center")
        text_kw.update(label_kwargs)
        atlas_vals = np.sort(np.unique(label_img))
        if atlas_vals[0] == 0:
            atlas_vals = atlas_vals[1:]  # remove 0
        atl_df = label_atlas.lookup_df
        for ic, area_id in enumerate(atlas_vals):
            leg = atl_df.loc[atl_df.id == area_id, "acronym"]
            if leg.shape[0] == 0:
                continue
            leg = leg.iloc[0]
            coords = np.vstack(np.where(label_img == area_id))
            mid_point = np.nanmedian(coords, 1)
            ax.text(
                mid_point[1],
                mid_point[0],
                s=leg,
                **text_kw,
            )

    return contours


def create_ctx_table(atlas=None):
    """Create a dataframe with cortical area info

    Args:
        atlas: either the name of a valid BrainGlobe atlas or the BrainGlobeAtlas
               instance
    Returns:
        ctx_df: a pandas DataFrame with information about cortical area and layers
    """

    if atlas is None:
        atlas = bga.bg_atlas.BrainGlobeAtlas("allen_mouse_25um")
    if isinstance(atlas, str):
        atlas = bga.bg_atlas.BrainGlobeAtlas(atlas)

    atlas.structures["Isocortex"]["id"]
    ctx_acr = atlas.get_structure_descendants("Isocortex")
    ctx_leaves = atlas.structures.tree.leaves(atlas.structures["Isocortex"]["id"])
    ctx_df = []
    for node in ctx_leaves:
        data = atlas.structures[node.identifier]
        # bga.structures element come with a mesh that kinda crash. Remove it
        valid_key = [k for k in data.keys() if "mesh" not in k.lower()]
        data = {k: data[k] for k in valid_key}

        full_name = data["name"]
        parent_id = data["structure_id_path"][-2]
        parent = atlas.structures[parent_id]
        if "layer" in full_name.lower():
            layer = full_name[full_name.lower().find("layer") + 5 :].strip()
            data = pd.Series(
                dict(
                    data,
                    cortical_area=parent["name"],
                    cortical_area_id=parent_id,
                    layer=layer,
                    area_acronym=parent["acronym"],
                )
            )
        elif data["id"] in [810, 819]:
            # two annoying area have layer specification but "layer" is not written
            layer = full_name.lower().split(" ")[-1].strip()
            data = pd.Series(
                dict(
                    data,
                    cortical_area=parent["name"],
                    cortical_area_id=parent_id,
                    layer=layer,
                    area_acronym=parent["acronym"],
                )
            )
        else:
            data = pd.Series(
                dict(
                    data,
                    cortical_area=parent["name"],
                    cortical_area_id=parent_id,
                    layer="nd",
                    area_acronym=parent["acronym"],
                )
            )
        ctx_df.append(data)
    return pd.DataFrame(ctx_df)


def peel_atlas(
    atlas, peel_list, axis="dorsal", get_index=False, which="first", verbose=True
):
    """Remove sequentially areas and generate external view

    This function will make an external view, remove first group of IDs in peel_list, make
    an external view and repeat. This effectively peels the atlas bit by bits.

    It returns the atlas ID in

    Args:
        atlas: annotation volume
        peel_list: list (of list) of area IDs to sequentially remove

    Returns:
        peeled_atlas_id: len(peel_list) list of external view with either the atlas ID
                         of each view (default, if get_index=False) or the index
                         generating this view
    """
    peeled_atlas = np.array(atlas, copy=True)  # make a copy to remove layers
    out = []
    for peel_index, atlas_ids in enumerate(peel_list):
        if verbose:
            print("Peeling level %d/%d" % (peel_index + 1, len(peel_list)), flush=True)
        layer_mask = np.isin(atlas, atlas_ids)
        peeled_atlas[layer_mask] = 0
        if verbose:
            print("   external view")
            out.append(external_view(atlas, axis, get_index=get_index, which=which))
    return out


def external_view(
    atlas, axis="dorsal", border_only=False, get_index=False, which="first"
):
    """Get a view from atlas from one side

    axis can be dorsal, ventral, left, right, front, back

    if get_index, doesn't return the view but the index of the pixels generating the view

    if which is first return the surface. If which is last, return the first index
    after the surface that is out of the brain. It is not equivalent to changing the
    axis if you have multiple pieces of brain overlapping"""

    # reorder atlas to put the relevant view as first axis
    if axis in ["front", "back"]:
        # nothing to do
        proj_atlas = atlas
    elif axis in ["dorsal", "ventral", "top", "bottom"]:
        proj_atlas = np.array(atlas.transpose([1, 0, 2]))
    elif axis in ["left", "right"]:
        proj_atlas = np.array(atlas.transpose([2, 0, 1]))
    else:
        raise IOError("Unknown view. Type better")

    if axis in ["ventral", "left", "bottom", "back"]:
        proj_atlas = proj_atlas[::-1, :, :]
    # make the atlas int8 to gaian space and still have 0, 1 and -1
    bin_atlas = np.array(proj_atlas != 0, dtype="int8")

    # now find the first value not outside brain (i.e. not 0)
    pad = np.zeros([1] + list(bin_atlas.shape[1:]), dtype="int8")
    is_in_brain = np.diff(np.vstack([pad, bin_atlas]), axis=0)
    if which == "first":
        # argmax return the first occurence of max (which is 1 here)
        first_in_brain = np.argmax(is_in_brain, axis=0)
    elif which == "last":
        # argmin return the first occurence of min (which is -1 here)
        first_in_brain = np.argmin(is_in_brain, axis=0)
    else:
        raise IOError("Unknown type. Which should be first or last")
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
        stacked = np.vstack(
            [np.zeros([1, label_image.shape[1], label_image.shape[2]]), label_image]
        )
        diff_updowm = np.diff(stacked, axis=0) != 0
        stacked = np.hstack(
            [np.zeros([label_image.shape[0], 1, label_image.shape[2]]), label_image]
        )
        diff_leftright = np.diff(stacked, axis=1) != 0
        stacked = np.dstack(
            [np.zeros([label_image.shape[0], label_image.shape[1], 1]), label_image]
        )
        diff_frontback = np.diff(stacked, axis=2) != 0
        diff = diff_leftright + diff_updowm + diff_frontback
        return diff

    diff_updowm = (
        np.diff(np.vstack([np.zeros([1, label_image.shape[1]]), label_image]), axis=0)
        != 0
    )
    diff_leftright = (
        np.diff(np.hstack([np.zeros([label_image.shape[0], 1]), label_image]), axis=1)
        != 0
    )
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
    pixel_volume = (pixel_size / 1000) ** 3
    out = dict()

    # first do the cortex
    for c, adf in cortex_df.groupby("area_acronym"):
        n_cells = np.isin(atlas_id, adf.id)
        area = np.isin(atlas, adf.id)
        area_d = dict(count=np.sum(n_cells), size=np.sum(area))
        area_d["volume"] = np.sum(area) * pixel_volume
        area_d["density"] = area_d["count"] / area_d["volume"]
        out[c] = area_d
    # now do rest of the brain
    all_ids = np.unique(atlas_id)
    is_ctx = np.isin(all_ids, cortex_df.id.values)
    rest_of_brain = all_ids[np.logical_not(is_ctx)]
    for area_id in rest_of_brain:
        if area_id == 0:
            # skip out of brain
            continue
        area_name = bg_atlas.structures.data[area_id]["acronym"]
        n_cells = np.isin(atlas_id, area_id)
        area = np.isin(atlas, area_id)
        area_d = dict(count=np.sum(n_cells), size=np.sum(area))
        area_d["volume"] = np.sum(area) * pixel_volume
        area_d["density"] = area_d["count"] / area_d["volume"]
        out[area_name] = area_d
    return out


def get_area_ids(areas, label_atlas, get_descendants=False):
    """Return the area ids from a list of area names or ids
    
    Args:
        area: single area name of ids
        label_atlas: brainglobe atlas instance
        get_descendants: if True, will return the descendants of the area
    
    Returns:
        dict
    """
    if isinstance(areas, str) or isinstance(areas, int):
        areas = [areas]
    
    output = {}
    for area in areas:
        if isinstance(area, str):
            atl_df = label_atlas.lookup_df
            area_id = atl_df.loc[atl_df.acronym == area, "id"]
            if area_id.shape[0] == 0:
                raise IOError(f"{area} is not a valid area name")
            area_id = area_id.iloc[0]
            acronym = area
        else:
            area_id = area
            acronym = label_atlas.structures[area]['acronym']
        
        id_list = [area_id]
        
        def _get_descendant(area_id, label_atlas):
            descendants = label_atlas.get_structure_descendants(area_id)
            desc_ids = [label_atlas.structures[desc]['id'] for desc in descendants]
            for desc in descendants:
                desc_ids += _get_descendant(desc, label_atlas)
            return desc_ids
        
        if get_descendants:
            id_list += _get_descendant(area_id, label_atlas)

        output[acronym] = id_list
    return output

