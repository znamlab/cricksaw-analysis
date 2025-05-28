import warnings
from pathlib import Path

import brainglobe_atlasapi as bga
import ccf_streamlines.projection as ccfproj
import numpy as np
import pandas as pd
import requests
import tqdm
from six import BytesIO


def plot_borders_and_areas(
    ax,
    label_img,
    areas_to_plot=None,
    color_kwargs=dict(),
    border_kwargs=dict(),
    label_kwargs=dict(),
    label_atlas=None,
    get_descendants=False,
    plot_borders=True,
    label_filled_areas=True,
    label_all_areas=False,
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
        color_kwargs (dict, optional): Keyword arguments for ax.contours of
            `areas_to_plot`
        border_kwargs (dict, optional): Keyword arguments for ax.contours of borders
        label_kwargs (dict, optional): Keyword arguments for ax.text of `areas_to_plot`
        label_atlas (Brainglobe atlas, optional): a brainglobe atlas. If provided, will
            add the name of each area in the center of that area (which might be at the
            midline for bilateral labels)
        get_descendants (bool, optional): if True, will also plot the descendants of the
            areas in `areas_to_plot`
        plot_borders (bool, optional): if True, will plot the borders of all areas
        label_filled_areas (bool, optional): if True, will label areas in areas_to_plot
        label_all_areas (bool, optional): if True, will label all areas in the atlas
    Returns:
        contours: contours of the filled areas
    """
    if areas_to_plot is None:
        areas_to_plot = []

    kwargs = dict(vmin=0, vmax=len(areas_to_plot) + 1)
    kwargs.update(color_kwargs)
    cont_kwargs = dict(colors="Grey")
    cont_kwargs.update(border_kwargs)
    text_kw = dict(color="w", verticalalignment="center", horizontalalignment="center")
    text_kw.update(label_kwargs)

    contours = []

    # first group subgroups of areas
    new_label = np.array(label_img)
    filled_areas = []
    to_label = {}
    for area_subgroup in areas_to_plot:
        area_to_fill = get_area_ids(area_subgroup, label_atlas, get_descendants)
        # paint all the areas in the subgroup with the same color
        main_id = None
        for acr, ids in area_to_fill.items():
            if main_id is None:
                main_id = ids[0]
                to_label[acr] = main_id
            for area_id in ids:
                new_label[new_label == area_id] = main_id
        filled_areas.append(main_id)

    dtypes = ["uint8", "uint16", "uint32"]
    dtype_limit = [256, 65536, 4294967295]

    if plot_borders:
        # create an image with continuous labelling of areas
        new_ids = np.unique(new_label)
        dtype = dtypes[np.searchsorted(dtype_limit, len(new_ids))]
        sorted_labels = np.sort(new_ids)
        bin_image = np.zeros(new_label.shape, dtype=dtype)
        for iar, area in enumerate(sorted_labels):
            bin_image[new_label == area] = iar + 1
        ax.contour(
            bin_image, levels=np.arange(0.5, len(new_ids) + 1, 0.5), **cont_kwargs
        )

    dtype = dtypes[np.searchsorted(dtype_limit, len(to_label))]
    bin_image = np.zeros(new_label.shape, dtype=dtype)
    for name, area_id in to_label.items():
        # create an image with 1 in the area and 0 everywhere else
        bin_image *= 0
        bin_image[new_label == area_id] = 1
        if not np.any(bin_image):
            continue
        contours.append(ax.contourf(bin_image, levels=np.array([0.5, 1.5]), **kwargs))
        if label_filled_areas:
            coords = np.vstack(np.where(bin_image))
            mid_point = np.nanmedian(coords, 1)
            ax.text(
                mid_point[1],
                mid_point[0],
                s=name,
                **text_kw,
            )

    if label_all_areas:
        atlas_vals = np.sort(np.unique(label_img))
        if atlas_vals[0] == 0:
            atlas_vals = atlas_vals[1:]  # remove 0
        to_label = {}
        for area_id in atlas_vals:
            to_label[label_atlas.lookup_df.query("id == area_id").iloc[0].acronym] = (
                area_id
            )
        labeled_atlas = label_img
        for name, area_id in to_label.items():
            if area_id == 0:
                continue
            coords = np.vstack(np.where(labeled_atlas == area_id))
            mid_point = np.nanmedian(coords, 1)
            ax.text(
                mid_point[1],
                mid_point[0],
                s=name,
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
                    layer="nd",  # codespell:ignore nd
                    area_acronym=parent["acronym"],
                )
            )
        ctx_df.append(data)
    return pd.DataFrame(ctx_df)


def peel_atlas(
    atlas, peel_list, axis="dorsal", get_index=False, which="first", verbose=True
):
    """Remove sequentially areas and generate external view

    This function will make an external view, remove first group of IDs in peel_list,
    make an external view and repeat. This effectively peels the atlas bit by bits.

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

    if get_index is True, doesn't return the view but the index of the pixels generating
    the view

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
        # argmax return the first occurrence of max (which is 1 here)
        first_in_brain = np.argmax(is_in_brain, axis=0)
    elif which == "last":
        # argmin return the first occurrence of min (which is -1 here)
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
            acronym = label_atlas.structures[area]["acronym"]

        id_list = [area_id]

        def _get_descendant(area_id, label_atlas):
            descendants = label_atlas.get_structure_descendants(area_id)
            desc_ids = [label_atlas.structures[desc]["id"] for desc in descendants]
            for desc in descendants:
                desc_ids += _get_descendant(desc, label_atlas)
            return desc_ids

        if get_descendants:
            id_list += _get_descendant(area_id, label_atlas)

        output[acronym] = id_list
    return output


def plot_coronal_view(
    ax, plane, label_atlas, hemisphere="both", area_colors={}, alpha=0.2
):
    """Plot a coronal view of the atlas

    Args:

        ax: Axes on which to plot
        plane: Plane of the view in atlas voxel coordinates
        label_atlas: Brainglobe atlas instance
        hemisphere: Hemisphere to plot (left, right, both)
        area_colors: Dictionary of area acronyms to their colors
        alpha: Opacity of the plotted areas

    Returns:
        cor_atlas: The coronal view of the atlas
    """
    midline = label_atlas.annotation.shape[2] // 2
    # Coronal view
    if hemisphere == "left":
        cor_atlas = np.array(label_atlas.annotation[plane, :, :midline])
    elif hemisphere == "right":
        cor_atlas = np.array(label_atlas.annotation[plane, :, midline:])
    else:
        cor_atlas = np.array(label_atlas.annotation[plane, :, :])

    cor_borders = get_border(cor_atlas)
    ax.imshow(1 - cor_borders, cmap="gray", vmin=0, vmax=1)
    for acr in area_colors:
        plot_borders_and_areas(
            ax,
            label_img=cor_atlas,
            areas_to_plot=[acr],
            color_kwargs=dict(colors=area_colors[acr], alpha=alpha),
            label_atlas=label_atlas,
            get_descendants=True,
            plot_borders=False,
            label_filled_areas=False,
        )
    return cor_atlas


def plot_flatmap(
    ax,
    ara_projection="flatmap_dorsal",
    hemisphere="both",
    area_colors={},
    alpha=0.2,
    ccf_streamlines_folder=None,
    label_areas=True,
):
    """Plot a flatmap of the atlas

    Args:

        ax: Axes on which to plot
        ara_projection: Projection of the atlas (flatmap_dorsal, flatmap_butterfly)
        hemisphere: Hemisphere to plot (left, right, both)
        area_colors: Dictionary of area acronyms to their colors
        alpha: Opacity of the plotted areas
        ccf_streamlines_folder: Folder where the CCF streamlines are stored.
        label_areas: Whether to label the areas. Default is True.


    Returns:
        flat_atlas: The flatmap of the atlas
    """
    if ccf_streamlines_folder is None:
        import flexiznam as flz

        project_folder = Path(flz.PARAMETERS["data_root"]["processed"]).parent
        ccf_streamlines_folder = project_folder / "resources" / "ccf_streamlines"

    bf_boundary_finder = ccfproj.BoundaryFinder(
        projected_atlas_file=ccf_streamlines_folder / f"{ara_projection}.nrrd",
        labels_file=ccf_streamlines_folder / "labelDescription_ITKSNAPColor.txt",
    )

    to_plot = []
    if hemisphere in ["left", "both"]:
        # We get the left hemisphere region boundaries with the default arguments
        bf_left_boundaries = bf_boundary_finder.region_boundaries()
        to_plot.append(bf_left_boundaries)

    if hemisphere in ["right", "both"]:
        # And we can get the right hemisphere boundaries that match up with
        # our projection if we specify the same configuration
        if ara_projection in ["flatmap_dorsal", "flatmap_butterfly"]:
            view_space_for_other_hemisphere = ara_projection
        else:
            view_space_for_other_hemisphere = True

        bf_right_boundaries = bf_boundary_finder.region_boundaries(
            # we want the right hemisphere boundaries, but located in the right place
            # to plot both hemispheres at the same time
            hemisphere="right_for_both" if hemisphere == "both" else "right",
            # we also want the hemispheres to be adjacent
            view_space_for_other_hemisphere=view_space_for_other_hemisphere,
        )
        to_plot.append(bf_right_boundaries)
    # Flatmap
    for boundaries in to_plot:
        for acronym, boundary_coords in boundaries.items():
            ax.plot(*boundary_coords.T, c="k", lw=0.5)
            if acronym in area_colors:
                ax.fill(*boundary_coords.T, c=area_colors[acronym], alpha=alpha, lw=0)
                if label_areas:
                    ax.text(
                        np.mean(boundary_coords[:, 0]),
                        np.mean(boundary_coords[:, 1]),
                        acronym,
                        fontsize=8,
                        color="k",
                        ha="center",
                        va="center",
                    )
    ax.set_aspect("equal")


def get_ara_retinotopic_map(keep_cropped_data=False):
    """Get the retinotopic map from the Allen Brain Atlas

    From: Biological variation in the sizes, shapes and locations of visual cortical
    areas in the mouse Waters J, Lee E, Gaudreault N, Griffin F, Lecoq J, et al. (2019)
    PLOS ONE 14(5): e0213924. https://doi.org/10.1371/journal.pone.0213924

    Args:
        keep_cropped_data: if True, will keep the cropped data, as in the original
            article. If False, pad the output to the full atlas size. Default is False

    Returns:
    mean_elevation_map, mean_azimuth_map: np.ndarray
        The elevation and azimuth maps on top projection of the Allen Mouse Brain Atlas
    """
    WKF_URL = "http://api.brain-map.org/api/v2/well_known_file_download/{}"

    def numpy_load_wkf(wkf_id, url=WKF_URL):
        url = url.format(wkf_id)
        r = requests.get(url)
        if r.status_code != 200:
            print("Error retrieving file from {}".format(url))
        else:
            return np.load(BytesIO(r.content))

    ISI1_azimuth_map_stack = numpy_load_wkf(745545159)
    ISI1_altitude_map_stack = numpy_load_wkf(745544088)
    mean_altitude_map = np.copy(ISI1_altitude_map_stack)
    mean_azimuth_map = np.copy(ISI1_azimuth_map_stack)

    for ii in range(mean_altitude_map.shape[0]):
        altitude_map = mean_altitude_map[ii, :, :]
        azimuth_map = mean_azimuth_map[ii, :, :]

        # convert all values with no data to NaNs
        altitude_map[altitude_map == altitude_map[0, 0]] = np.nan
        azimuth_map[azimuth_map == azimuth_map[0, 0]] = np.nan

        mean_altitude_map[ii, :, :] = altitude_map
        mean_azimuth_map[ii, :, :] = azimuth_map

    warnings.filterwarnings("ignore")
    mean_altitude_map = np.nanmean(mean_altitude_map, axis=0)
    mean_azimuth_map = np.nanmean(mean_azimuth_map, axis=0)
    warnings.filterwarnings("default")

    if keep_cropped_data:
        return mean_altitude_map, mean_azimuth_map

    atlas = bga.bg_atlas.BrainGlobeAtlas("allen_mouse_10um")
    ara_azimuth = np.zeros((atlas.shape[0], atlas.shape[2])) + np.nan
    ara_elevation = np.zeros((atlas.shape[0], atlas.shape[2])) + np.nan
    s = mean_azimuth_map.shape
    shift = [500, 0]
    ara_azimuth[shift[0] : s[0] + shift[0], shift[1] : s[1] + shift[1]] = (
        mean_azimuth_map
    )
    ara_elevation[shift[0] : s[0] + shift[0], shift[1] : s[1] + shift[1]] = (
        mean_altitude_map
    )

    # Make a symmetric map for the other hemisphere
    midline = int(atlas.shape[2] / 2)
    ara_azimuth[:, midline:] = np.flip(ara_azimuth[:, :midline], axis=1)
    ara_elevation[:, midline:] = np.flip(ara_elevation[:, :midline], axis=1)
    return ara_elevation, ara_azimuth


def move_out_of_area(
    pts: np.array,
    atlas,
    areas_to_empty="fiber tracts",
    valid_areas="grey",
    distance_threshold: float = 200,
    verbose: bool = True,
):
    """Move pts that are in `areas_to_empty` into the closest area

    Args:
        pts (np.array): array of points to move, in micron
        atlas (BrainGlobeAtlas): atlas to use
        areas_to_empty (str | list): area(s) to empty, must be valid `atlas.structure`
            name (acronym)
        valid_areas (str | list): area(s) to move the points to,  area(s) to empty, must
            be valid `atlas.structure` name (acronym)
        distance_threshold (float): maximum distance to move the points in microns
        verbose (bool): if True, will print some information

    Returns:
        pd.Dataframe: dataframe with `initial_coords`, `new_coords`, `distance_moved`,
            `new_area_id`, `new_area_acronym`, `moved`, `pts_index` columns
    """
    area_ids = np.zeros_like(pts[:, 0]) + np.nan

    # Get all the descendants of the area to empty
    id2empty_set: set[int] = set()
    if isinstance(areas_to_empty, str):
        areas_to_empty = [areas_to_empty]
    for area_to_empty in areas_to_empty:
        if area_to_empty == "outside":
            id2empty_set.add(0)
        else:
            invalid = atlas.structures.tree.leaves(
                atlas.structures[area_to_empty]["id"]
            )
            id2empty_set.update(leave.identifier for leave in invalid)
    id2empty = list(id2empty_set)

    # Find all points that are in the area to empty
    voxel_coords = np.round(pts / np.array(atlas.resolution)).astype(int)
    area_ids = atlas.annotation[
        voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]
    ]
    to_move = np.isin(area_ids, id2empty)
    if verbose:
        print(f"Found {to_move.sum()} points in {areas_to_empty}")
    if not to_move.sum():
        return pd.DataFrame(
            columns=[
                "initial_coords",
                "new_coords",
                "distance_moved",
                "new_area_id",
                "new_area_acronym",
                "moved",
                "pts_index",
            ]
        )

    # Find which areas are valid destinations
    idvalid_set: set[int] = set()
    if isinstance(valid_areas, str):
        valid_areas = [valid_areas]
    for valid_area in valid_areas:
        valid = atlas.structures.tree.leaves(atlas.structures[valid_area]["id"])
        idvalid_set.update(leave.identifier for leave in valid)
    idvalid = list(idvalid_set)
    if verbose:
        print(f"Found {len(idvalid)} acceptable target areas")

    window = np.round(distance_threshold / np.array(atlas.resolution)).astype(int)
    # Find the closest point in the valid area
    moved = []
    for pts_index in tqdm.tqdm(np.where(to_move)[0], disable=not verbose):
        pts_info = dict()
        point = pts[pts_index]
        pts_info["initial_coords"] = point
        pts_info["pts_index"] = pts_index
        if np.isnan(point).any():
            continue
        center_voxel = voxel_coords[pts_index]
        lims = np.vstack([center_voxel - window, center_voxel + window])
        lims = np.clip(lims, 0, np.array(atlas.annotation.shape) - 1)
        local_atlas = atlas.annotation[
            lims[0, 0] : lims[1, 0], lims[0, 1] : lims[1, 1], lims[0, 2] : lims[1, 2]
        ]
        pts_in_local_atlas = point / np.array(atlas.resolution)
        pts_in_local_atlas = pts_in_local_atlas - lims[0]
        valid_coords = np.array(np.where(np.isin(local_atlas, idvalid)))
        if not valid_coords.shape[1]:
            pts_info["new_coords"] = pts_info["initial_coords"]
            pts_info["distance_moved"] = 0
            pts_info["new_area_id"] = np.nan
            pts_info["new_area_acronym"] = "None"
            pts_info["moved"] = False
            moved.append(pts_info)
            continue
        rel_coord_voxel = valid_coords - pts_in_local_atlas[:, None]
        # make it into um
        rel_coord_um = rel_coord_voxel * np.array(atlas.resolution)[:, None]
        distance2pts = np.linalg.norm(rel_coord_um, axis=0)
        cl_id = np.argmin(distance2pts)
        closest_distance = distance2pts[cl_id]
        if closest_distance > distance_threshold:
            pts_info["new_coords"] = pts_info["initial_coords"]
            pts_info["distance_moved"] = 0
            pts_info["new_area_id"] = np.nan
            pts_info["new_area_acronym"] = "None"
            pts_info["moved"] = False
            moved.append(pts_info)
            continue
        pts_info["distance_moved"] = closest_distance
        closest_coord = valid_coords[:, cl_id] + lims[0]
        pts_info["new_coords"] = closest_coord * np.array(atlas.resolution)
        pts_info["new_area_id"] = atlas.annotation[
            closest_coord[0], closest_coord[1], closest_coord[2]
        ]
        pts_info["new_area_acronym"] = atlas.structures[pts_info["new_area_id"]][
            "acronym"
        ]
        pts_info["moved"] = True
        moved.append(pts_info)
    moved_df = pd.DataFrame(moved)
    if verbose:
        print(f"Moved {moved_df.moved.sum()}/{to_move.sum()} points")
    return moved_df
