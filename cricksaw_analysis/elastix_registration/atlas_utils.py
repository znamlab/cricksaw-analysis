"""Functions used to modify the atlas"""
from pathlib import Path

import SimpleITK as sitk
from sklearn.neighbors import KDTree
import numpy as np
import os
import pyqtgraph as pg  # needed for correct lasagna import.
import pandas as pd
from lasagna.io_libs import ara_json
from lasagna.tree import tree_parser

# from OpenEphys import utils
from skimage import segmentation, morphology
from scipy.ndimage.morphology import binary_dilation
from functools import partial
from scipy.optimize import least_squares


def get_cortex_borders(path_to_atlas, path_to_json):
    """Return the pia and the white matter around the cortex for one hemisphere (the left)"""

    ara_df, ara_tree = load_labels(path_to_json, return_df=True, return_tree=True)

    isocortex_id = ara_df[ara_df.name == "Isocortex"].id.iloc[0]
    ctx_ids = ara_tree.find_leaves(int(isocortex_id))
    # the claustrum is annoying. It make a tiny hole in the surface, add it as if it was cortex
    claustrum_id = ara_df[ara_df.name == "Claustrum"].id.iloc[0]
    ctx_ids.append(int(claustrum_id))
    atlas = sitk.ReadImage(path_to_atlas)
    atlas = sitk.GetArrayFromImage(atlas)

    # Midline is annoying. Cut the atlas in two and add a line of outside the brain on the side
    semi_atlas = np.array(atlas)
    semi_atlas[:, :, -int(np.floor(atlas.shape[2] / 2)) :] = 0
    semi_wm = sitk.GetArrayFromImage(
        binary_image(
            path_to_atlas, path_to_json, "fiber tracts", name_type="acronym", value=255
        )
    )
    semi_wm[:, :, -int(np.floor(atlas.shape[2] / 2)) :] = 0
    wm_outer_border = segmentation.find_boundaries(
        semi_wm, background=0, mode="outer", connectivity=2
    )

    # find the last pixel in the brain
    brain = semi_atlas != 0

    print("Find brain surface")
    brain_border = segmentation.find_boundaries(
        brain, background=0, mode="inner", connectivity=2
    )
    is_cortex = np.zeros(semi_atlas.shape, dtype=bool)
    for a in ctx_ids:
        is_cortex[semi_atlas == a] = True
    ctx_border = segmentation.find_boundaries(
        is_cortex, background=0, mode="inner", connectivity=2
    )

    pia = ctx_border & brain_border
    pia_clean = morphology.remove_small_objects(pia, min_size=10000, connectivity=2)
    bottom_ctx = ctx_border & wm_outer_border
    bottom_clean = morphology.remove_small_objects(
        bottom_ctx, min_size=10000, connectivity=2
    )
    return pia_clean, bottom_clean


def create_float_atlas(path_to_atlas, path_to_save=None, reload=True):
    """Create a version of the atlas that contains only float for elastix"""

    if path_to_save is not None:
        path_to_save = Path(path_to_save)
        path_to_translate = path_to_save.with_name(
            path_to_save.stem + "_translator.npy"
        )
    if (
        (path_to_save is not None)
        and reload
        and path_to_save.is_file()
        and path_to_translate.is_file()
    ):
        continuous_atlas = sitk.ReadImage(str(path_to_save))
        data = np.load(str(path_to_translate))
        translator = dict([(i, j) for i, j in zip(data[0], data[1])])
        return continuous_atlas, translator
    atlas = sitk.ReadImage(str(path_to_atlas))

    # Create a continuous atlas than can be cast to float
    atlas_array = np.asarray(sitk.GetArrayFromImage(atlas))
    continuous_atlas = np.zeros(atlas_array.shape, dtype=float)
    atlas_values = np.unique(atlas_array)
    translator = dict()
    for i, v in enumerate(atlas_values):
        continuous_atlas[atlas_array == v] = float(i)
        translator[float(i)] = v

    if path_to_save is not None:
        sitk.WriteImage(sitk.GetImageFromArray(continuous_atlas), str(path_to_save))
        array_version = np.array(list(translator.keys()))
        array_version = np.vstack(
            [array_version, np.array([translator[i] for i in array_version])]
        )
        np.save(str(path_to_translate), array_version)

    return continuous_atlas, translator


def translate_atlas(in_image, translator, path_to_save=None, out_dtype="int32"):
    """Translate an image following a directory"""
    data = sitk.GetArrayFromImage(in_image)
    output = np.zeros(data.shape, dtype=out_dtype)
    for k, v in translator.items():
        output[data == k] = v
    output = sitk.GetImageFromArray(output)
    if path_to_save is not None:
        sitk.WriteImage(output, path_to_save)
    return output


def create_borders_atlas(path_to_atlas):
    """Create a binary image with the border of all areas set to 1"""
    OUT_OFF_BRAIN_VALUE = 0  # value of a pixel that is outside of the brain
    atlas = sitk.ReadImage(path_to_atlas)
    array = sitk.GetArrayFromImage(atlas)
    borders = np.zeros(array.shape, dtype="uint8")

    # using skimage would label both side of the border. Use diff instead. The border is in one of the two area.
    # There is no good reason to select one or the other.
    for ip, plane in enumerate(array):
        # first look from left to right
        for il, line in enumerate(plane):
            bord = np.diff(np.hstack([0, line])) != 0
            borders[ip, il, :] = bord
        # then from bottom to top
        for ic, col in enumerate(plane.T):
            bord = np.diff(np.hstack([0, col])) != 0
            borders[ip, :, ic] += bord
    borders[borders != 0] = 255
    bord_img = sitk.GetImageFromArray(borders)

    return bord_img


def load_labels(path_to_json, return_df=False, return_tree=True):
    """
    Load the labels file, which must be in JSON


    The JSON should be the raw JSON from the ARA website

    Returns the labels as a tree structure that can be indexed by ID or/and a pandas dataframe
    """
    flattened_ara_tree, header = ara_json.import_data(path_to_json)
    table = flattened_ara_tree.strip().split("\n")
    out = []
    if return_df:
        ara_df = pd.DataFrame(
            data=[l.split("|") for l in table], columns=header.split("|")
        )
        ara_df.set_index("id", drop=False, inplace=True)
        out.append(ara_df)
    if return_tree:
        ara_tree = tree_parser.parse_file(table, col_sep="|", header_line=header)
        out.append(ara_tree)
    return tuple(out)


def binary_image(
    path_to_atlas, path_to_json, area_names, name_type="acronym", value=255
):
    """Create an image with area_names to `value` and the rest to 0"""
    # Make sure it is iterable
    # area_names = utils.make_list(area_names)
    ara_df, ara_tree = load_labels(path_to_json, return_df=True, return_tree=True)
    atlas = sitk.ReadImage(path_to_atlas)
    atlas = sitk.GetArrayFromImage(atlas)

    binary = np.zeros(atlas.shape, dtype="uint8")
    for area in area_names:
        good_id = ara_df.loc[:, name_type] == area
        if np.sum(good_id) == 0:
            print("Area %s not found" % area)
            continue
        elif np.sum(good_id) > 1:
            raise ValueError("Multiple match")
        area_data = ara_df[good_id].iloc[0]
        # use the tree to find all the leaves from that node
        node_ids = ara_tree.find_leaves(int(area_data.id))
        for n in node_ids:
            # atlas_id = ara_tree[n].data['atlas_id']
            binary[atlas == n] = value
    binary = sitk.GetImageFromArray(binary)
    return binary


def get_sub_area_id(area, area_column, atlas_labels, atlas_tree):
    if isinstance(area, str):
        area = [area]
    valid_values = []
    for a in area:
        val = atlas_labels[atlas_labels[area_column] == a].id
        assert val.shape[0] == 1
        val = int(val.iloc[0])
        valid_values.extend(get_descendent(atlas_tree, val, [val]))
    return list(np.unique(valid_values))


def plot_borders_and_areas(
    ax,
    label_img,
    areas_to_plot,
    color_kwargs=dict(),
    border_dilatation=2,
    area_dilatation=0,
    contour_version=False,
    cont_kwargs=dict(),
):
    """Plot the atlas borders and highlight areas

    contour_version is a lot slower but tried to make something vectorial

    :param label_img: a 2d image of labels
    :param areas_to_plot: a list (of list) of area ids to label.
    :param color_kwargs: kwargs for imshow of the borders (useful for color specification)
    :param border_dilatation: number of iteration of dilatation of borders
    :param area_dilatation:  number of iteration of dilatation of areas. Useful to render very thin structures
    :return: img, border the image label and the image of borders
    """
    kwargs = dict(vmin=0, vmax=len(areas_to_plot) + 1)
    kwargs.update(color_kwargs)
    contours = []
    if not contour_version:
        img = np.zeros(label_img.shape, dtype=float)
        for i_ar, ar_list in enumerate(areas_to_plot):
            good = np.isin(label_img, ar_list)
            if area_dilatation > 0:
                good = binary_dilation(good, iterations=area_dilatation)
            img[good] += i_ar + 1

        border = get_border(label_img)
        if border_dilatation > 0:
            border = binary_dilation(border, iterations=border_dilatation)
        border = np.asarray(border, dtype=float)
        border[border == 0] = np.nan
        img[img == 0] = np.nan
        ax.imshow(img, **kwargs)
        ax.imshow(border, vmin=-1, vmax=2, cmap="Greys")
        ax.set_axis_off()
        return img, border

    # instead of plotting the border, make a contour for each area
    # frist group subgroups of areas
    new_label = np.array(label_img)
    filled_areas = []
    for area_subgroup in areas_to_plot:
        filled_areas.append(area_subgroup[0])
        for area in area_subgroup:
            new_label[new_label == area] = area_subgroup[0]

    bin_image = np.zeros(new_label.shape, dtype="int8")
    for area in np.unique(new_label):
        # create an image with 1 in the area and 0 everywhere else
        bin_image *= 0
        bin_image[new_label == area] = 1
        ax.contour(
            bin_image, levels=[0.5], cmap="Greys", vmin=-1, vmax=2, **cont_kwargs
        )

        if area in filled_areas:
            i_area = filled_areas.index(area)
            bin_image[new_label == area] += i_area
            contours.append(
                ax.contourf(bin_image, levels=np.array([0.5, 1.5]) + i_area, **kwargs)
            )
    ax.set_aspect("equal")
    ax.set_axis_off()
    return contours


def move_out_of_wm(
    pts,
    dist_threshold,
    atlas_folder,
    path_to_json,
    atlas=None,
    atlas_labels=None,
    atlas_tree=None,
    tree_dict=None,
    ctx_df=None,
):
    """
    Move the points to the closest area out of the white matter

    `dist_threshold` must be in `atlas` unit (so in voxels)
    """
    if atlas_tree is None or atlas_labels is None:
        atlas_labels, atlas_tree = load_labels(
            path_to_json, return_df=True, return_tree=True
        )
    if atlas is None:
        atlas = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(atlas_folder, "atlas.mhd"))
        )
    if ctx_df is None:
        ctx_df = create_ctx_table(path_to_json=path_to_json)
    tree_dict = get_trees_to_border(atlas_folder, tree_dict)

    # find atlas values
    atlas_id = atlas[pts[:, 0], pts[:, 1], pts[:, 2]]
    # find which areas are in white matters
    wm_ind = get_descendent(atlas_tree, 1009)
    white_cells = np.isin(atlas_id, wm_ind)
    white_coords = pts[white_cells, :]
    if not len(white_coords):
        return pts
    dist, index = tree_dict["l6wm_border"].query(white_coords)
    dist = np.squeeze(dist)
    to_move = np.zeros_like(white_cells)
    to_move[np.where(white_cells)[0][dist < dist_threshold]] = True
    new_coord = pts[:]
    border_pts = np.asarray(tree_dict["l6wm_border"].data)
    closest_pts = np.asarray(border_pts[np.squeeze(index), :], dtype=pts.dtype)
    new_coord[to_move] = closest_pts[dist < dist_threshold]
    return new_coord


def create_ctx_table(path_to_json):
    """Create a dataframe with cortical area info"""

    ara_df, ara_tree = load_labels(path_to_json, return_df=True, return_tree=True)
    isocortex_id = ara_df[ara_df.name == "Isocortex"].id.iloc[0]
    ctx_ids = ara_tree.find_leaves(int(isocortex_id))

    ctx_df = []
    for id in ctx_ids:
        node = ara_tree[id]
        full_name = node.data["name"]
        if "layer" in full_name.lower():
            layer = full_name[full_name.lower().find("layer") + 5 :].strip()
            parent = ara_tree[node.parent]
            data = pd.Series(
                dict(
                    cortical_area=parent.data["name"],
                    cortical_area_id=node.parent,
                    layer=layer,
                    sub_area=full_name,
                    sub_area_id=id,
                    acronym=parent.data["acronym"],
                ),
                name=id,
            )
        elif id in [810, 819]:
            # two annoying area have layer specification but not layer written in their name
            layer = full_name.lower().split(" ")[-1].strip()
            parent = ara_tree[node.parent]
            data = pd.Series(
                dict(
                    cortical_area=parent.data["name"],
                    cortical_area_id=node.parent,
                    layer=layer,
                    sub_area=full_name,
                    sub_area_id=id,
                ),
                name=id,
            )
        else:
            data = pd.Series(
                dict(
                    cortical_area=full_name,
                    cortical_area_id=id,
                    layer="nd",
                    sub_area="none",
                    sub_area_id=np.nan,
                ),
                name=id,
            )
        ctx_df.append(data)
    return pd.DataFrame(ctx_df)


def get_closest_with_interpolation(kdtree, pts_to_move, kd_data=None, k=3):
    """Find the k nearest neighbour and return the average weighted by distance"""
    dist, indices = kdtree.query(pts_to_move, return_distance=True, k=k)
    if kd_data is None:
        kd_data = np.asarray(kdtree.data)
    pts_3d = np.zeros([k, pts_to_move.shape[0], pts_to_move.shape[1]])
    for i in range(k):
        pts_3d[i] = kd_data[indices[:, i]]
    # pts_3d = np.dstack([kd_data[i, :] for i in indices])

    # make a weighted average on the triangle to find the real point
    # reshape in coord, num pts, num neighbours to weight
    pts_3d = pts_3d.transpose((2, 1, 0))
    # for exact matches, keep them
    exact = dist[:, 0] == 0
    weighted_av = np.zeros_like(pts_to_move)
    weighted_av[exact] = pts_to_move[exact]
    # for the rest do a wieghted average
    to_av = np.logical_not(exact)
    weight = 1 / dist[to_av]
    weighted_av[to_av] = (np.sum(pts_3d[:, to_av] * weight, 2) / np.sum(weight, 1)).T
    return weighted_av


def get3d_position(
    coords2d,
    atlas_folder,
    atlas=None,
    hemisphere="right",
    vertices=None,
    vt=None,
    index2dto3d=None,
):
    """Get the 3D coordinates corresponding to a UV position"""

    if vertices is None or vt is None or index2dto3d is None:
        bff_file = os.path.join(atlas_folder, "bff_output.obj")
        vertices, vt, index3dto2d = load_flat_trans(bff_file)
        index2dto3d = dict([(v, k) for k, v in index3dto2d.items()])

    if hemisphere.lower() == "right":
        # the vertices where measured on an isolated left hemisphere.
        # Put them back half a brain to the right in mirror
        if atlas is None:
            atlas = sitk.GetArrayFromImage(
                sitk.ReadImage(os.path.join(atlas_folder, "atlas.mhd"))
            )
        vertices[:, 2] = -vertices[:, 2] + int(atlas.shape[2])

    vt_tree = KDTree(vt)
    # find the 3 closest 2d support point
    dist, indices = vt_tree.query(coords2d, return_distance=True, k=3)
    index_3d = np.asarray([[index2dto3d[i] for i in l] for l in indices])
    pts_3d = np.dstack([vertices[i, :] for i in index_3d])

    # make a weighted average on the triangle to find the real point
    # reshape in coord, num pts, num neighbours to weight
    pts_3d = pts_3d.transpose((1, 2, 0))
    weighted_av = np.sum(pts_3d * dist, 2) / np.sum(dist, 1)
    return weighted_av.T


def get_2d_position(coords3d, atlas_folder, atlas=None, hemisphere="right"):
    """Get the UV position from a 3d coordinates

    3D coordinates must be such that coords3d[:,2], coords3d[:,1] is a coronal plane"""
    bff_file = os.path.join(atlas_folder, "bff_output.obj")
    vertices, vt, index3dto2d = load_flat_trans(bff_file)

    if hemisphere.lower() == "right":
        # the vertices where measured on an isolated left hemisphere.
        # Put them back half a brain to the right in mirror
        if atlas is None:
            atlas = sitk.GetArrayFromImage(
                sitk.ReadImage(os.path.join(atlas_folder, "atlas.mhd"))
            )
        vertices[:, 2] = -vertices[:, 2] + int(atlas.shape[2])
    elif hemisphere.lower() != "left":
        raise IOError("Hemisphere should be right or left")

    vertice_tree = KDTree(vertices)
    # find the closest support point
    indices = vertice_tree.query(coords3d, return_distance=False)
    index_2d = np.asarray([index3dto2d[i] for i in indices[:, 0]])
    pts_2d = vt[index_2d, :]
    return pts_2d


def write_obj(target, verts, normals, faces, zero_based=True):
    """write an obj file for vertices, normales and faces"""
    if not zero_based:
        faces = faces + 1
    thefile = open(target, "w")
    for item in verts:
        thefile.write("v {0} {1} {2}\n".format(item[0], item[1], item[2]))
    for item in normals:
        thefile.write("vn {0} {1} {2}\n".format(item[0], item[1], item[2]))
    for item in faces:
        thefile.write(
            "f {0}//{0} {1}//{1} {2}//{2}\n".format(item[0], item[1], item[2])
        )
    thefile.close()


def load_flat_trans(path2flat_def, return_normale=False):
    """Load the 3D to 2D transform from obj file"""

    with open(path2flat_def, "r") as objdef:
        contents = objdef.read().split("\n")

    # use regexp instead
    import re

    v_pattern = re.compile("v (\d+.?\d*) (\d+.?\d*) (\d+.?\d*)")
    vt_pattern = re.compile("vt (\d+.?\d*) (\d+.?\d*)")
    f_pattern = re.compile("f (\d+)/(\d+) (\d+)/(\d+) (\d+)/(\d+)\n")
    with open(path2flat_def, "r") as objdef:
        contents = objdef.read()
    vertices = np.array(v_pattern.findall(contents), dtype=float)
    vt = np.array(vt_pattern.findall(contents), dtype=float)
    faces = np.array(f_pattern.findall(contents), dtype=int)
    f2d = faces[:, 1::2].reshape(-1) - 1  # remove one to have 0 based
    f3d = faces[:, :-1:2] - 1  # remove one to have 0 based
    flatf3d = f3d.reshape(-1)
    index3dto2d = dict([(three, two) for two, three in zip(f2d, flatf3d)])

    if not return_normale:
        return vertices, vt, index3dto2d

    # also get normales
    # from https://sites.google.com/site/dlampetest/python/calculating-normals-of-a-triangle-mesh-using-numpy
    # and https://stackoverflow.com/questions/6656358/calculating-normals-in-a-triangle-mesh/6661242#6661242
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[f3d]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # Now i need to find the faces of each vertex and sum their normales
    norm[f3d[:, 0]] += n
    norm[f3d[:, 1]] += n
    norm[f3d[:, 2]] += n

    # normalise
    lens = np.sqrt(norm[:, 0] ** 2 + norm[:, 1] ** 2 + norm[:, 2] ** 2)
    norm[:, 0] /= lens
    norm[:, 1] /= lens
    norm[:, 2] /= lens

    return vertices, vt, index3dto2d, norm


def get_cortical_borders(
    path2layeratlas,
    path2wmatlas,
    l1=100,
    l2=122,
    l3=144,
    l4=166,
    l5=188,
    l6a=210,
    l6b=232,
):
    """Find the border between layers

    The border is taken inside the upper layer so that the l6 to wm border is in l6"""

    borders = dict()
    layer = sitk.GetArrayFromImage(sitk.ReadImage(path2layeratlas))
    is_l6 = np.isin(layer, [l6a, l6b])
    is_l5 = layer == l5
    is_l234 = np.isin(layer, [l4, l3, l2])
    is_l1 = layer == l1
    is_wm = sitk.GetArrayFromImage(sitk.ReadImage(path2wmatlas)) == 255

    ordered_layers = [is_l1, is_l234, is_l5, is_l6, is_wm]
    dilated_layers = [[] for i in ordered_layers]
    layer_names = ["l1", "l234", "l5", "l6", "wm"]
    for i, layer_img in enumerate(ordered_layers):
        print("Dilate %s" % layer_names[i])
        dilated_layers[i] = binary_dilation(layer_img)

    # now find borders
    for i in range(len(ordered_layers) - 1):
        name = layer_names[i] + layer_names[i + 1] + "_border"
        border = np.logical_and(dilated_layers[i + 1], ordered_layers[i])
        borders[name] = np.vstack(np.where(border)).T

    return borders


def get_trees_to_border(atlas_folder, tree_dict=None):
    """Get the KDTree for all borders between layers"""
    border_names = [
        "pial_coords",
        "l1l234_border",
        "l234l5_border",
        "l5l6_border",
        "l6wm_border",
    ]
    if tree_dict is None:
        tree_dict = dict()
    for b in border_names:
        if b in tree_dict:
            continue
        bord_pts = np.load(os.path.join(atlas_folder, "%s.npy" % b))
        if b == "pial_coords":  # I saved it transposed ....
            bord_pts = bord_pts.T
        tree_dict[b] = KDTree(bord_pts)
    return tree_dict


def project_pts_to_wm(pts, atlas_folder, tree_dict=None):
    """Project pts to the closest border between layer and then iteratively towards the wm"""

    if not len(pts):
        return pts
    tree_dict = get_trees_to_border(atlas_folder, tree_dict)
    border_names = [
        "pial_coords",
        "l1l234_border",
        "l234l5_border",
        "l5l6_border",
        "l6wm_border",
    ]
    border_names = border_names[::-1]

    print("Find closest border")
    distance = np.zeros([pts.shape[0], len(tree_dict)])
    for i, b in enumerate(border_names):
        d, _ = tree_dict[b].query(pts, return_distance=True)
        distance[:, i] = d[:, 0]

    # find the smallest
    closest_border = np.argmin(distance, axis=1)
    wm_pts = np.array(pts)

    #  process iteratively until all the pts are on the l1 border
    level = np.max(closest_border)
    while level >= 0:
        print("Move to %s" % border_names[level])
        deep_pts = closest_border == level
        # move these pts up
        # find the tree that gives the upper level
        tree = tree_dict[border_names[level]]
        wm_pts[deep_pts] = get_closest_with_interpolation(tree, wm_pts[deep_pts])
        closest_border[deep_pts] -= 1
        level = np.max(closest_border)

    return wm_pts


def _distance_to_pia_normal(
    uv,
    single_pts,
    vt,
    atlas_folder,
    hemisphere,
    atlas,
    vertices,
    index2dto3d,
    pia_tree,
    path_to_wm,
):
    """Calculate the distance from a normale to the pia at coordinates u,v to a single 3d point"""

    # first from u,v find the normal vector
    # get the closest 3d pts
    coords = get3d_position(
        coords2d=np.atleast_2d([uv[0], uv[1]]),
        atlas=atlas,
        hemisphere=hemisphere,
        atlas_folder=atlas_folder,
        vertices=vertices,
        vt=vt,
        index2dto3d=index2dto3d,
    )
    # find closest pia_pts
    ind = pia_tree.query(coords, return_distance=False)
    # get the normal
    normal = path_to_wm[ind[0, 0]]
    # get the distance
    dst = lineseg_dist(single_pts, normal[:, 0], normal[:, 1])
    return dst


def _find_shortest_dist_to_normal(
    single_pts,
    atlas_folder,
    atlas,
    vt,
    vertices,
    path_to_wm,
    pia_tree,
    index2dto3d,
    pia_pts,
):
    midline = atlas.shape[2] / 2
    if single_pts[2] > midline:
        hemisphere = "right"
    else:
        hemisphere = "left"
    kwargs = dict(
        vt=vt,
        atlas_folder=atlas_folder,
        hemisphere=hemisphere,
        atlas=atlas,
        vertices=vertices,
        index2dto3d=index2dto3d,
        pia_tree=pia_tree,
        path_to_wm=path_to_wm,
        single_pts=single_pts,
    )
    func = partial(_distance_to_pia_normal, **kwargs)
    closest = pia_tree.query(np.atleast_2d(single_pts), return_distance=False)[0, 0]
    x0 = get_2d_position(np.atleast_2d(pia_pts[closest]), atlas_folder, atlas=atlas)[0]
    out = least_squares(func, x0)

    coord = get3d_position(
        np.atleast_2d(out.x),
        atlas_folder=atlas_folder,
        vt=vt,
        vertices=vertices,
        index2dto3d=index2dto3d,
        hemisphere=hemisphere,
        atlas=atlas,
    )
    closest = pia_tree.query(coord, return_distance=False)[0, 0]
    return pia_pts[closest]


def oignon_plot(
    layer, atlas, ctx_table, orientation="dorsal", border_only=False, atlas_df=None
):
    """Remove all layers above `layer` and find the atlas seen from the top

    Set layer to 'wm' to remove the whole cortex"""
    peeled_atlas = peel_atlas(layer, atlas, ctx_table, atlas_df=atlas_df)
    proj_atlas = external_view(peeled_atlas, axis=orientation, border_only=border_only)
    return proj_atlas


def oignon_index(layer, atlas, ctx_table, orientation="dorsal", atlas_df=None):
    """Return the index of the atlas that would form the projection one one axis once peeled"""
    peeled_atlas = peel_atlas(layer, atlas, ctx_table, atlas_df=atlas_df)
    ind = external_view(peeled_atlas, axis=orientation, get_index=True)
    return ind


def external_view(
    atlas, axis="dorsal", border_only=False, get_index=False, which="first"
):
    """Get a view from atlas from one side

    axis can be dorsal, ventral, left, right, front, back

    if get_index, doesn't return the view but the index of the pixels generating the view

    if which is first return the surface. If which is last, return the first index after the surface that is out of the
     brain. It is not equivalent to changing the axis if you have multiple pieces of brain overlapping"""

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
    # make the atlas int8 to gain space and still have 0, 1 and -1
    bin_atlas = np.array(proj_atlas != 0, dtype="int8")

    # now find the first value not outside brain (i.e. not 0)
    pad = np.zeros([1] + list(bin_atlas.shape[1:]), dtype="int8")
    is_in_brain = np.diff(np.vstack([pad, bin_atlas]), axis=0)
    if which == "first":
        # look for the first 1
        first_in_brain = np.argmax(
            is_in_brain, axis=0
        )  # argmax return the first occurence of max (which is 1 here)
    elif which == "last":
        # look for the first -1
        first_in_brain = np.argmin(
            is_in_brain, axis=0
        )  # argmin return the first occurence of min (which is -1 here)
    else:
        raise IOError("Unknown type. `which` should be first or last")
    if get_index:
        return first_in_brain
    # get the first value in brain for each
    x, y = np.meshgrid(*[np.arange(s) for s in first_in_brain.shape])
    proj_atlas = proj_atlas[first_in_brain, x.T, y.T]

    if border_only:
        proj_atlas = get_border(proj_atlas)
    return proj_atlas


def get_descendent(atlas_tree, area_id, descendent=None):
    """
    Return the id of all areas below area_id

    area_id itself is not included in the output
    descendent is the list in which the ids will be added. Use
    get_descendent(atlas_tree, area_id, [area_id]) if you want to include
    the parent id in the output
    """
    root = atlas_tree[area_id]
    if descendent is None:
        descendent = []
    # add this layer
    descendent.extend(root.children)
    # follow down on the next layer
    for child in root.children:
        descendent = get_descendent(atlas_tree, child, descendent)
    return descendent


def peel_atlas(layer, atlas, ctx_table, get_peel=False, atlas_df=None):
    """Remove all layers above `layer` and find the atlas seen from the top

    Set layer to 'wm' to remove the whole cortex"""
    ordered_list = ["1", "2", "2/3", "4", "5", "6a", "6b", "wm"]
    sclist = ["SCzo", "SCsg", "SCop", "SCig", "SCiw", "SCdg", "SCdw"]
    if layer in ordered_list:
        to_remove = ordered_list[: ordered_list.index(layer)]
        to_remove = np.asarray(
            ctx_table[ctx_table.layer.isin(to_remove)].index, dtype="uint32"
        )
    elif layer in sclist:
        assert atlas_df is not None
        to_remove = [
            atlas_df.loc[atlas_df.acronym == sc, "id"]
            for sc in sclist[: sclist.index(layer)]
        ]
        to_remove = np.squeeze(np.asarray(to_remove, dtype="uint32"))
    else:
        to_remove = np.asarray(layer, dtype="uint32")
    peel = np.isin(atlas, to_remove)
    if get_peel:
        return peel
    peeled_atlas = np.array(atlas, dtype="uint32")
    peeled_atlas[peel] = 0
    return peeled_atlas


def project_pts_to_pia(pts, atlas_folder, method="optimize", tree_dict=None):
    """Project pts to the closest border between layer and then iteratively towards the pia

    method can be `optimize`, for a slow 2D search on the UV coordinate for the point with the best projection
    or `via_wm` to go down to the white matter and then find the closest predefined path"""

    if not len(pts):
        return pts
    if method.lower() == "optimize":
        bff_file = os.path.join(atlas_folder, "bff_output.obj")
        vertices, vt, index3dto2d = load_flat_trans(bff_file)
        index2dto3d = dict([(v, k) for k, v in index3dto2d.items()])
        atlas = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(atlas_folder, "atlas.mhd"))
        )
        pia_pts = np.load(os.path.join(atlas_folder, "pial_coords.npy")).T
        pia_tree = KDTree(pia_pts)
        path_to_wm = np.load(os.path.join(atlas_folder, "path_from_pia_to_wm.npy"))
        coords = np.zeros_like(pts)
        print("projecting")
        n = len(pts)
        for i, single_pts in enumerate(pts):
            coords[i] = _find_shortest_dist_to_normal(
                single_pts,
                atlas_folder,
                atlas,
                vt,
                vertices,
                path_to_wm,
                pia_tree,
                index2dto3d,
                pia_pts,
            )
            print("%d / %d" % (i + 1, n))
    elif method.lower() == "via_wm":
        # project the point to wm
        wm_coords = project_pts_to_wm(pts, atlas_folder, tree_dict)
        # get the predefined path
        path = np.load(os.path.join(atlas_folder, "path_from_pia_to_wm.npy"))
        wm_tree = KDTree(path[:, :, 1])
        # find the closest 3 wm
        dist, indices = wm_tree.query(wm_coords, return_distance=True, k=3)
        # do the interpolation but between the pial coords
        pts_3d = np.zeros([3, pts.shape[0], pts.shape[1]])
        for i in range(3):
            pts_3d[i] = path[indices[:, i], :, 0]  # 0 to get the pial surface
        pts_3d = pts_3d.transpose((2, 1, 0))
        # for exact matches, keep them
        exact = dist[:, 0] == 0
        weighted_av = np.zeros_like(pts)
        weighted_av[exact] = pts_3d[:, exact, 0].T
        # for the rest do a wieghted average
        to_av = np.logical_not(exact)
        weight = 1 / dist[to_av]
        weighted_av[to_av] = (
            np.sum(pts_3d[:, to_av] * weight, 2) / np.sum(weight, 1)
        ).T
        coords = weighted_av
    else:
        raise IOError("Method must be `via_wm` or `optimize`")
    return coords


def get_path_across_layers(pia_pts, atlas_folder, tree_dict=None):
    """For each point on the surface find the shortest way to the white matter through all layers

    Requires the border points created by get_cortical_borders"""
    out = project_pts_to_wm(pia_pts, atlas_folder, tree_dict)
    path = np.dstack([pia_pts, out])
    # border_names = ['l1l234_border', 'l234l5_border', 'l5l6_border', 'l6wm_border']
    # borders = dict()
    # for b in border_names:
    #     borders[b] = np.load(os.path.join(atlas_folder, '%s.npy' % b))
    #
    # # query pts iteratively
    # path = np.zeros([len(pia_pts), 3, len(border_names) + 1])
    # path[:, :, 0] = pia_pts
    #
    # for i, layer in enumerate(border_names):
    #     print('Look for %s' % layer)
    #     print('Make a tree')
    #     bord_pts = borders[layer]
    #     tree = KDTree(bord_pts)
    #     print('Find the points')
    #     new_pts = tree.query(path[:, :, i], return_distance=False)
    #     print('Populate')
    #     path[:, :, i + 1] = bord_pts[new_pts[:, 0], :]

    return path


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


def lineseg_dist(p, a, b):
    """Calculate the distance between a line segment and point"""
    # normalized tangent vector
    d = np.divide(b - a, np.linalg.norm(b - a))

    # signed parallel distance components
    s = np.dot(a - p, d)
    t = np.dot(p - b, d)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, 0])

    # perpendicular distance component
    c = np.cross(p - a, d)

    return np.hypot(h, np.linalg.norm(c))
