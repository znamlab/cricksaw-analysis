"""
Functions linked to registration in itself

- register_one_step
- register_roi
"""

import os

import SimpleITK
from numpy import argmax


def apply_registration(
    transform_folder,
    path_to_image,
    root_name,
    suffix="",
    int_image=False,
    spacing=None,
    size=None,
    transformix_command="transformix",
):
    """Apply registration to an image

    :param transform_folder: folder to look for elastix transform
    :param path_to_image: path to an image file
    :param root_name: prefix name for the transform folder
    :param suffix: a string to append at the end of generated file names
    :param int_image: bool, should I make sure that the image stay an integer?
    :return:
    """
    file_names = os.listdir(transform_folder)
    transform_dict = {}
    for fName in file_names:
        if not os.path.isdir(os.path.join(transform_folder, fName)):
            continue
        if not fName.startswith(root_name + "_elastix_out_step"):
            continue
        elastix_files = os.listdir(os.path.join(transform_folder, fName))
        trans_files = [i for i in elastix_files if i.startswith("TransformParameters")]
        if len(trans_files) == 0:
            raise IOError("No transformation files for %s. Fix that" % fName)
        part = [int(i.split(".")[1]) for i in trans_files]
        good_file = trans_files[argmax(part)]
        transform_dict[fName] = good_file

    print("Doing %s" % path_to_image)
    assert os.path.exists(path_to_image)
    _, img_name = os.path.split(path_to_image)
    img_name, ext = os.path.splitext(img_name)

    # Apply the transformations
    steps = transform_dict.keys()
    print("%i steps to apply." % len(steps))
    for step in sorted(steps):
        print("    Step %s" % step)
        # Create a temporary folder to hold transformix output
        out_transformix_folder = step.replace("elastix", "transformix%s" % suffix)
        out_transformix_folder = os.path.join(transform_folder, out_transformix_folder)
        if not os.path.isdir(out_transformix_folder):
            os.mkdir(out_transformix_folder)

        trans_param_file = os.path.join(transform_folder, step, transform_dict[step])
        if int_image:
            _modify_parameter_file(
                trans_param_file, out_transformix_folder, True, size, spacing
            )
            trans_param_file = os.path.join(
                out_transformix_folder, transform_dict[step]
            )
        elif size is not None or spacing is not None:
            _modify_parameter_file(
                trans_param_file, out_transformix_folder, False, size, spacing
            )
            trans_param_file = os.path.join(
                out_transformix_folder, transform_dict[step]
            )
        # (FinalBSplineInterpolationOrder   0)
        # (ResultImagePixelType        "int")
        #
        # # Do the transformation
        # # There are nasty bugs in SimpleElastix. That fails with segfault:
        # # (see https: // github.com / SuperElastix / SimpleElastix / issues / 113)
        # transformix_image_filter = SimpleITK.TransformixImageFilter()
        # transformix_image_filter.SetTransformParameterMap(SimpleITK.ReadParameterFile(
        #   trans_param_file))
        # transformix_image_filter.SetMovingImage(SimpleITK.ReadImage(path_to_image))
        # transformix_image_filter.SetOutputDirectory(out_transformix_folder)
        # output = transformix_image_filter.Execute()
        # img = transformix_image_filter.GetResultImage()

        # So do it in bash
        bashcommand = r"%s -in %s -tp %s -out %s" % (
            transformix_command,
            path_to_image,
            trans_param_file,
            out_transformix_folder,
        )
        print("    Executing: %s" % bashcommand)
        os.system(bashcommand)

        path_to_image = os.path.join(out_transformix_folder, "result.mhd")

    # After make a copy of the last transform
    # Read the output
    res_img = SimpleITK.ReadImage(path_to_image)
    # Create a new pts file with output data
    path_to_image = img_name + "_transformed_%s%s%s" % (step, suffix, ext)
    path_to_image = os.path.join(transform_folder, path_to_image)
    SimpleITK.WriteImage(res_img, path_to_image)

    return res_img, path_to_image


def _modify_parameter_file(
    original_param_path, target_folder, int_image=False, size=None, spacing=None
):
    print("Modifying %s" % original_param_path)
    trans_param = SimpleITK.ReadParameterFile(original_param_path)
    if int_image:
        trans_param["FinalBSplineInterpolationOrder"] = ("0",)
        trans_param["ResultImagePixelType"] = ("int",)
    if size is not None:
        trans_param["Size"] = tuple(size)
    if spacing is not None:
        trans_param["Spacing"] = tuple(spacing)
    [_, fname] = os.path.split(original_param_path)
    if trans_param["InitialTransformParametersFileName"][0] != "NoInitialTransform":
        prev_step = trans_param["InitialTransformParametersFileName"]
        [_, prev_fname] = os.path.split(prev_step[0])
        trans_param["InitialTransformParametersFileName"] = (
            os.path.join(target_folder, prev_fname),
        )
        _modify_parameter_file(prev_step[0], target_folder, int_image, size, spacing)
    SimpleITK.WriteParameterFile(trans_param, os.path.join(target_folder, fname))


def registration_single_step(
    fixed_image,
    moving_image,
    path_to_save,
    prefix,
    step_number,
    transform_list,
    fixed_pts=None,
    moving_pts=None,
    parameter_map_dictionary=None,
    fixed_mask=None,
    moving_mask=None,
):
    """Do one single step of the registration.

    Default transform names are:
    'translation', 'rigid', 'affine', 'bspline', 'pts_translation', 'pts_rigid',
    'pts_affine', 'pts_bspline'

    :param fixed_image: SimpleITK image or file name
    :param moving_image: SimpleITK image or file name
    :param path_to_save: root folder to save data
    :param prefix: name to prepend to output (RR43_s04_s07 for instance)
    :param transform_list: list of transform to apply sequentially. Must be keys of
        `parameter_map_dictionary`
    :param fixed_pts: path to point file if needed
    :param moving_pts: path to point file if needed
    :param parameter_map_dictionary: a dictionary of parameter map. If None, the
        default will be loaded (works only if the M drive is mount in `/mnt/microscopy)`
    :param fixed_mask: Mask to apply to the fix image
    :param moving_mask: Mask to apply to the moving image
    :return: transformed image
    """

    if parameter_map_dictionary is None:
        parameter_map_dictionary = load_parameter_maps()

    if not isinstance(fixed_image, SimpleITK.SimpleITK.Image):
        fixed_image = SimpleITK.ReadImage(fixed_image)
    if not isinstance(moving_image, SimpleITK.SimpleITK.Image):
        moving_image = SimpleITK.ReadImage(moving_image)
    fixed_image.SetSpacing([1, 1, 1])
    moving_image.SetSpacing([1, 1, 1])
    if fixed_mask is not None and not isinstance(fixed_mask, SimpleITK.SimpleITK.Image):
        fixed_mask = SimpleITK.ReadImage(fixed_mask)
        fixed_mask = SimpleITK.Cast(fixed_mask, SimpleITK.sitkUInt8)
    if moving_mask is not None and not isinstance(
        moving_mask, SimpleITK.SimpleITK.Image
    ):
        moving_mask = SimpleITK.ReadImage(moving_mask)
        moving_mask = SimpleITK.Cast(moving_mask, SimpleITK.sitkUInt8)

    # Create a directory to save elastix log
    out_elastix_dir = os.path.join(
        path_to_save, prefix + "_elastix_out_step%02i" % step_number
    )
    if not os.path.isdir(out_elastix_dir):
        os.mkdir(out_elastix_dir)

    # init elastix image filter
    elastix_image_filter = SimpleITK.ElastixImageFilter()
    elastix_image_filter.SetOutputDirectory(out_elastix_dir)
    elastix_image_filter.SetMovingImage(moving_image)
    elastix_image_filter.SetFixedImage(fixed_image)
    if len(transform_list):
        elastix_image_filter.SetParameterMap(
            [parameter_map_dictionary[i] for i in transform_list]
        )
    else:
        print("No transform given, use default set")

    # Add points if needed
    if fixed_pts is not None:
        elastix_image_filter.SetFixedPointSetFileName(fixed_pts)
        elastix_image_filter.SetMovingPointSetFileName(moving_pts)

    change_sampler = False
    if fixed_mask is not None:
        elastix_image_filter.SetFixedMask(fixed_mask)
        change_sampler = True
    if moving_mask is not None:
        elastix_image_filter.SetMovingMask(moving_mask)
        change_sampler = True
    if change_sampler:
        print("There is a mask, Ill switch to randomMask sampler")
        new_params = []
        for param in elastix_image_filter.GetParameterMap():
            param["ImageSampler"] = ["RandomSparseMask"]
            new_params.append(param)
        elastix_image_filter.SetParameterMap(new_params)

    transformed_image = elastix_image_filter.Execute()
    SimpleITK.WriteImage(
        transformed_image,
        os.path.join(
            out_elastix_dir, prefix + "_registration_step%02i.mhd" % step_number
        ),
    )
    return transformed_image


def load_parameter_maps(
    param_path="/mnt/microscopy/Data/MF_data/Fabia/mice/RR/elastix_transforms",
):
    """Load parameter maps

    :param param_path: default
        '/mnt/microscopy/Data/MF_data/Fabia/mice/RR/elastix_transforms'
    :return pm_dict: dictionary of parameter maps
    """
    pm_dict = {}
    for fname in os.listdir(param_path):
        [rad, ext] = os.path.splitext(fname)
        if ext.lower() != ".txt":
            continue
        try:
            pm = SimpleITK.ReadParameterFile(os.path.join(param_path, fname))
            pm_dict[rad] = pm
            # Create a copy of parameter map using points
            pm = SimpleITK.ReadParameterFile(os.path.join(param_path, fname))
            pm["Metric"] = list(pm["Metric"]) + [
                "CorrespondingPointsEuclideanDistanceMetric"
            ]
            pm["Registration"] = ["MultiMetricMultiResolutionRegistration"]
            pm_dict["pts_" + rad] = pm
        except Exception:
            print("Fail to load: %s. Ignore the file" % fname)
            continue
    return pm_dict
