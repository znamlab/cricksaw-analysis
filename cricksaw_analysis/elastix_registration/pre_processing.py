import multiprocessing
import os
import xml.etree.ElementTree as ET

import numpy as np
import SimpleITK as sitk
import tifffile
import yaml
from joblib import Parallel, delayed
from PIL import Image
from skimage.transform import downscale_local_mean, resize

N_CORES = multiprocessing.cpu_count()


def get_downsampled_roi(
    original_image, on_focus_plane, z_in_atlas, roi_files=None, root_path=None
):
    raise NotImplementedError


def get_downsample_factor(path2tiff, outsize=25):
    """Get the factor needed to scale `path2tiff` so that pixels have `outsize` width


    :param path2tiff: path to tiff file (imagej or ome.tiff)
    :param outsize: size in the same unit as the pixels in the tiff (probably um)
    :return: xy_reduction_factor, outmetadata
    """
    print("Reading metadata")
    with tifffile.TiffFile(path2tiff) as tiffImage:
        metadata = dict()
        for tag in tiffImage[0].tags.values():
            metadata[tag.name.strip()] = tag.value
        x_res = metadata["x_resolution"]
        y_res = metadata["y_resolution"]
        pixel_size = [x_res[1] / x_res[0], y_res[1] / y_res[0]]
        # sometimes it seems that image resolution has a weird unit factor
        if "resolution_unit" in metadata:
            resu = metadata["resolution_unit"]
            try:
                resu = int(resu)
                if resu == 1:  # that means no absolute unit
                    pass
                elif resu == 2:  # that means inch
                    raise NotImplementedError
                elif resu == 3:  # that means cm
                    pixel_size = [i * 1e4 for i in pixel_size]  # switch to micrometers
                else:
                    raise IOError("weird unit")
            except ValueError:
                pass
        if tiffImage.is_ome:
            root = ET.fromstring(metadata["image_description"])
            # look for the image field
            ome = [r for r in root if r.tag.endswith("Image")]
            assert len(ome) == 1
            ome = ome[0]
            found = False
            for pixel_description in ome:
                if pixel_description.tag.endswith("Pixels"):
                    found = True
                    metadata.update(pixel_description.attrib)
                    break
            assert found
            # check that the pixel size are similar
            assert (float(metadata["PhysicalSizeX"]) - pixel_size[0]) < 0.01
            pixel_size = [
                float(metadata["PhysicalSizeX"]),
                float(metadata["PhysicalSizeY"]),
            ]
            assert metadata["PhysicalSizeXUnit"] == metadata["PhysicalSizeYUnit"]
            outmetadata = {"unit": metadata["PhysicalSizeYUnit"]}
        elif tiffImage.is_imagej:
            imagej_metadata = (
                metadata["image_description"].strip().decode("utf-8").split("\n")
            )
            for line in imagej_metadata:
                k, v = line.split("=")
                metadata[k] = v
            outmetadata = {}
            for k in ["spacing", "unit", "nslices"]:
                if k in metadata:
                    outmetadata[k] = metadata[k]
        else:
            outmetadata = {"unit": "nd"}  # codespell:ignore nd

    # Metadata are processed. Get downsample factor
    assert pixel_size[0] == pixel_size[1]  # It should be square pixels
    # output pixel size should be 25
    xy_reduction_factor = pixel_size[0] / outsize
    print("Got it. Scaling is about %.2f" % xy_reduction_factor)

    return xy_reduction_factor, outmetadata


def get_brainsaw_downsample_factor(path2recipe, outsize=25):
    """Get downsampling factor form a yaml recipe as created by brainsaw


    :param path2recipe: full path to the yaml file
    :param outsize: pixel size in XYZ after downsampling in same unit as input
        (presumably microns)
    :return: xy reduction factor
    :return: z_reduction factor
    """
    with open(path2recipe, "r") as stream:
        data = yaml.load(stream, Loader=yaml.FullLoader)
    voxel_size = data["VoxelSize"]
    x, y, z = [voxel_size[i] for i in ["X", "Y", "Z"]]
    if (
        z == 0
    ):  # if it's zero that means that there is a single plane per slice. Use slice
        # thickness as Z
        z = data["mosaic"]["sliceThickness"] * 1000  # put in microns
    assert (x > 0) and (z > 0)
    if np.abs(x - y) > 0.1:
        print("x and y differ. Will use x pixel size")
    print("Voxel size is %.2f in XY, %.2f in Z" % (x, z))
    xy_reduction_factor = x / outsize
    z_reduction_factor = z / outsize
    return xy_reduction_factor, z_reduction_factor


def upsample_brainsaw(
    path2img,
    path2recipe,
    z_planes_ori=None,
    z_planes_target=None,
    target=None,
    insize=25,
    check_for_ram=True,
    downsamplefactor=1,
):
    """Opposite of downsample_brainsaw. Upsample the registered atlas for instance to
    the original size

    This does not interpolate in Z but just takes the closest

    :param path2img: path to file to upsample
    :param path2recipe: path to the recipe file to read target pixel size
    :param z_planes_ori: z planes of the original image to upsample (incompatible with
        z_planes_target)
    :param z_planes_target: z planes of the target image to upsample (incompatible with
         z_planes_ori)
    :param target: file name (including extension) to save data. If None return
           array but don't write on disk
    :param insize: pixel size in XYZ of the input (25 if you use the classic atlas)
    :param n_cores: number of cores for multiprocessing
    :param check_for_ram: ask user confirmation if stack is more than 2Gb
    :param downsamplefactor: if stitchedImages_050 for instance
    :return big_data: array of downsampled data
    """

    z_planes_specified = np.sum(
        [int(i is None) for i in [z_planes_ori, z_planes_target]]
    )
    if z_planes_specified != 1:
        raise IOError("z_planes_ori OR z_planes_target must be supply")

    xy_reduction_factor, z_reduction_factor = get_brainsaw_downsample_factor(
        path2recipe, outsize=insize
    )
    xy_reduction_factor = xy_reduction_factor * downsamplefactor
    xy_increase_factor = 1 / xy_reduction_factor

    # now find the z_planes_ori if I need to start from the target
    if z_planes_ori is None:
        z_planes_ori = np.round(np.asarray(z_planes_target) * z_reduction_factor)
    z_planes_ori = np.atleast_1d(
        np.asarray(z_planes_ori, dtype=int)
    )  # to be able to index with that

    # read the small image
    img_data = sitk.GetArrayFromImage(sitk.ReadImage(path2img))
    # output of registration is in Z,X,Y, reshape to X,Y,Z
    img_data = img_data.transpose([1, 2, 0])
    # keep only relevant planes to free ram
    img_data = np.atleast_3d(img_data[:, :, z_planes_ori])
    out_shape = np.asarray(img_data.shape[:-1], dtype=float)
    out_shape = np.asarray(np.round(out_shape * xy_increase_factor), dtype=int)

    ram_size = (
        img_data.dtype.itemsize * np.prod(out_shape) * len(z_planes_ori) / 1024 / 1024
    )
    print("Big data is about %i Mo of ram." % ram_size)
    if (ram_size > 2000) and check_for_ram:
        keep_on = input(
            "That looks like a lot of RAM. Should I do it? [Y]/n " % ram_size
        )
        while keep_on.lower() not in ("", "y", "n"):
            keep_on = input("I did not get that. Should I keep on? Y for yes, n for no")
        if keep_on.lower() == "n":
            print("Stop here")
            return

    output = np.zeros(
        [out_shape[0], out_shape[1], len(z_planes_ori)], dtype=img_data.dtype
    )
    for i, plan in enumerate(z_planes_ori):
        small_data = img_data[:, :, i]
        big_data = np.array(
            Image.fromarray(small_data).resize(out_shape[::-1], Image.NEAREST),
            dtype=small_data.dtype,
        )
        output[:, :, i] = big_data
    return output


def downsample_brainsaw(
    path2tiffs,
    path2recipe,
    target=None,
    outsize=25,
    n_cores=N_CORES,
    check_for_ram=True,
    downsamplefactor=1,
):
    """Downsample data acquired by brainsaw

    :param path2tiffs: path to file to downsample
    :param path2recipe: path to the recipe file to read pixel size
    :param target: file name (including extension) to save data. If None return
           array but don't write on disk
    :param outsize: pixel size in XYZ after downsampling in same unit as input
           (presumably microns)
    :param n_cores: number of cores for multiprocessing
    :param check_for_ram: ask user confirmation if stack is more than 2Gb
    :param downsamplefactor: if stitchedImages_050 for instance
    :return small_data: array of downsampled data
    """

    xy_reduction_factor, z_reduction_factor = get_brainsaw_downsample_factor(
        path2recipe, outsize=outsize
    )
    xy_reduction_factor = xy_reduction_factor * downsamplefactor
    # Now get tiff file list
    tiff_files = sorted(
        [
            os.path.join(path2tiffs, fname)
            for fname in os.listdir(path2tiffs)
            if fname.endswith(".tif")
        ]
    )

    # do the first image
    small_data = _downsample_singlepage_bs(tiff_files[0], xy_reduction_factor)

    # calculate intermediate file size
    ram_size = (
        small_data.dtype.itemsize
        * np.prod(small_data.shape)
        * len(tiff_files)
        / 1024
        / 1024
    )
    print("Small data is about %i Mo of ram." % ram_size)
    if (ram_size > 2000) and check_for_ram:
        keep_on = input(
            "That looks like a lot of RAM. Should I do it? [Y]/n " % ram_size
        )
        while keep_on.lower() not in ("", "y", "n"):
            keep_on = input("I did not get that. Should I keep on? Y for yes, n for no")
        if keep_on.lower() == "n":
            print("Stop here")
            return

    # do all the others
    results = Parallel(n_jobs=n_cores)(
        delayed(_downsample_singlepage_bs)(i, xy_reduction_factor)
        for i in tiff_files[1:]
    )
    # add back the first stack
    results.insert(0, small_data)
    small_data = np.dstack(results).transpose([2, 0, 1])

    out_shape = list(small_data.shape)  # because output would be an unmutable tuple
    out_shape[0] = int(np.round(out_shape[0] * z_reduction_factor))
    if z_reduction_factor < 1:
        print("Downsampling in Z by %.2f" % z_reduction_factor)
        # same, first bin and then resize
        factor = int(np.floor(1 / z_reduction_factor))
        small_data = downscale_local_mean(small_data, (factor, 1, 1))
        small_data = resize(small_data, out_shape)
    elif z_reduction_factor > 1:
        print("UPSAMPLING in Z by %.2f" % z_reduction_factor)
        ram_size *= z_reduction_factor
        print("New size will be %i Mo." % ram_size)
        if (ram_size > 2000) and check_for_ram:
            keep_on = input(
                "I will take more than %i Mo of ram. Should I do it? [Y]/n " % ram_size
            )
            while keep_on.lower() not in ("", "y", "n"):
                keep_on = input(
                    "I did not get that. Should I keep on? Y for yes, n for no"
                )
            if keep_on.lower() == "n":
                print("Stop here")
                return
        small_data = resize(small_data, out_shape)
    else:
        print("No need to change Z.")

    if target is not None:
        _, ext = os.path.splitext(target)
        if ext == ".mhd":
            print("Writing %s" % target)
            mhd_image = sitk.GetImageFromArray(small_data)
            sitk.WriteImage(mhd_image, target)
        else:
            raise NotImplementedError
    else:
        print("No target. Do not save")
    return small_data


def _downsample_singlepage_bs(path2tiff, xy_reduction_factor):
    """Downsample a single page tiff file in xy. Made for brainsaw

    :param path2tiff:
    :param xy_reduction_factor:
    :return:
    """
    with tifffile.TiffFile(path2tiff) as tiffImage:
        assert len(tiffImage.pages) == 1
        page = tiffImage.pages[0]
        data = page.asarray()

    # First, bin in xy
    factor = int(np.floor(1 / xy_reduction_factor))
    small_data = downscale_local_mean(data, (factor, factor))
    # then interpolate to proper shape
    out_shape = np.array(
        np.round(np.asarray(data.shape) * xy_reduction_factor), dtype=int
    )
    small_data = resize(small_data, out_shape)
    return small_data


def _downsample(data, xy_reduction_factor, n_cores=N_CORES):
    """Function doing the downsampling.

    :param path2tiff:
    :param factor:
    :return:
    """
    if data.ndim == 2:
        # I have a multi page tiff. Make it 3D
        data = data.reshape([1] + list(data.shape))
    new_shape = np.array(
        np.floor(np.array(data.shape) * xy_reduction_factor), dtype=int
    )
    new_shape[0] = data.shape[0]
    results = Parallel(n_jobs=n_cores)(delayed(resize)(i, new_shape[1:]) for i in data)
    small_data = np.dstack(results).transpose([2, 0, 1])
    return small_data


def downsample_tiff(path2tiff, target, outsize=25, save_by_page=False):
    """Downsample a single tiff and write the output to disk

     The Z scaling is not changed

     This function should work with ImageJ files and with some ome.tiff files but for
     anything else, parsing the metadata  might fail (needed to get the pixel size)

    :param path2tiff: path to file to downsample
    :param target: file name to write the output
    :param outsize: pixel size in XY after downsampling in same unit as input
        (presumably microns)
    :param save_by_page: if true save a tiff file per page and target must be an
        existing directory (default false)
    :return: array of downsampled data if not save_by_page, else target path
    :return: xy reduction factor
    """
    if save_by_page:
        assert os.path.isdir(target)

    xy_reduction_factor, out_metadata = get_downsample_factor(path2tiff, outsize)
    path2file, filename = os.path.split(path2tiff)
    filename, ext = os.path.splitext(filename)

    print("Start scaling")
    with tifffile.TiffFile(path2tiff) as tiffImage:
        res = 1 / outsize

        # Process page by page
        rescaled = []
        for ip, page in enumerate(tiffImage):
            print("page %i" % ip)
            # load the data. This will load the whole tiff file. I hope it's not too big
            data = page.asarray()
            small_data = _downsample(data, xy_reduction_factor)
            if small_data.dtype == np.float64:
                print("!!! WARNING Downsampling float64 image to 32 !!!!")
                small_data = np.asarray(small_data, dtype=np.float32)
            if save_by_page:
                target_file = os.path.join(target, "%s_page%i%s" % (filename, ip, ext))
                tifffile.imsave(
                    target_file,
                    small_data,
                    imagej=True,
                    resolution=(res, res),
                    metadata=out_metadata,
                )
            else:
                rescaled.append(small_data)
    print("done scaling")
    if save_by_page:
        return target, xy_reduction_factor
    else:
        small_data = np.vstack(rescaled)
        tifffile.imsave(
            target,
            small_data,
            imagej=True,
            resolution=(res, res),
            metadata=out_metadata,
        )
        return small_data, xy_reduction_factor


def pre_processing(
    original_image, on_focus_plane, z_in_atlas, roi_files=None, root_path=None
):
    """Downsample the image and the ROIs. Create a volume with one image

    :param original_image: path to tiff image used to define ROIs
    :param on_focus_plane: plane in the tiff that is on focus (0 based)
    :param z_in_atlas: plane in the atlas corresponding to the tiff (0 based)
    :param roi_files: path to pts roi files
    :param root_path: path to save data (in downsamples and volume subfolders). Use
        original image parent if none
    :return:
    """

    ori_home, root_name = os.path.split(original_image)
    if not root_name.endswith(".ome.tiff"):
        print("!!!!!! WARNING I was expecting an ome_tiff. What happened? !!!!!")
    root_name = root_name[:-9]
    if root_path is None:
        root_path, _ = os.path.split(ori_home)
        path2volume = os.path.join(root_path, "volume_for_registration")
        if not os.path.isdir(path2volume):
            os.mkdir(path2volume)
        path2downsample = os.path.join(root_path, "downsampled_for_registration")
        if not os.path.isdir(path2downsample):
            os.mkdir(path2downsample)
    else:
        path2volume = root_path
        path2downsample = root_path

    downsample_image = os.path.join(path2downsample, root_name + "_%ium.tiff" % 25)
    small_data, xy_reduction_factor = downsample_tiff(
        original_image, target=downsample_image, outsize=25
    )
    # Make a volume (the ARA has 528 planes)
    volume = np.zeros([528, small_data.shape[1], small_data.shape[2]], dtype="uint16")
    volume[z_in_atlas] = small_data[on_focus_plane, :, :]
    mhd_image = sitk.GetImageFromArray(volume)
    sitk.WriteImage(
        mhd_image, os.path.join(path2volume, root_name + "_volume_25um.mhd")
    )
    mhd_image.GetSpacing()

    if roi_files is None:
        print("ooo")
        return mhd_image
    raise NotImplementedError
