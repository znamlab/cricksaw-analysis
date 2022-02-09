from pathlib import Path

import numpy as np
from brainglobe_napari_io.cellfinder import reader_xml
from napari import Viewer
from skimage import morphology, measure
from tifffile import tifffile
import flexiznam as fzm


def find_injection_site(img, area, percentile=99.99):
    """Find the injection site in a volume

    This find the brightest blob in an image. It works well to find the injection site
    on a downsampled cricksaw brain (50um/px for instance)

    Args:
        img (np.array): 2D or 3D array with image data
        area (float): 2D area of small objects to ignore. Must be smaller than the
                      cross-section of the injection site
        percentile (float): percentile for initial thresholding.

    Returns:
        inj_site (np.array): bool array of the same size as img with True in the
                             injection site
    """

    threshold = np.percentile(img, percentile)
    inj_site = img > threshold
    inj_site = morphology.binary_dilation(inj_site)
    inj_site = morphology.remove_small_objects(inj_site, area)
    labeled = measure.label(inj_site)
    prop = measure.regionprops(labeled)
    # find biggest object
    area = [p.area for p in prop]
    label = [p.label for p in prop]
    biggest = label[np.argmax(area)]
    inj_site = labeled == biggest
    return inj_site


def cell_in_volume(cell_coords, volume_mask, cell_scale_um, volume_scale_um):
    """Find which cells are in the volume

    Args:
        cell_coords (array): coordinate of the cells array of Nx3
        volume_mask (bool array): 3D image with True inside the volume, False outside
        cell_scale_um (array): vector of 3 elements, unit size for cell coordinates
                               along the A/P, D/V and M/L axis
        volume_scale_um (float): voxel size, assuming isotropic voxel

    Returns:
        cells_in_volume (array): subsample of cell_coords for cells that fit in the
                                 volume
    """
    # find in which pixels are the cells
    px_cells = np.array(np.round(cell_coords * cell_scale_um / volume_scale_um),
                        dtype=int)
    in_volume = volume_mask[px_cells[:, 0], px_cells[:, 1], px_cells[:, 2]]
    cells_in_volume = cell_coords[in_volume]
    return cells_in_volume


def find_cell_in_injection_site(project, mouse, channel, pixel_size_um,
                                cell_scale_um,
                                inj_min_width=100,
                                inj_vol_percentile=99.99,
                                cell_type='Cells',
                                cell_kwargs=None,
                                view_in_napari=False):
    """Find cells in the injection site

    Threshold and binarise an image to find the injection site, then load cells and
    return the coordinate of those falling in the injection volume

    Args:
        project (str): name of the project to find the data
        mouse (str): name of the mouse
        channel (int): channel containing the fluorescence data
        pixel_size_um (int): size of the voxel to use for downsampled data volume
        cell_scale_um (array): vector of 3 elements, unit size for cell coordinates
                               along the A/P, D/V and M/L axis
        inj_min_width (float): minimal width of the injection site in um, all blobs
                               smaller than this will be excluded before dilating
        inj_vol_percentile (float): percentile for initial thresholding
        cell_type (str): name of the cell type in the cell xml, usually 'Cells' or
                         'Non Cells'
        cell_kwargs (dict): kwargs for pts layers. Used only if view_in_napari is True
        view_in_napari (bool): plot the results in a napari viewer

    Returns:
        cell_in_inj (array): coordinates of the cells found in the injection site
        inj_site (array): mask of the injection site
        cell_out_inj (array): coordinates of cells *not* in the injection site
        [optional] viewer (napari.Viewer): Only if view_in_napari is True
    """
    roots = {k: Path(v) for k, v in fzm.PARAMETERS['data_root'].items()}
    # find proper image
    img_path = roots['raw'] / project / mouse / 'downsampled_stacks' / ('%03d_micron' %
                                                                        pixel_size_um)
    chan_data = img_path.glob('*ch%02d*' % channel)
    img_path = dict()
    for fname in chan_data:
        if fname.suffix in img_path:
            raise IOError('Multiple files for %s.' % fname)
        img_path[fname.suffix] = fname

    img = tifffile.imread(img_path['.tif'])
    area = int(inj_min_width ** 2 / pixel_size_um)
    inj_site = find_injection_site(img, area, percentile=inj_vol_percentile)

    # get cells
    processed = roots['processed'] / project / mouse
    cell_xml = processed / ('cellfinder_results_%03d' % pixel_size_um) / 'points' \
               / 'cell_classification.xml'

    cell_classification = reader_xml.xml_reader(str(cell_xml))
    cell_classification = {c[1]['name']: c for c in cell_classification}
    cells = cell_classification[cell_type]
    cell_in_inj = cell_in_volume(cells[0], inj_site, cell_scale_um=cell_scale_um,
                                 volume_scale_um=pixel_size_um)
    cell_out_inj = cell_in_volume(cells[0], ~inj_site, cell_scale_um=cell_scale_um,
                                  volume_scale_um=pixel_size_um)

    if view_in_napari:
        df_kwargs = dict(size=[5, 10, 10], symbol='disc', edge_width=0.5,
                         face_color=np.array([190, 174, 212, 125.]) / 255)
        if cell_kwargs is None:
            cell_kwargs = df_kwargs
        else:
            cell_kwargs.update(df_kwargs)
        cell_kwargs['scale'] = cell_scale_um

        viewer = Viewer()
        viewer.add_image(img,
                         name='%s chan %d %dum' % (mouse, channel, pixel_size_um),
                         scale=np.ones(3) * pixel_size_um)

        viewer.add_labels(inj_site, name='Inj site %s cleaned' % mouse,
                          scale=np.ones(3) * pixel_size_um)
        cells[1].update(cell_kwargs)
        viewer.add_points(data=cell_out_inj, **cells[1])

        kwargs = dict(cells[1])
        kwargs['face_color'] = np.array([253, 192, 134, 125]) / 255
        kwargs['name'] = 'Cells in injection site'
        viewer.add_points(data=cell_in_inj, **kwargs)
        return cell_in_inj, inj_site, cell_out_inj, viewer
    return cell_in_inj, inj_site, cell_out_inj
