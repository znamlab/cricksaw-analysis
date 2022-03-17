from pathlib import Path
import tifffile
import pandas as pd


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
