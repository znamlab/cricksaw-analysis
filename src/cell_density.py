"""Small script to threshold downsampled data and find cells in injection site"""
from napari.viewer import Viewer

from pathlib import Path
from brainglobe_napari_io.cellfinder import reader_xml
import tifffile
import numpy as np

processed = Path('D:/')
raw = Path('D:/')
mouse = 'BRYC64.2h'
channel = 3
pixel_size = 50
path = processed / mouse / 'downsampled_stacks' / ('%02d_micron' % pixel_size)

# find proper image
chan_data = path.glob('*ch%02d*' % channel)
img_path = dict()
for fname in chan_data:
    if fname.suffix in img_path:
        raise IOError('Multiple files for %s.' % fname)
    img_path[fname.suffix] = fname

img = tifffile.imread(img_path['.tif'])
threshold = np.percentile(img, 99.99)
inj_site = img > threshold


viewer = Viewer()

img_l = viewer.add_image(img, name='%s chan %d %dum' % (mouse, channel, pixel_size))
inj_site_l = viewer.add_labels(inj_site, name='Inj site %s' % (channel))

from skimage import morphology
from scipy import ndimage as ndi
inj_site = morphology.binary_dilation(inj_site)
inj_site_l = viewer.add_labels(inj_site, name='Inj site %s dilated' % (mouse))
inj_site_c = morphology.remove_small_objects(inj_site,
                                           area_threshold=int(750**3/pixel_size))
inj_site_m = viewer.add_labels(inj_site_c, name='Inj site %s cleaned' % (mouse))
labeled = ndi.label(inj_site_c)
labelled_inj_site = viewer.add_labels(labeled, name='%s labeled' % (
    mouse))
