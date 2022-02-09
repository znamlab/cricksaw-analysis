from napari.viewer import Viewer
from brainglobe_napari_io.cellfinder import reader_xml
import tifffile

viewer = Viewer()

# load single channel of a multitiff
path = '/Users/blota/Data/BRYC64.2h_injection_site_fixed_dimension.tif'
channel = 0
img_data = tifffile.imread(path)
img = viewer.add_image(img_data[:, channel, :, :], contrast_limits=[0, 2500],
                       name='BRYC64.2h')

# load cells
path = '/Users/blota/Data/cellfinder_results_050/points/cell_classification.xml'
data = reader_xml.xml_reader(path)
for dataset in data:
    viewer.add_points(dataset[0], **dataset[1])

path = '/Users/blota/Data/cellfinder_results_050/points/cell_classification.xml'
data = reader_xml.xml_reader(path)
for dataset in data:
    viewer.add_points(dataset[0], **dataset[1])
