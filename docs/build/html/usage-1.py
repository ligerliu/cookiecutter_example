import numpy as np
from pkg_resources import resource_filename
#import os

#os.chdir('/Users/jiliangliu')
#filename = resource_filename(__name__,'package_example')
#path = filename + '/example_data/example_data.npz'
path = resource_filename('package_example','example_data/example_data.npz')
a = np.load(path)
saxs = a['example_data'].item()['saxs']
reflectivity = a['example_data'].item()['reflectivity']
transmission = a['example_data'].item()['transmission']
q_reflc = a['example_data'].item()['q_reflc']
SLD = a['example_data'].item()['SLD']
SLDS = a['example_data'].item()['SLDS']
incident_angle = a['example_data'].item()['incident_angle']
wavelength = 0.9184
detector_distance = 1.2
pixel_size = 75*1e-6
scale_factor = 10
beamcenter_y = 256

from package_example.converter_saxs2gisaxs import convert_saxs2gisaxs

gisaxs = convert_saxs2gisaxs(saxs, detector_distance, wavelength,
         beamcenter_y, reflectivity, transmission, q_reflc, SLD,
         SLDS, pixel_size,incident_angle,scale_factor)
import matplotlib.pyplot as plt
fig,ax = plt.subplots(1,2)
ax[0].imshow(np.log(saxs))
ax[1].imshow(np.log(gisaxs))