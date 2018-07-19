=====
Usage
=====

Start by importing package_example.

.. code-block:: python

    import package_example
    
    
Functions
--------------------------
.. autofunction:: package_example.converter_saxs2gisaxs.gisaxs_construction
.. autofunction:: package_example.converter_saxs2gisaxs.gisaxs_full
.. autofunction:: package_example.converter_saxs2gisaxs.convert_saxs2gisaxs

Example
--------------------------
.. ipython:: python
   
   import numpy as np
   from pkg_resources import resource_filename
   
   a = np.load(resource_filename('example_data/example_data.npz'))
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

Plots
------------------------
.. plot:: 
   
   import numpy as np
   from pkg_resources import resource_filename
   
   a = np.load(resource_filename('example_data/example_data.npz'))
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
   
   fig,ax = plt.subplots(1,2,figsize = (6,3))
   ax[0].plot(q_reflc,np.log(reflectivity))
   ax[1].plot(q_reflc,transmission)
   
   fig,ax = plt.subplots(1,2)
   ax[0].imshow(np.log(saxs),vmin=1,vmax=12)
   ax[1].imshow(np.log(gisaxs),vmin=1,vmax=12)

