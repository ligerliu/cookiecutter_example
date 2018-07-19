import numpy as np


def gisaxs_construction(SAXS,
                        incident_angle,
                        SLD,
                        reflectivity,
                        transmission,
                        q_reflc,
                        qz,
                        Qz,
                        wavelength):
    """
    Computer the GISAXS pattern from SAXS pattern using DWBA
    
    See http://gisaxs.com/index.php/DWBA
    
    Parameters
    ----------
    SAXS: 2D numpy array
          small angle scattering pattern 2D image.
    incident_angle: float
          Incident angle in degrees.
    SLD:  float
          Scattering Length Density,
          parameter determine critical angle,
          usually in scale x1e-6
    reflectivity:  1D numpy.array
    transmission: 1D numpy.array
    q_reflc: 1D numpy.array
             q for reflectivity and transmission.
    qz: 1D numpy array
        q for detector space in vertical direction.
    Qz: 1D numpy array
        q for reciprocal space of SAXS in vertical direction.
    wavelength: float
        wavelength of incident X-ray.
    
    Returns
    --------
    GISAXS pattern: 2D numpy array
        GISAXS only, theoritically below critical angle of material no scattering.
    """
    incident_anlge = np.radians(incident_angle)
    ct_f = np.degrees(np.arcsin(wavelength*np.sqrt(16*np.pi*SLD)/4/np.pi))
#    ct_f is critical angle which calculated from snell law using SLD
    film_n = 1-(np.radians(ct_f)/2**.5)**2
    ambient_n =1.
    qz = np.sort(qz) 
#    make sure qz always incline order
    horizon_qz_index = np.nanargmin(np.abs(qz-2*np.pi*np.sin(incident_anlge)/wavelength))
#   qz position of horizon within GISAXS pattern
    k0 = 2*np.pi/wavelength
#   k0 is precalculate parameter
#   q = k0*1/d, d correlate the distance in real space
    two_theta = 2*np.arcsin(qz[horizon_qz_index:]/2/k0)
#   two theta angle in rciprocal space, radians
    alpha_incident_eff = np.arccos(np.cos(incident_anlge)*ambient_n/film_n)
#   correct incident angle distortion due to refraction
    r_f = reflectivity[np.nanargmin(np.abs(q_reflc-2*k0*np.sin(alpha_incident_eff)))]
    
    t_f = transmission[np.nanargmin(np.abs(q_reflc-2*k0*np.sin(alpha_incident_eff)))]
        
    two_theta_r = np.arccos(np.cos(two_theta-incident_anlge) * \
                  ambient_n/film_n)-alpha_incident_eff
    two_theta_d = np.arccos(np.cos(two_theta-incident_anlge) * \
                  ambient_n/film_n)+alpha_incident_eff
    qz_r = 2*np.pi*2*np.sin(two_theta_r/2)/wavelength
    qz_d = 2*np.pi*2*np.sin(two_theta_d/2)/wavelength   
    qz_f = 2*k0*np.sin(np.arccos(np.cos(2*np.arcsin(qz[horizon_qz_index:]/2/k0)\
           -incident_anlge)*ambient_n/film_n))
    
    reflc_params = np.interp(qz_f,
                            q_reflc[np.nanargmin(np.abs(q_reflc-np.nanmin(qz_f))):\
                            np.nanargmin(np.abs(q_reflc-np.nanmax(qz_f)))], \
                            reflectivity[np.nanargmin(np.abs(q_reflc-np.nanmin(qz_f))): \
                            np.nanargmin(np.abs(q_reflc-np.nanmax(qz_f)))])
    trans_params = np.interp(qz_f,
                            q_reflc[np.nanargmin(np.abs(q_reflc-np.nanmin(qz_f))): \
                            np.nanargmin(np.abs(q_reflc-np.nanmax(qz_f)))], \
                            transmission[np.nanargmin(np.abs(q_reflc-np.nanmin(qz_f))): \
                            np.nanargmin(np.abs(q_reflc-np.nanmax(qz_f)))])
                        
#   im_GISAXS = np.zeros((len(qz_r),np.shape(SAXS)[1]))
    im_GISAXS = np.zeros((len(qz_r), np.shape(SAXS)[1]))
    TT = np.zeros((len(qz_r), np.shape(SAXS)[1]))
    TR = np.zeros((len(qz_r), np.shape(SAXS)[1]))
    RT = np.zeros((len(qz_r), np.shape(SAXS)[1]))
    RR = np.zeros((len(qz_r), np.shape(SAXS)[1]))
    
    for i in range(len(qz_r)):
        r_index = np.argmin(np.abs(qz_r[i]-Qz))
        d_index = np.argmin(np.abs(qz_d[i]-Qz))
        im_GISAXS[i,:] = (trans_params[i]**2*t_f**2*SAXS[d_index.astype(int), :]+
        t_f**2*reflc_params[i]**2*SAXS[r_index.astype(int), :]+
        trans_params[i]**2*r_f**2*SAXS[r_index.astype(int), :]+
        r_f**2*reflc_params[i]**2*SAXS[d_index.astype(int), :])
        TT[i,:] = trans_params[i]**2*t_f**2*SAXS[d_index.astype(int), :]
        TR[i,:] = t_f**2*reflc_params[i]**2*SAXS[r_index.astype(int), :]
        RT[i,:] = trans_params[i]**2*r_f**2*SAXS[r_index.astype(int), :]
        RR[i,:] = r_f**2*reflc_params[i]**2*SAXS[d_index.astype(int), :]
    return im_GISAXS


def gisaxs_full(SAXS,
                GISAXS,
                incident_angle,
                SLD,
                SLDS,
                qz,
                wavelength,
                beamcenter_y,
                detector_distance,
                pixel_size,
                scale_factor=10):
    """
    Computer a GISAXS pattern include the GTSAXS pattern；
    GTSAXS pattern is SAXS pattern with refraction dsitortion and 
    intensity of GTSAXS is scale down by scale factor.
    
    Parameters
    ----------
    SAXS:   2D numpy array;
    GISAXS: 2D numpy array;
    incident_angle: float
            incident angle of X-ray, degrees;
    SLD:  float
          scattering length density of film;
    SLDS: float
          scattering length density of substrate;
    qz: float
        q of vertical detector space; 1/Angstrom;
    wavelength: float
        wave length of incident X-ray
    beamcenter_y: int
        beam center at vertical detector space, pixel;
    detector_distance: float
        sample to detector distance, meter.
    pixel_size: float, meter.
    scale_factor: int 
        the factor scale down SAXS to simulate GTSAXS.
    
    Returns
    -------
    GISAXS: 2D numpy array
        including GISAXS and GTSAXS
    """
    im_full = np.zeros(np.shape(SAXS))
    im_full[:GISAXS.shape[0],:] = np.flipud(GISAXS)
    
    k0 = 2*np.pi/wavelength
    ct_f = np.degrees(np.arcsin(wavelength*np.sqrt(16*np.pi*SLD)/4/np.pi))
    ct_si = np.degrees(np.arcsin(wavelength*np.sqrt(16*np.pi*SLDS)/4/np.pi))
    qz_cr = np.copy(qz)
    incident_angle = np.radians(incident_angle)
#   theta correlate the pixel position of horizon, below horizon is GTSAXS; above is GISAXS
    theta = np.round(2*np.sin(incident_angle/2)*detector_distance/(pixel_size)).astype(int)
    
#   refraction correction of qz for GISAXS and GTSAXS had been calculated basing on paper: 
#   Lu, X. et al., J. of Appl. Cryst.2013.doi: 10.1107/S0021889812047887
    
#   GISAXS qz refraction correction
    if incident_angle <= np.radians(ct_f):
        qz_cr[:beamcenter_y-theta] = k0*(np.sqrt((qz[:beamcenter_y-theta]/k0 - \
                                     np.sin(incident_angle))**2-np.sin(incident_angle)**2))
    else:
        qz_cr[:beamcenter_y-theta] = k0*(np.sqrt((np.sin(incident_angle)**2 - \
                                     np.sin(ct_f*np.pi/180)**2))+\
                                     np.sqrt((qz[:beamcenter_y-theta]/k0 - \
                                     np.sin(incident_angle))**2-\
                                     np.sin(ct_f*np.pi/180)**2))
#   GTSXAS qz refraction correction
    if incident_angle <= np.radians(ct_f):
        qz_cr[beamcenter_y-theta:] = k0*(-np.sqrt((qz[beamcenter_y-theta:]/ \
                                     k0-np.sin(incident_angle))**2-\
									 np.sin(incident_angle)**2+np.sin(ct_si*np.pi/180)**2))
    else:    
        qz_cr[beamcenter_y-theta:] = k0*(np.sqrt((np.sin(incident_angle)**2 - \
                                     np.sin(ct_f*np.pi/180)**2))-\
                                     np.sqrt((qz[beamcenter_y-theta:] / \
                                     k0-np.sin(incident_angle))**2-\
                                     np.sin(ct_f*np.pi/180)**2+np.sin(ct_si*np.pi/180)**2))
    
    for i in range(int(beamcenter_y-theta), len(qz)):
        im_full[i,:] = SAXS[np.argmin(np.abs(qz[i]-qz)).astype(int), :]/\
                       scale_factor
    
    im_full[np.isnan(im_full)]=0.01
    return im_full


def convert_saxs2gisaxs(SAXS,
                        detector_distance,
                        wavelength,
                        beamcenter_y,
                        reflectivity,
                        transmission, 
                        q_reflc, 
						SLD, 
                        SLDS,
                        pixel_size,
						incident_angle,
                        scale_factor):
    
    """
    Computer a GISAXS pattern include the GTSAXS pattern；
    this converter function will calcualte the qz and collibrate the SAXS 
    for GISAXS
    
    Parameters
    ----------
    SAXS:  2D numpy array
         Small Angle Xray Scattering pattern
    detector_distance :  float
         sample to detector distance
    wavelength :  float
         wave length of incident X-ray
    beamcenter_y :  int
         beam center at vertical detector space, pixel;
    reflectivity :  1D numpy.array
         reflectivity coefficient of material
    transmission :  1D numpy.array
         transmission coefficient of material
    q_reflc : 1D numpy array
         q for reflectivity and transmission curve;
    SLD : float
         scattering length density of film
    SLDS : float
         scattering length density of substrate
    pixel_size :  float
         actual pixel size of detector, in meter
    incident_angle :  float
         incident angle of X-ray, degrees
    scale_factor :  int
         the factor scale down SAXS to simulate GTSAXS
    
    Returns
    -------
    GISAXS pattern :  2D numpy.array
         including GISAXS and correlated GTSAXS
    """
    im = SAXS
    if beamcenter_y < im.shape[0]/2:  
        im = np.flipud(im)
        beamcenter_y = im.shape[0] - beamcenter_y
    qz = 2*np.pi*2*np.sin(np.arcsin((beamcenter_y-\
         np.arange(0, im.shape[0],1))*\
         pixel_size/detector_distance)/2)/wavelength
    im_GISAXS  = gisaxs_construction(SAXS=SAXS,
                                     incident_angle=incident_angle,
                                     SLD=SLD,
                                     reflectivity=reflectivity,
                                     transmission=transmission,
                                     q_reflc=q_reflc,
                                     qz=qz,
                                     Qz=qz,
                                     wavelength=wavelength)                            
    
    SLDS = SLDS
    
    im_full = gisaxs_full(SAXS=SAXS,
                          GISAXS=im_GISAXS,
                          incident_angle=incident_angle,
                          SLD=SLD,
                          SLDS=SLDS,
                          qz=qz,
                          wavelength=wavelength,
                          beamcenter_y=beamcenter_y,
                          detector_distance=detector_distance,
                          pixel_size=pixel_size,
                          scale_factor=scale_factor)
    return im_full
