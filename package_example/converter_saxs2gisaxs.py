import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io as sio
import multiprocessing
from scipy import misc
import scipy.io as sio

PATH_TO_SAVE = '/Users/jiliangliu/Downloads/for_jiliang/gisaxs_image/'
os.chdir('/Users/jiliangliu/Dropbox/gisaxs_sim')
#I make SLD, SLDS =0, reflc, trans = 1, current condition only pattern shift count, we could try whether GAN could work on this or not then continue include more distortion
# SLD corresponds to the SLD of film
def GISAXS_constrcution(SAXS,
                        incident_angle,
                        SLD,
                        reflectivity,
                        transmission,
                        q_reflc,
                        qz,
                        Qz,
                        wavelength):
    "
    Computer the GISAXS pattern from SAXS pattern using DWBA
    See http://gisaxs.com/index.php/DWBA
    Parameters
    -----------
    SAXS: small angle scattering pattern 2D image, numpy.array
    incident_angle: Incident angle in degrees, float
    SLD: Scattering Length Density, float, parameter determine critical angle
         usually in scale x1e-6
    reflectivity: reflectivity curve, 1D numpy.array
    transimission: transmission curve, 1D numpy.array
    q_reflc: q for reflectivity and transmission curve, 1D numpy.array
    qz: q for detector space in vertical direction, 1D numpy.array
    Qz: q for reciprocal space of SAXS in vertical direction, 1D numpy.array
    wavelength: wavelength of incident X-ray, float 
    "
    incident_anlge = np.radians(incident_angle)
    ct_f = np.degrees(np.arcsin(wavelength*np.sqrt(16*np.pi*SLD)/4/np.pi))#0.0928039405254*.9
    #ct_f is critical angle which calculated from snell law using SLD
    film_n = 1-(np.radians(ct_f)/2**.5)**2
    ambient_n =1.
    qz = np.sort(qz) 
    #make sure qz always incline order
    horizon_qz_index = np.nanargmin(np.abs(qz-2*np.pi*np.sin(incident_anlge)/wavelength))
    #qz position of horizon within GISAXS pattern
    k0 = 2*np.pi/wavelength
    #k0 is precalculate parameter
    #q = k0*1/d, d correlate the distance in real space
    two_theta = 2*np.arcsin(qz[horizon_qz_index:]/2/k0)
    #two theta angle in rciprocal space, radians
    alpha_incident_eff = np.arccos(np.cos(incident_anlge)*ambient_n/film_n)
    #correct incident angle distortion due to refraction
    r_f = reflctivity[np.nanargmin(np.abs(q_reflc-2*k0*np.sin(alpha_incident_eff)))]
    
    t_f = transimission[np.nanargmin(np.abs(q_reflc-2*k0*np.sin(alpha_incident_eff)))]
        
    two_theta_r = np.arccos(np.cos(two_theta-incident_anlge)*ambient_n/film_n)-alpha_incident_eff
    two_theta_d = np.arccos(np.cos(two_theta-incident_anlge)*ambient_n/film_n)+alpha_incident_eff
    qz_r = 2*np.pi*2*np.sin(two_theta_r/2)/wavelength
    qz_d = 2*np.pi*2*np.sin(two_theta_d/2)/wavelength   
    qz_f = 2*k0*np.sin(np.arccos(np.cos(2*np.arcsin(qz[horizon_qz_index:]/2/k0)\
           -incident_anlge)*ambient_n/film_n))
    
    reflc_params = np.interp(qz_f,
                            q_reflc[np.nanargmin(np.abs(q_reflc-np.nanmin(qz_f))):\
                            np.nanargmin(np.abs(q_reflc-np.nanmax(qz_f)))],
                            reflctivity[np.nanargmin(np.abs(q_reflc-np.nanmin(qz_f))):\
                            np.nanargmin(np.abs(q_reflc-np.nanmax(qz_f)))])
    trans_params = np.interp(qz_f,
                            q_reflc[np.nanargmin(np.abs(q_reflc-np.nanmin(qz_f))):\
                            np.nanargmin(np.abs(q_reflc-np.nanmax(qz_f)))],
                            transmission[np.nanargmin(np.abs(q_reflc-np.nanmin(qz_f))):\
                            np.nanargmin(np.abs(q_reflc-np.nanmax(qz_f)))])
                        
    #im_GISAXS = np.zeros((len(qz_r),np.shape(SAXS)[1]))
    im_GISAXS = np.zeros((len(qz_r),np.shape(SAXS)[1]))
    TT = np.zeros((len(qz_r),np.shape(SAXS)[1]))
    TR = np.zeros((len(qz_r),np.shape(SAXS)[1]))
    RT = np.zeros((len(qz_r),np.shape(SAXS)[1]))
    RR = np.zeros((len(qz_r),np.shape(SAXS)[1]))
    
    for i in range(len(qz_r)):
        r_index = np.argmin(np.abs(qz_r[i]-Qz))
        d_index = np.argmin(np.abs(qz_d[i]-Qz))
        im_GISAXS[i,:] = (trans_params[i]**2*t_f**2*SAXS[d_index.astype(int),:]+
        t_f**2*reflc_params[i]**2*SAXS[r_index.astype(int),:]+
        trans_params[i]**2*r_f**2*SAXS[r_index.astype(int),:]+
        r_f**2*reflc_params[i]**2*SAXS[d_index.astype(int),:])
        TT[i,:] = trans_params[i]**2*t_f**2*SAXS[d_index.astype(int),:]
        TR[i,:] = t_f**2*reflc_params[i]**2*SAXS[r_index.astype(int),:]
        RT[i,:] = trans_params[i]**2*r_f**2*SAXS[r_index.astype(int),:]
        RR[i,:] = r_f**2*reflc_params[i]**2*SAXS[d_index.astype(int),:]
    return im_GISAXS,TT,TR,RT,RR,horizon_qz_index,qz_f,reflc_params,trans_params


# SLDS corresponds to the SLD of substrate
def GISAXS_full(SAXS,GISAXS,alpha_incident,
				SLD,SLDS,qz,wavelength,
				beamcenter_y,detector_distance,
				pixel_size,scale_factor=5):
    im_2 = np.zeros(np.shape(SAXS))
    #im_2 = np.zeros((256,256))
    im_2[:GISAXS.shape[0],:] = np.flipud(GISAXS)
    
    k0 = 2*np.pi/wavelength
    ct_f = np.degrees(np.arcsin(wavelength*np.sqrt(16*np.pi*SLD)/4/np.pi))
    ct_si = np.degrees(np.arcsin(wavelength*np.sqrt(16*np.pi*SLDS)/4/np.pi))
    qz_cr = np.copy(qz)
    alpha_incident = np.radians(alpha_incident)
    theta = np.round(2*np.sin(alpha_incident/2)*detector_distance/(pixel_size)).astype(int)
    
    # GISAXS
    if alpha_incident <= np.radians(ct_f):
        qz_cr[:beamcenter_y-theta] = k0*(np.sqrt((qz[:beamcenter_y-theta]/k0-np.sin(alpha_incident))**2-np.sin(alpha_incident)**2))
    else:
        qz_cr[:beamcenter_y-theta] = k0*(np.sqrt((np.sin(alpha_incident)**2-np.sin(ct_f*np.pi/180)**2))+\
                                    	np.sqrt((qz[:beamcenter_y-theta]/k0-np.sin(alpha_incident))**2-\
										np.sin(ct_f*np.pi/180)**2))
    # GTSXAS
    if alpha_incident <= np.radians(ct_f):
        qz_cr[beamcenter_y-theta:] = k0*(-np.sqrt((qz[beamcenter_y-theta:]/k0-np.sin(alpha_incident))**2-\
										np.sin(alpha_incident)**2+np.sin(ct_si*np.pi/180)**2))
    else:    
        qz_cr[beamcenter_y-theta:] = k0*(np.sqrt((np.sin(alpha_incident)**2-np.sin(ct_f*np.pi/180)**2))-\
      									np.sqrt((qz[beamcenter_y-theta:]/k0-np.sin(alpha_incident))**2-\
										np.sin(ct_f*np.pi/180)**2+np.sin(ct_si*np.pi/180)**2))
    
    for i in range(int(beamcenter_y-theta),len(qz)):
        im_2[i,:] = SAXS[np.argmin(np.abs(qz[i]-qz)).astype(int),:]/scale_factor
    
    im_2[np.isnan(im_2)]=0.01
    return im_2#[:256,:]


def convert_one_image(im_name,detector_distance,wavelength,
						beamcenter_y,beamcenter_x,reflc, q_reflc, 
						SLD, SLDS, trans_index, pixel_size,
						alpha_incident=0.18,plot_im=False, save_plot=True):
    im = im_name
    if beamcenter_y < im.shape[0]/2:  
        im = np.flipud(im)#/np.min(im))
        beamcenter_y = im.shape[0] - beamcenter_y
    #im here corresponds to the SAXS pattern, the example has beam center at upper portion of image, ths I flip image, 
    #it better to put center in the center of image or lower part of image
    #set alpha_incident from 0.12, 0.14, 0.16, 0.18, 0.20 then one SAXS will pair with five GISAXS see GAN could figure out the shift in GISAXS
    # detector_distance = f['attributes']['sample_det_distance']['value'].value
    # wavelength = f['attributes']['wavelength']['value'].value
    # beamcenter_y = f['attributes']['shape']['item000'].value-f['attributes']['beamy0']['value'].value
    # beamcenter_x = f['attributes']['beamx0']['value'].value
    ##
    #beamcenter_y,beamcenter_y corresponds to the pixel posisiton of diffraction pattern. should be able to know in SAXS generator
    #wavelength,deteector_distance,alpha_incident should be known as experiment parameters, may set detector distance to 5, wavelength 0.9184,
    #set different alpha incident, then could pair one SAXS pattern to several GISAXS pattern.
    qz = 2*np.pi*2*np.sin(np.arcsin((beamcenter_y-np.arange(0,im.shape[0],1))*\
		pixel_size/detector_distance)/2)/wavelength
    #qz calculation is ycenter and SAXS row number correlate image shape[0]–f['attributes']['shape']['item000']
    #qx = 2*np.pi*2*np.sin(np.arcsin((beamcenter_x-np.arange(0,im.shape[1],1))*75*1e-6/detector_distance)/2)/wavelength
    #same for column direction
    # os.chdir('/Volumes/NO NAME/GISAXS_reconstruction_function/without_fringe')

    # reflc = np.ones((len(reflc),)) 
    # trans_index = np.ones((len(trans_index),))
    # SLD = 0

    im_GISAXS,TT,TR,RT,RR,horizon_qz_index,\
    qz_f,reflc_params,trans_params = GISAXS_constrcution(SAXS=im,\
                            incident_angle=alpha_incident,SLD=SLD,\
                            reflectivity=reflc,transmission=trans_index,q_reflc=q_reflc,\
                            qz=qz,Qz=qz,wavelength=wavelength, reflc=reflc, trans_index=trans_index)                            

    SLDS = SLDS#30*1e-6

    im_full = GISAXS_full(SAXS=im,GISAXS=im_GISAXS,alpha_incident=alpha_incident,
						SLD=SLD,SLDS=SLDS,
                    	qz=qz,wavelength=wavelength,
						beamcenter_y=beamcenter_y,
						detector_distance=detector_distance, 
						pixel_size=pixel_size,scale_factor=50)
    return im_full,qz_f,reflc_params,trans_params
