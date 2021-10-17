#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 09:42:33 2021

@author: Amanda Aguiar

This code will allow us to predict the satellite positions from a certain place
and obtain different types of plots
"""

import skyfield
import numpy as np
import matplotlib.pyplot as plt
from skyfield.api import load, wgs84
from skyfield.api import EarthSatellite
from skyfield.api import N, S, E, W 
from spacetrack import SpaceTrackClient
from datetime import datetime
import datetime as DT
import os
import healpy as hp
from mpl_toolkits.basemap import Basemap
import pandas as pd
import stcache

import satpred as satpred

#mirar si cuadra lo de hor_axis
#quitar identity y passworkd y directorios

plt.close('all') #The figures that were obtained in the previous run are closed

###############################################################################
################################# INPUT DATA ##################################
###############################################################################

#WARNING: ALL THE INPUT VALUES MUST BE WRITTEN AS INDICATED IN THE COMMENTS,
#OTHERWISE THE CODE WILL NOT WORK 

################################ Location data ################################

#The name of the place we want to observe is defined

##SUPPORTED INPUT VALUES: #string with the name of the place
observation_place = 'OT'

#The coordinates of the observation place are defined
##SUPPORTED INPUT VALUES: #int or float numbers with the proper coordinates
####DEFAULT VALUE: Teide Observatory data
altitude = 2390.
longitude = 16.509722*W
latitude = 28.309722*N

#The air temperature and pressure in the observation place at the moment in
# which we want to obtain the plot are defined. These values can usually be 
#found at this next link for the Teide Observatory:
#https://www.iac.es/index.php/en/observatorios-de-canarias/teide-observatory/weather  
#They are used in order to estimate where an object might appear in the 
#sky after the Earth's atmosphere has refracted its image higher. If they are
#not defined, the altitude that is obtained corresponds to the position
#you would see with no atmosphere
#This is just an estimation, since the effects of the atmosphere cannot be 
#predicted to high precision

##SUPPORTED INPUT VALUES: #int or float number that correspond to the air 
                          #temperature in ºC or None and air pressure in mb or 
                          #None. If None, this  effect is not taken into 
                          #consideration
temperature = None #Cº
pressure = None #mb

#################################### Dates ####################################

#A starting and an ending time for THE WHOLE CODE must be selected
#This will allow us to select all bodies that are associated with an ID and 
#TLEs that were on orbit in this date range
#If the body was not on orbit in the selected time, it will simply not be taken
#into account in the calculations
#If the calculations are made for a series of dates, these parameters define 
#the selected date range

##SUPPORTED INPUT VALUES: #Dates with the format 'YYYY-MM-DD HH:MM:SS' or simply
                          #'YYYY-MM-DD'. Write 'now' to select the current 
                          #instant according to the time of our computer                                       
start =  '2020-05-10 03:00:00'
end =  '2020-05-11 03:35:00'

#The number of seconds between dates in which the calculations will me made
#is also selected (resolution)

##SUPPORTED INPUT VALUES: #int or float number in seconds 
seconds_sk = 80 #s

#The time in which we want to obtain the plots and files that correspond to
#a single date is selected

##SUPPORTED INPUT VALUES: #Dates with the format 'YYYY-MM-DD HH:MM:SS' or simply
                          #'YYYY-MM-DD'. Write 'now' to select the current 
                          #instant according to the time of our computer
time = start         

#We decide on a number of difference between the selected dates in order to
#decide if it is worth it to use more than one TLE
tol_sec = 48*60*60 #Tolerance seconds 

############################### Satellite data ################################

#The basic data of the bodies, which will allow us to select we ones we are
#interested in is obtained
filename = '/example/path_basic_data/Basic_sat_data.txt'
sat_IDs, sat_names, companies, launch, decay, comment = np.loadtxt(filename, unpack=True, delimiter = ';', dtype=str) 
sat_IDs_int = sat_IDs.astype(int)

############################# Satellite selection #############################

#We determine the way in which we select the satellites we want to study

##SUPPORTED INPUT VALUES: #'Group', 'Name', 'Company'
sel_mode = 'Name' 

#The specific objects we want to study are selected

##SUPPORTED INPUT VALUES: #If mode_sel = 'Group':
                          ##'On orbit': All active objects between the chosen dates
                          #are selected
                          ##'On orbit satellites': All satellites that were on
                          #orbit between the chosen dates are selected
                          ##'On orbit STARLINK satellites': All on orbit 
                          #STARLINK satellites between the chosen dates are 
                          #selected
                          #If model_sel = 'Name' or model_sel = 'Company', we 
                          #must write the names or NORAD IDs and companies 
                          #respectively inside  a list and as string objects
                          #with all capital letters
                            #WARNING: It is better to use the ID of the 
                            #satellites rather than the name of the 
                            #satellite in order to avoid problems due to the 
                            #different format and spelling
sat_list =  ['44236']

sat_sel_str = satpred.selection(filename = filename, mode_sel = sel_mode, sat_sel = sat_list, start = start, \
              end = end)
sat_sel = sat_sel_str.astype(int)

############################## Two-Line Elements ##############################

#The TLEs are used in order to make the calculations are selected and 
#extracted as skyfield.sgp4lib.EarthSatellite objects

#We indicate if we want to retrieve the TLEs from the TLE cache or from 
#storage

##SUPPORTED INPUT VALUES: #'cache', 'storage'
TLE_ret = 'cache'

##WARNING: The TLE local cache has several bugs if we use it to extract data
#after December 2020. This is currently a work in progress

if TLE_ret == 'cache':
    
    #The path in which the neccesary TLEs are saved is introduced, as well as
    #the user and password of the SpaceTrack account in case the necessary
    #TLEs are not in the local cache yet 
    path_TLE = '/example/path_TLE/'
    identity = 'example_user@not_real.com'
    password = 'example_password_not_real'


elif  TLE_ret == 'storage':
    
    #The path in which the neccesary TLEs are saved is introduced
    path_TLE = '/example/path_TLE/'

################################# Output data #################################

#We decide which figures and calculations we want to writing the proper names
#that are contained in the following list:
##SUPPORTED INPUT VALUES: #'ELvsAZ (static)','ELvsAZ (trace)', 
                          #'LATvsLON (static)', 'LATvsLON (healpix map)', 
                          #'Earth map (static)', 'Earth map (trace)',
                          #'Distance (histogram)', 'Rise&Set'
calc_list = ['ELvsAZ (static)']

#The path in which the possible outputs are saved is defined
path_outputs = '/example/path_outputs/'

##SUPPORTED INPUT VALUES: #'Yes', 'No'
save_file = 'No'
save_plot = 'No'
save_histogram = 'No'

#We decide if we want the azimuth axis to be between 0 and 360 deg or between
#-180 and 180 deg 
##SUPPORTED INPUT VALUES: #'0 to 360', '-180 to 180'
hor_axis = '0 to 360'

###############################################################################
################################ CALCULATIONS #################################
###############################################################################

################################ List of dates ################################

#A list of dates between the starting and ending date is created with a certain
#period of time between the consecutive ones
date_start = satpred.time_sel(time)[1]
date_end = satpred.time_sel(end)[1]
date_modified = date_start
list_dates = [date_modified.strftime('%Y-%m-%d %H:%M:%S')]


while date_modified < date_end:
    date_modified+=DT.timedelta(seconds=seconds_sk)
    list_dates.append(date_modified.strftime('%Y-%m-%d %H:%M:%S'))  

    
####################### TLEs extraction and parameters ########################
    
#The TLEs both for the static and for the multiple-dates plots are obtained
    
#The satellite objects are obtained for the static plots if we have decided
#that we want to obtain them

if any(t[-8:] == '(static)' for t in calc_list) or 'Distance (histogram)' in calc_list:
   
    if TLE_ret == 'cache':
        
        satellites = satpred.TLE_sel_stcache(identity = identity, \
                            password = password, \
                            path_TLE = path_TLE,\
                            time = time, sat_sel = sat_sel)
             
    elif  TLE_ret == 'storage':
        
        #The satellite objects are obtained for the static plots if we have decided
        #that we want to obtain them
        satellites = satpred.TLE_sel(path_TLEs = path_TLE, \
                                   time = time, sat_sel = sat_sel)
        
    if satellites != []:
        
        par_st = satpred.parameters(filename=filename,satellites=satellites, sat_sel=sat_sel, time=start)

sat_IDs_list, sat_names_list, alt_list, az_list, h_list, lat_list, long_list, l_list, b_list, color_list = par_st
       
#The satellite objects are obtained for the non-static plots if we have decided
#that we want to obtain them

#In this case, lists of lists are needed 
sat_IDs_suplist = [ [] for _ in range(len(list_dates))]
sat_names_suplist = [ [] for _ in range(len(list_dates))]
alt_suplist = [ [] for _ in range(len(list_dates))]
az_suplist = [ [] for _ in range(len(list_dates))]
h_suplist = [ [] for _ in range(len(list_dates))]
lat_suplist = [ [] for _ in range(len(list_dates))]
long_suplist = [ [] for _ in range(len(list_dates))]
l_suplist = [ [] for _ in range(len(list_dates))]
b_suplist = [ [] for _ in range(len(list_dates))]
color_suplist = [ [] for _ in range(len(list_dates))]

if any(t[-11:] == '(animation)' for t in calc_list) or any(t[-7:] == '(trace)' for t in calc_list) or any(t[-13:] == '(healpix map)' for t in calc_list):

    #The first and last date are compared to see if we are interested
    #in skipping TLEs or if it is not worth it
    first_date = datetime.strptime(list_dates[0], '%Y-%m-%d %H:%M:%S') 
    last_date = datetime.strptime(list_dates[-1], '%Y-%m-%d %H:%M:%S')
    
    #If it is best to use only one TLE so that the code is faster
    if (last_date - first_date).total_seconds() <= tol_sec:
        
        if TLE_ret == 'cache':
            
            satellites = satpred.TLE_sel_stcache(identity = identity, \
                                password = password, \
                                path_TLE = path_TLE,\
                                time = list_dates[0], sat_sel = sat_sel)

        elif  TLE_ret == 'storage':

            satellites = satpred.TLE_sel(path_TLEs = path_TLE, \
                       time = list_dates[0], sat_sel = sat_sel)
            
        for p in range(len(list_dates)):
            
            par = satpred.parameters(filename=filename,altitude = altitude, longitude = longitude,\
               latitude = latitude,satellites=satellites, sat_sel=sat_sel, \
               time=list_dates[p], observation_place = observation_place,\
               hor_axis = hor_axis)
            
            sat_IDs_suplist[p] = par[0]
            sat_names_suplist[p] = par[1]
            alt_suplist[p] = par[2]
            az_suplist[p] = par[3]
            h_suplist[p] = par[4]
            lat_suplist[p] = par[5]
            long_suplist[p] = par[6]
            l_suplist[p] = par[7]
            b_suplist[p] = par[8]    
            color_suplist[p] = par[9]   
            
            print(list_dates[p])
            
    else:
    
        for p in range(len(list_dates)):
            
            if TLE_ret == 'cache':
                
                satellites = satpred.TLE_sel_stcache(identity = identity, \
                                    password = password, \
                                    path_TLE = path_TLE,\
                                    time = list_dates[p], sat_sel = sat_sel)
    
                
            elif  TLE_ret == 'storage':
    
                satellites = satpred.TLE_sel(path_TLEs = path_TLE, \
                           time = list_dates[p], sat_sel = sat_sel)
    
                
            if satellites !=[]:        
        
                par = satpred.parameters(filename=filename,altitude = altitude, longitude = longitude,\
               latitude = latitude,satellites=satellites, sat_sel=sat_sel, \
               time=list_dates[p], observation_place = observation_place,\
               hor_axis = hor_axis)
                
                sat_IDs_suplist[p] = par[0]
                sat_names_suplist[p] = par[1]
                alt_suplist[p] = par[2]
                az_suplist[p] = par[3]
                h_suplist[p] = par[4]
                lat_suplist[p] = par[5]
                long_suplist[p] = par[6]
                l_suplist[p] = par[7]
                b_suplist[p] = par[8]    
                color_suplist[p] = par[9]   
                
                print(satellites)
        

############################ STATIC EL vs AZ PLOT #############################
    
if 'ELvsAZ (static)' in calc_list:
    
    fig_orb, ax_orb = plt.subplots(figsize = (5*4.,5.))  
    title = satpred.time_sel(time)[1].strftime('%Y-%m-%d %H:%M:%S') + ' UTC'
    
    ax_orb.set_title('OT\n' + title)
    ax_orb.set_xlabel('AZ[degree]')
    ax_orb.set_ylabel('EL[degree]')
    ax_orb.set_ylim(-90., 0.)
    
    if hor_axis == '0 to 360':
        ax_orb.set_xlim(0, 360)
    if hor_axis == '-180 to 180':
        ax_orb.set_xlim(-180, 180)
    
    for k in range(len(sat_IDs_list)):
        ax_orb.plot(az_list[k], alt_list[k], '.', label = sat_names_list[k] + ' (' + str(sat_IDs_list[k]) + ')', markersize=3, color=color_list[k])

    if len(sat_sel) < 20:                                                   
        ax_orb.legend(loc='upper left')


############################# DISTANCE HISTOGRAM  #############################

if 'Distance (histogram)' in calc_list:
    
    if 'ELvsAZ (static)' in calc_list:
        fig_hist, ax_hist = plt.subplots(figsize = (15.,10.5))
        satpred.dist_hist(fig_hist, ax_hist, h_list, save_histogram = 'Yes', hist_saved = 'Histogram.png', nticks=1, nbins=1)
        
    else:
        print('The height histogram can only be made if the ELvsAZ (static) is retreived first')

######################### EL vs AZ PLOT LEAVING TRACE #########################

if 'ELvsAZ (trace)' in calc_list:
    
    fig_orb_t, ax_orb_t = plt.subplots(figsize = (5*4.,5.))
    
    title = satpred.time_sel(time)[1].strftime('%Y-%m-%d %H:%M:%S') + ' UTC'
    
    ax_orb_t.set_title('OT\n' + title)
    ax_orb_t.set_xlabel('AZ[degree]')
    ax_orb_t.set_ylabel('EL[degree]')
    ax_orb_t.set_ylim(-90., 0.)
    
    if hor_axis == '0 to 360':
        ax_orb_t.set_xlim(0, 360)
    if hor_axis == '-180 to 180':
        ax_orb_t.set_xlim(-180, 180)
    
    for p in range(len(list_dates)):
        
        for i in range(len(sat_sel)): 
            
            if az_suplist[p] != []:
                
                if p ==0:
        
                    ax_orb_t.plot(az_suplist[p][i], alt_suplist[p][i], '.', label = sat_names_suplist[p][i] + ' (' + str(sat_IDs_suplist[p][i]) + ')', markersize=3, color=color_suplist[p][i])
                    
                else:    
                    
                    ax_orb_t.plot(az_suplist[p][i], alt_suplist[p][i], '.', markersize=3, color=color_suplist[p][i])

    if len(sat_sel) < 20:                                                   
        ax_orb_t.legend(loc='upper left')

########################### STATIC LAT vs LON PLOT  ###########################
        
if 'LATvsLON (static)' in calc_list:
    
    fig_orb_latlon_st, ax_orb_latlon_st = plt.subplots(figsize = (5*4.,5.))  
    title = satpred.time_sel(time)[1].strftime('%Y-%m-%d %H:%M:%S') + ' UTC'
    
    ax_orb_latlon_st.set_title('OT\n' + title)
    ax_orb_latlon_st.set_xlabel('LAT[degree]')
    ax_orb_latlon_st.set_ylabel('LON[degree]')
    ax_orb_latlon_st.set_ylim(-90., 0.)
    
    if hor_axis == '0 to 360':
        ax_orb_latlon_st.set_xlim(0, 360)
    if hor_axis == '-180 to 180':
        ax_orb_latlon_st.set_xlim(-180, 180)
    
    for k in range(len(sat_IDs_list)):
        
        ax_orb_latlon_st.plot(l_list[k], b_list[k], '.', label = sat_names_list[k] + ' (' + str(sat_IDs_list[k]) + ')', markersize=3, color=color_list[k])

    if len(sat_sel) < 20:                                                   
        ax_orb_latlon_st.legend(loc='upper left')

######################## STATIC LAT vs LON HEALPIX MAP ########################
        
if 'LATvsLON (healpix map)' in calc_list:
    
    l_box= 6
    nside = 2**l_box
    print('Map resolution: ' + str(np.degrees(hp.nside2resol(nside))))

    data = np.linspace(0,len(sat_sel),len(sat_sel))
    
    m = np.zeros(hp.nside2npix(nside))
    
    for p in range(len(list_dates)):
        
        for i in range(len(sat_sel)): 
            
            if l_suplist[p] != []:
        
                pixel_indices = hp.ang2pix(nside, l_suplist[p][i], b_suplist[p][i], lonlat=True)
                m[pixel_indices] = data + 10*(i+1) 
                
    hp.mollview(map=m, fig=len(calc_list), xsize=800, title= 'CHOOSE TITLE')            
    hp.graticule()
    
############################## STATIC EARTH MAP ###############################

if 'Earth map (static)' in calc_list:

    fig = plt.figure(figsize=(12,9))
    
    me = Basemap(projection='mill',
               llcrnrlat = -90,
               urcrnrlat = 90,
               llcrnrlon = -180,
               urcrnrlon = 180,
               resolution = 'c')
    
    me.drawcoastlines()
    
    me.drawparallels(np.arange(-90,90,10),labels=[True,False,False,False])
    me.drawmeridians(np.arange(-180,180,30),labels=[0,0,0,1])
    
    #Current position is highlighted
    longitude = 16.509722*W
    latitude = 28.309722*N
    
    me.scatter(longitude,latitude,latlon=True, s=100, c='blue', marker='*', alpha=1, linewidth=1, zorder=1)
    
    me.scatter(long_list[0], lat_list[0],latlon=True, s=50, c='red', marker='.', alpha=1, linewidth=1, zorder=1)
    
########################### EARTH MAP LEAVING TRACE ###########################

if 'Earth map (trace)' in calc_list:

    fig_t, ax_t = plt.subplots(figsize = (12,9)) 
    
    me_t = Basemap(projection='mill',
               llcrnrlat = -90,
               urcrnrlat = 90,
               llcrnrlon = -180,
               urcrnrlon = 180,
               resolution = 'c')
    
    me_t.drawcoastlines()
    
    me_t.drawparallels(np.arange(-90,90,10),labels=[True,False,False,False])
    me_t.drawmeridians(np.arange(-180,180,30),labels=[0,0,0,1])
    
    #Current position is highlighted
    longitude = 16.509722*W
    latitude = 28.309722*N
    
    me_t.scatter(longitude,latitude,latlon=True, s=100, c='blue', marker='*', alpha=1, linewidth=1, zorder=1)
    
    for p in range(len(list_dates[0:-1])):
        
        for i in range(len(sat_sel)): 
            
            if long_suplist[p] != []:
                                                
                me_t.scatter(long_suplist[p][i], lat_suplist[p][i],latlon=True, s=5, c='red', marker='.', alpha=1, linewidth=1, zorder=1)


###################### RISING AND SETTING OF SATELLITES #######################
                
if 'Rise&Set' in calc_list:
    
    satpred.rise_set(filename=filename, satellites = satellites, sat_sel = sat_sel,\
             start = start, end = end, altitude_sat = 35.)

