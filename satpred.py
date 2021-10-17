#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 15:56:05 2021

@author: Amanda Aguiar

This is a code with the functions that are needed to predict the orbits of 
satellites
"""
#The necessary modules are imported
import skyfield
import numpy as np
import matplotlib.pyplot as plt
from skyfield.api import load, wgs84
from skyfield.api import EarthSatellite
from skyfield.api import N, S, E, W 
import sys 
import warnings
from spacetrack import SpaceTrackClient
import datetime as DT
from datetime import datetime
import pandas as pd
import dateutil.relativedelta
from skyfield.framelib import galactic_frame
import stcache
from astropy import units as u
from astropy.coordinates import SkyCoord
import os

###############################################################################
################################ TIME OBJECTS #################################
###############################################################################

def time_sel(time = 'now'):
    
    ############################## INTPUT VALUES ##############################
    
    #time: #Instant for which we want to create the time objects in UTC time
               #SUPPORTED INPUT VALUES: The following format must be used: 
               #either 'YYYY-MM-DD HH:MM:SS' or simply 'YYYY-MM-DD'. In the 
               #latter case, the 00:00:00 is the selected hour of the day. If
               #We write 'now', the current instant of our computer is selected
                   #DEFAULT VALUE = 'now': The current date of our computer is 
                   #selected
      
    ############################## OUTPUT VALUES ##############################             
    
    #t:         #skyfield.timelib.Time object of the selected instant
    
    #date:      #datetime.datetime object of the selected instant
    
    #date_list: #List chosen with the year, month, day, hour, minute and second
                    
    ############################### DEFINITION ################################
    
    #If the current time is selected, the current instant is selected
    if time == 'now':                                                                                                                                                                   
        time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    #In every case we determine the year, month and day of the selected date    
    year, month, day = int(time[0:4]), int(time[5:7]), int(time[8:10])
    
    #If we include the hour, minute and second values, they are taken into
    #account. If not, we choose the first second of the selected day
    if len(time) > 10:
        hour, minute, second = int(time[11:13]), int(time[14:16]), int(time[17:20])
        
    else:
        hour, minute, second = 0, 0, 0
        
    #Two types of time objects that correspond to the same date are created 
    t = ts.utc(year, month, day, hour, minute, second) #skyfield.timelib.Time
    date = datetime(year, month, day, hour, minute, second) #datetime.datetime
    
    #A list with the year, month, day, hour, minute and second is also extracted
    date_list = [year, month, day, hour, minute, second]
    
    return t, date, date_list

###############################################################################
############################# SATELLITE SELECTION #############################
###############################################################################

def selection(filename, mode_sel = 'Group', sat_sel = 'On orbit',\
               start = '2012-11-01', end = 'now'):    
    
    ############################## INTPUT VALUES ##############################
    
    #filename:  #Name of the file that contains the basic satellite information
                #we need
                    #SUPPORTED INPUT VALUES: Name of the file with the complete
                    #directory                                              
    
    #mode_sel:  #Way in which we select the objects we want to study
                    #SUPPORTED INPUT VALUES: 'Group', 'Name', 'Company'
                        #DEFAULT VALUE = 'Group': A certain group of objects is 
                        #selected
                            #WARNING: The company names have been obtained just
                            #by searching for the objects names with the same
                            #"prefix" in their names, so this is not an 
                            #entirely right criteria, though it is useful to
                            #select the sallites that belong to big companies
                            #such as STARLINK or ONEWEB
                            #---------- THIS IS A WORK IN PROGRESS -----------#
    
    #sat_sel:   #The satellite or  group of objects we want to study
                    #SUPPORTED INPUT VALUES: If mode_sel = 'Group':
                    ##'On orbit': All on-orbit objects between the chosen dates
                    #are selected
                    ##'On orbit satellites': All satellites that were on orbit  
                    #between the chosen dates are selected
                    ##'On orbit STARLINK satellites': All on orbit STARLINK 
                    #satellites between the chosen dates are selected
                    #If model_sel = 'Name' or model_sel = 'Company', we must
                    #write the names or NORAD IDs and companies respectively inside 
                    #a list and as string objects with all capital letters
                        #DEFAULT VALUE = 'On orbit': The objects that were 
                        #on orbit at the selected time are considered
                            #WARNING: It is better to use the ID of the 
                            #satellites rather than the name of the 
                            #satellite in order to avoid problems due to the 
                            #different format and spelling
                
    #start:     #Starting date of the date range in which we want to make the 
                #study. This parameter is only relevant if the chosen selected
                #mode corresponds to 'Group' and we want to select the on orbit
                #satellites
                    #SUPPORTED INPUT VALUES: The following format must be used: 
                    #either 'YYYY-MM-DD HH:MM:SS' or simply 'YYYY-MM-DD'. If 
                    #we write 'now', the current instant of the computer is 
                    #selected
                        #DEFAULT VALUE = '2012-11-01': The date in which 
                        #QUIJOTE started operations
                        
    #end:       #Ending date of the date range in which we want to make the 
                #study. This parameter is only relevant if the chosen selected
                #mode corresponds to 'Group'
                    #SUPPORTED INPUT VALUES: See start to learn more about this
                    #parameter
                        #DEFAULT VALUE = 'now': The current time of the
                        #computer is selected
                        
    ############################## OUTPUT VALUES ############################## 
    
    #sat_sel_arr: #Array with the IDs of the selected objects     

    ############################### DEFINITION ################################
    
    #The values of the file with the basic object informations that
    #are associated to a NORAD ID are obtained
    sat_IDs, sat_names, companies, launch, decay, comment = np.loadtxt(filename,\
                                       unpack=True, delimiter = ';', dtype=str) 
    
    #All of the resulting lists contain str parameters
    #sat_IDs:   #List of objects NORAD IDs 
    #sat_names: #List of objects names
    #companies: #List of prefixes in the object names that allow us to select
                #either the company or the common part of the object names
    #launch:    #List of launch dates of the objects 
    #decay:     #List of decay dates of the objects 
    #comment:   #List of comments of the nature of the objects
                  
    #The dates among which it is studied if the satellite is on orbit or not
    #are defined as skyfield.timelib.Time variables and datetime.datetime 
    #variables
    
    ts = load.timescale()
    t_start, date_start = time_sel(start)[0], time_sel(start)[1]
    t_end, date_end = time_sel(end)[0], time_sel(end)[1]     
    
    #If the end date is previous to the start date, the code will not run
    if date_start > date_end:
        sys.exit('The starting time must be previous to the ending time')
        if not sys.warnoptions:
            warnings.simplefilter("ignore")
            
    #An empty array is created in order to save the names of the objects
    #we want to study
    sat_sel_arr = np.array([])
            
    ######Selection by group
    if mode_sel == 'Group':
        
        #All basic satellite data is check in order to select the objects we 
        #want
        #If the object has been on orbit at any point between the selected 
        #dates, it is chosen 
        for j in range(len(sat_IDs)):
            
            #Launch date
            t1, date1 = time_sel(launch[j])[0], time_sel(launch[j])[1]
            
            #Decay date 
            if decay[j] != 'ACTIVE':
                t2, date2 = time_sel(decay[j])[0], time_sel(decay[j])[1]
            else:
                t2, date2 = time_sel('now')[0], time_sel('now')[1]
            
            #If the object was on orbit between the two selected dates, 
            #it is selected
            if date1 <= date_end and date2 >= date_start:
                
                if sat_sel == 'On orbit':
                    
                    sat_sel_arr = np.append(sat_sel_arr, sat_IDs[j])
                
                #If the on orbit object is a satellite, it is selected
                elif sat_sel == 'On orbit satellites':
                    
                    if comment[j]=='SATELLITE':
                         sat_sel_arr = np.append(sat_sel_arr, sat_IDs[j])
                         
                #If the on orbit object is a STARLINK satellite, it is selected
                elif sat_sel == 'On orbit STARLINK satellites':
                    if companies[j]=='STARLINK' and comment[j]=='SATELLITE':
                         sat_sel_arr = np.append(sat_sel_arr, sat_IDs[j])
                          
    ######Selection by name        
    if mode_sel == 'Name':

        for k in range(len(sat_sel)):
            
            #Depending on whether an ID or a name is introduced, we select the
            #objects by their name or by their ID
            if sat_sel[0].isnumeric()==True:
                sat_sel_arr = np.append(sat_sel_arr, sat_IDs[np.where(sat_IDs==sat_sel[k])[0]]) 

            elif type(sat_sel[0]) == str and sat_sel[0].isnumeric()==False:
                sat_sel_arr = np.append(sat_sel_arr, sat_IDs[np.where(sat_names==sat_sel[k])[0]])
                
            sat_sel_arr = sat_sel_arr.astype(int)
    
    ######Selection by company            
    if mode_sel == 'Company':
        
        #The satellites with the same prefix in their name are selected
        for p in range(len(sat_IDs)):
            
            if companies[p]==sat_sel and comment[p]=='SATELLITE':
                sat_sel_arr = np.append(sat_sel_arr, sat_IDs[p]) 
                     
    
        sat_sel_arr = sat_sel_arr.astype(int)
            
    return sat_sel_arr


###############################################################################
############################### TLE SELECTION  ################################
###############################################################################

def TLE_sel(sat_sel, path_TLEs, time = 'now'):

    ############################## INTPUT VALUES ##############################
    
    #sat_sel:   #NORAD IDs of the objects we want to study
                    #SUPPORTED INPUT VALUES: array with the IDs of the 
                    #selected satellites as integers
    
    #path_TLEs: #Directory in which all the TLEs can be found
                    #SUPPORTED INPUT VALUES: str with the complete name of the
                    #directory

    #time:     #Instant for which we want to create the time objects in UTC time
                   #SUPPORTED INPUT VALUES: The following format must be used: 
                   #either 'YYYY-MM-DD HH:MM:SS' or simply 'YYYY-MM-DD'. In the 
                   #latter case, the 00:00:00 is the selected hour of the day
                   #If we write 'now', the current instant of our computer is 
                   #selected
                       #DEFAULT VALUE = 'now': The current date of our computer  
                       #is selected
                      
    ############################## OUTPUT VALUES ##############################
    
    #satellites: #List with the skyfield.sgp4lib.EarthSatellite objects we are 
                 #interested in
                          
    ############################### DEFINITION ################################   
    
    date_sel_start = time_sel(time)[1]
    
    #If the starting date corresponds to the future, the newest file we have is
    #selected
    if date_sel_start >= datetime.now() or time == 'now':
        time = datetime.now().strftime('%Y-%m-%d')
        date_sel_start = time_sel(time)[1]  
        files_path = sorted(os.listdir(path_TLEs))
        files = [files_path[-1][3:-4]]
    
    #If that is not the case, the TLE that is closest to the selected date
    #is selected     
    else:    
        #The year and month of the selected starting date are obtained      
        year_start  = time_sel(time)[2][0]
        month_start  = time_sel(time)[2][1]
        
        if year_start < 2012:
            sys.exit('There is no data available for the selected date')
        
        #We select the present, previous and next file in order to check in  
        #which one the date is closer
        if year_start == 2012 and month_start == 1:
            before = date_sel_start
            
        else:
            before = date_sel_start + dateutil.relativedelta.relativedelta(months=-1)
            
        if year_start == 2020 and month_start == 12:
            after = date_sel_start 
            
        else: 
            
            after = date_sel_start + dateutil.relativedelta.relativedelta(months=1)
           
        #We make a list of every month that takes place between the two dates that 
        #were selected
        files = (pd.date_range(before, after, freq='MS').strftime("%Y-%m").tolist())

    ts = load.timescale() #returns a 'Timescale' built using official Earth 
                          #rotation data

    #The TLE data is extracted and introduced in a single list
    TLE_data = []
    
    for p in range(len(files)):
        name_TLE =  path_TLEs + 'tle' + files[p] + '.txt'
        TLE_data+= load.tle_file(name_TLE, ts=ts)
        
    #The data is saved in an empty list of lists. Arrays cannot be used because 
    #it is possible to have a different number of files of each object
    satellites_sublist = [ [] for _ in range(len(sat_sel))]
    
    #An empty list in which the skyfield.sgp4lib.EarthSatellite objects will be
    #retrieved is created
    satellites = []
    
    #The TLEs of the satellites we are interested in are obtained
    for i in range(len(TLE_data)):
        #The ID of the satellites we are interested in are selected so that 
        #we can introduce their TLEs in the lists of lists
        name = str(TLE_data[i])
        ind1 = name.find('#') + 1
        ind2 = name.find(' epoch')
        
        for j in range(len(sat_sel)):
                        
            if name[ind1:ind2] == str(sat_sel[j]):
                satellites_sublist[j].append(TLE_data[i])
                
    #We choose the TLE whose date is closest to the one that has been selected
    for z in range(len(satellites_sublist)):
        #All dates of the object are extracted in an array as long as we have 
        #TLEs
        dates_arr = np.array([])
        
        for n in range(len(satellites_sublist[z])):
        
            #We obtain the date by using the name of the 
            #skyfield.sgp4lib.EarthSatellite object
            satellites_str = str(satellites_sublist[z][n])
            inddate1 = satellites_str.find(' epoch ')  + len(' epoch ')
            inddate2 = satellites_str.find(' UTC') 
               
            date = satellites_str[inddate1:inddate2]
            dates_arr = np.append(dates_arr, time_sel(date)[1]) 
            
        if len(satellites_sublist[z]) !=0: 
            #The difference between all dates and the selected one are 
            #calculated and the index that corresponds to the closest one
            #is determined
            difference_dates = abs(dates_arr - date_sel_start)
            minim = min(difference_dates)
            index = np.where(difference_dates==minim)[0][0]
        
            #The skyfield.sgp4lib.EarthSatellite objects are introduced in the
            #list 
            satellites.append(satellites_sublist[z][index]) 

    return satellites

def TLE_sel_stcache(identity, password, \
                    path_TLE, sat_sel, time = '2012-01-01'):

    ############################## INTPUT VALUES ##############################
    
    #identity:  #SpaceTrack username
                    #SUPPORTED INPUT VALUES: str object with the username
                    
    #password:  #SpaceTrack username
                    #SUPPORTED INPUT VALUES: str object with the password 
    
    #path_TLEs: #See TLE_sel() to learn more about this input parameter
    
    #sat_sel:   #See TLE_sel() to learn more about this input parameter
    
    #time:      #See TLE_sel() to learn more about this input parameter
    
    ############################## OUTPUT VALUES ##############################
    
    #satellites: #List with the skyfield.sgp4lib.EarthSatellite objects we are 
                 #interested in
    
    ############################### DEFINITION ################################    
    
    #The year, month, day, hour, minute and second of the selected date are 
    #obtained, as well as a datetime object
    year, month, day, hour, minute, second = time_sel(time)[2]
    date_sel = time_sel(time)[1]
    
    #The TLEs corresponding to that date are selected
    TLE = stcache.TLEClient(identity, password).get_tle_for_day(year, month, day)
    TLE = TLE.split('\r\n')
    
    #The name with which we want to save a txt with the data is chosen
    name_TLE = path_TLE + time + '.txt'

    #The txt file is created
    with open(name_TLE, "w") as outfile:
        outfile.write("\n".join(TLE)) 
    
    ts = load.timescale() #returns a 'Timescale' built using official Earth 
                          #rotation data
    
    #The TLE data is extracted
    TLE_data = load.tle_file(name_TLE, ts=ts)
     
    #The data is saved in an empty list of lists. Arrays cannot be used because 
    #it is possible to have a different number of files of each object
    satellites_sublist = [ [] for _ in range(len(sat_sel))]
    
    #An empty list in which the skyfield.sgp4lib.EarthSatellite objects will be
    #retrieved is created
    satellites = []
    
    #The TLEs of the satellites we are interested in are obtained
    for i in range(len(TLE_data)):
        #The ID of the satellites we are interested in are selected so that 
        #we can introduce their TLEs in the lists of lists
        name = str(TLE_data[i])
        ind1 = name.find('#') + 1
        ind2 = name.find(' epoch')
        
        for j in range(len(sat_sel)):
                        
            if name[ind1:ind2] == str(sat_sel[j]):
                satellites_sublist[j].append(TLE_data[i])
                
    #We choose the TLE whose date is closest to the one that has been selected
    for z in range(len(satellites_sublist)):
        #All dates of the object are extracted in an array as long as we have 
        #TLEs
        dates_arr = np.array([])
        
        for n in range(len(satellites_sublist[z])):
        
            #We obtain the date by using the name of the 
            #skyfield.sgp4lib.EarthSatellite object
            satellites_str = str(satellites_sublist[z][n])
            inddate1 = satellites_str.find(' epoch ')  + len(' epoch ')
            inddate2 = satellites_str.find(' UTC') 
               
            date = satellites_str[inddate1:inddate2]
            dates_arr = np.append(dates_arr, time_sel(date)[1]) 
            
        if len(satellites_sublist[z]) !=0: 
            #The difference between all dates and the selected one are 
            #calculated and the index that corresponds to the closest one
            #is determined
            difference_dates = abs(dates_arr - date_sel)
            minim = min(difference_dates)
            index = np.where(difference_dates==minim)[0][0]
        
            #The skyfield.sgp4lib.EarthSatellite objects are introduced in the
            #list 
            satellites.append(satellites_sublist[z][index]) 
                        
    return satellites

###############################################################################
############################## COLOR SELECTION  ###############################
###############################################################################

#Returns a function that maps each index in 0, 1, ..., n-1 to a  RGB color
#This function will allow us to associate a single color to a satellite
#when plotting its evolution over time 
def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)

###############################################################################
####################### ORBIT PARAMETERS OF SATELLITES ########################
###############################################################################

ts = load.timescale()

def parameters(filename, satellites, sat_sel, altitude = 2390., longitude = 16.509722*W,\
               latitude = 28.309722*N, observation_place = 'OT',\
               hor_axis = '0 to 360', temperature = None, pressure = None, \
               time = '2012-11-01'):
    
    ############################## INTPUT VALUES ##############################
    
    #filename:          #See selection() to learn more about this input parameter
                          
    #satellites:        #Name of the files in which the necessary TLEs are found
                            #SUPPORTED INPUT VALUES: list of str with the complete
                            #name of the .txt files in which the TLEs can be found
                            
    #sat_sel:           #See TLE_sel() to learn more about this input parameter 

    #altitude:          #Altitude of the selected location
                            #SUPPORTED INPUT VALUES: int or float number that 
                            #correspond to this parameter in m 
                                #DEFAULT VALUE = 2390.: Altitude of the Teide 
                                #Observatory (OT) in m
                        
    #longitude:         #Longitude of the selected location 
                            #SUPPORTED INPUT VALUES: int or float number that 
                            #correspond to this parameter in decimal degrees
                                #DEFAULT VALUE = 16.509722*W, which corresponds to 
                                #16º 30´ 35″ West or -16d30m35s
                        
    #latitude:          #Latitude of the selected location 
                            #SUPPORTED INPUT VALUES: int or float number that 
                            #correspond to this parameter in decimal degrees
                                #DEFAULT VALUE = 28.309722*N, which corresponds to 
                                #28º 18´ 00″ North or +28d18m0s
                          
    #observation_place: #Name of the location in which the observations are 
                        #made
                            #SUPPORTED INPUT VALUES: string with the name
                                #DEFAULT VALUE = 'OT': Teide Observatory

    #hor_axis:           #Range of the azimuth values in deg
                            #SUPPORTED INPUT VALUES: '0 to 360', '-180 to 180'
                                 #DEFAULT VALUE = '0 to 360'
                          
    #save_file:         #Whether we want to save a file with all the calculations for
                        #the selected time or not 
                            #SUPPORTED INPUT VALUES: 'Yes', 'No'
                                 #DEFAULT VALUE = 'Yes': the data is saved
                      
    #save_plot:         #Whether we want to save a plot with all the calculations for 
                         #the selected time or not 
                             #SUPPORTED INPUT VALUES: 'Yes', 'No'
                                 #DEFAULT VALUE = 'Yes': the plot is saved
                          
    #path_outputs:      #Directory in which we want to save the output files 
                             #SUPPORTED INPUT VALUES: str with the directory
                              
    #temperature:       #Current air temperature in the selected location
                             #SUPPORTED INPUT VALUES: int or float number that 
                             #correspond to this parameter in ºC or None
                                 #DEFAULT VALUE = None: This parameter changes by the
                                 #day, so it is not adequate to have a default 
                                 #value
    
    #pressure:          #Current air pressure in the selected location
                             #SUPPORTED INPUT VALUES: int or float number that 
                             #correspond to this parameter in mb or None
                                 #DEFAULT VALUE = None: This parameter changes by 
                                 #the day, so it is not adequate to have a default 
                                 #value
                          
    #sel_mode:          #See selection() to learn more about this input parameter
                        #In this case, it is only use in order to make the plot 
                        #according to the selection mode                           
                          
    #WARNING: the temperature and pressure values can usually be found at:
    #https://www.iac.es/index.php/en/observatorios-de-canarias/teide-observatory/weather  
    #This data are used in order to estimate where an object might appear in the 
    #sky after the Earth's atmosphere has refracted its image higher. If they are
    #not defined, the altitude that is obtained corresponds to the position
    #you would see with no atmosphere
    #This is just an estimation, since the effects of the atmosphere cannot be 
    #predicted to high precision
    
    ############################## OUTPUT VALUES ##############################
    
    #az_list:   #List of azimuth angles in degrees of every satellite that is  
                #considered in the same order as in the satellites input 
                #parameter
    
    #alt_list:  #List of altitude angles in degrees of every satellite that is  
                #considered in the same order as in the satellites input 
                #parameter
                
    #h_list:    #List of vertical distance above sea level in km of every 
                #satellite that is considered in the same order as in the
                #satellites input parameter
                
    #ra_list:   #List of right ascension in hours of every satellite that is 
                #considered in the same order as in the satellites input 
                #parameter
                
    #dec_list:  #List of declination in degrees of every satellite that is 
                #considered in the same order as in the satellites input 
                #parameter
                
    #l_list:    #List of galactic longitude angles in degrees of every   
                #satellite that is considered in the same order as in the  
                #satellites input parameter
    
    #b_list:    #List of galactic altitude angles in degrees of every 
                #satellite that is considered in the same order as in the 
                #satellites input parameter

    ############################### DEFINITION ################################ 
    
    ts = load.timescale()

    #The values of the file with the basic object informations are obtained
    #See selection() to learn more about this parameters  
    sat_IDs, sat_names, companies = np.loadtxt(filename, unpack=True, delimiter = ';', dtype=str, usecols = (0,1,2)) 
            
    #We create a skyfield.toposlib.GeographicPosition for the selected 
    #parameters
    observatory = wgs84.latlon(latitude, longitude, altitude)
    
    #Since we want to work with int number, we change the data type of the 
    #elements in the sat_IDs list
    sat_IDs_int = sat_IDs.astype(int)
    
    #The time objects for the calculations are determined                      
    t = time_sel(time)[0]
    
    #Empty lists to save the data are created
    alt_list = []
    az_list = []
    h_list = []
    
    l_list = []
    b_list = []
    
    lat_list = []
    long_list = []
    
    #The names and IDs are also obtained so that the satellites can be
    #identified
    sat_IDs_list = []
    sat_names_list = []
    
    #A list of colors for each satellite is created
    cmap = get_cmap(len(sat_sel))
    color_list = []

    #The position coordinates and the distance are determined    
    for j in range(len(satellites)): 
        
        satellite = satellites[j]
        string = str(satellite)
        #The index of the selected satellite in the basic information file
        #is determined
        ind_ID1 = string.find("#") + 1
        ind_ID2 = string.find(" epoch")   
        ID = int(string[ind_ID1:ind_ID2])
        ind = np.where(sat_IDs_int == ID)[0][0]       
                              
        #The topocentric values are determines
        difference = satellite - observatory
        topocentric = difference.at(t)
        
        #Different parameters are found and introduced in the empty lists
        
        #The altitude, azimuth and distance are determined
        alt, az, dist = topocentric.altaz(temperature_C = temperature, pressure_mbar = pressure)
        alt, az, dist = alt.degrees, az.degrees, dist.km
        
        #The satellite height is determined
        geocentric = satellite.at(t)
        subpoint = wgs84.subpoint(geocentric)
        h = subpoint.elevation.km
        
        #Here one could add some criteria to discard the satellites that are
        #on the deployment phase, according to their height

        #The right ascension, declination and distance are determined
        ra, dec, dist2 = topocentric.radec(epoch='date')
        ra, dec, dist2 = ra.hours, dec.degrees, dist2.km*3.24078e-17 
        
        #The latitude and longitude are determined
        lat = subpoint.latitude.degrees
        long = subpoint.longitude.degrees   
        
        #The galatic longitude and latitude are determined
        c_icrs = SkyCoord(ra=ra, dec=dec, distance = dist2*u.kpc, unit=('hour','deg',u.kpc), frame='icrs')
        c_gal = c_icrs.galactic
        
        l = c_gal.l.degree
        b = c_gal.b.degree
                 
        if hor_axis == '-180 to 180' and az > 180:
            az = az - 360
            
        if hor_axis == '-180 to 180' and l > 180:
            l = l - 360
         
        #The results are introduced in lists
        alt_list.append(alt)
        az_list.append(az)
        h_list.append(h)
        
        lat_list.append(lat)
        long_list.append(long)
        
        l_list.append(l)
        b_list.append(b)
        
        sat_IDs_list.append(sat_IDs[ind])
        sat_names_list.append(sat_names[ind])
        
        color_list.append(cmap(j))
            
        return sat_IDs_list, sat_names_list, alt_list, az_list, h_list, \
    lat_list, long_list, l_list, b_list, color_list
                  

###############################################################################
############################# DISTANCE HISTOGRAM  #############################
###############################################################################

def dist_hist(fig_hist, ax_hist, h_list, hist_saved, save_histogram = 'Yes',\
              nbins = 50, normalized = False, nticks = 30, time = '2012-11-01'):

    ############################## INTPUT VALUES ##############################
    
    #h_list:        #List of vertical distance above sea level in km of every 
                    #satellite that is considered in the same order as in the
                    #satellites input parameter  
                         #SUPPORTED INPUT VALUES: list of distances in km

    #hist_saved:     #Name with which we want to save the histogram 
                         #SUPPORTED INPUT VALUES: str character
    
    #save_histogram: #Whether we want to save a histogram of the distance of 
                     #the objects for the selected time or not 
                         #SUPPORTED INPUT VALUES: 'Yes', 'No'
                             #DEFAULT VALUE = 'Yes': the plot is saved
                          
    #nbins:          #Number of bins of the histogram
                         #SUPPORTED INPUT VALUES: int number 
                             #DEFAULT VALUE = 50: the histogram has 50 bins  
    
    #normalized:     #We decide if we want the histogram to be normalized
                         #SUPPORTED INPUT VALUES: 'True', 'False'
                             #DEFAULT VALUE = 'False': the histogram is not 
                             #normalized

    #nticks:         #Number of ticks in the horizontal axis of the histogram
                         #SUPPORTED INPUT VALUES: int number 
                             #DEFAULT VALUE = 30: the histogram has 30 ticks
                             #in the horizontal axis

    ############################## OUTPUT VALUES ##############################

    #We obtain a histogram with the chosen attributes     
                         
    ############################### DEFINITION ################################ 
    
    h_list = np.array(h_list)
    ind = np.where(h_list < 50000)[0]
    h_list = h_list[ind]

    h_list = sorted(h_list) #We make the ticks look good

    ax_hist.set_xlabel(r'Distance (km)')
    ax_hist.set_title('Satellites over distance (' +  str(time) + ')')
    ax_hist.set_xticks(np.round(np.linspace(h_list[0], h_list[-1], nticks), 2))
    ax_hist.set_xticklabels(np.round(np.linspace(h_list[0], h_list[-1], nticks),\
                                     2), rotation = 45)
    entries, bin_edges, patches = ax_hist.hist(h_list, bins=int(nbins), \
                                               density = normalized)

    if save_histogram == 'Yes':
        fig_hist.savefig(hist_saved, dpi=500, \
                         bbox_inches='tight')    
 
###############################################################################
###################### RISING AND SETTING OF SATELLITES #######################
###############################################################################
    
def rise_set(filename, satellites, sat_sel, altitude = 2390., \
             longitude = 16.509722*W,latitude = 28.309722*N, \
             start = '2012-12-01', end = '2012-12-02', altitude_sat = 35.):

    ############################## INTPUT VALUES ##############################
    
    #filename:     #See parameters() to learn more about this parameter
    
    #satellites:   #See TLE_sel() to learn more about this parameter
    
    #altitude:     #See parameters() to learn more about this parameter
    
    #longitude:    #See parameters() to learn more about this parameter
    
    #latitude:     #See parameters() to learn more about this parameter
    
    #sat_sel:      #See TLE_sel() to learn more about this parameter
    
    #start:        #See selection() to learn more about this parameter
    
    #end:          #See selection() to learn more about this parameter
    
    #altitude_sat: #Altitude in degrees in which we want to know when the 
                   #satellite rises or sets
                       #SUPPORTED INPUT VALUES: float number 
                           #DEFAULT VALUES = 35., which is the elevation at
                           #which the QUIJOTE telescope works 
  
    ############################## OUTPUT VALUES ##############################

    #The times in which the satellite rises above a certain altitude, sets 
    #below this same altitude and culminates are printed

    ############################### DEFINITION ################################ 
    
    #The values of the file with the basic object informations are obtained
    sat_IDs, sat_names, companies, launch, decay, comment = np.loadtxt(filename,\
                                       unpack=True, delimiter = ';', dtype=str)
    sat_IDs_int = sat_IDs.astype(int)
    #See selection() to learn more about this parameters              

    
    #We create a skyfield.toposlib.GeographicPosition for the selected 
    #parameters
    observatory = wgs84.latlon(latitude, longitude, altitude)
    
    #Both the starting and ending time of this study are selected
    t_start = time_sel(start)[0]
    t_end = time_sel(end)[0]

    for j in range(len(satellites)):
        
        satellite = satellites[j]
        
        string = str(satellite)
        
        #The index of the selected satellite in the basic information file
        #is determined
        ind_ID1 = string.find("#") + 1
        ind_ID2 = string.find(" epoch")   
        ID = int(string[ind_ID1:ind_ID2])
        ind = np.where(sat_IDs_int == ID)[0][0]       
        
        t, events = satellite.find_events(observatory, t_start, t_end, \
                                          altitude_degrees=altitude_sat)
        
        print('')
        
        for ti, event in zip(t, events):
            name = ('rises above '+ str(altitude_sat) +  '°', 'culminates', 'sets below '+ str(altitude_sat) +  '°')[event]
            print(ti.utc_strftime('%Y-%m-%d %H:%M:%S'), \
                  sat_names[ind] + ' (' + str(sat_IDs[ind]) + ') ' + name)
            
