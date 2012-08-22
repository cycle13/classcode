# -*- coding: utf-8 -*-
# <nbformat>2</nbformat>

# <codecell>

#!/usr/bin/python

# imports
import netCDF4

# Add a couple of user defined functions
import os, datetime, pylab
from weather_modules import *
from utilities_modules import *

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# <codecell>

# Set user options
fpath = '/home/scavallo/data/gfs_4_20120822_0000_000.nc'
level_option = 50000 # Pascals; set to -1 for sea level pressure
date_string = '2012082200' # yyyymmddhh

# <codecell>

# Open the netcdf file and read select variables
f = netCDF4.Dataset(fpath,'r')
lons = f.variables['lon_0'][:]
lats = f.variables['lat_0'][::-1] # Read in reverse direction
levs = f.variables['lv_ISBL0'][:]
if ( level_option == -1 ) :
   plotvar  = f.variables['PRMSL_P0_L101_GLL0'][::-1,:]/100
else:
    levelindex = pylab.find(levs==level_option)
    plotvar = f.variables['HGT_P0_L100_GLL0'][levelindex,::-1,:].squeeze() # Reverse latitude dimension
f.close            

# <codecell>

lonin = lons
plotvar, lons = addcyclic(plotvar, lonin)

# Refresh the dimensions
[X,Y] = np.meshgrid(lons,lats)
[ny,nx] = np.shape(X)

levelh = level_option / 100 # Convert level to hPa for title

yyyy = date_string[0:4]
mm = date_string[4:6]
dd = date_string[6:8]
hh = date_string[8:10]

if (level_option == -1 ):
   titletext = 'Sea level pressure valid  ' +mm +'/' +dd +'/' +yyyy +' at ' +'UTC'
else:
   titletext = str(levelh)+' hPa geopotential heights valid ' +mm +'/' +dd +'/' +yyyy +' at ' +'UTC'
print titletext

# <codecell>

# Set global figure properties
golden = (pylab.sqrt(5)+1.)/2.
figprops = dict(figsize=(8., 16./golden),dpi=128)
adjustprops = dict(left=0.15, bottom=0.1, right=0.90, top = 0.93, wspace=0.2, hspace=0.2)

# <codecell>
# Setting contour interval done here
if (level_option == -1 ):
    cint = 4
    cbar_min = 1012-20*cint
    cbar_max = 1012+20*cint
else:
    base_cntr = 5580 # a contour close to standard atmospheric value
    cint = 60 # contour interval
    cbar_min = base_cntr-20*cint
    cbar_max = base_cntr+20*cint
    
cflevs = np.arange(cbar_min,cbar_max+1,cint)

fig = plt.figure(**figprops)   # New figure   
ax1 = fig.add_axes([0.1,0.1,0.8,0.8])

map = Basemap(projection='ortho', lat_0 = 50, lon_0 = 260,
               resolution = 'l', area_thresh = 1000.,ax=ax1)
# draw countries, states, and differentiate land from water areas.
map.drawcountries()
map.drawstates()
map.drawlsmask(land_color='0.7',ocean_color='white',lakes=True)

# draw lat/lon grid lines every 30 degrees.
map.drawmeridians(np.arange(0, 360, 30))
map.drawparallels(np.arange(-90, 90, 30))


x, y = map(X, Y)
CS = map.contour(x,y,plotvar,cflevs,colors='k',linewidths=1.5)
plt.clabel(CS, inline=1, fontsize=10)
ax1.set_title(titletext)

figname = "example"
save_name=figname+".png"
fig.savefig(save_name)

plt.show()

