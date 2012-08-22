#!/usr/bin/python

import numpy as np
from mstats import *
Cp = 1004.5;
Cv = 717.5;
Rd = 287.04; 
Rv = 461.6;
RvRd = Rv / Rd;
g = 9.81; 
L = 2.50e6; 
Talt = 288.1500; 
Tfrez = 273.1500; 
To = 300; 
Po = 101325;
Pr = 1000.;
lapsesta = 6.5 / 1000;
kappa = Rd / Cp;
epsil = Rd/Rv;
pi = 3.14159265;
pid = pi/180;
R_earth = 6371200; 
omeg_e = (2*pi) / (24*3600);
eo = 6.11;
missval = -9999;
eps = 2.2204e-16

def temp_to_theta(temp, pres):
    ''' Compute potential temperature '''
    ''' '''
    ''' theta: Input potential temperature (K) '''
    ''' pres:  Input pressure (Pa)'''
    ''' temp:  Output temperature (K)'''
    return temp * (100000. / pres) ** 0.286

def theta_to_temp(theta, pres):
    ''' Compute temperature '''
    ''' '''
    ''' temp:  Input temperature (K)''' 
    ''' pres:  Input pressure (Pa)'''
    ''' theta: Output potential temperature (K)'''
    return theta * (pres / 100000.) ** 0.286

def td_to_mixrat(tdew, pres):
    ''' Convert from dewpoint temperature to water vapor mixing ratio '''
    ''' '''
    ''' tdew:   Input dewpoint temperature (K)'''
    ''' pres:   Input pressure (Pa)'''
    ''' mixrat: Output water vapor mixing ratio (kg/kg)'''    
    pres = pres/100
    mixrat = eo / (pres * RvRd)  * np.exp( (L/Rv)*((1/Tfrez) - (1 / tdew) ) )     
    return mixrat
    
def mixrat_to_td(qvap, pres):
    ''' Convert from water vapor mixing ratio to dewpoint temperature '''
    ''' '''
    ''' qvap: Input water vapor mixing ratio (kg/kg)'''
    ''' pres: Input pressure (Pa)'''
    ''' tdew: Output dewpoint temperature (K)'''    
    pres = pres/100
    evap = qvap * pres * RvRd;
    tdew = 1/((1/Tfrez) - (Rv/L)*np.log(evap/eo))
    return tdew

def claus_clap(temp):
    ''' Compute saturation vapor pressure '''
    ''' '''
    ''' temp: Input temperature (K)  '''
    ''' esat: Output satuation vapor pressure (Pa)'''    
    esat = (eo * np.exp( (L / Rv) * ( 1/Tfrez - 1/temp) ) ) * 100    
    return esat

def thetae(thta, temp, esat):
    ''' Compute equivalent potential temperature '''
    ''' '''
    ''' thta:   Input potential temperature (K) '''
    ''' temp:   Input temperature (K) '''
    ''' esat:   Input saturation vapor pressure (Pa)'''
    ''' thetae: Output equivalent potential temperature (K)'''
    thout = thta * np.exp( (L * esat) / (Cp * temp) )
    return thout

def calc_gradient(fldin, dx, dy, dz):
    '''
    Computes the horizontal gradient of any scalar given a constant
    grid spacing in the x, y, and z directions.
    
    fldin: Input scalar
    dx: Input x grid spacing (must be single value)
    dy: Input y grid spacing (must be single value)
    dz: Input z grid spacing (must be single value)
    '''
    dfdx, dfdy, dfdz = np.gradient(fldin, dx, dy, dz)    
    return dfdx, dfdy, dfdz

def latlon_to_dlatdlon(lats,lons):
    """
    Return arrays with the spacing between latitude and longitudes

    The gradients are computed using central differences in the interior
    and first differences at the boundaries. The returned gradient hence has
    the same shape as the input array.

    Parameters
    ----------
    lats : vector of latitudes in degrees
    lons : vector of longitudes in degrees

    Returns
    -------
    dlat : array with differences in latitudes between grid points with size (lats,lons)
    dlon : array with differences in longitudes between grid points with size (lats,lons)
    
    Examples
    --------    
    dlat,dlon = latlon_to_dlatdlon(lats,lons)
        
    """
    
    nlat = len(lats)
    nlon = len(lons) 
    latarr = np.zeros((nlat,nlon))
    lonarr = np.zeros((nlat,nlon))     
    dlatarr = np.zeros((nlat,nlon))
    dlonarr = np.zeros((nlat,nlon))
    
          
    for jj in range(0,nlat):       
       for ii in range(0,nlon):          
          latarr[jj,ii] = lats[jj]	   
          lonarr[jj,ii] = lons[ii]

                     
    latrad = latarr*(pi/180)

    # use central differences on interior and first differences on endpoints

    otype = latarr.dtype.char
    if otype not in ['f', 'd', 'F', 'D']:
        otype = 'd'        

    dlats = np.zeros_like(lats).astype(otype)	
    dlats[1:-1] = (lats[2:] - lats[:-2])
    dlats[0] = (lats[1] - lats[0])
    dlats[-1] = (dlats[-2] - dlats[-1])
    
    dlons = np.zeros_like(lons).astype(otype)	
    dlons[1:-1] = (lons[2:] - lons[:-2])
    dlons[0] = (lons[1] - lons[0])
    dlons[-1] = (dlons[-2] - dlons[-1])        

    # Since we differenced in the reverse direction, change the sign
    dlats = -1*dlats    
           
    for jj in range(0,nlat):       
       for ii in range(0,nlon):          
          dlonarr[jj,ii] = dlons[ii]
	  dlatarr[jj,ii] = dlats[jj]	  
                              
    
    return dlatarr, dlonarr   
    
def gradient_cartesian(f, *varargs):
    """
    Return the gradient of an N-dimensional array on an evenly spaced grid.

    The gradient is computed using central differences in the interior
    and first differences at the boundaries. The returned gradient hence has
    the same shape as the input array.

    Parameters
    ----------
    f : N-dimensional array containing samples of a scalar function.
        If 2-D, must be ordered as f(y,x)
	If 3-D, must be ordered as f(z,y,x) or f(p,y,x)
    `*varargs` : scalars
          0, 1, or N scalars specifying the sample distances in each direction,
          that is: `dz`, `dy`, `dx`, ... The default distance is 1.
          
	  If a vector is specified as the first argument of three, then the difference
	     of this vector will be taken here.


    Returns
    -------
    g : ndarray
      N arrays of the same shape as `f` giving the derivative of `f` with
      respect to each dimension.

    Examples
    --------    
    temperature = temperature(pressure,y,x)
    levs = pressure vector
    dy = scalar or array of grid spacing in y direction
    dx = scalar or vector of grid spacing in x direction
    >>> dfdz, dfdy = gradient_evenspaced(fldin, dy, dx)  
    >>> dfdz, dfdy, dfdx = gradient_evenspaced(fldin, dz, dy, dx)  
    >>> dfdp, dfdy, dfdx = gradient_evenspaced(fldin, levs, dy, dx)  
    """    
    N = len(f.shape)  # number of dimensions        
    n = len(varargs)        
    argsin = list(varargs)
    
    if N != n:
       raise SyntaxError("dimensions of input array must match the number of remaining argumens")
    
    df = np.gradient(f)
    
    if n == 1:        
        dy = argsin[0]
	
	dfdy = df[0]
    elif n == 2:
        dy = argsin[0]
	dx = argsin[1]
        
	dfdy = df[0]
        dfdx = df[1]		
    elif n == 3:        
        levs = argsin[0]
	dy = argsin[1]
        dx = argsin[2]
        
	dfdz = df[0]
        dfdy = df[1]
        dfdx = df[2]	
    else:
        raise SyntaxError(
                "invalid number of arguments")    
               
    otype = f.dtype.char
    if otype not in ['f', 'd', 'F', 'D']:
        otype = 'd'        
    
    
    try:
       M = len(dx.shape)
    except:
       M = 1
    dyarr = np.zeros_like(f).astype(otype)
    dxarr = np.zeros_like(f).astype(otype) 
    if M == 1:  
       dyarr[:] = dy
       dxarr[:] = dx
       if N == 1:
          ny = np.shape(f)
       elif N == 2:
          ny, nx = np.shape(f)     
       else:
          nz, ny, nx = np.shape(f)
       
    else:         
       if N == 1:
          ny = np.shape(f)
	  for jj in range(0,ny):           
	     dyarr[jj,ii] = dy[jj]	
       elif N == 2:
	  ny, nx = np.shape(f)                   
	  for jj in range(0,ny):       
             for ii in range(0,nx):          
        	dyarr[jj,ii] = dy[jj]	   
		dxarr[jj,ii] = dx[ii]
       else:
	  nz, ny, nx = np.shape(f)
	  for kk in range(0,nz):
             for jj in range(0,ny):       
        	for ii in range(0,nx):          
                   dyarr[kk,jj,ii] = dy[jj]	
		   dxarr[kk,jj,ii] = dx[ii]
                         
    if n==1:       
       dfdy = dfdy/dx
       
       return dfdy
    elif n==2:                     
       dfdy = dfdy/dy
       dfdx = dfdx/dx
       
       return dfdy,dfdx
    elif n==3:            
       dfdy = dfdy/dy
       dfdx = dfdx/dx    
    
       nzz = np.shape(levs)   
       print nzz
       if not nzz:
          nzz=0
              
       if nzz>1:                       	    	     	    
	    zin = levs
	    dz = np.zeros_like(zin).astype(otype)	
            dz[1:-1] = (zin[2:] - zin[:-2])/2
            dz[0] = (zin[1] - zin[0])
            dz[-1] = (zin[-1] - zin[-2])
	    dz = dz*-1 # assume the model top is the first index and the lowest model is the last index
	    
	    dx3 = np.ones_like(f).astype(otype)      	   	    
	    for kk in range(0,nz):	       
	       dx3[kk,:,:] = dz[kk]       	                
       else:
            dx3 = np.ones_like(f).astype(otype)       
            dx3[:] = dx[0] 
	                 
       dfdz = dfdz/dx3          
       return dfdz,dfdy,dfdx    

def gradient_sphere(f, *varargs):
    """
    Return the gradient of a 2-dimensional array on a sphere given a latitude
    and longitude vector.

    The gradient is computed using central differences in the interior
    and first differences at the boundaries. The returned gradient hence has
    the same shape as the input array.

    Parameters
    ----------
    f : A 2-dimensional array containing samples of a scalar function.
    latvec: latitude vector
    lonvec: longitude vector

    Returns
    -------
    g : dfdx and dfdy arrays of the same shape as `f` giving the derivative of `f` with
        respect to each dimension.

    Examples
    --------
    temperature = temperature(pressure,latitude,longitude)
    levs = pressure vector
    lats = latitude vector
    lons = longitude vector
    >>> tempin = temperature[5,:,:]   	       
    >>> dfdlat, dfdlon = gradient_sphere(tempin, lats, lons)      
    
    >>> dfdp, dfdlat, dfdlon = gradient_sphere(temperature, levs, lats, lons)   
    
    based on gradient function from /usr/lib64/python2.6/site-packages/numpy/lib/function_base.py
    """
    
    R_earth = 6371200;              
    N = len(f.shape)  # number of dimensions        
    n = len(varargs)        
    argsin = list(varargs)
    
    if N != n:
       raise SyntaxError("dimensions of input array must match the number of remaining argumens")
    
    df = np.gradient(f)
    
    if n == 1:        
        lats = argsin[0]
	
	dfdy = df[0]
    elif n == 2:
        lats = argsin[0]
	lons = argsin[1]
        
	dfdy = df[0]
        dfdx = df[1]		
    elif n == 3:        
        levs = argsin[0]
	lats = argsin[1]
        lons = argsin[2]
        
	dfdz = df[0]
        dfdy = df[1]
        dfdx = df[2]	
    else:
        raise SyntaxError(
                "invalid number of arguments")    
               
    otype = f.dtype.char
    if otype not in ['f', 'd', 'F', 'D']:
        otype = 'd'        
    
    latarr = np.zeros_like(f).astype(otype)
    lonarr = np.zeros_like(f).astype(otype)
    if N == 1:
       nlat = np.shape(f)       
       for jj in range(0,nlat):       
          latarr[jj,ii] = lats[jj]	 
       lonarr = latarr
       lons = lats
    elif N == 2:
       nlat, nlon = np.shape(f)                   
       for jj in range(0,nlat):       
          for ii in range(0,nlon):          
             latarr[jj,ii] = lats[jj]	   
	     lonarr[jj,ii] = lons[ii]
    else:
       nz, nlat, nlon = np.shape(f)
       for kk in range(0,nz):
          for jj in range(0,nlat):       
             for ii in range(0,nlon):          
                latarr[kk,jj,ii] = lats[jj]	
		lonarr[kk,jj,ii] = lons[ii]
                     
    latrad = latarr*(pi/180)

    # use central differences on interior and first differences on endpoints

    outvals = []

    dlats = np.zeros_like(lats).astype(otype)	
    dlats[1:-1] = (lats[2:] - lats[:-2])
    dlats[0] = (lats[1] - lats[0])
    dlats[-1] = (dlats[-2] - dlats[-1])
    
    dlons = np.zeros_like(lons).astype(otype)	
    dlons[1:-1] = (lons[2:] - lons[:-2])
    dlons[0] = (lons[1] - lons[0])
    dlons[-1] = (dlons[-2] - dlons[-1])        

    # Since we differenced in the reverse direction, change the sign
    dlats = -1*dlats
    
    dlatarr = np.tile(dlats,[nlon,1])    
    dlatarr = np.reshape(dlatarr,[nlat,nlon])
       
    dlonarr = np.zeros_like(f).astype(otype)    
    if N==2:
       for jj in range(0,nlat):       
          for ii in range(0,nlon):          
             dlonarr[jj,ii] = dlons[ii]	  
    elif N==3:
       for kk in range(0,nz):
          for jj in range(0,nlat):       
             for ii in range(0,nlon):
	        dlonarr[kk,jj,ii] = dlons[ii]	          
                     
    dlatsrad = dlatarr*(pi/180)
    dlonsrad = dlonarr*(pi/180)    
    latrad = latarr*(pi/180)
        
    if n==1:
       dx1 = R_earth * dlatsrad
       dfdy = dfdy/dx1
       
       return dfdy
    elif n==2:       
       dx1 = R_earth * dlatsrad
       dx2 = R_earth * np.cos(latrad) * dlonsrad         
       
       dfdy = dfdy/dx1
       dfdx = dfdx/dx2
       
       return dfdy,dfdx
    elif n==3:            
       dx1 = R_earth * dlatsrad
       dx2 = R_earth * np.cos(latrad) * dlonsrad         
              
       dfdy = dfdy/dx1
       dfdx = dfdx/dx2    
    
       nzz = np.shape(levs)   
       print nzz
       if not nzz:
          nzz=0
              
       if nzz>1:                       	    	     	    
	    zin = levs
	    dz = np.zeros_like(zin).astype(otype)	
            dz[1:-1] = (zin[2:] - zin[:-2])/2
            dz[0] = (zin[1] - zin[0])
            dz[-1] = (zin[-1] - zin[-2])
	    dz = dz*-1 # assume the model top is the first index and the lowest model is the last index
	    
	    dx3 = np.ones_like(f).astype(otype)      	   	    
	    for kk in range(0,nz):	       
	       dx3[kk,:,:] = dz[kk]       	                
       else:
            dx3 = np.ones_like(f).astype(otype)       
            dx3[:] = dx[0] 
	                 
       dfdz = dfdz/dx3          
       return dfdz,dfdy,dfdx
       



	
def _get_gradients(u, v, dx, dy):
    #Helper function for getting convergence and vorticity from 2D arrays
    dudx, dudy = np.gradient(u, dx, dy)
    dvdx, dvdy = np.gradient(v, dx, dy)
    return dudx, dudy, dvdx, dvdy

def vertical_vorticity(u, v, dx, dy, grid_opt):
    '''
Calculate the vertical vorticity of the horizontal wind. The grid
must have a constant spacing in each direction.

u, v : 2 dimensional arrays
Arrays with the x and y components of the wind, respectively.
X must be the first dimension and y the second.

dx : scalar or array
The grid spacing in the x-direction 

dy : scalar or array
The grid spacing in the y-direction

grid_opt: 1 for cartesian grid
          2 for lat/lon grid

Returns : 2 dimensional array
The vertical vorticity
'''
    
    if grid_opt == 1:
       dudy,dudx = gradient_cartesian(u, dy, dx)      
       dvdy,dvdx = gradient_cartesian(v, dy, dx)      
    else: 
       dudy,dudx = gradient_sphere(u, dy, dx)      
       dvdy,dvdx = gradient_sphere(v, dy, dx)          
    
    return dvdx - dudy

def h_convergence(u, v, dx, dy):
    '''
Calculate the horizontal convergence of the horizontal wind. The grid
must have a constant spacing in each direction.

u, v : 2 dimensional arrays
Arrays with the x and y components of the wind, respectively.
X must be the first dimension and y the second.

dx : scalar
The grid spacing in the x-direction

dy : scalar
The grid spacing in the y-direction

Returns : 2 dimensional array
The horizontal convergence
'''
    dudx, dudy, dvdx, dvdy = _get_gradients(u, v, dx, dy)
    return dudx + dvdy
