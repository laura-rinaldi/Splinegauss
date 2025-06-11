#!/usr/bin/env python
# coding: utf-8



import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, splrep, splev
from scipy.interpolate import BSpline, PPoly
                     
                      

def compute_spline_boundary(X,Y,spline_parms,spline_type, *args, **kwargs):
    if len(args) + len(kwargs) < 3:
        spline_parms=np.array([[4 , len(X)]]); 
    #end

    if np.shape(spline_parms)[0] == 0:
        spline_parms=np.array([[4 , len(X)]]); 
    #end

    if len(args) + len(kwargs) < 4:
        spline_type='periodic';
    #end

#................ splines definition ................ 

    L = np.shape(spline_parms)[0]

    Sx=[]
    Sy=[]
    for i in range(L): # define spline

        # first and last index to consider
        if i == 0:
            imin=0;
        else:
            imin = spline_parms[i-1,1]
        #end
        
        imax=spline_parms[i,1]

        tL=np.arange(imin,imax)
        xL=X[imin:imax] 
        yL=Y[imin:imax]

        SxL,SyL=compute_parametric_spline(tL,xL,yL,spline_parms[i,0],spline_type)

        Sx.append(SxL)
        Sy.append(SyL)

        
        
        return Sx, Sy
#end




#--------------------------------------------------------------------------
# compute_parametric_spline
#--------------------------------------------------------------------------

def compute_parametric_spline(s,x,y,spline_order, spline_type, *args, **kwargs):
    
   
    if len(args) + len(kwargs) < 5:
        spline_type = 'periodic'
    #end

    if len(spline_type) == 0:
        spline_type = 'periodic'
    #end


    # ................ splines definition ................
     


    if spline_order==4:
                ppx=CubicSpline(s,x, bc_type=spline_type)
                ppy=CubicSpline(s,y, bc_type=spline_type)
                
    else:
        
                ppx= splrep(s,x,spline_order)
                ppy= splrep(s,y,spline_order)
        #end


    

    if isinstance(ppx, BSpline):
        ppx = PPoly.from_spline(ppx)
        ppy = PPoly.from_spline(ppy)
    #end

    return ppx, ppy
