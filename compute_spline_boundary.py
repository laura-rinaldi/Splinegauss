#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, splrep, splev
from scipy.interpolate import BSpline, PPoly
from scipy.interpolate import make_interp_spline

# In[9]:




#..........................................................................
# OBJECT:
#
# This subroutine, starting from the sampling values of the boundary in X
# and Y (i.e. sampling points are the rows of the matrix [X Y]) it computes
# the parametrical description of the boundary in the "x" and "y" variable,
# via many spline, with possible different order, as described in
# "spline_parms".
#
# INPUT:
#
# X,Y: sampling points (X,Y) described as column vectors.
#
# spline_parms: spline order and block (vector).
#
#            [spline_parms(i,1)=2] : piecewise linear splines.
#            [spline_parms(i,1)=4] : cubic splines
#                         (depending on "spltypestring" in input).
#            [spline_parms(i,1)=k] : in the case k is not 2 or 4 it
#                          chooses the k-th order splines (for additional
#                          help digit "help spapi" in matlab shell).
#
#            [spline_parms(:,2)]: vector of final components of a block.
#
#            example:
#
#            "spline_parms=[2 31; 4 47; 8 67]" means that from the 1st
#             vertex to the 31th vertex we have an order 2 spline (piecewise
#             linear), from the 32th vertex to the 47th we use a 4th order
#             spline (i.e. a cubic and periodic spline by default), from
#             the 48th to the 67th (and final!) we use an 8th order spline.
#
# spline_type: in case of cubic spline, it describes the type of spline
#      by string. It is used by Matlab built-in routine "csape". Available
#      choices are found at
#
#             https://www.mathworks.com/help/curvefit/csape.html
#
#      and can be
#
#        'complete'   : match endslopes (as given in VALCONDS, with
#                     default as under *default*).
#       'not-a-knot' : make spline C^3 across first and last interior
#                     break (ignoring VALCONDS if given).
#       'periodic'   : match first and second derivatives at first
#                     data point with those at last data point (ignoring
#                     VALCONDS if given).
#       'second'     : match end second derivatives (as given in
#                    VALCONDS, with default [0 0], i.e., as in variational).
#       'variational': set end second derivatives equal to zero
#                     (ignoring VALCONDS if given).
#
# OUTPUT:
#
# Sx: vector whose k-th component is a spline describing the k-th spline
#     component describing the boundary, i.e. for certain [t(k),t(k+1] we
#     have that "x(t)=s(t)" with "s=Sx(k)".
#
# Sy: vector whose k-th component is a spline describing the k-th spline
#     component describing the boundary in the variable "y", i.e. for
#     certain [t(k),t(k+1] we have that "y(t)=s(t)" with "s=Sy(k)".
#
# Important note: Sx and Sy have the same "breaks" in each component, this
# meaning that "(Sx(k)).breaks=(Sy(k)).breaks".
#..........................................................................


# ................ troubleshooting ................ 

# Detting default as periodic cubic spline in case "spline_parms" is not
# declared.
                      
                      

def compute_spline_boundary(X,Y,spline_parms,spline_type, *args, **kwargs):
    if not spline_parms.any():
        spline_parms=np.array([[4 , len(X)]]); 
    #end

    if np.shape(spline_parms) == 0:
        spline_parms=np.array([[4 , len(X)]]); 
    #end

    if not spline_type:
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
    
    #..........................................................................
    # Object:
    #
    # Compute parametric spline relavant parameters "ppx", "ppy" so that a
    # point at the boundary of the  domain has coordinates (x(s),y(s))
    #
    # Input:
    # s: parameter data.
    # x: determine spline x(s) interpolating (s,x)
    # y: determine spline y(s) interpolating (s,y)
    # spline_order: spline order (i.e. degree + 1)
    # spline_type: string with the spline type i.e.
    #             'complete'   : match endslopes (as given in VALCONDS, with
    #                     default as under *default*).
    #             'not-a-knot' : make spline C^3 across first and last interior
    #                     break (ignoring VALCONDS if given).
    #             'periodic'   : match first and second derivatives at first
    #                     data point with those at last data point (ignoring
    #                     VALCONDS if given).
    #             'second'     : match end second derivatives (as given in
    #                    VALCONDS, with default [0 0], i.e., as in variational).
    #             'variational': set end second derivatives equal to zero
    #                     (ignoring VALCONDS if given).
    #     If "spline_type" is not declared or equal to "[]", we use 'periodic'.
    #
    # Output:
    # ppx: spline x(t) data
    # ppy: spline y(t) data
    #..........................................................................

    # ................ troubleshooting ................ 

    # Detting default as periodic cubic spline in case "spline_parms" is not
    # declared.

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
                print(s,x,y)
                ppx = make_interp_spline(s, x, k=spline_order - 1)
                ppy = make_interp_spline(s, y, k=spline_order - 1)
                #ppy= splrep(s,y, k=spline_order-1)
        #end


    

    if isinstance(ppx, BSpline):
        ppx = PPoly.from_spline(ppx, extrapolate=True)
    if isinstance(ppy, BSpline):
        ppy = PPoly.from_spline(ppy, extrapolate=True)
    #end

    ppx = clean_ppoly(ppx)
    ppy = clean_ppoly(ppy)
    return ppx, ppy


def clean_ppoly(pp):
    coeffs = pp.c
    x = pp.x

    # Trova segmenti con intervallo positivo (no duplicati nei nodi)
    mask = np.diff(x) > 1e-12
    new_x = x[:-1][mask]
    new_x = np.append(new_x, x[-1])  # reinserisce ultimo nodo

    # Applica stesso filtro ai coefficienti
    new_c = coeffs[:, mask]

    # Ricrea un nuovo oggetto PPoly
    clean_pp = PPoly(c=new_c, x=new_x, extrapolate=pp.extrapolate)
    return clean_pp