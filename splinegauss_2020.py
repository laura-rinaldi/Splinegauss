#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, splrep, splev
from scipy.interpolate import BSpline, PPoly
import scipy.special
from scipy.special import roots_legendre
from scipy.special import gamma, gammaln


def splinegauss_2020(ade,Sx,Sy):

   
    [Sx_out,Sy_out,rot_matrix,xi]=domain_rotation(Sx,Sy);
    
    # Compute cubature formula over "D", having degree of precision "ade".

    XYW_D = splinegauss_2020_basic(ade,Sx_out,Sy_out,xi);

    XY_D=XYW_D[0:2]; 
    XY=np.array(XY_D).T@rot_matrix; 
 
    XYW=[XY[:,0], XY[:,1], XYW_D[2].T];


    return XYW




def splinegauss_2020_basic(ade,Sx,Sy,xi, *args, **kwargs):


    if (np.shape(xi) == 0):
        xi=0; 
    #end
    # .............. formula computation ..............

    XYW=[];

    # Number of spline curvilinear sides
    L=len(Sx);

    
    XYW_X=[];

    for i in range(L):
        # Determine curvilinear sides
        SXL = Sx[i]  
        SYL = Sy[i]

        # The degree of precision in x and y directions
        ade_x = ade

        # p_i = spline degree (order - 1)
        p_i = 4 - 1
        ade_y = (ade + 2) * p_i - 1

        # Generate tensorial quadrature rule on the reference square
        XYW_square, XYW_X = tensorial_cubature_square(ade_x, ade_y, XYW_X)

        # Map the quadrature rule to the curvilinear patch
        XYW_L = rule_map(XYW_square, SXL, SYL, xi)

        # Append results
        XYW.extend(XYW_L)

    return XYW



def tensorial_cubature_square(ade_x,ade_y,XW_X, *args, **kwargs):

   
    if XW_X is None:
        XW_X = np.array([]).reshape(0, 2)

    Nx = int(np.ceil((ade_x + 1) / 2))

    # Compute rule in x direction if not provided
    if np.shape(XW_X)[0] == 0:
        ab = r_jacobi(Nx, 0, 0)
        XW_X = gauss(Nx, ab)
        Xx = XW_X[ 0]
        Wx = XW_X[ 1]
    else:
        Xx = XW_X[ 0]
        Wx = XW_X[ 1]

    # Rule in y direction
    Ny = int(np.ceil((ade_y + 1) / 2))
    if Nx == Ny:
        XW_Y = XW_X
    else:
        ab = r_jacobi(Ny, 0, 0)
        XW_Y = gauss(Ny, ab)

    Xy = XW_Y[ 0]
    Wy = XW_Y[ 1]

    # Tensorial rule assembly
    X, Y = np.meshgrid(Xx, Xy)
    X = X.T.flatten()
    Y = Y.T.flatten()


    WWx, WWy = np.meshgrid(Wx, Wy)
    W = (WWx * WWy).T.flatten()

    XYW = np.column_stack((X, Y, W))
    return XYW, XW_X



def gauss(N,ab):
    
    N0=np.shape(ab)[0]; 
    if N0<N:
           raise ValueError('input array ab too short')
       #end
    N=int(N)
    J=np.zeros((N,N));
    for n in range(N):
           J[n,n]=ab[n,0]; 
       #end
    for n in range(1, N):
           J[n,n-1]=np.sqrt(ab[n,1]);
           J[n-1,n]=J[n,n-1];
       #end
    D, V = np.linalg.eig(J)  # Compute eigenvalues (D) and eigenvectors (V)

    # Sort eigenvalues and rearrange eigenvectors accordingly
    I = np.argsort(D)        # Get the sorted indices of eigenvalues
    D = D[I]                 # Reorder eigenvalues
    V = V[:, I]              # Reorder eigenvectors
    xw=[D , ab[0,1]*np.transpose(V[0,:])**2];

    return xw



# ...................... ORTH. POLY. RECURSION  ......................... 

def r_jacobi(N,a,b):
    nu = (b - a) / (a + b + 2)

    # Compute mu based on the given conditions
    if a + b + 2 > 128:
        mu = np.exp((a + b + 1) * np.log(2) + (gammaln(a + 1) + gammaln(b + 1) - gammaln(a + b + 2)))
    else:
        mu = 2**(a + b + 1) * (gamma(a + 1) * gamma(b + 1) / gamma(a + b + 2))

    # Handle N=1 case
    if N == 1:
        return np.array([[nu, mu]])

    # Compute coefficients
    N -= 1
    N=int(N)
    n = np.arange(1, N + 1)
    nab = 2 * n + a + b
  
    A = np.hstack([[nu], (b**2 - a**2) * np.ones(N) / (nab * (nab + 2))])

    # Compute B coefficients
    nab = nab[1:] 
    B1 = 4 * (a + 1) * (b + 1) / ((a + b + 2)**2 * (a + b + 3))
    B = 4 * (n[1:] + a) * (n[1:] + b) * n[1:] * (n[1:] + a + b) / (nab**2 * (nab + 1) * (nab - 1))

    # Construct final coefficient matrix
    ab = np.column_stack([A, np.hstack([[mu], B1, B])])
    return ab






def rule_map(XYW,S1,S2,xi):

   
    tau=XYW[:,0]; 
    u=XYW[:,1]; 
    w=XYW[:,2];
    

   
    [breaks,coefs,l,k] = [S2.x.T, S2.c.T, np.shape(S2.x)[0] - 1, np.shape(S2.c)[0] ];

    d= 1  
  
    replicated = np.tile(np.arange(k-1, 0, -1), (d * l, 1))

    # Now multiply with the corresponding portion of coefs
    result = replicated * coefs[:, :k-1]
    # Now, create the PPoly object S2_prime from breaks and coefs

    S2_prime = PPoly(result.T,breaks);


    t=S1.x.T; # row vector of spline breaks
    M=len(u);
    L=len(t)-1;

    delta_t=np.diff(t); # row vector
    delta_t_rep=np.tile(delta_t,(M,1)).flatten();
   
    aver_t = (t[:-1] + t[1:]) / 2 

    
    aver_t_rep = np.tile(aver_t, (M, 1)).T.flatten()

    tau_rep = np.tile(tau, (L, 1)).flatten()
   
    u_rep=np.tile(u,(L,1)).flatten();
    w_rep=np.tile(w,(L,1)).flatten();



    # ........ Computing cubature nodes and weights ........

    qij_u=(delta_t_rep/2)*u_rep.T+aver_t_rep;

    S1_qij_u=S1(qij_u);
    term1=delta_t_rep/4;

    term2=S1_qij_u-xi;
    term3=S2_prime(qij_u);

    
    W=term1*term2*term3*w_rep;

    X=(term2/2)*tau_rep+(S1_qij_u+xi)/2;
    Y=S2(qij_u);

    XYW_domain=np.array([X, Y, W]);
        
    return XYW_domain



def domain_rotation(Sx_in,Sy_in):


    control_pts=[];
    for k in range(len(Sx_in)):


        SxL=Sx_in[k]; 
        SyL=Sy_in[k];
        t= SxL.x;
        Xt=SxL(t);
        Yt=SyL(t);
        control_pts.append(np.column_stack((Xt, Yt)))
    control_pts = np.vstack(control_pts)    
    #end


    distances = points2distances(control_pts);
    
    max_distance = np.max(np.max(distances));
    
    
    [i1,i2]=np.where(distances == max_distance);
    i1=i1[0]; 
    i2=i2[0]; 

    vertex_1=control_pts[i1,:];
    vertex_2=control_pts[i2,:];
    direction_axis=(vertex_2-vertex_1)/max_distance;

    rot_angle_x=np.arccos(direction_axis[0]);
    rot_angle_y=np.arccos(direction_axis[1]);

    if rot_angle_y <= np.pi/2:
        if rot_angle_x <= np.pi/2:
            rot_angle=-rot_angle_y;
        else:
            rot_angle=rot_angle_y;
        #end
    else:
        if rot_angle_x <= np.pi/2:
            rot_angle=np.pi-rot_angle_y;
        else:
            rot_angle=rot_angle_y;
        #end
    #end


    # CLOCKWISE ROTATION.
    rot_matrix=np.array([[np.cos(rot_angle) ,np.sin(rot_angle)],[ -np.sin(rot_angle), np.cos(rot_angle)]]);

    number_sides=np.shape(control_pts)[0]-1;

    axis_abscissa=rot_matrix@np.transpose(vertex_1); 
    xi=axis_abscissa[0];


    M=len(Sx_in);
    Sx_out=[]
    Sy_out=[]

    for k in range(M):

        SxL_in=Sx_in[k];
        SyL_in=Sy_in[k];
        
        t_vals = SxL_in.x  

        
        
        x_vals = SxL_in(t_vals)
        y_vals = SyL_in(t_vals)
        
        SxL_out_A = rot_matrix[0][0] * SxL_in.c
        
        SxL_out_B= rot_matrix[0][1] * SyL_in.c
        
        SxL_out= SxL_out_B + SxL_out_A
        

        SyL_out_A= rot_matrix[1][0] * SxL_in.c
        SyL_out_B= rot_matrix[1][1] * SyL_in.c
        
        SyL_out= SyL_out_B + SyL_out_A


        Sx_o=PPoly(SxL_out, t_vals)
        Sy_o=PPoly(SyL_out, t_vals)
        
        Sx_out.append(Sx_o)
        Sy_out.append(Sy_o)
        

    #end
    return [Sx_out,Sy_out,rot_matrix,xi] 



#--------------------------------------------------------------------------
# points2distances.
#--------------------------------------------------------------------------

def points2distances(points):
   
    # Get dimensions.
    [numpoints,dim]=np.shape(points);

    # All inner products between points.
    distances=points@np.transpose(points);

    # Vector of squares of norms of points.
    lsq=np.diag(distances);

    # Distance matrix.
    distances=np.sqrt(np.tile(lsq, (numpoints,1))+np.transpose(np.tile(lsq, (numpoints,1)))-2*distances)
    return distances










