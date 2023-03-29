import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR  # for building SVR model
import sklearn.metrics as sm
from gradients import compute_face_phi, dphidx, dphidy, init
import time
import sys

# give this one a new name later. the idea is to take the paths and use the 
# (hopefully structured the exact same way) data and return a pandas datafrasme

def one_method_to_do_it_all(path_tec:str, path_mesh:str):#, path_xc_yc:str,path_mesh:str,path_k_eps_rans:str):
    
    x,y,p,u,v,uu,vv,ww,uv,eps_DNS,k_DNS,ni,nj = dat_to_variable_arrays(path_tec)

    # set Neumann of p at upper and lower boundaries
    p[:, 1]  = p[:, 2]
    p[:, -1] = p[:, -1 - 1]

    # set periodic b.c on west boundary
    # SHOULD WE USE THIS?
    u[0, :]  = u[-1, :]
    v[0, :]  = v[-1, :]
    p[0, :]  = p[-1, :]
    uu[0, :] = uu[-1, :]

    # x and y are fo the cell centers. The dphidx_dy routine needs the face coordinate, xf2d, yf2d
    # load them
    xc_yc = np.loadtxt(path_mesh)
    xf = np.reshape(xc_yc[:, 0], (nj - 1, ni - 1)).transpose()
    yf = np.reshape(xc_yc[:, 1], (nj - 1, ni - 1)).transpose()

    # compute cell centers
    xp = 0.25 * (x[0:-1, 0:-1] + x[0:-1, 1:] + x[1:, 0:-1] + x[1:, 1:])
    yp = 0.25 * (y[0:-1, 0:-1] + y[0:-1, 1:] + y[1:, 0:-1] + y[1:, 1:])

    # delete last row/col
    x = np.delete(np.delete(x, -1, 1), -1, 0)
    y = np.delete(np.delete(y, -1, 1), -1, 0)
    xp = np.delete(np.delete(xp, -1, 1), -1, 0)
    yp = np.delete(np.delete(yp, -1, 1), -1, 0)

    # compute geometric quantities
    areaw, areawx, areawy, areas, areasx, areasy, vol, fx, fy = init(x, y, xp, yp)


    # delete first/last row/col
    u       = np.delete(np.delete(u, [0,-1], 1), [0,-1], 0)
    v       = np.delete(np.delete(v, [0,-1], 1), [0,-1], 0)
    p       = np.delete(np.delete(p, [0,-1], 1), [0,-1], 0)
    k_DNS   = np.delete(np.delete(k_DNS, [0,-1], 1), [0,-1], 0)
    uu      = np.delete(np.delete(uu, [0,-1], 1), [0,-1], 0)
    vv      = np.delete(np.delete(vv, [0,-1], 1), [0,-1], 0)
    ww      = np.delete(np.delete(ww, [0,-1], 1), [0,-1], 0)
    uv      = np.delete(np.delete(uv, [0,-1], 1), [0,-1], 0)
    eps_DNS = np.delete(np.delete(eps_DNS, [0,-1], 1), [0,-1], 0)

    ni -= 2
    nj -= 2

    # eps at last cell upper cell wrong. fix it.
    eps_DNS[:, -1] = eps_DNS[:, -2]

    # compute face value of U and V
    u_face_w, u_face_s = compute_face_phi(u, fx, fy, ni, nj)  # Error with dimensions for u2d vs fx,fy
    v_face_w, v_face_s = compute_face_phi(v, fx, fy, ni, nj)

    # x derivatives
    dudx = dphidx(u_face_w, u_face_s, areawx, areasx, vol)
    dvdx = dphidx(v_face_w, v_face_s, areawx, areasx, vol)

    # y derivatives
    dudy = dphidy(u_face_w, u_face_s, areawy, areasy, vol)
    dvdy = dphidy(v_face_w, v_face_s, areawy, areasy, vol)

    omega = eps_DNS / k_DNS / 0.09



    return 0



def dat_to_variable_arrays(path:str):

    tec = np.genfromtxt(path, dtype=None, comments="%").transpose() #turn every column in the .dat file to a row in tec
    k_DNS = 0.5 * (tec[5] + tec[6] + tec[7])                        #Turbulent kinetic energy (?)

    nu,ni,nj = get_ni_nj(path)                      #get the dimentions on of the data, based on path name

    k_DNS = np.reshape(k_DNS, (nj, ni)).transpose() #k_DNS has the shape [ni,nj]
    tec = tec.reshape(10,nj,ni)                     #tec is reshaped so every variable has the shape [nj,ni]
    tec = np.transpose(tec,axes=[0,2,1])            #tec is transposed so that every variable has the shape [ni,nj]

    x       = tec[0]
    y       = tec[1]
    p       = tec[2]
    u       = tec[3]
    v       = tec[4]
    uu      = tec[5]
    vv      = tec[6]
    ww      = tec[7]
    uv      = tec[8]
    eps_DNS = tec[9]

    return x,y,p,u,v,uu,vv,ww,uv,eps_DNS,k_DNS,ni,nj

def get_ni_nj(path:str) -> tuple[float,int,int]:
    
    if path == "small_wave/tec.dat" or path == "large_wave/tec_large.dat":
        return 0.0001,170,194
    if path == "one_hill/tec_OneHill.dat":
        return (1./10595.),162,162
    if path == "two_hills/tecTwoHills.dat":
        return (1./10595.),402,162

    return 0,0,0