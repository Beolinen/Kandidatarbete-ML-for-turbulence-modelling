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

def one_method_to_do_it_all(path_tec:str):#, path_xc_yc:str,path_mesh:str,path_k_eps_rans:str):
    
    x,y,p,u,v,uu,vv,ww,uv,eps_DNS,k_DNS,ni,nj = dat_to_variable_arrays(path_tec)

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