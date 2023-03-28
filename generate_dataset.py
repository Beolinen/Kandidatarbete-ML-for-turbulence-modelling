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

def one_method_to_do_it_all(path_k_eps_rans:str, path_tec:str, path_xc_yc:str,path_mesh:str):
    
    tec = np.genfromtxt(path_tec, dtype=None, comments="%").transpose()

    x,y,p,u,v,uu,vv,ww,uv,eps_DNS = tec[0],tec[1],tec[2],tec[3],tec[4],tec[5],tec[6],tec[7],tec[8],tec[9] 
    k_DNS = 0.5 * (uu + vv + ww)  # Turbulent kinetic energy (?)
    
    ni,nj,nu = get_ni_nj(y,x)

    viscos = nu

    u2d = np.reshape(u, (nj, ni)).transpose()
    v2d = np.reshape(v, (nj, ni)).transpose()
    p2d = np.reshape(p, (nj, ni)).transpose()
    x2d = np.reshape(x, (nj, ni)).transpose()
    y2d = np.reshape(y, (nj, ni)).transpose()
    uu2d = np.reshape(uu, (nj, ni)).transpose()  # =mean{v'_1v'_1}
    uv2d = np.reshape(uv, (nj, ni)).transpose()  # =mean{v'_1v'_2}
    vv2d = np.reshape(vv, (nj, ni)).transpose()  # =mean{v'_2v'_2}
    ww2d = np.reshape(ww, (nj, ni)).transpose()
    k2d = np.reshape(k_DNS, (nj, ni)).transpose()  # =mean{0.5(v'_iv'_i)}
    eps_DNS2d = np.reshape(eps_DNS, (nj, ni)).transpose()

    # set Neumann of p at upper and lower boundaries
    p2d[:, 1] = p2d[:, 2]
    p2d[:, -1] = p2d[:, -1 - 1]

    # set periodic b.c on west boundary
    # SHOULD WE USE THIS?
    u2d[0, :] = u2d[-1, :]
    v2d[0, :] = v2d[-1, :]
    p2d[0,] = p2d[-1, :]
    uu2d[0, :] = uu2d[-1, :]

    # x and y are fo the cell centers. The dphidx_dy routine needs the face coordinate, xf2d, yf2d
    # load them
    xc_yc = np.loadtxt(path_mesh)
    xf = xc_yc[:, 0]
    yf = xc_yc[:, 1]
    xf2d = np.reshape(xf, (nj - 1, ni - 1)).transpose()
    yf2d = np.reshape(yf, (nj - 1, ni - 1)).transpose()

    xp2d = 0.25 * (x2d[0:-1, 0:-1] + x2d[0:-1, 1:] + x2d[1:, 0:-1] + x2d[1:, 1:])
    yp2d = 0.25 * (y2d[0:-1, 0:-1] + y2d[0:-1, 1:] + y2d[1:, 0:-1] + y2d[1:, 1:])

    # delete last row
    x2d = np.delete(x2d, -1, 0)
    y2d = np.delete(y2d, -1, 0)
    xp2d = np.delete(xp2d, -1, 0)
    yp2d = np.delete(yp2d, -1, 0)

    # delete last columns
    x2d = np.delete(x2d, -1, 1)
    y2d = np.delete(y2d, -1, 1)
    xp2d = np.delete(xp2d, -1, 1)
    yp2d = np.delete(yp2d, -1, 1)

    # compute geometric quantities
    areaw, areawx, areawy, areas, areasx, areasy, vol, fx, fy = init(x2d, y2d, xp2d, yp2d)

    # delete first/last row/col
    u2d = np.delete(np.delete(u2d, [0,-1], 1), [0,-1], 0)
    v2d = np.delete(np.delete(v2d, [0,-1], 1), [0,-1], 0)
    p2d = np.delete(np.delete(p2d, [0,-1], 1), -[0,-1], 0)
    k2d = np.delete(np.delete(k2d, [0,-1], 1), [0,-1], 0)
    uu2d = np.delete(np.delete(uu2d, [0,-1], 1), [0,-1], 0)
    vv2d = np.delete(np.delete(vv2d, [0,-1], 1), [0,-1], 0)
    ww2d = np.delete(np.delete(ww2d, [0,-1], 1), [0,-1], 0)
    uv2d = np.delete(np.delete(uv2d, [0,-1], 1), [0,-1], 0)
    eps_DNS2d = np.delete(np.delete(eps_DNS2d, [0,-1], 1), [0,-1], 0)

    ni -= 2
    nj -= 2

    # eps at last cell upper cell wrong. fix it.
    eps_DNS2d[:, -1] = eps_DNS2d[:, -2]

    # compute face value of U and V
    u2d_face_w, u2d_face_s = compute_face_phi(u2d, fx, fy, ni, nj)  # Error with dimensions for u2d vs fx,fy
    v2d_face_w, v2d_face_s = compute_face_phi(v2d, fx, fy, ni, nj)

    # x derivatives
    dudx = dphidx(u2d_face_w, u2d_face_s, areawx, areasx, vol)
    dvdx = dphidx(v2d_face_w, v2d_face_s, areawx, areasx, vol)

    # y derivatives
    dudy = dphidy(u2d_face_w, u2d_face_s, areawy, areasy, vol)
    dvdy = dphidy(v2d_face_w, v2d_face_s, areawy, areasy, vol)

    omega = eps_DNS2d / k2d / 0.09

    cmy_DNS = np.array(-uv2d / (k2d * (dudy + dvdx)) * omega)
    cmy_DNS = np.where(abs(dudy + dvdx) < 1, 1, cmy_DNS)
    cmy_DNS = np.where(cmy_DNS > 0, cmy_DNS, 1)
    cmy_DNS = np.where(cmy_DNS <= 3, cmy_DNS, 1)

    duidxj = np.array((dudx ** 2 + 0.5 * (dudy ** 2 + 2 * dudy * dvdx + dvdx ** 2) + dvdy ** 2) ** 0.5)
            
    return 0

def get_ni_nj(y:np.array, x:np.array):
    if max(y) == 1.:
        ni = 170
        nj = 194
        nu = 1. / 10000.
    else:
        nu = 1. / 10595.
        if max(x) > 8.:
            nj = 162
            ni = 162
        else:
            ni = 402
            nj = 162
    return ni,nj,nu 