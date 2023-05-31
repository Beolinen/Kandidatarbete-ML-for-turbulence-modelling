import numpy as np
import pandas as pd
from gradients import compute_face_phi, dphidx, dphidy, init

# give this one a new name later. the idea is to take the paths and use the 
# (hopefully structured the exact same way) data and return a pandas datafrasme
def dat_2_dataset(path_tec:str, path_mesh:str, df:bool):
    x,y,p,u,v,uu,vv,ww,uv,eps,k,ni,nj = dat_2_var_arr(path_tec)

    # set Neumann of p at upper and lower boundaries
    p[:, 1]  = p[:, 2]
    p[:, -1] = p[:, -1 - 1]

    # set periodic b.c on west boundary
    # SHOULD WE USE THIS?
    u[0, :]  = u[-1, :]
    v[0, :]  = v[-1, :]
    p[0, :]  = p[-1, :]
    uu[0, :] = uu[-1, :]

    mesh = np.loadtxt(path_mesh).transpose()
    xf = np.reshape(mesh[0], (nj - 1, ni - 1)).transpose()
    yf = np.reshape(mesh[1], (nj - 1, ni - 1)).transpose()


    # compute cell centers
    xp = 0.25 * (xf[0:-1, 0:-1] + xf[0:-1, 1:] + xf[1:, 0:-1] + xf[1:, 1:])  
    yp = 0.25 * (yf[0:-1, 0:-1] + yf[0:-1, 1:] + yf[1:, 0:-1] + yf[1:, 1:])  

    # compute geometric quantities
    areaw, areawx, areawy, areas, areasx, areasy, vol, fx, fy = init(xf, yf, xp, yp)

    # delete first/last row/col
    x   = np.delete(np.delete(x,   [0,-1], 1), [0,-1], 0)
    y   = np.delete(np.delete(y,   [0,-1], 1), [0,-1], 0)
    p   = np.delete(np.delete(p,   [0,-1], 1), [0,-1], 0)
    u   = np.delete(np.delete(u,   [0,-1], 1), [0,-1], 0)
    v   = np.delete(np.delete(v,   [0,-1], 1), [0,-1], 0)
    uu  = np.delete(np.delete(uu,  [0,-1], 1), [0,-1], 0)
    vv  = np.delete(np.delete(vv,  [0,-1], 1), [0,-1], 0)
    ww  = np.delete(np.delete(ww,  [0,-1], 1), [0,-1], 0)
    uv  = np.delete(np.delete(uv,  [0,-1], 1), [0,-1], 0)
    eps = np.delete(np.delete(eps, [0,-1], 1), [0,-1], 0)
    k   = np.delete(np.delete(k,   [0,-1], 1), [0,-1], 0)

    ni -= 2
    nj -= 2

    # eps at last cell upper cell wrong. fix it.
    eps[:, -1] = eps[:, -2]

    # compute face value of U and V
    u_face_w, u_face_s = compute_face_phi(u, fx, fy, ni, nj)  # Error with dimensions for u2d vs fx,fy
    v_face_w, v_face_s = compute_face_phi(v, fx, fy, ni, nj)

    # x derivatives
    dudx = dphidx(u_face_w, u_face_s, areawx, areasx, vol)
    dvdx = dphidx(v_face_w, v_face_s, areawx, areasx, vol)

    # y derivatives
    dudy = dphidy(u_face_w, u_face_s, areawy, areasy, vol)
    dvdy = dphidy(v_face_w, v_face_s, areawy, areasy, vol)

    # Alt. 1 ger bättre cmy
    # vist_DNS=abs(uv)/dudy
    # omega=(eps/0.09/vist_DNS)**0.5
    # cmy = np.array(abs(uv) / (k * (dudy)) * omega)

    # Alt. 2
    omega = eps / k / 0.09
    cmy = np.array(-uv / (k * (dudy + dvdx)) * omega)
    

    # cmy = np.where(cmy > 0, cmy, 1)
    # cmy = np.where(cmy <= 3, cmy, 1)

    duidxj = np.array((dudx ** 2 + 0.5 * (dudy ** 2 + 2 * dudy * dvdx + dvdx ** 2) + dvdy ** 2) ** 0.5)

    if(df):
        return pd.DataFrame({
        'dudx'  :dudx.transpose().flatten(),
        'dvdx'  :dvdx.transpose().flatten(),
        'dudy'  :dvdy.transpose().flatten(),
        'dvdy'  :dudy.transpose().flatten(),
        'cmy'   :cmy.transpose().flatten(),
        'duidxj':duidxj.transpose().flatten(),
        'x'     :x.transpose().flatten(),
        'y'     :y.transpose().flatten(),
        'p'     :p.transpose().flatten(),
        'u'     :u.transpose().flatten(),
        'v'     :v.transpose().flatten(),
        'uu'    :uu.transpose().flatten(),
        'vv'    :vv.transpose().flatten(),
        'ww'    :ww.transpose().flatten(),
        'uv'    :uv.transpose().flatten(),
        'eps'   :eps.transpose().flatten(),
        'k'     :k.transpose().flatten(),
        })


    return  dudx,dvdx,dudy,dvdy,cmy,duidxj,x,y,p,u,v,uu,vv,ww,uv,eps,k

def dat_2_var_arr(path:str):

    tec = np.genfromtxt(path, dtype=None, comments="%").transpose() # turn every column in the .dat file to a row in tec
    k = 0.5 * (tec[5] + tec[6] + tec[7])                            # Turbulent kinetic energy (?)

    nu,ni,nj = get_ni_nj(path)              # get the dimentions on of the data, based on path name

    k = np.reshape(k, (nj, ni)).transpose() # k has the shape [ni,nj]
    tec = tec.reshape(10,nj,ni)             # tec is reshaped so every variable has the shape [nj,ni]
    tec = np.transpose(tec,axes=[0,2,1])    # tec is transposed so that every variable has the shape [ni,nj]

    x   = tec[0]
    y   = tec[1]
    p   = tec[2]
    u   = tec[3]
    v   = tec[4]
    uu  = tec[5]
    vv  = tec[6]
    ww  = tec[7]
    uv  = tec[8]
    eps = tec[9] # /nu ??

    return x,y,p,u,v,uu,vv,ww,uv,eps,k,ni,nj

def get_ni_nj(path:str) -> tuple[float,int,int]:
    if path == "small_wave/tec.dat":
        return 0.0001,170,194
    if path == "large_wave/tec.dat":
        return 0.0001,170,194
    if path == "one_hill/tec.dat":
        return (1./10595.),162,162
    if path == "two_hills/tec.dat":
        return (1./10595.),402,162
    return 0,0,0
