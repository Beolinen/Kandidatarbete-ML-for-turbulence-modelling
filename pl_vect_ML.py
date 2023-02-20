import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
# from dphidx_dy import dphidx_dy # Kommer ersättas av gradients nedan
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR  # for building SVR model
import random
import gradients
from gradients import compute_face_phi, dphidx, dphidy, init

plt.interactive(True)

# plt.close('all')

# read data file
tec = np.genfromtxt("tec.dat", dtype=None,comments="%")

# text='VARIABLES = X Y P U V u2 v2 w2 uv diss'
x = tec[:, 0] # pos in x-axis
y = tec[:, 1] # pos in y-axis
p = tec[:, 2] # pressure
u = tec[:, 3] #  mean velocity 
v = tec[:, 4] #  mean velocity 
uu = tec[:, 5] #  turbulent velocity (stress)
vv = tec[:, 6] #  turbulent velocity (stress)
ww = tec[:, 7] #  turbulent velocity (stress)
uv = tec[:, 8] # Reynold stress? 
eps = tec[:, 9] # Dissipation
k = 0.5 * (uu + vv + ww)


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

viscos = nu 

# Preparing the matrixes for the gradients
u2d = np.reshape(u, (nj, ni)) 
v2d = np.reshape(v, (nj, ni)) 
p2d = np.reshape(p, (nj, ni)) 
x2d = np.reshape(x, (nj, ni)) 
y2d = np.reshape(y, (nj, ni)) 
uu2d = np.reshape(uu, (nj, ni))  # = mean{v'_1v'_1}
uv2d = np.reshape(uv, (nj, ni))  # = mean{v'_1v'_2}
vv2d = np.reshape(vv, (nj, ni))  # = mean{v'_2v'_2}
ww2d = np.reshape(ww, (nj, ni))  # = mean{v'_2v'_2}
k2d=np.reshape(k,(nj,ni))
eps2d=np.reshape(eps,(nj,ni))

u2d = np.transpose(u2d)
v2d = np.transpose(v2d)
p2d = np.transpose(p2d)
x2d = np.transpose(x2d)
y2d = np.transpose(y2d)
uu2d = np.transpose(uu2d)
vv2d = np.transpose(vv2d)
uv2d = np.transpose(uv2d)
ww2d = np.transpose(ww2d)
k2d = np.transpose(k2d)
eps2d = np.transpose(eps2d)


# set periodic b.c on west boundary
# u2d[0,:]=u2d[-1,:]
# v2d[0,:]=v2d[-1,:]
# p2d[0,:]=p2d[-1,:]
# uu2d[0,:]=uu2d[-1,:]

#  v---------------------------------- ANVÄNDS INTE ----------------------------------v
# # read k and eps from a 2D RANS simulations. They should be used for computing the damping function f
# k_eps_RANS = np.loadtxt("k_eps_RANS.dat")
# k_RANS=k_eps_RANS[:,0]
# eps_RANS=k_eps_RANS[:,1]
# vist_RANS=k_eps_RANS[:,2]

# ntstep=k_RANS[0]

# k_RANS2d=np.reshape(k_RANS,(nj,ni))/ntstep
# eps_RANS2d=np.reshape(eps_RANS,(nj,ni))/ntstep
# vist_RANS2d=np.reshape(vist_RANS,(nj,ni))/ntstep
#  ^---------------------------------- ANVÄNDS INTE ----------------------------------^

# x and y are fo the cell centers. The dphidx_dy routine needs the face coordinate, xf2d, yf2d
# load them
xc_yc = np.loadtxt("xc_yc.dat")
xf = xc_yc[:,0]
yf = xc_yc[:,1]
x2d = np.reshape(xf,(nj,ni))
y2d = np.reshape(yf,(nj,ni))
x2d = np.transpose(x2d)
y2d = np.transpose(y2d)

# compute cell centers
xp2d = 0.25 * (x2d[0:-1,0:-1] + x2d[0:-1,1:] + x2d[1:,0:-1] + x2d[1:,1:]) # (169,193)
yp2d = 0.25 * (y2d[0:-1,0:-1] + y2d[0:-1,1:] + y2d[1:,0:-1] + y2d[1:,1:]) # (169,193)


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
areaw,areawx,areawy,areas,areasx,areasy,vol,fx,fy = init(x2d,y2d,xp2d,yp2d)

# print('x2d.shape', x2d.shape)
# print('u2d.shape', u2d.shape)

# delete last row
u2d = np.delete(u2d, -1, 0)
v2d = np.delete(v2d, -1, 0)
p2d = np.delete(p2d, -1, 0)
k2d = np.delete(k2d, -1, 0)
uu2d = np.delete(uu2d, -1, 0)
vv2d = np.delete(vv2d, -1, 0)
ww2d = np.delete(ww2d, -1, 0)
uv2d = np.delete(uv2d, -1, 0)
eps2d = np.delete(eps2d, -1, 0)
# k_RANS2d = np.delete(k_RANS2d, -1, 0)
# eps_RANS2d = np.delete(eps_RANS2d, -1, 0)
# vist_RANS2d = np.delete(vist_RANS2d, -1, 0)

# delete first row
u2d = np.delete(u2d, 0, 0)
v2d = np.delete(v2d, 0, 0)
p2d = np.delete(p2d, 0, 0)
k2d = np.delete(k2d, 0, 0)
uu2d = np.delete(uu2d, 0, 0)
vv2d = np.delete(vv2d, 0, 0)
ww2d = np.delete(ww2d, 0, 0)
uv2d = np.delete(uv2d, 0, 0)
eps2d = np.delete(eps2d, 0, 0)
# k_RANS2d = np.delete(k_RANS2d, 0, 0)
# eps_RANS2d = np.delete(eps_RANS2d, 0, 0)
# vist_RANS2d = np.delete(vist_RANS2d, 0, 0)

# delete last columns
u2d = np.delete(u2d, -1, 1)
v2d = np.delete(v2d, -1, 1)
p2d = np.delete(p2d, -1, 1)
k2d = np.delete(k2d, -1, 1)
uu2d = np.delete(uu2d, -1, 1)
vv2d = np.delete(vv2d, -1, 1)
ww2d = np.delete(ww2d, -1, 1)
uv2d = np.delete(uv2d, -1, 1) 
eps2d = np.delete(eps2d, -1, 1)
# k_RANS2d = np.delete(k_RANS2d, -1, 1)
# eps_RANS2d = np.delete(eps_RANS2d, -1, 1)
# vist_RANS2d = np.delete(vist_RANS2d, -1, 1) 

# delete first columns
u2d = np.delete(u2d, 0, 1) # (168,192)
v2d = np.delete(v2d, 0, 1) # (168,192)
p2d = np.delete(p2d, 0, 1)
k2d = np.delete(k2d, 0, 1)
uu2d = np.delete(uu2d, 0, 1)
vv2d = np.delete(vv2d, 0, 1)
ww2d = np.delete(ww2d, 0, 1)
uv2d = np.delete(uv2d, 0, 1)
eps2d = np.delete(eps2d, 0, 1) 
# k_RANS2d = np.delete(k_RANS2d, 0, 1)
# eps_RANS2d = np.delete(eps_RANS2d, 0, 1)
# vist_RANS2d = np.delete(vist_RANS2d, 0, 1) 

ni = ni-2
nj = nj-2

# eps at last cell upper cell wrong. fix it.
eps2d[:,-1] = eps2d[:,-2]

# print('new x2d.shape',x2d.shape) 
# print('new u2d.shape',u2d.shape)  

# compute face value of U and V 
u2d_face_w,u2d_face_s = compute_face_phi(u2d,fx,fy,ni,nj) # ger(169,193) vs (168,192)
v2d_face_w,v2d_face_s = compute_face_phi(v2d,fx,fy,ni,nj) 

# x derivatives
dudx = dphidx(u2d_face_w,u2d_face_s,areawx,areasx,vol)
dvdx = dphidx(v2d_face_w,u2d_face_s,areawx,areasx,vol)

# y derivatives
dudy = dphidx(u2d_face_w,u2d_face_s,areawy,areasy,vol)
dvdy = dphidx(v2d_face_w,u2d_face_s,areawy,areasy,vol)


#Prepare data to fit model -
#Compute C_my and ||duidxj|| to train model
# tau = k/eps
#1 (2/3)*k 
NY = - uv2d / (dudy + dvdx) 
cmy = NY*eps2d/(k2d**2) 

# Find values above one 
cmy_rans = cmy > 1

# Cmu cannot be negative
cmy=np.where(cmy > 0,cmy,1)

# if the velocity gradient are very small: set cmu=1
cmy=np.where(abs(dudy+dvdx)  < 1,1,cmy)

cmy_rans = np.reshape(cmy_rans, nj*ni)

duidxj = np.array((dudy**2 + dudx**2 + dvdy**2 + dvdx**2)**0.5)

#ML-metod
duidxj = duidxj.reshape(-1,1)
scaler = MinMaxScaler()

duidxj_scaled = []
duidxj_scaled = scaler.fit_transform(duidxj)

X = np.zeros((len(duidxj_scaled),1)) 
X[:,0] = duidxj_scaled[:,0] 
Y = cmy_rans

#Bygger upp SVR modell
model = SVR(kernel = 'rbf', C = 1, epsilon = 0.0001)
SVR = model.fit(X,Y.flatten())

# ------------------------------ TEST med nytt case ---------------------------------------
#Kolla på KDtree för nearest neighbour och fixa bättre modell approx. 
# Prepare data for new case, same method as previously
tec_large = np.genfromtxt("tec_large.dat", dtype=None,comments="%")

x_large = tec_large[:, 0] 
y_large = tec_large[:, 1] 
p_large = tec_large[:, 2] 
u_large = tec_large[:, 3] 
v_large = tec_large[:, 4] 
uu_large = tec_large[:, 5] 
vv_large = tec_large[:, 6] 
ww_large = tec_large[:, 7] 
uv_large = tec_large[:, 8] 
eps_large = tec_large[:, 9] 
k_large = 0.5 * (uu_large + vv_large + ww_large)

if max(y_large) == 1.:
    ni = 170
    nj = 194
    nu = 1. / 10000.
else:
    nu = 1. / 10595.
    if max(x_large) > 8.:
        nj = 162
        ni = 162
    else:
        ni = 402
        nj = 162

viscos = nu 

u2d_large = np.transpose(np.reshape(u_large, (nj, ni))) 
v2d_large = np.transpose(np.reshape(v_large, (nj, ni))) 
p2d_large = np.transpose(np.reshape(p_large, (nj, ni))) 
k2d_large = np.transpose(np.reshape(k_large, (nj, ni))) 
uu2d_large = np.transpose(np.reshape(uu_large, (nj, ni))) 
vv2d_large = np.transpose(np.reshape(vv_large, (nj, ni))) 
ww2d_large = np.transpose(np.reshape(ww_large, (nj, ni))) 
uv2d_large = np.transpose(np.reshape(uv_large, (nj, ni)))
eps2d_large = np.transpose(np.reshape(eps_large, (nj, ni))) 


# x and y are fo the cell centers. The dphidx_dy routine needs the face coordinate, xf2d, yf2d
# load them 
xc_yc = np.loadtxt("xc_yc_large.dat") 
xf_large = xc_yc[:,0] 
yf_large = xc_yc[:,1] 
x2d_large = np.reshape(xf_large,(nj,ni))
y2d_large = np.reshape(yf_large,(nj,ni))
x2d_large = np.transpose(x2d_large)
y2d_large = np.transpose(y2d_large)

# compute cell centers
xp2d_large = 0.25 * (x2d_large[0:-1,0:-1] + x2d_large[0:-1,1:] + x2d_large[1:,0:-1] + x2d_large[1:,1:])
yp2d_large = 0.25 * (y2d_large[0:-1,0:-1] + y2d_large[0:-1,1:] + y2d_large[1:,0:-1] + y2d_large[1:,1:])

# delete last row
x2d_large = np.delete(x2d_large, -1, 0)
y2d_large = np.delete(y2d_large, -1, 0)
xp2d_large = np.delete(xp2d_large, -1, 0)
yp2d_large = np.delete(yp2d_large, -1, 0)

# delete last columns
x2d_large = np.delete(x2d_large, -1, 1)
y2d_large = np.delete(y2d_large, -1, 1)
xp2d_large = np.delete(xp2d_large, -1, 1)
yp2d_large = np.delete(yp2d_large, -1, 1)

# compute geometric quantities
areaw_large,areawx_large,areawy_large,areas_large,areasx_large,areasy_large,vol_large,fx_large,fy_large = init(x2d_large, y2d_large, xp2d_large, yp2d_large)

# print('x2d.shape', x2d_large.shape)
# print('u2d.shape', u2d_large.shape)

# delete last row
u2d_large = np.delete(u2d_large, -1, 0)
v2d_large = np.delete(v2d_large, -1, 0)
p2d_large = np.delete(p2d_large, -1, 0)
k2d_large = np.delete(k2d_large, -1, 0)
uu2d_large = np.delete(uu2d_large, -1, 0)
vv2d_large = np.delete(vv2d_large, -1, 0)
ww2d_large = np.delete(ww2d_large, -1, 0)
uv2d_large = np.delete(uv2d_large, -1, 0)
eps2d_large = np.delete(eps2d_large, -1, 0)
# k_RANS2d = np.delete(k_RANS2d, -1, 0)
# eps_RANS2d = np.delete(eps_RANS2d, -1, 0)
# vist_RANS2d = np.delete(vist_RANS2d, -1, 0)

# delete first row
u2d_large = np.delete(u2d_large, 0, 0)
v2d_large = np.delete(v2d_large, 0, 0)
p2d_large = np.delete(p2d_large, 0, 0)
k2d_large = np.delete(k2d_large, 0, 0)
uu2d_large = np.delete(uu2d_large, 0, 0)
vv2d_large = np.delete(vv2d_large, 0, 0)
ww2d_large = np.delete(ww2d_large, 0, 0)
uv2d_large = np.delete(uv2d_large, 0, 0)
eps2d_large = np.delete(eps2d_large, 0, 0)
# k_RANS2d = np.delete(k_RANS2d, 0, 0)
# eps_RANS2d = np.delete(eps_RANS2d, 0, 0)
# vist_RANS2d = np.delete(vist_RANS2d, 0, 0)

# delete last columns
u2d_large = np.delete(u2d_large, -1, 1)
v2d_large = np.delete(v2d_large, -1, 1)
p2d_large = np.delete(p2d_large, -1, 1)
k2d_large = np.delete(k2d_large, -1, 1)
uu2d_large = np.delete(uu2d_large, -1, 1)
vv2d_large = np.delete(vv2d_large, -1, 1)
ww2d_large = np.delete(ww2d_large, -1, 1)
uv2d_large = np.delete(uv2d_large, -1, 1) 
eps2d_large = np.delete(eps2d_large, -1, 1)
# k_RANS2d = np.delete(k_RANS2d, -1, 1)
# eps_RANS2d = np.delete(eps_RANS2d, -1, 1)
# vist_RANS2d = np.delete(vist_RANS2d, -1, 1) 

# delete first columns
u2d_large = np.delete(u2d_large, 0, 1)
v2d_large = np.delete(v2d_large, 0, 1)
p2d_large = np.delete(p2d_large, 0, 1)
k2d_large = np.delete(k2d_large, 0, 1)
uu2d_large = np.delete(uu2d_large, 0, 1)
vv2d_large = np.delete(vv2d_large, 0, 1)
ww2d_large = np.delete(ww2d_large, 0, 1)
uv2d_large = np.delete(uv2d_large, 0, 1)
eps2d_large = np.delete(eps2d_large, 0, 1) 
# k_RANS2d = np.delete(k_RANS2d, 0, 1)
# eps_RANS2d = np.delete(eps_RANS2d, 0, 1)
# vist_RANS2d = np.delete(vist_RANS2d, 0, 1) 

ni = ni-2
nj = nj-2

# eps at last cell upper cell wrong. fix it.
eps2d_large[:,-1] = eps2d_large[:,-2]

# print('new x2d.shape',x2d.shape) 
# print('new u2d.shape',u2d.shape)  

# compute face value of U and V 
u2d_face_w_large,u2d_face_s_large = compute_face_phi(u2d_large,fx_large,fy_large,ni,nj) 
v2d_face_w_large,v2d_face_s_large = compute_face_phi(v2d_large,fx_large,fy_large,ni,nj) 

# x derivatives
dudx_large = dphidx(u2d_face_w_large,u2d_face_s_large,areawx_large,areasx_large,vol_large)
dvdx_large = dphidx(v2d_face_w_large,u2d_face_s_large,areawx_large,areasx_large,vol_large)

# x derivatives
dudy_large = dphidx(u2d_face_w_large,u2d_face_s_large,areawy_large,areasy_large,vol_large)
dvdy_large = dphidx(v2d_face_w_large,u2d_face_s_large,areawy_large,areasy_large,vol_large)


duidxj_test = np.array((dudy_large**2 + dudx_large**2 + dvdy_large**2 + dvdx_large**2)**0.5) 
duidxj_test = duidxj_test.reshape(-1,1) 

duidxj_test_scaled = scaler.fit_transform(duidxj_test) 

X_test = np.zeros((len(duidxj_test),1)) 
X_test[:,0] = duidxj_test_scaled[:,0] 

# Predict C_my using model created before.
y_svr = model.predict(X_test) 
y_svr_no_scale = scaler.inverse_transform(y_svr.reshape(-1,1)) 
y_svr_no_scale = y_svr_no_scale.flatten() 

X_test_no_scale = scaler.inverse_transform(X_test)

# Actual C_my_large using RANS. 
# NY2d_large = - uv2d_large / (dudy_large + dvdx_large)
# cmy_large_RANS = NY2d_large*eps2d_large/(k2d_large**2) 
# cmy_large_RANS = np.reshape(cmy_large_RANS, nj*ni)


# Calculate error
errorML = (np.std(y_svr - cmy))/(np.mean(y_svr**2))**0.5 
error = (np.std(0.09 - cmy))/(np.mean(0.09**2))**0.5 
errorOmega = (np.std(1-cmy))/(np.mean((1)**2))**0.5 

# Print error
print("RMS-felet med ML är", errorML) 
print("RMS-felet med standardmodell (C_my = 0.09) är", error) 
print("RMS-felet med standardmodell ,k-omega, (C_my = -1) är", errorOmega) 

# Plotta lösning 
plt.scatter(X_test_no_scale, cmy, marker = "o", s = 10, c = "green", label = "Target") 
plt.scatter(X_test_no_scale, y_svr, marker = "o", s= 10, c = "blue", label = "Prediction") 
plt.legend(loc="lower right", prop=dict(size=12)) 
plt.xlabel("$ || \partial u_i^+ / \partial x_j^+ ||$") 
plt.ylabel("$C^{k-\omega}_\mu$") 
plt.title("$C^{k-\omega}_\mu = f( ||\partial u_i^+/ \partial x_j^+||)$") 
# plt.savefig("Modell1.png")
plt.show()