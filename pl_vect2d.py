import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from dphidx_dy import dphidx_dy
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR  # for building SVR model
import random

# read data file
tec=np.genfromtxt("/Users/benjaminjonsson/Programmering/Kandidat/small_wave/tec.dat", dtype=None,comments="%")

# text='VARIABLES = X Y P U V u2 v2 w2 uv mu_sgs prod'
x = tec[:, 0]
y = tec[:, 1]
p = tec[:, 2]
u = tec[:, 3]
v = tec[:, 4]
uu = tec[:, 5]
vv = tec[:, 6]
ww = tec[:, 7]
uv = tec[:, 8]
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

u2d = np.reshape(u, (nj, ni))
v2d = np.reshape(v, (nj, ni))
p2d = np.reshape(p, (nj, ni))
x2d = np.reshape(x, (nj, ni))
y2d = np.reshape(y, (nj, ni))
uu2d = np.reshape(uu, (nj, ni))  # =mean{v'_1v'_1}
uv2d = np.reshape(uv, (nj, ni))  # =mean{v'_1v'_2}
vv2d = np.reshape(vv, (nj, ni))  # =mean{v'_2v'_2}
k2d = np.reshape(k, (nj, ni))  # =mean{0.5(v'_iv'_i)}

u2d = np.transpose(u2d)
v2d = np.transpose(v2d)
p2d = np.transpose(p2d)
x2d = np.transpose(x2d)
y2d = np.transpose(y2d)
uu2d = np.transpose(uu2d)
vv2d = np.transpose(vv2d)
uv2d = np.transpose(uv2d)
k2d = np.transpose(k2d)

# set periodic b.c on west boundary
# u2d[0,:]=u2d[-1,:]
# v2d[0,:]=v2d[-1,:]
# p2d[0,:]=p2d[-1,:]
# uu2d[0,:]=uu2d[-1,:]


# read k and eps from a 2D RANS simulations. They should be used for computing the damping function f
k_eps_RANS = np.loadtxt("/Users/benjaminjonsson/Programmering/Kandidat/small_wave/k_eps_RANS.dat")
k_RANS = k_eps_RANS[:, 0]
diss_RANS = k_eps_RANS[:, 1]
vist_RANS = k_eps_RANS[:, 2]

ntstep = k_RANS[0]

k_RANS_2d = np.reshape(k_RANS, (ni, nj)) / ntstep  # modeled turbulent kinetic energy
diss_RANS_2d = np.reshape(diss_RANS, (ni, nj)) / ntstep  # modeled dissipation
vist_RANS_2d = np.reshape(vist_RANS, (ni, nj)) / ntstep  # turbulent viscosity, AKN model

# set small values on k & eps at upper and lower boundaries to prevent NaN on division
diss_RANS_2d[:, 0] = 1e-10
k_RANS_2d[:, 0] = 1e-10
vist_RANS_2d[:, 0] = nu
diss_RANS_2d[:, -1] = 1e-10
k_RANS_2d[:, -1] = 1e-10
vist_RANS_2d[:, -1] = nu

# set Neumann of p at upper and lower boundaries
p2d[:, 1] = p2d[:, 2]
p2d[:, -1] = p2d[:, -1 - 1]

# x and y are fo the cell centers. The dphidx_dy routine needs the face coordinate, xf2d, yf2d
# load them
xc_yc = np.loadtxt("/Users/benjaminjonsson/Programmering/Kandidat/small_wave/xc_yc.dat")
xf = xc_yc[:, 0]
yf = xc_yc[:, 1]
xf2d = np.reshape(xf, (nj, ni))
yf2d = np.reshape(yf, (nj, ni))
xf2d = np.transpose(xf2d)
yf2d = np.transpose(yf2d)

# delete last row
xf2d = np.delete(xf2d, -1, 0)
yf2d = np.delete(yf2d, -1, 0)

# delete last columns
xf2d = np.delete(xf2d, -1, 1)
yf2d = np.delete(yf2d, -1, 1)

# compute the gradient dudx, dudy at point P
dudx = np.zeros((ni, nj))
dudy = np.zeros((ni, nj))
dvdx = np.zeros((ni, nj))
dvdy = np.zeros((ni, nj))

dudx, dudy = dphidx_dy(xf2d, yf2d, u2d)
dvdx, dvdy = dphidx_dy(xf2d, yf2d, v2d)

#Prepare data to fit model
indexDrop = []
E = k_RANS
K = diss_RANS
NY = vist_RANS

for i in range(len(diss_RANS)):
    if diss_RANS[i] == 0:
        indexDrop.append(i)

E = np.delete(diss_RANS, indexDrop)
K = np.delete(k_RANS, indexDrop)
NY = np.delete(vist_RANS, indexDrop)
UV = np.delete(uv, indexDrop)
DUDY = np.delete(dudy,indexDrop)
DVDX = np.delete(dvdx,indexDrop)

omega = E/(K*0.09)

#Compute C_my and ||duidxj|| to train model
#Option 2 and 3 seem to work

#2 k-eps method
cmy_rans = np.array(NY*E/(K**2))

#3 k-omega method
#cmy_rans = np.array(NY*omega/K)

duidxj = np.array((dudy**2 + dudx**2 + dvdy**2 + dvdx**2)**0.5)
duidxj = np.delete(duidxj, indexDrop)

#ML-metod
duidxj = duidxj.reshape(-1,1)

scaler = MinMaxScaler()

duidxj_scaled = []
duidxj_scaled = scaler.fit_transform(duidxj)

X = np.zeros((len(duidxj_scaled),1))
X[:,0] = duidxj_scaled[:,0]
Y = cmy_rans

model = SVR(kernel = 'rbf', C = 1, epsilon = 0.001)
SVR = model.fit(X,Y.flatten())

#TEST med nytt case
#Prepare data for new case, same method as previously
tec_large = np.genfromtxt("/Users/benjaminjonsson/Programmering/Kandidat/large_wave/tec_large.dat", dtype=None,comments="%")

u_large = tec_large[:,3]
v_large = tec_large[:,4]
y_large = tec_large[:,1]
x_large = tec_large[:,0]
uv_large = tec_large[:,8]

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

xc_yc_large = np.loadtxt("/Users/benjaminjonsson/Programmering/Kandidat/large_wave/xc_yc_large.dat")
xf_large = xc_yc_large[:, 0]
yf_large = xc_yc_large[:, 1]
xf2d_large = np.reshape(xf_large, (nj, ni))
yf2d_large = np.reshape(yf_large, (nj, ni))
xf2d_large = np.transpose(xf2d_large)
yf2d_large = np.transpose(yf2d_large)

#Delete last row
xf2d_large = np.delete(xf2d_large, -1, 0)
yf2d_large = np.delete(yf2d_large, -1, 0)

#Delete last columns
xf2d_large = np.delete(xf2d_large, -1, 1)
yf2d_large = np.delete(yf2d_large, -1, 1)

#Compute the gradient dudx, dudy at point P
dudx_large = np.zeros((ni, nj))
dudy_large = np.zeros((ni, nj))
dvdx_large = np.zeros((ni, nj))
dvdy_large = np.zeros((ni, nj))

dudx_large, dudy_large = dphidx_dy(xf2d_large, yf2d_large, u2d_large)
dvdx_large, dvdy_large = dphidx_dy(xf2d_large, yf2d_large, v2d_large)

duidxj_test = np.array((dudy_large**2 + dudx_large**2 + dvdy_large**2 + dvdx_large**2)**0.5)
duidxj_test = np.delete(duidxj_test,indexDrop)
duidxj_test = duidxj_test.reshape(-1,1)

duidxj_test_scaled = scaler.fit_transform(duidxj_test)

X_test = np.zeros((len(duidxj_test),1))
X_test[:,0] = duidxj_test_scaled[:,0]

y_svr = model.predict(X_test)

X_test_no_scale = scaler.inverse_transform(X_test)

#Calculate correct C_my for prediction
k_eps_RANS_large = np.loadtxt("/Users/benjaminjonsson/Programmering/Kandidat/small_wave/k_eps_RANS.dat")
k_RANS_large = k_eps_RANS_large[:, 0]
diss_RANS_large = k_eps_RANS_large[:, 1]
vist_RANS_large = k_eps_RANS_large[:, 2]

indexDrop_large = []
E_large = k_RANS_large
K_large = diss_RANS_large
NY_large = vist_RANS_large

for i in range(len(diss_RANS_large)):
    if diss_RANS_large[i] == 0:
        indexDrop_large.append(i)

E_large = np.delete(diss_RANS_large, indexDrop_large)
K_large= np.delete(k_RANS_large, indexDrop_large)
NY_large = np.delete(vist_RANS_large, indexDrop_large)
UV_large = np.delete(uv_large, indexDrop_large)
DUDY_large = np.delete(dudy_large,indexDrop_large)
DVDX_large = np.delete(dvdx_large,indexDrop_large)

omega_large = E_large/(K_large*0.09)

#k-omega
#cmy_rans_large = np.array(NY_large*omega_large/K_large)

#k-eps
cmy_rans_large = np.array(NY*E/(K**2))

#Calculate error
errorML = (np.std(y_svr - cmy_rans_large))/(np.mean(y_svr**2))**0.5
error = (np.std(0.09 - cmy_rans_large))/(np.mean(0.09**2))**0.5
errorOmega = (np.std(1-cmy_rans_large))/(np.mean((1)**2))**0.5

predictOwnCase = model.predict(X)
errorOwnCase = (np.std(predictOwnCase - cmy_rans))/(np.mean(predictOwnCase**2))**0.5

#Print error
print("RMS-felet med ML är", errorML)
print("RMS-felet med standardmodell (C_my = 0.09) är", error)
print("RMS-felet med standardmodell ,k-omega, (C_my = 1) är", errorOmega)
print("Error in fitting case is",errorOwnCase)

#Plotta lösning
plt.figure("Test")
plt.scatter(X_test_no_scale,cmy_rans_large,marker = "o", s = 10, c = "green",label = "Target")
plt.scatter(X_test_no_scale,y_svr,marker = "o", s= 10, c = "blue", label = "Prediction")
plt.legend(loc="lower right",prop=dict(size=12))
plt.xlabel("$ || \partial u_i^+ / \partial x_j^+ ||$")
plt.ylabel("$C_\mu$")
plt.title("$C_\mu = f( ||\partial u_i^+/ \partial x_j^+||)$")
plt.savefig("Modell2_small_test_large.png")
plt.show()