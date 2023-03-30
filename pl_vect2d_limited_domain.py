# ----------------------------------------------Import Packages------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from gradients import compute_face_phi, dphidx, dphidy, init
from sklearn.metrics import mean_squared_error
import sklearn.metrics as sm
import time
import warnings
import matplotlib.cbook
import sys

# ----------------------------------------------Read Data Original Case----------------------------------------------

# PROBLEMS START AT THESE VALUES
# LIMITS LARGE CASE 90-COLUMN 163-ROW
# LIMITS SMALL CASE 116-COLLUMN 135-ROW
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

# read data file
st = time.process_time()
tec = np.genfromtxt("large_wave/tec_large.dat", dtype=None, comments="%")

print("Starting script")
# text='VARIABLES = X Y P U V u2 v2 w2 uv eps'
# Define variables from text-file
x = tec[:, 0]
y = tec[:, 1]
p = tec[:, 2]
u = tec[:, 3]
v = tec[:, 4]
uu = tec[:, 5]
vv = tec[:, 6]
ww = tec[:, 7]
uv = tec[:, 8]
k_DNS = 0.5 * (uu + vv + ww)
eps_DNS = tec[:, 9]

# Define matrix dimensions
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

# Reshape vectors to wanted dimensions
u2d = np.reshape(u, (nj, ni))
v2d = np.reshape(v, (nj, ni))
p2d = np.reshape(p, (nj, ni))
x2d = np.reshape(x, (nj, ni))
y2d = np.reshape(y, (nj, ni))
uu2d = np.reshape(uu, (nj, ni))  # =mean{v'_1v'_1}
uv2d = np.reshape(uv, (nj, ni))  # =mean{v'_1v'_2}
vv2d = np.reshape(vv, (nj, ni))  # =mean{v'_2v'_2}
ww2d = np.reshape(ww, (nj, ni))
k2d = np.reshape(k_DNS, (nj, ni))  # =mean{0.5(v'_iv'_i)}
eps_DNS2d = np.reshape(eps_DNS, (nj, ni))

u2d = np.transpose(u2d)
v2d = np.transpose(v2d)
p2d = np.transpose(p2d)
x2d = np.transpose(x2d)
y2d = np.transpose(y2d)
uu2d = np.transpose(uu2d)
vv2d = np.transpose(vv2d)
ww2d = np.transpose(ww2d)
uv2d = np.transpose(uv2d)
k2d = np.transpose(k2d)
eps_DNS2d = np.transpose(eps_DNS2d)

# set Neumann of p at upper and lower boundaries
p2d[:, 1] = p2d[:, 2]
p2d[:, -1] = p2d[:, -1 - 1]

u2d[0, :] = u2d[-1, :]
v2d[0, :] = v2d[-1, :]
p2d[0, :] = p2d[-1, :]
uu2d[0, :] = uu2d[-1, :]

# x and y are fo the cell centers. The dphidx_dy routine needs the face coordinate, xf2d, yf2d
# load them
xc_yc = np.loadtxt("large_wave/mesh_large.dat")
xf = xc_yc[:, 0]
yf = xc_yc[:, 1]
xf2d = np.reshape(xf, (nj - 1, ni - 1))
yf2d = np.reshape(yf, (nj - 1, ni - 1))
xf2d = np.transpose(xf2d)
yf2d = np.transpose(yf2d)

# compute cell centers
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

# delete last row
u2d = np.delete(u2d, -1, 0)
v2d = np.delete(v2d, -1, 0)
p2d = np.delete(p2d, -1, 0)
k2d = np.delete(k2d, -1, 0)
uu2d = np.delete(uu2d, -1, 0)
vv2d = np.delete(vv2d, -1, 0)
ww2d = np.delete(ww2d, -1, 0)
uv2d = np.delete(uv2d, -1, 0)
eps_DNS2d = np.delete(eps_DNS2d, -1, 0)

# delete first row
u2d = np.delete(u2d, 0, 0)
v2d = np.delete(v2d, 0, 0)
p2d = np.delete(p2d, 0, 0)
k2d = np.delete(k2d, 0, 0)
uu2d = np.delete(uu2d, 0, 0)
vv2d = np.delete(vv2d, 0, 0)
ww2d = np.delete(ww2d, 0, 0)
uv2d = np.delete(uv2d, 0, 0)
eps_DNS2d = np.delete(eps_DNS2d, 0, 0)

# delete last columns
u2d = np.delete(u2d, -1, 1)
v2d = np.delete(v2d, -1, 1)
p2d = np.delete(p2d, -1, 1)
k2d = np.delete(k2d, -1, 1)
uu2d = np.delete(uu2d, -1, 1)
vv2d = np.delete(vv2d, -1, 1)
ww2d = np.delete(ww2d, -1, 1)
uv2d = np.delete(uv2d, -1, 1)
eps_DNS2d = np.delete(eps_DNS2d, -1, 1)

# delete first columns
u2d = np.delete(u2d, 0, 1)
v2d = np.delete(v2d, 0, 1)
p2d = np.delete(p2d, 0, 1)
k2d = np.delete(k2d, 0, 1)
uu2d = np.delete(uu2d, 0, 1)
vv2d = np.delete(vv2d, 0, 1)
ww2d = np.delete(ww2d, 0, 1)
uv2d = np.delete(uv2d, 0, 1)
eps_DNS2d = np.delete(eps_DNS2d, 0, 1)

xp2d = xp2d[:, -163::]
yp2d = yp2d[:, -163::]

ni = ni - 2
nj = nj - 2

# eps at last cell upper cell wrong. fix it.
eps_DNS2d[:, -1] = eps_DNS2d[:, -2]

print('new x2d.shape', x2d.shape)
print('new u2d.shape', u2d.shape)

# compute face value of U and V
u2d_face_w, u2d_face_s = compute_face_phi(u2d, fx, fy, ni, nj)  # Error with dimensions for u2d vs fx,fy
v2d_face_w, v2d_face_s = compute_face_phi(v2d, fx, fy, ni, nj)

# x derivatives
dudx = dphidx(u2d_face_w, u2d_face_s, areawx, areasx, vol)
dvdx = dphidx(v2d_face_w, v2d_face_s, areawx, areasx, vol)

# y derivatives
dudy = dphidy(u2d_face_w, u2d_face_s, areawy, areasy, vol)
dvdy = dphidy(v2d_face_w, v2d_face_s, areawy, areasy, vol)

# ----------------------------------------------Project-Group Work---------------------------------------------------
# ----------------------------------------------ML-Method Original Case----------------------------------------------
print("Data read")
print("Starting ML")

omega = eps_DNS2d / k2d / 0.09

# Compute C_my and ||duidxj|| to train model
cmy_DNS = np.array(-uv2d / (k2d * (dudy + dvdx)) * omega)
cmy_DNS = cmy_DNS[:, -163::]

# cmy_DNS = np.where(cmy_DNS > 3, 1, cmy_DNS)
# cmy_DNS = np.where(cmy_DNS < 0, 1, cmy_DNS)

duidxj = np.array((dudx ** 2 + 0.5 * (dudy ** 2 + 2 * dudy * dvdx + dvdx ** 2) + dvdy ** 2) ** 0.5)
duidxj = duidxj[:, -163::]

k2d = k2d[:, -163::]
uv2d = uv2d[:, -163::]
eps_DNS2d = eps_DNS2d[:, -163::]
p2d = p2d[:, -163::]
vv2d = vv2d[:, -163::]
uu2d = uu2d[:, -163::]

# ML-metod
# The MinMaxScaler works by first computing the minimum and maximum values of each feature in the training data. 
# It then scales the values of each feature such that the minimum value is mapped to 0 and the maximum value is mapped to 1.
scaler1 = StandardScaler()
scaler2 = StandardScaler()
scaler3 = StandardScaler()
scaler4 = StandardScaler()
scaler5 = StandardScaler()
scaler6 = StandardScaler()
scaler7 = StandardScaler()

# Reshape Data
duidxj = duidxj.reshape(-1, 1)
k_scaled_reshaped = k2d.reshape(-1, 1)
uv_scaled_reshaped = uv2d.reshape(-1, 1)
eps_scaled_reshaped = eps_DNS2d.reshape(-1, 1)
p2d_scaled_reshaped = p2d.reshape(-1, 1)
vv2d_scaled_reshaped = vv2d.reshape(-1, 1)
uu2d_scaled_reshaped = uu2d.reshape(-1, 1)

# scale data
duidxj_scaled = scaler1.fit_transform(duidxj)
k_scaled = scaler2.fit_transform(k_scaled_reshaped)
uv_scaled = scaler3.fit_transform(uv_scaled_reshaped)
eps_scaled = scaler4.fit_transform(eps_scaled_reshaped)
p2d_scaled = scaler5.fit_transform(p2d_scaled_reshaped)
vv2d_scaled = scaler6.fit_transform(vv2d_scaled_reshaped)
uu2d_scaled = scaler7.fit_transform(uu2d_scaled_reshaped)

# 2 columns for 3D plot, 1 for 2D --> comment second column
X = np.zeros((len(duidxj_scaled), 7))
X[:, 0] = duidxj_scaled[:, 0]  # Origanal training data
X[:, 1] = uv_scaled[:, 0]  # Original training data
X[:, 2] = k_scaled[:, 0]
X[:, 3] = eps_scaled[:, 0]
X[:, 4] = p2d_scaled[:, 0]
X[:, 5] = vv2d_scaled[:, 0]
X[:, 6] = uu2d_scaled[:, 0]

Y = cmy_DNS

# plt.figure()
# plt.plot(yp2d[-1,:],uv2d[-1,:], 'o')
# plt.figure()
# plt.plot(yp2d_large[-1,:],k_DNS2d[-1,:], 'b-')
# plt.figure()
# plt.plot(yp2d_large[-1,:],omega_large[-1,:], 'b--')
# plt.figure()
# plt.plot(yp2d_large[-1,:],dudy_large[-1,:], 'r-')
# plt.figure()
# plt.plot(yp2d_large[-1,:],dvdx_large[-1,:], 'k-')

# plt.figure()
# plt.plot(xp2d[:,-135::],cmy_DNS[:,-135::]) #rad -100 ger ickefysikaliska värden
# plt.show()
# sys.exit()

# Fit model
model = SVR(kernel='rbf', C=1, epsilon=0.001)
SVR = model.fit(X, Y.flatten())

# ----------------------------------------------Test With New Case------------------------------------------------
# ----------------------------------------------Read Data Large Case----------------------------------------------
print("Reading new case")

tec_large = np.genfromtxt("small_wave/tec.dat", dtype=None, comments="%")

u_large = tec_large[:, 3]
v_large = tec_large[:, 4]
y_large = tec_large[:, 1]
x_large = tec_large[:, 0]
uv_large = tec_large[:, 8]
uu_large = tec_large[:, 5]
vv_large = tec_large[:, 6]
ww_large = tec_large[:, 7]
k_DNS_large = 0.5 * (uu_large + vv_large + ww_large)
eps_DNS_large = tec_large[:, 9]
p_large = tec_large[:, 2]

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

viscous_large = nu

u2d_large = np.transpose(np.reshape(u_large, (nj, ni)))
v2d_large = np.transpose(np.reshape(v_large, (nj, ni)))
uv2d_large = np.transpose(np.reshape(uv_large, (nj, ni)))
k_DNS2d = np.transpose(np.reshape(k_DNS_large, (nj, ni)))
eps_DNS2d_large = np.transpose(np.reshape(eps_DNS_large, (nj, ni)))
x2d_large = np.transpose(np.reshape(x_large, (nj, ni)))
y2d_large = np.transpose(np.reshape(y_large, (nj, ni)))
p2d_large = np.transpose(np.reshape(p_large, (nj, ni)))
uu2d_large = np.transpose(np.reshape(uu_large, (nj, ni)))
vv2d_large = np.transpose(np.reshape(vv_large, (nj, ni)))
ww2d_large = np.transpose(np.reshape(ww_large, (nj, ni)))

# set Neumann of p at upper and lower boundaries
p2d_large[:, 1] = p2d_large[:, 2]
p2d_large[:, -1] = p2d_large[:, -1 - 1]

u2d_large[0, :] = u2d_large[-1, :]
v2d_large[0, :] = v2d_large[-1, :]
p2d_large[0, :] = p2d_large[-1, :]
uu2d_large[0, :] = uu2d_large[-1, :]

xc_yc_large = np.loadtxt("small_wave/mesh.dat")
xf_large = xc_yc_large[:, 0]
yf_large = xc_yc_large[:, 1]
xf2d_large = np.reshape(xf_large, (nj - 1, ni - 1))
yf2d_large = np.reshape(yf_large, (nj - 1, ni - 1))
xf2d_large = np.transpose(xf2d_large)
yf2d_large = np.transpose(yf2d_large)

# compute cell centers
xp2d_large = 0.25 * (x2d_large[0:-1, 0:-1] + x2d_large[0:-1, 1:] + x2d_large[1:, 0:-1] + x2d_large[1:, 1:])
yp2d_large = 0.25 * (y2d_large[0:-1, 0:-1] + y2d_large[0:-1, 1:] + y2d_large[1:, 0:-1] + y2d_large[1:, 1:])

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
areaw, areawx, areawy, areas, areasx, areasy, vol, fx, fy = init(x2d_large, y2d_large, xp2d_large, yp2d_large)

# delete last row
u2d_large = np.delete(u2d_large, -1, 0)
v2d_large = np.delete(v2d_large, -1, 0)
p2d_large = np.delete(p2d_large, -1, 0)
k_DNS2d = np.delete(k_DNS2d, -1, 0)
uu2d_large = np.delete(uu2d_large, -1, 0)
vv2d_large = np.delete(vv2d_large, -1, 0)
ww2d_large = np.delete(ww2d_large, -1, 0)
uv2d_large = np.delete(uv2d_large, -1, 0)
eps_DNS2d_large = np.delete(eps_DNS2d_large, -1, 0)

# delete first row
u2d_large = np.delete(u2d_large, 0, 0)
v2d_large = np.delete(v2d_large, 0, 0)
p2d_large = np.delete(p2d_large, 0, 0)
k_DNS2d = np.delete(k_DNS2d, 0, 0)
uu2d_large = np.delete(uu2d_large, 0, 0)
vv2d_large = np.delete(vv2d_large, 0, 0)
ww2d_large = np.delete(ww2d_large, 0, 0)
uv2d_large = np.delete(uv2d_large, 0, 0)
eps_DNS2d_large = np.delete(eps_DNS2d_large, 0, 0)

# delete last columns
u2d_large = np.delete(u2d_large, -1, 1)
v2d_large = np.delete(v2d_large, -1, 1)
p2d_large = np.delete(p2d_large, -1, 1)
k_DNS2d = np.delete(k_DNS2d, -1, 1)
uu2d_large = np.delete(uu2d_large, -1, 1)
vv2d_large = np.delete(vv2d_large, -1, 1)
ww2d_large = np.delete(ww2d_large, -1, 1)
uv2d_large = np.delete(uv2d_large, -1, 1)
eps_DNS2d_large = np.delete(eps_DNS2d_large, -1, 1)

# delete first columns
u2d_large = np.delete(u2d_large, 0, 1)
v2d_large = np.delete(v2d_large, 0, 1)
p2d_large = np.delete(p2d_large, 0, 1)
k_DNS2d = np.delete(k_DNS2d, 0, 1)
uu2d_large = np.delete(uu2d_large, 0, 1)
vv2d_large = np.delete(vv2d_large, 0, 1)
ww2d_large = np.delete(ww2d_large, 0, 1)
uv2d_large = np.delete(uv2d_large, 0, 1)
eps_DNS2d_large = np.delete(eps_DNS2d_large, 0, 1)

# xp2d_large = xp2d_large[:,-163::]
# yp2d_large = yp2d_large[:,-163::]

ni = ni - 2
nj = nj - 2

# eps at last cell upper cell wrong. fix it.
eps_DNS2d_large[:, -1] = eps_DNS2d_large[:, -2]

print('new x2d.shape', x2d.shape)
print('new u2d.shape', u2d.shape)

# compute face value of U and V
u2d_face_w, u2d_face_s = compute_face_phi(u2d_large, fx, fy, ni, nj)  # Error with dimensions for u2d vs fx,fy
v2d_face_w, v2d_face_s = compute_face_phi(v2d_large, fx, fy, ni, nj)

# x derivatives
dudx_large = dphidx(u2d_face_w, u2d_face_s, areawx, areasx, vol)
dvdx_large = dphidx(v2d_face_w, v2d_face_s, areawx, areasx, vol)

# y derivatives
dudy_large = dphidy(u2d_face_w, u2d_face_s, areawy, areasy, vol)
dvdy_large = dphidy(v2d_face_w, v2d_face_s, areawy, areasy, vol)

# ----------------------------------------------ML-Method Large Case----------------------------------------------
print("Starting ML new case")

# Calculate correct C_my for prediction
omega_large = eps_DNS2d_large / k_DNS2d / 0.09

cmy_DNS_large = np.array(-uv2d_large / (k_DNS2d * (dudy_large + dvdx_large)) * omega_large)
# cmy_DNS_large = cmy_DNS_large[:,-163::]

cmy_DNS_large = np.where(cmy_DNS_large > 2, 1, cmy_DNS_large)
cmy_DNS_large = np.where(cmy_DNS_large < 0, 1, cmy_DNS_large)

# plt.figure()
# plt.plot(yp2d_large[-1,:],uv2d_large[-1,:], 'o')
# plt.figure()
# plt.plot(yp2d_large[-1,:],k_DNS2d[-1,:], 'b-')
# plt.figure()
# plt.plot(yp2d_large[-1,:],omega_large[-1,:], 'b--')
# plt.figure()
# plt.plot(yp2d_large[-1,:],dudy_large[-1,:], 'r-')
# plt.figure()
# plt.plot(yp2d_large[-1,:],dvdx_large[-1,:], 'k-')

# plt.figure()
# plt.plot(yp2d_large[-100,:],cmy_DNS_large[-100,:], 'ro') #rad -100 ger ickefysikaliska värden
# plt.show()
# sys.exit()

# np.array is used to convert the list to an array
duidxj_test = np.array((dudx_large ** 2 + 0.5 * (
            dudy_large ** 2 + 2 * dudy_large * dvdx_large + dvdx_large ** 2) + dvdy_large ** 2) ** 0.5)
# duidxj_test = duidxj_test[:,-163::]

# k_DNS2d = k_DNS2d[:,-163::]
# uv2d_large = uv2d_large[:,-163::]
# eps_DNS2d_large = eps_DNS2d_large[:,-163::]
# p2d_large = p2d_large[:,-163::]
# vv2d_large = vv2d_large[:,-163::]
# uu2d_large = uu2d_large[:,-163::]
# ww2d_large = ww2d_large[:,-163::]

# Reshape data
duidxj_test_reshape = duidxj_test.reshape(-1, 1)
uv_large_reshape = uv2d_large.reshape(-1, 1)
k_scaled_reshape = k_DNS2d.reshape(-1, 1)
eps_large_reshape = eps_DNS2d_large.reshape(-1, 1)
p2d_large_reshape = p2d_large.reshape(-1, 1)
vv2d_large_reshape = vv2d_large.reshape(-1, 1)
uu2d_large_reshape = uu2d_large.reshape(-1, 1)

# Scale data
duidxj_test_scaled = scaler1.transform(duidxj_test_reshape)
k_scaled_large = scaler2.transform(k_scaled_reshape)
uv_large_scaled = scaler3.transform(uv_large_reshape)
eps_large_scaled = scaler4.transform(eps_large_reshape)
p2d_large_scaled = scaler5.transform(p2d_large_reshape)
vv2d_large_scale = scaler6.transform(vv2d_large_reshape)
uu2d_large_scale = scaler7.transform(uu2d_large_reshape)

# x (ammount of input variables) columns for 3D plot, 1 for 2D --> comment second column, (k gives unrealistic results)
X_test = np.zeros((len(duidxj_test_scaled), 7))
X_test[:, 0] = duidxj_test_scaled[:, 0]
X_test[:, 1] = uv_large_scaled[:, 0]
X_test[:, 2] = k_scaled_large[:, 0]
X_test[:, 3] = eps_large_scaled[:, 0]
X_test[:, 4] = p2d_large_scaled[:, 0]
X_test[:, 5] = vv2d_large_scale[:, 0]
X_test[:, 6] = uu2d_large_scale[:, 0]

y_svr = model.predict(X_test)
y_svr = np.reshape(y_svr, (ni, nj))

intensity_large = (uu2d_large ** 2 + vv2d_large ** 2 + ww2d_large ** 2) ** 0.5

# ----------------------------------------------Calculate Error----------------------------------------------
print("Calculating error")

errorML = (np.std(y_svr.flatten() - cmy_DNS_large.flatten())) / (np.mean(y_svr ** 2)) ** 0.5
error = (np.std(0.09 - cmy_DNS_large.flatten())) / (np.mean(0.09 ** 2)) ** 0.5
errorOmega = (np.std(1 - cmy_DNS_large.flatten())) / (np.mean((1) ** 2)) ** 0.5

predictOwnCase = model.predict(X)
# errorOwnCase = (np.std(predictOwnCase.flatten() - cmy_DNS.flatten()))/(np.mean(predictOwnCase**2))**0.5

# Print error
print("Coefficient of varience med ML är", errorML)
print("Coefficient of varience med standardmodell (C_my = 0.09) är", error)
print("Coefficient of varience med standardmodell ,k-omega, (C_my = 1) är", errorOmega)
# print("Coefficient of varience in fitting case is",errorOwnCase)

# RMS ERROR
c_k_eps = []
c_k_omega = []
for i in range(len(cmy_DNS_large.flatten())):
    c_k_eps.append(0.09)
    c_k_omega.append(1)

errorRMS_ML = mean_squared_error(cmy_DNS_large.flatten(), y_svr.flatten())
errorRMS = mean_squared_error(cmy_DNS_large.flatten(), c_k_eps)
errorRMS_Omega = mean_squared_error(cmy_DNS_large.flatten(), c_k_omega)
# errorRMS_Own_Case = mean_squared_error(cmy_DNS.flatten(), y_svr.flatten())

print("RMS-felet med ML är", errorRMS_ML)
print("RMS-felet med standardmodell (C_my = 0.09) är", errorRMS)
print("RMS-felet med standardmodell ,k-omega, (C_my = 1) är", errorRMS_Omega)
# print("Error in fitting case is",errorRMS_Own_Case)

# -------------------------------------Calculate Error with Sklearn metrics-------------------------------
print("------------------------------------")
print("Errors with machine-learning model:")
print("Mean absolute error =", round(sm.mean_absolute_error(cmy_DNS_large.flatten(), y_svr.flatten()), 2))
print("Mean squared error =", round(sm.mean_squared_error(cmy_DNS_large.flatten(), y_svr.flatten()), 2))
print("Median absolute error =", round(sm.median_absolute_error(cmy_DNS_large.flatten(), y_svr.flatten()), 2))
print("Explain variance score =", round(sm.explained_variance_score(cmy_DNS_large.flatten(), y_svr.flatten()), 2))
print("R2 score =", round(sm.r2_score(cmy_DNS_large.flatten(), y_svr.flatten()), 2))
print("------------------------------------")
print("Error with standard model:")
print("Mean absolute error =",
      round(sm.mean_absolute_error(cmy_DNS_large.flatten(), [1] * len(cmy_DNS_large.flatten())), 2))
print("Mean squared error =",
      round(sm.mean_squared_error(cmy_DNS_large.flatten(), [1] * len(cmy_DNS_large.flatten())), 2))
print("Median absolute error =",
      round(sm.median_absolute_error(cmy_DNS_large.flatten(), [1] * len(cmy_DNS_large.flatten())), 2))
print("Explain variance score =",
      round(sm.explained_variance_score(cmy_DNS_large.flatten(), [1] * len(cmy_DNS_large.flatten())), 2))
print("R2 score =", round(sm.r2_score(cmy_DNS_large.flatten(), [1] * len(cmy_DNS_large.flatten())), 2))

# ----------------------------------------------Plot Solution----------------------------------------------
et = time.process_time()
print("Time elapsed: " + str(et - st))
print("Plotting")

# plt.figure("height restriction")
# plt.plot(xp2d_large[:,-163],cmy_DNS_large[:,-163])
# plt.xlabel("x [m]")
# plt.ylabel("$C_{\mu}^{k-\omega}$")
# plt.title("$C_\mu ^{k-\omega} \in [x_0,x_n]x[y_0,y_{lim}]$")
# plt.savefig("C_my_domain_restricted_height")

# plt.figure("width restriction")
# plt.plot(yp2d_large[-90,:],cmy_DNS_large[-90,:])
# plt.xlabel("y [m]")
# plt.ylabel("$C_{\mu}^{k-\omega}$")
# plt.title("$C_\mu ^{k-\omega} \in [x_{lim},x_n]x[y_0,y_n]$")
# plt.savefig("C_my_domain_restricted_width")

plt.figure("Test")
plt.scatter(yp2d_large, cmy_DNS_large, marker="o", s=10, c="green", label="Target")
plt.scatter(yp2d_large, y_svr, marker="o", s=10, c="blue", label="Prediction")
plt.ylabel("$C_{\mu}^{k-\omega}$")
plt.xlabel("y [m]")
plt.title("$C_\mu^{k-\omega}$")
plt.legend(loc="upper right", prop=dict(size=12))
plt.savefig("Modell_large_test_small_2d.png")

fig3d = plt.figure("3d-Test")
ax = plt.axes(projection='3d')
surf = ax.plot_surface(xp2d[:, -135::], yp2d[:, -135::], cmy_DNS[:, -135::], cmap=cm.coolwarm, label="Target")
# ax.scatter(xp2d_large,yp2d_large,y_svr,marker = "o", s= 10, c = "blue", label = "Prediction")
ax.set_xlabel("$x [m]$")
ax.set_ylabel("$y [m]$")
ax.set_zlabel("$C_\mu^{k-\omega}$")
plt.title("$C_\mu^{k-\omega}$ 3d plot")
fig3d.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig("Modell_large_test_small_3d.png")

fig1, ax1 = plt.subplots()
plt.subplots_adjust(left=0.20, bottom=0.20)
fig1.colorbar(plt.contourf(xp2d_large, yp2d_large, cmy_DNS_large, 1000), ax=ax1, label="$C_\mu$")
plt.axis([0, 4, -0.4, 1])
plt.title("Values of $C_\mu$ (DNS) in the area $[x_{lim},x_n] x [y_0,y_n]$")
plt.xlabel("$x [m]$")
plt.ylabel("$y [m]$")
plt.savefig("C_my_in_domain.png")

fig2, ax2 = plt.subplots()
plt.subplots_adjust(left=0.20, bottom=0.20)
fig2.colorbar(plt.contourf(xp2d_large, yp2d_large, y_svr, 1000), ax=ax2, label="$C_\mu$")
plt.axis([0, 4, -0.4, 1])
plt.title("Values of $C_\mu$ (Prediction) in the area $[x_{lim},x_n]$ x $[y_0,y_n]$")
plt.xlabel("$x [m]$")
plt.ylabel("$y [m]$")
plt.savefig("C_my_pred_in_domain_filter.png")

fig3, ax3 = plt.subplots()
plt.subplots_adjust(left=0.20, bottom=0.20)
fig3.colorbar(plt.contourf(xp2d_large, yp2d_large, intensity_large, 1000), ax=ax3, label="$\partial v /\partial y$")
plt.axis([0, 4, -0.4, 1])
plt.title("Values of $dvdy$ (DNS) in the area $[x_0,x_n]$ x $[y_0,y_n]$")
plt.xlabel("$x [m]$")
plt.ylabel("$y [m]$")
plt.savefig("dvdy_in_domain_filter.png")

plt.show()
