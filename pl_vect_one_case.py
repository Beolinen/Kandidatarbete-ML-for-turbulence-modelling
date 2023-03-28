# ----------------------------------------------Import Packages------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR  # for building SVR model
from gradients import compute_face_phi, dphidx, dphidy, init
import time
import warnings
import matplotlib.cbook
# -------------------------------------------------start-------------------------------------------------
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

# read data file
st = time.process_time()
tec = np.genfromtxt("large_wave/tec_large.dat", dtype=None, comments="%")

print("Starting script")
# text='VARIABLES = X Y P U V u2 v2 w2 uv eps'
# Define variables from text-file
x = tec[:, 0]  # Cell center x
y = tec[:, 1]  # Cell center y
p = tec[:, 2]  # Pressure
u = tec[:, 3]  # Instant/laminar velocity in x
v = tec[:, 4]  # Instant/laminar velocity in y
uu = tec[:, 5]  # Stress
vv = tec[:, 6]  # Stress
ww = tec[:, 7]  # Stress
uv = tec[:, 8]  # Stress
k_DNS = 0.5 * (uu + vv + ww)  # Turbulent kinetic energy (?)
eps_DNS = tec[:, 9]  # Dissipation

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

# set periodic b.c on west boundary
# SHOULD WE USE THIS?
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

ni = ni - 2
nj = nj - 2

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

print("Data read")
print("Starting ML")
# ----------------------------------------------Project-Group Work---------------------------------------------------
# ----------------------------------------------ML-Method Original Case----------------------------------------------
omega = eps_DNS2d / k2d / 0.09

# Compute C_my and ||duidxj|| to train model
cmy_DNS = np.array(-uv2d / (k2d * (dudy + dvdx)) * omega)
cmy_DNS = cmy_DNS.flatten()

duidxj = np.array((dudx ** 2 + 0.5 * (dudy ** 2 + 2 * dudy * dvdx + dvdx ** 2) + dvdy ** 2) ** 0.5)
duidxj = duidxj.flatten()


# create indices for all data
index = np.arange(0, len(cmy_DNS.flatten()), dtype=int)

# number of elements of test data, 20%
n_test = int(0.2 * len(cmy_DNS))

# the rest is for training data
n_svr = len(cmy_DNS) - n_test

# pick 20% elements randomly (test data)
index_test = np.random.choice(index, size=n_test, replace=False)
cmy_test = cmy_DNS[index_test]
duidxj_test = duidxj[index_test]

# delete testing data from 'all data' => training data
cmy_training = np.delete(cmy_DNS, index_test)
duidxj_training = np.delete(duidxj, index_test)

# Reshape training data
duidxj_training = duidxj_training.reshape(-1, 1)

# Scale training data
scaler = StandardScaler()
duidxj_training_scaled = scaler.fit_transform(duidxj_training)

X = np.zeros((len(cmy_training), 1))
y = cmy_training
X[:, 0] = duidxj_training_scaled[:, 0]

print('starting SVR')

# choose Machine Learning model
C = 0.1
eps = 0.001
model = SVR(epsilon=eps, C=C)

# Fit the model
svr = model.fit(X, y.flatten())

# Reshape test data
cmy_test = cmy_test.reshape(-1, 1)
duidxj_test = duidxj_test.reshape(-1, 1)

# Scale test data
scaler2 = StandardScaler()
duidxj_test_scaled = scaler2.fit_transform(duidxj_test)

X_test = np.zeros((len(duidxj_test), 1))
X_test[:, 0] = duidxj_test_scaled[:, 0]

cmy_predict = model.predict(X_test)

duidxj_training_no_scale = scaler2.inverse_transform(duidxj_test_scaled)

errorML = (np.std(cmy_predict.flatten() - cmy_test.flatten())) / (np.mean(cmy_predict ** 2)) ** 0.5
print('\nRMS error using ML turbulence model', errorML)