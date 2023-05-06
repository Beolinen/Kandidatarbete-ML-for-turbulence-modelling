# ----------------------------------------------Import Packages------------------------------------------------------
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR  # for building SVR model
from gradients import compute_face_phi, dphidx, dphidy, init
import sklearn.metrics as sm

# read data file
st = time.process_time()
tec = np.genfromtxt("large_wave/tec.dat", dtype=None, comments="%")

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
xc_yc = np.loadtxt("large_wave/mesh.dat")
xf = xc_yc[:, 0]
yf = xc_yc[:, 1]
xf2d = np.reshape(xf, (nj - 1, ni - 1))
yf2d = np.reshape(yf, (nj - 1, ni - 1))
xf2d = np.transpose(xf2d)
yf2d = np.transpose(yf2d)

# compute cell centers
xp2d = 0.25 * (xf2d[0:-1, 0:-1] + xf2d[0:-1, 1:] + xf2d[1:, 0:-1] + xf2d[1:, 1:])  # Borde vara yf2d och xf2d
yp2d = 0.25 * (yf2d[0:-1, 0:-1] + yf2d[0:-1, 1:] + yf2d[1:, 0:-1] + yf2d[1:, 1:])  # Borde vara yf2d och xf2d

# compute geometric quantities
areaw, areawx, areawy, areas, areasx, areasy, vol, fx, fy = init(xf2d, yf2d, xp2d, yp2d)

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
vist_DNS = abs(uv2d) / dudy
omega_DNS = (eps_DNS2d / 0.09 / vist_DNS) ** 0.5
cmy_DNS = abs(uv2d) / (k2d * dudy) * omega_DNS
cmy_DNS = np.nan_to_num(cmy_DNS, posinf=1, neginf=1)

# omega = eps_DNS2d / k2d / 0.09
# cmy_DNS = np.array(-uv2d / (k2d * (dudy + dvdx)) * omega)
cmy_DNS[:, 0] = 1
duidxj = np.array((dudx ** 2 + 0.5 * (dudy ** 2 + 2 * dudy * dvdx + dvdx ** 2) + dvdy ** 2) ** 0.5)
filterx = duidxj > 15


cmy_DNSS = cmy_DNS[filterx]
k2d = k2d[filterx]
uv2d = uv2d[filterx]
eps_DNS2d = eps_DNS2d[filterx]
p2d = p2d[filterx]
vv2d = vv2d[filterx]
uu2d = uu2d[filterx]
duidxj = duidxj[filterx]

# Scaler
scaler_duidxj = StandardScaler()  # I ML CHANNEL HAR ALLA INPUTS EGNA SCALERS
scaler_k = StandardScaler()
scaler_uv = StandardScaler()
scaler_eps = StandardScaler()
scaler_p2d = StandardScaler()
scaler_vv = StandardScaler()
scaler_uu = StandardScaler()

# Reshape Data
duidxj = duidxj.reshape(-1, 1)
k_scaled_reshaped = k2d.reshape(-1, 1)
uv_scaled_reshaped = uv2d.reshape(-1, 1)
eps_scaled_reshaped = eps_DNS2d.reshape(-1, 1)
p2d_scaled_reshaped = p2d.reshape(-1, 1)
vv2d_scaled_reshaped = vv2d.reshape(-1, 1)
uu2d_scaled_reshaped = uu2d.reshape(-1, 1)

# Transform
duidxj_scaled = scaler_duidxj.fit_transform(duidxj)
k_scaled = scaler_k.fit_transform(k_scaled_reshaped)
uv_scaled = scaler_uv.fit_transform(uv_scaled_reshaped)
eps_scaled = scaler_eps.fit_transform(eps_scaled_reshaped)
p2d_scaled = scaler_p2d.fit_transform(p2d_scaled_reshaped)
vv2d_scaled = scaler_vv.fit_transform(vv2d_scaled_reshaped)
uu2d_scaled = scaler_uu.fit_transform(uu2d_scaled_reshaped)

X = np.zeros((len(duidxj_scaled), 7))
X[:, 0] = duidxj_scaled[:, 0]  # Origanal training data
X[:, 1] = uv_scaled[:, 0]  # Original training data
X[:, 2] = k_scaled[:, 0]
X[:, 3] = eps_scaled[:, 0]
X[:, 4] = p2d_scaled[:, 0]
X[:, 5] = vv2d_scaled[:, 0]
X[:, 6] = uu2d_scaled[:, 0]

Y = cmy_DNSS

# Choose model

model = SVR(kernel='rbf', C=100, epsilon=0.01)
SVR = model.fit(X, Y.flatten())

print("Reading test case")
# ----------------------------------------------Test With New Case------------------------------------------------
# ----------------------------------------------Read Data Large Case----------------------------------------------
tec_2 = np.genfromtxt("small_wave/tec.dat", dtype=None, comments="%")

u_2 = tec_2[:, 3]
v_2 = tec_2[:, 4]
y_2 = tec_2[:, 1]
x_2 = tec_2[:, 0]
uv_2 = tec_2[:, 8]
uu_2 = tec_2[:, 5]
vv_2 = tec_2[:, 6]
ww_2 = tec_2[:, 7]
k_DNS_2 = 0.5 * (uu_2 + vv_2 + ww_2)
eps_DNS_2 = tec_2[:, 9]
p_2 = tec_2[:, 2]

if max(y_2) == 1.:
    ni = 170
    nj = 194
    nu = 1. / 10000.
else:
    nu = 1. / 10595.
    if max(x_2) > 8.:
        nj = 162
        ni = 162
    else:
        ni = 402
        nj = 162

viscous_2 = nu

u2d_2 = np.transpose(np.reshape(u_2, (nj, ni)))
v2d_2 = np.transpose(np.reshape(v_2, (nj, ni)))
uv2d_2 = np.transpose(np.reshape(uv_2, (nj, ni)))
k_DNS2d = np.transpose(np.reshape(k_DNS_2, (nj, ni)))
eps_DNS2d_2 = np.transpose(np.reshape(eps_DNS_2, (nj, ni)))
x2d_2 = np.transpose(np.reshape(x_2, (nj, ni)))
y2d_2 = np.transpose(np.reshape(y_2, (nj, ni)))
p2d_2 = np.transpose(np.reshape(p_2, (nj, ni)))
uu2d_2 = np.transpose(np.reshape(uu_2, (nj, ni)))
vv2d_2 = np.transpose(np.reshape(vv_2, (nj, ni)))
ww2d_2 = np.transpose(np.reshape(ww_2, (nj, ni)))

# set Neumann of p at upper and lower boundaries
p2d_2[:, 1] = p2d_2[:, 2]
p2d_2[:, -1] = p2d_2[:, -1 - 1]

# set periodic b.c on west boundary
u2d_2[0, :] = u2d_2[-1, :]
v2d_2[0, :] = v2d_2[-1, :]
p2d_2[0, :] = p2d_2[-1, :]
uu2d_2[0, :] = uu2d_2[-1, :]

xc_yc_2 = np.loadtxt("small_wave/mesh.dat")
xf_2 = xc_yc_2[:, 0]
yf_2 = xc_yc_2[:, 1]
xf2d_2 = np.reshape(xf_2, (nj - 1, ni - 1))
yf2d_2 = np.reshape(yf_2, (nj - 1, ni - 1))
xf2d_2 = np.transpose(xf2d_2)
yf2d_2 = np.transpose(yf2d_2)

# compute cell centers
xp2d_2 = 0.25 * (xf2d_2[0:-1, 0:-1] + xf2d_2[0:-1, 1:] + xf2d_2[1:, 0:-1] + xf2d_2[1:, 1:])
yp2d_2 = 0.25 * (yf2d_2[0:-1, 0:-1] + yf2d_2[0:-1, 1:] + yf2d_2[1:, 0:-1] + yf2d_2[1:, 1:])

# compute geometric quantities
areaw, areawx, areawy, areas, areasx, areasy, vol, fx, fy = init(xf2d_2, yf2d_2, xp2d_2, yp2d_2)

# delete last row
u2d_2 = np.delete(u2d_2, -1, 0)
v2d_2 = np.delete(v2d_2, -1, 0)
p2d_2 = np.delete(p2d_2, -1, 0)
k_DNS2d = np.delete(k_DNS2d, -1, 0)
uu2d_2 = np.delete(uu2d_2, -1, 0)
vv2d_2 = np.delete(vv2d_2, -1, 0)
ww2d_2 = np.delete(ww2d_2, -1, 0)
uv2d_2 = np.delete(uv2d_2, -1, 0)
eps_DNS2d_2 = np.delete(eps_DNS2d_2, -1, 0)

# delete first row
u2d_2 = np.delete(u2d_2, 0, 0)
v2d_2 = np.delete(v2d_2, 0, 0)
p2d_2 = np.delete(p2d_2, 0, 0)
k_DNS2d = np.delete(k_DNS2d, 0, 0)
uu2d_2 = np.delete(uu2d_2, 0, 0)
vv2d_2 = np.delete(vv2d_2, 0, 0)
ww2d_2 = np.delete(ww2d_2, 0, 0)
uv2d_2 = np.delete(uv2d_2, 0, 0)
eps_DNS2d_2 = np.delete(eps_DNS2d_2, 0, 0)

# delete last columns
u2d_2 = np.delete(u2d_2, -1, 1)
v2d_2 = np.delete(v2d_2, -1, 1)
p2d_2 = np.delete(p2d_2, -1, 1)
k_DNS2d = np.delete(k_DNS2d, -1, 1)
uu2d_2 = np.delete(uu2d_2, -1, 1)
vv2d_2 = np.delete(vv2d_2, -1, 1)
ww2d_2 = np.delete(ww2d_2, -1, 1)
uv2d_2 = np.delete(uv2d_2, -1, 1)
eps_DNS2d_2 = np.delete(eps_DNS2d_2, -1, 1)

# delete first columns
u2d_2 = np.delete(u2d_2, 0, 1)
v2d_2 = np.delete(v2d_2, 0, 1)
p2d_2 = np.delete(p2d_2, 0, 1)
k_DNS2d = np.delete(k_DNS2d, 0, 1)
uu2d_2 = np.delete(uu2d_2, 0, 1)
vv2d_2 = np.delete(vv2d_2, 0, 1)
ww2d_2 = np.delete(ww2d_2, 0, 1)
uv2d_2 = np.delete(uv2d_2, 0, 1)
eps_DNS2d_2 = np.delete(eps_DNS2d_2, 0, 1)

ni = ni - 2
nj = nj - 2

# eps at last cell upper cell wrong. fix it.
eps_DNS2d_2[:, -1] = eps_DNS2d_2[:, -2]

# compute face value of U and V
u2d_face_w, u2d_face_s = compute_face_phi(u2d_2, fx, fy, ni, nj)  # Error with dimensions for u2d vs fx,fy
v2d_face_w, v2d_face_s = compute_face_phi(v2d_2, fx, fy, ni, nj)

# x derivatives
dudx_2 = dphidx(u2d_face_w, u2d_face_s, areawx, areasx, vol)
dvdx_2 = dphidx(v2d_face_w, v2d_face_s, areawx, areasx, vol)

# y derivatives
dudy_2 = dphidy(u2d_face_w, u2d_face_s, areawy, areasy, vol)
dvdy_2 = dphidy(v2d_face_w, v2d_face_s, areawy, areasy, vol)

print("Starting ML on new case")
# ----------------------------------------------ML-Method Large Case----------------------------------------------
# Calculate correct C_my for prediction
vist_DNS_2 = abs(uv2d_2) / dudy_2
omega_DNS_2 = (eps_DNS2d_2 / 0.09 / vist_DNS_2) ** 0.5
cmy_DNS_2 = abs(uv2d_2) / (k_DNS2d * dudy_2) * omega_DNS_2
cmy_DNS_2 = np.nan_to_num(cmy_DNS_2, posinf=1, neginf=1)

# omega_2 = eps_DNS2d_2 / k_DNS2d / 0.09
# cmy_DNS_2 = np.array(-uv2d_2 / (k_DNS2d * (dudy_2 + dvdx_2)) * omega_2)
cmy_DNS_2[:, 0] = 1

# np.array is used to convert the list to an array
duidxj_test = np.array((dudx_2 ** 2 + 0.5 * (
        dudy_2 ** 2 + 2 * dudy_2 * dvdx_2 + dvdx_2 ** 2) + dvdy_2 ** 2) ** 0.5)

# Reshape data
duidxj_test_reshape = duidxj_test.reshape(-1, 1)
uv_2_reshape = uv2d_2.reshape(-1, 1)
k_scaled_reshape = k_DNS2d.reshape(-1, 1)
eps_2_reshape = eps_DNS2d_2.reshape(-1, 1)
p2d_2_reshape = p2d_2.reshape(-1, 1)
vv2d_2_reshape = vv2d_2.reshape(-1, 1)
uu2d_2_reshape = uu2d_2.reshape(-1, 1)

# Scale data
duidxj_test_scaled = scaler_duidxj.transform(duidxj_test_reshape)
k_scaled_2 = scaler_k.transform(k_scaled_reshape)
uv_2_scaled = scaler_uv.transform(uv_2_reshape)
eps_2_scaled = scaler_eps.transform(eps_2_reshape)
p2d_2_scaled = scaler_p2d.transform(p2d_2_reshape)
vv2d_2_scale = scaler_vv.transform(vv2d_2_reshape)
uu2d_2_scale = scaler_uu.transform(uu2d_2_reshape)

# x (ammount of input variables)
X_test = np.zeros((len(duidxj_test_scaled), 7))
X_test[:, 0] = duidxj_test_scaled[:, 0]
X_test[:, 1] = uv_2_scaled[:, 0]
X_test[:, 2] = k_scaled_2[:, 0]
X_test[:, 3] = eps_2_scaled[:, 0]
X_test[:, 4] = p2d_2_scaled[:, 0]
X_test[:, 5] = vv2d_2_scale[:, 0]
X_test[:, 6] = uu2d_2_scale[:, 0]

y_svr = SVR.predict(X_test)
y_svr = np.reshape(y_svr, (ni, nj))


X_test_no_scale = scaler_duidxj.inverse_transform(X_test)

print("Calculating error")
# ----------------------------------------------Calculate Error----------------------------------------------
errorML = (np.std(y_svr - cmy_DNS_2)) / (np.mean(y_svr ** 2)) ** 0.5
errorOmega = (np.std(1 - cmy_DNS_2.flatten())) / (np.mean(1 ** 2)) ** 0.5

predictOwnCase = SVR.predict(X)
errorOwnCase = (np.std(predictOwnCase.flatten() - cmy_DNSS.flatten())) / (np.mean(predictOwnCase ** 2)) ** 0.5

# Print error
print("------------------------------------")
print("Coefficient of varience med ML är", errorML)
print("Coefficient of varience med standardmodell ,k-omega, (C_my = 1) är", errorOmega)
print("Coefficient of varience in fitting case is", errorOwnCase)
print("------------------------------------")
# ----------------------------------------------Plot Cmy in domain----------------------------------------------
jet = plt.get_cmap("jet")

fig1, ax1 = plt.subplots()
plt.subplots_adjust(left=0.20, bottom=0.20)
fig1.colorbar(plt.contourf(xp2d, yp2d, cmy_DNS, levels=1000, cmap=jet), ax=ax1, label="$C_\mu$")
plt.axis([0, 3.5, -0.4, 1])
plt.title("Values of $C_\mu$ (DNS large) in the area $[x_0,x_n] x [y_0,y_n]$")
plt.xlabel("$x [m]$")
plt.ylabel("$y [m]$")

# plot the
fig2, ax2 = plt.subplots()
plt.subplots_adjust(left=0.20, bottom=0.20)
fig2.colorbar(plt.contourf(xp2d_2, yp2d_2, cmy_DNS_2, levels=1000, cmap=jet), ax=ax2, label="$C_\mu$")
plt.axis([0, 3.5, -0.4, 1])
plt.title("Values of $C_\mu$ (DNS small) in the area $[x_0,x_n]$ x $[y_0,y_n]$")
plt.xlabel("$x [m]$")
plt.ylabel("$y [m]$")

fig3, ax3 = plt.subplots()
plt.subplots_adjust(left=0.20, bottom=0.20)
fig3.colorbar(plt.contourf(xp2d_2, yp2d_2, y_svr, levels=1000, cmap=jet), ax=ax3, label="$C_\mu$")
plt.axis([0, 3.5, -0.4, 1])
plt.title("Values of $C_\mu$ (Prediction) in the area $[x_0,x_n]$ x $[y_0,y_n]$")
plt.xlabel("$x [m]$")
plt.ylabel("$y [m]$")

plt.show()
