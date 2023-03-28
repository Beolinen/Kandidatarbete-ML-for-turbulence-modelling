

# ----------------------------------------------Import Packages------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR  # for building SVR model
import sklearn.metrics as sm
from gradients import compute_face_phi, dphidx, dphidy, init
import time
import sys
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

# ----------------------------------------------Read Data Original Case----------------------------------------------

def dat_to_variable_arrays(path:str):
    tec = np.genfromtxt(path, dtype=None, comments="%").transpose()
    k_DNS = 0.5 * (tec[5] + tec[6] + tec[7])  # Turbulent kinetic energy (?)

    nu,ni,nj = get_ni_nj(tec[1],tec[0])

    k_DNS = np.reshape(k_DNS, (nj, ni)).transpose() #k_DNS has the shape [ni,nj]
    tec = tec.reshape(10,nj,ni)                     #tec is reshaped so every variable has the shape [nj,ni]
    tec = np.transpose(tec,axes=[0,2,1])            #tec is transposed

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
  
def get_ni_nj(y:np.array, x:np.array):
    if max(y) == 1.:
        ni,nj = 170,194
        nu = 1. / 10000.
    else:
        nu = 1. / 10595.
        if max(x) > 8.:
            ni,nj = 162,162
        else:
            ni,nj = 402,162
    return nu,ni,nj    

# read data file
st = time.process_time()


print("Starting script")

#get variables from .dat
x,y,p,u,v,uu,vv,ww,uv,eps_DNS,k_DNS,ni,nj = dat_to_variable_arrays("large_wave/tec_large.dat")


# set Neumann of p at upper and lower boundaries
p[:, 1]  = p[:, 2]
p[:, -1] = p[:, -1 - 1]

# set periodic b.c on west boundary
# SHOULD WE USE THIS?
u[0, :]  = u[-1, :]
v[0, :]  = v[-1, :]
p[0,]    = p[-1, :]
uu[0, :] = uu[-1, :]

# x and y are fo the cell centers. The dphidx_dy routine needs the face coordinate, xf2d, yf2d
# load them
xc_yc = np.loadtxt("large_wave/mesh_large.dat").transpose()
xf = np.reshape(xc_yc[0], (nj-1, ni-1)).transpose()
yf = np.reshape(xc_yc[1], (nj-1, ni-1)).transpose()

# compute cell centers
xp = 0.25 * (x[0:-1, 0:-1] + x[0:-1, 1:] + x[1:, 0:-1] + x[1:, 1:])
yp = 0.25 * (y[0:-1, 0:-1] + y[0:-1, 1:] + y[1:, 0:-1] + y[1:, 1:])

# delete last row
x = np.delete(x, -1, 0)
y = np.delete(y, -1, 0)
xp = np.delete(xp, -1, 0)
yp = np.delete(yp, -1, 0)

# delete last columns
x = np.delete(x, -1, 1)
y = np.delete(y, -1, 1)
xp = np.delete(xp, -1, 1)
yp = np.delete(yp, -1, 1)

# compute geometric quantities
areaw, areawx, areawy, areas, areasx, areasy, vol, fx, fy = init(x, y, xp, yp)

# delete first/last row/col
u = np.delete(np.delete(u, [0,-1], 1), [0,-1], 0)
v = np.delete(np.delete(v, [0,-1], 1), [0,-1], 0)
p = np.delete(np.delete(p, [0,-1], 1), [0,-1], 0)
uu = np.delete(np.delete(uu, [0,-1], 1), [0,-1], 0)
vv = np.delete(np.delete(vv, [0,-1], 1), [0,-1], 0)
ww = np.delete(np.delete(ww, [0,-1], 1), [0,-1], 0)
uv = np.delete(np.delete(uv, [0,-1], 1), [0,-1], 0)
eps_DNS = np.delete(np.delete(eps_DNS, [0,-1], 1), [0,-1], 0)
k_DNS = np.delete(np.delete(k_DNS, [0,-1], 1), [0,-1], 0)

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

print("Data read")
print("Starting ML")
# ----------------------------------------------Project-Group Work---------------------------------------------------
# ----------------------------------------------ML-Method Original Case----------------------------------------------
omega = eps_DNS / k_DNS / 0.09

# Compute C_my and ||duidxj|| to train model
cmy_DNS = np.array(-uv / (k_DNS * (dudy + dvdx)) * omega)
cmy_DNS = np.where(abs(dudy + dvdx) < 1, 1, cmy_DNS)
cmy_DNS = np.where(cmy_DNS > 0, cmy_DNS, 1)
cmy_DNS = np.where(cmy_DNS <= 3, cmy_DNS, 1)

duidxj = np.array((dudx ** 2 + 0.5 * (dudy ** 2 + 2 * dudy * dvdx + dvdx ** 2) + dvdy ** 2) ** 0.5)

# ML-metod
scaler = StandardScaler()

# Reshape Data
duidxj = duidxj.reshape(-1, 1)
k_scaled_reshaped = k_DNS.reshape(-1, 1)
uv_scaled_reshaped = uv.reshape(-1, 1)
eps_scaled_reshaped = eps_DNS.reshape(-1, 1)
p2d_scaled_reshaped = p.reshape(-1, 1)
vv2d_scaled_reshaped = vv.reshape(-1, 1)
uu2d_scaled_reshaped = uu.reshape(-1, 1)

# scale data
duidxj_scaled = scaler.fit_transform(duidxj)
k_scaled = scaler.fit_transform(k_scaled_reshaped)
uv_scaled = scaler.fit_transform(uv_scaled_reshaped)
eps_scaled = scaler.fit_transform(eps_scaled_reshaped)
p2d_scaled = scaler.fit_transform(p2d_scaled_reshaped)
vv2d_scaled = scaler.fit_transform(vv2d_scaled_reshaped)
uu2d_scaled = scaler.fit_transform(uu2d_scaled_reshaped)

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

# Choose model
model = SVR(kernel='rbf', C=1, epsilon=0.001)
SVR = model.fit(X, Y.flatten())

print("Reading new case")
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

nu,ni,nj = get_ni_nj(y_2,x_2)

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
p2d_2[0,] = p2d_2[-1, :]
uu2d_2[0, :] = uu2d_2[-1, :]

xc_yc_2 = np.loadtxt("small_wave/mesh.dat")
xf_2 = xc_yc_2[:, 0]
yf_2 = xc_yc_2[:, 1]
xf2d_2 = np.reshape(xf_2, (nj - 1, ni - 1))
yf2d_2 = np.reshape(yf_2, (nj - 1, ni - 1))
xf2d_2 = np.transpose(xf2d_2)
yf2d_2 = np.transpose(yf2d_2)

# compute cell centers
xp2d_2 = 0.25 * (x2d_2[0:-1, 0:-1] + x2d_2[0:-1, 1:] + x2d_2[1:, 0:-1] + x2d_2[1:, 1:])
yp2d_2 = 0.25 * (y2d_2[0:-1, 0:-1] + y2d_2[0:-1, 1:] + y2d_2[1:, 0:-1] + y2d_2[1:, 1:])

# delete last row
x2d_2 = np.delete(x2d_2, -1, 0)
y2d_2 = np.delete(y2d_2, -1, 0)
xp2d_2 = np.delete(xp2d_2, -1, 0)
yp2d_2 = np.delete(yp2d_2, -1, 0)

# delete last columns
x2d_2 = np.delete(x2d_2, -1, 1)
y2d_2 = np.delete(y2d_2, -1, 1)
xp2d_2 = np.delete(xp2d_2, -1, 1)
yp2d_2 = np.delete(yp2d_2, -1, 1)

# compute geometric quantities
areaw, areawx, areawy, areas, areasx, areasy, vol, fx, fy = init(x2d_2, y2d_2, xp2d_2, yp2d_2)

# delete first/last row/col
u2d_2 = np.delete(np.delete(u2d_2, [0,-1], 1), [0,-1], 0)
v2d_2 = np.delete(np.delete(v2d_2, [0,-1], 1), [0,-1], 0)
p2d_2 = np.delete(np.delete(p2d_2, [0,-1], 1), [0,-1], 0)
k_DNS2d = np.delete(np.delete(k_DNS2d, [0,-1], 1), [0,-1], 0)
uu2d_2 = np.delete(np.delete(uu2d_2, [0,-1], 1), [0,-1], 0)
vv2d_2 = np.delete(np.delete(vv2d_2, [0,-1], 1), [0,-1], 0)
ww2d_2 = np.delete(np.delete(ww2d_2, [0,-1], 1), [0,-1], 0)
uv2d_2 = np.delete(np.delete(uv2d_2, [0,-1], 1), [0,-1], 0)
eps_DNS2d_2 = np.delete(np.delete(eps_DNS2d_2, [0,-1], 1), [0,-1], 0)


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

print("Starting ML new case")
# ----------------------------------------------ML-Method Large Case----------------------------------------------
'''Reshape is used to fit arrays to the dimensions of the data.
   scaler.fit_transform is used to scale the data to a range between 0 and 1.'''

# Calculate correct C_my for prediction
omega_2 = eps_DNS2d_2 / k_DNS2d / 0.09

cmy_DNS_2 = np.array(-uv2d_2 / (k_DNS2d * (dudy_2 + dvdx_2)) * omega_2)
cmy_DNS_large = np.where(abs(dudy_2 + dvdx_2) < 1, 1, cmy_DNS_2)
cmy_DNS_2 = np.where(cmy_DNS_2 > 0, cmy_DNS_2, 1)
cmy_DNS_2 = np.where(cmy_DNS_2 <= 3, cmy_DNS_2, 1)

# plt.figure()
# plt.plot(yp2d_2[-1,:],uv2d_2[-1,:], 'o')
# plt.figure()
# plt.plot(yp2d_2[-1,:],k_DNS2d[-1,:], 'b-')
# plt.figure()
# plt.plot(yp2d_2[-1,:],omega_2[-1,:], 'b--')
# plt.figure()
# plt.plot(yp2d_2[-1,:],dudy_2[-1,:], 'r-')
# plt.figure()
# plt.plot(yp2d_2[-1,:],dvdx_2[-1,:], 'k-')
#
# plt.figure()
# plt.plot(yp2d_2[-100,:],cmy_DNS_2[-100,:], 'ro') #rad -100 ger ickefysikaliska värden
# plt.show()

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
duidxj_test_scaled = scaler.fit_transform(duidxj_test_reshape)
k_scaled_2 = scaler.fit_transform(k_scaled_reshape)
uv_2_scaled = scaler.fit_transform(uv_2_reshape)
eps_2_scaled = scaler.fit_transform(eps_2_reshape)
p2d_2_scaled = scaler.fit_transform(p2d_2_reshape)
vv2d_2_scale = scaler.fit_transform(vv2d_2_reshape)
uu2d_2_scale = scaler.fit_transform(uu2d_2_reshape)

# x (ammount of input variables) columns for 3D plot, 1 for 2D --> comment second column, (k gives unrealistic results)
X_test = np.zeros((len(duidxj_test_scaled), 7))
X_test[:, 0] = duidxj_test_scaled[:, 0]
X_test[:, 1] = uv_2_scaled[:, 0]
X_test[:, 2] = k_scaled_2[:, 0]
X_test[:, 3] = eps_2_scaled[:, 0]
X_test[:, 4] = p2d_2_scaled[:, 0]
X_test[:, 5] = vv2d_2_scale[:, 0]
X_test[:, 6] = uu2d_2_scale[:, 0]

y_svr = model.predict(X_test)

X_test_no_scale = scaler.inverse_transform(X_test)

print("Calculating error")
# ----------------------------------------------Calculate Error----------------------------------------------
errorML = (np.std(y_svr.flatten() - cmy_DNS_2.flatten())) / (np.mean(y_svr ** 2)) ** 0.5
error = (np.std(0.09 - cmy_DNS_2.flatten())) / (np.mean(0.09 ** 2)) ** 0.5
errorOmega = (np.std(1 - cmy_DNS_2.flatten())) / (np.mean(1 ** 2)) ** 0.5

predictOwnCase = model.predict(X)
errorOwnCase = (np.std(predictOwnCase.flatten() - cmy_DNS.flatten())) / (np.mean(predictOwnCase ** 2)) ** 0.5

# Print error
print("Coefficient of varience med ML är", errorML)
print("Coefficient of varience med standardmodell (C_my = 0.09) är", error)
print("Coefficient of varience med standardmodell ,k-omega, (C_my = 1) är", errorOmega)
print("Coefficient of varience in fitting case is", errorOwnCase)

# RMS ERROR
c_k_eps = [0.09] * len(cmy_DNS_2.flatten())
c_k_omega = [1] * len(cmy_DNS_2.flatten())

# -------------------------------------Calculate Error with Sklearn metrics--------------------------------
# Mean absolute error: This is the average of absolute errors of all the data points in the given dataset.

# Mean squared error: This is the average of the squares of the errors of all the data points in the given dataset.
# It is one of the most popular metrics out there!

# Median absolute error: This is the median of all the errors in the given dataset.
# The main advantage of this metric is that it's robust to outliers.
# A single bad point in the test dataset wouldn't skew the entire error metric, as opposed to a mean error metric.

# Explained variance score: This score measures how well our model can account for the variation in our dataset.
# A score of 1.0 indicates that our model is perfect.

# R2 score: This is pronounced as R-squared, and this score refers to the coefficient of determination.
# This tells us how well the unknown samples will be predicted by our model.
# The best possible score is 1.0, but the score can be negative as well.
print("Errors with machine-learning model:")
print("Mean absolute error =", round(sm.mean_absolute_error(cmy_DNS_2.flatten(), y_svr.flatten()), 2))
print("Mean squared error =", round(sm.mean_squared_error(cmy_DNS_2.flatten(), y_svr.flatten()), 2))
print("Median absolute error =", round(sm.median_absolute_error(cmy_DNS_2.flatten(), y_svr.flatten()), 2))
print("Explain variance score =", round(sm.explained_variance_score(cmy_DNS_2.flatten(), y_svr.flatten()), 2))
print("R2 score =", round(sm.r2_score(cmy_DNS_2.flatten(), y_svr.flatten()), 2))

print("Error with standard model:")
print("Mean absolute error =",
      round(sm.mean_absolute_error(cmy_DNS_2.flatten(), [1] * len(cmy_DNS_2.flatten())), 2))
print("Mean squared error =",
      round(sm.mean_squared_error(cmy_DNS_2.flatten(), [1] * len(cmy_DNS_2.flatten())), 2))
print("Median absolute error =",
      round(sm.median_absolute_error(cmy_DNS_2.flatten(), [1] * len(cmy_DNS_2.flatten())), 2))
print("Explain variance score =",
      round(sm.explained_variance_score(cmy_DNS_2.flatten(), [1] * len(cmy_DNS_2.flatten())), 2))
print("R2 score =", round(sm.r2_score(cmy_DNS_2.flatten(), [1] * len(cmy_DNS_2.flatten())), 2))
et = time.process_time()
print("Time elapsed: " + str(et - st))
print("Plotting")

# ----------------------------------------------Plot Solution----------------------------------------------
plt.figure("Test")

plt.scatter(X_test_no_scale[:, 0], cmy_DNS_2, marker="o", s=10, c="green", label="Target")
plt.scatter(X_test_no_scale[:, 0], y_svr, marker="o", s=10, c="blue", label="Prediction")
plt.xlabel("$||S_{ij}||$")
plt.ylabel("$C_{\mu}^{k-\omega}$")
plt.title("$C_\mu^{k-\omega} = f( ||S_{i j}||,uv)$")

plt.legend(loc="upper right", prop=dict(size=12))
plt.savefig("pictures/Modell_large_test_small_S_ij_model.png")

# ----------------------------------------------Plot Cmy in domain----------------------------------------------
# Fix dimensions of x and y, by deleting first row and column or last row and column

x = np.delete(x, -1, 0)
# x2d = np.delete(x2d,  0,0)
x = np.delete(x, -1, 1)
# x2d = np.delete(x2d,  0,1)

y = np.delete(y, -1, 0)
# y2d = np.delete(y2d,  0,0)
y = np.delete(y, -1, 1)
# y2d = np.delete(y2d,  0,1)

# SHOULD PLOTS USE x2d or xp2d, y2d or xp2d (large/small)
# plot the
fig1, ax1 = plt.subplots()
plt.subplots_adjust(left=0.20, bottom=0.20)
fig1.colorbar(plt.contourf(xp, yp, cmy_DNS, 1000, cmap=plt.get_cmap("plasma")), ax=ax1, label="$C_\mu$")
plt.axis([0, 3.5, -0.4, 1])
plt.title("Values of $C_\mu$ (DNS large) in the area $[x_0,x_n] x [y_0,y_n]$")
plt.xlabel("$x [m]$")
plt.ylabel("$y [m]$")
plt.savefig("pictures/C_my_in_domain.png")

# plot the 
y_svr = np.reshape(y_svr, (ni, nj))

fig2, ax2 = plt.subplots()
plt.subplots_adjust(left=0.20, bottom=0.20)
fig2.colorbar(plt.contourf(xp2d_2, yp2d_2, cmy_DNS_2, 1000, cmap=plt.get_cmap("plasma")), ax=ax2,
              label="$C_\mu$")  # y_svr
plt.axis([0, 3.5, -0.4, 1])
plt.title("Values of $C_\mu$ (DNS small) in the area $[x_0,x_n]$ x $[y_0,y_n]$")
plt.xlabel("$x [m]$")
plt.ylabel("$y [m]$")
plt.savefig("pictures/C_my_pred_in_domain.png")

fig3, ax3 = plt.subplots()
plt.subplots_adjust(left=0.20, bottom=0.20)
fig3.colorbar(plt.contourf(xp2d_2, yp2d_2, abs(dudy_2 + dvdx_2), 1000, cmap=plt.get_cmap("plasma")), ax=ax3,
              label="$||S_{ij}||$")
plt.axis([0, 3.5, -0.4, 1])
plt.title("Values of $||S_{ij}||$ (DNS) in the area $[x_0,x_n]$ x $[y_0,y_n]$")
plt.xlabel("$x [m]$")
plt.ylabel("$y [m]$")
plt.savefig("pictures/S_ij_in_domain.png")

fig4, ax4 = plt.subplots()
plt.subplots_adjust(left=0.20, bottom=0.20)
fig4.colorbar(plt.contourf(xp2d_2, yp2d_2, u2d_2, 1000, cmap=plt.get_cmap("plasma")), ax=ax4, label="$u(x,y)$")
plt.axis([0, 3.5, -0.4, 1])
plt.title("Values of $u(x,y)$ (DNS) in the area $[x_0,x_n]$ x $[y_0,y_n]$")
plt.xlabel("$x [m]$")
plt.ylabel("$y [m]$")
plt.savefig("pictures/u_in_domain.png")

fig5, ax5 = plt.subplots()
plt.subplots_adjust(left=0.20, bottom=0.20)
fig5.colorbar(plt.contourf(xp2d_2, yp2d_2, v2d_2, 1000, cmap=plt.get_cmap("plasma")), ax=ax5, label="$v(x,y)$")
plt.axis([0, 3.5, -0.4, 1])
plt.title("Values of $v(x,y)$ (DNS) in the area $[x_0,x_n]$ x $[y_0,y_n]$")
plt.xlabel("$x [m]$")
plt.ylabel("$y [m]$")
plt.savefig("pictures/v_in_domain.png")

fig6, ax6 = plt.subplots()
plt.subplots_adjust(left=0.20, bottom=0.20)
fig6.colorbar(plt.contourf(xp2d_2, yp2d_2, dudy_2, 1000, cmap=plt.get_cmap("plasma")), ax=ax6,
              label="$\partial u /\partial y$")
plt.axis([0, 3.5, -0.4, 1])
plt.title("Values of $dudy$ (DNS) in the area $[x_0,x_n]$ x $[y_0,y_n]$")
plt.xlabel("$x [m]$")
plt.ylabel("$y [m]$")
plt.savefig("pictures/dudy_in_domain.png")

fig7, ax7 = plt.subplots()
plt.subplots_adjust(left=0.20, bottom=0.20)
fig7.colorbar(plt.contourf(xp2d_2, yp2d_2, dvdx_2, 1000, cmap=plt.get_cmap("plasma")), ax=ax7,
              label="$\partial v /\partial x$")
plt.axis([0, 3.5, -0.4, 1])
plt.title("Values of $dvdx$ (DNS) in the area $[x_0,x_n]$ x $[y_0,y_n]$")
plt.xlabel("$x [m]$")
plt.ylabel("$y [m]$")
plt.savefig("pictures/dvdx_in_domain.png")

fig8, ax8 = plt.subplots()
plt.subplots_adjust(left=0.20, bottom=0.20)
fig8.colorbar(plt.contourf(xp2d_2, yp2d_2, uu2d_2, 1000, cmap=plt.get_cmap("plasma")), ax=ax8, label="uu2d")
plt.axis([0, 3.5, -0.4, 1])
plt.title("Values of $uu$ (DNS) in the area $[x_0,x_n]$ x $[y_0,y_n]$")
plt.xlabel("$x [m]$")
plt.ylabel("$y [m]$")
plt.savefig("pictures/uu_in_domain.png")

fig9, ax9 = plt.subplots()
plt.subplots_adjust(left=0.20, bottom=0.20)
fig9.colorbar(plt.contourf(xp2d_2, yp2d_2, dvdy_2, 1000, cmap=plt.get_cmap("plasma")), ax=ax9,
              label="$\partial v /\partial y$")
plt.axis([0, 3.5, -0.4, 1])
plt.title("Values of $dvdy$ (DNS) in the area $[x_0,x_n]$ x $[y_0,y_n]$")
plt.xlabel("$x [m]$")
plt.ylabel("$y [m]$")
plt.savefig("pictures/dvdy_in_domain.png")

fig2, ax2 = plt.subplots()
plt.subplots_adjust(left=0.20, bottom=0.20)
fig2.colorbar(plt.contourf(xp2d_2, yp2d_2, y_svr, 1000, cmap=plt.get_cmap("plasma")), ax=ax2, label="$C_\mu$")  # y_svr
plt.axis([0, 3.5, -0.4, 1])
plt.title("Values of $C_\mu$ (Prediction) in the area $[x_0,x_n]$ x $[y_0,y_n]$")
plt.xlabel("$x [m]$")
plt.ylabel("$y [m]$")
plt.savefig("pictures/C_my_pred_in_domain.png")

plt.show()
