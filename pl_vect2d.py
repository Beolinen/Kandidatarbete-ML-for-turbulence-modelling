#----------------------------------------------Import Packages----------------------------------------------
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR  # for building SVR model
from gradients import compute_face_phi,dphidx,dphidy,init

#----------------------------------------------Read Data Original Case----------------------------------------------
# read data file
tec=np.genfromtxt("small_wave/tec.dat",skip_header=2)

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
eps_DNS = tec[:,9]

#Define matrix dimensions
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

#Reshape vectors to wanted dimensions
u2d = np.reshape(u, (nj, ni))
v2d = np.reshape(v, (nj, ni))
p2d = np.reshape(p, (nj, ni))
x2d = np.reshape(x, (nj, ni))
y2d = np.reshape(y, (nj, ni))
uu2d = np.reshape(uu, (nj, ni))  # =mean{v'_1v'_1}
uv2d = np.reshape(uv, (nj, ni))  # =mean{v'_1v'_2}
vv2d = np.reshape(vv, (nj, ni))  # =mean{v'_2v'_2}
ww2d = np.reshape(ww,(nj,ni))
k2d = np.reshape(k_DNS, (nj, ni))  # =mean{0.5(v'_iv'_i)}
eps_DNS2d = np.reshape(eps_DNS,(nj,ni))

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

# Set Neumann boundary conditions of p at upper and lower boundaries, what is p?
p2d[:, 1] = p2d[:, 2]
p2d[:, -1] = p2d[:, -1 - 1]

# x and y are fo the cell centers. The dphidx_dy routine needs the face coordinate, xf2d, yf2d 
#integrerningsgränser för punkter
# load them
xc_yc = np.loadtxt("small_wave/xc_yc.dat")
xf = xc_yc[:, 0]
yf = xc_yc[:, 1]
xf2d = np.reshape(xf, (nj, ni)).transpose()
yf2d = np.reshape(yf, (nj, ni)).transpose()


# compute cell centers
xp2d = 0.25*(x2d[0:-1,0:-1]+x2d[0:-1,1:]+x2d[1:,0:-1]+x2d[1:,1:])
yp2d = 0.25*(y2d[0:-1,0:-1]+y2d[0:-1,1:]+y2d[1:,0:-1]+y2d[1:,1:])

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

ni = ni-2
nj = nj-2

# eps at last cell upper cell wrong. fix it.
eps_DNS2d[:,-1]=eps_DNS2d[:,-2]

print('new x2d.shape', x2d.shape)
print('new u2d.shape', u2d.shape)

# compute face value of U and V
u2d_face_w,u2d_face_s = compute_face_phi(u2d, fx, fy, ni, nj) #Error with dimensions for u2d vs fx,fy
v2d_face_w,v2d_face_s = compute_face_phi(v2d, fx, fy, ni, nj)

# x derivatives
dudx = dphidx(u2d_face_w, u2d_face_s, areawx, areasx, vol)
dvdx = dphidx(v2d_face_w, u2d_face_s, areawx, areasx, vol)

# x derivatives
dudy = dphidx(u2d_face_w, u2d_face_s, areawy, areasy, vol)
dvdy = dphidx(v2d_face_w, u2d_face_s, areawy, areasy, vol)

#----------------------------------------------Project-Group Work---------------------------------------------------
#----------------------------------------------ML-Method Original Case----------------------------------------------
omega = eps_DNS2d/k2d/0.09

# np.maximum is used to avoid division by zero
k2d = np.maximum(k2d, viscos*omega)

#Compute C_my and ||duidxj|| to train model

cmy_DNS = np.array(-uv2d/(k2d*(dudy + dvdx))*omega) 

cmy_DNS = np.where(abs(dudy + dvdx)  < 1,1, cmy_DNS)
cmy_DNS = np.where(cmy_DNS > 0, cmy_DNS,1)
cmy_DNS = np.where(cmy_DNS <= 2, cmy_DNS, 1)

duidxj = np.array((0.5*dudy**2 + dudx**2 + dvdy**2 + 0.5*dvdx**2)**0.5)

#ML-metod
# The MinMaxScaler works by first computing the minimum and maximum values of each feature in the training data. 
# It then scales the values of each feature such that the minimum value is mapped to 0 and the maximum value is mapped to 1.
scaler = MinMaxScaler()

duidxj = duidxj.reshape(-1, 1)

duidxj_scaled = []
duidxj_scaled = scaler.fit_transform(duidxj)

k_scaled = k2d.reshape(-1, 1)
k_scaled = scaler.fit_transform(k_scaled)

uv_scaled = uv2d.reshape(-1,1)
uv_scaled = scaler.fit_transform(uv_scaled)

#2 columns for 3D plot, 1 for 2D --> comment second column
X = np.zeros((len(duidxj_scaled), 2))
X[:,0] = duidxj_scaled[:,0]

print 

#Choose model
#X[:,1] = k_scaled[:, 0]
X[:,1] = uv_scaled[:, 0]

Y = cmy_DNS

model = SVR(kernel = 'rbf', C = 1, epsilon = 0.001)
SVR = model.fit(X, Y.flatten())

# plot the results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y, c='green', label='target', alpha=0.5)
ax.scatter(X[:,0], X[:,1], SVR.predict(X), c='blue', label='prediction', alpha=0.5)
ax.set_xlabel('duidxj')
ax.set_ylabel('k')
ax.set_zlabel('cmy')
ax.legend()
plt.savefig('plots/cmy_k_small.png')

#----------------------------------------------Test With New Case------------------------------------------------
#----------------------------------------------Read Data Large Case----------------------------------------------
tec_large = np.genfromtxt("large_wave/tec_large.dat", dtype=None,comments="%")

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

xc_yc_large = np.loadtxt("large_wave/xc_yc_large.dat")
xf_large = xc_yc_large[:, 0]
yf_large = xc_yc_large[:, 1]
xf2d_large = np.reshape(xf_large, (nj, ni))
yf2d_large = np.reshape(yf_large, (nj, ni))
xf2d_large = np.transpose(xf2d_large)
yf2d_large = np.transpose(yf2d_large)

# compute cell centers
xp2d_large = 0.25*(x2d_large[0:-1, 0:-1]+x2d_large[0:-1,1:]+x2d_large[1:,0:-1]+x2d_large[1:, 1:])
yp2d_large = 0.25*(y2d_large[0:-1, 0:-1]+y2d_large[0:-1,1:]+y2d_large[1:,0:-1]+y2d_large[1:, 1:])

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

ni = ni-2
nj = nj-2

# eps at last cell upper cell wrong. fix it.
eps_DNS2d[:, -1] = eps_DNS2d[:, -2]

print('new x2d.shape', x2d.shape)
print('new u2d.shape', u2d.shape)

# compute face value of U and V
u2d_face_w,u2d_face_s = compute_face_phi(u2d_large, fx, fy, ni, nj) #Error with dimensions for u2d vs fx,fy
v2d_face_w,v2d_face_s = compute_face_phi(v2d_large, fx, fy, ni, nj)

# x derivatives
dudx_large = dphidx(u2d_face_w, u2d_face_s, areawx, areasx, vol)
dvdx_large = dphidx(v2d_face_w, u2d_face_s, areawx, areasx, vol)

# x derivatives
dudy_large = dphidx(u2d_face_w, u2d_face_s, areawy, areasy, vol)
dvdy_large = dphidx(v2d_face_w, u2d_face_s, areawy, areasy, vol)

#----------------------------------------------ML-Method Large Case----------------------------------------------
# Reshape is used to fit arrays to the dimensions of the data.
# scaler.fit_transform is used to scale the data to a range between 0 and 1.

#Calculate correct C_my for prediction
omega_large = eps_DNS2d_large/k_DNS2d/0.09

# np.maximum is used to avoid division by zero
k_DNS2d = np.maximum(k_DNS2d,viscous_large*omega_large)

cmy_DNS_large = np.array(-uv2d_large/(k_DNS2d*(dudy_large + dvdx_large))*omega_large)

# np.where is used to find the indices where the condition is true
cmy_DNS_large = np.where(abs(dudy_large+dvdx_large)  < 1, 1, cmy_DNS_large)
cmy_DNS_large = np.where(cmy_DNS_large > 0, cmy_DNS_large, 1)
cmy_DNS_large = np.where(cmy_DNS_large <= 2, cmy_DNS_large, 1)

# np.array is used to convert the list to an array
duidxj_test = np.array((0.5*dudy_large**2 + dudx_large**2 + dvdy_large**2 + 0.5*dvdx_large**2)**0.5)
duidxj_test = duidxj_test.reshape(-1, 1)
duidxj_test_scaled = scaler.fit_transform(duidxj_test)

k_scaled_large = k_DNS2d.reshape(-1, 1)
k_scaled_large = scaler.fit_transform(k_scaled_large)

uv_large_scaled = uv2d_large.reshape(-1, 1)
uv_large_scaled = scaler.fit_transform(uv_large_scaled)

#2 columns for 3D plot, 1 for 2D --> comment second column
X_test = np.zeros((len(duidxj_test), 2))
X_test[:,0] = duidxj_test_scaled[:, 0]

# Choose which model to use: k or uv   (k gives unrealistic results)
#X_test[:,1] = k_scaled_large[:,0]
X_test[:, 1] = uv_large_scaled[:, 0]

y_svr = model.predict(X_test)

X_test_no_scale = scaler.inverse_transform(X_test)

#----------------------------------------------Calculate Error----------------------------------------------
errorML = (np.std(y_svr.flatten() - cmy_DNS_large.flatten()))/(np.mean(y_svr**2))**0.5
error = (np.std(0.09 - cmy_DNS_large.flatten()))/(np.mean(0.09**2))**0.5
errorOmega = (np.std(1-cmy_DNS_large.flatten()))/(np.mean((1)**2))**0.5

predictOwnCase = model.predict(X)
errorOwnCase = (np.std(predictOwnCase.flatten() - cmy_DNS.flatten()))/(np.mean(predictOwnCase**2))**0.5

#Print error
print("RMS-felet med ML är", errorML)
print("RMS-felet med standardmodell (C_my = 0.09) är", error)
print("RMS-felet med standardmodell ,k-omega, (C_my = 1) är", errorOmega)
print("Error in fitting case is", errorOwnCase)

#----------------------------------------------Plot Solution----------------------------------------------
plt.figure("Test")

#plt.scatter(X_test_no_scale[:,0],cmy_DNS_large,marker = "o", s = 10, c = "green",label = "Target")
#plt.scatter(X_test_no_scale[:,0],y_svr,marker = "o", s= 10, c = "blue", label = "Prediction")

ax = plt.axes(projection = '3d')

# plot the surface
ax.scatter(X_test_no_scale[:,0], X_test_no_scale[:,1], cmy_DNS_large, s=10, c="green", label="Target")
ax.scatter(X_test_no_scale[:,0], X_test_no_scale[:,1], y_svr, s=10, c="blue", label="Prediction")

#ax.scatter(X_test_no_scale[:,0], X_test_no_scale[:,1], cmy_DNS_large, marker="o", s=10, c="green", label="Target")
#ax.scatter(X_test_no_scale[:,0], X_test_no_scale[:,1], y_svr,marker="o", s=10, c="blue", label="Prediction")
plt.legend(loc="upper right", prop=dict(size=12))
ax.set_xlabel("$ || S_{i j}||$")
ax.set_ylabel("$uv$")
ax.set_zlabel("$C_\mu^{k-\omega}$")
plt.title("$C_\mu^{k-\omega} = f( ||S_{i j}||,uv)$")
plt.savefig("plots/Modell_small_test_large_S_ij_uv_only_model.png")

#----------------------------------------------Plot Cmy in domain----------------------------------------------
#Fix dimensions of x and y, by deleting first row and column or last row and column
x2d = np.delete(x2d, -1,0)
#x2d = np.delete(x2d,  0,0)
x2d = np.delete(x2d, -1,1)
#x2d = np.delete(x2d,  0,1)

y2d = np.delete(y2d, -1,0)
#y2d = np.delete(y2d,  0,0)
y2d = np.delete(y2d, -1,1)
#y2d = np.delete(y2d,  0,1)

# plot the 
fig1, ax1 = plt.subplots()
plt.subplots_adjust(left=0.20, bottom=0.20)
plt.contourf(x2d, y2d, cmy_DNS_large, 5)
fig1.colorbar(plt.contourf(x2d, y2d, cmy_DNS, 5), ax=ax1, label = "$C_\my$")
plt.axis([0,3,-0.25,1.25])
plt.title('Values of $C_\mu$ in the area $[x_0,x_n] \times [y_0,y_n]$')
plt.xlabel("$x [m]$")
plt.ylabel("$y [m]$")
plt.savefig("plots/C_my_in_domain.png")

plt.show()