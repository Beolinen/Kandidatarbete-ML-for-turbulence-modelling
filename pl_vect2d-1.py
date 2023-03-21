#----------------------------------------------Import Packages----------------------------------------------
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR  # for building SVR model
from gradients import compute_face_phi,dphidx,dphidy,init
from sklearn.metrics import mean_squared_error

#----------------------------------------------Read Data Original Case----------------------------------------------
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
# read data file
tec=np.genfromtxt("/Users/benjaminjonsson/Programmering/Kandidat/large_wave/tec_large.dat", dtype=None,comments="%")

# text='VARIABLES = X Y P U V u2 v2 w2 uv eps'
#Define variables from text-file
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

# set Neumann of p at upper and lower boundaries
p2d[:, 1] = p2d[:, 2]
p2d[:, -1] = p2d[:, -1 - 1]

# x and y are fo the cell centers. The dphidx_dy routine needs the face coordinate, xf2d, yf2d
# load them
xc_yc = np.loadtxt("/Users/benjaminjonsson/Programmering/Kandidat/large_wave/xc_yc_large.dat")
xf = xc_yc[:, 0]
yf = xc_yc[:, 1]
xf2d = np.reshape(xf, (nj, ni))
yf2d = np.reshape(yf, (nj, ni))
xf2d = np.transpose(xf2d)
yf2d = np.transpose(yf2d)

# compute cell centers
xp2d=0.25*(x2d[0:-1,0:-1]+x2d[0:-1,1:]+x2d[1:,0:-1]+x2d[1:,1:])
yp2d=0.25*(y2d[0:-1,0:-1]+y2d[0:-1,1:]+y2d[1:,0:-1]+y2d[1:,1:])

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

print('new x2d.shape',x2d.shape)
print('new u2d.shape',u2d.shape)

# compute face value of U and V
u2d_face_w,u2d_face_s=compute_face_phi(u2d,fx,fy,ni,nj) #Error with dimensions for u2d vs fx,fy
v2d_face_w,v2d_face_s=compute_face_phi(v2d,fx,fy,ni,nj)

# x derivatives
dudx=dphidx(u2d_face_w,u2d_face_s,areawx,areasx,vol)
dvdx=dphidx(v2d_face_w,v2d_face_s,areawx,areasx,vol)

# y derivatives
dudy=dphidy(u2d_face_w,u2d_face_s,areawy,areasy,vol)
dvdy=dphidy(v2d_face_w,v2d_face_s,areawy,areasy,vol)


#----------------------------------------------Project-Group Work---------------------------------------------------
#----------------------------------------------ML-Method Original Case----------------------------------------------
omega = eps_DNS2d/k2d/0.09

# np.maximum is used to avoid division by zero
k2d=np.maximum(k2d,viscos*omega)

#Compute C_my and ||duidxj|| to train model
cmy_DNS = np.array(-uv2d/(k2d*(dudy + dvdx))*omega) 
cmy_DNS = np.where(abs(dudy + dvdx)  < 1, 1, cmy_DNS)
cmy_DNS = np.where(cmy_DNS > 0, cmy_DNS, 1)
cmy_DNS = np.where(cmy_DNS <= 2, cmy_DNS, 1)

duidxj = np.array((dudx**2 + 0.5*(dudy**2 + 2*dudy*dvdx + dvdx**2) + dvdy**2)**0.5)

#ML-metod
# The MinMaxScaler works by first computing the minimum and maximum values of each feature in the training data. 
# It then scales the values of each feature such that the minimum value is mapped to 0 and the maximum value is mapped to 1.
scaler = StandardScaler()

# Reshape Data
duidxj = duidxj.reshape(-1,1)
k_scaled_reshaped = k2d.reshape(-1,1)
uv_scaled_reshaped = uv2d.reshape(-1,1)
eps_scaled_reshaped = eps_DNS2d.reshape(-1, 1)

# scale data
duidxj_scaled = scaler.fit_transform(duidxj)
k_scaled = scaler.fit_transform(k_scaled_reshaped)
uv_scaled = scaler.fit_transform(uv_scaled_reshaped)
eps_scaled = scaler.fit_transform(eps_scaled_reshaped)

#2 columns for 3D plot, 1 for 2D --> comment second column
X = np.zeros((len(duidxj_scaled),1))
X[:,0] = duidxj_scaled[:,0]  # Origanal training data

#X[:,1] = uv_scaled[:,0] # Original training data
#X[:,2] = k_scaled[:,0]
#X[:,3] = eps_scaled[:,0]

Y = cmy_DNS

#Choose model
model = SVR(kernel = 'rbf', C = 1, epsilon = 0.001)
SVR = model.fit(X,Y.flatten())

#----------------------------------------------Test With New Case------------------------------------------------
#----------------------------------------------Read Data Large Case----------------------------------------------
tec_large = np.genfromtxt("/Users/benjaminjonsson/Programmering/Kandidat/small_wave/tec.dat", dtype=None,comments="%")

u_large = tec_large[:,3]
v_large = tec_large[:,4]
y_large = tec_large[:,1]
x_large = tec_large[:,0]
uv_large = tec_large[:,8]
uu_large = tec_large[:, 5]
vv_large = tec_large[:, 6]
ww_large = tec_large[:, 7]
k_DNS_large = 0.5 * (uu_large + vv_large + ww_large)
eps_DNS_large = tec_large[:,9]
p_large = tec_large[:,2]

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
uv2d_large = np.transpose(np.reshape(uv_large,(nj,ni)))
k_DNS2d = np.transpose(np.reshape(k_DNS_large,(nj,ni)))
eps_DNS2d_large = np.transpose(np.reshape(eps_DNS_large,(nj,ni)))
x2d_large = np.transpose(np.reshape(x_large,(nj,ni)))
y2d_large = np.transpose(np.reshape(y_large,(nj,ni)))
p2d_large = np.transpose(np.reshape(p_large,(nj,ni)))
uu2d_large = np.transpose(np.reshape(uu_large,(nj,ni)))
vv2d_large = np.transpose(np.reshape(vv_large,(nj,ni)))
ww2d_large = np.transpose(np.reshape(ww_large,(nj,ni)))

xc_yc_large = np.loadtxt("/Users/benjaminjonsson/Programmering/Kandidat/small_wave/xc_yc.dat")
xf_large = xc_yc_large[:, 0]
yf_large = xc_yc_large[:, 1]
xf2d_large = np.reshape(xf_large, (nj, ni))
yf2d_large = np.reshape(yf_large, (nj, ni))
xf2d_large = np.transpose(xf2d_large)
yf2d_large = np.transpose(yf2d_large)

# compute cell centers
xp2d_large=0.25*(x2d_large[0:-1,0:-1]+x2d_large[0:-1,1:]+x2d_large[1:,0:-1]+x2d_large[1:,1:])
yp2d_large=0.25*(y2d_large[0:-1,0:-1]+y2d_large[0:-1,1:]+y2d_large[1:,0:-1]+y2d_large[1:,1:])

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
areaw,areawx,areawy,areas,areasx,areasy,vol,fx,fy = init(x2d_large,y2d_large,xp2d_large,yp2d_large)

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
eps_DNS2d[:,-1]=eps_DNS2d[:,-2]

print('new x2d.shape',x2d.shape)
print('new u2d.shape',u2d.shape)

# compute face value of U and V
u2d_face_w,u2d_face_s=compute_face_phi(u2d_large,fx,fy,ni,nj) #Error with dimensions for u2d vs fx,fy
v2d_face_w,v2d_face_s=compute_face_phi(v2d_large,fx,fy,ni,nj)

# x derivatives
dudx_large=dphidx(u2d_face_w,u2d_face_s,areawx,areasx,vol)
dvdx_large=dphidx(v2d_face_w,v2d_face_s,areawx,areasx,vol)

# y derivatives
dudy_large=dphidy(u2d_face_w,u2d_face_s,areawy,areasy,vol)
dvdy_large=dphidy(v2d_face_w,v2d_face_s,areawy,areasy,vol)
        
#----------------------------------------------ML-Method Large Case----------------------------------------------
'''Reshape is used to fit arrays to the dimensions of the data.
   scaler.fit_transform is used to scale the data to a range between 0 and 1.'''

#Calculate correct C_my for prediction
omega_large = eps_DNS2d_large/k_DNS2d/0.09

# np.maximum is used to avoid division by zero
k_DNS2d=np.maximum(k_DNS2d,viscous_large*omega_large)

cmy_DNS_large = np.array(-uv2d_large/(k_DNS2d*(dudy_large + dvdx_large))*omega_large)

cmy_DNS_large = np.where(abs(dudy_large+dvdx_large)  < 1, 1, cmy_DNS_large)
cmy_DNS_large = np.where(cmy_DNS_large > 0, cmy_DNS_large, 1)
cmy_DNS_large = np.where(cmy_DNS_large <= 2, cmy_DNS_large, 1)

# np.array is used to convert the list to an array
duidxj_test = np.array((dudx_large**2 + 0.5*(dudy_large**2 + 2*dudy_large*dvdx_large+ dvdx_large**2) + dvdy_large**2)**0.5)

# Reshape data
duidxj_test_reshape = duidxj_test.reshape(-1,1)
uv_large_reshape = uv2d_large.reshape(-1,1)
k_scaled_reshape = k_DNS2d.reshape(-1,1)
eps_large_reshape = eps_DNS2d_large.reshape(-1,1)

# Scale data
duidxj_test_scaled = scaler.fit_transform(duidxj_test_reshape)
k_scaled_large = scaler.fit_transform(k_scaled_reshape)
uv_large_scaled = scaler.fit_transform(uv_large_reshape)
eps_large_scaled = scaler.fit_transform(eps_large_reshape)

#x (ammount of input variables) columns for 3D plot, 1 for 2D --> comment second column
X_test = np.zeros((len(duidxj_test_scaled),1))
X_test[:,0] = duidxj_test_scaled[:,0]

# Choose which model to use: k or uv   (k gives unrealistic results)
#X_test[:,1] = uv_large_scaled[:,0]
#X_test[:,2] = k_scaled_large[:,0]
#X_test[:,3] = eps_large_scaled[:,0]

y_svr = model.predict(X_test)

X_test_no_scale = scaler.inverse_transform(X_test)

#----------------------------------------------Calculate Error----------------------------------------------
errorML = (np.std(y_svr.flatten() - cmy_DNS_large.flatten()))/(np.mean(y_svr**2))**0.5
error = (np.std(0.09 - cmy_DNS_large.flatten()))/(np.mean(0.09**2))**0.5
errorOmega = (np.std(1-cmy_DNS_large.flatten()))/(np.mean((1)**2))**0.5

predictOwnCase = model.predict(X)
errorOwnCase = (np.std(predictOwnCase.flatten() - cmy_DNS.flatten()))/(np.mean(predictOwnCase**2))**0.5

#Print error
print("Coefficient of varience med ML är", errorML)
print("Coefficient of varience med standardmodell (C_my = 0.09) är", error)
print("Coefficient of varience med standardmodell ,k-omega, (C_my = 1) är", errorOmega)
print("Coefficient of varience in fitting case is",errorOwnCase)

#RMS ERROR
c_k_eps = []
c_k_omega = []
for i in range(len(cmy_DNS_large.flatten())):
    c_k_eps.append(0.09)
    c_k_omega.append(1)

errorRMS_ML = mean_squared_error(cmy_DNS_large.flatten(),y_svr.flatten())
errorRMS = mean_squared_error(cmy_DNS_large.flatten(), c_k_eps)
errorRMS_Omega = mean_squared_error(cmy_DNS_large.flatten(), c_k_omega)
errorRMS_Own_Case = mean_squared_error(cmy_DNS.flatten(), y_svr.flatten())

print("RMS-felet med ML är", errorRMS_ML)
print("RMS-felet med standardmodell (C_my = 0.09) är", errorRMS)
print("RMS-felet med standardmodell ,k-omega, (C_my = 1) är", errorRMS_Omega)
print("Error in fitting case is",errorRMS_Own_Case)

#----------------------------------------------Plot Solution----------------------------------------------
plt.figure("Test")

plt.scatter(X_test_no_scale[:,0],cmy_DNS_large,marker = "o", s = 10, c = "green",label = "Target")
plt.scatter(X_test_no_scale[:,0],y_svr,marker = "o", s= 10, c = "blue", label = "Prediction")
plt.xlabel("$||S_{ij}||$")
plt.ylabel("$C_{\mu}^{k-\omega}$")
plt.title("$C_\mu^{k-\omega} = f( ||S_{i j}||,uv)$")

#ax = plt.axes(projection = '3d')

#Plot k
#ax.scatter(X_test_no_scale[:,0],X_test_no_scale[:,2],cmy_DNS_large,marker = "o", s = 10, c = "green",label = "Target") 
#ax.scatter(X_test_no_scale[:,0],X_test_no_scale[:,2],y_svr,marker = "o", s= 10, c = "blue", label = "Prediction") 

#Plot uv
#x.scatter(X_test_no_scale[:,0],X_test_no_scale[:,1],cmy_DNS_large,marker = "o", s = 10, c = "green",label = "Target") 
#ax.scatter(X_test_no_scale[:,0],X_test_no_scale[:,1],y_svr,marker = "o", s= 10, c = "blue", label = "Prediction")

#ax.set_xlabel("$ || S_{i j}||$")
#ax.set_ylabel("$K$")
#ax.set_ylabel("$uv$")
#ax.set_zlabel("$C_\mu^{k-\omega}$")

plt.legend(loc="upper right",prop=dict(size=12))
plt.savefig("Modell_large_test_small_S_ij_model.png")

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
fig1.colorbar(plt.contourf(x2d, y2d, cmy_DNS_large, 1000), ax=ax1, label = "$C_\mu$")
plt.axis([0,4,-0.4,1])
plt.title("Values of $C_\mu$ (DNS) in the area $[x_0,x_n] x [y_0,y_n]$")
plt.xlabel("$x [m]$")
plt.ylabel("$y [m]$")
plt.savefig("C_my_in_domain.png")

# plot the 
y_svr = np.reshape(y_svr,(ni,nj))

fig2, ax2 = plt.subplots()
plt.subplots_adjust(left=0.20, bottom=0.20)
fig2.colorbar(plt.contourf(x2d, y2d, y_svr, 1000), ax=ax2, label = "$C_\mu$")
plt.axis([0,4,-0.4,1])
plt.title("Values of $C_\mu$ (Prediction) in the area $[x_0,x_n]$ x $[y_0,y_n]$")
plt.xlabel("$x [m]$")
plt.ylabel("$y [m]$")
plt.savefig("C_my_pred_in_domain.png")

fig3, ax3 = plt.subplots()
plt.subplots_adjust(left=0.20, bottom=0.20)
fig3.colorbar(plt.contourf(x2d, y2d, duidxj_test, 1000), ax=ax3, label = "$||S_{ij}||$")
plt.axis([0,4,-0.4,1])
plt.title("Values of $||S_{ij}||$ (DNS) in the area $[x_0,x_n]$ x $[y_0,y_n]$")
plt.xlabel("$x [m]$")
plt.ylabel("$y [m]$")
plt.savefig("S_ij_in_domain.png")

fig4, ax4 = plt.subplots()
plt.subplots_adjust(left=0.20, bottom=0.20)
fig4.colorbar(plt.contourf(x2d, y2d, u2d_large, 1000), ax=ax4, label = "$u(x,y)$")
plt.axis([0,4,-0.4,1])
plt.title("Values of $u(x,y)$ (DNS) in the area $[x_0,x_n]$ x $[y_0,y_n]$")
plt.xlabel("$x [m]$")
plt.ylabel("$y [m]$")
plt.savefig("u_in_domain.png")

fig5, ax5 = plt.subplots()
plt.subplots_adjust(left=0.20, bottom=0.20)
fig5.colorbar(plt.contourf(x2d, y2d, v2d_large, 1000), ax=ax5, label = "$v(x,y)$")
plt.axis([0,4,-0.4,1])
plt.title("Values of $v(x,y)$ (DNS) in the area $[x_0,x_n]$ x $[y_0,y_n]$")
plt.xlabel("$x [m]$")
plt.ylabel("$y [m]$")
plt.savefig("v_in_domain.png")

fig6, ax6 = plt.subplots()
plt.subplots_adjust(left=0.20, bottom=0.20)
fig6.colorbar(plt.contourf(x2d, y2d, dudy_large, 1000), ax=ax6, label = "$\partial u /\partial y$")
plt.axis([0,4,-0.4,1])
plt.title("Values of $dudy$ (DNS) in the area $[x_0,x_n]$ x $[y_0,y_n]$")
plt.xlabel("$x [m]$")
plt.ylabel("$y [m]$")
plt.savefig("dudy_in_domain.png")

fig7, ax7 = plt.subplots()
plt.subplots_adjust(left=0.20, bottom=0.20)
fig7.colorbar(plt.contourf(x2d, y2d, dvdx_large, 1000), ax=ax7, label = "$\partial v /\partial x$")
plt.axis([0,4,-0.4,1])
plt.title("Values of $dvdx$ (DNS) in the area $[x_0,x_n]$ x $[y_0,y_n]$")
plt.xlabel("$x [m]$")
plt.ylabel("$y [m]$")
plt.savefig("dvdx_in_domain.png")

fig8, ax8 = plt.subplots()
plt.subplots_adjust(left=0.20, bottom=0.20)
fig8.colorbar(plt.contourf(x2d, y2d, uu2d, 1000), ax=ax8, label = "uu2d")
plt.axis([0,4,-0.4,1])
plt.title("Values of $uu$ (DNS) in the area $[x_0,x_n]$ x $[y_0,y_n]$")
plt.xlabel("$x [m]$")
plt.ylabel("$y [m]$")
plt.savefig("uu_in_domain.png")

fig9, ax9 = plt.subplots()
plt.subplots_adjust(left=0.20, bottom=0.20)
fig9.colorbar(plt.contourf(x2d, y2d, dvdy_large, 1000), ax=ax9, label = "$\partial v /\partial y$")
plt.axis([0,4,-0.4,1])
plt.title("Values of $dvdy$ (DNS) in the area $[x_0,x_n]$ x $[y_0,y_n]$")
plt.xlabel("$x [m]$")
plt.ylabel("$y [m]$")
plt.savefig("dvdy_in_domain.png")

plt.show()

print(f" dudy i {dudx[140][5]}")
print(f" dudy i+1 {dudx[141][5]}")

dv = (u2d[141][5]-u2d[140][5])/(x2d[141][5]-x2d[140][5])
print(f"Test dudy i {dv}")