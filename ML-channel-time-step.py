import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVR
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams.update({'font.size': 22})

viscos=1/5200

plt.close('all')
plt.interactive(True)

# load DNS data
DNS_mean=np.genfromtxt("LM_Channel_5200_mean_prof.dat",comments="%")
y_DNS=DNS_mean[:,0]
yplus_DNS=DNS_mean[:,1]
u_DNS=DNS_mean[:,2]
dudy_DNS=np.gradient(u_DNS,y_DNS)

DNS_stress=np.genfromtxt("LM_Channel_5200_vel_fluc_prof.dat",comments="%")
u2_DNS=DNS_stress[:,2]
v2_DNS=DNS_stress[:,3]
w2_DNS=DNS_stress[:,4]
uv_DNS=DNS_stress[:,5]
k_DNS=0.5*(u2_DNS+v2_DNS+w2_DNS)

DNS_RSTE=np.genfromtxt("LM_Channel_5200_RSTE_k_prof.dat",comments="%")
eps_DNS=DNS_RSTE[:,7]/viscos # it is scaled with ustar**4/viscos


# fix wall
eps_DNS[0]=eps_DNS[1]
vist_DNS=abs(uv_DNS)/dudy_DNS

tau_DNS=k_DNS/eps_DNS


a11_DNS=u2_DNS/k_DNS-0.66666
a22_DNS=v2_DNS/k_DNS-0.66666
a33_DNS=w2_DNS/k_DNS-0.66666

c2=(2*a11_DNS+a33_DNS)/tau_DNS**2/dudy_DNS**2

c2_mean=np.trapz(c2,y_DNS)


# load data from k-omega RANS
data = np.loadtxt('y_u_k_om_uv_5200-RANS-code.txt')
y_rans = data[:,0]
k_rans = data[:,2]
om_rans = data[:,3]
eps_rans=0.08*k_rans*om_rans
# interpolate to DNS grid
tau_rans=k_rans/eps_rans

# interpolate to DNS grid
tau_rans_DNS=np.interp(y_DNS, y_rans, tau_rans)



################## file 2, AKN
data=np.loadtxt('y_u_k_diss_nut_AKN_5200.dat')
y=data[:,0]
u=data[:,1]
k=data[:,2]
eps=data[:,3]
tau=k/eps

yplus=y/viscos

y_akn=y
k_akn=k
eps_akn=eps
yplus_akn = yplus

# interpolate to DNS grid
tau_akn_DNS=np.interp(y_DNS, y_rans, tau)

################## file 3, k-omega
data=np.loadtxt('y_u_k_om_nut_peng_5200.dat')
y=data[:,0]
u=data[:,1]
k=data[:,2]
om=data[:,3]
yplus=y/viscos
eps_rans=0.08*k_rans*om_rans

ustar=(viscos*u[0]/y[0])**0.5
yplus=y*ustar/viscos

eps_peng=0.08*k*om
k_peng=k
om_peng=om
yplus_peng = y/viscos
tau_peng=k_peng/eps_peng

# interpolate to DNS grid
tau_peng_DNS=np.interp(y_DNS, y_rans, tau_peng)

# time scale from DNS
u2_non_DNS=0.6666*k_DNS+0.82/12*k_DNS*tau_DNS**2*dudy_DNS**2
v2_non_DNS=0.6666*k_DNS-0.5/12*k_DNS*tau_DNS**2*dudy_DNS**2
w2_non_DNS=0.6666*k_DNS-0.16/12*k_DNS*tau_DNS**2*dudy_DNS**2

# time scale from k-omega model
u2_non_rans=0.6666*k_DNS+0.82/12*k_DNS*tau_rans_DNS**2*dudy_DNS**2
v2_non_rans=0.6666*k_DNS-0.5/12*k_DNS*tau_rans_DNS**2*dudy_DNS**2
w2_non_rans=0.6666*k_DNS-0.16/12*k_DNS*tau_rans_DNS**2*dudy_DNS**2

# time scale from PDH k-omega model
u2_non_peng=0.6666*k_DNS+0.82/12*k_DNS*tau_peng_DNS**2*dudy_DNS**2
v2_non_peng=0.6666*k_DNS-0.5/12*k_DNS*tau_peng_DNS**2*dudy_DNS**2
w2_non_peng=0.6666*k_DNS-0.16/12*k_DNS*tau_peng_DNS**2*dudy_DNS**2

# time scale from PDH k-omega model
u2_non_akn=0.6666*k_DNS+0.82/12*k_DNS*tau_akn_DNS**2*dudy_DNS**2
v2_non_akn=0.6666*k_DNS-0.5/12*k_DNS*tau_akn_DNS**2*dudy_DNS**2
w2_non_akn=0.6666*k_DNS-0.16/12*k_DNS*tau_akn_DNS**2*dudy_DNS**2

# plot stresses time-scale-DNS
################### 
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(yplus_DNS,u2_DNS,'b-',label='uu, DNS')
plt.plot(yplus_DNS,v2_DNS,'r-',label='vv, DNS')
plt.plot(yplus_DNS,w2_DNS,'k-',label='ww, DNS')
plt.plot(yplus_DNS,k_DNS,'g-',label='k, DNS')
plt.plot(yplus_DNS[::30],u2_non_DNS[::30],'bo',label='uu')
plt.plot(yplus_DNS[::30],v2_non_DNS[::30],'ro',label='vv')
plt.plot(yplus_DNS[::30],w2_non_DNS[::30],'ko',label='ww')
plt.axis([0, 5200, 0,10])
plt.legend(loc="best",fontsize=14)
plt.savefig('stresses-DNS-and-non-linear-time-scale-DNS.png',bbox_inches='tight')

# plot stresses time-scale-DNS, k-omega
################### 
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(yplus_DNS,u2_DNS,'b-',label='uu, DNS')
plt.plot(yplus_DNS,v2_DNS,'r-',label='vv, DNS')
plt.plot(yplus_DNS,w2_DNS,'k-',label='ww, DNS')
plt.plot(yplus_DNS,k_DNS,'g-',label='k, DNS')
plt.plot(yplus_DNS[::30],u2_non_rans[::30],'bo',label='uu')
plt.plot(yplus_DNS[::30],v2_non_rans[::30],'ro',label='vv')
plt.plot(yplus_DNS[::30],w2_non_rans[::30],'ko',label='ww')
plt.axis([0, 5200, 0,10])
plt.legend(loc="best",fontsize=14)
plt.savefig('stresses-DNS-and-non-linear-time-scale-k-omega.png',bbox_inches='tight')

# plot stresses time-scale-DNS, k-omega Peng
################### 
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(yplus_DNS,u2_DNS,'b-',label='uu, DNS')
plt.plot(yplus_DNS,v2_DNS,'r-',label='vv, DNS')
plt.plot(yplus_DNS,w2_DNS,'k-',label='ww, DNS')
plt.plot(yplus_DNS,k_DNS,'g-',label='k, DNS')
plt.plot(yplus_DNS[::30],u2_non_peng[::30],'bo',label='uu')
plt.plot(yplus_DNS[::30],v2_non_peng[::30],'ro',label='vv')
plt.plot(yplus_DNS[::30],w2_non_peng[::30],'ko',label='ww')
plt.axis([0, 5200, 0,10])
plt.legend(loc="best",fontsize=14)
plt.savefig('stresses-DNS-and-non-linear-time-scale-k-omega-peng.png',bbox_inches='tight')

# plot stresses time-scale-DNS, k-omega
################### 
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(yplus_DNS,u2_DNS,'b-',label='uu, DNS')
plt.plot(yplus_DNS,v2_DNS,'r-',label='vv, DNS')
plt.plot(yplus_DNS,w2_DNS,'k-',label='ww, DNS')
plt.plot(yplus_DNS,k_DNS,'g-',label='k, DNS')
plt.plot(yplus_DNS[::30],u2_non_akn[::30],'bo',label='uu')
plt.plot(yplus_DNS[::30],v2_non_akn[::30],'ro',label='vv')
plt.plot(yplus_DNS[::30],w2_non_akn[::30],'ko',label='ww')
plt.axis([0, 5200, 0,10])
plt.legend(loc="best",fontsize=14)
plt.savefig('stresses-DNS-and-non-linear-time-scale-akn.png',bbox_inches='tight')

# plot time scales, 
################### 
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(yplus_DNS,tau_DNS,'b-',label='DNS')
plt.plot(yplus_DNS,tau_rans_DNS,'r-',label='RANS, k-omega')
plt.plot(yplus_DNS,tau_peng_DNS,'k-',label='RANS, Peng k-omega')
plt.plot(yplus_DNS,tau_akn_DNS,'b--',label='RANS, AKN k-eps')
plt.axis([0, 2000, 0,0.6])
plt.legend(loc="best",fontsize=14)
plt.savefig('rime-scale-DNS-and-non-liner.png',bbox_inches='tight')


# vist and diss of k-omega model agree well with DNS, but not k. Hence omega is taken from diss and vist
# vist = cmu*k**2/eps
# omega = eps/k = eps/(vist*eps/cmu)**0.5 = (eps/vist/cmu)**0.5
omega_DNS=(eps_DNS/0.09/vist_DNS)**0.5


# turbulence model: uv = -cmu*k/omega*dudy => cmu=-uv/(k*dudy)*omega
# Input data: dudy
# output, to be predicted: cmu. interpolate to k-omega grid
cmu_DNS=-uv_DNS/(k_DNS*dudy_DNS)*omega_DNS
# fix cmu at the wall
cmu_DNS[0]=1
cmu_all_data=cmu_DNS

# input dudy
dudy_all_data=dudy_DNS


# create indices for all data
index= np.arange(0,len(cmu_all_data), dtype=int)

# number of elements of test data, 20%
n_test=int(0.2*len(cmu_all_data))

# pick 20% elements randomly (test data)
index_test=np.random.choice(index, size=n_test, replace=False)
# pick every 5th elements 
#index_test=index[::5]

dudy_test=dudy_all_data[index_test]
cmu_out_test=cmu_all_data[index_test]
n_test=len(dudy_test)

# delete testing data from 'all data' => training data
dudy_in=np.delete(dudy_all_data,index_test)
cmu_out=np.delete(cmu_all_data,index_test)
n_svr=len(cmu_out)

# re-shape
dudy_in=dudy_in.reshape(-1, 1)

# scale input data 
scaler_dudy=StandardScaler()
dudy_in=scaler_dudy.fit_transform(dudy_in)

# setup X (input) and y (output)
X=np.zeros((n_svr,1))
y=cmu_out
X[:,0]=dudy_in[:,0]

print('starting SVR')

# choose Machine Learning model
C=1
eps=0.001
# use Linear model
#model = LinearSVR(epsilon = eps , C = C, max_iter=1000)
model = SVR(kernel='rbf', epsilon = eps, C = C)

# Fit the model
svr = model.fit(X, y.flatten())

#  re-shape test data
dudy_test=dudy_test.reshape(-1, 1)

# scale test data
dudy_test=scaler_dudy.transform(dudy_test)

# setup X (input) for testing (predicting)
X_test=np.zeros((n_test,1))
X_test[:,0]=dudy_test[:,0]

# predict cmu
cmu_predict= model.predict(X_test)

# find difference between ML prediction and target
cmu_error=np.std(cmu_predict-cmu_out_test)/\
(np.mean(cmu_predict**2))**0.5
print('\nRMS error using ML turbulence model',cmu_error)

################### 2D scatter top view plot all points, both test and y_svr
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
ax=plt.gca()

# plot all points
plt.scatter(scaler_dudy.inverse_transform(dudy_test), cmu_out_test,marker='o', s=20.2,c='green',label='target')
plt.scatter(scaler_dudy.inverse_transform(dudy_test), cmu_predict,marker='o', s=20.2,c='blue',label='predicted')

#label axes
ax.set_ylabel(r'$C_\mu$')
plt.xlabel('$\partial U^+/\partial y$')
plt.axis([0,2500,0,1.1])
plt.legend(loc="upper left",fontsize=14)

axins1 = inset_axes(ax1, width="50%", height="50%", loc='upper right', borderpad=0.1)
# reduce fotnsize 
axins1.tick_params(axis = 'both', which = 'major', labelsize = 10)
plt.scatter(scaler_dudy.inverse_transform(dudy_test), cmu_out_test,marker='o', s=20.2,c='green')
plt.scatter(scaler_dudy.inverse_transform(dudy_test), cmu_predict,marker='o', s=20.2,c='blue')
axins1.yaxis.set_label_position("left")
axins1.yaxis.tick_left()
axins1.xaxis.set_label_position("bottom")
axins1.xaxis.tick_bottom()
plt.ylabel("$C_\mu$")
plt.xlabel("$\partial U^+/\partial y$")
plt.axis([0, 100, 0.4,1])


plt.savefig('scatter-cmu-vs-dudy-svr-and-test.png',bbox_inches='tight')

