#----------------Imports----------------

import numpy as np
import torch 
import matplotlib.pyplot as plt
import warnings
import matplotlib.cbook

#----------------Parameters----------------
#Ignore matplotlib deprecation warnings in output
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

viscos=1/5200

# load RANS data created by rans.m (which can be downloaded)
# load DNS data
DNS_mean=np.genfromtxt("LM_Channel_5200_mean_prof.dat",comments="%")
y_DNS=DNS_mean[:,0]
yplus_DNS=DNS_mean[:,1]
u_DNS=DNS_mean[:,2]
dudy_DNS=np.gradient(u_DNS,y_DNS)

DNS_stress=np.genfromtxt("LM_Channel_5200_vel_fluc_prof.dat",comments="%")
uu_DNS=DNS_stress[:,2]
vv_DNS=DNS_stress[:,3]
ww_DNS=DNS_stress[:,4]
uv_DNS=DNS_stress[:,5]
uw_DNS = DNS_stress[:,6]
vw_DNS = DNS_stress[:,7]
k_DNS=0.5*(uu_DNS+vv_DNS+ww_DNS)

DNS_RSTE=np.genfromtxt("LM_Channel_5200_RSTE_k_prof.dat",comments="%")
eps_DNS=DNS_RSTE[:,7]/viscos # it is scaled with ustar**4/viscos
# fix wall
eps_DNS[0]=eps_DNS[1]

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

data=np.loadtxt('y_u_k_om_nut_peng_5200.dat')
y=data[:,0]
u=data[:,1]
k=data[:,2]
om=data[:,3]
yplus=y/viscos
eps_rans=0.08*k*om

ustar=(viscos*u[0]/y[0])**0.5
yplus=y*ustar/viscos

eps_peng=0.08*k*om
k_peng=k
om_peng=om
yplus_peng = y/viscos
tau_peng=k_peng/eps_peng

# interpolate to DNS grid
tau_peng_DNS=np.interp(y_DNS, y_rans, tau_peng)


tau_DNS = k_DNS/eps_DNS

# dont train on, uu, vv, ww, uv, uw, vw 
# Maybe mixed terms are ok (just not uu,vv,ww)

#-----------------Data_manipulation--------------------

# Delete first value for all interesting data
uv_DNS = np.delete(uv_DNS, 0)
vv_DNS = np.delete(vv_DNS, 0)
ww_DNS = np.delete(ww_DNS, 0)
uw_DNS = np.delete(uw_DNS,0)
vw_DNS = np.delete(vw_DNS,0)
k_DNS = np.delete(k_DNS, 0)
eps_DNS = np.delete(eps_DNS, 0)
dudy_DNS = np.delete(dudy_DNS, 0)
yplus_DNS = np.delete(yplus_DNS,0)
uu_DNS = np.delete(uu_DNS,0)
tau_DNS = np.delete(tau_DNS,0)
tau_akn_DNS = np.delete(tau_akn_DNS,0)
tau_rans_DNS = np.delete(tau_rans_DNS,0)
tau_peng_DNS = np.delete(tau_peng_DNS,0)

# Calculate ny_t and time-scale tau
viscous_t = k_DNS**2/eps_DNS
#tau = viscous_t/np.abs(uv_DNS)
#tau = (k_DNS/eps_DNS) #Keep this, dont expect "normal values" anymore because of this
#tau = 1/omega_RANS RANS Time scale (better apparently) (0,13 instead of 0,8)

# Calculate c_1, c_2 of the Non-linear Eddy Viscosity Model
# Array for storing c_1, c_2, & c_3

c_0_DNS = -6*(ww_DNS/k_DNS - 2/3)/(tau_DNS**2*dudy_DNS**2)
c_2_DNS = ((ww_DNS/k_DNS - 2/3) + 2*(uu_DNS/k_DNS - 2/3))/(tau_DNS**2*dudy_DNS**2)

c_0_AKN = -6*(ww_DNS/k_DNS - 2/3)/(tau_akn_DNS**2*dudy_DNS**2)
c_2_AKN = ((ww_DNS/k_DNS - 2/3) + 2*(uu_DNS/k_DNS - 2/3))/(tau_akn_DNS**2*dudy_DNS**2)

c_0_RANS = -6*(ww_DNS/k_DNS - 2/3)/(tau_rans_DNS**2*dudy_DNS**2)
c_2_RANS = ((ww_DNS/k_DNS - 2/3) + 2*(uu_DNS/k_DNS - 2/3))/(tau_rans_DNS**2*dudy_DNS**2)

c_0_PENG = -6*(ww_DNS/k_DNS - 2/3)/(tau_peng_DNS**2*dudy_DNS**2)
c_2_PENG = ((ww_DNS/k_DNS - 2/3) + 2*(uu_DNS/k_DNS - 2/3))/(tau_peng_DNS**2*dudy_DNS**2)
# c_0 = np.trapz(c_0,yplus_DNS)
# c_2 = np.trapz(c_2,yplus_DNS)

# print(c_0)
# print(c_2)

# c_0 = 0.16
# c_2 = 0.2

#Way to get weighted mean
# c_0konst = np.mean(np.trapz(c_0,y_DNS))
# c_2konst = np.mean(np.trapz(c_2,y_DNS))

ww_tau_DNS = ((c_0_DNS)*(tau_DNS**2*dudy_DNS**2)/(-6) + 2/3)*k_DNS
uu_tau_DNS = ((1/12)*tau_DNS**2*dudy_DNS**2*((c_0_DNS) + 6*(c_2_DNS)) + 2/3)*k_DNS
vv_tau_DNS = ((1/12)*tau_DNS**2*dudy_DNS**2*((c_0_DNS) - 6*(c_2_DNS)) + 2/3)*k_DNS

ww_tau_AKN = ((c_0_AKN)*(tau_akn_DNS**2*dudy_DNS**2)/(-6) + 2/3)*k_DNS
uu_tau_AKN = ((1/12)*tau_akn_DNS**2*dudy_DNS**2*((c_0_AKN) + 6*(c_2_AKN)) + 2/3)*k_DNS
vv_tau_AKN = ((1/12)*tau_akn_DNS**2*dudy_DNS**2*((c_0_AKN) - 6*(c_2_AKN)) + 2/3)*k_DNS

ww_tau_RANS = ((c_0_RANS)*(tau_rans_DNS**2*dudy_DNS**2)/(-6) + 2/3)*k_DNS
uu_tau_RANS = ((1/12)*tau_rans_DNS**2*dudy_DNS**2*((c_0_RANS) + 6*(c_2_RANS)) + 2/3)*k_DNS
vv_tau_RANS = ((1/12)*tau_rans_DNS**2*dudy_DNS**2*((c_0_RANS) - 6*(c_2_RANS)) + 2/3)*k_DNS

ww_tau_PENG = ((c_0_PENG)*(tau_peng_DNS**2*dudy_DNS**2)/(-6) + 2/3)*k_DNS
uu_tau_PENG = ((1/12)*tau_peng_DNS**2*dudy_DNS**2*((c_0_PENG) + 6*(c_2_PENG)) + 2/3)*k_DNS
vv_tau_PENG = ((1/12)*tau_peng_DNS**2*dudy_DNS**2*((c_0_PENG) - 6*(c_2_PENG)) + 2/3)*k_DNS

#-----------------Plotting--------------------
fig1= plt.figure()
plt.subplots_adjust(left=0.25,bottom=0.20)
plt.plot(uu_tau_DNS,yplus_DNS,'b--',label = "tau DNS")
plt.plot(uu_tau_AKN,yplus_DNS,'k',label = "tau AKN")
plt.plot(uu_tau_RANS,yplus_DNS,'c',label = "tau RANS")
plt.plot(uu_tau_PENG,yplus_DNS,'g',label = "tau PENG")
# plt.plot(uu_DNS,yplus_DNS,'r--',label = "DNS")

plt.xlabel("$\overline{u'u'}^+$")
plt.ylabel("$y^+$")
plt.legend(loc="best",fontsize=12)
    
plt.show()
# fig2= plt.figure()
# plt.subplots_adjust(left=0.25,bottom=0.20)
# plt.plot(vv,yplus_DNS,'b',label = "Approx")
# plt.plot(vv_DNS,yplus_DNS,'r--', label = "DNS")
# #plt.axis([0, 1.5, 10,5000])
# plt.ylim(1,5000)
# plt.xlabel("$\overline{v'v'}^+$")
# plt.ylabel("$y^+$")
# plt.legend(loc="best",fontsize=16)

# fig3 = plt.figure()
# plt.subplots_adjust(left=0.25,bottom=0.20)
# plt.plot(ww,yplus_DNS,'b', label = "Approx")
# plt.plot(ww_DNS,yplus_DNS,'r--', label = "DNS")
# #plt.axis([0, 1.5, 10,5000])
# plt.ylim(1,5000)
# plt.xlabel("$\overline{w'w'}^+$")
# plt.ylabel("$y^+$")
# plt.legend(loc="best",fontsize=16)

# plt.figure()
# plt.plot(tau_DNS,yplus_DNS,label = "DNS")
# plt.plot(tau_akn_DNS,yplus_DNS, label = "AKN")
# plt.plot(tau_rans_DNS,yplus_DNS, label = "RANS")
# plt.plot(tau_peng_DNS,yplus_DNS,label = "PENG")

# plt.legend(loc = "best", fontsize = 12)
# plt.show()