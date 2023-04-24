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
y= data[:,0]
u= data[:,1]
k= data[:,2]
om= data[:,3]
diss1=0.09*k*om
ustar=(viscos*u[0]/y[0])**0.5
yplus=y*ustar/viscos

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

# Calculate ny_t and time-scale tau
viscous_t = k_DNS**2/eps_DNS 
tau = viscous_t/abs(uv_DNS)

# Calculate c_1, c_2 of the Non-linear Eddy Viscosity Model
# Array for storing c_1, c_2, & c_3

c_0 = -2*(ww_DNS/k_DNS - 2/3)/(tau**2*dudy_DNS**2)
c_2 = 2*((ww_DNS/k_DNS - 2/3) + (uu_DNS/k_DNS - 2/3))/(tau**2*dudy_DNS**2)

ww = (c_0*(tau**2*dudy_DNS**2)/(-2) + 2/3)*k_DNS
uu = ((1/12)*tau**2*dudy_DNS**2*(c_0 + 6*c_2) + 2/3)*k_DNS
vv = ((1/12)*tau**2*dudy_DNS**2*(c_0 - 6*c_2) + 2/3)*k_DNS

#-----------------Plotting--------------------
fig1 = plt.figure()
plt.subplots_adjust(left=0.25,bottom=0.20)
plt.plot(vv,yplus_DNS,'b')
plt.plot(vv_DNS,yplus_DNS,'r--')
plt.axis([0, 1.5, 10,5000])
plt.xlabel("$\overline{u'u'}^+$")
plt.ylabel("$y^+$")
plt.legend(loc="best",fontsize=16)

plt.show()
