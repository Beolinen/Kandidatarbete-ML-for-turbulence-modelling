#----------------Imports----------------

import numpy as np
import torch 
import matplotlib.pyplot as plt
import warnings
import matplotlib.cbook

#----------------Parameters----------------
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

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

# Calculate c_1, c_2, & c_3 of the Non-linear Eddy Viscosity Model
# Array for storing c_1, c_2, & c_3
c = np.zeros((3, len(k_DNS)))

for i in range(len(k_DNS)):

    # Equation 14.2
    a_vector = np.array([[uu_DNS[i]**2/k_DNS[i] - 2/3],\
        [vv_DNS[i]**2/k_DNS[i] - 2/3],\
        [ww_DNS[i]**2/k_DNS[i] - 2/3]])
    
    # Equation 14.4
    B_matrix = (1/12)*(tau[i]*dudy_DNS[i])**2*np.array([[1,6,1],[1,-6,1],[-2,0,-2]])

    # Since B is a singular matrix, we use the pseudo-inverse to approximate c
    B_pinv = np.linalg.pinv(B_matrix)
    c[:,i] = np.dot(B_pinv, a_vector).flatten()


# Calculate c_1 average
c_1_avg = np.mean(c[0,:])
c_2_avg = np.mean(c[1,:])
c_3_avg = np.mean(c[2,:])
print("c_1 average: ", c_1_avg)
print("c_2 average: ", c_2_avg)
print("c_3 average: ", c_3_avg)


# ML using pytorch to estimate c_1, c_2, & c_3




#Calculating target using approximation
uu = (1/12)*(tau[i]*dudy_DNS[i])**2*(c[0,:] + 6*c[1,:] + c[2,:])
vv = (1/12)*(tau[i]*dudy_DNS[i])**2*(c[0,:] - 6*c[1,:] + c[2,:])
ww = (1/12)*(tau[i]*dudy_DNS[i])**2*(-2*c[0,:] - 2*c[2,:])


#-----------------Plotting--------------------

fig1 = plt.figure()
plt.subplots_adjust(left=0.25,bottom=0.20)
plt.plot(uu,yplus_DNS,'b-',label='RANS')
plt.plot(uu_DNS,yplus_DNS,'r--',label='DNS')
plt.axis([0, 10, 0, 5200])
plt.xlabel("$\overline{u'u'}^+$")
plt.ylabel("$y^+$")
plt.legend(loc="best",fontsize=16)

fig2 = plt.figure()
plt.subplots_adjust(left=0.25,bottom=0.20)
plt.plot(abs(vv),yplus_DNS,'b-',label='RANS')
plt.plot(vv_DNS,yplus_DNS,'r--',label='DNS')
plt.axis([0, 3, 0, 5200])
plt.xlabel("$\overline{v'v'}^+$")
plt.ylabel("$y^+$")
plt.legend(loc="best",fontsize=16)

fig3 = plt.figure()
plt.subplots_adjust(left=0.25,bottom=0.20)
plt.plot(abs(ww),yplus_DNS,'b-',label='RANS')
plt.plot(ww_DNS,yplus_DNS,'r--',label='DNS')
plt.axis([0, 3, 0, 5200])
plt.xlabel("$\overline{w'w'}^+$")
plt.ylabel("$y^+$")
plt.legend(loc="best",fontsize=16)

plt.show()

#u'u' decent, w'w' decent, v'v' bad, not expected
