import numpy as np
import torch 

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


# Create matrixes and so on
# -------------------------------------
# Delete first value for all interesting data
uv_DNS = np.delete(uv_DNS, 0)
vv_DNS = np.delete(vv_DNS, 0)
ww_DNS = np.delete(ww_DNS, 0)
uw_DNS = np.delete(uw_DNS,0)
vw_DNS = np.delete(vw_DNS,0)
k_DNS = np.delete(k_DNS, 0)
eps_DNS = np.delete(eps_DNS, 0)
dudy_DNS = np.delete(dudy_DNS, 0)

#Calculate ny_t and time-scale tau
viscous_t = k_DNS**2/eps_DNS 
tau = viscous_t/abs(uv_DNS)

#Calculate c_1,c_2,c_3
for i in range(len(k_DNS)):
    B_matrix = (1/12)*(tau[i]*dudy_DNS[i])**2*np.array([[1,6,1],[1,-6,1],[-2,0,-2]])

    a_matrix = np.array([[uu_DNS[i]**2/k_DNS[i] - 2/3],\
        [vv_DNS[i]**2/k_DNS[i] - 2/3],\
        [ww_DNS[i]**2/k_DNS[i] - 2/3]])

    # print(B_matrix)
    # print(a_vector)

    #try:  # Inte en bra lösning lol
    c = np.linalg.solve(B_matrix, a_matrix)
    print(c)


#ML using pytorch to estimate c_1,c_2,c_3