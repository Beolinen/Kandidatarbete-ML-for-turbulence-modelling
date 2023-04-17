import scipy.io as sio
import numpy as np
import sys
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

plt.interactive(True)
plt.close('all')

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

print('ustar',ustar)

dudy=np.gradient(u,y)

nj=len(y)

uu=np.zeros(nj)
vv=np.zeros(nj)
ww=np.zeros(nj)
uv=np.zeros(nj)

# compute uu, vv, ww with EARSM
# S. Wallin and A. V. Johansson, 
# “An explicit algebraic Reynolds stress model for incompressible and compressible turbulent flows,” 
# J. Fluid Mech. 403, 89 (2000).

for j in range (0,nj-1):

    diss=diss1[j]
    rk=k[j]
    ttau=rk/diss

    om12=ttau*0.5*dudy[j]
    om21=-om12
    om22=0.
    om11=0.
    s11=0.
    s12=ttau*0.5*dudy[j]
    s21=s12
    s22=0.
    s33=0.
    vor=(-2.*om12**2)
    str1=(s11**2+s12**2+s21**2+s22**2)

      
    cpr1=9./4.*(1.8-1.)
    p1=(1./27.*cpr1**2+9./20.*str1-2./3.*vor)*cpr1
    p2=p1**2-(1./9.*cpr1**2+9./10.*str1+2./3.*vor)**3
      
    if p2 >  0:
       if p1-p2**0.5 >= 0:
          sigg=1.
       else:
          sigg=-1.
       un=cpr1/3.+(p1+p2**0.5)**(1./3.)+sigg*(abs(p1-p2**0.5))**(1./3.)
    else:
       un=cpr1/3.+2.*(p1**2-p2)**(1./6.)*np.cos(1./3.*np.arccos(p1/(np.sqrt(p1**2-p2))))
      
    const=6./5.
    beta1=-const*un/(un**2-2.*vor)
    beta4=beta1/un

    uu[j]=2./3.*rk+rk*beta1*s11+rk*beta4*(s12*om21-om12*s21)
    vv[j]=2./3.*rk+rk*beta1*s22+rk*beta4*(s21*om12-om21*s12)
    ww[j]=2./3.*rk
    uv[j]=rk*(beta1*s12+beta4*(s11*om12-om12*s22))

#************ U
fig2 = plt.figure()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.semilogx(yplus,u,'b-',label='RANS')
plt.semilogx(yplus_DNS,u_DNS,'r--',label='DNS')
plt.ylabel("$U^+$")
plt.xlabel("$y^+$")
plt.axis([1, 5200, 0, 28])
plt.legend(loc="best",fontsize=16)
plt.savefig('pictures/u_log_5200-channel.png')


#************ uu
fig2 = plt.figure()
plt.subplots_adjust(left=0.25,bottom=0.20)
plt.plot(uu,yplus,'b-',label='RANS')
plt.plot(uu_DNS,yplus_DNS,'r--',label='DNS')
plt.axis([0, 10, 0, 5200])
plt.xlabel("$\overline{u'u'}^+$")
plt.ylabel("$y^+$")
plt.legend(loc="best",fontsize=16)
plt.savefig('pictures/uu_python.png')

#************ vv
fig2 = plt.figure()
plt.subplots_adjust(left=0.25,bottom=0.20)
plt.plot(vv,yplus,'b-',label='RANS')
plt.plot(vv_DNS,yplus_DNS,'r--',label='DNS')
plt.axis([0, 1.5, 0, 5200])
plt.xlabel("$\overline{v'v'}^+$")
plt.ylabel("$y^+$")
plt.legend(loc="best",fontsize=16)
plt.plot(vv_DNS,yplus_DNS,'r--')
plt.savefig('pictures/vv_python.png')


#************ ww
fig2 = plt.figure()
plt.subplots_adjust(left=0.25,bottom=0.20)
plt.plot(ww,yplus,'b-',label='RANS')
plt.plot(ww_DNS,yplus_DNS,'r--',label='DNS')
plt.axis([0, 3.0, 0, 5200])
plt.xlabel("$\overline{v'v'}^+$")
plt.ylabel("$y^+$")
plt.legend(loc="best",fontsize=16)
plt.plot(ww_DNS,yplus_DNS,'r--')
plt.savefig('pictures/ww_python.png')


#************ uv
fig2 = plt.figure()
plt.subplots_adjust(left=0.25,bottom=0.20)
plt.plot(uv,yplus,'b-',label='RANS')
plt.plot(uv_DNS,yplus_DNS,'r--',label='DNS')
plt.axis([-1, 0 , 0, 5200])
plt.xlabel("$\overline{u'v'}^+$")
plt.ylabel("$y^+$")
plt.plot(uv_DNS,yplus_DNS,'r--')
plt.savefig('pictures/uv_python.png')