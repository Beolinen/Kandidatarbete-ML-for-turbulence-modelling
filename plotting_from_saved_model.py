#----------------Imports----------------

import numpy as np
import torch 
import matplotlib.pyplot as plt
import warnings
import matplotlib.cbook
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import StandardScaler
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
#print(dudy_DNS)

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
# uv_DNS = np.delete(uv_DNS, 0)
# vv_DNS = np.delete(vv_DNS, 0)
# ww_DNS = np.delete(ww_DNS, 0)
# uw_DNS = np.delete(uw_DNS,0)
# vw_DNS = np.delete(vw_DNS,0)
# k_DNS = np.delete(k_DNS, 0)
# eps_DNS = np.delete(eps_DNS, 0)
# dudy_DNS = np.delete(dudy_DNS, 0)
# yplus_DNS = np.delete(yplus_DNS,0)
# uu_DNS = np.delete(uu_DNS,0)
# tau_DNS = np.delete(tau_DNS,0)
# tau_akn_DNS = np.delete(tau_akn_DNS,0)
# tau_rans_DNS = np.delete(tau_rans_DNS,0)
# tau_peng_DNS = np.delete(tau_peng_DNS,0)
#print(yplus_DNS[6])
uv_DNS = uv_DNS[10:-1:]
vv_DNS = vv_DNS[10:-1:]
ww_DNS = ww_DNS[10:-1:]
uw_DNS = uw_DNS[10:-1:]
vw_DNS = vw_DNS[10:-1:]
k_DNS = k_DNS[10:-1:]
eps_DNS = eps_DNS[10:-1:]
dudy_DNS = dudy_DNS[10:-1:]
yplus_DNS = yplus_DNS[10:-1:]
uu_DNS = uu_DNS[10:-1:]
tau_DNS = tau_DNS[10:-1:]
tau_akn_DNS = tau_akn_DNS[10:-1:]
tau_rans_DNS = tau_rans_DNS[10:-1:]
tau_peng_DNS = tau_peng_DNS[10:-1:]

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

def reshape_those_fuckers(*args):
    return [arg.reshape(-1,1) for arg in args]


dudy_squared_DNS = (dudy_DNS**2).reshape(-1,1)
dudy_DNS = dudy_DNS.reshape(-1,1)
dudy_squared_DNS_scaled = StandardScaler().fit_transform(dudy_squared_DNS)
dudy_DNS_scaled = StandardScaler().fit_transform(dudy_DNS)
X = np.concatenate((dudy_DNS,dudy_squared_DNS_scaled),axis=1)


# transpose the target vector to make it a column vector  
c = np.array([c_0_DNS,c_2_DNS])
y = c.transpose()


#tau, dudy, k, uu, vv, ww, yplus_DNS
test_var = np.concatenate((reshape_those_fuckers(tau_DNS,dudy_DNS,k_DNS,uu_DNS,vv_DNS,ww_DNS,yplus_DNS, c[0,:],c[1,:])),axis=1)

# split the feature matrix and target vector into training and validation sets
# test_size=0.2 means we reserve 20% of the data for validation
# random_state=42 is a fixed seed for the random number generator, ensuring reproducibility

# random_state = randrange(100)
random_state = 90

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,random_state= random_state)
X_train, X_val, test_var_train, test_var_val = train_test_split(X, test_var, test_size=0.2,random_state= random_state)

# convert the numpy arrays to PyTorch tensors with float32 data type
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# create PyTorch datasets and dataloaders for the training and validation sets
# a TensorDataset wraps the feature and target tensors into a single dataset
# a DataLoader loads the data in batches and shuffles the batches if shuffle=True
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class ThePredictionMachine(nn.Module):

    def __init__(self):
        
        super(ThePredictionMachine, self).__init__()

        self.input   = nn.Linear(2, 50)     
        self.hidden1 = nn.Linear(50, 25) 
        self.hidden_1 = nn.Linear(25,25)
        self.hidden_2 = nn.Linear(25,25)
        self.hidden2 = nn.Linear(25, 2)     

    def forward(self, x):
        x = nn.functional.relu(self.input(x))
        x = nn.functional.relu(self.hidden1(x))
        x = nn.functional.relu(self.hidden_1(x))
        x = nn.functional.relu(self.hidden_2(x))
        x = self.hidden2(x)
        return x

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches

    print(f"Avg loss: {test_loss:>8f} \n")

model = torch.load("trained_models/nn_model_c0_c2.pt")
preds = model(X_val_tensor)
c_NN = preds.detach().numpy()

#tau, dudy, k, uu, vv, ww, yplus, c_0,c_2
ww_NN = ((c_NN[:,0])*(test_var_val[:,0]**2*test_var_val[:,1]**2)/(-6) + 2/3)*test_var_val[:,2]
uu_NN = ((1/12)*test_var_val[:,0]**2*test_var_val[:,1]**2*((c_NN[:,0]) + 6*(c_NN[:,1])) + 2/3)*test_var_val[:,2]
vv_NN = ((1/12)*test_var_val[:,0]**2*test_var_val[:,1]**2*((c_NN[:,0]) - 6*(c_NN[:,1])) + 2/3)*test_var_val[:,2]

c0 = 0.16
c2 = 0.11

ww_const = ((c0)*(test_var_val[:,0]**2*test_var_val[:,1]**2)/(-6) + 2/3)*test_var_val[:,2]
uu_const = ((1/12)*test_var_val[:,0]**2*test_var_val[:,1]**2*((c0) + 6*(c2)) + 2/3)*test_var_val[:,2]
vv_const = ((1/12)*test_var_val[:,0]**2*test_var_val[:,1]**2*((c0) - 6*(c2)) + 2/3)*test_var_val[:,2]


c0 = np.mean(c_NN[:,0])
c2 = np.mean(c_NN[:,1])

ww_NN_const = ((c0)*(test_var_val[:,0]**2*test_var_val[:,1]**2)/(-6) + 2/3)*test_var_val[:,2]
uu_NN_const = ((1/12)*test_var_val[:,0]**2*test_var_val[:,1]**2*((c0) + 6*(c2)) + 2/3)*test_var_val[:,2]
vv_NN_const = ((1/12)*test_var_val[:,0]**2*test_var_val[:,1]**2*((c0) - 6*(c2)) + 2/3)*test_var_val[:,2]


#-----------------Plotting--------------------
# fig1, (ax0,ax1,ax2)= plt.subplots(nrows = 3 ,ncols = 1, sharex = True, figsize = (6,9))
# ax0.scatter(test_var_val[:,3],test_var_val[:,6],s = 10, marker = "o",color = "r",label = "DNS")
# ax0.scatter(uu_const,test_var_val[:,6],s = 10,marker = "o", color = "b" ,label = "const c")
# ax0.scatter(uu_NN_const,test_var_val[:,6],s = 10,marker = "o", color = "m" ,label = "mean c from NN")
# ax0.scatter(uu_NN,test_var_val[:,6],s = 10,marker = "o", color = "k" ,label = "NN")

# ax0.axis([-5,15,0,5200])
# ax0.set_xlabel("$\overline{u'u'}^+$")
# ax0.set_ylabel("$y^+$")
# ax0.set_title("Approximating Reynolds stresses using a Neural Network")
# ax0.legend(loc="best",fontsize=12)

# ax1.scatter(test_var_val[:,4],test_var_val[:,6],s = 10, marker = "o",color = "r",label = "DNS")
# ax1.scatter(vv_const,test_var_val[:,6],s = 10,marker = "o", color = "b" ,label = "const c")
# ax1.scatter(vv_NN_const,test_var_val[:,6],s = 10,marker = "o", color = "m" ,label = "const c")
# ax1.scatter(vv_NN,test_var_val[:,6],s = 10,marker = "o", color = "k" ,label = "NN")

# ax1.axis([-5,15,0,5200])
# ax1.set_xlabel("$\overline{v'v'}^+$")
# ax1.set_ylabel("$y^+$")
# # ax1.set_title("Approximating $\overline{v'v'}$")
# ax1.legend(loc="best",fontsize=12)

# ax2.scatter(test_var_val[:,5],test_var_val[:,6],s = 10, marker = "o",color = "r",label = "DNS")
# ax2.scatter(ww_const,test_var_val[:,6],s = 10,marker = "o", color = "b" ,label = "const c")
# ax2.scatter(ww_NN_const,test_var_val[:,6],s = 10,marker = "o", color = "m" ,label = "const c")
# ax2.scatter(ww_NN,test_var_val[:,6],s = 10,marker = "o", color = "k" ,label = "NN")

# ax2.axis([-5,15,0,5200])
# plt.ylabel("$y^+$")
# ax2.legend(loc="best",fontsize=12)
# # ax2.set_title("Approximating $\overline{w'w'}$")
# ax2.set_xlabel("$\overline{w'w'}$")
# fig1.savefig("plots/NN-model.png")

# fig2, (ax3,ax4)= plt.subplots(nrows = 2, ncols = 1, sharex = True)
# ax3.scatter(test_var_val[:,7],test_var_val[:,6], s = 10,marker = "o", color = "r", label = "Target")
# ax3.scatter(c_NN[:,0],test_var_val[:,6], s = 10,marker = "o", color = "b", label = "NN")
# ax3.axis([-0.25,1,0,5000])
# ax3.set_title("Approximating $c_0$ and $c_2$ using Neural Network")
# ax3.set_xlabel("$c_0$")
# ax3.set_ylabel("$y^+$")
# ax3.legend(loc = "best", fontsize = 12)

# ax4.scatter(test_var_val[:,8],test_var_val[:,6], s = 10,marker = "o", color = "r", label = "Target")
# ax4.scatter(c_NN[:,1],test_var_val[:,6], s = 10,marker = "o", color = "b", label = "NN")
# ax4.axis([-0.25,1,0,5000])
# #ax4.set_title("Approximating $c_2$ using Neural Network")
# ax4.set_xlabel("$c_2$")
# ax4.set_ylabel("$y^+$")
# ax4.legend(loc = "best", fontsize = 12)
# fig2.savefig("plots/c_approximation_NN.png")

#------------------------TEST WITH NEW DATA--------------------
DNS_mean=np.genfromtxt("vel_11000_DNS.dat",comments="%")
y_DNS=DNS_mean[:,0]
yplus_DNS=DNS_mean[:,1]
u_DNS=DNS_mean[:,2]
dudy_DNS= np.gradient(u_DNS,y_DNS)

uu_DNS = DNS_mean[:,3]**2
vv_DNS = DNS_mean[:,4]**2
ww_DNS = DNS_mean[:,5]**2
k_DNS = 0.5*(uu_DNS + vv_DNS + ww_DNS)

DNS_stress=np.genfromtxt("bud_11000.prof",comments="%")
eps_DNS = DNS_stress[:,4]

tau_DNS = k_DNS/eps_DNS

yplus_DNS = np.delete(yplus_DNS,0)
dudy_DNS = np.delete(dudy_DNS,0)
uu_DNS = np.delete(uu_DNS,0)
vv_DNS = np.delete(vv_DNS,0)
ww_DNS = np.delete(ww_DNS,0)
k_DNS = np.delete(k_DNS,0)
eps_DNS = np.delete(eps_DNS,0)
tau_DNS = np.delete(tau_DNS,0)

yplus_DNS = np.delete(yplus_DNS,-1)
dudy_DNS = np.delete(dudy_DNS,-1)
uu_DNS = np.delete(uu_DNS,-1)
vv_DNS = np.delete(vv_DNS,-1)
ww_DNS = np.delete(ww_DNS,-1)
k_DNS = np.delete(k_DNS,-1)
eps_DNS = np.delete(eps_DNS,-1)
tau_DNS = np.delete(tau_DNS,-1)

dudy_squared_DNS = (dudy_DNS**2).reshape(-1,1)
dudy_DNS = dudy_DNS.reshape(-1,1)

dudy_squared_DNS_scaled = StandardScaler().fit_transform(dudy_squared_DNS)
dudy_DNS_scaled = StandardScaler().fit_transform(dudy_DNS)

X = np.concatenate((dudy_DNS,dudy_squared_DNS_scaled),axis=1)
X_val_tensor = torch.tensor(X, dtype=torch.float32)

preds2 = model(X_val_tensor)
c_NN = preds2.detach().numpy()

c_0_DNS = -6*(ww_DNS/k_DNS - 2/3)/(tau_DNS**2*dudy_DNS[:,0]**2)
c_2_DNS = ((ww_DNS/k_DNS - 2/3) + 2*(uu_DNS/k_DNS - 2/3))/(tau_DNS**2*dudy_DNS[:,0]**2)

ww_NN = ((c_NN[:,0])*(tau_DNS**2*dudy_DNS[:,0]**2)/(-6) + 2/3)*k_DNS
uu_NN = ((1/12)*tau_DNS**2*dudy_DNS[:,0]**2*((c_NN[:,0]) + 6*(c_NN[:,1])) + 2/3)*k_DNS
vv_NN = ((1/12)*tau_DNS**2*dudy_DNS[:,0]**2*((c_NN[:,0]) - 6*(c_NN[:,1])) + 2/3)*k_DNS

c0 = 0.16
c2 = 0.11

ww_const = ((c0)*(tau_DNS**2*dudy_DNS**2)/(-6) + 2/3)*k_DNS
uu_const = ((1/12)*tau_DNS**2*dudy_DNS**2*((c0) + 6*(c2)) + 2/3)*k_DNS
vv_const = ((1/12)*tau_DNS**2*dudy_DNS**2*((c0) - 6*(c2)) + 2/3)*k_DNS

fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
ax=plt.gca()
plt.scatter(c_0_DNS,yplus_DNS,s = 15, marker = "o",color = "r",label = "DNS")
plt.scatter(c_NN[:,0][0::4],yplus_DNS[0::4],s = 20,marker = "o", color = "b" ,alpha = 0.5,label = "NN test")
# plt.plot(c_0_DNS,yplus_DNS,color = "r", label = "DNS")
# plt.plot(c_NN[:,0],yplus_DNS,color = "b", label = "NN test")
plt.axis([0,1000,0,7000])
plt.xlabel("$c_0$")
plt.ylabel("$y^+$")
plt.legend(loc = "lower right", fontsize = 12)
plt.title("Test of trained model")

axins1 = inset_axes(ax1, width="30%", height="30%", loc='upper left', borderpad=2)
plt.scatter(c_0_DNS,yplus_DNS,s = 15, marker = "o",color = "r",label = "DNS")
plt.scatter(c_NN[:,0][0::4],yplus_DNS[0::4],s = 20,marker = "o", color = "b" ,alpha = 0.5,label = "NN test")
# plt.plot(c_0_DNS,yplus_DNS,color = "r", label = "DNS")
# plt.plot(c_NN[:,0],yplus_DNS,color = "b", label = "NN test")
plt.axis([0,0.1,0,7000])
axins1.set_yticks([])
axins1.set_xticks([0,0.05,0.1])
plt.xlabel("$c_0$")
plt.savefig("plots/c_0_test.png")

fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
ax=plt.gca()
plt.scatter(c_2_DNS,yplus_DNS,s = 15, marker = "o",color = "r",label = "DNS")
plt.scatter(c_NN[:,1][0::4],yplus_DNS[0::4],s = 20,marker = "o", color = "b" ,alpha = 0.5,label = "NN test")
plt.axis([0,1000,0,7000])
plt.xlabel("$c_2$")
plt.ylabel("$y^+$")
plt.legend(loc = "lower right", fontsize = 12)
plt.title("Test of trained model")

axins1 = inset_axes(ax1, width="50%", height="50%", loc='upper right', borderpad=0.1)
plt.scatter(c_2_DNS,yplus_DNS,s = 15, marker = "o",color = "r",label = "DNS")
plt.scatter(c_NN[:,1][0::4],yplus_DNS[0::4],s = 20,marker = "o", color = "b" ,alpha = 0.5,label = "NN test")
plt.axis([0,0.1,0,7000])
axins1.set_yticks([])
axins1.set_xticks([0,0.05,0.1])
plt.xlabel("$c_2$")
plt.savefig("plots/c_2_test.png")

plt.show()
