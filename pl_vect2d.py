import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from dphidx_dy import dphidx_dy
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR  # for building SVR model
import plotly.graph_objects as go  # for data visualization
import plotly.express as px  # for data visualization

plt.rcParams.update({'font.size': 22})
plt.interactive(True)

# read data file
tec=np.genfromtxt("/Users/benjaminjonsson/Programmering/Kandidat/small_wave/tec.dat", dtype=None,comments="%")

# text='VARIABLES = X Y P U V u2 v2 w2 uv mu_sgs prod'

x = tec[:, 0]
y = tec[:, 1]
p = tec[:, 2]
u = tec[:, 3]
v = tec[:, 4]
uu = tec[:, 5]
vv = tec[:, 6]
ww = tec[:, 7]
uv = tec[:, 8]
k = 0.5 * (uu + vv + ww)

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

u2d = np.reshape(u, (nj, ni))
v2d = np.reshape(v, (nj, ni))
p2d = np.reshape(p, (nj, ni))
x2d = np.reshape(x, (nj, ni))
y2d = np.reshape(y, (nj, ni))
uu2d = np.reshape(uu, (nj, ni))  # =mean{v'_1v'_1}
uv2d = np.reshape(uv, (nj, ni))  # =mean{v'_1v'_2}
vv2d = np.reshape(vv, (nj, ni))  # =mean{v'_2v'_2}
k2d = np.reshape(k, (nj, ni))  # =mean{0.5(v'_iv'_i)}

u2d = np.transpose(u2d)
v2d = np.transpose(v2d)
p2d = np.transpose(p2d)
x2d = np.transpose(x2d)
y2d = np.transpose(y2d)
uu2d = np.transpose(uu2d)
vv2d = np.transpose(vv2d)
uv2d = np.transpose(uv2d)
k2d = np.transpose(k2d)

# set periodic b.c on west boundary
# u2d[0,:]=u2d[-1,:]
# v2d[0,:]=v2d[-1,:]
# p2d[0,:]=p2d[-1,:]
# uu2d[0,:]=uu2d[-1,:]


# read k and eps from a 2D RANS simulations. They should be used for computing the damping function f
k_eps_RANS = np.loadtxt("/Users/benjaminjonsson/Programmering/Kandidat/small_wave/k_eps_RANS.dat")
k_RANS = k_eps_RANS[:, 0]
diss_RANS = k_eps_RANS[:, 1]
vist_RANS = k_eps_RANS[:, 2]

ntstep = k_RANS[0]

k_RANS_2d = np.reshape(k_RANS, (ni, nj)) / ntstep  # modeled turbulent kinetic energy
diss_RANS_2d = np.reshape(diss_RANS, (ni, nj)) / ntstep  # modeled dissipation
vist_RANS_2d = np.reshape(vist_RANS, (ni, nj)) / ntstep  # turbulent viscosity, AKN model

# set small values on k & eps at upper and lower boundaries to prevent NaN on division
diss_RANS_2d[:, 0] = 1e-10
k_RANS_2d[:, 0] = 1e-10
vist_RANS_2d[:, 0] = nu
diss_RANS_2d[:, -1] = 1e-10
k_RANS_2d[:, -1] = 1e-10
vist_RANS_2d[:, -1] = nu

# set Neumann of p at upper and lower boundaries
p2d[:, 1] = p2d[:, 2]
p2d[:, -1] = p2d[:, -1 - 1]

# x and y are fo the cell centers. The dphidx_dy routine needs the face coordinate, xf2d, yf2d
# load them
xc_yc = np.loadtxt("/Users/benjaminjonsson/Programmering/Kandidat/small_wave/xc_yc.dat")
xf = xc_yc[:, 0]
yf = xc_yc[:, 1]
xf2d = np.reshape(xf, (nj, ni))
yf2d = np.reshape(yf, (nj, ni))
xf2d = np.transpose(xf2d)
yf2d = np.transpose(yf2d)

# delete last row
xf2d = np.delete(xf2d, -1, 0)
yf2d = np.delete(yf2d, -1, 0)
# delete last columns
xf2d = np.delete(xf2d, -1, 1)
yf2d = np.delete(yf2d, -1, 1)

# compute the gradient dudx, dudy at point P
dudx = np.zeros((ni, nj))
dudy = np.zeros((ni, nj))
dvdx = np.zeros((ni, nj))
dvdy = np.zeros((ni, nj))

dudx, dudy = dphidx_dy(xf2d, yf2d, u2d)
dvdx, dvdy = dphidx_dy(xf2d, yf2d, v2d)

df = pd.read_csv('/Users/benjaminjonsson/Programmering/Kandidat/k_eps_RANS2.csv', encoding='utf-8', sep=';')

k_rans = df['k'].to_numpy()
e_rans = df['e'].to_numpy()
ny_rans = df['ny'].to_numpy()

indexDrop = []
E = k_rans
K = e_rans
NY = ny_rans

for i in range(len(e_rans)):
    if e_rans[i] == 0:
        indexDrop.append(i)

E = np.delete(k_rans, indexDrop)
K = np.delete(e_rans, indexDrop)
NY = np.delete(ny_rans, indexDrop)

cmy_rans = np.array((NY * E)/(K**2))

#duidxj = np.array(np.sqrt(dudx**2 + dudy**2 + dvdx**2 + dvdy**2))
duidxj = np.array(np.sqrt(dudx**2))
duidxj = np.delete(duidxj, indexDrop)

#ML-metod
duidxj = duidxj.reshape(-1,1)

scaler = MinMaxScaler()

duidxj_scaled = []
duidxj_scaled = scaler.fit_transform(duidxj[:])

X = duidxj_scaled
Y = cmy_rans

model = SVR(kernel = 'rbf', C = 1, epsilon = 10)

SVR = model.fit(X,Y)
xrange = np.linspace(X.min(),X.max(),len(X))

y_svr = model.predict(xrange.reshape(-1,1))
y_svr_no_scale = scaler.inverse_transform(y_svr.reshape(-1,1))
y_svr_no_scale = y_svr_no_scale.flatten()

#Plotta lösning
plt.scatter(xrange,y_svr_no_scale)
plt.savefig("bild.png")

#TEST med nytt case
k_eps_RANS = np.loadtxt("/Users/benjaminjonsson/Programmering/Kandidat/large_wave/k_eps_RANS_large.dat")
k_RANS_large = k_eps_RANS[:, 0]
diss_RANS_large = k_eps_RANS[:, 1]
ny_t_RANS_large = k_eps_RANS[:,2]

k_RANS_large = np.delete(k_RANS_large,indexDrop)
diss_RANS_large = np.delete(diss_RANS_large,indexDrop)
ny_t_RANS_large = np.delete(ny_t_RANS_large,indexDrop)

ny_t_predict = y_svr_no_scale*k_RANS_large**2/diss_RANS_large
ny_t_standard = 0.09*k_RANS_large**2/diss_RANS_large

errorML = (np.std(ny_t_predict - ny_t_RANS_large))/(np.mean(ny_t_predict**2))**0.5/(np.mean(ny_t_RANS_large**2))**0.5
error = (np.std(ny_t_standard - ny_t_RANS_large))/(np.mean(ny_t_standard**2))**0.5/(np.mean(ny_t_RANS_large**2))**0.5
print("Felet med ML är", errorML)
print("Felet med standardmodell är", error)

print(np.mean(y_svr_no_scale))
