import numpy as np
import matplotlib.pyplot as plt
from dphidx_dy import dphidx_dy
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR  # for building SVR model
import plotly.graph_objects as go  # for data visualization
import plotly.express as px  # for data visualization

# read data file
tec = np.genfromtxt("tec.dat", dtype=None, comments="%")

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
eps_DNS = tec[:, 9]
k_DNS = 0.5 * (uu + vv + ww)

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
k2d = np.reshape(k_DNS, (nj, ni))  # =mean{0.5(v'_iv'_i)}

u2d = np.transpose(u2d)
v2d = np.transpose(v2d)
p2d = np.transpose(p2d)
x2d = np.transpose(x2d)
y2d = np.transpose(y2d)
uu2d = np.transpose(uu2d)
vv2d = np.transpose(vv2d)
uv2d = np.transpose(uv2d)
k2d = np.transpose(k2d)

# set Neumann of p at upper and lower boundaries
p2d[:, 1] = p2d[:, 2]
p2d[:, -1] = p2d[:, -1 - 1]

# x and y are fo the cell centers. The dphidx_dy routine needs the face coordinate, xf2d, yf2d
# load them
xc_yc = np.loadtxt("xc_yc.dat")
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


indexDrop = []
for i in range(len(k_DNS)):
    if k_DNS[i] == 0:
        indexDrop.append(i)

eps_DNS = np.delete(eps_DNS, indexDrop)
k_DNS = np.delete(k_DNS, indexDrop)
uv = np.delete(uv, indexDrop)
dudy = np.delete(dudy, indexDrop)
dvdx = np.delete(dvdx, indexDrop)

cmy_DNS = ((-uv * eps_DNS) / (k_DNS**2 * (dudy + dvdx)))
np.savetxt('k_DNS', k_DNS, delimiter=',')
np.savetxt('cmy_DNS', cmy_DNS, delimiter=',')
np.savetxt('eps_DNS', eps_DNS, delimiter=',')


x = np.linspace(0, len(cmy_DNS), len(cmy_DNS))
plt.figure('test')
plt.scatter(x, cmy_DNS)

plt.show()
