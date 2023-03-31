import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

viscos = 1 / 5200

# load DNS data
DNS_mean = np.genfromtxt("LM_Channel_5200_mean_prof.dat", comments="%")
y_DNS = DNS_mean[:, 0]
u_DNS = DNS_mean[:, 2]

DNS_stress = np.genfromtxt("LM_Channel_5200_vel_fluc_prof.dat", comments="%")
uv_DNS = DNS_stress[:, 5]

# load RANS data
y_u_k_om_uv = np.loadtxt("y_u_k_om_uv_5200-RANS-code.txt")

y = y_u_k_om_uv[:, 0]
u = y_u_k_om_uv[:, 1]
k = y_u_k_om_uv[:, 2]
om = y_u_k_om_uv[:, 3]
uv = y_u_k_om_uv[:, 4]

# use only half channel
nj2 = int(len(y) / 2)
y = y[0:nj2]
u = u[0:nj2]
k = k[0:nj2]
om = om[0:nj2]

dudy = np.gradient(u, y)
vist = k / om

# turbulence model: uv = -cmu*k/omega*dudy
# Input data: dudy*k/omega, uv
# output, to be predicted: cmu

# input uv: interpolate uv_DNS to k-omega grid
uv_DNS_all_data = np.interp(y, y_DNS, uv_DNS)
# input dudy*k/omega
k_over_omega_all_data = k / om * dudy
# output cmu
cmu_all_data = uv_DNS_all_data / k_over_omega_all_data

# create indices for all data
index = np.arange(0, len(k_over_omega_all_data), dtype=int)

# number of elements of test data, 20%
n_test = int(0.2 * len(k_over_omega_all_data))

# the rest is for training data
n_svr = len(k_over_omega_all_data) - n_test

# pick 20% elements randomly (test data)
index_test = np.random.choice(index, size=n_test, replace=False)
k_over_omega_test = k_over_omega_all_data[index_test]
uv_DNS_test = uv_DNS_all_data[index_test]
cmu_out_test = cmu_all_data[index_test]

# delete testing data from 'all data' => training data
k_over_omega_in = np.delete(k_over_omega_all_data, index_test)
uv_DNS_in = np.delete(uv_DNS_all_data, index_test)
cmu_out = np.delete(cmu_all_data, index_test)

# re-shape
k_over_omega_in = k_over_omega_in.reshape(-1, 1)
uv_DNS_in = uv_DNS_in.reshape(-1, 1)

# scale input data 
scaler_k_over_omega = StandardScaler()
scaler_uv = StandardScaler()
k_over_omega_in = scaler_k_over_omega.fit_transform(k_over_omega_in)
uv_DNS_in = scaler_uv.fit_transform(uv_DNS_in)

# setup X (input) and y (output)
X = np.zeros((n_svr, 2))
y = cmu_out
X[:, 0] = k_over_omega_in[:, 0]
X[:, 1] = uv_DNS_in[:, 0]

print('starting SVR')

# choose Machine Learning model
C = 0.1
eps = 0.001
# use Linear model
model = LinearSVR(epsilon=eps, C=C)

# Fit the model
svr = model.fit(X, y.flatten())

#  re-shape test data
k_over_omega_test = k_over_omega_test.reshape(-1, 1)
uv_DNS_test = uv_DNS_test.reshape(-1, 1)

# scale test data
k_over_omega_test = scaler_k_over_omega.transform(k_over_omega_test)
uv_DNS_test = scaler_uv.transform(uv_DNS_test)

# setup X (input) for testing (predicting)
X_test = np.zeros((n_test, 2))
X_test[:, 0] = k_over_omega_test[:, 0]
X_test[:, 1] = uv_DNS_test[:, 0]

# predict cmu
cmu_predict = model.predict(X_test)

# invert scaling
k_over_omega_test_no_scale = \
    scaler_k_over_omega.inverse_transform(k_over_omega_test)
uv_DNS_test_no_scale = \
    scaler_uv.inverse_transform(uv_DNS_test)

# flatten
uv_DNS_test_no_scale = uv_DNS_test_no_scale.flatten()

# compute uv
uv_predict = k_over_omega_test_no_scale.flatten() * cmu_predict.flatten()

# find difference between ML prediction and target
uv_rms = np.std(uv_predict - uv_DNS_test_no_scale) / \
         (np.mean(uv_predict ** 2)) ** 0.5 / (np.mean(uv_DNS_test_no_scale ** 2)) ** 0.5
print('\nRMS error using ML turbulence model', uv_rms)

# find difference between standard model (constant cmu) and target
cmu = -1
uv_standard = k_over_omega_test_no_scale.flatten() * cmu
uv_rms = np.std(uv_standard - uv_DNS_test_no_scale) / \
         (np.mean(uv_standard ** 2)) ** 0.5 / (np.mean(uv_DNS_test_no_scale ** 2)) ** 0.5
print('\nRMS error using standard turbulence model', uv_rms)
