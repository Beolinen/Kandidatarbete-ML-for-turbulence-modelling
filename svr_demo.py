# Importera biblioteket
from sklearn.svm import SVR
import numpy as np

# Skapa data för x-funktionen
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel()
noise = np.random.randint(-1.5,1.5,100) * np.random.uniform(low=0, high=0.5, size=(100,))
y = y + noise

# Skapa en SVR-modell
C = 0.5
gamma = 0.1
svr_rbf = SVR(kernel='rbf', C=C, gamma=gamma)

# Träna modellen
svr_rbf.fit(X, y)

# Skapa testdata
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]

# Gör en prediktion med modellen
y_pred = svr_rbf.predict(X_test)

# Plotta resultaten
import matplotlib.pyplot as plt
plt.scatter(X, y, color='darkorange', label='SVR prediction')
plt.plot(X_test, y_pred, color='navy', label='Hyperplan')
plt.plot(X_test, y_pred-C, linestyle='dashed', color='red', label='SVR-eps')
plt.plot(X_test, y_pred+C, linestyle='dashed', color='red', label='SVR+eps')

plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()




