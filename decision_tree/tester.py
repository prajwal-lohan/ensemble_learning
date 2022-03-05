from sklearn import tree
import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# reg = my decision tree regressor
skreg = tree.DecisionTreeRegressor()

# # Classification dataset
N = 1000
X, Y = make_moons(N, noise=0.2)
# plt.scatter(X[:, 0], X[:, 1], c=Y)
# plt.show()

# Regression dataset
N = 200
X = np.linspace(-1, 1, N)
Y = X ** 2 + np.random.normal(0, 0.07, N)
X = X.reshape(-1, 1)
plt.scatter(X, Y)
plt.show()

# fit X and y and then predict with sklearn dt and my dt to compare