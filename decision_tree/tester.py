from decision_tree.DecisionTreeRegressor import DecisionTreeRegressor
from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # # Classification Example
    # N = 1000
    # X, Y = make_moons(N, noise=0.2)
    # plt.scatter(X[:, 0], X[:, 1], c=Y)
    # plt.show()

    # Regression Example
    N = 200
    X = np.linspace(-1, 1, N)
    y = X ** 2 + np.random.normal(0, 0.07, N)
    X = X.reshape(-1, 1)
    # plt.scatter(X, y,c="b")
    # plt.show()

    # fit X and y and then predict with sklearn dt and my dt to compare
    reg = DecisionTreeRegressor(criterion='mse', max_depth=4)
    skreg = tree.DecisionTreeRegressor(criterion='mse', max_depth=4)

    reg.fit(X, y)
    skreg.fit(X, y)

    my_pred = reg.predict(X)
    sk_pred = skreg.predict(X)

    print("R2 score = ", reg.score(my_pred, y))
    print("sklearn score = ", skreg.score(X, y))

    plt.scatter(X, y, c="b", linewidths=0.1)
    plt.plot(X, sk_pred, c="g")
    plt.plot(X, my_pred, c="r")
    plt.show()

    reg.show_tree(["dummy_feature"])
