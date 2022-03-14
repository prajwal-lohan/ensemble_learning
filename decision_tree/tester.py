from decision_tree.DecisionTreeRegressor import DecisionTreeRegressor

# import decision_tree.Classifier as cls

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

    # Regression Example. Comment out the unwanted dataset
    # 1st test trial dataset    ####################################
    N = 200
    X = np.linspace(-1, 1, N)
    y = X ** 2 + np.random.normal(0, 0.07, N)
    ################################################################

    # 2nd test trial dataset    ####################################
    X = np.sort(5 * np.random.rand(80, 1), axis=0)
    y = np.sin(X).ravel()
    y[::5] += 3 * (0.5 - np.random.rand(16))
    ################################################################

    X = X.reshape(-1, 1)

    # fit X and y and then predict with sklearn dt and my dt to compare
    d = 4
    reg = DecisionTreeRegressor(criterion='mse', max_depth=d)
    skreg = tree.DecisionTreeRegressor(criterion='mse', max_depth=d)

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


    # Visualizing tree with load_boston dataset
    from sklearn.datasets import load_boston

    boston = load_boston()
    X, y = boston.data, boston.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    reg = DecisionTreeRegressor(criterion='mse', max_depth=3)
    reg.fit(X, y)

    pred = reg.predict(X_test)
    print("R2 score = ", reg.score(pred, y_test))

    reg.show_tree(boston.feature_names)
