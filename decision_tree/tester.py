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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    reg = DecisionTreeRegressor(criterion='mse', max_depth=4)
    skreg = tree.DecisionTreeRegressor(criterion='mse', max_depth=4)

    reg.fit(X, y)
    skreg.fit(X, y)

    # reg.fit(X_train, y_train)
    # skreg.fit(X_train, y_train)

    my_pred = reg.predict(X)
    sk_pred = skreg.predict(X)

    # my_pred = reg.predict(X_test)
    # sk_pred = skreg.predict(X_test)

    # print("My dt result: ", my_pred)
    # print("Sklearn dt result: ", sk_pred)

    # plt.scatter(X_train, y_train)
    # plt.scatter(X_test, y_test)

    print("R2 score = ",reg.score(my_pred,y))
    print("sklearn score = ",skreg.score(X,y))

    plt.scatter(X, y, c="b", linewidths=0.1)
    plt.plot(X, sk_pred, c="g")
    plt.plot(X, my_pred, c="r")
    plt.show()


    reg.show_tree(["dummy_feature"])