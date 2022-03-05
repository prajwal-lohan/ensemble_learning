import numpy as np


class DecisionTreeRegressor():
    def __init__(self,
                 criterion='mse',
                 max_depth=None,
                 min_samples_split=None,
                 min_samples_leaf=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=None,
                 min_impurity_split=None):

        self.n_samples = None
        self.impurity_decrease = None
        self.left_child = None
        self.right_child = None
        self.depth = 0
        self.root = None
        self.criterion = criterion
        self.max_depth = max_depth

    def fit(self, X, y):
        # initialise root node
        self.root = DecisionTreeRegressor()

        # grow the tree recursively starting from root node
        self.root.grow_tree(self, X, y)
        return

    def predict(self, X):

        return

    def grow_tree(self, X, y):

        # find best split
        self.best_split(self,X,y)


        # split the tree using this best split
        self.split_tree(self, X, y)

        return

    def split_tree(self):

        return

    def best_split(self):

        return

    def node_impurity(self, criterion, y):
        if criterion == 'mse':
            return np.mean((y - np.mean(y)) ** 2)
        elif criterion == 'mae':
            return
        return
