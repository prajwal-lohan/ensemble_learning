import numpy as np


class DecisionTreeRegressor():
    def __init__(self,
                 criterion='mse',
                 max_depth=None,
                 min_samples_split=None,
                 min_samples_leaf=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=None):

        self.split_feature = None
        self.split_value = None
        self.n_samples = None
        self.impurity_decrease = None
        self.left_child = None
        self.right_child = None
        self.depth = 0
        self.root = None
        self.criterion = criterion
        self.max_depth = max_depth

        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease

    def fit(self, X, y):
        # initialise root node
        self.root = DecisionTreeRegressor(criterion=self.criterion,
                                          max_depth=self.max_depth,
                                          min_impurity_decrease=self.min_impurity_decrease)

        # grow the tree recursively starting from root node
        self.root.grow_tree(X, y)

    def predict(self, X):

        return

    def grow_tree(self, X, y):

        # Stop growing tree according to stopping conditions
        if self.depth >= self.max_depth:
            return

        self.n_samples = X.shape[0]

        # find best split
        self.best_split(X, y)

        # todo: Question: if for best split, min_samples_leaf is not satisfied for one child,
        #  but it is satisfied for another case of split (not max impurity decrease), what to do?
        #  can this situation ever occur?

        if self.split_feature is None:
            return

        # todo: check impurity decrease condition

        # split the tree using this best split
        self.split_tree(X, y)

        return

    def split_tree(self, X, y):

        # Split X and y according to split_value
        left_indices = [X[:, self.split_feature <= self.split_value]]
        right_indices = [X[:, self.split_feature > self.split_value]]

        X_left = X[left_indices]
        y_left = y[left_indices]

        X_right = X[right_indices]
        y_right = y[right_indices]

        # Create left child and grow it recursively
        self.left_child = DecisionTreeRegressor(criterion=self.criterion,
                                                max_depth=self.max_depth,
                                                min_impurity_decrease=self.min_impurity_decrease)
        self.left_child.depth = self.depth + 1
        self.left_child.grow_tree(X_left, y_left)

        # Create right child and grow it recursively
        self.right_child = DecisionTreeRegressor(criterion=self.criterion,
                                                 max_depth=self.max_depth,
                                                 min_impurity_decrease=self.min_impurity_decrease)
        self.right_child.depth = self.depth + 1
        self.right_child.grow_tree(X_right, y_right)

    def best_split(self, X, y):

        return

    def node_impurity(self, criterion, y):

        if criterion == 'mse':
            return np.mean((y - np.mean(y)) ** 2)

        elif criterion == 'mae':
            return np.mean(y - np.mean(y))  # todo: is this correct?
        return
