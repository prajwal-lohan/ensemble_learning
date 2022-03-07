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

        self.split_feature = None  # Feature used to decide the split
        self.split_value = None  # Value to decide the split
        self.value = None  # Value assigned (if a leaf node)
        self.n_samples = None  # Number of samples in the node
        self.impurity_decrease = None  # Impurity decreased by splitting the node at best feature and value
        self.left_child = None  # Left child node
        self.right_child = None  # Right child node
        self.depth = 0  # Depth of tree till this node
        self.root = None  # Root node needed when first initialising the tree
        self.criterion = criterion  # Criterion to find node impurity
        self.max_depth = max_depth  # Maximum depth allowed for growing the tree

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

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # grow the tree recursively starting from root node
        self.root.grow_tree(X, y)

    def predict(self, X):

        return

    def grow_tree(self, X, y):

        self.value = np.mean(y)

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

        if (self.min_impurity_decrease is not None) and (self.impurity_decrease < self.min_impurity_decrease):
            return
        # todo: check impurity decrease condition

        # split the tree using this best split
        self.split_tree(X, y)

        return

    def split_tree(self, X, y):

        # Split X and y according to split_value
        left_indices = X[:, self.split_feature] <= self.split_value
        right_indices = X[:, self.split_feature] > self.split_value

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
        max_impurity_dec = 0.0
        split_feature = None
        split_value = None

        node_impurity = self.node_impurity(y)

        for col in range(X.shape[1]):
            col_unique = np.unique(X[:, col])
            col_means = np.convolve(col_unique, np.ones(2), 'valid') / 2
            for col_mean in col_means:
                left_indices = y[:, col] <= col_mean
                right_indices = y[:, col] > col_mean

                y_left = y[left_indices]
                y_right = y[right_indices]

                impurity_left = self.node_impurity(y_left)
                impurity_right = self.node_impurity(y_right)

                impurity_dec = node_impurity - ((len(y_left) * impurity_left + len(y_right) * impurity_right) / len(y))

                if impurity_dec > max_impurity_dec:
                    max_impurity_dec = impurity_dec
                    split_feature = col
                    split_value = col_mean

        self.split_feature = split_feature
        self.split_value = split_value
        self.impurity_decrease = max_impurity_dec  # Used to compare with min_impurity_dec to decide whether to split or not

    def node_impurity(self, y):

        if self.criterion == 'mse':
            return np.mean((y - np.mean(y)) ** 2)

        elif self.criterion == 'mae':
            return np.mean(y - np.mean(y))  # todo: is this correct?
        return
