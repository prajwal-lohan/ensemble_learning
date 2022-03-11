import numpy as np
import pydot


class DecisionTreeRegressor():
    def __init__(self,
                 criterion='mse',
                 max_depth=None,
                 min_samples_split=None,
                 min_samples_leaf=None,
                 min_impurity_decrease=None):

        self.n_features = None
        self.split_feature = None  # Feature used to decide the split
        self.split_value = None  # Value to decide the split
        self.value = None  # Value assigned (if a leaf node)
        self.n_samples = None  # Number of samples in the node
        self.impurity_decrease = None  # Impurity decreased by splitting the node at best feature and value
        self.left_child = None  # Left child node
        self.right_child = None  # Right child node
        self.depth = 0  # Depth of tree till this node
        self.criterion = criterion  # Criterion to find node impurity
        self.max_depth = max_depth  # Maximum depth allowed for growing the tree

        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease

        # todo: add value checks? like passing -ve min_impurity_decrease should not be allowed

    def fit(self, X, y):

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.n_features = X.shape[1]

        # grow the tree recursively starting from root node
        self.grow_tree(X, y)

    def predict(self, X):

        result = []
        for row in range(X.shape[0]):
            # result.append(self.root.predict_one_row(X[row, :]))
            result.append(self.predict_one_row(X[row, :]))

        return np.array(result)

    def predict_one_row(self, row):
        if self.split_feature is None:
            return self.value

        else:
            if row[self.split_feature] <= self.split_value:
                return self.left_child.predict_one_row(row)
            else:
                return self.right_child.predict_one_row(row)

    def grow_tree(self, X, y):

        self.n_samples = X.shape[0]
        self.value = np.mean(y)

        # Stop growing tree according to stopping conditions
        if self.depth >= self.max_depth:
            return

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
                left_indices = X[:, col] <= col_mean
                right_indices = X[:, col] > col_mean

                y_left = y[left_indices]
                y_right = y[right_indices]

                impurity_left = self.node_impurity(y_left)
                impurity_right = self.node_impurity(y_right)

                impurity_dec = node_impurity - ((len(y_left) * impurity_left + len(y_right) * impurity_right) / len(y))
                # impurity_dec = node_impurity - (impurity_left + impurity_right)
                # todo: check which one to use

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


    def show_tree(self, feature_names):
        if self.n_features is None:
            raise ValueError("Decision tree must be fitted before visualizing")
        if len(feature_names) != self.n_features:
            raise ValueError(
                "Length of feature_names should be same as number of features in dataset used for fitting the tree")

        graph = pydot.Dot(graph_type='digraph', strict=True)

        # self.node_num = 0
        node_name = "root"
        if self.split_feature is not None:
            feature_line = str(feature_names[self.split_feature]) + " <= " + str(round(self.split_value, 3)) + "\n"
        else:
            feature_line = ""
        root_label = feature_line + \
                     "samples = " + str(self.n_samples) + "\n" + "value = " + str(round(self.value, 3))

        root_node = pydot.Node(name=node_name, label=root_label)
        graph.add_node(root_node)

        self.add_child_nodes(graph, node_name, feature_names)

        graph.write_png("Decision_Tree.png")

    # def add_child_nodes(self):
    def add_child_nodes(self, graph, parent_node_name, feature_names):
        # If no child nodes exist, don't do anything
        if self.split_feature is None:
            return

        # Children exist

        # Add left child
        left_node_name = parent_node_name + "_left"

        # Add feature split line in label if left child has further children
        if self.left_child.split_feature is not None:
            left_feature_line = str(feature_names[self.left_child.split_feature]) + " <= " + \
                                str(round(self.left_child.split_value, 3)) + "\n"
        else:
            left_feature_line = ""

        left_label = left_feature_line + "samples = " + str(self.left_child.n_samples) + "\n" + "value = " + str(
            round(self.left_child.value, 3))
        left_node = pydot.Node(name=left_node_name, label=left_label)
        graph.add_node(left_node)
        left_edge = pydot.Edge(parent_node_name, left_node_name)
        graph.add_edge(left_edge)

        # Add children of left child recursively
        self.left_child.add_child_nodes(graph, left_node_name, feature_names)

        # Add right child
        right_node_name = parent_node_name + "_right"

        # Add feature split line in label if right child has further children
        if self.right_child.split_feature is not None:
            right_feature_line = str(feature_names[self.right_child.split_feature]) + " <= " + \
                                 str(round(self.right_child.split_value, 3)) + "\n"
        else:
            right_feature_line = ""

        right_label = right_feature_line + "samples = " + str(self.right_child.n_samples) + "\n" + \
                      "value = " + str(round(self.right_child.value, 3))
        right_node = pydot.Node(name=right_node_name, label=right_label)
        graph.add_node(right_node)
        right_edge = pydot.Edge(parent_node_name, right_node_name)
        graph.add_edge(right_edge)

        # Add children of right child recursively
        self.right_child.add_child_nodes(graph, right_node_name, feature_names)
