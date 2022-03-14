# Import libraries
import numpy as np
from collections import Counter


class Node:
    def __init__(self):
        self.index = None
        self.value = None
        self.left = None
        self.right = None
        self.gini = None
        self.depth = 0


class DecisionTreeClassifier:
    def __init__(self,
                 max_depth=None,
                 min_size=None):

        self.n_features = None
        self.max_depth = max_depth
        self.min_size = min_size  # minimum size of the node
        self.node = Node()

        '''
    Function to calculate the gini of the node 
    '''

    def calculate_gini(self, node):
        # Create a list of classes
        classes = []
        for row in node:
            classes.append((row[-1]))
        list_of_classes = np.unique(classes)

        # For each class we calculate the total number of instances of that class in the node
        n_instances = []
        for cls in list_of_classes:
            n_instance_class = sum(classes == cls)
            n_instances.append(n_instance_class)

        # Finally calculate the gini for that node. We get the node size by summing up all the instances and the gini is returned
        node_size = sum(n_instances)
        gini_node = 1
        for n in n_instances:
            if node_size == 0:
                continue
            gini_node -= (n / node_size) ** 2
        return gini_node

    '''
  Function to get the best split and store node information in a dictionary

  '''

    def get_split(self, node):
        # We take each feature and value and calculate the gini at this split. By the end we will have found the best split.
        best_gini = self.calculate_gini(node)
        best_feature, best_split = None, None
        best_left, best_right = None, None
        for k in range(self.n_features):
            for j in range(len(node)):
                current_split_value = node[j][k]
                left, right = list(), list()
                for i in range(len(node)):
                    if node[i][k] <= current_split_value:
                        left.append(node[i])
                    else:
                        right.append(node[i])

                # calculate gini at current split
                region_size = len(left) + len(right)
                weighted_gini = (len(left) * self.calculate_gini(left) + len(right) * self.calculate_gini(
                    right)) / region_size

                # if current gini calc is better than best gini so far we update the information on the best split
                if weighted_gini < best_gini:
                    best_gini, best_feature, best_split, best_left, best_right = weighted_gini, k, current_split_value, left, right

        result_node = Node()
        result_node.index = best_feature
        result_node.value = best_split
        result_node.left = best_left
        result_node.right = best_right
        result_node.gini = best_gini
        return result_node

    '''
  Function to assign classes to the terminal nodes  
  '''

    def final_label(self, node):
        list_1 = []
        for row in node:
            list_1.append((row[-1]))
        # Create a dictionary with a count of each class and we assign the final label as the class with the maximum number of counts
        node_dict = dict((i, list_1.count(i)) for i in list_1)
        label = max(node_dict, key=node_dict.get)
        # print(label, ":",sum(node_dict.values()),node_dict)
        return label

    '''
  Function to recursively split nodes out. 

  '''

    def recursive_split(self, node):
        left = node.left
        right = node.right
        if node.depth < self.max_depth:
            if node.index is not None:

                if left is not None and len(left) > 1:
                    temp_left = self.get_split(left)
                    if temp_left.index is not None:
                        node.left = temp_left
                        node.left.depth = node.depth + 1
                        self.recursive_split(node.left)
                    else:
                        node.left = self.final_label(left)

                if right is not None and len(right) > 1:
                    temp_right = self.get_split(right)
                    if temp_right.index is not None:
                        node.right = temp_right
                        node.right.depth = node.depth + 1
                        self.recursive_split(node.right)
                    else:
                        node.right = self.final_label(right)

        else:
            node.left = self.final_label(left)
            node.right = self.final_label(right)
        return

    '''
  Function to finally build out the whole tree and return a list of predictions
  '''

    def fit(self, X, y):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.n_features = X.shape[1]

        dataset = self.dataset(X, y)

        root = self.get_split(dataset)  # initially split the dataset
        # grow the tree recursively starting from root
        self.recursive_split(root)
        self.node = root

        return root

    def predict(self, X):
        root = self.node
        predictions = []
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        for row in range(X.shape[0]):
            predictions.append(self.predict_one_row(X[row, :], root))  # ***tree
        return np.array(predictions)

    '''
  Function to predict the label, given the input dataset
  '''

    def predict_one_row(self, row, node):
        # We traverse the tree until we reach a terminal node, and return the label that is assigned to this node.

        if isinstance(node, Node):
            if row[node.index] <= node.value:
                return self.predict_one_row(row, node.left)
            else:
                return self.predict_one_row(row, node.right)
        elif isinstance(node, list):
            labels = [int(l[-1]) for l in node]
            return Counter(labels).most_common(1)[0][0]
        else:
            return int(node)

    '''
  Function to print the tree 
  '''

    def print_tree(self, node=None, depth=0):
        # If the node is a dictionary, it means it is not a terminal node, so it keeps traversing the tree until a terminal node is reached
        if node is None:
            node = self.node

        if isinstance(node, Node):
            depth = node.depth
            print(depth * "  ", "X", node.index, "<=", node.value)
            self.print_tree(node.left, depth + 1)
            self.print_tree(node.right, depth + 1)
        elif isinstance(node, list):
            labels = [int(l[-1]) for l in node]
            print(depth * " ", labels)
        else:
            print(depth * " ", int(node))

    '''
  Function to create one dataset by merging X and Y. 
  '''

    def dataset(self, X, Y):
        dataset = list()
        for i in range(len(X)):
            row_concat = np.append(X[i], Y[i])
            dataset.append(row_concat)
        return dataset

    '''
  Function to return the accuracy of our predictions
  '''

    def accuracy(self, predictions, Y):
        # We take the number of correct predictions/total length of the test set
        correct = 0
        for i in range(len(Y)):
            if predictions[i] == Y[i]:
                correct += 1

        accuracy = correct / len(Y)
        print("Accuracy: ", accuracy)

    # tests


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # Testing on make_moons data
    from sklearn.datasets import make_moons

    X, Y = make_moons(100, noise=0.2)
    clf = DecisionTreeClassifier(max_depth=4)
    # vars(clf)
    clf.fit(X, Y)
    pred = clf.predict(X)
    clf.accuracy(pred, Y)
    # clf.print_tree()

    plot_step = 0.02

    # Display of the decision surface
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    plt.xlabel(1)
    plt.ylabel(2)
    plt.axis("tight")


    n_classes = 3
    plot_colors = "by"  # blue-red-yellow
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(Y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=i, cmap=plt.cm.Paired)
        # plt.scatter(X[idx, 0], X[idx, 1], label=i, cmap=plt.cm.Paired)
    plt.axis("tight")
    plt.suptitle("Decision surface of a decision tree using paired features")
    plt.legend()
    # plt.savefig('fig.png')
    plt.show()
