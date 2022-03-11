
# Implementation of a decision tree from scratch

## Classification

### General overview: how it works
The output of a call on DecisionTreeClassifier is a dictionary that contains all information relating to a classification decision tree. Everytime a node is split, it will store this information within this dictionary, and this nested dictionary procedure continues until our constraints are met: either we have reached the desired maximum depth of our tree, or the node size is less than our desired minimum size, or the node is completely pure (only contains one class). 

### Criteria:
In this case we have used the Gini coefficient as the criteria to split the nodes, given it is the most popular criterion in the classification case. 

### Attributes 

### Functions
Below you can find a short description of the functions. 

calculate_gini: this calculates the gini of any node. 

get_split: given the parent node, this returns the information relating to the two child nodes, which are found based on the smallest gini. It returns a dictionary containing the node's index to split on, the value to split by, the rows from the dataset in the left and right node, and the gini at this optimal split. 

final_label: this assigns a class to the terminal nodes


recursive_split: allows our tree to continue splitting until our constraints are violated. 

grow_tree: builds out our tree 


predict_one_row: given a row in the dataset, it outputs the predicted class.

predict: predicts a class for each row in the dataset and appends this to a list. 

dataset: merges our X and Y dataset into one numpy array for simplicity

fit: fits our tree on the training data

accuracy: returns the accuracy of our predictions on the test set. 
