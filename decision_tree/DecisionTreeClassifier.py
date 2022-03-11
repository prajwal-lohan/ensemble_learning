#Import libraries
import numpy as np

'''
Function to calculate the gini of the node 
'''

def calculate_gini(node):
  #Create a list of classes 
  classes=[] 
  for row in node:
    classes.append((row[-1])) 
  list_of_classes= np.unique(classes) 

  #For each class we calculate the total number of instances of that class in the node
  n_instances=[] 
  for cls in list_of_classes: 
    n_instance_class=sum(classes==cls) 
    n_instances.append(n_instance_class)
  
  #Finally calculate the gini for that node. We get the node size by summing up all the instances and the gini is returned 
  node_size=sum(n_instances) 
  gini_node=1
  for n in n_instances:  
    if node_size==0: 
      continue 
    gini_node-=(n/node_size)**2
  return gini_node

'''
Function to get the best split and store node information in a dictionary

'''

def get_split(node): 
  #We take each feature and value and calculate the gini at this split. By the end we will have found the best split. 
  best_gini,best_feature, best_split=1, None, None 
  for k in range(n_features): 
    for j in range(len(node)): 
      current_split_value=node[j][k] 
      left, right=list(),list() 
      for i in range(len(node)):  
        if node[i][k]<=current_split_value:  
          left.append(node[i])
        else:
          right.append(node[i])

      #calculate gini at current split
      region_size=len(left)+len(right) 
      weighted_gini=len(left)/(region_size)*calculate_gini(left) +len(right)/(region_size)*calculate_gini(right) 

      #if current gini calc is better than best gini so far we update the information on the best split 
      if weighted_gini<best_gini: 
        best_gini, best_feature, best_split, best_left,best_right=weighted_gini, k, current_split_value, left, right
  
  #Store node information in a dictionary
  return {'index':best_feature, 'value': best_split, 'left':best_left, 'right':best_right, 'gini':best_gini} 

'''
Function to assign classes to the terminal nodes  
''' 

def final_label(node):
  classes=[]
  for row in node:
    classes.append((row[-1]))

  #Create a dictionary with a count of each class and we assign the final label as the class with the maximum number of counts
  node_dict=dict((i, classes.count(i)) for i in classes)  
  label=max(node_dict, key=node_dict.get) 
  print(label, ":",sum(node_dict.values()),node_dict)
  return label


'''
Function to recursively split nodes out. 
'''
def recursive_split(node, max_depth, depth, min_size):
  left, right=node['left'], node['right']
  del node['left'], node['right']
  
  #Checking whether the node is a terminal node, or whether the impurity =0 - if so we will assign a label. 
  if not left or not right or node['gini']==0:
    node['left']=node['right']=final_label(left+right)
    return

  #Checking if we have exceeded the maximum depth  - otherwise we will keep calling the function
  if depth>=max_depth:
    node['left'], node['right']=final_label(left), final_label(right)
    return

  #Checking if we are below the minimum size - otherwise we will keep calling the function
  if len(left)<=min_size:
    node['left']=final_label(left)
  else:
    node['left']=get_split(left)
    recursive_split(node['left'], max_depth, depth+1, min_size) 
    
  if len(right)<=min_size:
    node['right']=final_label(right)

  else:
    node['right']=get_split(right) 
    recursive_split(node['right'], max_depth, depth+1, min_size)     

'''
Function to build out the tree 
'''
#We start with the initial full dataset and it keeps calling the recursive split function until the constraints are violated

def build_tree(node, max_depth, min_size):
  root=get_split(dataset) 
  recursive_split(root, max_depth, min_size, 1) 
  return root

'''
Function to print the tree 
'''
def print_tree(node, depth=0):
  #If the node is a dictionary, it means it is not a terminal node, so it keeps traversing the tree until a terminal node is reached
  if type(node)==dict: 
    print(depth*"  ","X", node['index'], "<=", node['value'])
    print_tree(node['left'], depth+1)
    print_tree(node['left'], depth+1)
  else:
    print(depth*" ", node)

'''
Function to predict the label, given the input dataset
'''

def predict(node, row): 
  #We traverse the tree until we reach a terminal node, and return the label that is assigned to this node. 
	if row[node['index']] <= node['value']:
		if type(node['left'])==dict:
			return predict(node['left'], row) 
		else:
			return node['left']
	else:
		if type(node['right'])==dict:
			return predict(node['right'], row)
		else:
			return node['right']


'''
Function to finally build out the whole tree and return a list of predictions
'''

def final_tree(dataset, max_depth, min_size):
  tree=build_tree(dataset, max_depth, min_size) 
  predictions=list()
  for row in dataset: 
    prediction=predict(tree, row)
    predictions.append(prediction)
  return predictions

'''
Function to return the accuracy of our predictions
'''
def accuracy(predictions, Y):
  #We take the number of correct predictions/total length of the test set
  correct=0
  for i in range(len(Y)):
    if predictions[i]==Y[i]:
      correct+=1

  accuracy=correct/len(Y)
  print("Accuracy: ", accuracy)

'''
Function to create one dataset by merging X and Y. 
'''

def dataset(X, Y):
  dataset=list()
  for i in range(len(X)):
    row_concat=np.append(X[i],Y[i])
    dataset.append(row_concat)
  return dataset

#tests

#Get X, Y
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
N=100
X,Y=make_moons(N, noise=0.2)
plt.scatter(X[:,0], X[:,1],c=Y)
plt.show()


#Run and output terminal nodes, accuracy
n_features=X.shape[1]
dataset=dataset(X,Y)
accuracy(final_tree(dataset, max_depth=1, min_size=1), Y)

