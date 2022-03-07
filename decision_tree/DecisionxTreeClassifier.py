#libraries
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import collections

#sample dataset
from sklearn.datasets import make_moons
N=1000
X,Y=make_moons(N, noise=0.2)
plt.scatter(X[:,0], X[:,1],c=Y)
plt.show()

def calculate_gini(X):
  classes= np.unique(Y) #first we get the number of classes from our Y
  n_instances=[] #create a list that will store the number of each class for that node
  for cls in classes: #for each class
    n_instance_class=sum(X==cls) #calculate the total number of instances of that class in the node
    n_instances.append(n_instance_class)
  
  #now calculate the gini for that node
  node_size=sum(n_instances) #get node size by summing up all the instances 
  gini_node=1
  for n in n_instances:  #for each instance in each class: 
    if node_size==0: 
      continue 
    gini_node-=(n/node_size)**2
  return gini_node
  
  n_features=X.shape[1] #get number of variables from dataset
d=dict(zip(list(range(n_features)), sorted(zip(*X)))) #for each feature/var for the full dataset we add all values to a dictionary. 

def get_split(X, Y):
  best_gini=1 #initiate best gini value
  best_feature, best_split=None, None
  for k in X.keys(): #for each key (feature/variable)
    feature_number=list(X)[k] #which variable/feature number we are splitting by e.g first of two vars
    for j in range(len(Y)): #for each value in dictionary key 
      current_split_value=X[k][j] #we split by this value
      left_node_class, right_node_class=[],[] #create empty lists for the classes of left and right nodes 
      left_node_x, right_node_x= collections.defaultdict(list),collections.defaultdict(list) #create empty lists for the left and right nodes variable values 
      for i in range(len(Y)):  #iterate over each value in the dataset
        if d[k][i]<=current_split_value: #if current datapoint is less than split value it goes into left node, else right 
          left_node_class.append(Y[i])
          left_node_x[feature_number].append(X[feature_number][i]) #we append it and need to append the corresponding other x values 
          left_node_x[0].append(X[0][i]) #****
        else:
          right_node_class.append(Y[i]) 
          right_node_x[feature_number].append(X[feature_number][i]) #we append it and need to append the corresponding other x values **
          right_node_x[0].append(X[0][i]) 

      global best_node_left, best_node_left_x, best_node_right, best_node_right_x
      #calculate gini at current split
      region_size=len(left_node_class)+len(right_node_class) #first get number of values in each node
      weighted_gini=len(left_node_class)/(region_size)*calculate_gini(left_node_class) +len(right_node_class)/(region_size)*calculate_gini(right_node_class)  #calculate the weighted gini
      #print('X',feature_number,  current_split_value, 'Gini=', weighted_gini)
      if weighted_gini<best_gini: #if current gini calc is better than best gini so far:
        best_gini, best_feature, best_split, best_node_left,best_node_right, best_node_left_x,best_node_right_x =weighted_gini, feature_number, current_split_value, left_node_class, right_node_class, right_node_x #set new best values
#print the split
  print("Best split is X",best_feature, "<", best_split, "Gini=", best_gini )

def final_label(node):
  node_dict=dict((i, node.count(i)) for i in node) #create a dictionary with a count of each class 
  label=max(node_dict, key=node_dict.get) #take class with maximum number of counts
  print("label:", label)
  return label

def recursive_split(X, max_depth, depth):
  if len(best_node_right)==0: #if a left or right node is empty or we are at a depth larger than the max_depth of the tree, it is a terminal node so assign label to it 
    right_child_node=final_label(best_node_right)
    #print("right node terminates")
    return
  if len(best_node_left)==0:
    left_child_node=final_label(best_node_left)
    return
  if depth>=max_depth:
    final_label(best_node_right)
    final_label(best_node_left)
    return
  else:
    left_child_node=get_split(best_node_left_x, best_node_left) #else we will continue splitting and building out the tree
    recursive_split(left_child_node, max_depth, depth+1) 
    right_child_node=get_split(best_node_right_x, best_node_right) 
    recursive_split(right_child_node, max_depth, depth+1) 
  
  def predict():
  
