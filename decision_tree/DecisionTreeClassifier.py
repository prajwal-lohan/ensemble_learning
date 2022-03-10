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

#calc gini for the node  
def calculate_gini(node):
  classes=[] #create a list for the classes
  for row in node:
    classes.append((row[-1])) #we create a list of classes (Y)
  
  list_of_classes= np.unique(classes) #first we get the number of classes from our Y
  n_instances=[] #create a list that will store the number of each class for that node
  for cls in list_of_classes: #for each class
    
    n_instance_class=sum(classes==cls) #calculate the total number of instances of that class in the node
    n_instances.append(n_instance_class)
  
  #now calculate the gini for that node
  node_size=sum(n_instances) #get node size by summing up all the instances 
  gini_node=1
  for n in n_instances:  #for each instance in each class: 
    if node_size==0: 
      continue 
    gini_node-=(n/node_size)**2
  return gini_node
  
  n_features=X.shape[1]

#get split
def get_split(node): 
  best_gini,best_feature, best_split=1, None, None #initiate best gini value

  for k in range(n_features): #for each key (feature/variable)
    #feature_number=list(X)[k] #which variable/feature number we are splitting by e.g first of two vars
    for j in range(len(node)): #for each value in dictionary key 
      current_split_value=node[j][k] #we split by this value
      left, right=list(),list() #create empty lists for the classes of left and right nodes 
      for i in range(len(node)):  #iterate over each value in the dataset
        if node[i][k]<=current_split_value: #if current datapoint is less than split value it goes into left node, else right . HH TO DOUBLE CHECK IF <= OR<
          left.append(node[i])
        else:
          right.append(node[i])

      #calculate gini at current split
      region_size=len(left)+len(right) #first get number of values in each node
      weighted_gini=len(left)/(region_size)*calculate_gini(left) +len(right)/(region_size)*calculate_gini(right)  #calculate the weighted gini

      if weighted_gini<best_gini: #if current gini calc is better than best gini so far:
        best_gini, best_feature, best_split, best_left,best_right=weighted_gini, k, current_split_value, left, right#set new best values
  return {'index':best_feature, 'value': best_split, 'left':best_left, 'right':best_right, 'gini':best_gini} #STORE node information in a dictionary

def final_label(node):
  classes=[]
  for row in node:
    classes.append((row[-1])) #we create a list of classes (Y)
  node_dict=dict((i, classes.count(i)) for i in classes) #create a dictionary with a count of each class 
  label=max(node_dict, key=node_dict.get) #take class with maximum number of counts
  print(label, ":",sum(node_dict.values()),node_dict)
  return label

def recursive_split(node, max_depth, depth, min_size):
  left, right=node['left'], node['right']
  del node['left'], node['right']
  
  if not left or not right or node['gini']==0:
    node['left']=node['right']=final_label(left+right)
    return

  #check if we have exceeded the maximum epth
  if depth>=max_depth:
    node['left'], node['right']=final_label(left), final_label(right)
    return

    #checking if we are below the minimum size
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
    
   def build_tree(node, max_depth, min_size):
  root=get_split(dataset) #here we start with the initial full dataset
  recursive_split(root, max_depth, min_size, 1) # it will keep calling this function (as input into next run).d, Y, max_depth, depth)
  return root

def predict(node, row): 
	if row[node['index']] < node['value']:
		if type(node['left'])==dict:
			return predict(node['left'], row) #keep iterating
		else:
			return node['left']
	else:
		if type(node['right'])==dict:
			return predict(node['right'], row)
		else:
			return node['right']

tree=build_tree(dataset, 1, 1) 
tree
predictions=list()
for row in dataset: 
  prediction=predict(tree, row)
  predictions.append(prediction)
  
 def accuracy(predictions, Y):
  correct=0
  for i in range(len(Y)):
    if predictions[i]==Y[i]:
      correct+=1

  accuracy=correct/len(Y)
  print(accuracy)

accuracy(predictions, Y) 
