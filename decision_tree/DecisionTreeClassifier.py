#Import libraries
import numpy as np

class DecisionTreeClassifier(object):
  def __init__(self, 
               max_depth=None, 
               min_size=None):
               #*args,
               #**kwargs):
    
    #super(DecisionTreeClassifier, self).__init__(*args, **kwargs)
    #self.__dict__ = self

    self.n_features=None
    self.depth=0
    self.max_depth=max_depth
    self.min_size=min_size #minimum size of the node
    self.node=None #none
    self.left=None
    self.right=None
    self.tree=None
    '''
    Function to calculate the gini of the node 
    '''

  def calculate_gini(self, node):
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

  def get_split(self, node): 
    #We take each feature and value and calculate the gini at this split. By the end we will have found the best split. 
    best_gini,best_feature, best_split=1, None, None 
    for k in range(self.n_features): 
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
        weighted_gini=len(left)/(region_size)*self.calculate_gini(left) +len(right)/(region_size)*self.calculate_gini(right) 

        #if current gini calc is better than best gini so far we update the information on the best split 
        if weighted_gini<best_gini: 
          best_gini, best_feature, best_split, best_left,best_right=weighted_gini, k, current_split_value, left, right
    
    #Store node information in a dictionary
    return {'index':best_feature, 'value': best_split, 'left':best_left, 'right':best_right, 'gini':best_gini} 

  '''
  Function to assign classes to the terminal nodes  
  ''' 

  def final_label(self, node):
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
  def recursive_split(self, node, depth=0):
    
    #Checking whether the node is a terminal node, or whether the impurity =0 - if so we will assign a label. 
    if not self.left or not self.right or node['gini']==0:
      node['left']=node['right']=self.final_label(self.left+self.right)
      return

    #Checking if we have exceeded the maximum depth  - otherwise we will keep calling the function
    if self.depth>=self.max_depth:
      node['left'], node['right']=self.final_label(self.left), self.final_label(self.right)
      return

    #Checking if we are below the minimum size - otherwise we will keep calling the function
    if len(self.left)<=self.min_size:
      self.left=self.final_label(self.left)
    else:
      self.left=self.get_split(self.left)['left']
      self.depth=self.depth+1
      self.recursive_split(self, node['left']) 
      
    if len(self.right)<=self.min_size:
      node['right']=self.final_label(self.right)

    else:
      self.right=self.get_split(self.right)['right']
      self.depth=self.depth+1 
      self.recursive_split(self, node['right'])   #**

  '''
  Function to build out the tree 
  '''
  #We start with the initial full dataset and it keeps calling the recursive split function until the constraints are violated

  def grow_tree(self, dataset):
    root=self.get_split(dataset) 
    self.recursive_split(self, root)
    #self.recursive_split(root, self.max_depth, self.min_size, 1) 
    return root

  '''
  Function to print the tree 
  '''
  def print_tree(self, node, depth=0):
    #If the node is a dictionary, it means it is not a terminal node, so it keeps traversing the tree until a terminal node is reached
    if type(node)==dict: 
      print(depth*"  ","X", node['index'], "<=", node['value'])
      self.print_tree(node['left'], depth+1)
      self.print_tree(node['left'], depth+1)
    else:
      print(depth*" ", node)

  '''
  Function to predict the label, given the input dataset
  '''

  def predict_one_row(self, node, row): 
    #We traverse the tree until we reach a terminal node, and return the label that is assigned to this node. 
    if row[node['index']] <= node['value']:
      if type(node['left'])==dict:
        return self.predict(node['left'], row) 
      else:
        return node['left']
    else:
      if type(node['right'])==dict:
        return self.predict(node['right'], row)
      else:
        return node['right']

  #def prediction_list(self, X):
  def predict(self, dataset):
    predictions=list()
    for row in dataset: 
      predictions.append(self.predict_one_row(self.tree, row)) #***tree
    return predictions

  '''
  Function to create one dataset by merging X and Y. 
  '''

  def dataset(self, X, Y):
    dataset=list()
    for i in range(len(X)):
      row_concat=np.append(X[i],Y[i])
      dataset.append(row_concat)
    return dataset

  '''
  Function to finally build out the whole tree and return a list of predictions
  '''

  def fit(self, X, y):

      if X.ndim == 1:
          X = X.reshape(-1, 1)
      if y.ndim == 1:
          y = y.reshape(-1, 1)

      self.n_features = X.shape[1]
      self.max_depth=1
      self.min_size=1
      dataset=self.dataset(X, y)
      # grow the tree recursively starting from root 
      self.grow_tree(dataset)
      
      #self.tree=self.grow_tree(dataset)

  '''
  Function to return the accuracy of our predictions
  '''
  def accuracy(self, predictions, Y):
    #We take the number of correct predictions/total length of the test set
    correct=0
    for i in range(len(Y)):
      if predictions[i]==Y[i]:
        correct+=1

    accuracy=correct/len(Y)
    print("Accuracy: ", accuracy)



  #tests
if __name__=="__main__":
  import sys
  from sklearn.datasets import make_moons
  X,Y=make_moons(100, noise=0.2)
  clf=DecisionTreeClassifier(max_depth=1, min_size=1)
  #vars(clf)
  clf.fit(X, Y)
  #clf.predict(Y)
  #clf.accuracy()

