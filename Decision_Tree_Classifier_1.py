#!/usr/bin/env python
# coding: utf-8

# ### Decision-Tree-Classifier
# ### Decision Tree Classifier in Python with Scikit Learn
# 
# #### Prediction using Decision Tree Algorithm
# 
# #### Problem Statement : Create the Decision tree Classifier and Visualize it Graphically.

# ## Import libraries

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import tree


# ### Read dataset

# In[2]:


cwd = os.getcwd() + '/Downloads/'
data = cwd + 'Iris.csv'

df = pd.read_csv(data)
df


# ### Description of the Data

# In[3]:


df.describe()


# ### Check for any missing null values

# In[4]:


df.isnull().sum()


# ### Checking columns count of "Species"

# In[5]:


df['Species'].value_counts()


# In[6]:


df['PetalWidthCm'].plot.hist()
plt.show()


# In[7]:


df = df.drop(df.columns[0],axis=1)
df


# ### Transform Non-Numerical column into Numerical Column:

# In[8]:


df['Species'] = df.Species.replace({"Iris-setosa" : 0 ,"Iris-versicolor" : 1,"Iris-virginica" : 2})


# ### Extracting Independent and Dependent Variable :

# In[9]:


x = df[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
X = x.values
y = df['Species'].values
y


# ### Now Split the dataset into training and test set:

# In[10]:


(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=1)


#  ### Fitting a Decision - Tree algorithm to the Training set:

# In[11]:


dtc = DecisionTreeClassifier(ccp_alpha=0.01)
dtc.fit(X_train, y_train)
dtc.score(X_test, y_test)


# In[12]:


clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)


# In[13]:


clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)


# In[14]:


dtc.get_params()


# In[15]:


clf_gini.get_params()


# In[16]:


clf_entropy.get_params()


# ### Predicting the test result :

# In[17]:


y_pred_gini = clf_gini.predict(X_test)
y_pred_gini


# In[18]:


y_pred_entropy = clf_entropy.predict(X_test)
y_pred_entropy


# ### Test accuracy of the result :

# In[19]:


print ("Accuracy is :", accuracy_score(y_test,y_pred_gini)*100)


# In[20]:


print ("Accuracy is :", accuracy_score(y_test,y_pred_entropy)*100)


# ### Test precision of the result :

# In[21]:


from sklearn.metrics import precision_score
print ("Precision is :",precision_score(y_test, y_pred_gini,average='weighted')*100)


# In[22]:


from sklearn.metrics import precision_score
print ("Precision is :",precision_score(y_test, y_pred_entropy,average='weighted')*100)


# ### Test recall of the result :

# In[23]:


from sklearn.metrics import recall_score
print ("Recall is :",recall_score(y_test, y_pred_gini,average='weighted')*100)


# In[24]:


from sklearn.metrics import recall_score
print ("Recall is :",recall_score(y_test, y_pred_entropy,average='weighted')*100)


# ### Test classification report of the result :

# In[25]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_gini,target_names=['Iris-setosa','Iris-versicolor','Iris-virginica']))


# In[26]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_entropy,target_names=['Iris-setosa','Iris-versicolor','Iris-virginica']))


# ### Creation of Confusion Matrix :

# In[27]:


results_gini = confusion_matrix(y_test, y_pred_gini) 
print ('Confusion Matrix :')
print(results_gini) 


# In[28]:


results_entropy = confusion_matrix(y_test, y_pred_entropy) 
print ('Confusion Matrix :')
print(results_entropy) 


# In[29]:


feature_names = x.columns
feature_names


# In[30]:


clf_gini.feature_importances_


# In[31]:


clf_entropy.feature_importances_


# In[32]:


dtc.feature_importances_


# In[33]:


feature_importance_gini = pd.DataFrame(clf_gini.feature_importances_,index = feature_names).sort_values(0,ascending= False)
feature_importance_gini


# In[34]:


feature_importance_entropy = pd.DataFrame(clf_entropy.feature_importances_,index = feature_names).sort_values(0,ascending= False)
feature_importance_entropy


# In[35]:


features_gini = list(feature_importance_gini[feature_importance_gini[0]>0].index)
features_gini


# In[36]:


features_entropy = list(feature_importance_entropy[feature_importance_entropy[0]>0].index)
features_entropy


# In[37]:


feature_importance_gini.head(10).plot(kind='bar')


# In[38]:


feature_importance_entropy.head(10).plot(kind='bar')


# ### Visualising Result :
# #### Visualize the Graph :

# In[39]:


from sklearn import tree
from matplotlib import pyplot as plt
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf_gini,feature_names=feature_names,class_names=None
                   , filled=True,fontsize=25)


# In[40]:


from sklearn import tree
from matplotlib import pyplot as plt
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf_entropy,feature_names=feature_names,class_names=None
                   , filled=True,fontsize=25)


# ### Conclusion : Hence Decision Tree Classifier is created and Visualized it graphically.
