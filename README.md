# Supervised Learning - Decision Tree prediction and visualization
Problem statement:  To predict the safety of the car. Create the Decision Tree Classifier and Visualize it graphically.

## Mandatory Requirement: 

Place the dataset in downloads folder.

requirements.txt : Contains all the libraries used in the code. Which needs to be installed first.

## Used Python Packages:

1) sklearn :

      a) In python, sklearn is a machine learning package which include a lot of ML algorithms.
      
      b) Here, we are using some of its modules like train_test_split, DecisionTreeClassifier,accuracy_score,precision_score,recall_score and           classification_report.
      
2) NumPy :

      a) It is a numeric python module which provides fast maths functions for calculations.
      
      b) It is used to read data in numpy arrays and for manipulation purpose.
      
3) Pandas :

      a) Used to read and write different files.
      
      b) Data manipulation can be done easily with dataframes.

## Installation of the packages :
In Python, sklearn is the package which contains all the required packages to implement Machine learning algorithm. You can install the sklearn package by following the commands given below.

a) using pip :
pip install -U scikit-learn
Before using the above command make sure you have scipy and numpy packages installed.
If you don’t have pip. You can install it using python get-pip.py

b) using conda :
conda install scikit-learn
                                  
## Assumptions we make while using Decision tree :
1) At the beginning, we consider the whole training set as the root.
2) Attributes are assumed to be categorical for information gain and for gini index, attributes are assumed to be continuous.
3) On the basis of attribute values records are distributed recursively.
4) We use statistical methods for ordering attributes as root or internal node.
5) While implementing the decision tree we will go through the following two phases:

A) Building Phase
   a) Preprocess the dataset.
   b) Split the dataset from train and test using Python sklearn package.
   c) Train the classifier.
   
B) Operational Phase
   a) Make predictions.
   b) Calculate the accuracy.
                                   

### Terms used in code :

Gini index and information gain both of these methods are used to select from the n attributes of the dataset which attribute would be placed at the root node or the internal node.

1) Gini index
2) Entropy
3) Information Gain
4) Accuracy score
5) Confusion Matrix
