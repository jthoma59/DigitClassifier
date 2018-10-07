import numpy as np
from sklearn.cross_validation import train_test_split

#Used to plot image
import matplotlib.pyplot as pt

#To load training data set
import pandas as pd

#Import classifier
from sklearn.tree import DecisionTreeClassifier

#Data includes label (what the digit is) and its various pixels
#convert into 2d numpy array
data = pd.read_csv("DigitData/train.csv").as_matrix()

#Assign classifier to clf
clf = DecisionTreeClassifier()

#Build dataset
X = data[0:, 1:]
y = data[0:, 0]

#Create sets for testing and training data
#.5 = separate into halves
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=.5)

#Train the classifier
clf.fit(x_train,y_train)

#Get predictions after classification
predicitions = clf.predict(x_test)

#See resulting accuracy
from sklearn.metrics import accuracy_score
print("Decision Tree: ")
print(accuracy_score(y_test, predicitions))
